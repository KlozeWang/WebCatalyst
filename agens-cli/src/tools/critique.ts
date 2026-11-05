/**
 * `critique` tool — the external-judge half of the build/critique loop.
 *
 * Why a tool, not a pure skill: we want an INDEPENDENT model call with its
 * own system prompt and strict JSON schema, so the feedback isn't biased
 * by the builder agent's reasoning context. Think "external judge".
 *
 * Design note: this tool intentionally returns NO numeric scores. A
 * scalar score collapses too much information and tempts the agent to
 * stop the moment a threshold is crossed. Instead, the judge writes a
 * structured list of observed issues + concrete suggestions, plus a
 * binary verdict (`ship_it` vs `needs_revision`). The agent then loops
 * on the issues until none are `major`.
 *
 * Input:
 *   - taskPrompt:     the original user request (what they asked for)
 *   - screenshotPath: PNG/JPEG produced by Playwright MCP
 *   - artifactPath?:  optional source file path for additional context
 *   - focus?:         optional caller-supplied areas to pay extra attention to
 *
 * Output (always JSON):
 *   {
 *     "summary":   "one-sentence overall impression",
 *     "issues":    [{ "area", "severity", "observation", "suggestion" }],
 *     "wellDone":  ["things to preserve, so the agent doesn't undo them"],
 *     "verdict":   "ship_it" | "needs_revision"
 *   }
 *
 * The tool reuses the user's BYOK provider config so critique calls hit the
 * same endpoint as the builder agent (no second key to configure). If the
 * user is on GitHub Copilot (no BYOK), the tool fails loudly — vision
 * critique requires a vision-capable endpoint we can POST to directly.
 */
import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { defineTool } from "@github/copilot-sdk";
import chalk from "chalk";
import { z } from "zod";
import type { ProviderFile } from "../userconfig.js";

const CRITIC_SYSTEM_PROMPT = `You are a meticulous external design + product judge evaluating a visual
artifact produced by a coding agent. You receive:
  1. The user's original task prompt.
  2. A screenshot of the artifact in a browser.
  3. Optionally, the source code.

Your job is NOT to assign a score. Your job is to describe, in precise
words, what is wrong with the artifact and how to fix it — so another
agent can read your feedback and the screenshot together and improve the
work.

Respond with a single JSON object matching this schema exactly (no
markdown fences, no prose outside JSON):

{
  "summary": "<one sentence overall impression — what works and what doesn't>",
  "issues": [
    {
      "area":        "<where in the artifact, e.g. 'hero', 'nav', 'card grid', 'footer', 'gameplay loop'>",
      "severity":    "major" | "minor" | "nit",
      "observation": "<what you see that's wrong, in concrete visual terms>",
      "suggestion":  "<a specific change to make — sizes, colors, structure, copy>"
    }
  ],
  "wellDone": [ "<things that already work and the agent should NOT change>" ],
  "verdict": "ship_it" | "needs_revision"
}

Severity guide:
  - major: hurts task fidelity, breaks layout, looks unprofessional, or
           any user would notice immediately. Must be fixed.
  - minor: noticeable polish issue. Should be fixed before shipping.
  - nit:   a refinement; nice to have but not required.

Dimensions to look at (use them as a checklist, not as output fields):
  - task fidelity:   does it actually do what the user asked?
  - visual quality:  layout, typography hierarchy, color, spacing, polish
  - interactivity:   if applicable, do hover/click/animation states work?
  - correctness:     no broken layout, no overflow, no console errors visible
  - originality:     is the design generic, or thoughtful?
  - accessibility:   contrast, semantic structure (judge from screenshot)

Verdict rules:
  - "needs_revision" if there is ANY issue with severity "major", OR if
    there are 3+ "minor" issues.
  - "ship_it" only when the artifact would satisfy a demanding user as-is.

Observations and suggestions MUST be specific and visual. Good:
"The hero heading is roughly 32px; at the 1440px viewport it gets lost
above the 18px subhead. Bump heading to 56-64px and add 24px of top
padding." Bad: "make it nicer", "improve typography".

Aim for 2-6 issues. Be honest — most first drafts need revision. Do not
invent issues to pad the list; if it really ships, return an empty
issues array and verdict "ship_it".`;

const parameters = z.object({
  taskPrompt: z.string().describe("The original user task / prompt that the artifact is meant to fulfill. Pass verbatim — do NOT paraphrase."),
  screenshotPath: z.string().describe("Absolute or session-relative path to a PNG/JPEG screenshot of the rendered artifact."),
  artifactPath: z.string().optional().describe("Optional path to the artifact source (e.g. index.html) to include for context."),
  focus: z.array(z.string()).optional().describe("Optional list of areas / dimensions the caller especially wants the judge to scrutinize (e.g. 'typography', 'interactivity', 'hero section')."),
});

type Severity = "major" | "minor" | "nit";

type CritiqueIssue = {
  area: string;
  severity: Severity;
  observation: string;
  suggestion: string;
};

type CritiqueResult = {
  summary: string;
  issues: CritiqueIssue[];
  wellDone: string[];
  verdict: "ship_it" | "needs_revision";
  /** Provenance: which judge actually wrote this. Useful for trajectories
   *  + for spotting "oh the wrong model got called" at a glance. */
  _judge?: {
    requestedModel: string;
    responseModel?: string;
    baseUrl: string;
    latencyMs: number;
  };
};

function guessMime(p: string): string {
  const ext = extname(p).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".webp") return "image/webp";
  if (ext === ".gif") return "image/gif";
  return "image/png";
}

export function makeCritiqueTool(provider: ProviderFile | undefined, model: string) {
  return defineTool("critique", {
    description:
      "Send the original task prompt + a screenshot of the rendered artifact to an external judge model. " +
      "Returns a structured list of observed issues with severity + concrete suggestions, plus a verdict " +
      "('ship_it' or 'needs_revision'). No numeric scores — use the issue list to drive the next iteration. " +
      "Loop until verdict='ship_it' or no 'major' issues remain (cap at ~5 iterations).",
    parameters,
    skipPermission: true,
    handler: async (args): Promise<CritiqueResult | string> => {
      if (!provider?.baseUrl) {
        return "ERROR: critique requires a BYOK provider (set provider.baseUrl in config). Cannot call vision endpoint.";
      }

      // Load image as base64 data URL so it works against any OpenAI-compatible endpoint.
      let imageDataUrl: string;
      try {
        const buf = await readFile(args.screenshotPath);
        imageDataUrl = `data:${guessMime(args.screenshotPath)};base64,${buf.toString("base64")}`;
      } catch (e) {
        return `ERROR: failed to read screenshot at ${args.screenshotPath}: ${(e as Error).message}`;
      }

      let artifactSnippet = "";
      if (args.artifactPath) {
        try {
          const src = await readFile(args.artifactPath, "utf8");
          artifactSnippet = src.length > 12_000 ? src.slice(0, 12_000) + "\n…[truncated]" : src;
        } catch {
          // non-fatal — judge can still work from the screenshot
        }
      }

      const userParts: any[] = [
        { type: "text", text: `Task prompt:\n${args.taskPrompt}` },
        { type: "image_url", image_url: { url: imageDataUrl } },
      ];
      if (artifactSnippet) {
        userParts.push({ type: "text", text: `Source code:\n\`\`\`\n${artifactSnippet}\n\`\`\`` });
      }
      if (args.focus?.length) {
        userParts.push({ type: "text", text: `Caller asks you to pay extra attention to: ${args.focus.join(", ")}` });
      }

      // POST directly — we're an OpenAI-compatible client here, intentionally
      // bypassing the SDK runtime so the critic call is fully isolated.
      const url = provider.baseUrl.replace(/\/+$/, "") + "/chat/completions";
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (provider.bearerToken) headers["Authorization"] = `Bearer ${provider.bearerToken}`;
      else if (provider.apiKey) headers["Authorization"] = `Bearer ${provider.apiKey}`;

      // Surface provenance live so the user can see WHO is judging.
      process.stderr.write(
        chalk.gray(`  · critique judge=${model} @ ${provider.baseUrl}\n`),
      );

      const startedAt = Date.now();
      let raw: string;
      let responseModel: string | undefined;
      try {
        const res = await fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model,
            messages: [
              { role: "system", content: CRITIC_SYSTEM_PROMPT },
              { role: "user", content: userParts },
            ],
            temperature: 0.2,
            response_format: { type: "json_object" },
          }),
        });
        if (!res.ok) {
          const body = await res.text().catch(() => "");
          return `ERROR: critic endpoint returned ${res.status} from ${url} (model=${model}): ${body.slice(0, 500)}`;
        }
        const data = (await res.json()) as any;
        raw = data?.choices?.[0]?.message?.content ?? "";
        // OpenAI-compatible servers echo the model that actually served
        // the request — useful when a gateway silently rewrites models.
        responseModel = typeof data?.model === "string" ? data.model : undefined;
      } catch (e) {
        return `ERROR: critic call failed (model=${model}, url=${url}): ${(e as Error).message}`;
      }
      const latencyMs = Date.now() - startedAt;

      // Strip stray fences just in case the model ignored response_format
      const cleaned = raw.trim().replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "");
      let parsed: CritiqueResult;
      try {
        parsed = JSON.parse(cleaned);
      } catch {
        return `ERROR: judge returned non-JSON content. Raw output:\n${raw.slice(0, 1500)}`;
      }

      // Defensive normalization: ensure required fields exist with sane types
      // so downstream agents can rely on the shape without null-checks everywhere.
      if (!Array.isArray(parsed.issues)) parsed.issues = [];
      if (!Array.isArray(parsed.wellDone)) parsed.wellDone = [];
      if (typeof parsed.summary !== "string") parsed.summary = "";

      // Recompute verdict deterministically from issue severities so the
      // result is consistent regardless of what the judge wrote — same
      // rule the system prompt asks for.
      const majors = parsed.issues.filter((i) => i?.severity === "major").length;
      const minors = parsed.issues.filter((i) => i?.severity === "minor").length;
      parsed.verdict = majors > 0 || minors >= 3 ? "needs_revision" : "ship_it";

      parsed._judge = {
        requestedModel: model,
        responseModel,
        baseUrl: provider.baseUrl,
        latencyMs,
      };
      // One-line live summary so the user sees the verdict and who issued it.
      const verdictTag =
        parsed.verdict === "ship_it" ? chalk.green("ship_it") : chalk.yellow("needs_revision");
      process.stderr.write(
        chalk.gray(
          `  · critique → ${verdictTag} ` +
            `(${majors} major, ${minors} minor, ${parsed.issues.length - majors - minors} nit) ` +
            `[judge=${responseModel ?? model}, ${latencyMs}ms]\n`,
        ),
      );
      return parsed;
    },
  });
}
