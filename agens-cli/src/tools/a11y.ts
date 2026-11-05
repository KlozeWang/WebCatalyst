/**
 * `a11y_audit` tool — run axe-core against a rendered URL using the
 * singleton Playwright Browser. Returns a compact summary of violations
 * the agent can act on.
 *
 * Why a separate tool, not "ask critique to grade accessibility":
 *   - Vision models guess at a11y; axe gives ground truth.
 *   - Cheap (no extra LLM call).
 *   - Returns specific selectors and fixes the agent can patch.
 */
import { defineTool } from "@github/copilot-sdk";
import { z } from "zod";
// axe-core ships as CJS; the bundled rule source lives on the default export.
import axe from "axe-core";
import { newPage, closePage } from "../playwrightHost.js";

const axeSource: string = (axe as unknown as { source: string }).source;

const parameters = z.object({
  url: z.string().describe("URL to audit. Use `file://<absolute>/index.html` for local artifacts."),
  passThreshold: z.number().int().min(0).optional().describe("Max allowed violations to call it passed (default 0)."),
});

interface AxeNode { target: string[]; html: string }
interface AxeViolation { id: string; impact: string | null; help: string; nodes: AxeNode[] }
interface AxeResult { url: string; violations: AxeViolation[] }

declare global {
  // axe is injected at runtime via addScriptTag; declare for evaluate().
  interface Window { axe: { run(): Promise<AxeResult> } }
}

export const a11yTool = defineTool("a11y_audit", {
  description:
    "Run axe-core accessibility audit on a rendered URL (via Playwright). Returns a structured list of violations with selectors and impact, so the agent can fix specific issues. Use after `html-artisan` produces an artifact, before declaring it done.",
  parameters,
  skipPermission: true,
  handler: async (args) => {
    let page;
    try {
      page = await newPage();
    } catch (e) {
      return `ERROR: failed to launch browser: ${(e as Error).message}`;
    }

    try {
      await page.goto(args.url, { waitUntil: "networkidle", timeout: 30_000 });
      await page.addScriptTag({ content: axeSource });
      const result = await page.evaluate(async () => await window.axe.run());

      const threshold = args.passThreshold ?? 0;
      const violations = (result.violations ?? []).map((v) => ({
        id: v.id,
        impact: v.impact,
        help: v.help,
        nodes: v.nodes.slice(0, 3).map((n) => ({
          target: n.target.join(", "),
          html: n.html.slice(0, 200),
        })),
      }));
      return {
        url: result.url,
        violationCount: violations.length,
        passed: violations.length <= threshold,
        violations,
      };
    } catch (e) {
      return `ERROR: a11y audit failed: ${(e as Error).message}`;
    } finally {
      await closePage(page);
    }
  },
});
