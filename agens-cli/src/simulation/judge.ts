import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import chalk from "chalk";
import type { ProviderFile, SimulationJudgeRoutingConfig } from "../userconfig.js";
import { ACTION_DECISION_SYSTEM_PROMPT } from "./prompts.js";
import type {
  ActionHint,
  JudgeProvenance,
  LoadedText,
  PsychometricObservation,
  SimulationAction,
  SimulationDecision,
} from "./types.js";
import { clampMs, normalizeMaybeMs } from "./utils.js";

const RETRYABLE_HTTP_STATUSES = new Set([402, 403, 429]);
const RETRYABLE_REQUEST_MAX_RETRIES = 2;
const DISABLE_TEMPERATURE_ENV = "AGENS_SIM_JUDGE_DISABLE_TEMPERATURE";
const RETRYABLE_REQUEST_RETRY_DELAY_MS = 5_000;
const PILLOW_COMPRESSION_SCRIPT = String.raw`
import sys
from PIL import Image, ImageOps

image = ImageOps.exif_transpose(Image.open(sys.argv[1]))
image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
if image.mode in ("RGBA", "LA"):
    background = Image.new("RGB", image.size, "white")
    background.paste(image, mask=image.getchannel("A"))
    image = background
else:
    image = image.convert("RGB")
image.save(sys.stdout.buffer, format="JPEG", quality=75, optimize=True, progressive=True)
`;

type CompressedImage = { data: Buffer; mimeType: string };
type ImageCompressor = (path: string) => Promise<CompressedImage>;

class ActionValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ActionValidationError";
  }
}

export async function decideAction(args: {
  provider?: ProviderFile;
  routing?: SimulationJudgeRoutingConfig;
  model: string;
  taskPrompt?: string;
  screenshotPath: string;
  persona?: LoadedText;
  metrics?: LoadedText;
  artifact?: LoadedText;
  url: string;
  actionHints: ActionHint[];
  episodeHistory: Array<Record<string, unknown>>;
  turnIndex: number;
  turnLimit: number;
  frameMs: number;
  maxAdvanceMs: number;
}): Promise<SimulationDecision> {
  const baseTextParts = [
    `URL:\n${args.url}`,
    `Task prompt:\n${args.taskPrompt ?? "Simulate realistic user play until the task is complete or the user would stop."}`,
    `Turn:\n${args.turnIndex + 1} of ${args.turnLimit}`,
    `Reference clock intervals:\n${JSON.stringify({
      referenceFrameIntervalMs: args.frameMs,
      maxPostActionAdvanceMs: args.maxAdvanceMs,
    }, null, 2)}`,
    ...(args.episodeHistory.length
      ? [`Episode history:\n${JSON.stringify(args.episodeHistory, null, 2)}`]
      : ["Episode history:\n[]"]),
    `Available interactive elements:\n${JSON.stringify(args.actionHints, null, 2)}`,
    ...(args.persona ? [`User persona (${args.persona.path}):\n${args.persona.text}`] : []),
    ...(args.metrics ? [`Psychometric metrics (${args.metrics.path}):\n${args.metrics.text}`] : []),
    ...(args.artifact ? [`Artifact source code (${args.artifact.path}):\n\`\`\`\n${args.artifact.text}\n\`\`\``] : []),
  ];
  let correctionPrompt: string | undefined;
  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const raw = await callVisionJudge({
      provider: args.provider,
      routing: args.routing,
      model: args.model,
      systemPrompt: ACTION_DECISION_SYSTEM_PROMPT,
      screenshotPath: args.screenshotPath,
      textParts: correctionPrompt ? [...baseTextParts, correctionPrompt] : baseTextParts,
      progressLabel: attempt === 1 ? "user simulation" : `user simulation retry ${attempt - 1}`,
    });
    try {
      const decision = normalizeDecision(raw.parsed);
      decision._judge = raw.provenance;
      process.stderr.write(
        chalk.gray(`  · user simulation action=${decision.action.type} [judge=${raw.provenance.responseModel ?? args.model}]\n`),
      );
      return decision;
    } catch (err) {
      if (!(err instanceof ActionValidationError) || attempt >= maxAttempts) throw err;
      correctionPrompt = [
        "Your previous response was invalid and was not executed.",
        `Validation error: ${(err as Error).message}`,
        "Return corrected JSON only, using the same schema.",
        "Reminder: combo.keys must contain 1 or 2 keyboard keys only.",
        `Previous invalid JSON:\n${JSON.stringify(raw.parsed, null, 2)}`,
      ].join("\n");
      process.stderr.write(chalk.yellow(`  · user simulation invalid action; retrying: ${(err as Error).message}\n`));
    }
  }
  throw new ActionValidationError("Failed to produce a valid simulation action.");
}

export type JudgeTextPart = { kind: "text"; text: string };
export type JudgeAttachment = { kind: "image"; path: string; mimeType: string };
export type JudgeContentPart = JudgeTextPart | JudgeAttachment;

export async function callJsonJudge(args: {
  provider?: ProviderFile;
  routing?: SimulationJudgeRoutingConfig;
  model: string;
  systemPrompt: string;
  textParts: string[];
  attachments: JudgeAttachment[];
  contentParts?: JudgeContentPart[];
  progressLabel: string;
  /** @internal Test hook; production uses Pillow through PYTHON_BIN/python3. */
  imageCompressor?: ImageCompressor;
}): Promise<{ parsed: any; provenance: JudgeProvenance }> {
  const provider = selectJudgeProvider(args.provider, args.routing);
  if (!provider?.baseUrl) {
    throw new Error("user_simulation requires a provider baseUrl for its judge.");
  }

  const contentParts = args.contentParts ?? [
    ...args.textParts.map((text): JudgeContentPart => ({ kind: "text", text })),
    ...args.attachments,
  ];
  const useResponsesApi = provider.wireApi === "responses";
  const hasImages = contentParts.some((part) => part.kind === "image");
  const buildUserParts = async (compressImages: boolean): Promise<any[]> => {
    const userParts: any[] = [];
    for (const part of contentParts) {
      if (part.kind === "text") {
        userParts.push({ type: useResponsesApi ? "input_text" : "text", text: part.text });
        continue;
      }
      const compressed = compressImages
        ? await (args.imageCompressor ?? compressImageWithPillow)(part.path)
        : undefined;
      const file = compressed?.data ?? await readFile(part.path);
      const mimeType = compressed?.mimeType ?? part.mimeType;
      const dataUrl = `data:${mimeType};base64,${file.toString("base64")}`;
      userParts.push(useResponsesApi
        ? { type: "input_image", image_url: dataUrl }
        : { type: "image_url", image_url: { url: dataUrl } });
    }
    return userParts;
  };

  const url = provider.baseUrl.replace(/\/+$/, "") + (useResponsesApi ? "/responses" : "/chat/completions");
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (provider.bearerToken) headers.Authorization = `Bearer ${provider.bearerToken}`;
  else if (provider.apiKey) headers.Authorization = `Bearer ${provider.apiKey}`;

  const disableTemperature = process.env[DISABLE_TEMPERATURE_ENV] === "true";
  const buildRequestBody = async (compressImages: boolean): Promise<string> => {
    const userParts = await buildUserParts(compressImages);
    return JSON.stringify(useResponsesApi
      ? {
          model: args.model,
          input: [
            { role: "system", content: [{ type: "input_text", text: args.systemPrompt }] },
            { role: "user", content: userParts },
          ],
          text: { format: { type: "json_object" } },
        }
      : {
          model: args.model,
          messages: [
            { role: "system", content: args.systemPrompt },
            { role: "user", content: userParts },
          ],
          ...(!disableTemperature ? { temperature: 0.2 } : {}),
          response_format: { type: "json_object" },
        });
  };
  let requestBody = await buildRequestBody(false);

  process.stderr.write(chalk.gray(`  · ${args.progressLabel} judge=${args.model} @ ${provider.baseUrl}\n`));
  const startedAt = Date.now();
  let data: any;
  let retryableRequestRetries = 0;
  let compressedRetryUsed = false;
  while (true) {
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers,
        body: requestBody,
      });
    } catch (error) {
      if (retryableRequestRetries >= RETRYABLE_REQUEST_MAX_RETRIES) throw error;
      retryableRequestRetries += 1;
      const detail = error instanceof Error ? error.message : String(error);
      process.stderr.write(
        chalk.yellow(
          `  · ${args.progressLabel} judge fetch failed: ${detail}; retry ${retryableRequestRetries}/${RETRYABLE_REQUEST_MAX_RETRIES} in ${RETRYABLE_REQUEST_RETRY_DELAY_MS / 1_000}s\n`,
        ),
      );
      await new Promise<void>((resolve) => setTimeout(resolve, RETRYABLE_REQUEST_RETRY_DELAY_MS));
      continue;
    }
    if (res.ok) {
      data = await res.json();
      break;
    }

    const body = await res.text().catch(() => "");
    if (res.status === 400 && hasImages && !compressedRetryUsed) {
      const originalBytes = Buffer.byteLength(requestBody);
      try {
        requestBody = await buildRequestBody(true);
      } catch (error) {
        throw new Error(
          `judge returned 400 from ${url}, then Pillow image compression failed: ${(error as Error).message}; ` +
          `original response: ${body.slice(0, 500)}`,
        );
      }
      compressedRetryUsed = true;
      process.stderr.write(
        chalk.yellow(
          `  · ${args.progressLabel} judge returned 400; retrying once with Pillow-compressed images ` +
          `(${originalBytes} -> ${Buffer.byteLength(requestBody)} request bytes)\n`,
        ),
      );
      continue;
    }
    if (RETRYABLE_HTTP_STATUSES.has(res.status) && retryableRequestRetries < RETRYABLE_REQUEST_MAX_RETRIES) {
      retryableRequestRetries += 1;
      process.stderr.write(
        chalk.yellow(
          `  · ${args.progressLabel} judge returned ${res.status}; retry ${retryableRequestRetries}/${RETRYABLE_REQUEST_MAX_RETRIES} in ${RETRYABLE_REQUEST_RETRY_DELAY_MS / 1_000}s\n`,
        ),
      );
      await new Promise<void>((resolve) => setTimeout(resolve, RETRYABLE_REQUEST_RETRY_DELAY_MS));
      continue;
    }
    throw new Error(`judge returned ${res.status} from ${url}: ${body.slice(0, 500)}`);
  }

  if (data === undefined) {
    throw new Error(`judge request exhausted retries without a response from ${url}`);
  }
  const raw = useResponsesApi
    ? String((data?.output ?? [])
        .flatMap((item: any) => item?.content ?? [])
        .filter((item: any) => item?.type === "output_text")
        .map((item: any) => item?.text ?? "")
        .join(""))
    : String(data?.choices?.[0]?.message?.content ?? "");
  const responseModel = typeof data?.model === "string" ? data.model : undefined;
  const cleaned = raw.trim().replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "");

  let parsed: any;
  try {
    parsed = JSON.parse(cleaned);
  } catch {
    throw new Error(`judge returned non-JSON content: ${raw.slice(0, 1500)}`);
  }

  return {
    parsed,
    provenance: {
      requestedModel: args.model,
      responseModel,
      baseUrl: provider.baseUrl,
      latencyMs: Date.now() - startedAt,
    },
  };
}

async function callVisionJudge(args: {
  provider?: ProviderFile;
  routing?: SimulationJudgeRoutingConfig;
  model: string;
  systemPrompt: string;
  screenshotPath: string;
  textParts: string[];
  progressLabel: string;
}): Promise<{ parsed: any; provenance: JudgeProvenance }> {
  return callJsonJudge({
    provider: args.provider,
    routing: args.routing,
    model: args.model,
    systemPrompt: args.systemPrompt,
    textParts: args.textParts,
    attachments: [{ kind: "image", path: args.screenshotPath, mimeType: "image/png" }],
    progressLabel: args.progressLabel,
  });
}

function selectJudgeProvider(
  primaryProvider: ProviderFile | undefined,
  routing: SimulationJudgeRoutingConfig | undefined,
): ProviderFile | undefined {
  if (!routing) return primaryProvider;
  if (!Number.isFinite(routing.ratio) || routing.ratio < 0) {
    throw new Error("simulation judge routing ratio must be a finite non-negative number.");
  }
  const additionalProbability = routing.ratio / (routing.ratio + 1);
  return Math.random() < additionalProbability
    ? { ...routing.additionalProvider, wireApi: "completions" }
    : primaryProvider;
}

function compressImageWithPillow(path: string): Promise<CompressedImage> {
  const pythonBin = process.env.PYTHON_BIN ?? "python3";
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonBin, ["-c", PILLOW_COMPRESSION_SCRIPT, path], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    proc.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
    proc.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
    proc.on("error", (error) => {
      reject(new Error(`could not start ${pythonBin}: ${error.message}`));
    });
    proc.on("close", (code) => {
      if (code !== 0) {
        const detail = Buffer.concat(stderr).toString("utf8").trim();
        reject(new Error(`${pythonBin} Pillow compressor exited with code ${code}${detail ? `: ${detail}` : ""}`));
        return;
      }
      const data = Buffer.concat(stdout);
      if (!data.length) {
        reject(new Error(`${pythonBin} Pillow compressor produced an empty image`));
        return;
      }
      resolve({ data, mimeType: "image/jpeg" });
    });
  });
}

function normalizeDecision(raw: any): SimulationDecision {
  const action = normalizeAction(raw?.action ?? raw);
  return {
    summary: typeof raw?.summary === "string" ? raw.summary : "",
    action,
    targetState: typeof raw?.targetState === "string" && raw.targetState.trim()
      ? raw.targetState
      : "The page visibly reflects the selected user action.",
    expectedTransition: typeof raw?.expectedTransition === "string" ? raw.expectedTransition : undefined,
    psychometricObservation: normalizePsychometric(raw?.psychometricObservation),
  };
}

function normalizePsychometric(raw: any): PsychometricObservation | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  return {
    positive: Array.isArray(raw.positive) ? raw.positive.filter((x: unknown) => typeof x === "string") : undefined,
    negative: Array.isArray(raw.negative) ? raw.negative.filter((x: unknown) => typeof x === "string") : undefined,
  };
}

function normalizeAction(raw: any): SimulationAction {
  const type = String(raw?.type ?? "").toLowerCase();
  const waitMs = clampMs(normalizeMaybeMs(raw?.waitMs, raw?.waitSeconds) ?? 0, 0, 10_000, "action.waitMs");
  const withPreWait = <T extends Exclude<SimulationAction, { type: "terminate" }>>(action: T): T =>
    waitMs > 0 ? { ...action, waitMs } : action;

  if (type === "terminate" || type === "stop") {
    return {
      type: "terminate",
      reason: typeof raw.reason === "string" ? raw.reason : undefined,
    };
  }
  if (type === "click" || type === "hover") {
    const action: any = { type };
    if (typeof raw.selector === "string") action.selector = raw.selector;
    if (typeof raw.text === "string") action.text = raw.text;
    if (Number.isFinite(raw.x)) action.x = Number(raw.x);
    if (Number.isFinite(raw.y)) action.y = Number(raw.y);
    if (type === "click" && Number.isFinite(raw.durationMs)) action.durationMs = Number(raw.durationMs);
    return withPreWait(action);
  }
  if (type === "mousemove" || type === "mouse_move" || type === "look") {
    return withPreWait({
      type: "mouseMove",
      dx: Number.isFinite(raw.dx) ? Number(raw.dx) : 0,
      dy: Number.isFinite(raw.dy) ? Number(raw.dy) : 0,
      ...(Number.isFinite(raw.durationMs) ? { durationMs: Number(raw.durationMs) } : {}),
      ...(Number.isFinite(raw.steps) ? { steps: Number(raw.steps) } : {}),
    });
  }
  if (type === "press") {
    return withPreWait({
      type: "press",
      key: typeof raw.key === "string" ? raw.key : "Enter",
      ...(Number.isFinite(raw.durationMs) ? { durationMs: Number(raw.durationMs) } : {}),
    });
  }
  if (type === "combo") {
    const keys = normalizeComboKeys(raw?.keys);
    return withPreWait({
      type: "combo",
      keys,
      ...(Number.isFinite(raw.durationMs) ? { durationMs: Number(raw.durationMs) } : {}),
    });
  }
  if (type === "type") {
    return withPreWait({
      type: "type",
      selector: typeof raw.selector === "string" ? raw.selector : undefined,
      text: typeof raw.text === "string" ? raw.text : "",
    });
  }
  if (type === "wait") {
    return withPreWait({ type: "wait", ms: Number.isFinite(raw.ms) ? Number(raw.ms) : 0 });
  }
  throw new ActionValidationError(`Unsupported simulation action type: ${type || "(missing)"}`);
}

function normalizeComboKeys(rawKeys: unknown): string[] {
  if (!Array.isArray(rawKeys)) {
    throw new ActionValidationError("combo.keys must be an array of 1 or 2 keyboard key strings.");
  }
  const keys: string[] = [];
  for (const rawKey of rawKeys) {
    if (typeof rawKey !== "string" || !rawKey.trim()) {
      throw new ActionValidationError("combo.keys must contain only non-empty keyboard key strings.");
    }
    const key = rawKey.trim();
    if (!keys.includes(key)) keys.push(key);
  }
  if (keys.length < 1) {
    throw new ActionValidationError("combo.keys must contain at least 1 keyboard key.");
  }
  if (keys.length > 2) {
    throw new ActionValidationError("combo.keys must contain no more than 2 keyboard keys.");
  }
  return keys;
}
