import { createHash, randomUUID } from "node:crypto";
import {
  access,
  lstat,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { isDeepStrictEqual } from "node:util";
import { pathToFileURL } from "node:url";
import { PNG } from "pngjs";
import type { Browser, BrowserContext, CDPSession, Page } from "playwright";
import { resolveBrowser } from "../browser.js";
import {
  advanceClock,
  launchBrowser,
  moveMouseToViewportCenter,
  pauseClockNow,
  performAction,
} from "./playwright.js";
import type { ActionExecutionMetadata, SimulationAction } from "./types.js";
import { turnLabel, writeJson } from "./utils.js";

const TRACE_SCHEMA_VERSION = 1;
const DEFAULT_FRAME_MS = 50;
const REPLAY_VIEWPORT = { width: 1440, height: 900 } as const;
const REPLAY_SIMILARITY_THRESHOLD = 0.95;
const SOURCE_SNIPPET_LIMIT = 512;
const TRACEABLE_ACTIONS = new Set<SimulationAction["type"]>(["click", "mouseMove", "press", "combo"]);

export interface TraceAnalysisRequest {
  inputPath: string;
  threshold: number;
  frameMs?: number;
}

export interface TraceAnalysisRuntime {
  cwd: string;
  browser?: string;
  headed?: boolean;
}

export type TraceSkipReason = "below-threshold" | "unsupported-action";
export type ReplayReliability = "normal" | "degraded";

export interface SourcePosition {
  line: number;
  column: number;
}

export interface HtmlSourceRange {
  utf16Start: number;
  utf16End: number;
  utf8ByteStart: number;
  utf8ByteEnd: number;
  start: SourcePosition;
  end: SourcePosition;
  snippet: string;
  snippetTruncated: boolean;
}

export interface ExecutedBlock {
  executionCount: number;
  range: HtmlSourceRange;
}

export interface ExecutedFunction {
  functionName: string;
  executionCount: number;
  isBlockCoverage: boolean;
  range: HtmlSourceRange;
  blocks: ExecutedBlock[];
}

export interface MappedArtifactCoverage {
  functions: ExecutedFunction[];
  summary: {
    executedScriptCount: number;
    mappedScriptCount: number;
    excludedScriptCount: number;
    unmappedArtifactScriptCount: number;
    executedFunctionCount: number;
    executedRangeCount: number;
  };
  warnings: string[];
}

export interface TraceTurnResult {
  turnIndex: number;
  action: SimulationAction;
  transitionScore: number;
  selected: boolean;
  skipReasons: TraceSkipReason[];
  status: "pending" | "replayed" | "failed";
  originalExecutionSuccess?: boolean;
  replayExecution?: ActionExecutionMetadata;
  simulatedAdvanceMs: number;
  screenshotSimilarity?: number;
  replayReliability?: ReplayReliability;
  domPath?: string;
  codeTracePath?: string;
  coverageSummary?: MappedArtifactCoverage["summary"];
  warnings: string[];
}

export interface TraceAnalysisResult {
  schemaVersion: 1;
  ok: boolean;
  input: {
    requestedPath: string;
    runDir: string;
    traceDir: string;
    artifactPath: string;
    artifactSha256: string;
    threshold: number;
  };
  outputDir: string;
  sourceScope: "artifact-inline-main-frame-javascript";
  traceableActionTypes: Array<"click" | "mouseMove" | "press" | "combo">;
  replay: {
    browser: string;
    viewport: { width: number; height: number };
    frameMs: number;
    clockStartMs?: number;
    legacyDefaults: true;
    similarityThreshold: number;
    firstDriftTurn?: number;
  };
  totalTurnCount: number;
  selectedTurnCount: number;
  processedTurnCount: number;
  turns: TraceTurnResult[];
  browserMessages: Array<{ type: "console-error" | "page-error"; text: string }>;
  warnings: string[];
  errors: string[];
  startedAt: string;
  finishedAt?: string;
}

export interface RawCoverageRange {
  startOffset: number;
  endOffset: number;
  count: number;
}

export interface RawCoverageFunction {
  functionName: string;
  isBlockCoverage: boolean;
  ranges: RawCoverageRange[];
}

export interface RawScriptCoverage {
  scriptId: string;
  url: string;
  functions: RawCoverageFunction[];
}

export interface ScriptParsedMetadata {
  scriptId: string;
  url: string;
  startLine: number;
  startColumn: number;
  endLine?: number;
  endColumn?: number;
  /** False for iframe and isolated-world scripts observed by the page CDP session. */
  isMainFrameDefaultWorld?: boolean;
}

interface ResolvedTraceInput {
  requestedPath: string;
  runDir: string;
  traceDir: string;
  artifactPath: string;
  outputDir: string;
}

interface LoadedTraceTurn {
  turnIndex: number;
  action: SimulationAction;
  transitionScore: number;
  simulatedAdvanceMs: number;
  terminated: boolean;
  originalExecution?: Record<string, unknown>;
  originalAfterScreenshotPath: string;
}

interface LoadedTraceBundle {
  artifactText: string;
  artifactSha256: string;
  turns: LoadedTraceTurn[];
}

interface CoverageCapture {
  coverage: RawScriptCoverage[];
  scripts: Map<string, ScriptParsedMetadata>;
  sources: Map<string, string>;
}

class ReplayActionError extends Error {
  constructor(message: string, readonly metadata: ActionExecutionMetadata) {
    super(message);
    this.name = "ReplayActionError";
  }
}

export class TraceAnalysisRunError extends Error {
  constructor(message: string, readonly result: TraceAnalysisResult) {
    super(message);
    this.name = "TraceAnalysisRunError";
  }
}

export async function runTraceAnalysis(
  request: TraceAnalysisRequest,
  runtime: TraceAnalysisRuntime,
): Promise<TraceAnalysisResult> {
  const threshold = validateThreshold(request.threshold);
  const frameMs = validateFrameMs(request.frameMs ?? DEFAULT_FRAME_MS);
  const resolved = await resolveTraceInput(request.inputPath, runtime.cwd);
  const bundle = await loadTraceBundle(resolved);
  const browserChoice = resolveBrowser(runtime.browser);
  if (browserChoice.channel === "firefox") {
    throw new Error("agens trace requires a Chromium-family browser; Firefox does not expose CDP precise coverage.");
  }
  await assertReplaceableOutputDirectory(resolved.outputDir);

  const turns = bundle.turns.map((turn): TraceTurnResult => {
    const skipReasons = traceSkipReasons(turn.transitionScore, turn.action, threshold);
    return {
      turnIndex: turn.turnIndex,
      action: turn.action,
      transitionScore: turn.transitionScore,
      selected: skipReasons.length === 0,
      skipReasons,
      status: "pending",
      originalExecutionSuccess: finiteBoolean(turn.originalExecution?.success),
      simulatedAdvanceMs: turn.simulatedAdvanceMs,
      warnings: [],
    };
  });
  const selectedTurnCount = turns.filter((turn) => turn.selected).length;
  const startedAt = new Date().toISOString();
  const result: TraceAnalysisResult = {
    schemaVersion: TRACE_SCHEMA_VERSION,
    ok: false,
    input: {
      requestedPath: resolved.requestedPath,
      runDir: resolved.runDir,
      traceDir: resolved.traceDir,
      artifactPath: resolved.artifactPath,
      artifactSha256: bundle.artifactSha256,
      threshold,
    },
    outputDir: resolved.outputDir,
    sourceScope: "artifact-inline-main-frame-javascript",
    traceableActionTypes: ["click", "mouseMove", "press", "combo"],
    replay: {
      browser: browserChoice.label,
      viewport: { ...REPLAY_VIEWPORT },
      frameMs,
      legacyDefaults: true,
      similarityThreshold: REPLAY_SIMILARITY_THRESHOLD,
    },
    totalTurnCount: turns.length,
    selectedTurnCount,
    processedTurnCount: 0,
    turns,
    browserMessages: [],
    warnings: [
      "legacy-trace-replay-defaults: viewport=1440x900 and the configured frameMs are used because existing result.json files do not persist replay metadata.",
      "offline replay cannot reproduce unrecorded randomness; screenshot similarity is advisory.",
    ],
    errors: [],
    startedAt,
  };

  const stagingDir = await createStagingDirectory(resolved.outputDir);
  let browser: Browser | undefined;
  let context: BrowserContext | undefined;
  let collector: PreciseCoverageCollector | undefined;
  let runtimeFailure: Error | undefined;

  try {
    browser = await launchBrowser(runtime.browser, runtime.headed);
    context = await browser.newContext({ viewport: { ...REPLAY_VIEWPORT } });
    const clockStartMs = Date.now();
    result.replay.clockStartMs = clockStartMs;
    await context.clock.install({ time: clockStartMs });
    const page = await context.newPage();
    attachBrowserDiagnostics(page, result.browserMessages);
    collector = new PreciseCoverageCollector(
      await context.newCDPSession(page),
      pathToFileURL(resolved.artifactPath).href,
    );
    await collector.initialize();
    await page.goto(pathToFileURL(resolved.artifactPath).href, {
      waitUntil: "networkidle",
      timeout: 30_000,
    });
    await moveMouseToViewportCenter(page);

    let firstDriftTurn: number | undefined;
    for (let index = 0; index < bundle.turns.length; index++) {
      const loadedTurn = bundle.turns[index]!;
      const turnResult = result.turns[index]!;
      await pauseClockNow(context, page);

      let coverageCapture: CoverageCapture | undefined;
      let coverageStarted = false;
      try {
        if (turnResult.selected) {
          await collector.start();
          coverageStarted = true;
        }
        const replayExecution = await replayTurn(page, context, loadedTurn, frameMs);
        turnResult.replayExecution = replayExecution;
        if (coverageStarted) {
          coverageStarted = false;
          coverageCapture = await collector.stop();
        }
      } catch (error) {
        if (coverageStarted) await collector.stopIfActive().catch(() => undefined);
        turnResult.status = "failed";
        if (error instanceof ReplayActionError) turnResult.replayExecution = error.metadata;
        throw error;
      }

      const similarity = await compareReplayScreenshot(page, loadedTurn.originalAfterScreenshotPath, turnResult.warnings);
      if (similarity !== undefined) {
        turnResult.screenshotSimilarity = similarity;
        if (similarity < REPLAY_SIMILARITY_THRESHOLD && firstDriftTurn === undefined) {
          firstDriftTurn = loadedTurn.turnIndex;
          result.replay.firstDriftTurn = firstDriftTurn;
        }
      }
      turnResult.replayReliability = firstDriftTurn === undefined ? "normal" : "degraded";

      try {
        if (turnResult.selected && coverageCapture) {
          const mapped = mapArtifactCoverage({
            artifactText: bundle.artifactText,
            artifactUrl: pathToFileURL(resolved.artifactPath).href,
            coverage: coverageCapture.coverage,
            scripts: coverageCapture.scripts,
            sources: coverageCapture.sources,
          });
          turnResult.warnings.push(...mapped.warnings);
          turnResult.coverageSummary = mapped.summary;
          const label = turnLabel(loadedTurn.turnIndex);
          const domName = `${label}-dom.html`;
          const codeTraceName = `${label}-code-trace.json`;
          const dom = await page.content();
          await writeFile(path.join(stagingDir, domName), dom, "utf8");
          await writeJson(path.join(stagingDir, codeTraceName), {
            schemaVersion: TRACE_SCHEMA_VERSION,
            turnIndex: loadedTurn.turnIndex,
            action: loadedTurn.action,
            transitionScore: loadedTurn.transitionScore,
            threshold,
            replayReliability: turnResult.replayReliability,
            screenshotSimilarity: turnResult.screenshotSimilarity,
            source: {
              path: resolved.artifactPath,
              sha256: bundle.artifactSha256,
              offsetEncoding: "UTF-16 code units plus UTF-8 byte offsets",
              lineColumnBase: 1,
              snippetLimit: SOURCE_SNIPPET_LIMIT,
            },
            window: transitionWindow(loadedTurn, turnResult.replayExecution),
            functions: mapped.functions,
            coverageSummary: mapped.summary,
            warnings: turnResult.warnings,
          });
          turnResult.domPath = domName;
          turnResult.codeTracePath = codeTraceName;
        }
      } catch (error) {
        turnResult.status = "failed";
        throw error;
      }

      turnResult.status = "replayed";
      result.processedTurnCount += 1;

      if (loadedTurn.terminated) break;
    }
    result.ok = true;
  } catch (error) {
    runtimeFailure = toError(error);
    result.errors.push(runtimeFailure.message);
  } finally {
    if (collector) await collector.dispose().catch((error) => result.warnings.push(`coverage-dispose: ${toError(error).message}`));
    if (context) await context.close().catch((error) => result.warnings.push(`context-close: ${toError(error).message}`));
    if (browser) await browser.close().catch((error) => result.warnings.push(`browser-close: ${toError(error).message}`));
  }

  result.finishedAt = new Date().toISOString();
  await writeJson(path.join(stagingDir, "trace-result.json"), result);
  await replaceOutputDirectory(stagingDir, resolved.outputDir);

  if (runtimeFailure) throw new TraceAnalysisRunError(runtimeFailure.message, result);
  return result;
}

export function traceSkipReasons(
  transitionScore: number,
  action: SimulationAction,
  threshold: number,
): TraceSkipReason[] {
  const reasons: TraceSkipReason[] = [];
  if (transitionScore < threshold) reasons.push("below-threshold");
  if (!TRACEABLE_ACTIONS.has(action.type)) reasons.push("unsupported-action");
  return reasons;
}

export function mapArtifactCoverage(args: {
  artifactText: string;
  artifactUrl: string;
  coverage: RawScriptCoverage[];
  scripts: Map<string, ScriptParsedMetadata>;
  sources: Map<string, string>;
}): MappedArtifactCoverage {
  const lineStarts = buildLineStarts(args.artifactText);
  const functions: ExecutedFunction[] = [];
  const warnings: string[] = [];
  let executedScriptCount = 0;
  let mappedScriptCount = 0;
  let excludedScriptCount = 0;
  let unmappedArtifactScriptCount = 0;

  for (const scriptCoverage of args.coverage) {
    const executedFunctions = scriptCoverage.functions.filter(hasExecutedRange);
    if (!executedFunctions.length) continue;
    executedScriptCount += 1;
    if (!sameResourceUrl(scriptCoverage.url, args.artifactUrl)) {
      excludedScriptCount += 1;
      continue;
    }

    const metadata = args.scripts.get(scriptCoverage.scriptId);
    const source = args.sources.get(scriptCoverage.scriptId);
    if (metadata?.isMainFrameDefaultWorld === false) {
      excludedScriptCount += 1;
      continue;
    }
    const scriptStart = metadata
      ? offsetAtPosition(lineStarts, metadata.startLine, metadata.startColumn, args.artifactText.length)
      : undefined;
    if (
      !metadata ||
      source === undefined ||
      scriptStart === undefined ||
      args.artifactText.slice(scriptStart, scriptStart + source.length) !== source
    ) {
      unmappedArtifactScriptCount += 1;
      warnings.push(`unmapped-artifact-script: scriptId=${scriptCoverage.scriptId}`);
      continue;
    }

    mappedScriptCount += 1;
    for (const rawFunction of executedFunctions) {
      const positiveRanges = rawFunction.ranges.filter((range) => range.count > 0);
      const outer = positiveRanges[0];
      if (!outer) continue;
      const mappedOuter = mapRange(args.artifactText, lineStarts, source, scriptStart, outer);
      if (!mappedOuter) {
        warnings.push(`unmapped-function-range: scriptId=${scriptCoverage.scriptId} function=${rawFunction.functionName || "<anonymous>"}`);
        continue;
      }
      const blocks: ExecutedBlock[] = [];
      for (const block of positiveRanges.slice(1)) {
        const mappedBlock = mapRange(args.artifactText, lineStarts, source, scriptStart, block);
        if (mappedBlock) blocks.push({ executionCount: block.count, range: mappedBlock });
      }
      functions.push({
        functionName: rawFunction.functionName || "<anonymous>",
        executionCount: outer.count,
        isBlockCoverage: rawFunction.isBlockCoverage,
        range: mappedOuter,
        blocks: blocks.sort(compareExecutedBlocks),
      });
    }
  }

  functions.sort(compareExecutedFunctions);

  return {
    functions,
    summary: {
      executedScriptCount,
      mappedScriptCount,
      excludedScriptCount,
      unmappedArtifactScriptCount,
      executedFunctionCount: functions.length,
      executedRangeCount: functions.reduce((sum, fn) => sum + 1 + fn.blocks.length, 0),
    },
    warnings,
  };
}

function compareExecutedFunctions(left: ExecutedFunction, right: ExecutedFunction): number {
  return left.range.utf16Start - right.range.utf16Start ||
    left.range.utf16End - right.range.utf16End ||
    left.functionName.localeCompare(right.functionName);
}

function compareExecutedBlocks(left: ExecutedBlock, right: ExecutedBlock): number {
  return left.range.utf16Start - right.range.utf16Start ||
    left.range.utf16End - right.range.utf16End;
}

export function rgbSimilarity(
  width: number,
  height: number,
  first: Uint8Array,
  second: Uint8Array,
): number {
  const expectedLength = width * height * 4;
  if (width <= 0 || height <= 0 || first.length !== expectedLength || second.length !== expectedLength) return 0;
  let difference = 0;
  for (let i = 0; i < expectedLength; i += 4) {
    difference += Math.abs(first[i]! - second[i]!);
    difference += Math.abs(first[i + 1]! - second[i + 1]!);
    difference += Math.abs(first[i + 2]! - second[i + 2]!);
  }
  const similarity = 1 - difference / (width * height * 3 * 255);
  return Math.round(Math.max(0, Math.min(1, similarity)) * 1_000_000) / 1_000_000;
}

class PreciseCoverageCollector {
  private readonly scripts = new Map<string, ScriptParsedMetadata>();
  private readonly executionContexts = new Map<number, { frameId?: string; isDefault?: boolean }>();
  private active = false;
  private initialized = false;
  private mainFrameId?: string;

  constructor(
    private readonly session: CDPSession,
    private readonly artifactUrl: string,
  ) {}

  async initialize(): Promise<void> {
    this.session.on("Page.frameNavigated", (event: { frame?: { id?: string; parentId?: string } }) => {
      if (event.frame?.id && event.frame.parentId === undefined) this.mainFrameId = event.frame.id;
    });
    this.session.on("Runtime.executionContextCreated", (event: {
      context?: { id?: number; auxData?: { frameId?: string; isDefault?: boolean } };
    }) => {
      if (typeof event.context?.id === "number") {
        this.executionContexts.set(event.context.id, event.context.auxData ?? {});
      }
    });
    this.session.on("Runtime.executionContextDestroyed", (event: { executionContextId?: number }) => {
      if (typeof event.executionContextId === "number") this.executionContexts.delete(event.executionContextId);
    });
    this.session.on("Runtime.executionContextsCleared", () => this.executionContexts.clear());
    this.session.on("Debugger.scriptParsed", (event: ScriptParsedMetadata & {
      executionContextId?: number;
      executionContextAuxData?: { frameId?: string; isDefault?: boolean };
    }) => {
      const context = event.executionContextAuxData ?? (
        typeof event.executionContextId === "number"
          ? this.executionContexts.get(event.executionContextId)
          : undefined
      );
      this.scripts.set(event.scriptId, {
        scriptId: event.scriptId,
        url: event.url,
        startLine: event.startLine,
        startColumn: event.startColumn,
        endLine: event.endLine,
        endColumn: event.endColumn,
        isMainFrameDefaultWorld:
          context?.frameId === this.mainFrameId && context?.isDefault === true,
      });
    });
    await this.session.send("Page.enable");
    const frameTree = await this.session.send("Page.getFrameTree") as {
      frameTree?: { frame?: { id?: string } };
    };
    this.mainFrameId = frameTree.frameTree?.frame?.id;
    await this.session.send("Runtime.enable");
    await this.session.send("Debugger.enable");
    await this.session.send("Profiler.enable");
    this.initialized = true;
  }

  async start(): Promise<void> {
    if (!this.initialized) throw new Error("coverage collector was not initialized");
    if (this.active) throw new Error("coverage collection is already active");
    await this.session.send("Profiler.startPreciseCoverage", {
      callCount: true,
      detailed: true,
      allowTriggeredUpdates: false,
    });
    this.active = true;
  }

  async stop(): Promise<CoverageCapture> {
    if (!this.active) throw new Error("coverage collection is not active");
    try {
      const response = await this.session.send("Profiler.takePreciseCoverage") as { result?: RawScriptCoverage[] };
      const coverage = Array.isArray(response.result) ? response.result : [];
      const sources = new Map<string, string>();
      for (const scriptCoverage of coverage) {
        if (!hasExecutedScript(scriptCoverage) || !sameResourceUrl(scriptCoverage.url, this.artifactUrl)) continue;
        try {
          const source = await this.session.send("Debugger.getScriptSource", {
            scriptId: scriptCoverage.scriptId,
          }) as { scriptSource?: string };
          if (typeof source.scriptSource === "string") sources.set(scriptCoverage.scriptId, source.scriptSource);
        } catch {
          // The mapper will report this artifact script as unmapped.
        }
      }
      return {
        coverage,
        scripts: new Map(this.scripts),
        sources,
      };
    } finally {
      this.active = false;
      await this.session.send("Profiler.stopPreciseCoverage").catch(() => undefined);
    }
  }

  async stopIfActive(): Promise<void> {
    if (!this.active) return;
    this.active = false;
    await this.session.send("Profiler.stopPreciseCoverage");
  }

  async dispose(): Promise<void> {
    await this.stopIfActive().catch(() => undefined);
    if (!this.initialized) return;
    this.initialized = false;
    await this.session.send("Profiler.disable").catch(() => undefined);
    await this.session.send("Debugger.disable").catch(() => undefined);
    await this.session.send("Runtime.disable").catch(() => undefined);
    await this.session.send("Page.disable").catch(() => undefined);
    await this.session.detach().catch(() => undefined);
  }
}

async function resolveTraceInput(rawInputPath: string, cwd: string): Promise<ResolvedTraceInput> {
  const requestedPath = path.resolve(cwd, rawInputPath);
  const inputStat = await stat(requestedPath).catch((error) => {
    throw new Error(`Trace input does not exist at ${requestedPath}: ${toError(error).message}`);
  });
  if (!inputStat.isDirectory()) throw new Error(`Trace input must be a directory: ${requestedPath}`);

  let runDir: string;
  let traceDir: string;
  if (await pathExists(path.join(requestedPath, "artifact", "index.html"))) {
    runDir = requestedPath;
    traceDir = await resolveTraceDirectoryFromRun(runDir);
  } else if (await pathExists(path.join(requestedPath, "result.json"))) {
    traceDir = requestedPath;
    const traceRoot = path.dirname(traceDir);
    if (path.basename(traceRoot) !== "trace") {
      throw new Error(`A concrete trace directory must be located under <run>/trace/: ${traceDir}`);
    }
    runDir = path.dirname(traceRoot);
  } else {
    throw new Error(`Expected a reward run directory or concrete trace directory: ${requestedPath}`);
  }

  const artifactPath = path.join(runDir, "artifact", "index.html");
  if (!await pathExists(artifactPath)) throw new Error(`Missing artifact HTML: ${artifactPath}`);
  const traceId = path.basename(traceDir);
  if (!traceId || traceId === "." || traceId === "..") throw new Error(`Invalid trace directory name: ${traceDir}`);
  return {
    requestedPath,
    runDir,
    traceDir,
    artifactPath,
    outputDir: path.join(runDir, "trace-analysis", traceId),
  };
}

async function resolveTraceDirectoryFromRun(runDir: string): Promise<string> {
  const traceRoot = path.join(runDir, "trace");
  const candidates = await listTraceDirectories(traceRoot);
  const rewardResultPath = path.join(runDir, "reward-result.json");
  if (await pathExists(rewardResultPath)) {
    const rewardResult = await readJsonObject(rewardResultPath);
    const recorded = nestedString(rewardResult, "paths", "runDir");
    if (recorded) {
      const relocated = candidates.find((candidate) => path.basename(candidate) === path.basename(recorded));
      if (relocated) return relocated;
      const exact = path.resolve(runDir, recorded);
      if (await pathExists(path.join(exact, "result.json"))) return exact;
    }
  }
  if (candidates.length === 1) return candidates[0]!;
  if (!candidates.length) throw new Error(`No trace/*/result.json found under ${runDir}`);
  throw new Error(`Multiple trace directories found under ${runDir}; pass the concrete trace/<timestamp> directory.`);
}

async function listTraceDirectories(traceRoot: string): Promise<string[]> {
  const entries = await readdir(traceRoot, { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return [];
    throw error;
  });
  const candidates: string[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const candidate = path.join(traceRoot, entry.name);
    if (await pathExists(path.join(candidate, "result.json"))) candidates.push(candidate);
  }
  return candidates.sort();
}

async function loadTraceBundle(resolved: ResolvedTraceInput): Promise<LoadedTraceBundle> {
  const artifactText = await readFile(resolved.artifactPath, "utf8");
  const artifactSha256 = createHash("sha256").update(artifactText).digest("hex");
  const resultPath = path.join(resolved.traceDir, "result.json");
  const rawResult = await readJsonObject(resultPath);
  if (!Array.isArray(rawResult.turns)) throw new Error(`${resultPath} must contain a turns array.`);
  const turns: LoadedTraceTurn[] = [];

  for (let index = 0; index < rawResult.turns.length; index++) {
    const rawTurn = plainObject(rawResult.turns[index], `${resultPath} turns[${index}]`);
    const turnIndex = requiredInteger(rawTurn.turnIndex, `${resultPath} turns[${index}].turnIndex`);
    if (turnIndex !== index) throw new Error(`${resultPath} turn indices must be contiguous from 0; found ${turnIndex} at position ${index}.`);
    const label = turnLabel(turnIndex);
    const decisionPath = path.join(resolved.traceDir, `${label}-decision.json`);
    const evaluationPath = path.join(resolved.traceDir, `${label}-transition-evaluation.json`);
    const decision = await readJsonObject(decisionPath);
    const evaluation = await readJsonObject(evaluationPath);
    const action = parseSimulationAction(decision.action, `${decisionPath} action`);
    const resultAction = parseSimulationAction(rawTurn.action, `${resultPath} turns[${index}].action`);
    if (!isDeepStrictEqual(action, resultAction)) {
      throw new Error(`Action mismatch between ${decisionPath} and result.json turn ${turnIndex}.`);
    }
    const evaluationTurn = evaluation.turnIndex;
    if (evaluationTurn !== undefined && requiredInteger(evaluationTurn, `${evaluationPath} turnIndex`) !== turnIndex) {
      throw new Error(`Turn index mismatch in ${evaluationPath}.`);
    }
    const transitionScore = requiredFiniteNumber(evaluation.score, `${evaluationPath} score`);
    if (transitionScore < 0 || transitionScore > 100) throw new Error(`${evaluationPath} score must be between 0 and 100.`);
    const simulatedAdvanceMs = requiredFiniteNumber(
      rawTurn.simulatedAdvanceMs,
      `${resultPath} turns[${index}].simulatedAdvanceMs`,
    );
    if (simulatedAdvanceMs < 0) throw new Error(`${resultPath} turn ${turnIndex} simulatedAdvanceMs must be non-negative.`);
    const terminated = rawTurn.terminated === true || action.type === "terminate";
    if (terminated && index !== rawResult.turns.length - 1) throw new Error(`Terminate action at turn ${turnIndex} must be the final turn.`);
    turns.push({
      turnIndex,
      action,
      transitionScore,
      simulatedAdvanceMs,
      terminated,
      originalExecution: isRecord(rawTurn.actionExecution) ? rawTurn.actionExecution : undefined,
      originalAfterScreenshotPath: path.join(resolved.traceDir, `${label}-after.png`),
    });
  }
  return { artifactText, artifactSha256, turns };
}

async function replayTurn(
  page: Page,
  context: BrowserContext,
  turn: LoadedTraceTurn,
  frameMs: number,
): Promise<ActionExecutionMetadata> {
  const preWaitMs = finiteNonNegative(turn.action.type === "terminate" ? undefined : turn.action.waitMs) ?? 0;
  if (preWaitMs > 0) await advanceClock(context, page, preWaitMs, frameMs);
  const executableAction = actionForReplay(turn.action, turn.originalExecution);
  const execution = await performAction(page, context, executableAction, frameMs);
  execution.metadata.preWaitMs = preWaitMs || undefined;
  if (execution.metadata.success !== true) {
    throw new ReplayActionError(
      `Replay action failed at turn ${turn.turnIndex}: ${execution.metadata.executionError?.message ?? "unknown action error"}`,
      execution.metadata,
    );
  }
  const consumedMs = preWaitMs + execution.advanceMs;
  const remainingMs = turn.simulatedAdvanceMs - consumedMs;
  if (remainingMs < -0.5) {
    throw new ReplayActionError(
      `Replay timing exceeded recorded simulatedAdvanceMs at turn ${turn.turnIndex}: consumed=${consumedMs}, recorded=${turn.simulatedAdvanceMs}.`,
      execution.metadata,
    );
  }
  if (remainingMs > 0) await advanceClock(context, page, remainingMs, frameMs);
  return execution.metadata;
}

function actionForReplay(action: SimulationAction, originalExecution?: Record<string, unknown>): SimulationAction {
  if (action.type === "terminate") return action;
  const { waitMs: _waitMs, ...withoutWait } = action;
  const recordedDuration = finiteNonNegative(originalExecution?.durationMs);
  const hadExplicitDuration = "durationMs" in action && action.durationMs !== undefined;
  if (
    recordedDuration !== undefined &&
    (recordedDuration > 0 || hadExplicitDuration) &&
    (withoutWait.type === "click" || withoutWait.type === "press" || withoutWait.type === "combo" || withoutWait.type === "mouseMove")
  ) {
    return { ...withoutWait, durationMs: recordedDuration } as SimulationAction;
  }
  return withoutWait as SimulationAction;
}

function transitionWindow(turn: LoadedTraceTurn, execution?: ActionExecutionMetadata): Record<string, unknown> {
  const preWaitMs = execution?.preWaitMs ?? 0;
  const actionDurationMs = execution?.durationMs ?? (turn.action.type === "wait" ? turn.action.ms ?? 0 : 0);
  return {
    includesPreWait: true,
    includesActionDuration: true,
    includesAfterDelay: true,
    preWaitMs,
    actionDurationMs,
    afterDelayMs: Math.max(0, turn.simulatedAdvanceMs - preWaitMs - actionDurationMs),
    simulatedAdvanceMs: turn.simulatedAdvanceMs,
  };
}

async function compareReplayScreenshot(
  page: Page,
  originalPath: string,
  warnings: string[],
): Promise<number | undefined> {
  if (!await pathExists(originalPath)) {
    warnings.push(`replay-similarity-skipped: missing ${originalPath}`);
    return undefined;
  }
  try {
    const [originalBuffer, replayBuffer] = await Promise.all([
      readFile(originalPath),
      page.screenshot({ fullPage: true }),
    ]);
    const original = PNG.sync.read(originalBuffer);
    const replay = PNG.sync.read(replayBuffer);
    if (original.width !== replay.width || original.height !== replay.height) {
      warnings.push(
        `replay-screenshot-size-mismatch: original=${original.width}x${original.height} replay=${replay.width}x${replay.height}`,
      );
      return 0;
    }
    return rgbSimilarity(original.width, original.height, original.data, replay.data);
  } catch (error) {
    warnings.push(`replay-similarity-skipped: ${toError(error).message}`);
    return undefined;
  }
}

function mapRange(
  artifactText: string,
  lineStarts: number[],
  source: string,
  scriptStart: number,
  raw: RawCoverageRange,
): HtmlSourceRange | undefined {
  if (
    !Number.isInteger(raw.startOffset) ||
    !Number.isInteger(raw.endOffset) ||
    raw.startOffset < 0 ||
    raw.endOffset < raw.startOffset ||
    raw.endOffset > source.length
  ) return undefined;
  const utf16Start = scriptStart + raw.startOffset;
  const utf16End = scriptStart + raw.endOffset;
  if (utf16End > artifactText.length) return undefined;
  const fullSnippet = artifactText.slice(utf16Start, utf16End);
  return {
    utf16Start,
    utf16End,
    utf8ByteStart: Buffer.byteLength(artifactText.slice(0, utf16Start), "utf8"),
    utf8ByteEnd: Buffer.byteLength(artifactText.slice(0, utf16End), "utf8"),
    start: positionAtOffset(lineStarts, utf16Start),
    end: positionAtOffset(lineStarts, utf16End),
    snippet: fullSnippet.slice(0, SOURCE_SNIPPET_LIMIT),
    snippetTruncated: fullSnippet.length > SOURCE_SNIPPET_LIMIT,
  };
}

function buildLineStarts(text: string): number[] {
  const starts = [0];
  for (let index = 0; index < text.length; index++) {
    if (text.charCodeAt(index) === 10) starts.push(index + 1);
  }
  return starts;
}

function offsetAtPosition(
  lineStarts: number[],
  zeroBasedLine: number,
  zeroBasedColumn: number,
  textLength: number,
): number | undefined {
  if (!Number.isInteger(zeroBasedLine) || !Number.isInteger(zeroBasedColumn) || zeroBasedLine < 0 || zeroBasedColumn < 0) {
    return undefined;
  }
  const lineStart = lineStarts[zeroBasedLine];
  if (lineStart === undefined) return undefined;
  const offset = lineStart + zeroBasedColumn;
  return offset <= textLength ? offset : undefined;
}

function positionAtOffset(lineStarts: number[], offset: number): SourcePosition {
  let low = 0;
  let high = lineStarts.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (lineStarts[middle]! <= offset) low = middle + 1;
    else high = middle - 1;
  }
  const lineIndex = Math.max(0, high);
  return {
    line: lineIndex + 1,
    column: offset - lineStarts[lineIndex]! + 1,
  };
}

function hasExecutedRange(fn: RawCoverageFunction): boolean {
  return fn.ranges.some((range) => Number.isFinite(range.count) && range.count > 0);
}

function hasExecutedScript(script: RawScriptCoverage): boolean {
  return script.functions.some(hasExecutedRange);
}

function sameResourceUrl(first: string, second: string): boolean {
  try {
    const left = new URL(first);
    const right = new URL(second);
    left.hash = "";
    right.hash = "";
    return left.href === right.href;
  } catch {
    return first === second;
  }
}

function parseSimulationAction(value: unknown, label: string): SimulationAction {
  const raw = plainObject(value, label);
  if (typeof raw.type !== "string") throw new Error(`${label}.type must be a string.`);
  validateOptionalWait(raw.waitMs, `${label}.waitMs`);
  switch (raw.type) {
    case "click":
      validateOptionalString(raw.selector, `${label}.selector`);
      validateOptionalString(raw.text, `${label}.text`);
      validateOptionalNumber(raw.x, `${label}.x`);
      validateOptionalNumber(raw.y, `${label}.y`);
      validateOptionalDuration(raw.durationMs, `${label}.durationMs`);
      if (!raw.selector && !raw.text && !(Number.isFinite(raw.x) && Number.isFinite(raw.y))) {
        throw new Error(`${label} click requires selector, text, or x/y coordinates.`);
      }
      return raw as unknown as SimulationAction;
    case "hover":
      validateOptionalString(raw.selector, `${label}.selector`);
      validateOptionalString(raw.text, `${label}.text`);
      validateOptionalNumber(raw.x, `${label}.x`);
      validateOptionalNumber(raw.y, `${label}.y`);
      if (!raw.selector && !raw.text && !(Number.isFinite(raw.x) && Number.isFinite(raw.y))) {
        throw new Error(`${label} hover requires selector, text, or x/y coordinates.`);
      }
      return raw as unknown as SimulationAction;
    case "mouseMove":
      requiredFiniteNumber(raw.dx, `${label}.dx`);
      requiredFiniteNumber(raw.dy, `${label}.dy`);
      validateOptionalDuration(raw.durationMs, `${label}.durationMs`);
      if (raw.steps !== undefined && requiredInteger(raw.steps, `${label}.steps`) <= 0) {
        throw new Error(`${label}.steps must be positive.`);
      }
      return raw as unknown as SimulationAction;
    case "press":
      if (typeof raw.key !== "string" || !raw.key.trim()) throw new Error(`${label}.key must be a non-empty string.`);
      validateOptionalDuration(raw.durationMs, `${label}.durationMs`);
      return raw as unknown as SimulationAction;
    case "combo":
      if (!Array.isArray(raw.keys) || raw.keys.length < 1 || raw.keys.length > 2 || raw.keys.some((key) => typeof key !== "string" || !key.trim())) {
        throw new Error(`${label}.keys must contain 1 or 2 non-empty strings.`);
      }
      validateOptionalDuration(raw.durationMs, `${label}.durationMs`);
      return raw as unknown as SimulationAction;
    case "type":
      if (typeof raw.selector !== "string" || !raw.selector) throw new Error(`${label}.selector must be a non-empty string.`);
      if (typeof raw.text !== "string") throw new Error(`${label}.text must be a string.`);
      return raw as unknown as SimulationAction;
    case "wait":
      validateOptionalDuration(raw.ms, `${label}.ms`);
      return raw as unknown as SimulationAction;
    case "terminate":
      validateOptionalString(raw.reason, `${label}.reason`);
      return raw as unknown as SimulationAction;
    default:
      throw new Error(`${label}.type is unsupported: ${raw.type}`);
  }
}

function attachBrowserDiagnostics(
  page: Page,
  messages: TraceAnalysisResult["browserMessages"],
): void {
  const push = (type: "console-error" | "page-error", text: string): void => {
    if (messages.length >= 100) return;
    messages.push({ type, text: text.slice(0, 2_000) });
  };
  page.on("pageerror", (error) => push("page-error", error.message));
  page.on("console", (message) => {
    if (message.type() === "error") push("console-error", message.text());
  });
}

async function createStagingDirectory(outputDir: string): Promise<string> {
  const parent = path.dirname(outputDir);
  await mkdir(parent, { recursive: true });
  const staging = path.join(parent, `.${path.basename(outputDir)}.tmp-${process.pid}-${randomUUID()}`);
  await mkdir(staging);
  return staging;
}

async function assertReplaceableOutputDirectory(outputDir: string): Promise<void> {
  const info = await lstat(outputDir).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return undefined;
    throw error;
  });
  if (!info) return;
  if (info.isSymbolicLink()) throw new Error(`Refusing to replace symlinked trace-analysis output: ${outputDir}`);
  if (!info.isDirectory()) throw new Error(`Trace-analysis output exists and is not a directory: ${outputDir}`);
}

async function replaceOutputDirectory(stagingDir: string, outputDir: string): Promise<void> {
  const parent = path.dirname(outputDir);
  const backup = path.join(parent, `.${path.basename(outputDir)}.bak-${process.pid}-${randomUUID()}`);
  const hadExisting = await pathExists(outputDir);
  if (hadExisting) await rename(outputDir, backup);
  try {
    await rename(stagingDir, outputDir);
  } catch (error) {
    if (hadExisting && !await pathExists(outputDir)) await rename(backup, outputDir).catch(() => undefined);
    throw error;
  }
  if (hadExisting) await rm(backup, { recursive: true, force: true });
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonObject(filePath: string): Promise<Record<string, unknown>> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    throw new Error(`Failed to read JSON at ${filePath}: ${toError(error).message}`);
  }
  return plainObject(parsed, filePath);
}

function plainObject(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${label} must be a JSON object.`);
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nestedString(value: Record<string, unknown>, first: string, second: string): string | undefined {
  const nested = value[first];
  if (!isRecord(nested)) return undefined;
  return typeof nested[second] === "string" ? nested[second] : undefined;
}

function requiredInteger(value: unknown, label: string): number {
  const number = requiredFiniteNumber(value, label);
  if (!Number.isInteger(number)) throw new Error(`${label} must be an integer.`);
  return number;
}

function requiredFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} must be a finite number.`);
  return value;
}

function validateOptionalString(value: unknown, label: string): void {
  if (value !== undefined && typeof value !== "string") throw new Error(`${label} must be a string when present.`);
}

function validateOptionalNumber(value: unknown, label: string): void {
  if (value !== undefined) requiredFiniteNumber(value, label);
}

function validateOptionalDuration(value: unknown, label: string): void {
  if (value === undefined) return;
  if (requiredFiniteNumber(value, label) < 0) throw new Error(`${label} must be non-negative.`);
}

function validateOptionalWait(value: unknown, label: string): void {
  validateOptionalDuration(value, label);
}

function finiteNonNegative(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function finiteBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function validateThreshold(value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 100) throw new Error("threshold must be between 0 and 100.");
  return value;
}

function validateFrameMs(value: number): number {
  if (!Number.isInteger(value) || value <= 0 || value > 1_000) throw new Error("frameMs must be an integer between 1 and 1000.");
  return value;
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}
