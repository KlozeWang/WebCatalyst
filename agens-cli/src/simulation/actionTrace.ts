import { createHash } from "node:crypto";
import type { CDPSession, Page } from "playwright";
import type {
  ActionTraceByteRun,
  ActionTraceTurnSummary,
  OnlineActionTraceAttribution,
  OnlineActionTraceConfig,
  SimulationAction,
  UserSimulationTurn,
} from "./types.js";

const TRACE_SCHEMA_VERSION = 1 as const;
const DEFAULT_MAX_PAUSE_EVENTS_PER_ACTION = 4096;
const TRACEABLE_ACTIONS = new Set<SimulationAction["type"]>(["click", "mouseMove", "press", "combo"]);

interface ScriptParsedEvent {
  scriptId: string;
  url: string;
  startLine: number;
  startColumn: number;
  executionContextId?: number;
  executionContextAuxData?: { frameId?: string; isDefault?: boolean };
}

interface ScriptRecord {
  scriptId: string;
  url: string;
  startLine: number;
  startColumn: number;
  isMainFrameDefaultWorld: boolean;
  source?: string;
  sourceLineStarts?: number[];
  htmlUtf16Start?: number;
}

interface RawCoverageRange {
  startOffset: number;
  endOffset: number;
  count: number;
}

interface RawCoverageFunction {
  functionName: string;
  ranges: RawCoverageRange[];
}

interface RawScriptCoverage {
  scriptId: string;
  url: string;
  functions: RawCoverageFunction[];
}

interface FunctionRecord {
  key: string;
  scriptId: string;
  functionName: string;
  sourceStart: number;
  sourceEnd: number;
  htmlUtf16Start: number;
  htmlUtf16End: number;
  breakpointInstalled: boolean;
}

interface PausedCallFrame {
  location?: { scriptId?: string; lineNumber?: number; columnNumber?: number };
}

interface CapturedTurnTrace {
  turnIndex: number;
  actionType: SimulationAction["type"];
  pauseEventCount: number;
  stacks: string[][];
  instrumentedFunctionKeys: Set<string>;
  executedFunctionKeys: Set<string>;
  partial: boolean;
  warnings: string[];
  errors: string[];
}

interface DepthRange {
  utf16Start: number;
  utf16End: number;
  depth: number;
}

/** Whether an action participates in online action tracing. */
export function isOnlineTraceableAction(action: SimulationAction): boolean {
  return TRACEABLE_ACTIONS.has(action.type);
}

export function validateOnlineActionTraceConfig(config: OnlineActionTraceConfig): Required<OnlineActionTraceConfig> {
  if (!Number.isFinite(config.threshold) || config.threshold < 0 || config.threshold > 100) {
    throw new Error("actionTrace.threshold must be a finite number between 0 and 100");
  }
  const cap = config.maxPauseEventsPerAction ?? DEFAULT_MAX_PAUSE_EVENTS_PER_ACTION;
  if (!Number.isInteger(cap) || cap <= 0) {
    throw new Error("actionTrace.maxPauseEventsPerAction must be a positive integer");
  }
  return {
    enabled: config.enabled ?? true,
    threshold: config.threshold,
    maxPauseEventsPerAction: cap,
  };
}

/**
 * Remove duplicate and shallower breakpoint captures. CDP pauses once at each
 * instrumented function entry, so a single leaf call commonly yields
 * [leaf, caller] and [caller]. Only the maximal chain should assign depths.
 */
export function reduceMaximalCallStacks(stacks: readonly (readonly string[])[]): string[][] {
  const unique = new Map<string, string[]>();
  for (const stack of stacks) {
    if (!stack.length) continue;
    unique.set(JSON.stringify(stack), [...stack]);
  }
  const ordered = [...unique.values()].sort((left, right) => right.length - left.length);
  return ordered.filter((candidate, index) => !ordered.slice(0, index).some((longer) =>
    longer.length > candidate.length && isSuffix(candidate, longer),
  ));
}

/** Merge overlapping source ranges, retaining the minimum call depth. */
export function mergeDepthRanges(ranges: readonly DepthRange[], artifactText: string): ActionTraceByteRun[] {
  const events = new Map<number, { add: number[]; remove: number[] }>();
  for (const range of ranges) {
    const start = Math.max(0, Math.min(artifactText.length, range.utf16Start));
    const end = Math.max(start, Math.min(artifactText.length, range.utf16End));
    if (start === end || !Number.isInteger(range.depth) || range.depth < 0) continue;
    const startEvent = events.get(start) ?? { add: [], remove: [] };
    startEvent.add.push(range.depth);
    events.set(start, startEvent);
    const endEvent = events.get(end) ?? { add: [], remove: [] };
    endEvent.remove.push(range.depth);
    events.set(end, endEvent);
  }
  if (!events.size) return [];

  const positions = [...events.keys()].sort((left, right) => left - right);
  const active = new Map<number, number>();
  const utf16Runs: DepthRange[] = [];
  let previous = positions[0]!;
  for (const position of positions) {
    if (position > previous && active.size) {
      const depth = Math.min(...active.keys());
      const prior = utf16Runs.at(-1);
      if (prior && prior.utf16End === previous && prior.depth === depth) {
        prior.utf16End = position;
      } else {
        utf16Runs.push({ utf16Start: previous, utf16End: position, depth });
      }
    }
    const event = events.get(position)!;
    for (const depth of event.remove) {
      const count = active.get(depth) ?? 0;
      if (count <= 1) active.delete(depth);
      else active.set(depth, count - 1);
    }
    for (const depth of event.add) active.set(depth, (active.get(depth) ?? 0) + 1);
    previous = position;
  }

  const boundaryOffsets = utf8OffsetsAt(
    artifactText,
    [...new Set(utf16Runs.flatMap((run) => [run.utf16Start, run.utf16End]))],
  );
  return utf16Runs.map((run) => ({
    utf8ByteStart: boundaryOffsets.get(run.utf16Start)!,
    utf8ByteEnd: boundaryOffsets.get(run.utf16End)!,
    depth: run.depth,
  }));
}

/**
 * Collect exact JavaScript call chains in the browser used by simulation.
 * Breakpoint failures are intentionally recorded as attribution failures and
 * never allowed to fail the simulation reward itself.
 */
export class OnlineActionTraceCollector {
  private readonly session: CDPSession;
  private readonly config: Required<OnlineActionTraceConfig>;
  private readonly artifactLineStarts: number[];
  private readonly artifactSha256: string;
  private readonly scripts = new Map<string, ScriptRecord>();
  private readonly functions = new Map<string, FunctionRecord>();
  private readonly functionsByScript = new Map<string, FunctionRecord[]>();
  private readonly executionContexts = new Map<number, { frameId?: string; isDefault?: boolean }>();
  private readonly captures = new Map<number, CapturedTurnTrace>();
  private readonly setupWarnings: string[] = [];
  private readonly setupErrors: string[] = [];
  private readonly pendingPauseHandlers = new Set<Promise<void>>();
  private mainFrameId?: string;
  private activeTurn?: CapturedTurnTrace;
  private coverageActive = false;
  private initialized = false;
  private disposed = false;

  private constructor(
    session: CDPSession,
    private readonly artifactUrl: string,
    private readonly artifactText: string,
    config: OnlineActionTraceConfig,
  ) {
    this.config = validateOnlineActionTraceConfig(config);
    this.session = session;
    this.artifactLineStarts = buildLineStarts(artifactText);
    this.artifactSha256 = createHash("sha256").update(artifactText).digest("hex");
  }

  /** Create the collector because Playwright's CDP-session factory is async. */
  static async create(
    page: Page,
    artifactUrl: string,
    artifactText: string,
    config: OnlineActionTraceConfig,
  ): Promise<OnlineActionTraceCollector> {
    return new OnlineActionTraceCollector(
      await page.context().newCDPSession(page),
      artifactUrl,
      artifactText,
      config,
    );
  }

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
    this.session.on("Debugger.scriptParsed", (event: ScriptParsedEvent) => this.recordScript(event));
    this.session.on("Debugger.paused", (event: { callFrames?: PausedCallFrame[] }) => {
      const promise = this.handlePaused(event).catch((error) => {
        const message = `paused-handler: ${toError(error).message}`;
        if (this.activeTurn) {
          this.activeTurn.partial = true;
          this.activeTurn.errors.push(message);
        } else {
          this.setupWarnings.push(message);
        }
      });
      this.pendingPauseHandlers.add(promise);
      void promise.finally(() => this.pendingPauseHandlers.delete(promise));
    });

    await this.session.send("Page.enable");
    const frameTree = await this.session.send("Page.getFrameTree") as {
      frameTree?: { frame?: { id?: string } };
    };
    this.mainFrameId = frameTree.frameTree?.frame?.id;
    await this.session.send("Runtime.enable");
    await this.session.send("Debugger.enable");
    await this.session.send("Profiler.enable");
    await this.session.send("Debugger.setPauseOnExceptions", { state: "none" });
    await this.session.send("Debugger.setSkipAllPauses", { skip: true });
    this.initialized = true;
  }

  /** Discover page-load functions and install exact function-entry breakpoints. */
  async prepare(): Promise<void> {
    if (!this.initialized) throw new Error("online action trace collector was not initialized");
    const response = await this.session.send("Profiler.getBestEffortCoverage") as { result?: RawScriptCoverage[] };
    await this.registerCoverageFunctions(response.result ?? []);
    if (!this.functions.size) {
      this.setupWarnings.push("no traceable inline artifact JavaScript functions were discovered");
    }
  }

  /** Returns false for action types that intentionally do not participate. */
  async startTurn(turnIndex: number, action: SimulationAction): Promise<boolean> {
    if (!isOnlineTraceableAction(action)) return false;
    if (!this.initialized || this.disposed) throw new Error("online action trace collector is not active");
    if (this.activeTurn) throw new Error(`online action trace turn ${this.activeTurn.turnIndex} is still active`);
    const capture: CapturedTurnTrace = {
      turnIndex,
      actionType: action.type,
      pauseEventCount: 0,
      stacks: [],
      instrumentedFunctionKeys: new Set(
        [...this.functions.values()].filter((fn) => fn.breakpointInstalled).map((fn) => fn.key),
      ),
      executedFunctionKeys: new Set(),
      partial: false,
      warnings: [],
      errors: [],
    };
    this.captures.set(turnIndex, capture);
    this.activeTurn = capture;
    try {
      await this.session.send("Profiler.startPreciseCoverage", {
        callCount: true,
        detailed: true,
        allowTriggeredUpdates: false,
      });
      this.coverageActive = true;
    } catch (error) {
      capture.partial = true;
      capture.warnings.push(`precise-coverage-start: ${toError(error).message}`);
    }
    try {
      await this.session.send("Debugger.setSkipAllPauses", { skip: false });
    } catch (error) {
      capture.partial = true;
      capture.errors.push(`debugger-enable-pauses: ${toError(error).message}`);
    }
    return true;
  }

  async finishTurn(): Promise<void> {
    const capture = this.activeTurn;
    if (!capture) return;
    await this.session.send("Debugger.setSkipAllPauses", { skip: true }).catch((error) => {
      capture.partial = true;
      capture.warnings.push(`debugger-disable-pauses: ${toError(error).message}`);
    });
    await this.waitForPauseHandlers();

    let coverage: RawScriptCoverage[] = [];
    if (this.coverageActive) {
      try {
        const response = await this.session.send("Profiler.takePreciseCoverage") as { result?: RawScriptCoverage[] };
        coverage = response.result ?? [];
      } catch (error) {
        capture.partial = true;
        capture.warnings.push(`precise-coverage-take: ${toError(error).message}`);
      } finally {
        this.coverageActive = false;
        await this.session.send("Profiler.stopPreciseCoverage").catch((error) => {
          capture.partial = true;
          capture.warnings.push(`precise-coverage-stop: ${toError(error).message}`);
        });
      }
    }

    this.activeTurn = undefined;

    // A newly created function cannot be breakpointed retroactively, but make
    // it available to subsequent actions in the same browser session.
    if (coverage.length) {
      await this.registerCoverageFunctions(coverage).catch((error) => {
        capture.partial = true;
        capture.warnings.push(`dynamic-function-registration: ${toError(error).message}`);
      });
    }
    for (const scriptCoverage of coverage) {
      for (const rawFunction of scriptCoverage.functions) {
        if (!rawFunction.ranges.some((range) => range.count > 0)) continue;
        const outer = rawFunction.ranges[0];
        if (!outer) continue;
        const key = functionKey(scriptCoverage.scriptId, outer.startOffset, outer.endOffset);
        if (this.functions.has(key)) capture.executedFunctionKeys.add(key);
      }
    }
    const capturedKeys = new Set(capture.stacks.flat());
    const missed = [...capture.executedFunctionKeys].filter((key) =>
      !capture.instrumentedFunctionKeys.has(key) || !capturedKeys.has(key),
    );
    if (missed.length) {
      capture.partial = true;
      capture.warnings.push(`precise coverage found ${missed.length} executed function(s) without a captured call frame`);
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    if (this.activeTurn) await this.finishTurn().catch(() => undefined);
    this.disposed = true;
    await this.waitForPauseHandlers();
    await this.session.send("Debugger.setSkipAllPauses", { skip: true }).catch(() => undefined);
    if (this.coverageActive) {
      this.coverageActive = false;
      await this.session.send("Profiler.stopPreciseCoverage").catch(() => undefined);
    }
    await this.session.send("Profiler.disable").catch(() => undefined);
    await this.session.send("Debugger.disable").catch(() => undefined);
    await this.session.send("Runtime.disable").catch(() => undefined);
    await this.session.send("Page.disable").catch(() => undefined);
    await this.session.detach().catch(() => undefined);
  }

  buildAttribution(turns: readonly UserSimulationTurn[]): OnlineActionTraceAttribution {
    const summaries: ActionTraceTurnSummary[] = [];
    const depthRanges: DepthRange[] = [];
    let selectedActionCount = 0;
    let attributedActionCount = 0;
    let selectedFailureCount = 0;
    let selectedPartialCount = 0;

    for (const turn of turns) {
      const score = finiteNumber(turn.transitionEvaluation?.score);
      const traceable = isOnlineTraceableAction(turn.action);
      const selected = traceable && score !== undefined && score >= this.config.threshold;
      const capture = this.captures.get(turn.turnIndex);
      const base = {
        turnIndex: turn.turnIndex,
        actionType: turn.action.type,
        transitionScore: score,
        selected,
        pauseEventCount: capture?.pauseEventCount ?? 0,
        stackCount: capture?.stacks.length ?? 0,
        functionCount: 0,
        warnings: capture?.warnings.length ? [...capture.warnings] : undefined,
        errors: capture?.errors.length ? [...capture.errors] : undefined,
      };
      if (!traceable) {
        summaries.push({ ...base, status: "not-traceable" });
        continue;
      }
      if (!selected) {
        summaries.push({ ...base, status: score === undefined ? "failed" : "below-threshold" });
        continue;
      }

      selectedActionCount += 1;
      if (!capture) {
        selectedFailureCount += 1;
        summaries.push({
          ...base,
          status: "failed",
          errors: [...(base.errors ?? []), "no online trace capture was available for the selected action"],
        });
        continue;
      }

      const maximalStacks = reduceMaximalCallStacks(capture.stacks);
      const depths = new Map<string, number>();
      for (const stack of maximalStacks) {
        stack.forEach((key, depth) => depths.set(key, Math.min(depths.get(key) ?? Number.POSITIVE_INFINITY, depth)));
      }
      for (const [key, depth] of depths) {
        const fn = this.functions.get(key);
        if (!fn) continue;
        depthRanges.push({ utf16Start: fn.htmlUtf16Start, utf16End: fn.htmlUtf16End, depth });
      }
      const functionCount = depths.size;
      if (functionCount) attributedActionCount += 1;
      let status: ActionTraceTurnSummary["status"];
      if (capture.errors.length && !functionCount) {
        status = "failed";
        selectedFailureCount += 1;
      } else if (capture.partial || capture.errors.length) {
        status = "partial";
        selectedPartialCount += 1;
      } else if (!functionCount) {
        status = "empty";
      } else {
        status = "complete";
      }
      summaries.push({ ...base, status, functionCount });
    }

    const runs = mergeDepthRanges(depthRanges, this.artifactText);
    let status: OnlineActionTraceAttribution["status"];
    if (!selectedActionCount) status = "empty";
    else if (!runs.length && selectedFailureCount === selectedActionCount) status = "failed";
    else if (selectedFailureCount || selectedPartialCount || this.setupErrors.length) status = "partial";
    else if (!runs.length) status = "empty";
    else status = "complete";

    return {
      schemaVersion: TRACE_SCHEMA_VERSION,
      enabled: true,
      status,
      sourceScope: "artifact-inline-main-frame-javascript",
      offsetEncoding: "utf-8-bytes",
      artifactSha256: this.artifactSha256,
      artifactUtf8ByteLength: Buffer.byteLength(this.artifactText, "utf8"),
      threshold: this.config.threshold,
      maxPauseEventsPerAction: this.config.maxPauseEventsPerAction,
      selectedActionCount,
      attributedActionCount,
      runs,
      turns: summaries,
      warnings: this.setupWarnings.length ? [...this.setupWarnings] : undefined,
      errors: this.setupErrors.length ? [...this.setupErrors] : undefined,
    };
  }

  private recordScript(event: ScriptParsedEvent): void {
    const context = event.executionContextAuxData ?? (
      typeof event.executionContextId === "number" ? this.executionContexts.get(event.executionContextId) : undefined
    );
    this.scripts.set(event.scriptId, {
      scriptId: event.scriptId,
      url: event.url,
      startLine: event.startLine,
      startColumn: event.startColumn,
      isMainFrameDefaultWorld: context?.frameId === this.mainFrameId && context?.isDefault === true,
    });
  }

  private async registerCoverageFunctions(coverage: readonly RawScriptCoverage[]): Promise<void> {
    const pending: FunctionRecord[] = [];
    for (const scriptCoverage of coverage) {
      if (!sameResourceUrl(scriptCoverage.url, this.artifactUrl)) continue;
      const script = this.scripts.get(scriptCoverage.scriptId);
      if (!script?.isMainFrameDefaultWorld) continue;
      if (!await this.loadAndMapScript(script)) continue;
      const source = script.source!;
      const scriptStart = script.htmlUtf16Start!;
      for (const rawFunction of scriptCoverage.functions) {
        const outer = rawFunction.ranges[0];
        if (!outer || outer.startOffset < 0 || outer.endOffset <= outer.startOffset || outer.endOffset > source.length) {
          continue;
        }
        // Precise coverage represents the script top level as an anonymous
        // function. It is not a user call-chain node.
        if (!rawFunction.functionName && outer.startOffset === 0 && outer.endOffset === source.length) continue;
        const key = functionKey(script.scriptId, outer.startOffset, outer.endOffset);
        if (this.functions.has(key)) continue;
        const record: FunctionRecord = {
          key,
          scriptId: script.scriptId,
          functionName: rawFunction.functionName || "<anonymous>",
          sourceStart: outer.startOffset,
          sourceEnd: outer.endOffset,
          htmlUtf16Start: scriptStart + outer.startOffset,
          htmlUtf16End: scriptStart + outer.endOffset,
          breakpointInstalled: false,
        };
        this.functions.set(key, record);
        const byScript = this.functionsByScript.get(script.scriptId) ?? [];
        byScript.push(record);
        byScript.sort((left, right) =>
          left.sourceStart - right.sourceStart || left.sourceEnd - right.sourceEnd,
        );
        this.functionsByScript.set(script.scriptId, byScript);
        pending.push(record);
      }
    }

    for (let start = 0; start < pending.length; start += 64) {
      const chunk = pending.slice(start, start + 64);
      const results = await Promise.allSettled(chunk.map((fn) => this.installFunctionBreakpoint(fn)));
      results.forEach((result, index) => {
        if (result.status === "rejected") {
          const fn = chunk[index]!;
          this.setupWarnings.push(
            `breakpoint-install: ${fn.functionName} (${fn.key}): ${toError(result.reason).message}`,
          );
        }
      });
    }
  }

  private async loadAndMapScript(script: ScriptRecord): Promise<boolean> {
    if (script.source !== undefined) return script.htmlUtf16Start !== undefined;
    try {
      const response = await this.session.send("Debugger.getScriptSource", { scriptId: script.scriptId }) as {
        scriptSource?: string;
      };
      if (typeof response.scriptSource !== "string") return false;
      script.source = response.scriptSource;
      script.sourceLineStarts = buildLineStarts(script.source);
      const start = offsetAtPosition(
        this.artifactLineStarts,
        script.startLine,
        script.startColumn,
        this.artifactText.length,
      );
      if (start === undefined || this.artifactText.slice(start, start + script.source.length) !== script.source) {
        this.setupWarnings.push(`unmapped-artifact-script: scriptId=${script.scriptId}`);
        return false;
      }
      script.htmlUtf16Start = start;
      return true;
    } catch (error) {
      this.setupWarnings.push(`script-source: scriptId=${script.scriptId}: ${toError(error).message}`);
      return false;
    }
  }

  private async installFunctionBreakpoint(fn: FunctionRecord): Promise<void> {
    const script = this.scripts.get(fn.scriptId);
    if (!script?.sourceLineStarts) return;
    const start = debuggerLocationAtOffset(script, fn.sourceStart);
    const end = debuggerLocationAtOffset(script, fn.sourceEnd);
    const possible = await this.session.send("Debugger.getPossibleBreakpoints", {
      start,
      end,
      restrictToFunction: true,
    }) as { locations?: Array<{ scriptId: string; lineNumber: number; columnNumber?: number }> };
    const location = possible.locations?.[0];
    if (!location) throw new Error("no possible function-entry breakpoint location");
    await this.session.send("Debugger.setBreakpoint", { location });
    fn.breakpointInstalled = true;
  }

  private async handlePaused(event: { callFrames?: PausedCallFrame[] }): Promise<void> {
    const capture = this.activeTurn;
    try {
      if (capture) {
        capture.pauseEventCount += 1;
        const stack: string[] = [];
        for (const frame of event.callFrames ?? []) {
          const fn = this.functionAtLocation(frame.location);
          if (fn) stack.push(fn.key);
        }
        if (stack.length) capture.stacks.push(stack);
        if (capture.pauseEventCount >= this.config.maxPauseEventsPerAction) {
          capture.partial = true;
          capture.warnings.push(
            `pause event cap ${this.config.maxPauseEventsPerAction} reached; attribution may be partial`,
          );
          await this.session.send("Debugger.setSkipAllPauses", { skip: true }).catch(() => undefined);
        }
      }
    } finally {
      await this.session.send("Debugger.resume").catch(() => undefined);
    }
  }

  private functionAtLocation(
    location: PausedCallFrame["location"],
  ): FunctionRecord | undefined {
    if (!location?.scriptId || !Number.isInteger(location.lineNumber)) return undefined;
    const script = this.scripts.get(location.scriptId);
    if (!script?.sourceLineStarts) return undefined;
    const offset = sourceOffsetAtDebuggerLocation(script, location.lineNumber!, location.columnNumber ?? 0);
    if (offset === undefined) return undefined;
    let best: FunctionRecord | undefined;
    for (const fn of this.functionsByScript.get(location.scriptId) ?? []) {
      if (fn.sourceStart > offset) break;
      if (offset < fn.sourceEnd && (!best || fn.sourceEnd - fn.sourceStart < best.sourceEnd - best.sourceStart)) {
        best = fn;
      }
    }
    return best;
  }

  private async waitForPauseHandlers(): Promise<void> {
    while (this.pendingPauseHandlers.size) {
      await Promise.allSettled([...this.pendingPauseHandlers]);
    }
  }
}

export function failedOnlineActionTraceAttribution(
  config: OnlineActionTraceConfig,
  artifactText: string,
  error: unknown,
): OnlineActionTraceAttribution {
  const resolved = validateOnlineActionTraceConfig(config);
  return {
    schemaVersion: TRACE_SCHEMA_VERSION,
    enabled: true,
    status: "failed",
    sourceScope: "artifact-inline-main-frame-javascript",
    offsetEncoding: "utf-8-bytes",
    artifactSha256: createHash("sha256").update(artifactText).digest("hex"),
    artifactUtf8ByteLength: Buffer.byteLength(artifactText, "utf8"),
    threshold: resolved.threshold,
    maxPauseEventsPerAction: resolved.maxPauseEventsPerAction,
    selectedActionCount: 0,
    attributedActionCount: 0,
    runs: [],
    turns: [],
    errors: [toError(error).message],
  };
}

function functionKey(scriptId: string, start: number, end: number): string {
  return `${scriptId}:${start}:${end}`;
}

function isSuffix(candidate: readonly string[], longer: readonly string[]): boolean {
  const offset = longer.length - candidate.length;
  return candidate.every((value, index) => longer[offset + index] === value);
}

function buildLineStarts(text: string): number[] {
  const starts = [0];
  for (let index = 0; index < text.length; index++) {
    if (text.charCodeAt(index) === 10) starts.push(index + 1);
  }
  return starts;
}

function offsetAtPosition(
  lineStarts: readonly number[],
  line: number,
  column: number,
  textLength: number,
): number | undefined {
  if (!Number.isInteger(line) || !Number.isInteger(column) || line < 0 || column < 0 || line >= lineStarts.length) {
    return undefined;
  }
  const offset = lineStarts[line]! + column;
  return offset <= textLength ? offset : undefined;
}

function debuggerLocationAtOffset(
  script: ScriptRecord,
  offset: number,
): { scriptId: string; lineNumber: number; columnNumber: number } {
  const starts = script.sourceLineStarts!;
  let low = 0;
  let high = starts.length;
  while (low + 1 < high) {
    const middle = Math.floor((low + high) / 2);
    if (starts[middle]! <= offset) low = middle;
    else high = middle;
  }
  const localColumn = offset - starts[low]!;
  return {
    scriptId: script.scriptId,
    lineNumber: script.startLine + low,
    columnNumber: localColumn + (low === 0 ? script.startColumn : 0),
  };
}

function sourceOffsetAtDebuggerLocation(
  script: ScriptRecord,
  lineNumber: number,
  columnNumber: number,
): number | undefined {
  const localLine = lineNumber - script.startLine;
  if (localLine < 0 || localLine >= script.sourceLineStarts!.length) return undefined;
  const localColumn = columnNumber - (localLine === 0 ? script.startColumn : 0);
  if (localColumn < 0) return undefined;
  const offset = script.sourceLineStarts![localLine]! + localColumn;
  return offset <= script.source!.length ? offset : undefined;
}

function utf8OffsetsAt(text: string, positions: number[]): Map<number, number> {
  const sorted = [...positions].sort((left, right) => left - right);
  const offsets = new Map<number, number>();
  let priorUtf16 = 0;
  let priorUtf8 = 0;
  for (const position of sorted) {
    priorUtf8 += Buffer.byteLength(text.slice(priorUtf16, position), "utf8");
    offsets.set(position, priorUtf8);
    priorUtf16 = position;
  }
  return offsets;
}

function sameResourceUrl(left: string, right: string): boolean {
  try {
    const first = new URL(left);
    const second = new URL(right);
    first.hash = "";
    second.hash = "";
    return first.href === second.href;
  } catch {
    return left === right;
  }
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}
