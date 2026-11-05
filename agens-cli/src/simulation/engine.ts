import { access, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Browser, BrowserContext, Page } from "playwright";
import type { ResolvedUserSimulationConfig } from "../userconfig.js";
import { createActionOverlay } from "./actionOverlay.js";
import { decideAction } from "./judge.js";
import { observeAfterAction } from "./observer.js";
import { advanceClock, launchBrowser, moveMouseToViewportCenter, pauseClockNow, performAction, readActionHints } from "./playwright.js";
import type {
  ProgressObservation,
  RunUserSimulationInput,
  SimulationAction,
  TransitionFailureMode,
  TurnVisualEvaluation,
  UserSimulationResult,
  UserSimulationRuntime,
  UserSimulationTurn,
  VideoSegment,
} from "./types.js";
import {
  clampCount,
  clampMs,
  createRunDir,
  loadOptionalText,
  normalizeUrl,
  summarizeTurns,
  turnLabel,
  writeJson,
} from "./utils.js";
import { processSimulationVideos } from "./video.js";
import { evaluateGlobalSimulation } from "./globalJudge.js";
import { JudgeTaskQueue } from "./judgeTaskQueue.js";
import { evaluateTurnJudges } from "./visualJudges.js";
import {
  failedOnlineActionTraceAttribution,
  OnlineActionTraceCollector,
  validateOnlineActionTraceConfig,
} from "./actionTrace.js";

export type {
  ActionHint,
  JudgeProvenance,
  LoadedText,
  ProgressObservation,
  PsychometricObservation,
  RunUserSimulationInput,
  SimulationAction,
  SimulationDecision,
  UserSimulationResult,
  UserSimulationRuntime,
  UserSimulationTurn,
} from "./types.js";

export class UserSimulationRunError extends Error {
  constructor(message: string, readonly result: UserSimulationResult) {
    super(message);
    this.name = "UserSimulationRunError";
  }
}

export async function runUserSimulation(
  input: RunUserSimulationInput,
  runtime: UserSimulationRuntime,
): Promise<UserSimulationResult> {
  const stepMs = clampMs(input.stepMs ?? runtime.stepMs ?? 300, 0, 60_000, "stepMs");
  const frameMs = clampMs(input.frameMs ?? runtime.frameMs ?? 50, 1, 1_000, "frameMs");
  const maxAdvanceMs = clampMs(input.maxAdvanceMs ?? Math.max(5_000, stepMs * 10), 0, 120_000, "maxAdvanceMs");
  const turnLimit = clampCount(input.turnLimit ?? input.steps ?? runtime.turnLimit ?? runtime.defaultSteps ?? 20, 1, 1_000, "turnLimit");
  const saveFullRecord = input.saveFullRecord ?? runtime.saveFullRecord ?? false;
  const recordVideo = saveFullRecord && (input.recordVideo ?? runtime.recordVideo ?? true);
  const runDir = await createRunDir(runtime.cwd, input.outputDir ?? runtime.outputDir ?? ".agens-sim");
  const url = normalizeUrl(input.url, runtime.cwd);

  const resultPath = path.join(runDir, "result.json");
  const stateTracePath = path.join(runDir, "state_trace.jsonl");
  const actionTraceAttributionPath = path.join(runDir, "action-trace-attribution.json");

  const persona = await loadOptionalText("persona", input.personaPath ?? runtime.personaPath, runtime.cwd);
  const metrics = await loadOptionalText(
    "psychometric metrics",
    input.psychometricPath ?? runtime.psychometricPath,
    runtime.cwd,
  );
  const artifactPath = input.artifactPath ?? (input.includeCode ? sourcePathFromLocalUrl(url, runtime.cwd) : undefined);
  const artifact = await loadOptionalText("artifact source", artifactPath, runtime.cwd, 12_000, false);
  const artifactPrompt = await loadOptionalText(
    "artifact prompt",
    await promptPathFromLocalArtifact(
      artifact?.path ?? (artifactPath ? sourcePathFromLocalUrl(artifactPath, runtime.cwd) : undefined) ?? sourcePathFromLocalUrl(url, runtime.cwd),
    ),
    runtime.cwd,
    12_000,
    false,
  );

  let browser: Browser | undefined;
  let context: BrowserContext | undefined;
  let page: Page | undefined;
  let fullRecordPath: string | undefined;
  let fullRecordError: string | undefined;
  const videoProcessingErrors: string[] = [];
  const activeVideoSegments: VideoSegment[] = [];
  const turns: UserSimulationTurn[] = [];
  const traceEvents: unknown[] = [];
  let globalEvaluation: UserSimulationResult["globalEvaluation"];
  let videoStartedAtMs = Date.now();
  const judgeQueue = new JudgeTaskQueue();
  let judgeQueueDrained = false;
  const actionTraceConfig = input.actionTrace?.enabled === false ? undefined : input.actionTrace;
  let actionTraceCollector: OnlineActionTraceCollector | undefined;
  let actionTraceCollectorDisposed = false;
  let actionTraceArtifactText = "";
  let actionTraceSetupError: Error | undefined;
  let actionTraceAttribution: UserSimulationResult["actionTraceAttribution"];
  let persistedActionTracePath: string | undefined;

  if (actionTraceConfig) validateOnlineActionTraceConfig(actionTraceConfig);

  const disposeActionTrace = async (): Promise<void> => {
    if (!actionTraceCollector || actionTraceCollectorDisposed) return;
    actionTraceCollectorDisposed = true;
    await actionTraceCollector.dispose().catch((error) => {
      traceEvents.push({ phase: "action-trace-dispose-error", error: toError(error).message });
    });
  };
  const finalizeActionTrace = async (): Promise<void> => {
    if (!actionTraceConfig || actionTraceAttribution) return;
    try {
      actionTraceAttribution = actionTraceCollector
        ? actionTraceCollector.buildAttribution(turns)
        : failedOnlineActionTraceAttribution(
          actionTraceConfig,
          actionTraceArtifactText,
          actionTraceSetupError ?? new Error("online action trace collector was unavailable"),
        );
    } catch (error) {
      actionTraceSetupError = toError(error);
      traceEvents.push({ phase: "action-trace-finalize-error", error: actionTraceSetupError.message });
      actionTraceAttribution = failedOnlineActionTraceAttribution(
        actionTraceConfig,
        actionTraceArtifactText,
        actionTraceSetupError,
      );
    }
    try {
      await writeJson(actionTraceAttributionPath, actionTraceAttribution);
      persistedActionTracePath = actionTraceAttributionPath;
    } catch (error) {
      actionTraceAttribution.warnings = [
        ...(actionTraceAttribution.warnings ?? []),
        `attribution-write: ${toError(error).message}`,
      ];
      traceEvents.push({ phase: "action-trace-write-error", error: toError(error).message });
    }
  };

  try {
    browser = await launchBrowser(runtime.browser, runtime.headed);
    context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      ...(recordVideo
        ? { recordVideo: { dir: runDir, size: { width: 1440, height: 900 } } }
        : {}),
    });
    await context.clock.install({ time: Date.now() });
    videoStartedAtMs = Date.now();
    page = await context.newPage();
    if (actionTraceConfig) {
      try {
        if (!artifact?.path) throw new Error("online action trace requires an artifactPath");
        actionTraceArtifactText = await readFile(artifact.path, "utf8");
        actionTraceCollector = await OnlineActionTraceCollector.create(
          page,
          url,
          actionTraceArtifactText,
          actionTraceConfig,
        );
        await actionTraceCollector.initialize();
      } catch (error) {
        actionTraceSetupError = toError(error);
        traceEvents.push({ phase: "action-trace-setup-error", error: actionTraceSetupError.message });
        await disposeActionTrace();
        actionTraceCollector = undefined;
      }
    }
    const loadStartMs = Date.now();
    await page.goto(url, { waitUntil: "networkidle", timeout: 30_000 });
    if (actionTraceCollector) {
      try {
        await actionTraceCollector.prepare();
      } catch (error) {
        actionTraceSetupError = toError(error);
        traceEvents.push({ phase: "action-trace-prepare-error", error: actionTraceSetupError.message });
        await disposeActionTrace();
        actionTraceCollector = undefined;
      }
    }
    activeVideoSegments.push({
      label: "page-load",
      startMs: loadStartMs,
      endMs: Date.now(),
    });
    const initialMouseStartMs = Date.now();
    const initialMousePoint = await moveMouseToViewportCenter(page);
    activeVideoSegments.push({
      label: "initial-mouse-center",
      startMs: initialMouseStartMs,
      endMs: Date.now(),
      preBufferMs: 300,
      postBufferMs: 300,
      minDurationMs: 250,
    });
    traceEvents.push({
      phase: "initial-mouse-center",
      point: initialMousePoint,
    });

    traceEvents.push({
      phase: "judge-pipeline",
      mode: "queued",
      turnJudgeModel: runtime.turnJudge.model,
      decisionJudgeModel: runtime.decisionJudge.model,
    });

    for (let turnIndex = 0; turnIndex < turnLimit; turnIndex++) {
      judgeQueue.assertHealthy();
      await pauseClockNow(context, page);
      const turn = await runSimulationTurn({
        turnIndex,
        turnLimit,
        page,
        context,
        runtime,
        input,
        runDir,
        url,
        persona,
        metrics,
        artifact,
        frameMs,
        maxAdvanceMs,
        priorTurns: turns,
        activeVideoSegments,
        traceEvents,
        actionTraceCollector,
        assertCanContinue: () => judgeQueue.assertHealthy(),
      });
      turns.push(turn);
      judgeQueue.assertHealthy();
      traceEvents.push({ phase: "visual-evaluation-queued", judgeType: "combined", turnIndex });
      judgeQueue.enqueue(
        `${turnLabel(turnIndex)} combined transition/screenshot`,
        () => evaluateTurnJudges({
          turn,
          judge: runtime.turnJudge,
          taskPrompt: input.taskPrompt,
          runDir,
          traceEvents,
        }),
      );
      if (turn.terminated) break;
    }

    const video = page.video();
    await disposeActionTrace();
    await context.close();
    context = undefined;
    if (video) {
      const sourceVideoPath = await video.path().catch(() => undefined);
      if (sourceVideoPath) {
        const processing = await processSimulationVideos({
          sourceVideoPath,
          saveFullRecord,
          runDir,
          videoStartedAtMs,
          activeSegments: activeVideoSegments,
        });
        fullRecordPath = processing.fullRecordPath;
        fullRecordError = processing.fullRecordError;
        videoProcessingErrors.push(...processing.errors);
      }
    } else if (saveFullRecord) {
      fullRecordError = "save_full_record requires Playwright video recording; no source recording was produced.";
    }

    try {
      await judgeQueue.drain();
    } finally {
      judgeQueueDrained = true;
    }
    await finalizeActionTrace();

    if (runtime.extraGlobalJudger?.enabled) {
      globalEvaluation = await evaluateGlobalSimulation({
        judger: runtime.extraGlobalJudger,
        url,
        runDir,
        turns,
        turnLimit,
        taskPrompt: input.taskPrompt,
        artifactPrompt,
        visualEvaluationSummary: buildVisualEvaluationSummary(turns),
        terminated: turns.at(-1)?.terminated === true,
        terminationReason: turns.at(-1)?.terminationReason,
        totalSimulatedAdvanceMs: turns.reduce((sum, turn) => sum + turn.simulatedAdvanceMs, 0),
        preRolloutReference: input.preRolloutReference,
        skipPreRolloutVisualComparison: input.skipPreRolloutVisualComparison,
      });
      traceEvents.push({
        phase: "global-evaluation",
        evaluationPath: globalEvaluation.evaluationPath,
        score: globalEvaluation.score,
        unpenalizedScore: globalEvaluation.unpenalizedScore,
        preRolloutVisualPenalty: globalEvaluation.preRolloutVisualPenalty,
        preRolloutVisualComparisonSkipped: globalEvaluation.preRolloutVisualComparisonSkipped,
        preRolloutComparisonPath: globalEvaluation.preRolloutComparison?.evaluationPath,
        weightedScore: globalEvaluation.weightedScore,
        usabilityCap: globalEvaluation.usabilityCap,
        criticalBlocker: globalEvaluation.criticalBlocker,
      });
    }

    await writeFile(
      stateTracePath,
      traceEvents.map((entry) => JSON.stringify(entry)).join("\n") + (traceEvents.length ? "\n" : ""),
      "utf8",
    );

    const result = buildResult({
      url,
      runDir,
      turnLimit,
      turns,
      fullRecordPath,
      fullRecordError,
      videoProcessingErrors,
      stateTracePath,
      resultPath,
      globalEvaluation,
      actionTraceAttribution,
      actionTraceAttributionPath: persistedActionTracePath,
    });
    await writeJson(resultPath, result);
    return result;
  } catch (error) {
    let drainError: unknown;
    if (!judgeQueueDrained) {
      try {
        await judgeQueue.drain();
      } catch (queueError) {
        drainError = queueError;
      } finally {
        judgeQueueDrained = true;
      }
    }
    await disposeActionTrace();
    await finalizeActionTrace();
    let failure = toError(error);
    if (drainError) {
      failure = new AggregateError(
        [toError(error), toError(drainError)],
        `Simulation failed and the turn judge queue did not complete cleanly: ${toError(error).message}`,
      );
    }
    traceEvents.push({
      phase: "simulation-error",
      error: failure.message,
      errors: failure instanceof AggregateError
        ? failure.errors.map((item) => toError(item).message)
        : [failure.message],
    });
    const partialResult = buildResult({
      url,
      runDir,
      simulationError: failure.message,
      turnLimit,
      turns,
      fullRecordPath,
      fullRecordError,
      videoProcessingErrors,
      stateTracePath,
      resultPath,
      globalEvaluation,
      actionTraceAttribution,
      actionTraceAttributionPath: persistedActionTracePath,
    });
    const persistenceErrors: Error[] = [];
    try {
      await writeFile(
        stateTracePath,
        traceEvents.map((entry) => JSON.stringify(entry)).join("\n") + (traceEvents.length ? "\n" : ""),
        "utf8",
      );
    } catch (persistenceError) {
      persistenceErrors.push(toError(persistenceError));
    }
    try {
      await writeJson(resultPath, partialResult);
    } catch (persistenceError) {
      persistenceErrors.push(toError(persistenceError));
    }
    if (persistenceErrors.length) {
      failure = new AggregateError(
        [failure, ...persistenceErrors],
        `Simulation failed and partial artifacts could not be persisted completely: ${failure.message}`,
      );
      partialResult.simulationError = failure.message;
    }
    throw new UserSimulationRunError(failure.message, partialResult);
  } finally {
    await disposeActionTrace();
    if (context) {
      try { await context.close(); } catch { /* ignore */ }
    }
    if (browser) {
      try { await browser.close(); } catch { /* ignore */ }
    }
  }
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function sourcePathFromLocalUrl(url: string, cwd: string): string | undefined {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "file:") return undefined;
    return fileURLToPath(parsed);
  } catch {
    if (/^[a-zA-Z][a-zA-Z\d+.-]*:/.test(url)) return undefined;
    return path.resolve(cwd, url);
  }
}

async function promptPathFromLocalArtifact(sourcePath: string | undefined): Promise<string | undefined> {
  if (!sourcePath) return undefined;
  const promptPath = path.join(path.dirname(sourcePath), "prompt.txt");
  try {
    await access(promptPath);
    return promptPath;
  } catch {
    return undefined;
  }
}

export function runtimeFromResolvedConfig(
  cwd: string,
  sim: ResolvedUserSimulationConfig,
  browser?: string,
  headed?: boolean,
): UserSimulationRuntime {
  return {
    cwd,
    browser,
    headed,
    turnJudge: sim.turnJudge,
    decisionJudge: sim.decisionJudge,
    personaPath: sim.personaPath,
    psychometricPath: sim.psychometricPath,
    outputDir: sim.outputDir,
    stepMs: sim.stepMs,
    frameMs: sim.frameMs,
    recordVideo: sim.recordVideo,
    saveFullRecord: sim.saveFullRecord,
    turnLimit: sim.turnLimit,
    defaultSteps: sim.defaultSteps,
    extraGlobalJudger: sim.extraGlobalJudger,
  };
}

async function runSimulationTurn(args: {
  turnIndex: number;
  turnLimit: number;
  page: Page;
  context: BrowserContext;
  runtime: UserSimulationRuntime;
  input: RunUserSimulationInput;
  runDir: string;
  url: string;
  persona?: Awaited<ReturnType<typeof loadOptionalText>>;
  metrics?: Awaited<ReturnType<typeof loadOptionalText>>;
  artifact?: Awaited<ReturnType<typeof loadOptionalText>>;
  frameMs: number;
  maxAdvanceMs: number;
  priorTurns: UserSimulationTurn[];
  activeVideoSegments: VideoSegment[];
  traceEvents: unknown[];
  actionTraceCollector?: OnlineActionTraceCollector;
  assertCanContinue?: () => void;
}): Promise<UserSimulationTurn> {
  const label = turnLabel(args.turnIndex);
  const beforeScreenshotPath = path.join(args.runDir, `${label}-before.png`);
  const actionBaseScreenshotPath = path.join(args.runDir, `${label}-action-before.png`);
  const afterScreenshotPath = path.join(args.runDir, `${label}-after.png`);
  const decisionPath = path.join(args.runDir, `${label}-decision.json`);
  const actionOverlayPath = path.join(args.runDir, `${label}-action-overlay.png`);

  const beforeScreenshotStartMs = Date.now();
  await args.page.screenshot({ path: beforeScreenshotPath, fullPage: true });
  pushStillFrameSegment(args.activeVideoSegments, `${label}-before`, beforeScreenshotStartMs, Date.now());
  const actionHints = await readActionHints(args.page);
  const decision = await decideAction({
    provider: args.runtime.decisionJudge.provider,
    routing: args.runtime.decisionJudge.routing,
    model: args.runtime.decisionJudge.model,
    taskPrompt: args.input.taskPrompt,
    screenshotPath: beforeScreenshotPath,
    persona: args.persona,
    metrics: args.metrics,
    artifact: args.artifact,
    url: args.url,
    actionHints,
    episodeHistory: summarizeTurns(args.priorTurns),
    turnIndex: args.turnIndex,
    turnLimit: args.turnLimit,
    frameMs: args.frameMs,
    maxAdvanceMs: args.maxAdvanceMs,
  });
  args.assertCanContinue?.();
  await writeJson(decisionPath, decision);
  args.traceEvents.push({
    phase: "turn-decision",
    turnIndex: args.turnIndex,
    screenshotPath: beforeScreenshotPath,
    decisionPath,
    decision,
  });

  if (decision.action.type === "terminate") {
    const terminateScreenshotStartMs = Date.now();
    await args.page.screenshot({ path: afterScreenshotPath, fullPage: true });
    pushStillFrameSegment(args.activeVideoSegments, `${label}-terminate-after`, terminateScreenshotStartMs, Date.now());
    const turn: UserSimulationTurn = {
      turnIndex: args.turnIndex,
      beforeScreenshotPath,
      decisionPath,
      decision,
      action: decision.action,
      afterScreenshotPath,
      observationScreenshotPaths: [],
      observations: [],
      reachedGoal: true,
      totalWaitMs: 0,
      simulatedAdvanceMs: 0,
      terminated: true,
      terminationReason: decision.action.reason ?? decision.summary,
    };
    args.traceEvents.push({
      phase: "terminate",
      turnIndex: args.turnIndex,
      screenshotPath: afterScreenshotPath,
      reason: turn.terminationReason,
      decision,
    });
    return turn;
  }

  const preActionWaitMs = getPreActionWaitMs(decision.action);
  let actionBaseForOverlayPath = beforeScreenshotPath;
  if (preActionWaitMs > 0) {
    const preWaitStartMs = Date.now();
    await advanceClock(args.context, args.page, preActionWaitMs, args.frameMs);
    args.activeVideoSegments.push({
      label: `${label}-pre-action-wait`,
      startMs: preWaitStartMs,
      endMs: Date.now(),
    });
    const actionBaseScreenshotStartMs = Date.now();
    await args.page.screenshot({ path: actionBaseScreenshotPath, fullPage: true });
    pushStillFrameSegment(args.activeVideoSegments, `${label}-action-before`, actionBaseScreenshotStartMs, Date.now());
    actionBaseForOverlayPath = actionBaseScreenshotPath;
  }

  const executableAction = stripPreActionWait(decision.action);
  const turnObservations: ProgressObservation[] = [];
  let traceStarted = false;
  if (args.actionTraceCollector) {
    try {
      traceStarted = await args.actionTraceCollector.startTurn(args.turnIndex, executableAction);
    } catch (error) {
      args.traceEvents.push({
        phase: "action-trace-start-error",
        turnIndex: args.turnIndex,
        error: toError(error).message,
      });
    }
  }
  let actionExecution: Awaited<ReturnType<typeof performAction>>;
  let execution: Awaited<ReturnType<typeof observeAfterAction>>;
  let overlayPath: string | undefined;
  try {
    const actionSegmentStartMs = Date.now();
    actionExecution = await performAction(args.page, args.context, executableAction, args.frameMs, {
      onHeldActionCheckpoint: (checkpoint) => recordHeldActionObservation({
        page: args.page,
        runDir: args.runDir,
        turnIndex: args.turnIndex,
        preActionWaitMs,
        checkpoint,
        observations: turnObservations,
        videoSegments: args.activeVideoSegments,
      }),
    });
    actionExecution.metadata.preWaitMs = preActionWaitMs || undefined;
    args.activeVideoSegments.push({
      label: `${label}-action`,
      startMs: actionSegmentStartMs,
      endMs: Date.now(),
    });
    overlayPath = await createActionOverlay({
      beforeScreenshotPath: actionBaseForOverlayPath,
      outputPath: actionOverlayPath,
      action: executableAction,
      metadata: actionExecution.metadata,
    }).catch((err) => {
      args.traceEvents.push({
        phase: "action-overlay-error",
        turnIndex: args.turnIndex,
        error: (err as Error).message,
      });
      return undefined;
    });
    args.traceEvents.push({
      phase: "action-execution",
      turnIndex: args.turnIndex,
      action: decision.action,
      actionExecution: actionExecution.metadata,
      actionOverlayPath: overlayPath,
    });
    if (actionExecution.metadata.executionError) {
      args.traceEvents.push({
        phase: "action-execution-error",
        turnIndex: args.turnIndex,
        action: decision.action,
        actionExecution: actionExecution.metadata,
      });
    }
    execution = await observeAfterAction({
      page: args.page,
      context: args.context,
      action: executableAction,
      frameMs: args.frameMs,
      maxAdvanceMs: args.maxAdvanceMs,
      turnIndex: args.turnIndex,
      videoSegments: args.activeVideoSegments,
    });
  } finally {
    if (traceStarted) {
      await args.actionTraceCollector!.finishTurn().catch((error) => {
        args.traceEvents.push({
          phase: "action-trace-finish-error",
          turnIndex: args.turnIndex,
          error: toError(error).message,
        });
      });
    }
  }

  const afterScreenshotStartMs = Date.now();
  await args.page.screenshot({ path: afterScreenshotPath, fullPage: true });
  pushStillFrameSegment(args.activeVideoSegments, `${label}-after`, afterScreenshotStartMs, Date.now());
  const turn: UserSimulationTurn = {
    turnIndex: args.turnIndex,
    beforeScreenshotPath,
    decisionPath,
    decision,
    action: decision.action,
    actionExecution: actionExecution.metadata,
    actionOverlayPath: overlayPath,
    afterScreenshotPath,
    observationScreenshotPaths: turnObservations.map((o) => o.screenshotPath),
    observations: turnObservations,
    finalObservation: turnObservations.at(-1),
    reachedGoal: execution.reachedGoal,
    totalWaitMs: execution.totalWaitMs,
    simulatedAdvanceMs: preActionWaitMs + actionExecution.advanceMs + execution.totalWaitMs,
    terminated: false,
  };
  args.traceEvents.push(...turnObservations);
  args.traceEvents.push({
    phase: "turn-after",
    turnIndex: args.turnIndex,
    screenshotPath: afterScreenshotPath,
    reachedGoal: turn.reachedGoal,
    totalWaitMs: turn.totalWaitMs,
    simulatedAdvanceMs: turn.simulatedAdvanceMs,
  });
  return turn;
}

async function recordHeldActionObservation(args: {
  page: Page;
  runDir: string;
  turnIndex: number;
  preActionWaitMs: number;
  checkpoint: { checkpointMs: number; deltaMs: number; index: number };
  observations: ProgressObservation[];
  videoSegments: VideoSegment[];
}): Promise<void> {
  const screenshotPath = path.join(
    args.runDir,
    `${turnLabel(args.turnIndex)}-observe-${String(args.checkpoint.index).padStart(3, "0")}.png`,
  );
  const screenshotStart = Date.now();
  await args.page.screenshot({ path: screenshotPath, fullPage: true });
  const screenshotEnd = Date.now();
  args.videoSegments.push({
    label: `turn-${args.turnIndex}-held-observe-${args.checkpoint.index}`,
    startMs: screenshotStart,
    endMs: screenshotEnd,
    preBufferMs: 800,
    postBufferMs: 800,
    minDurationMs: 600,
  });
  args.observations.push({
    phase: "held-action-observe",
    turnIndex: args.turnIndex,
    screenshotPath,
    simulatedElapsedMs: args.preActionWaitMs + args.checkpoint.checkpointMs,
    reachedGoal: false,
    waitMs: args.checkpoint.deltaMs,
    observation: `Automatic observation at ${args.checkpoint.checkpointMs}ms during the held action.`,
    confidence: 0,
  });
}

function getPreActionWaitMs(action: SimulationAction): number {
  return action.type !== "terminate" && Number.isFinite(action.waitMs) && action.waitMs! > 0 ? action.waitMs! : 0;
}

function stripPreActionWait(action: SimulationAction): SimulationAction {
  if (action.type === "terminate" || action.waitMs === undefined) return action;
  const { waitMs: _waitMs, ...rest } = action;
  return rest as SimulationAction;
}

function pushStillFrameSegment(segments: VideoSegment[], label: string, startMs: number, endMs: number): void {
  segments.push({
    label,
    startMs,
    endMs,
    preBufferMs: 900,
    postBufferMs: 900,
    minDurationMs: 700,
  });
}

function buildResult(args: {
  url: string;
  runDir: string;
  simulationError?: string;
  turnLimit: number;
  turns: UserSimulationTurn[];
  fullRecordPath?: string;
  fullRecordError?: string;
  videoProcessingErrors: string[];
  stateTracePath: string;
  resultPath: string;
  globalEvaluation?: UserSimulationResult["globalEvaluation"];
  actionTraceAttribution?: UserSimulationResult["actionTraceAttribution"];
  actionTraceAttributionPath?: string;
}): UserSimulationResult {
  const observations = args.turns.flatMap((turn) => turn.observations);
  const finalTurn = args.turns.at(-1);
  const finalObservation = observations.at(-1);
  const fallbackAction: SimulationAction = { type: "terminate", reason: "No simulation turns were executed." };
  const totalSimulatedAdvanceMs = args.turns.reduce((sum, turn) => sum + turn.simulatedAdvanceMs, 0);
  const totalWaitMs = args.turns.reduce((sum, turn) => sum + turn.totalWaitMs, 0);

  return {
    url: args.url,
    runDir: args.runDir,
    simulationError: args.simulationError,
    turnLimit: args.turnLimit,
    turns: args.turns,
    terminated: finalTurn?.terminated === true,
    terminationReason: finalTurn?.terminationReason,
    turnCount: args.turns.length,
    finalScreenshotPath: finalTurn?.afterScreenshotPath ?? "",
    totalSimulatedAdvanceMs,
    fullRecordPath: args.fullRecordPath,
    fullRecordError: args.fullRecordError,
    videoProcessingErrors: args.videoProcessingErrors.length ? args.videoProcessingErrors : undefined,
    visualEvaluationSummary: buildVisualEvaluationSummary(args.turns),
    visualEvaluationErrors: args.turns.flatMap((turn) => turn.visualEvaluationErrors ?? []).length
      ? args.turns.flatMap((turn) => turn.visualEvaluationErrors ?? [])
      : undefined,
    globalEvaluation: args.globalEvaluation,
    globalEvaluationPath: args.globalEvaluation?.evaluationPath,
    actionTraceAttribution: args.actionTraceAttribution,
    actionTraceAttributionPath: args.actionTraceAttributionPath,
    stateTracePath: args.stateTracePath,
    resultPath: args.resultPath,
    action: finalTurn?.action ?? fallbackAction,
    decision: finalTurn?.decision ?? {
      summary: "No simulation turns were executed.",
      action: fallbackAction,
      targetState: "No page state was observed.",
    },
    beforeScreenshotPath: args.turns[0]?.beforeScreenshotPath ?? "",
    afterScreenshotPath: finalTurn?.afterScreenshotPath ?? "",
    observationScreenshotPaths: observations.map((o) => o.screenshotPath),
    observations,
    finalObservation,
    reachedGoal: finalTurn?.reachedGoal ?? false,
    totalWaitMs,
    simulatedAdvanceMs: totalSimulatedAdvanceMs,
    decisionPath: args.turns[0]?.decisionPath ?? "",
  };
}

function buildVisualEvaluationSummary(turns: UserSimulationTurn[]): UserSimulationResult["visualEvaluationSummary"] {
  const screenshot = turns.flatMap((turn) => turn.screenshotEvaluation ? [turn.screenshotEvaluation] : []);
  const transition = turns.flatMap((turn) => turn.transitionEvaluation ? [turn.transitionEvaluation] : []);
  if (!screenshot.length && !transition.length) return undefined;
  return {
    ...(screenshot.length ? { screenshot: summarizeEvaluations(screenshot) } : {}),
    ...(transition.length ? {
      transition: {
        ...summarizeEvaluations(transition),
        failureModes: summarizeFailureModes(transition),
      },
    } : {}),
  };
}

function summarizeEvaluations(evaluations: TurnVisualEvaluation[]): { count: number; averageScore: number } {
  const total = evaluations.reduce((sum, evaluation) => sum + evaluation.score, 0);
  const base = {
    count: evaluations.length,
    averageScore: Math.round((total / evaluations.length) * 100) / 100,
  };
  return {
    ...base,
    ...averageRewardField(evaluations, "overallReward", "averageOverallReward"),
    ...averageRewardField(evaluations, "visualStabilityReward", "averageVisualStabilityReward"),
    ...averageRewardField(evaluations, "executionReward", "averageExecutionReward"),
    ...averageRewardField(evaluations, "appFeedbackReward", "averageAppFeedbackReward"),
    ...averageRewardField(evaluations, "taskProgressReward", "averageTaskProgressReward"),
    ...averageRewardField(evaluations, "readabilityReward", "averageReadabilityReward"),
    ...averageRewardField(evaluations, "stateVisibilityReward", "averageStateVisibilityReward"),
    ...averageRewardField(evaluations, "screenshotAestheticReward", "averageScreenshotAestheticReward"),
  };
}

function averageRewardField(
  evaluations: TurnVisualEvaluation[],
  field: keyof Pick<
    TurnVisualEvaluation,
    | "overallReward"
    | "visualStabilityReward"
    | "executionReward"
    | "appFeedbackReward"
    | "taskProgressReward"
    | "readabilityReward"
    | "stateVisibilityReward"
    | "screenshotAestheticReward"
  >,
  outputField: string,
): Record<string, number> {
  const values = evaluations
    .map((evaluation) => evaluation[field])
    .filter((value): value is number => Number.isFinite(value));
  if (!values.length) return {};
  const total = values.reduce((sum, value) => sum + value, 0);
  return { [outputField]: Math.round((total / values.length) * 100) / 100 };
}

function summarizeFailureModes(
  evaluations: TurnVisualEvaluation[],
): Array<{ failureMode: TransitionFailureMode; count: number; recentExamples: Array<{ turnIndex: number; summary: string }> }> {
  const modes = evaluations
    .map((evaluation) => evaluation.failureMode)
    .filter((mode): mode is TransitionFailureMode => Boolean(mode) && mode !== "none");
  if (!modes.length) return [];

  const orderedModes: TransitionFailureMode[] = [
    "coordinate_miss",
    "action_ineffective",
    "illegal_or_rejected",
    "timing_issue",
    "uncertain",
  ];
  return orderedModes.flatMap((failureMode) => {
    const matching = evaluations.filter((evaluation) => evaluation.failureMode === failureMode);
    if (!matching.length) return [];
    return [{
      failureMode,
      count: matching.length,
      recentExamples: matching.slice(-3).map((evaluation) => ({
        turnIndex: evaluation.turnIndex,
        summary: evaluation.summary,
      })),
    }];
  });
}
