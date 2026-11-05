import type { ProviderFile, ResolvedSimulationJudge } from "../userconfig.js";

export type SimulationAction =
  | { type: "click"; selector?: string; x?: number; y?: number; text?: string; durationMs?: number; waitMs?: number }
  | { type: "hover"; selector?: string; x?: number; y?: number; text?: string; waitMs?: number }
  | { type: "mouseMove"; dx: number; dy: number; durationMs?: number; steps?: number; waitMs?: number }
  | { type: "press"; key: string; durationMs?: number; waitMs?: number }
  | { type: "combo"; keys: string[]; durationMs?: number; waitMs?: number }
  | { type: "type"; selector?: string; text: string; waitMs?: number }
  | { type: "wait"; ms?: number; waitMs?: number }
  | { type: "terminate"; reason?: string };

export type TransitionFailureMode =
  | "none"
  | "coordinate_miss"
  | "action_ineffective"
  | "illegal_or_rejected"
  | "timing_issue"
  | "uncertain";

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ActionHitTarget {
  pagePoint?: Point;
  hitElement?: {
    tag: string;
    id?: string;
    className?: string;
    text?: string;
  };
  elementRect?: Rect;
  localPoint?: Point;
  canvasSize?: { width: number; height: number };
}

export interface ActionExecutionMetadata {
  actionType: SimulationAction["type"];
  success?: boolean;
  warnings?: string[];
  preWaitMs?: number;
  requestedDurationMs?: number;
  durationMs?: number;
  activationMethod?: string;
  targetResolution?: {
    method: "coordinates" | "selector-text" | "selector" | "text" | "pointer-lock-canvas-current-mouse";
    selector?: string;
    text?: string;
    fallback?: string;
    matchCount?: number;
  };
  executionError?: {
    name?: string;
    message: string;
  };
  mouseCenterPoint?: Point;
  mouseMove?: {
    from: Point;
    to: Point;
    delta: Point;
    steps: number;
  };
  ignoredCoordinates?: Point;
  hitTarget?: ActionHitTarget;
}

export interface VideoSegment {
  label: string;
  startMs: number;
  endMs: number;
  preBufferMs?: number;
  postBufferMs?: number;
  minDurationMs?: number;
}

export interface SimulationDecision {
  summary: string;
  action: SimulationAction;
  targetState: string;
  expectedTransition?: string;
  psychometricObservation?: PsychometricObservation;
  _judge?: JudgeProvenance;
}

export interface ProgressObservation {
  phase: "held-action-observe" | "post-action-advance";
  turnIndex: number;
  screenshotPath: string;
  simulatedElapsedMs: number;
  reachedGoal: boolean;
  waitMs: number;
  observation: string;
  confidence: number;
  psychometricObservation?: PsychometricObservation;
  _judge?: JudgeProvenance;
}

export interface PsychometricObservation {
  positive?: string[];
  negative?: string[];
}

export interface JudgeProvenance {
  requestedModel: string;
  responseModel?: string;
  baseUrl: string;
  latencyMs: number;
}

export interface ExtraGlobalJudgerRuntime {
  enabled: boolean;
  judge: { model: string; provider?: ProviderFile };
}

export interface TurnVisualEvaluation {
  judgeType: "screenshot" | "transition";
  turnIndex: number;
  artifactPath: string;
  evaluationPath: string;
  /** Backward-compatible primary score. For new judges this equals overallReward when present. */
  score: number;
  /** Product-quality weighted reward. */
  overallReward?: number;
  /** Visual stability, rendering coherence, layout stability, and absence of broken frames. */
  visualStabilityReward?: number;
  /** TransitionJudger: whether the simulator action was delivered to the intended target. */
  executionReward?: number;
  /** TransitionJudger: whether the app/game feedback is clear, rule-consistent, and useful. */
  appFeedbackReward?: number;
  /** TransitionJudger: weakly weighted task/scenario progress. */
  taskProgressReward?: number;
  /** ScreenshotJudger: screenshot text/UI readability. */
  readabilityReward?: number;
  /** ScreenshotJudger: whether the screenshot exposes enough page/game state to evaluate. */
  stateVisibilityReward?: number;
  /** ScreenshotJudger: visual polish, composition, color, and aesthetic quality. */
  screenshotAestheticReward?: number;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  /** TransitionJudger: concise guidance for future action selection. */
  experience?: string;
  /** TransitionJudger: compact but detailed description of before.png state. */
  state?: string;
  /** TransitionJudger: normalized reason a transition failed or "none". */
  failureMode?: TransitionFailureMode;
  _judge?: JudgeProvenance;
}

export interface VisualEvaluationSummary {
  screenshot?: {
    count: number;
    averageScore: number;
    averageOverallReward?: number;
    averageVisualStabilityReward?: number;
    averageReadabilityReward?: number;
    averageStateVisibilityReward?: number;
    averageScreenshotAestheticReward?: number;
  };
  transition?: {
    count: number;
    averageScore: number;
    averageOverallReward?: number;
    averageVisualStabilityReward?: number;
    averageExecutionReward?: number;
    averageAppFeedbackReward?: number;
    averageTaskProgressReward?: number;
    failureModes?: Array<{
      failureMode: TransitionFailureMode;
      count: number;
      recentExamples: Array<{ turnIndex: number; summary: string }>;
    }>;
  };
}

export interface PreRolloutReference {
  promptId: string;
  screenshotPath: string;
  evaluationPath?: string | null;
  screenshotScore?: number | null;
  turnIndex: number;
  selection: string;
}

export interface PreRolloutVisualComparison {
  judgeType: "pre-rollout-visual-comparison";
  evaluationPath: string;
  referenceScreenshotPath: string;
  referenceEvaluationPath?: string;
  referenceScreenshotScore?: number;
  referenceTurnIndex: number;
  currentScreenshotPath: string;
  currentScreenshotScore: number;
  currentTurnIndex: number;
  degradationLevel: number;
  summary: string;
  referenceAdvantages: string[];
  currentAdvantages: string[];
  currentDeficiencies: string[];
  _judge?: JudgeProvenance;
}

export interface GlobalSimulationEvaluation {
  judgeType: "global";
  artifactPath: string;
  evaluationPath: string;
  score: number;
  /** Stage-2 score before the optional pre-rollout visual penalty. */
  unpenalizedScore?: number;
  preRolloutVisualPenalty?: number;
  preRolloutVisualComparisonSkipped?: boolean;
  preRolloutComparison?: PreRolloutVisualComparison;
  weightedScore: number;
  usabilityCap?: number;
  criticalBlocker: boolean;
  criticalBlockerReason?: string;
  confidence: number;
  visualStabilityReward: number;
  screenshotAestheticReward: number;
  feedbackClarityReward: number;
  interactionReliabilityReward: number;
  stateLegibilityReward: number;
  promptAlignmentReward: number;
  taskProgressReward: number;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  recurringIssues: Array<{
    issue: string;
    severity: "low" | "medium" | "high";
    turns: number[];
    evidence: string;
  }>;
  failureModeSummary: Array<{
    failureMode: TransitionFailureMode;
    count: number;
    impact: string;
  }>;
  experience: string;
  recommendations: string[];
  promptPath?: string;
  promptTruncated?: boolean;
  _judge?: JudgeProvenance;
}

export interface ActionHint {
  tag: string;
  selectorHint?: string;
  text?: string;
  rect: { x: number; y: number; width: number; height: number };
}

export interface RunUserSimulationInput {
  url: string;
  taskPrompt?: string;
  artifactPath?: string;
  includeCode?: boolean;
  personaPath?: string;
  psychometricPath?: string;
  outputDir?: string;
  stepMs?: number;
  frameMs?: number;
  recordVideo?: boolean;
  saveFullRecord?: boolean;
  turnLimit?: number;
  /** @deprecated Use turnLimit. */
  steps?: number;
  maxAdvanceMs?: number;
  preRolloutReference?: PreRolloutReference;
  skipPreRolloutVisualComparison?: boolean;
  /** Optional online JavaScript call-trace attribution for reward training. */
  actionTrace?: OnlineActionTraceConfig;
}

export interface OnlineActionTraceConfig {
  enabled?: boolean;
  /** Select actions whose transition judge score is at least this value. */
  threshold: number;
  /** Hard cap on Debugger.paused events collected for one action. */
  maxPauseEventsPerAction?: number;
}

export interface ActionTraceByteRun {
  /** Inclusive UTF-8 byte offset in the exact artifact HTML. */
  utf8ByteStart: number;
  /** Exclusive UTF-8 byte offset in the exact artifact HTML. */
  utf8ByteEnd: number;
  /** Leaf function is depth 0; its caller is depth 1, and so on. */
  depth: number;
}

export interface ActionTraceTurnSummary {
  turnIndex: number;
  actionType: SimulationAction["type"];
  transitionScore?: number;
  selected: boolean;
  status: "not-traceable" | "below-threshold" | "complete" | "partial" | "failed" | "empty";
  pauseEventCount: number;
  stackCount: number;
  functionCount: number;
  warnings?: string[];
  errors?: string[];
}

export interface OnlineActionTraceAttribution {
  schemaVersion: 1;
  enabled: true;
  status: "complete" | "partial" | "empty" | "failed";
  sourceScope: "artifact-inline-main-frame-javascript";
  offsetEncoding: "utf-8-bytes";
  artifactSha256: string;
  artifactUtf8ByteLength: number;
  threshold: number;
  maxPauseEventsPerAction: number;
  selectedActionCount: number;
  attributedActionCount: number;
  runs: ActionTraceByteRun[];
  turns: ActionTraceTurnSummary[];
  warnings?: string[];
  errors?: string[];
}

export interface UserSimulationRuntime {
  cwd: string;
  browser?: string;
  headed?: boolean;
  /** Per-turn screenshot and transition evaluator. */
  turnJudge: ResolvedSimulationJudge;
  /** Action decision model. */
  decisionJudge: ResolvedSimulationJudge;
  personaPath?: string;
  psychometricPath?: string;
  outputDir?: string;
  stepMs?: number;
  frameMs?: number;
  recordVideo?: boolean;
  saveFullRecord?: boolean;
  turnLimit?: number;
  extraGlobalJudger?: ExtraGlobalJudgerRuntime;
  /** @deprecated Use turnLimit. */
  defaultSteps?: number;
}

export interface UserSimulationResult {
  url: string;
  runDir: string;
  /** Present when a partial result was persisted before the simulation failed. */
  simulationError?: string;
  turnLimit: number;
  turns: UserSimulationTurn[];
  terminated: boolean;
  terminationReason?: string;
  turnCount: number;
  finalScreenshotPath: string;
  totalSimulatedAdvanceMs: number;
  fullRecordPath?: string;
  fullRecordError?: string;
  videoProcessingErrors?: string[];
  visualEvaluationSummary?: VisualEvaluationSummary;
  visualEvaluationErrors?: string[];
  globalEvaluation?: GlobalSimulationEvaluation;
  globalEvaluationPath?: string;
  actionTraceAttribution?: OnlineActionTraceAttribution;
  actionTraceAttributionPath?: string;
  stateTracePath: string;
  resultPath: string;
  /** @deprecated Single-turn compatibility field. */
  action: SimulationAction;
  /** @deprecated Single-turn compatibility field. */
  decision: SimulationDecision;
  /** @deprecated Single-turn compatibility field. */
  beforeScreenshotPath: string;
  /** @deprecated Single-turn compatibility field. */
  afterScreenshotPath: string;
  /** @deprecated Single-turn compatibility field. */
  observationScreenshotPaths: string[];
  /** @deprecated Single-turn compatibility field. */
  observations: ProgressObservation[];
  /** @deprecated Single-turn compatibility field. */
  finalObservation?: ProgressObservation;
  /** @deprecated Single-turn compatibility field. */
  reachedGoal: boolean;
  /** @deprecated Single-turn compatibility field. */
  totalWaitMs: number;
  /** @deprecated Single-turn compatibility field. */
  simulatedAdvanceMs: number;
  /** @deprecated Single-turn compatibility field. */
  decisionPath: string;
}

export interface UserSimulationTurn {
  turnIndex: number;
  beforeScreenshotPath: string;
  decisionPath: string;
  decision: SimulationDecision;
  action: SimulationAction;
  actionExecution?: ActionExecutionMetadata;
  actionOverlayPath?: string;
  afterScreenshotPath: string;
  observationScreenshotPaths: string[];
  observations: ProgressObservation[];
  finalObservation?: ProgressObservation;
  reachedGoal: boolean;
  totalWaitMs: number;
  simulatedAdvanceMs: number;
  terminated: boolean;
  terminationReason?: string;
  screenshotEvaluation?: TurnVisualEvaluation;
  transitionEvaluation?: TurnVisualEvaluation;
  visualEvaluationErrors?: string[];
}

export interface LoadedText {
  path: string;
  text: string;
  truncated: boolean;
}
