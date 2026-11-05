import path from "node:path";
import chalk from "chalk";
import { callJsonJudge, type JudgeContentPart } from "./judge.js";
import type {
  ExtraGlobalJudgerRuntime,
  GlobalSimulationEvaluation,
  LoadedText,
  PreRolloutReference,
  PreRolloutVisualComparison,
  TransitionFailureMode,
  TurnVisualEvaluation,
  UserSimulationTurn,
  VisualEvaluationSummary,
} from "./types.js";
import { clampNumber, turnLabel, writeJson } from "./utils.js";

const GLOBAL_JUDGER_SYSTEM_PROMPT = `You are GlobalJudger for an HTML game/user-experience simulation.
You receive structured per-turn transition and screenshot evaluations only. You do not receive screenshots. Do not invent unseen visual details; use screenshot summaries, strengths, weaknesses, and reward fields as visual evidence.

Your job is to judge the full play episode as a human product-quality reviewer. Consider whether the page/game is stable, understandable, playable, responsive, aligned with the provided prompt.txt when present, and whether critical flows actually work.

Use transition evaluations for interaction quality, feedback clarity, execution reliability, failureMode, experience, state, and task progress.
Use screenshot evaluations for visual stability, state legibility, readability, and screenshotAestheticReward.

Compute weightedScore before any cap:
weightedScore =
0.15 * visualStabilityReward
+ 0.15 * screenshotAestheticReward
+ 0.10 * feedbackClarityReward
+ 0.20 * interactionReliabilityReward
+ 0.10 * stateLegibilityReward
+ 0.20 * promptAlignmentReward
+ 0.10 * taskProgressReward

Only apply a usability cap as a severity guardrail when the weightedScore would otherwise overstate the real user experience. Do not set usabilityCap merely because there is a flaw. Ordinary UX weaknesses, ambiguous feedback, low task progress, occasional ineffective actions, missing polish, or partially unreliable secondary mechanics should reduce the reward dimensions but should not set usabilityCap.

Choose the highest cap consistent with the observed severity:
- Cap at 0-10 if the page is blank, crashed, corrupted, or impossible to evaluate.
- Cap at 20 if the page renders but is effectively non-interactive: no meaningful primary interaction works at all, or the app cannot leave an inert/static state.
- Cap at 35 if a required primary entry action such as Start, Play, Begin, Next, or an equivalent first-step control appears correctly targeted but is nonfunctional, so the main experience cannot begin.
- Cap at 45 if the artifact is substantially the wrong product or the requested core experience is absent, even if some unrelated UI or visuals work.
- Cap at 60 only for severe but non-total incompleteness: the main artifact is recognizable and partially usable, but the central loop or a required primary mechanic is absent, blocked, or impossible to complete.
- Cap at 70 for major prompt gaps where the main loop works and the user can make meaningful progress, but a prominent required feature/content area is missing or clearly nonfunctional.
- Cap at 80 for notable missing content, weak production breadth, or unreliable secondary mechanics when the primary experience remains playable.
- Cap at 90 for minor prompt omissions or polish gaps only when the requested score would otherwise be near-perfect.
- Leave usabilityCap null when the flaw is already well represented by low promptAlignmentReward, taskProgressReward, feedbackClarityReward, or interactionReliabilityReward.
- For broad prompts requesting many assets, levels, weapons, sounds, or polish, do not cap at 60 solely because every requested item was not observed in a short episode. Use reward dimensions, or a higher cap such as 70/80, unless the missing item blocks the primary loop or makes the artifact substantially not what was requested.
- Do not apply a cap for clear rule-based rejections such as Too Far, Need key, No ammo, cooldown, or invalid move.
- Do not apply a cap if failures are mostly coordinate_miss or otherwise caused by simulator/model execution; lower confidence instead.

Classify a critical blocker only when evidence shows a severe defect or required-content failure that blocks the main requested experience, not merely a weak or confusing interaction. In general, criticalBlocker should be true only for caps of 60 or lower; higher caps usually describe non-critical limitations.

Return only JSON with:
{
  "weightedScore": number from 0 to 100,
  "score": number from 0 to 100 after applying usabilityCap,
  "usabilityCap": number from 0 to 100 or null,
  "criticalBlocker": boolean,
  "criticalBlockerReason": string or null,
  "confidence": number from 0 to 100,
  "visualStabilityReward": number from 0 to 100,
  "screenshotAestheticReward": number from 0 to 100,
  "feedbackClarityReward": number from 0 to 100,
  "interactionReliabilityReward": number from 0 to 100,
  "stateLegibilityReward": number from 0 to 100,
  "promptAlignmentReward": number from 0 to 100,
  "taskProgressReward": number from 0 to 100,
  "summary": string,
  "strengths": string[],
  "weaknesses": string[],
  "recurringIssues": [{"issue": string, "severity": "low" | "medium" | "high", "turns": number[], "evidence": string}],
  "failureModeSummary": [{"failureMode": "coordinate_miss" | "action_ineffective" | "illegal_or_rejected" | "timing_issue" | "uncertain" | "none", "count": number, "impact": string}],
  "experience": string,
  "recommendations": string[]
}`;

const PRE_ROLLOUT_COMPARISON_SYSTEM_PROMPT = `You compare two screenshots of HTML games created for the exact same task prompt.
Image 1 is the retained pre-rollout reference. Image 2 is the current training rollout being evaluated.

Judge only visual presentation: scene quality, beauty, composition, visual hierarchy, layout organization, spacing, typography, color harmony, coherence, polish, and readability. Do not penalize ordinary differences in gameplay progress, animation timing, selected controls, scores, or transient state unless they materially harm visual organization.

If the screenshots do not show an obvious visual-quality difference, assign degradationLevel 0. Treat the provided screenshot scores only as a calibration signal: when their absolute delta is at most 8 points, default to no penalty unless Image 2 shows a clear, material visual degradation. Do not penalize small, ambiguous, or merely numerical differences.

Return degradationLevel from 0 to 100 measuring how much worse Image 2 is than Image 1:
- 0: Image 2 is visually equal to or better than Image 1.
- 1-20: Image 2 is only slightly worse.
- 21-50: Image 2 is noticeably worse in multiple visual dimensions.
- 51-80: Image 2 is substantially less polished, coherent, or organized.
- 81-100: Image 2 is dramatically worse or visually unusable relative to Image 1.

Do not assign a positive penalty merely because the images differ. Return only JSON with:
{
  "degradationLevel": number from 0 to 100,
  "summary": string,
  "referenceAdvantages": string[],
  "currentAdvantages": string[],
  "currentDeficiencies": string[]
}`;

export async function evaluateGlobalSimulation(args: {
  judger: ExtraGlobalJudgerRuntime;
  url: string;
  runDir: string;
  turns: UserSimulationTurn[];
  turnLimit: number;
  taskPrompt?: string;
  artifactPrompt?: LoadedText;
  visualEvaluationSummary?: VisualEvaluationSummary;
  terminated: boolean;
  terminationReason?: string;
  totalSimulatedAdvanceMs: number;
  preRolloutReference?: PreRolloutReference;
  skipPreRolloutVisualComparison?: boolean;
}): Promise<GlobalSimulationEvaluation> {
  const evaluationPath = path.join(args.runDir, "global-evaluation.json");
  const comparisonSkipped = args.skipPreRolloutVisualComparison === true && !args.preRolloutReference;
  const comparison = args.preRolloutReference
    ? await evaluatePreRolloutVisualComparison({
        judger: args.judger,
        runDir: args.runDir,
        turns: args.turns,
        taskPrompt: args.taskPrompt,
        reference: args.preRolloutReference,
      })
    : undefined;
  const input = buildGlobalJudgeInput(args);
  const raw = await callJsonJudge({
    provider: args.judger.judge.provider,
    model: args.judger.judge.model,
    systemPrompt: GLOBAL_JUDGER_SYSTEM_PROMPT,
    textParts: [
      `Task prompt:\n${args.taskPrompt ?? "Evaluate the full simulated user experience."}`,
      ...(args.artifactPrompt
        ? [`Artifact prompt (${args.artifactPrompt.path}${args.artifactPrompt.truncated ? ", truncated" : ""}):\n${args.artifactPrompt.text}`]
        : ["Artifact prompt:\n(none found)"]),
      `Structured simulation evidence:\n${JSON.stringify(input, null, 2)}`,
    ],
    attachments: [],
    progressLabel: "global visual evaluation",
  });
  const evaluation = normalizeGlobalEvaluation({
    raw: raw.parsed,
    runDir: args.runDir,
    evaluationPath,
    artifactPrompt: args.artifactPrompt,
    provenance: raw.provenance,
  });
  if (comparison || comparisonSkipped) {
    evaluation.unpenalizedScore = evaluation.score;
    evaluation.preRolloutVisualPenalty = comparison?.degradationLevel ?? 0;
    evaluation.preRolloutVisualComparisonSkipped = comparisonSkipped || undefined;
    evaluation.preRolloutComparison = comparison;
    evaluation.score = evaluation.unpenalizedScore - evaluation.preRolloutVisualPenalty;
  }
  await writeJson(evaluationPath, evaluation);
  const penaltyLabel = comparison
    ? ` stage2=${evaluation.unpenalizedScore}/100 penalty=${comparison.degradationLevel}/100`
    : comparisonSkipped
      ? ` stage2=${evaluation.unpenalizedScore}/100 penalty=0/100 stage1=skipped-no-reference`
      : "";
  process.stderr.write(chalk.gray(`  · global evaluation score=${evaluation.score}/100${penaltyLabel}\n`));
  return evaluation;
}

async function evaluatePreRolloutVisualComparison(args: {
  judger: ExtraGlobalJudgerRuntime;
  runDir: string;
  turns: UserSimulationTurn[];
  taskPrompt?: string;
  reference: PreRolloutReference;
}): Promise<PreRolloutVisualComparison> {
  const current = args.turns
    .filter((turn) => turn.screenshotEvaluation && Number.isFinite(turn.screenshotEvaluation.score))
    .sort((left, right) => {
      const scoreDifference = right.screenshotEvaluation!.score - left.screenshotEvaluation!.score;
      return scoreDifference || left.turnIndex - right.turnIndex;
    })[0];
  if (!current?.screenshotEvaluation) {
    throw new Error("Pre-rollout comparison requires at least one current screenshot evaluation.");
  }

  const evaluationPath = path.join(args.runDir, "pre-rollout-visual-comparison.json");
  const textParts = [
    `Task prompt:\n${args.taskPrompt ?? "Compare visual quality for the same HTML game task."}`,
    `Reference selection metadata:\n${JSON.stringify({
      promptId: args.reference.promptId,
      turnIndex: args.reference.turnIndex,
      screenshotScore: args.reference.screenshotScore,
      selection: args.reference.selection,
    }, null, 2)}`,
    `Current selection metadata:\n${JSON.stringify({
      turnIndex: current.turnIndex,
      screenshotScore: current.screenshotEvaluation.score,
    }, null, 2)}`,
  ];
  const contentParts: JudgeContentPart[] = [
    ...textParts.map((text): JudgeContentPart => ({ kind: "text", text })),
    { kind: "text", text: `Image 1: pre-rollout reference\nPath: ${args.reference.screenshotPath}` },
    { kind: "image", path: args.reference.screenshotPath, mimeType: "image/png" },
    { kind: "text", text: `Image 2: current evaluated HTML\nPath: ${current.beforeScreenshotPath}` },
    { kind: "image", path: current.beforeScreenshotPath, mimeType: "image/png" },
  ];
  const raw = await callJsonJudge({
    provider: args.judger.judge.provider,
    model: args.judger.judge.model,
    systemPrompt: PRE_ROLLOUT_COMPARISON_SYSTEM_PROMPT,
    textParts,
    attachments: [],
    contentParts,
    progressLabel: "pre-rollout visual comparison",
  });
  if (!Number.isFinite(raw.parsed?.degradationLevel)) {
    throw new Error("Pre-rollout visual comparison must return a numeric degradationLevel.");
  }
  const comparison: PreRolloutVisualComparison = {
    judgeType: "pre-rollout-visual-comparison",
    evaluationPath,
    referenceScreenshotPath: args.reference.screenshotPath,
    referenceEvaluationPath: args.reference.evaluationPath ?? undefined,
    referenceScreenshotScore: normalizeOptionalReward(args.reference.screenshotScore),
    referenceTurnIndex: args.reference.turnIndex,
    currentScreenshotPath: current.beforeScreenshotPath,
    currentScreenshotScore: current.screenshotEvaluation.score,
    currentTurnIndex: current.turnIndex,
    degradationLevel: clampNumber(raw.parsed.degradationLevel, 0, 100, 0),
    summary: typeof raw.parsed?.summary === "string" ? raw.parsed.summary : "",
    referenceAdvantages: normalizeStringArray(raw.parsed?.referenceAdvantages),
    currentAdvantages: normalizeStringArray(raw.parsed?.currentAdvantages),
    currentDeficiencies: normalizeStringArray(raw.parsed?.currentDeficiencies),
    _judge: raw.provenance,
  };
  await writeJson(evaluationPath, comparison);
  process.stderr.write(
    chalk.gray(`  · pre-rollout visual degradation=${comparison.degradationLevel}/100\n`),
  );
  return comparison;
}

function buildGlobalJudgeInput(args: {
  url: string;
  turns: UserSimulationTurn[];
  turnLimit: number;
  visualEvaluationSummary?: VisualEvaluationSummary;
  terminated: boolean;
  terminationReason?: string;
  totalSimulatedAdvanceMs: number;
}): Record<string, unknown> {
  return {
    run: {
      url: args.url,
      turnCount: args.turns.length,
      turnLimit: args.turnLimit,
      terminated: args.terminated,
      terminationReason: args.terminationReason,
      totalSimulatedAdvanceMs: args.totalSimulatedAdvanceMs,
      visualEvaluationSummary: args.visualEvaluationSummary,
    },
    turns: args.turns.map((turn) => ({
      turnIndex: turn.turnIndex,
      turnLabel: turnLabel(turn.turnIndex),
      action: turn.action,
      actionExecution: turn.actionExecution,
      timing: {
        totalWaitMs: turn.totalWaitMs,
        simulatedAdvanceMs: turn.simulatedAdvanceMs,
      },
      reachedGoal: turn.reachedGoal,
      terminated: turn.terminated,
      terminationReason: turn.terminationReason,
      decisionSummary: turn.decision.summary,
      transition: compactEvaluation(turn.transitionEvaluation),
      screenshot: compactEvaluation(turn.screenshotEvaluation),
      visualEvaluationErrors: turn.visualEvaluationErrors,
    })),
  };
}

function compactEvaluation(evaluation: TurnVisualEvaluation | undefined): Record<string, unknown> | undefined {
  if (!evaluation) return undefined;
  return {
    judgeType: evaluation.judgeType,
    turnIndex: evaluation.turnIndex,
    score: evaluation.score,
    overallReward: evaluation.overallReward,
    visualStabilityReward: evaluation.visualStabilityReward,
    executionReward: evaluation.executionReward,
    appFeedbackReward: evaluation.appFeedbackReward,
    taskProgressReward: evaluation.taskProgressReward,
    readabilityReward: evaluation.readabilityReward,
    stateVisibilityReward: evaluation.stateVisibilityReward,
    screenshotAestheticReward: evaluation.screenshotAestheticReward,
    summary: evaluation.summary,
    strengths: evaluation.strengths,
    weaknesses: evaluation.weaknesses,
    experience: evaluation.experience,
    state: evaluation.state,
    failureMode: evaluation.failureMode,
  };
}

function normalizeGlobalEvaluation(args: {
  raw: any;
  runDir: string;
  evaluationPath: string;
  artifactPrompt?: LoadedText;
  provenance: GlobalSimulationEvaluation["_judge"];
}): GlobalSimulationEvaluation {
  const visualStabilityReward = normalizeReward(args.raw?.visualStabilityReward);
  const screenshotAestheticReward = normalizeReward(args.raw?.screenshotAestheticReward);
  const feedbackClarityReward = normalizeReward(args.raw?.feedbackClarityReward);
  const interactionReliabilityReward = normalizeReward(args.raw?.interactionReliabilityReward);
  const stateLegibilityReward = normalizeReward(args.raw?.stateLegibilityReward);
  const promptAlignmentReward = normalizeReward(args.raw?.promptAlignmentReward);
  const taskProgressReward = normalizeReward(args.raw?.taskProgressReward);
  const computedWeightedScore = computeWeightedScore({
    visualStabilityReward,
    screenshotAestheticReward,
    feedbackClarityReward,
    interactionReliabilityReward,
    stateLegibilityReward,
    promptAlignmentReward,
    taskProgressReward,
  });
  const weightedScore = normalizeOptionalReward(args.raw?.weightedScore) ?? computedWeightedScore;
  const usabilityCap = normalizeOptionalReward(args.raw?.usabilityCap);
  const requestedScore = normalizeOptionalReward(args.raw?.score) ?? weightedScore;
  const score = usabilityCap === undefined ? requestedScore : Math.min(requestedScore, usabilityCap);
  return {
    judgeType: "global",
    artifactPath: args.runDir,
    evaluationPath: args.evaluationPath,
    score,
    weightedScore,
    usabilityCap,
    criticalBlocker: args.raw?.criticalBlocker === true,
    criticalBlockerReason: typeof args.raw?.criticalBlockerReason === "string" ? args.raw.criticalBlockerReason : undefined,
    confidence: clampNumber(args.raw?.confidence, 0, 100, 50),
    visualStabilityReward,
    screenshotAestheticReward,
    feedbackClarityReward,
    interactionReliabilityReward,
    stateLegibilityReward,
    promptAlignmentReward,
    taskProgressReward,
    summary: typeof args.raw?.summary === "string" ? args.raw.summary : "",
    strengths: normalizeStringArray(args.raw?.strengths),
    weaknesses: normalizeStringArray(args.raw?.weaknesses),
    recurringIssues: normalizeRecurringIssues(args.raw?.recurringIssues),
    failureModeSummary: normalizeFailureModeSummary(args.raw?.failureModeSummary),
    experience: typeof args.raw?.experience === "string" ? args.raw.experience : "",
    recommendations: normalizeStringArray(args.raw?.recommendations),
    promptPath: args.artifactPrompt?.path,
    promptTruncated: args.artifactPrompt?.truncated,
    _judge: args.provenance,
  };
}

function computeWeightedScore(rewards: {
  visualStabilityReward: number;
  screenshotAestheticReward: number;
  feedbackClarityReward: number;
  interactionReliabilityReward: number;
  stateLegibilityReward: number;
  promptAlignmentReward: number;
  taskProgressReward: number;
}): number {
  const score =
    0.15 * rewards.visualStabilityReward +
    0.15 * rewards.screenshotAestheticReward +
    0.10 * rewards.feedbackClarityReward +
    0.20 * rewards.interactionReliabilityReward +
    0.10 * rewards.stateLegibilityReward +
    0.20 * rewards.promptAlignmentReward +
    0.10 * rewards.taskProgressReward;
  return Math.round(score * 100) / 100;
}

function normalizeReward(raw: unknown): number {
  return clampNumber(raw, 0, 100, 0);
}

function normalizeOptionalReward(raw: unknown): number | undefined {
  if (!Number.isFinite(raw)) return undefined;
  return clampNumber(raw, 0, 100, 0);
}

function normalizeStringArray(raw: unknown): string[] {
  return Array.isArray(raw) ? raw.filter((x): x is string => typeof x === "string") : [];
}

function normalizeRecurringIssues(raw: unknown): GlobalSimulationEvaluation["recurringIssues"] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    if (!item || typeof item.issue !== "string") return [];
    const severity = item.severity === "high" || item.severity === "medium" || item.severity === "low"
      ? item.severity
      : "medium";
    const turns = Array.isArray(item.turns)
      ? item.turns.filter((turn: unknown): turn is number => Number.isInteger(turn))
      : [];
    return [{
      issue: item.issue,
      severity,
      turns,
      evidence: typeof item.evidence === "string" ? item.evidence : "",
    }];
  });
}

function normalizeFailureModeSummary(raw: unknown): GlobalSimulationEvaluation["failureModeSummary"] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    const failureMode = normalizeFailureMode(item?.failureMode);
    if (!failureMode) return [];
    return [{
      failureMode,
      count: Number.isInteger(item.count) && item.count >= 0 ? item.count : 0,
      impact: typeof item.impact === "string" ? item.impact : "",
    }];
  });
}

function normalizeFailureMode(raw: unknown): TransitionFailureMode | undefined {
  const allowed: TransitionFailureMode[] = [
    "none",
    "coordinate_miss",
    "action_ineffective",
    "illegal_or_rejected",
    "timing_issue",
    "uncertain",
  ];
  return allowed.includes(raw as TransitionFailureMode) ? raw as TransitionFailureMode : undefined;
}
