import path from "node:path";
import chalk from "chalk";
import { callJsonJudge, type JudgeContentPart } from "./judge.js";
import type {
  TransitionFailureMode,
  TurnVisualEvaluation,
  UserSimulationRuntime,
  UserSimulationTurn,
} from "./types.js";
import { clampNumber, turnLabel, writeJson } from "./utils.js";

const TURN_JUDGER_SYSTEM_PROMPT = [
  "You are the combined TransitionJudger and ScreenshotJudger for an HTML game quality evaluation system.",
  "You receive exactly two images in this order: Image 1 is the before-action screenshot and Image 2 is the after-action screenshot.",
  "Evaluate both judge roles independently in one response. Never treat the simulator decision as ground truth.",
  "",
  "TransitionJudger:",
  "Evaluate the two screenshots as evidence of the game's interaction and visual transition quality.",
  "Use action execution and hit-test metadata to determine whether the simulator delivered the requested action to the intended target.",
  "Action execution warnings are simulator diagnostics. If metadata warns that a canvas selector click lacked coordinates, treat it as evidence that the simulator clicked the canvas center; for a 2D canvas internal target this is usually coordinate_miss, not proof that the game failed to respond.",
  "If metadata contains success=false or executionError, treat the simulator action as not delivered. Set executionReward low and classify failureMode as action_ineffective, while still judging visual stability and app feedback from the images.",
  "The targetState, expectedTransition, and decision summary describe simulator intent only. Do not score whether that prediction was fulfilled.",
  "Score the HTML game itself: visual continuity, responsiveness, feedback clarity, rule consistency, animation/state coherence, absence of crashes or broken rendering, and whether the observed transition makes sense for visible game mechanics.",
  "Use multi-dimensional rewards. Human product-quality preference should prioritize stable, readable, coherent visuals and understandable feedback over immediate task progress. Task progress is a weak component and must not dominate overall reward.",
  "Strengths and weaknesses must describe the game's transition, feedback, coherence, and robustness, not whether the simulator's predicted target was achieved.",
  "If hit-test metadata shows the action point missed the intended target, use coordinate_miss; do not conclude that the action mechanism itself is ineffective.",
  "If the action is invalid under visible state or ordinary app/game rules, use illegal_or_rejected. Clear rejection can still receive high visualStabilityReward and appFeedbackReward.",
  "If an apparently valid, correctly delivered action produces no feedback, state change, partial progress, or consistent behavior, use action_ineffective; treat it as a likely implementation bug, selector/event mismatch, or execution-path issue and penalize it strongly.",
  "Do not punish the app heavily just because the simulator chose a poor action. A rejected or no-op action can still earn high visualStabilityReward and appFeedbackReward when the page remains stable and feedback is clear.",
  "Describe the before image state compactly and extract reusable experience from the observed mechanics and feedback.",
  "Transition reward dimensions:",
  "- visualStabilityReward: rendering stability, layout coherence, readable visuals, absence of flicker, broken frames, crashes, or state corruption. This is the most important reward.",
  "- appFeedbackReward: clarity, immediacy, and rule-consistency of the app/game response. Legal rejection can score high when it is understandable.",
  "- executionReward: whether the simulator delivered the requested action to the intended element, coordinate, key, or mouse target according to metadata.",
  "- taskProgressReward: weakly weighted progress toward the task/scenario, such as starting play, moving, scoring, collecting, or completing an objective.",
  "Compute transition overallReward = 0.40 * visualStabilityReward + 0.25 * appFeedbackReward + 0.25 * executionReward + 0.10 * taskProgressReward.",
  "Transition rubric:",
  "- 90-100 overallReward: Stable, readable, coherent visuals with clear feedback and correct simulator execution; task progress may be partial or absent if a no-op or rejection is understandable.",
  "- 70-89 overallReward: Mostly stable and understandable, with minor ambiguity, weak feedback, or limited task progress.",
  "- 40-69 overallReward: Usable but meaningfully weak in feedback, execution clarity, visual stability, or interaction coherence.",
  "- 1-39 overallReward: Severe visual instability, broken rendering, execution mismatch, or an apparently valid action producing no understandable feedback.",
  "- 0 overallReward: The two images are blank, corrupted, missing essential page content, or impossible to evaluate.",
  "Transition failureMode definitions:",
  '- "none": the action succeeded or any failure is not relevant.',
  '- "coordinate_miss": the action point missed the intended target or landed on the wrong element or region.',
  '- "action_ineffective": the target was correct and the action should be valid by visible state, but the app did not respond as expected.',
  '- "illegal_or_rejected": the action was invalid by visible state or game rules, missing prerequisites, resource constraints, cooldowns, distance checks, or mechanics and was rejected or correctly no-oped.',
  '- "timing_issue": the action is plausible but appears too early, too late, or not held long enough.',
  '- "uncertain": there is not enough evidence to classify the failure.',
  "",
  "ScreenshotJudger:",
  "Evaluate Image 2, the after-action screenshot, for screenshot usability only.",
  "Focus on rendering quality, visual stability, visible page/game state, layout stability, readability, clipping, occlusion, artifacts, and whether the image is useful for judging page state.",
  "Do not score task completion, action success, game progress, or whether the simulator made a good choice. A screenshot can receive a high score even when the action failed, as long as the visual state is clear and stable.",
  "Rate screenshotAestheticReward separately for visual polish, composition, color harmony, spacing, and overall presentation quality. It is not included in screenshot score; the run-level global judge uses it.",
  "Compute screenshot score = 0.45 * visualStabilityReward + 0.30 * stateVisibilityReward + 0.25 * readabilityReward.",
  "Screenshot rubric:",
  "- 90-100: The screenshot is sharp, complete, readable, stable, and clearly shows page state after the action.",
  "- 70-89: The screenshot is generally useful, with minor clipping, small readability issues, or non-critical visual artifacts.",
  "- 40-69: The screenshot is partially useful but has meaningful occlusion, blurry text, layout instability, cropped important content, or ambiguous state.",
  "- 1-39: The screenshot is mostly unusable due to blank areas, severe clipping, unreadable UI, broken rendering, or failure to show relevant state.",
  "- 0: The screenshot is blank, missing, corrupted, or impossible to evaluate.",
  "",
  "Return only JSON with exactly these two top-level objects:",
  "{",
  '  "transitionEvaluation": {',
  '    "overallReward": number from 0 to 100,',
  '    "visualStabilityReward": number from 0 to 100,',
  '    "executionReward": number from 0 to 100,',
  '    "appFeedbackReward": number from 0 to 100,',
  '    "taskProgressReward": number from 0 to 100,',
  '    "summary": string,',
  '    "strengths": string[],',
  '    "weaknesses": string[],',
  '    "failureMode": "none" | "coordinate_miss" | "action_ineffective" | "illegal_or_rejected" | "timing_issue" | "uncertain",',
  '    "experience": string,',
  '    "state": string',
  "  },",
  '  "screenshotEvaluation": {',
  '    "score": number from 0 to 100,',
  '    "visualStabilityReward": number from 0 to 100,',
  '    "readabilityReward": number from 0 to 100,',
  '    "stateVisibilityReward": number from 0 to 100,',
  '    "screenshotAestheticReward": number from 0 to 100,',
  '    "summary": string,',
  '    "strengths": string[],',
  '    "weaknesses": string[]',
  "  }",
  "}",
].join("\n");

export interface EvaluateTurnJudgesArgs {
  turn: UserSimulationTurn;
  judge: UserSimulationRuntime["turnJudge"];
  taskPrompt?: string;
  runDir: string;
  traceEvents: unknown[];
}

export async function evaluateTurnJudges(args: EvaluateTurnJudgesArgs): Promise<void> {
  try {
    const evaluations = await evaluateCombinedTurn(args);
    args.turn.transitionEvaluation = evaluations.transition;
    args.turn.screenshotEvaluation = evaluations.screenshot;
    args.traceEvents.push({
      phase: "visual-evaluation",
      judgeType: "transition",
      turnIndex: args.turn.turnIndex,
      evaluationPath: evaluations.transition.evaluationPath,
      score: evaluations.transition.score,
      failureMode: evaluations.transition.failureMode,
    });
    args.traceEvents.push({
      phase: "visual-evaluation",
      judgeType: "screenshot",
      turnIndex: args.turn.turnIndex,
      evaluationPath: evaluations.screenshot.evaluationPath,
      score: evaluations.screenshot.score,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    pushVisualError(args.turn, args.traceEvents, message);
    throw new Error("Combined turn judge failed for " + turnLabel(args.turn.turnIndex) + ": " + message, { cause: err });
  }
}

async function evaluateCombinedTurn(
  args: EvaluateTurnJudgesArgs,
): Promise<{ transition: TurnVisualEvaluation; screenshot: TurnVisualEvaluation }> {
  const label = turnLabel(args.turn.turnIndex);
  const transitionPath = path.join(args.runDir, label + "-transition-evaluation.json");
  const screenshotPath = path.join(args.runDir, label + "-screenshot-evaluation.json");
  const textParts = [
    "Task prompt:\n" + (args.taskPrompt ?? "Evaluate this simulated user turn."),
    "Turn index:\n" + args.turn.turnIndex,
    "Chosen action:\n" + JSON.stringify(args.turn.action, null, 2),
    "Action execution metadata:\n" + JSON.stringify(args.turn.actionExecution ?? {}, null, 2),
    "Simulator intent only, not a scoring target - decision summary:\n" + args.turn.decision.summary,
    "Simulator intent only, not a scoring target - predicted target state:\n" + args.turn.decision.targetState,
    "Simulator intent only, not a scoring target - predicted transition:\n" + (args.turn.decision.expectedTransition ?? ""),
    "Observer reached target:\n" + String(args.turn.reachedGoal),
    "Timing:\n" + JSON.stringify({
      totalWaitMs: args.turn.totalWaitMs,
      simulatedAdvanceMs: args.turn.simulatedAdvanceMs,
    }, null, 2),
    "Image order is fixed: Image 1 is before the action; Image 2 is after the action. Both are attached even when byte-identical.",
  ];
  const contentParts: JudgeContentPart[] = [
    ...textParts.map((text): JudgeContentPart => ({ kind: "text", text })),
    { kind: "text", text: "Image 1: before\nPath: " + args.turn.beforeScreenshotPath },
    { kind: "image", path: args.turn.beforeScreenshotPath, mimeType: "image/png" },
    { kind: "text", text: "Image 2: after\nPath: " + args.turn.afterScreenshotPath },
    { kind: "image", path: args.turn.afterScreenshotPath, mimeType: "image/png" },
  ];
  const raw = await callJsonJudge({
    provider: args.judge.provider,
    routing: args.judge.routing,
    model: args.judge.model,
    systemPrompt: TURN_JUDGER_SYSTEM_PROMPT,
    textParts,
    attachments: [],
    contentParts,
    progressLabel: "combined transition/screenshot evaluation",
  });
  if (!isObject(raw.parsed?.transitionEvaluation) || !isObject(raw.parsed?.screenshotEvaluation)) {
    throw new Error("combined judge response must contain transitionEvaluation and screenshotEvaluation objects");
  }
  const transition = normalizeEvaluation({
    raw: raw.parsed.transitionEvaluation,
    judgeType: "transition",
    turnIndex: args.turn.turnIndex,
    artifactPath: args.turn.afterScreenshotPath,
    evaluationPath: transitionPath,
    provenance: raw.provenance,
  });
  const screenshot = normalizeEvaluation({
    raw: raw.parsed.screenshotEvaluation,
    judgeType: "screenshot",
    turnIndex: args.turn.turnIndex,
    artifactPath: args.turn.afterScreenshotPath,
    evaluationPath: screenshotPath,
    provenance: raw.provenance,
  });
  await Promise.all([
    writeJson(transitionPath, transition),
    writeJson(screenshotPath, screenshot),
  ]);
  process.stderr.write(
    chalk.gray(
      "  · combined evaluation " + label +
      " transition=" + transition.score + "/100 screenshot=" + screenshot.score + "/100\n",
    ),
  );
  return { transition, screenshot };
}

function normalizeEvaluation(args: {
  raw: any;
  judgeType: "screenshot" | "transition";
  turnIndex: number;
  artifactPath: string;
  evaluationPath: string;
  provenance: TurnVisualEvaluation["_judge"];
}): TurnVisualEvaluation {
  const overallReward = normalizeOptionalReward(args.raw?.overallReward);
  const score = overallReward ?? clampNumber(args.raw?.score, 0, 100, 0);
  return {
    judgeType: args.judgeType,
    turnIndex: args.turnIndex,
    artifactPath: args.artifactPath,
    evaluationPath: args.evaluationPath,
    score,
    overallReward,
    visualStabilityReward: normalizeOptionalReward(args.raw?.visualStabilityReward),
    executionReward: args.judgeType === "transition" ? normalizeOptionalReward(args.raw?.executionReward) : undefined,
    appFeedbackReward: args.judgeType === "transition" ? normalizeOptionalReward(args.raw?.appFeedbackReward) : undefined,
    taskProgressReward: args.judgeType === "transition" ? normalizeOptionalReward(args.raw?.taskProgressReward) : undefined,
    readabilityReward: args.judgeType === "screenshot" ? normalizeOptionalReward(args.raw?.readabilityReward) : undefined,
    stateVisibilityReward: args.judgeType === "screenshot" ? normalizeOptionalReward(args.raw?.stateVisibilityReward) : undefined,
    screenshotAestheticReward: args.judgeType === "screenshot"
      ? normalizeOptionalReward(args.raw?.screenshotAestheticReward)
      : undefined,
    summary: typeof args.raw?.summary === "string" ? args.raw.summary : "",
    strengths: normalizeStringArray(args.raw?.strengths),
    weaknesses: normalizeStringArray(args.raw?.weaknesses),
    experience: typeof args.raw?.experience === "string" ? args.raw.experience : undefined,
    state: typeof args.raw?.state === "string" ? args.raw.state : undefined,
    failureMode: args.judgeType === "transition" ? normalizeFailureMode(args.raw?.failureMode) : undefined,
    _judge: args.provenance,
  };
}

function normalizeOptionalReward(raw: unknown): number | undefined {
  if (!Number.isFinite(raw)) return undefined;
  return clampNumber(raw, 0, 100, 0);
}

function normalizeStringArray(raw: unknown): string[] {
  return Array.isArray(raw) ? raw.filter((x): x is string => typeof x === "string") : [];
}

function normalizeFailureMode(raw: unknown): TransitionFailureMode {
  const value = typeof raw === "string" ? raw : "";
  if (
    value === "none" ||
    value === "coordinate_miss" ||
    value === "action_ineffective" ||
    value === "illegal_or_rejected" ||
    value === "timing_issue" ||
    value === "uncertain"
  ) {
    return value;
  }
  return "uncertain";
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function pushVisualError(
  turn: UserSimulationTurn,
  traceEvents: unknown[],
  message: string,
): void {
  turn.visualEvaluationErrors = [...(turn.visualEvaluationErrors ?? []), "combined: " + message];
  traceEvents.push({
    phase: "visual-evaluation-error",
    judgeType: "combined",
    turnIndex: turn.turnIndex,
    error: message,
  });
}
