import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { runUserSimulation, UserSimulationRunError } from "./engine.js";
import type {
  TurnVisualEvaluation,
  UserSimulationResult,
  UserSimulationRuntime,
  VisualEvaluationSummary,
  GlobalSimulationEvaluation,
  OnlineActionTraceAttribution,
  OnlineActionTraceConfig,
  PreRolloutReference,
} from "./types.js";
import { writeJson } from "./utils.js";

export interface RewardSimulationMetadata {
  globalStep?: string | number;
  promptId?: string | number;
  sampleId?: string | number;
  rolloutId?: string | number;
}

export interface RewardSimulationRequest extends RewardSimulationMetadata {
  html?: string;
  htmlFile?: string;
  taskPrompt?: string;
  outputRoot?: string;
  turnLimit?: number;
  stepMs?: number;
  frameMs?: number;
  maxAdvanceMs?: number;
  saveFullRecord?: boolean;
  recordVideo?: boolean;
  preRolloutReference?: PreRolloutReference;
  skipPreRolloutVisualComparison?: boolean;
  actionTrace?: OnlineActionTraceConfig;
}

export interface RewardSimulationResult {
  ok: boolean;
  score: number;
  scoreSource: "globalEvaluation.score / 100";
  globalScore?: number;
  unpenalizedGlobalScore?: number;
  preRolloutVisualPenalty?: number;
  transitionScore?: number;
  screenshotScore?: number;
  metadata: Required<Record<keyof RewardSimulationMetadata, string>>;
  paths: {
    sampleDir: string;
    artifactDir: string;
    artifactPath: string;
    runDir?: string;
    resultPath?: string;
    stateTracePath?: string;
    globalEvaluationPath?: string;
    preRolloutComparisonPath?: string;
    actionTraceAttributionPath?: string;
    rewardResultPath: string;
  };
  actionTraceAttribution?: OnlineActionTraceAttribution;
  visualEvaluationSummary?: VisualEvaluationSummary;
  transitionEvaluations: TurnVisualEvaluation[];
  screenshotEvaluations: TurnVisualEvaluation[];
  globalEvaluation?: GlobalSimulationEvaluation;
  errors: string[];
}

export async function runRewardSimulation(
  request: RewardSimulationRequest,
  runtime: UserSimulationRuntime,
): Promise<RewardSimulationResult> {
  const metadata = normalizeMetadata(request);
  const sampleDir = buildSampleDir(runtime.cwd, request.outputRoot ?? ".agens-reward", metadata);
  const artifactDir = path.join(sampleDir, "artifact");
  const artifactPath = path.join(artifactDir, "index.html");
  const promptPath = path.join(artifactDir, "prompt.txt");
  const traceRoot = path.join(sampleDir, "trace");
  const rewardResultPath = path.join(sampleDir, "reward-result.json");

  await mkdir(artifactDir, { recursive: true });
  await mkdir(traceRoot, { recursive: true });

  const errors: string[] = [];
  try {
    const html = await readHtml(request, runtime.cwd);
    await writeFile(artifactPath, html, "utf8");
    if (request.taskPrompt) {
      await writeFile(promptPath, request.taskPrompt, "utf8");
    }

    const result = await runUserSimulation(
      {
        url: pathToFileURL(artifactPath).href,
        taskPrompt: request.taskPrompt,
        artifactPath,
        includeCode: true,
        outputDir: traceRoot,
        turnLimit: request.turnLimit,
        stepMs: request.stepMs,
        frameMs: request.frameMs,
        maxAdvanceMs: request.maxAdvanceMs,
        saveFullRecord: request.saveFullRecord,
        recordVideo: request.recordVideo,
        preRolloutReference: request.preRolloutReference,
        skipPreRolloutVisualComparison: request.skipPreRolloutVisualComparison,
        actionTrace: request.actionTrace,
      },
      forceRewardGlobalJudger(runtime),
    );
    const rewardResult = buildRewardResult({
      metadata,
      sampleDir,
      artifactDir,
      artifactPath,
      rewardResultPath,
      result,
      errors,
    });
    await writeJson(rewardResultPath, rewardResult);
    return rewardResult;
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
    const rewardResult = buildRewardResult({
      metadata,
      sampleDir,
      artifactDir,
      artifactPath,
      rewardResultPath,
      result: error instanceof UserSimulationRunError ? error.result : undefined,
      errors,
    });
    await writeJson(rewardResultPath, rewardResult);
    return rewardResult;
  }
}

function buildRewardResult(args: {
  metadata: Required<Record<keyof RewardSimulationMetadata, string>>;
  sampleDir: string;
  artifactDir: string;
  artifactPath: string;
  rewardResultPath: string;
  result?: UserSimulationResult;
  errors: string[];
}): RewardSimulationResult {
  const transitionEvaluations = args.result?.turns.flatMap((turn) =>
    turn.transitionEvaluation ? [turn.transitionEvaluation] : [],
  ) ?? [];
  const screenshotEvaluations = args.result?.turns.flatMap((turn) =>
    turn.screenshotEvaluation ? [turn.screenshotEvaluation] : [],
  ) ?? [];
  const globalScore = finiteNumber(args.result?.globalEvaluation?.score);
  const unpenalizedGlobalScore = finiteNumber(args.result?.globalEvaluation?.unpenalizedScore);
  const preRolloutVisualPenalty = finiteNumber(args.result?.globalEvaluation?.preRolloutVisualPenalty);
  const transitionScore = finiteNumber(args.result?.visualEvaluationSummary?.transition?.averageScore);
  const screenshotScore = finiteNumber(args.result?.visualEvaluationSummary?.screenshot?.averageScore);
  const errors = [...args.errors];
  if (globalScore === undefined) {
    errors.push("Missing globalEvaluation.score; reward defaults to 0.");
  }

  return {
    ok: errors.length === 0,
    score: globalScore === undefined ? 0 : globalScore / 100,
    scoreSource: "globalEvaluation.score / 100",
    globalScore,
    unpenalizedGlobalScore,
    preRolloutVisualPenalty,
    transitionScore,
    screenshotScore,
    metadata: args.metadata,
    paths: {
      sampleDir: args.sampleDir,
      artifactDir: args.artifactDir,
      artifactPath: args.artifactPath,
      runDir: args.result?.runDir,
      resultPath: args.result?.resultPath,
      stateTracePath: args.result?.stateTracePath,
      globalEvaluationPath: args.result?.globalEvaluationPath,
      preRolloutComparisonPath: args.result?.globalEvaluation?.preRolloutComparison?.evaluationPath,
      actionTraceAttributionPath: args.result?.actionTraceAttributionPath,
      rewardResultPath: args.rewardResultPath,
    },
    actionTraceAttribution: args.result?.actionTraceAttribution,
    visualEvaluationSummary: args.result?.visualEvaluationSummary,
    transitionEvaluations,
    screenshotEvaluations,
    globalEvaluation: args.result?.globalEvaluation,
    errors,
  };
}

async function readHtml(request: RewardSimulationRequest, cwd: string): Promise<string> {
  if (request.html !== undefined) return request.html;
  if (!request.htmlFile) throw new Error("Reward simulation request must include html or htmlFile.");
  return await readFile(path.resolve(cwd, request.htmlFile), "utf8");
}

function forceRewardGlobalJudger(runtime: UserSimulationRuntime): UserSimulationRuntime {
  return {
    ...runtime,
    extraGlobalJudger: {
      enabled: true,
      judge: {
        model: runtime.decisionJudge.model,
        provider: runtime.decisionJudge.provider,
      },
    },
  };
}

function buildSampleDir(
  cwd: string,
  outputRoot: string,
  metadata: Required<Record<keyof RewardSimulationMetadata, string>>,
): string {
  const root = path.resolve(cwd, outputRoot);
  return path.join(
    root,
    `step-${safePathPart(metadata.globalStep)}`,
    `prompt-${safePathPart(metadata.promptId)}`,
    `sample-${safePathPart(metadata.sampleId)}`,
    `run-${timestamp()}`,
  );
}

function normalizeMetadata(request: RewardSimulationRequest): Required<Record<keyof RewardSimulationMetadata, string>> {
  const rolloutId = stringifyMeta(request.rolloutId) ?? "unknown";
  return {
    globalStep: stringifyMeta(request.globalStep) ?? "unknown",
    promptId: stringifyMeta(request.promptId) ?? "unknown",
    sampleId: stringifyMeta(request.sampleId) ?? rolloutId,
    rolloutId,
  };
}

function stringifyMeta(value: string | number | undefined): string | undefined {
  if (value === undefined || value === null) return undefined;
  return String(value);
}

function safePathPart(value: string): string {
  return value.replace(/[^a-zA-Z0-9._=-]+/g, "_").slice(0, 120) || "unknown";
}

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
