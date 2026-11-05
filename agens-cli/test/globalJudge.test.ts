import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { evaluateGlobalSimulation } from "../src/simulation/globalJudge.js";
import type { UserSimulationTurn } from "../src/simulation/types.js";

test("global judge receives structured evidence and prompt text without images or source input", async () => {
  const runDir = await mkdtemp(path.join(tmpdir(), "agens-global-judge-"));
  const turn: UserSimulationTurn = {
    turnIndex: 0,
    beforeScreenshotPath: "turn-000-before.png",
    decisionPath: "turn-000-decision.json",
    decision: {
      summary: "Start the game.",
      action: { type: "click", selector: "#start" },
      targetState: "Gameplay.",
    },
    action: { type: "click", selector: "#start" },
    afterScreenshotPath: "turn-000-after.png",
    observationScreenshotPaths: [],
    observations: [],
    reachedGoal: true,
    totalWaitMs: 50,
    simulatedAdvanceMs: 50,
    terminated: false,
  };

  const savedFetch = globalThis.fetch;
  let requestBody: any;
  try {
    globalThis.fetch = (async (_input, init) => {
      requestBody = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        model: "global-response-model",
        choices: [{ message: { content: "{}" } }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    await evaluateGlobalSimulation({
      judger: {
        enabled: true,
        judge: {
          model: "global-model",
          provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
        },
      },
      url: "file:///game/index.html",
      runDir,
      turns: [turn],
      turnLimit: 1,
      taskPrompt: "Play the game.",
      artifactPrompt: {
        path: path.join(runDir, "prompt.txt"),
        text: "ARTIFACT_PROMPT_SENTINEL",
        truncated: false,
      },
      terminated: false,
      totalSimulatedAdvanceMs: 50,
    });
  } finally {
    globalThis.fetch = savedFetch;
  }

  const messages = requestBody.messages as Array<{ content: any }>;
  const userParts = messages[1].content as Array<{ type: string; text?: string }>;
  const userText = userParts.map((part) => part.text ?? "").join("\n");
  assert.equal(userParts.some((part) => part.type === "image_url"), false);
  assert.match(userText, /ARTIFACT_PROMPT_SENTINEL/);
  assert.match(userText, /Structured simulation evidence/);
  assert.doesNotMatch(userText, /Artifact source code/);
  await rm(runDir, { recursive: true, force: true });
});

test("pre-rollout comparison selects the best current before screenshot and allows a negative score", async () => {
  const runDir = await mkdtemp(path.join(tmpdir(), "agens-global-pre-rollout-"));
  const referencePath = path.join(runDir, "reference.png");
  const currentLowPath = path.join(runDir, "current-low-before.png");
  const currentHighPath = path.join(runDir, "current-high-before.png");
  await Promise.all([
    writeFile(referencePath, Buffer.from("reference-image")),
    writeFile(currentLowPath, Buffer.from("current-low-image")),
    writeFile(currentHighPath, Buffer.from("current-high-image")),
  ]);

  const makeTurn = (turnIndex: number, beforeScreenshotPath: string, score: number): UserSimulationTurn => ({
    turnIndex,
    beforeScreenshotPath,
    decisionPath: `turn-${turnIndex}-decision.json`,
    decision: {
      summary: "Inspect the game.",
      action: { type: "click", selector: "#start" },
      targetState: "Gameplay.",
    },
    action: { type: "click", selector: "#start" },
    afterScreenshotPath: `turn-${turnIndex}-after.png`,
    observationScreenshotPaths: [],
    observations: [],
    reachedGoal: false,
    totalWaitMs: 50,
    simulatedAdvanceMs: 50,
    terminated: false,
    screenshotEvaluation: {
      judgeType: "screenshot",
      turnIndex,
      artifactPath: `turn-${turnIndex}-after.png`,
      evaluationPath: `turn-${turnIndex}-screenshot-evaluation.json`,
      score,
      summary: "Readable screenshot.",
      strengths: [],
      weaknesses: [],
    },
  });

  const savedFetch = globalThis.fetch;
  const requestBodies: any[] = [];
  try {
    globalThis.fetch = (async (_input, init) => {
      requestBodies.push(JSON.parse(String(init?.body)));
      const content = requestBodies.length === 1
        ? {
            degradationLevel: 80,
            summary: "The current scene is substantially less organized.",
            referenceAdvantages: ["Clear hierarchy"],
            currentAdvantages: [],
            currentDeficiencies: ["Weak composition"],
          }
        : { score: 30, weightedScore: 30 };
      return new Response(JSON.stringify({
        model: "global-response-model",
        choices: [{ message: { content: JSON.stringify(content) } }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    const evaluation = await evaluateGlobalSimulation({
      judger: {
        enabled: true,
        judge: {
          model: "global-model",
          provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
        },
      },
      url: "file:///game/index.html",
      runDir,
      turns: [makeTurn(0, currentLowPath, 70), makeTurn(1, currentHighPath, 95)],
      turnLimit: 2,
      taskPrompt: "Build the same game.",
      terminated: false,
      totalSimulatedAdvanceMs: 100,
      preRolloutReference: {
        promptId: "000004",
        screenshotPath: referencePath,
        evaluationPath: "/reference/turn-000-screenshot-evaluation.json",
        screenshotScore: 97,
        turnIndex: 0,
        selection: "highest-screenshot-evaluation-score",
      },
    });

    assert.equal(requestBodies.length, 2);
    const comparisonSystemPrompt = requestBodies[0].messages[0].content as string;
    assert.match(comparisonSystemPrompt, /absolute delta is at most 8 points/);
    assert.match(comparisonSystemPrompt, /default to no penalty/);
    const comparisonParts = requestBodies[0].messages[1].content as Array<{ type: string; text?: string }>;
    assert.equal(comparisonParts.filter((part) => part.type === "image_url").length, 2);
    assert.match(comparisonParts.map((part) => part.text ?? "").join("\n"), /Image 1: pre-rollout reference/);
    assert.match(comparisonParts.map((part) => part.text ?? "").join("\n"), /current-high-before\.png/);
    const globalParts = requestBodies[1].messages[1].content as Array<{ type: string }>;
    assert.equal(globalParts.some((part) => part.type === "image_url"), false);

    assert.equal(evaluation.unpenalizedScore, 30);
    assert.equal(evaluation.preRolloutVisualPenalty, 80);
    assert.equal(evaluation.score, -50);
    assert.equal(evaluation.preRolloutComparison?.currentTurnIndex, 1);
    assert.equal(evaluation.preRolloutComparison?.currentScreenshotPath, currentHighPath);
    const stored = JSON.parse(await readFile(path.join(runDir, "global-evaluation.json"), "utf8"));
    assert.equal(stored.score, -50);
  } finally {
    globalThis.fetch = savedFetch;
    await rm(runDir, { recursive: true, force: true });
  }
});

test("missing pre-rollout screenshot skips Stage 1 with zero penalty and still runs Stage 2", async () => {
  const runDir = await mkdtemp(path.join(tmpdir(), "agens-global-missing-pre-rollout-"));
  const turn: UserSimulationTurn = {
    turnIndex: 0,
    beforeScreenshotPath: "turn-000-before.png",
    decisionPath: "turn-000-decision.json",
    decision: {
      summary: "Inspect the game.",
      action: { type: "click", selector: "#start" },
      targetState: "Gameplay.",
    },
    action: { type: "click", selector: "#start" },
    afterScreenshotPath: "turn-000-after.png",
    observationScreenshotPaths: [],
    observations: [],
    reachedGoal: false,
    totalWaitMs: 50,
    simulatedAdvanceMs: 50,
    terminated: false,
  };

  const savedFetch = globalThis.fetch;
  let requestCount = 0;
  try {
    globalThis.fetch = (async () => {
      requestCount += 1;
      return new Response(JSON.stringify({
        model: "global-response-model",
        choices: [{ message: { content: JSON.stringify({ score: 64, weightedScore: 64 }) } }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    const evaluation = await evaluateGlobalSimulation({
      judger: {
        enabled: true,
        judge: {
          model: "global-model",
          provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
        },
      },
      url: "file:///game/index.html",
      runDir,
      turns: [turn],
      turnLimit: 1,
      taskPrompt: "Build the game.",
      terminated: false,
      totalSimulatedAdvanceMs: 50,
      skipPreRolloutVisualComparison: true,
    });

    assert.equal(requestCount, 1);
    assert.equal(evaluation.unpenalizedScore, 64);
    assert.equal(evaluation.preRolloutVisualPenalty, 0);
    assert.equal(evaluation.preRolloutVisualComparisonSkipped, true);
    assert.equal(evaluation.preRolloutComparison, undefined);
    assert.equal(evaluation.score, 64);
    const stored = JSON.parse(await readFile(path.join(runDir, "global-evaluation.json"), "utf8"));
    assert.equal(stored.preRolloutVisualPenalty, 0);
    assert.equal(stored.preRolloutVisualComparisonSkipped, true);
    assert.equal(stored.score, 64);
  } finally {
    globalThis.fetch = savedFetch;
    await rm(runDir, { recursive: true, force: true });
  }
});
