import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { evaluateTurnJudges } from "../src/simulation/visualJudges.js";
import type { UserSimulationTurn } from "../src/simulation/types.js";

const onePixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "base64",
);

test("combined turn judge sends exactly before/after images and writes compatible outputs", async () => {
  const runDir = await mkdtemp(path.join(tmpdir(), "agens-combined-judge-"));
  const beforePath = path.join(runDir, "turn-000-before.png");
  const afterPath = path.join(runDir, "turn-000-after.png");
  await Promise.all([
    writeFile(beforePath, onePixelPng),
    writeFile(afterPath, onePixelPng),
  ]);

  const turn: UserSimulationTurn = {
    turnIndex: 0,
    beforeScreenshotPath: beforePath,
    decisionPath: path.join(runDir, "turn-000-decision.json"),
    decision: {
      summary: "Click the visible Start button.",
      action: { type: "click", selector: "#start" },
      targetState: "Gameplay is visible.",
      expectedTransition: "The start screen changes to gameplay.",
    },
    action: { type: "click", selector: "#start" },
    actionExecution: { actionType: "click", success: true },
    actionOverlayPath: path.join(runDir, "must-not-be-read-overlay.png"),
    afterScreenshotPath: afterPath,
    observationScreenshotPaths: [path.join(runDir, "must-not-be-read-observation.png")],
    observations: [],
    reachedGoal: true,
    totalWaitMs: 300,
    simulatedAdvanceMs: 300,
    terminated: false,
  };

  const savedFetch = globalThis.fetch;
  let requestBody: any;
  try {
    globalThis.fetch = (async (_input, init) => {
      requestBody = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        model: "response-model",
        choices: [{
          message: {
            content: JSON.stringify({
              transitionEvaluation: {
                overallReward: 81,
                visualStabilityReward: 90,
                executionReward: 80,
                appFeedbackReward: 75,
                taskProgressReward: 70,
                summary: "Stable transition.",
                strengths: ["stable"],
                weaknesses: [],
                failureMode: "none",
                experience: "The Start button works.",
                state: "A start screen is visible.",
              },
              screenshotEvaluation: {
                score: 82,
                visualStabilityReward: 90,
                readabilityReward: 80,
                stateVisibilityReward: 72,
                screenshotAestheticReward: 77,
                summary: "Readable screenshot.",
                strengths: ["readable"],
                weaknesses: [],
              },
            }),
          },
        }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    await evaluateTurnJudges({
      turn,
      judge: {
        model: "turn-judge",
        provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
      },
      taskPrompt: "Start the game.",
      runDir,
      traceEvents: [],
    });
  } finally {
    globalThis.fetch = savedFetch;
  }

  const messages = requestBody.messages as Array<{ role: string; content: any }>;
  const systemPrompt = String(messages[0].content);
  const userParts = messages[1].content as Array<{ type: string; text?: string }>;
  assert.match(systemPrompt, /TransitionJudger/);
  assert.match(systemPrompt, /ScreenshotJudger/);
  assert.equal(userParts.filter((part) => part.type === "image_url").length, 2);
  assert.equal(userParts.some((part) => part.text?.includes("Artifact source code")), false);
  assert.equal(userParts.some((part) => part.text?.includes("must-not-be-read")), false);
  assert.equal(turn.transitionEvaluation?.score, 81);
  assert.equal(turn.screenshotEvaluation?.score, 82);

  const transitionFile = JSON.parse(await readFile(
    path.join(runDir, "turn-000-transition-evaluation.json"),
    "utf8",
  ));
  const screenshotFile = JSON.parse(await readFile(
    path.join(runDir, "turn-000-screenshot-evaluation.json"),
    "utf8",
  ));
  assert.equal(transitionFile.judgeType, "transition");
  assert.equal(screenshotFile.judgeType, "screenshot");
  await rm(runDir, { recursive: true, force: true });
});
