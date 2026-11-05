import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { decideAction } from "../src/simulation/judge.js";
import type { UserSimulationTurn } from "../src/simulation/types.js";
import { summarizeTurns } from "../src/simulation/utils.js";

const onePixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "base64",
);

test("decision keeps source code but excludes prior transition judge results", async () => {
  const runDir = await mkdtemp(path.join(tmpdir(), "agens-decision-context-"));
  const screenshotPath = path.join(runDir, "before.png");
  await writeFile(screenshotPath, onePixelPng);
  const priorTurn: UserSimulationTurn = {
    turnIndex: 0,
    beforeScreenshotPath: "turn-000-before.png",
    decisionPath: "turn-000-decision.json",
    decision: {
      summary: "Tried Start.",
      action: { type: "click", selector: "#start" },
      targetState: "Started.",
    },
    action: { type: "click", selector: "#start" },
    afterScreenshotPath: "turn-000-after.png",
    observationScreenshotPaths: [],
    observations: [],
    reachedGoal: false,
    totalWaitMs: 300,
    simulatedAdvanceMs: 300,
    terminated: false,
    transitionEvaluation: {
      judgeType: "transition",
      turnIndex: 0,
      artifactPath: "turn-000-after.png",
      evaluationPath: "turn-000-transition-evaluation.json",
      score: 17,
      summary: "TRANSITION_JUDGE_SENTINEL",
      strengths: [],
      weaknesses: ["TRANSITION_WEAKNESS_SENTINEL"],
      experience: "TRANSITION_EXPERIENCE_SENTINEL",
      state: "TRANSITION_STATE_SENTINEL",
      failureMode: "action_ineffective",
    },
  };
  const episodeHistory = summarizeTurns([priorTurn]);
  assert.equal("transitionExperience" in episodeHistory[0], false);
  assert.equal("beforeState" in episodeHistory[0], false);
  assert.equal("transitionScore" in episodeHistory[0], false);

  const savedFetch = globalThis.fetch;
  let requestBody: any;
  try {
    globalThis.fetch = (async (_input, init) => {
      requestBody = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        model: "decision-response-model",
        choices: [{
          message: {
            content: JSON.stringify({
              summary: "Try a keyboard action.",
              action: { type: "press", key: "Enter" },
              targetState: "The page reacts.",
            }),
          },
        }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    await decideAction({
      provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
      model: "decision-model",
      taskPrompt: "Start the game.",
      screenshotPath,
      artifact: {
        path: path.join(runDir, "index.html"),
        text: "<script>const DECISION_CODE_SENTINEL = true;</script>",
        truncated: false,
      },
      url: "file:///game/index.html",
      actionHints: [],
      episodeHistory,
      turnIndex: 1,
      turnLimit: 3,
      frameMs: 50,
      maxAdvanceMs: 5_000,
    });
  } finally {
    globalThis.fetch = savedFetch;
  }

  const messages = requestBody.messages as Array<{ content: any }>;
  const userText = (messages[1].content as Array<{ type: string; text?: string }>)
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("\n");
  assert.match(userText, /DECISION_CODE_SENTINEL/);
  assert.doesNotMatch(userText, /Simulation memory/);
  assert.doesNotMatch(userText, /TRANSITION_(JUDGE|WEAKNESS|EXPERIENCE|STATE)_SENTINEL/);
  await rm(runDir, { recursive: true, force: true });
});
