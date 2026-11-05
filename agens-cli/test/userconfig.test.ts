import assert from "node:assert/strict";
import test from "node:test";
import { applyEnv, redact, resolveUserSimulation, withDefaults, type UserConfig } from "../src/userconfig.js";

const simulationEnvNames = [
  "AGENS_SIM_JUDGE_MODEL",
  "AGENS_SIM_JUDGE_BASE_URL",
  "AGENS_SIM_JUDGE_API_KEY",
  "AGENS_SIM_JUDGE_PROVIDER_TYPE",
  "AGENS_SIM_JUDGE_WIRE_API",
  "AGENS_SIM_JUDGE_BASE_URL_ADDITION",
  "AGENS_SIM_JUDGE_API_KEY_ADDITION",
  "AGENS_SIM_JUDGE_ROUTING_RATIO",
  "AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL",
  "AGENS_SIM_SCREENSHOT_JUDGE_ENABLED",
  "AGENS_SIM_SCREENSHOT_JUDGE_MODEL",
  "AGENS_SIM_SCREENSHOT_JUDGE_BASE_URL",
  "AGENS_SIM_SCREENSHOT_JUDGE_API_KEY",
  "AGENS_SIM_SCREENSHOT_JUDGE_PROVIDER_TYPE",
] as const;

function withCleanSimulationEnv(run: () => void): void {
  const saved = Object.fromEntries(simulationEnvNames.map((name) => [name, process.env[name]]));
  try {
    for (const name of simulationEnvNames) delete process.env[name];
    run();
  } finally {
    for (const name of simulationEnvNames) {
      const value = saved[name];
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }
}

test("resolves strict turn and decision/global model roles on one provider", () => {
  const resolved = resolveUserSimulation(withDefaults({
    judge: {
      model: "turn-model",
      provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
    },
    userSimulation: { extraGlobalJudger: { enabled: true, model: "decision-model" } },
  }));

  assert.equal(resolved.turnJudge.model, "turn-model");
  assert.equal(resolved.decisionJudge.model, "decision-model");
  assert.deepEqual(resolved.turnJudge.provider, resolved.decisionJudge.provider);
  assert.deepEqual(resolved.extraGlobalJudger?.judge, resolved.decisionJudge);
});

test("decision/global model falls back to the turn judge model", () => {
  const resolved = resolveUserSimulation(withDefaults({
    judge: { model: "single-model", provider: { baseUrl: "http://judge.test/v1" } },
  }));
  assert.equal(resolved.turnJudge.model, "single-model");
  assert.equal(resolved.decisionJudge.model, "single-model");
});

test("environment maps SIM_JUDGE to turn evaluation and EXTRA_GLOBAL to decision/global", () => {
  withCleanSimulationEnv(() => {
    process.env.AGENS_SIM_JUDGE_MODEL = "turn-env";
    process.env.AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL = "decision-env";
    process.env.AGENS_SIM_JUDGE_BASE_URL = "http://env-judge.test/v1";
    const configured = applyEnv(withDefaults({}));
    const resolved = resolveUserSimulation(configured);
    assert.equal(resolved.turnJudge.model, "turn-env");
    assert.equal(resolved.decisionJudge.model, "decision-env");
    assert.equal(resolved.turnJudge.provider?.baseUrl, "http://env-judge.test/v1");
    assert.deepEqual(resolved.turnJudge.provider, resolved.decisionJudge.provider);
  });
});

test("environment adds weighted Chat Completions routing to decisions and turn evaluation", () => {
  withCleanSimulationEnv(() => {
    process.env.AGENS_SIM_JUDGE_BASE_URL = "http://primary.test/v1";
    process.env.AGENS_SIM_JUDGE_API_KEY = "primary-key";
    process.env.AGENS_SIM_JUDGE_WIRE_API = "responses";
    process.env.AGENS_SIM_JUDGE_BASE_URL_ADDITION = "http://additional.test/v1";
    process.env.AGENS_SIM_JUDGE_API_KEY_ADDITION = "additional-key";
    process.env.AGENS_SIM_JUDGE_ROUTING_RATIO = "2";

    const configured = applyEnv(withDefaults({}));
    const resolved = resolveUserSimulation(configured);

    assert.equal(resolved.turnJudge.routing?.ratio, 2);
    assert.deepEqual(resolved.turnJudge.routing, resolved.decisionJudge.routing);
    assert.deepEqual(resolved.turnJudge.routing?.additionalProvider, {
      baseUrl: "http://additional.test/v1",
      apiKey: "additional-key",
      type: "openai",
      wireApi: "completions",
    });
    assert.equal(resolved.extraGlobalJudger?.judge.provider?.baseUrl, "http://primary.test/v1");
    assert.equal("routing" in (resolved.extraGlobalJudger?.judge ?? {}), false);
    assert.equal(
      redact(configured).userSimulation?.judgeRouting?.additionalProvider.apiKey,
      "addi…-key",
    );
  });
});

test("additional judge routing defaults to a 1:1 split", () => {
  withCleanSimulationEnv(() => {
    process.env.AGENS_SIM_JUDGE_BASE_URL_ADDITION = "http://additional.test/v1";
    process.env.AGENS_SIM_JUDGE_API_KEY_ADDITION = "additional-key";
    const resolved = resolveUserSimulation(applyEnv(withDefaults({})));
    assert.equal(resolved.turnJudge.routing?.ratio, 1);
  });
  withCleanSimulationEnv(() => {
    process.env.AGENS_SIM_JUDGE_BASE_URL_ADDITION = "http://additional.test/v1";
    process.env.AGENS_SIM_JUDGE_API_KEY_ADDITION = "additional-key";
    process.env.AGENS_SIM_JUDGE_ROUTING_RATIO = "0";
    const resolved = resolveUserSimulation(applyEnv(withDefaults({})));
    assert.equal(resolved.turnJudge.routing?.ratio, 0);
  });
});

test("additional judge routing rejects a missing key or invalid ratio", () => {
  withCleanSimulationEnv(() => {
    process.env.AGENS_SIM_JUDGE_BASE_URL_ADDITION = "http://additional.test/v1";
    assert.throws(
      () => applyEnv(withDefaults({})),
      /AGENS_SIM_JUDGE_API_KEY_ADDITION is required/,
    );
  });
  for (const ratio of ["-1", "not-a-number", "Infinity"]) {
    withCleanSimulationEnv(() => {
      process.env.AGENS_SIM_JUDGE_BASE_URL_ADDITION = "http://additional.test/v1";
      process.env.AGENS_SIM_JUDGE_API_KEY_ADDITION = "additional-key";
      process.env.AGENS_SIM_JUDGE_ROUTING_RATIO = ratio;
      assert.throws(
        () => applyEnv(withDefaults({})),
        /AGENS_SIM_JUDGE_ROUTING_RATIO must be a finite non-negative number/,
      );
    });
  }
});

test("removed independent model configuration fails closed", () => {
  assert.throws(
    () => withDefaults({ userSimulation: { judge: { model: "old" } } } as UserConfig),
    /userSimulation\.judge was removed/,
  );
  assert.throws(
    () => withDefaults({
      userSimulation: { screenshotJudger: { enabled: true } },
    } as UserConfig),
    /userSimulation\.screenshotJudger was removed/,
  );
  withCleanSimulationEnv(() => {
    process.env.AGENS_SIM_SCREENSHOT_JUDGE_MODEL = "old";
    assert.throws(() => applyEnv(withDefaults({})), /AGENS_SIM_SCREENSHOT_JUDGE_MODEL was removed/);
  });
  withCleanSimulationEnv(() => {
    process.env.AGENS_SIM_SCREENSHOT_JUDGE_ENABLED = "true";
    assert.throws(() => applyEnv(withDefaults({})), /AGENS_SIM_SCREENSHOT_JUDGE_ENABLED was removed/);
  });
});
