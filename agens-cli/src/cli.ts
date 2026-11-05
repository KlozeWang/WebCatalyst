#!/usr/bin/env node
/**
 * agens — agent harness CLI built on the GitHub Copilot SDK.
 *
 * Precedence for every option:  CLI flag  >  env var  >  ~/.agens/config.json
 */
import path from "node:path";
import { mkdirSync, readFileSync, statSync } from "node:fs";
import { Command, InvalidArgumentError } from "commander";
import chalk from "chalk";
import { startRepl } from "./repl.js";
import { runOneShot } from "./oneshot.js";
import { listSkills } from "./skills.js";
import { runUpdate } from "./update.js";
import { runUserSimulation, runtimeFromResolvedConfig } from "./simulation/engine.js";
import { runRewardSimulation, type RewardSimulationRequest } from "./simulation/reward.js";
import { runTraceAnalysis, TraceAnalysisRunError } from "./simulation/trace.js";
import {
  CONFIG_PATH,
  loadConfigFile,
  saveConfigFile,
  applyEnv,
  withDefaults,
  resolveJudge,
  resolveUserSimulation,
  setByPath,
  getByPath,
  redact,
  type ProviderFile,
  type UserConfig,
} from "./userconfig.js";
import type { HarnessConfig } from "./config.js";

const program = new Command();

program
  .name("agens")
  .description("Agent harness built on the Copilot SDK, with built-in Playwright MCP.")
  .version("0.5.0");

// --- main run command (default) ---
program
  .argument("[prompt...]", "Optional one-shot prompt. If omitted, starts an interactive REPL.")
  .option("-p, --prompt <text>", "One-shot prompt (alternative to positional argument; useful for scripts)")
  .option("-m, --model <id>", "Model to use")
  .option("-s, --skill <name>", "Load a built-in skill (repeatable)", collect, [])
  .option("--base-url <url>", "Custom provider endpoint (BYOK)")
  .option("--api-key <key>", "API key for the custom provider")
  .option("--provider-type <t>", "openai | azure | anthropic")
  .option("--judge-model <id>", "Model for critique and per-turn simulation evaluation")
  .option("--judge-base-url <url>", "Endpoint for critique and simulation judges")
  .option("--judge-api-key <key>", "API key for critique and simulation judges")
  .option("--sim-judge-base-url <url>", "Shared endpoint for simulation decisions and judges")
  .option("--sim-judge-api-key <key>", "Shared API key for simulation decisions and judges")
  .option("--extra-global-judge", "Enable run-level global simulation evaluation")
  .option("--no-extra-global-judge", "Disable run-level global simulation evaluation")
  .option("--extra-global-judge-model <id>", "Model to use for simulation decisions and run-level global evaluation")
  .option("--sim-persona <file>", "Markdown persona file for user simulation")
  .option("--sim-metrics <file>", "Markdown psychometric metrics/guidelines file for user simulation")
  .option("--sim-output-dir <dir>", "Output directory for user simulation artifacts")
  .option("--sim-step-ms <ms>", "Default timing budget used when max advance is omitted", parseNonNegativeInt)
  .option("--sim-frame-ms <ms>", "Clock chunk size used while advancing held actions and deterministic observations", parsePositiveInt)
  .option("--sim-turn-limit <n>", "Max user simulation turns before stopping", parsePositiveInt)
  .option("--sim-save-full-record, --sim-save_full_record", "Also save a full-run MP4 recording for user simulation")
  .option("--no-sim-record-video", "Disable Playwright video recording for user simulation")
  .option("--no-playwright", "Disable the built-in Playwright MCP server")
  .option("--browser <name>", "Browser for Playwright: auto | msedge | chrome | chromium | firefox (default auto: prefers installed Edge/Chrome)")
  .option("--headed", "Show the Playwright browser window (default: headless / invisible)")
  .option("--cwd <dir>", "Working directory for the session", process.cwd())
  .option("-t, --trajectory <file>", "Write the full trajectory (prompt, response, events, images as base64) to this JSON file. One-shot mode only.")
  .option("--reasoning", "Show the model's chain-of-thought (assistant.reasoning) inline")
  .action(async (promptParts: string[], opts) => {
    const actionOpts = materializeOpts(opts);
    const harness = resolveHarnessConfig(actionOpts);
    // -p / --prompt wins over the positional argument; positional is the
    // fallback so existing `agens "..."` invocations keep working.
    const prompt = (actionOpts.prompt ?? promptParts.join(" ")).trim();
    try {
      if (prompt) await runOneShot(prompt, harness, { trajectoryPath: actionOpts.trajectory, showReasoning: !!actionOpts.reasoning });
      else await startRepl(harness);
    } catch (err) {
      console.error(chalk.red("agens error:"), err instanceof Error ? err.message : err);
      process.exit(1);
    }
  });

// --- direct user simulation ---
program
  .command("simulate <url>")
  .description("Run a multi-step user simulation against a rendered page URL")
  .option("--task <text>", "Task or scenario for the simulated user")
  .option("--artifact <file>", "Optional artifact source file to include in simulation decisions")
  .option("--include-code", "Include local artifact source code in simulation decision context")
  .option("--persona <file>", "Markdown persona file (overrides config)")
  .option("--metrics <file>", "Markdown psychometric metrics/guidelines file (overrides config)")
  .option("--out <dir>", "Output directory for screenshots, optional MP4 recording, decision, and trace files")
  .option("--step-ms <ms>", "Default timing budget used when max advance is omitted", parseNonNegativeInt)
  .option("--frame-ms <ms>", "Clock chunk size used while advancing held actions and deterministic observations", parsePositiveInt)
  .option("--turn-limit <n>", "Max simulation turns before stopping", parsePositiveInt)
  .option("--max-advance-ms <ms>", "Max simulated milliseconds for the default after-action delay", parseNonNegativeInt)
  .option("--save-full-record, --save_full_record", "Also save a full-run MP4 recording")
  .option("--no-record-video", "Disable Playwright video recording")
  .option("--json", "Print full JSON result")
  .option("-m, --model <id>", "Fallback model to use")
  .option("--base-url <url>", "Fallback provider endpoint")
  .option("--api-key <key>", "Fallback provider API key")
  .option("--provider-type <t>", "openai | azure | anthropic")
  .option("--judge-model <id>", "Model for critique and per-turn simulation evaluation")
  .option("--judge-base-url <url>", "Endpoint for critique and simulation judges")
  .option("--judge-api-key <key>", "API key for critique and simulation judges")
  .option("--sim-judge-base-url <url>", "Shared endpoint for simulation decisions and judges")
  .option("--sim-judge-api-key <key>", "Shared API key for simulation decisions and judges")
  .option("--extra-global-judge", "Enable run-level global simulation evaluation")
  .option("--no-extra-global-judge", "Disable run-level global simulation evaluation")
  .option("--extra-global-judge-model <id>", "Model to use for simulation decisions and run-level global evaluation")
  .option("--browser <name>", "Browser: auto | msedge | chrome | chromium | firefox")
  .option("--headed", "Show the Playwright browser window")
  .option("--cwd <dir>", "Working directory for resolving relative paths and output", process.cwd())
  .action(async (url: string, opts) => {
    const actionOpts = materializeOpts(opts);
    const harness = resolveHarnessConfig({
      ...actionOpts,
      simPersona: actionOpts.persona,
      simMetrics: actionOpts.metrics,
      simOutputDir: actionOpts.out,
      simStepMs: actionOpts.stepMs,
      simFrameMs: actionOpts.frameMs,
      simTurnLimit: actionOpts.turnLimit,
      simSaveFullRecord: actionOpts.saveFullRecord ?? actionOpts.save_full_record,
      simRecordVideo: actionOpts.recordVideo,
    });
    try {
      const result = await runUserSimulation(
        {
          url,
          taskPrompt: actionOpts.task,
          artifactPath: actionOpts.artifact,
          includeCode: !!actionOpts.includeCode,
          personaPath: actionOpts.persona,
          psychometricPath: actionOpts.metrics,
          outputDir: actionOpts.out,
          stepMs: actionOpts.stepMs,
          frameMs: actionOpts.frameMs,
          turnLimit: actionOpts.turnLimit,
          saveFullRecord: actionOpts.saveFullRecord ?? actionOpts.save_full_record,
          recordVideo: actionOpts.recordVideo,
          maxAdvanceMs: actionOpts.maxAdvanceMs,
        },
        runtimeFromResolvedConfig(
          harness.cwd,
          harness.userSimulation,
          harness.browser,
          harness.headed,
        ),
      );
      if (actionOpts.json) {
        console.log(JSON.stringify(result, null, 2));
        return;
      }
      console.log(chalk.bold.green("Simulation complete"));
      console.log(`  turns: ${chalk.cyan(String(result.turnCount))}/${result.turnLimit}`);
      console.log(`  final action: ${chalk.cyan(result.action.type)}`);
      if (result.terminated) console.log(`  terminated: ${result.terminationReason ?? "yes"}`);
      console.log(`  runDir: ${result.runDir}`);
      console.log(`  final:  ${result.finalScreenshotPath}`);
      if (result.fullRecordPath) console.log(`  full mp4: ${result.fullRecordPath}`);
      if (result.fullRecordError) console.log(chalk.yellow(`  full mp4: ${result.fullRecordError}`));
      if (result.visualEvaluationSummary?.screenshot) {
        console.log(`  screenshot eval: ${result.visualEvaluationSummary.screenshot.averageScore}/100 avg over ${result.visualEvaluationSummary.screenshot.count}`);
      }
      if (result.visualEvaluationSummary?.transition) {
        console.log(`  transition eval: ${result.visualEvaluationSummary.transition.averageScore}/100 avg over ${result.visualEvaluationSummary.transition.count}`);
      }
      if (result.globalEvaluation) {
        const cap = result.globalEvaluation.usabilityCap === undefined ? "" : ` cap=${result.globalEvaluation.usabilityCap}`;
        console.log(`  global eval: ${result.globalEvaluation.score}/100${cap}`);
      }
      console.log(`  result: ${result.resultPath}`);
    } catch (err) {
      console.error(chalk.red("simulate error:"), err instanceof Error ? err.message : err);
      process.exit(1);
    }
  });

// --- offline transition/code tracing ---
program
  .command("trace <run-or-trace-dir>")
  .description("Replay a retained reward trace and capture code coverage plus live DOM for qualifying turns")
  .requiredOption("--threshold <score>", "Minimum transition score from 0 to 100", parseScoreThreshold)
  .option("--frame-ms <ms>", "Clock chunk size for legacy trace replay (default 50)", parsePositiveInt)
  .option("--browser <name>", "Chromium-family browser: auto | msedge | chrome | chromium")
  .option("--headed", "Show the browser window")
  .option("--json", "Print the full trace-result JSON")
  .option("--cwd <dir>", "Working directory for resolving the input path", process.cwd())
  .action(async (inputPath: string, opts) => {
    const actionOpts = materializeOpts(opts);
    const harness = resolveHarnessConfig(actionOpts);
    try {
      const result = await runTraceAnalysis(
        {
          inputPath,
          threshold: actionOpts.threshold,
          frameMs: actionOpts.frameMs,
        },
        {
          cwd: harness.cwd,
          browser: harness.browser,
          headed: harness.headed,
        },
      );
      if (actionOpts.json) {
        console.log(JSON.stringify(result, null, 2));
        return;
      }
      console.log(chalk.bold.green("Trace analysis complete"));
      console.log(`  selected: ${chalk.cyan(String(result.selectedTurnCount))}/${result.totalTurnCount}`);
      console.log(`  processed: ${chalk.cyan(String(result.processedTurnCount))}/${result.totalTurnCount}`);
      if (result.replay.firstDriftTurn !== undefined) {
        console.log(chalk.yellow(`  replay drift: first detected at turn ${result.replay.firstDriftTurn}`));
      }
      console.log(`  output: ${result.outputDir}`);
    } catch (err) {
      if (actionOpts.json && err instanceof TraceAnalysisRunError) {
        console.log(JSON.stringify(err.result, null, 2));
      } else {
        console.error(chalk.red("trace error:"), err instanceof Error ? err.message : err);
        if (err instanceof TraceAnalysisRunError) console.error(chalk.gray(`partial result: ${err.result.outputDir}`));
      }
      process.exitCode = 1;
    }
  });

// --- verl reward simulation ---
program
  .command("simulate-reward")
  .description("Run a verl-compatible reward simulation from rollout HTML")
  .option("--request <file>", "JSON request file containing html/htmlFile and rollout metadata")
  .option("--request-stdin", "Read the JSON request from stdin")
  .option("--html-file <file>", "HTML artifact source file")
  .option("--html-stdin", "Read HTML artifact source from stdin")
  .option("--task <text>", "Task or scenario for the simulated user")
  .option("--out-root <dir>", "Root directory for retained reward artifacts")
  .option("--global-step <value>", "Training global step metadata")
  .option("--prompt-id <value>", "Prompt/group id metadata")
  .option("--sample-id <value>", "Rollout sample id metadata")
  .option("--rollout-id <value>", "Rollout id metadata")
  .option("--turn-limit <n>", "Max simulation turns before stopping", parsePositiveInt)
  .option("--step-ms <ms>", "Default timing budget used when max advance is omitted", parseNonNegativeInt)
  .option("--frame-ms <ms>", "Clock chunk size used while advancing held actions and deterministic observations", parsePositiveInt)
  .option("--max-advance-ms <ms>", "Max simulated milliseconds for the default after-action delay", parseNonNegativeInt)
  .option("--save-full-record, --save_full_record", "Also save a full-run MP4 recording")
  .option("--no-record-video", "Disable Playwright video recording")
  .option("--json", "Print the reward-result JSON")
  .option("-m, --model <id>", "Fallback model to use")
  .option("--base-url <url>", "Fallback provider endpoint")
  .option("--api-key <key>", "Fallback provider API key")
  .option("--provider-type <t>", "openai | azure | anthropic")
  .option("--judge-model <id>", "Model for critique and per-turn simulation evaluation")
  .option("--judge-base-url <url>", "Endpoint for critique and simulation judges")
  .option("--judge-api-key <key>", "API key for critique and simulation judges")
  .option("--sim-judge-base-url <url>", "Shared endpoint for simulation decisions and judges")
  .option("--sim-judge-api-key <key>", "Shared API key for simulation decisions and judges")
  .option("--extra-global-judge-model <id>", "Model to use for simulation decisions and run-level global evaluation")
  .option("--browser <name>", "Browser: auto | msedge | chrome | chromium | firefox")
  .option("--headed", "Show the Playwright browser window")
  .option("--cwd <dir>", "Working directory for resolving relative paths and output", process.cwd())
  .action(async (opts) => {
    const actionOpts = materializeOpts(opts);
    const harness = resolveHarnessConfig(actionOpts);
    try {
      const request = readRewardSimulationRequest(actionOpts, harness.cwd);
      const result = await runRewardSimulation(
        request,
        runtimeFromResolvedConfig(
          harness.cwd,
          harness.userSimulation,
          harness.browser,
          harness.headed,
        ),
      );
      if (actionOpts.json) {
        console.log(JSON.stringify(result, null, 2));
        return;
      }
      console.log(result.ok ? chalk.bold.green("Reward simulation complete") : chalk.bold.yellow("Reward simulation failed"));
      console.log(`  reward: ${chalk.cyan(String(result.score))}`);
      console.log(`  global: ${result.globalScore ?? "missing"}`);
      console.log(`  transition: ${result.transitionScore ?? "missing"}`);
      console.log(`  screenshot: ${result.screenshotScore ?? "missing"}`);
      console.log(`  result: ${result.paths.rewardResultPath}`);
      if (result.errors.length) {
        console.log(chalk.yellow(`  errors: ${result.errors.length}`));
      }
    } catch (err) {
      console.error(chalk.red("simulate-reward error:"), err instanceof Error ? err.message : err);
      process.exit(1);
    }
  });

// --- skills ---
program
  .command("skills")
  .description("List available built-in skills")
  .action(() => {
    const skills = listSkills();
    if (!skills.length) {
      console.log(chalk.yellow("No skills installed yet. Drop folders under skills/<name>/SKILL.md"));
      return;
    }
    console.log(chalk.bold("Built-in skills:"));
    for (const s of skills) console.log(`  ${chalk.cyan(s.name)}  ${chalk.gray("- " + s.path)}`);
  });

program
  .command("update")
  .description("Self-update agens to the latest published version")
  .action(runUpdate);

// --- config ---
const configCmd = program.command("config").description("Manage ~/.agens/config.json");

configCmd
  .command("init")
  .description("Write a template config file (won't overwrite existing)")
  .action(() => {
    const existing = loadConfigFile();
    if (Object.keys(existing).length) {
      console.log(chalk.yellow(`Config already exists at ${CONFIG_PATH} — leaving alone.`));
      return;
    }
    const template: UserConfig = {
      model: "gpt-5",
      provider: {
        type: "openai",
      },
      skills: [],
      playwright: true,
      userSimulation: {
        outputDir: ".agens-sim",
        stepMs: 300,
        frameMs: 50,
        recordVideo: true,
        saveFullRecord: false,
        turnLimit: 20,
      },
    };
    saveConfigFile(template);
    console.log(chalk.green(`Wrote template config to ${CONFIG_PATH}`));
    console.log(chalk.gray("Edit it, or use:  agens config set provider.baseUrl <url>"));
  });

configCmd
  .command("show")
  .description("Print the effective config (secrets redacted)")
  .action(() => {
    const merged = applyEnv(withDefaults(loadConfigFile()));
    console.log(JSON.stringify(redact(merged), null, 2));
    console.log(chalk.gray(`\n(source: built-in defaults + ${CONFIG_PATH} + env vars)`));
  });

configCmd
  .command("path")
  .description("Print the config file path")
  .action(() => console.log(CONFIG_PATH));

configCmd
  .command("set <key> <value>")
  .description("Set a config value, e.g. `config set provider.baseUrl http://...`")
  .action((key: string, value: string) => {
    const cfg = loadConfigFile() as Record<string, any>;
    // Coerce booleans/numbers for ergonomics
    let v: any = value;
    if (value === "true") v = true;
    else if (value === "false") v = false;
    else if (/^-?\d+(\.\d+)?$/.test(value)) v = Number(value);
    setByPath(cfg, key, v);
    saveConfigFile(cfg);
    console.log(chalk.green(`Set ${key} = ${typeof v === "string" ? v : JSON.stringify(v)}`));
  });

configCmd
  .command("get <key>")
  .description("Read a config value")
  .action((key: string) => {
    const cfg = loadConfigFile() as Record<string, any>;
    const v = getByPath(cfg, key);
    console.log(v === undefined ? "(unset)" : typeof v === "string" ? v : JSON.stringify(v));
  });

function readRewardSimulationRequest(opts: any, cwd: string): RewardSimulationRequest {
  if (opts.request && opts.requestStdin) {
    throw new Error("Use only one of --request or --request-stdin.");
  }
  if (opts.htmlStdin && opts.requestStdin) {
    throw new Error("Use only one stdin mode: --html-stdin or --request-stdin.");
  }

  const base = opts.request
    ? parseJsonObject(readFileSync(path.resolve(cwd, opts.request), "utf8"), opts.request)
    : opts.requestStdin
      ? parseJsonObject(readFileSync(0, "utf8"), "stdin")
      : {};
  const metadata = isPlainObject(base.metadata) ? base.metadata : {};
  const htmlFile = pickOpt<string>(opts, "htmlFile") ?? base.htmlFile;
  const html = opts.htmlStdin ? readFileSync(0, "utf8") : htmlFile ? undefined : base.html;

  return {
    ...base,
    html,
    htmlFile,
    taskPrompt: pickOpt<string>(opts, "task") ?? base.taskPrompt ?? base.task,
    outputRoot: pickOpt<string>(opts, "outRoot") ?? base.outputRoot ?? base.outRoot,
    globalStep: pickOpt<string>(opts, "globalStep") ?? base.globalStep ?? metadata.globalStep,
    promptId: pickOpt<string>(opts, "promptId") ?? base.promptId ?? metadata.promptId,
    sampleId: pickOpt<string>(opts, "sampleId") ?? base.sampleId ?? metadata.sampleId,
    rolloutId: pickOpt<string>(opts, "rolloutId") ?? base.rolloutId ?? metadata.rolloutId,
    turnLimit: pickOpt<number>(opts, "turnLimit") ?? base.turnLimit,
    stepMs: pickOpt<number>(opts, "stepMs") ?? base.stepMs,
    frameMs: pickOpt<number>(opts, "frameMs") ?? base.frameMs,
    maxAdvanceMs: pickOpt<number>(opts, "maxAdvanceMs") ?? base.maxAdvanceMs,
    saveFullRecord: pickOpt<boolean>(opts, "saveFullRecord", "save_full_record") ?? base.saveFullRecord,
    recordVideo: pickOpt<boolean>(opts, "recordVideo") ?? base.recordVideo,
  };
}

function parseJsonObject(text: string, label: string): Record<string, any> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`Failed to parse ${label} as JSON: ${(error as Error).message}`);
  }
  if (!isPlainObject(parsed)) {
    throw new Error(`${label} must contain a JSON object.`);
  }
  return parsed;
}

function isPlainObject(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function collect(value: string, prev: string[]): string[] {
  return prev.concat([value]);
}

function parseNonNegativeInt(value: string): number {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 0) {
    throw new InvalidArgumentError("expected a non-negative integer");
  }
  return n;
}

function parsePositiveInt(value: string): number {
  const n = Number(value);
  if (!Number.isInteger(n) || n <= 0) {
    throw new InvalidArgumentError("expected a positive integer");
  }
  return n;
}

function parseScoreThreshold(value: string): number {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0 || n > 100) {
    throw new InvalidArgumentError("expected a number between 0 and 100");
  }
  return n;
}

function materializeOpts(opts: any): any {
  if (!opts || typeof opts !== "object") return opts ?? {};
  if (typeof opts.opts === "function") {
    const globals = typeof opts.optsWithGlobals === "function" ? opts.optsWithGlobals() : {};
    return { ...globals, ...opts.opts() };
  }
  return opts;
}

function pickOpt<T = unknown>(opts: any, ...names: string[]): T | undefined {
  for (const name of names) {
    if (opts[name] !== undefined) return opts[name] as T;
  }
  return undefined;
}

function pickArgFlag(...flags: string[]): string | undefined {
  for (const flag of flags) {
    const i = process.argv.indexOf(flag);
    if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  }
  return undefined;
}

function resolveHarnessConfig(opts: any): HarnessConfig {
  opts = materializeOpts(opts);
  // Precedence: CLI flag > env var > config file > built-in defaults.
  const merged = applyEnv(withDefaults(loadConfigFile()));
  const model = pickOpt<string>(opts, "model") ?? merged.model ?? "gpt-5";
  const provider = { ...(merged.provider ?? {}) };
  const baseUrl = pickOpt<string>(opts, "baseUrl", "baseURL", "baseurl");
  const apiKey = pickOpt<string>(opts, "apiKey", "apikey");
  const providerType = pickOpt<ProviderFile["type"]>(opts, "providerType", "providertype");
  if (baseUrl) provider.baseUrl = baseUrl;
  if (apiKey) provider.apiKey = apiKey;
  if (providerType) provider.type = providerType;

  const skillsFromCli = (opts.skill as string[]) ?? [];
  const skills = skillsFromCli.length ? skillsFromCli : merged.skills ?? [];

  const enablePlaywright =
    opts.playwright === false ? false : merged.playwright !== false; // default true

  // Resolve judge with full precedence: CLI > env > config > fallback to main.
  const baseJudge = resolveJudge(merged);
  const judgeProvider = { ...(baseJudge.provider ?? {}) };
  const judgeBaseUrl = pickOpt<string>(opts, "judgeBaseUrl", "judgeBaseURL", "judgeBaseurl");
  const judgeApiKey = pickOpt<string>(opts, "judgeApiKey", "judgeApikey");
  const judgeModel = pickOpt<string>(opts, "judgeModel", "judgemodel");
  if (judgeBaseUrl) judgeProvider.baseUrl = judgeBaseUrl;
  if (judgeApiKey) judgeProvider.apiKey = judgeApiKey;
  const judge = {
    model: judgeModel ?? baseJudge.model,
    provider: judgeProvider.baseUrl ? judgeProvider : undefined,
  };

  const baseSimulation = resolveUserSimulation({ ...merged, judge });
  const simulationProvider = { ...(baseSimulation.turnJudge.provider ?? {}) };
  const simJudgeBaseUrl =
    pickOpt<string>(opts, "simJudgeBaseUrl", "simJudgeBaseURL", "simJudgeBaseurl") ??
    pickArgFlag("--sim-judge-base-url");
  const simJudgeApiKey =
    pickOpt<string>(opts, "simJudgeApiKey", "simJudgeApikey") ??
    pickArgFlag("--sim-judge-api-key");
  if (simJudgeBaseUrl) simulationProvider.baseUrl = simJudgeBaseUrl;
  if (simJudgeApiKey) simulationProvider.apiKey = simJudgeApiKey;
  const sharedSimulationProvider = simulationProvider.baseUrl ? simulationProvider : undefined;
  const turnJudge = {
    model: baseSimulation.turnJudge.model,
    provider: sharedSimulationProvider,
    ...(baseSimulation.turnJudge.routing ? { routing: baseSimulation.turnJudge.routing } : {}),
  };
  const simRecordVideo =
    opts.simRecordVideo === false || opts.recordVideo === false
      ? false
      : baseSimulation.recordVideo !== false;
  const extraGlobalJudgeEnabled = pickExplicitBooleanFlag("--extra-global-judge");
  const extraGlobalJudgeModel =
    pickOpt<string>(opts, "extraGlobalJudgeModel", "extraglobaljudgemodel") ??
    pickArgFlag("--extra-global-judge-model");
  const decisionJudge = {
    model: extraGlobalJudgeModel ?? baseSimulation.decisionJudge.model ?? turnJudge.model,
    provider: sharedSimulationProvider,
    ...(baseSimulation.decisionJudge.routing ? { routing: baseSimulation.decisionJudge.routing } : {}),
  };
  const userSimulation = {
    ...baseSimulation,
    turnJudge,
    decisionJudge,
    personaPath: pickOpt<string>(opts, "simPersona") ?? baseSimulation.personaPath,
    psychometricPath: pickOpt<string>(opts, "simMetrics") ?? baseSimulation.psychometricPath,
    outputDir: pickOpt<string>(opts, "simOutputDir") ?? baseSimulation.outputDir,
    stepMs: pickOpt<number>(opts, "simStepMs") ?? baseSimulation.stepMs,
    frameMs: pickOpt<number>(opts, "simFrameMs") ?? baseSimulation.frameMs,
    turnLimit: pickOpt<number>(opts, "simTurnLimit") ?? baseSimulation.turnLimit,
    saveFullRecord: pickOpt<boolean>(opts, "simSaveFullRecord", "simSave_full_record") ?? baseSimulation.saveFullRecord,
    recordVideo: simRecordVideo,
    defaultSteps: baseSimulation.defaultSteps,
    extraGlobalJudger: {
      enabled: extraGlobalJudgeEnabled ?? baseSimulation.extraGlobalJudger?.enabled ?? false,
      judge: {
        model: decisionJudge.model,
        provider: decisionJudge.provider,
      },
    },
  };

  // Resolve & validate the working directory up front. The SDK spawns a
  // child process with this as its cwd; if it doesn't exist, the spawn
  // fails and the user sees a cryptic "Cannot call write after a stream
  // was destroyed" much later. Create it on demand so `--cwd new-folder`
  // just works, and reject anything that exists but isn't a directory.
  const cwd = path.resolve(process.cwd(), opts.cwd as string);
  try {
    const st = statSync(cwd);
    if (!st.isDirectory()) {
      throw new Error(`--cwd path exists but is not a directory: ${cwd}`);
    }
  } catch (e: any) {
    if (e?.code === "ENOENT") {
      mkdirSync(cwd, { recursive: true });
      console.error(chalk.gray(`· created working directory ${cwd}`));
    } else {
      throw e;
    }
  }

  return {
    model,
    skills,
    enablePlaywright,
    cwd,
    provider: provider.baseUrl ? provider : undefined,
    judge,
    browser: opts.browser as string | undefined,
    headed: !!opts.headed,
    userSimulation,
  };
}

function pickExplicitBooleanFlag(positiveFlag: string): boolean | undefined {
  const negativeFlag = `--no-${positiveFlag.slice(2)}`;
  if (process.argv.includes(positiveFlag)) return true;
  if (process.argv.includes(negativeFlag)) return false;
  return undefined;
}

program.parseAsync(process.argv);
