/**
 * Three-layer user configuration:
 *
 *   1. Config file at ~/.agens/config.json  (lowest priority, persistent)
 *   2. Environment variables                (middle)
 *   3. CLI flags                            (highest, wins everything)
 *
 * Anything provider-related (BYOK endpoint, key, model) flows through here so
 * users have one obvious place to set "which LLM does agens talk to".
 *
 * File schema (all fields optional):
 *   {
 *     "model": "gpt-5.2",
 *     "provider": {
 *       "type": "openai",          // "openai" | "azure" | "anthropic"
 *       "baseUrl": "http://host:8080/v1",
 *       "apiKey": "sk-...",
 *       "bearerToken": "...",      // optional, takes precedence over apiKey
 *       "wireApi": "completions"   // "completions" | "responses"
 *     },
 *     "skills": ["web-research"],  // skills to auto-load every run
 *     "playwright": true
 *   }
 *
 * Env var overrides:
 *   AGENS_MODEL, AGENS_BASE_URL, AGENS_API_KEY, AGENS_BEARER_TOKEN,
 *   AGENS_PROVIDER_TYPE, AGENS_WIRE_API, AGENS_SKILLS_DIR
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync, chmodSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export const CONFIG_DIR = join(homedir(), ".agens");
export const CONFIG_PATH = join(CONFIG_DIR, "config.json");

export interface ProviderFile {
  type?: "openai" | "azure" | "anthropic";
  baseUrl?: string;
  apiKey?: string;
  bearerToken?: string;
  wireApi?: "completions" | "responses";
}

export interface SimulationJudgeRoutingConfig {
  /** Additional OpenAI-compatible endpoint used by simulation decisions and turn evaluation. */
  additionalProvider: ProviderFile;
  /** Additional-to-primary routing weight. A value of 1 means a 1:1 split. */
  ratio: number;
}

export interface ResolvedSimulationJudge {
  model: string;
  provider?: ProviderFile;
  routing?: SimulationJudgeRoutingConfig;
}

export interface JudgeConfig {
  /** Model for critique plus per-turn screenshot/transition evaluation. Falls back to top-level `model`. */
  model?: string;
  /** Shared provider for critique and all simulation judge roles. Falls back to top-level `provider`.
   *  Set this to point evaluation at a different endpoint without affecting the builder. */
  provider?: ProviderFile;
}

export interface ExtraGlobalJudgerConfig {
  /** Enable the run-level GlobalJudger. Defaults to false. */
  enabled?: boolean;
  /** Model used for action decisions and the run-level GlobalJudger. Falls back to the turn judge model. */
  model?: string;
}

export interface UserSimulationConfig {
  /** Optional run-level evaluator. Its model is also used for simulation decisions. */
  extraGlobalJudger?: ExtraGlobalJudgerConfig;
  /** Markdown file describing the simulated user's persona. */
  personaPath?: string;
  /** Markdown file describing psychometric metrics, scales, and guidelines. */
  psychometricPath?: string;
  /** Directory where screenshots, optional MP4 recordings, decisions, and traces are written. */
  outputDir?: string;
  /** Default timing budget used when maxAdvanceMs is omitted. */
  stepMs?: number;
  /** Chunk size used while advancing held actions and deterministic observations. */
  frameMs?: number;
  /** Record Playwright video when saveFullRecord is enabled. */
  recordVideo?: boolean;
  /** Also save a full-run MP4 recording when system ffmpeg supports it. */
  saveFullRecord?: boolean;
  /** Max simulated user turns before the episode stops. */
  turnLimit?: number;
  /** @deprecated Use turnLimit. */
  defaultSteps?: number;
  /** Simulation-only weighted endpoint routing, populated from environment variables. */
  judgeRouting?: SimulationJudgeRoutingConfig;
}

export interface ResolvedUserSimulationConfig extends Omit<UserSimulationConfig, "extraGlobalJudger" | "judgeRouting"> {
  /** Model used for per-turn screenshot and transition evaluations. */
  turnJudge: ResolvedSimulationJudge;
  /** Model used for action decisions. */
  decisionJudge: ResolvedSimulationJudge;
  extraGlobalJudger?: {
    enabled: boolean;
    judge: { model: string; provider?: ProviderFile };
  };
}

export interface UserConfig {
  model?: string;
  provider?: ProviderFile;
  /** Shared model + endpoint for critique and per-turn simulation evaluation. */
  judge?: JudgeConfig;
  /** Configuration for the user simulation engine/tool. */
  userSimulation?: UserSimulationConfig;
  skills?: string[];
  playwright?: boolean;
}

/**
 * Built-in defaults. Used when neither the config file nor env vars set a
 * value. Custom endpoints and credentials must be supplied by the user
 * through `agens config set ...`, env vars, or CLI flags.
 */
export const BUILTIN_DEFAULTS: UserConfig = {
  model: "claude-sonnet-4.6",
  provider: {
    type: "openai",
  },
  // Core skills always available unless user explicitly overrides `skills`.
  skills: ["html-artisan", "critic"],
  // Default judge: same endpoint, same model. Users can point this at
  // another OpenAI-compatible vision model for true multi-model critique:
  //   agens config set judge.model gpt-5.2
  //   agens config set judge.provider.baseUrl http://host:port/v1
  //   agens config set judge.provider.apiKey <key>
  judge: {
    model: "gpt-5.2",
  },
  userSimulation: {
    outputDir: ".agens-sim",
    stepMs: 300,
    frameMs: 50,
    recordVideo: true,
    saveFullRecord: false,
    turnLimit: 20,
    extraGlobalJudger: { enabled: false },
  },
  playwright: true,
};

export function loadConfigFile(): UserConfig {
  if (!existsSync(CONFIG_PATH)) return {};
  try {
    return JSON.parse(readFileSync(CONFIG_PATH, "utf8")) as UserConfig;
  } catch (err) {
    throw new Error(`Failed to parse ${CONFIG_PATH}: ${(err as Error).message}`);
  }
}

/** Merge built-in defaults under a user-supplied config (user wins). */
export function withDefaults(cfg: UserConfig): UserConfig {
  validateRemovedSimulationConfig(cfg);
  const provider = { ...(BUILTIN_DEFAULTS.provider ?? {}), ...(cfg.provider ?? {}) };
  const judgeProvider = { ...(BUILTIN_DEFAULTS.judge?.provider ?? {}), ...(cfg.judge?.provider ?? {}) };
  const judge: JudgeConfig = {
    ...(BUILTIN_DEFAULTS.judge ?? {}),
    ...(cfg.judge ?? {}),
    ...(Object.keys(judgeProvider).length ? { provider: judgeProvider } : {}),
  };
  return {
    ...BUILTIN_DEFAULTS,
    ...cfg,
    provider,
    judge,
    userSimulation: {
      ...(BUILTIN_DEFAULTS.userSimulation ?? {}),
      ...(cfg.userSimulation ?? {}),
    },
  };
}

export function saveConfigFile(cfg: UserConfig): void {
  if (!existsSync(CONFIG_DIR)) mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 });
  writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2) + "\n", { mode: 0o600 });
  try { chmodSync(CONFIG_PATH, 0o600); } catch { /* best-effort on Windows */ }
}

/** Apply env-var overrides on top of a file-loaded config (env wins). */
export function applyEnv(cfg: UserConfig): UserConfig {
  validateRemovedSimulationEnv(process.env);
  const out: UserConfig = { ...cfg, provider: { ...(cfg.provider ?? {}) } };
  const e = process.env;
  if (e.AGENS_MODEL) out.model = e.AGENS_MODEL;
  if (e.AGENS_BASE_URL) out.provider!.baseUrl = e.AGENS_BASE_URL;
  if (e.AGENS_API_KEY) out.provider!.apiKey = e.AGENS_API_KEY;
  if (e.AGENS_BEARER_TOKEN) out.provider!.bearerToken = e.AGENS_BEARER_TOKEN;
  if (e.AGENS_PROVIDER_TYPE) out.provider!.type = e.AGENS_PROVIDER_TYPE as ProviderFile["type"];
  if (e.AGENS_WIRE_API) out.provider!.wireApi = e.AGENS_WIRE_API as ProviderFile["wireApi"];
  if (e.AGENS_JUDGE_BASE_URL || e.AGENS_JUDGE_API_KEY || e.AGENS_JUDGE_PROVIDER_TYPE || e.AGENS_JUDGE_WIRE_API) {
    out.judge = {
      ...(out.judge ?? {}),
      provider: {
        ...(out.judge?.provider ?? {}),
        ...(e.AGENS_JUDGE_BASE_URL ? { baseUrl: e.AGENS_JUDGE_BASE_URL } : {}),
        ...(e.AGENS_JUDGE_API_KEY ? { apiKey: e.AGENS_JUDGE_API_KEY } : {}),
        ...(e.AGENS_JUDGE_PROVIDER_TYPE
          ? { type: e.AGENS_JUDGE_PROVIDER_TYPE as ProviderFile["type"] }
          : {}),
        ...(e.AGENS_JUDGE_WIRE_API
          ? { wireApi: e.AGENS_JUDGE_WIRE_API as ProviderFile["wireApi"] }
          : {}),
      },
    };
  }
  if (e.AGENS_SIM_JUDGE_MODEL) {
    out.judge = { ...(out.judge ?? {}), model: e.AGENS_SIM_JUDGE_MODEL };
  }
  if (e.AGENS_SIM_JUDGE_BASE_URL || e.AGENS_SIM_JUDGE_API_KEY || e.AGENS_SIM_JUDGE_PROVIDER_TYPE || e.AGENS_SIM_JUDGE_WIRE_API) {
    out.judge = {
      ...(out.judge ?? {}),
      provider: {
        ...(out.judge?.provider ?? {}),
        ...(e.AGENS_SIM_JUDGE_BASE_URL ? { baseUrl: e.AGENS_SIM_JUDGE_BASE_URL } : {}),
        ...(e.AGENS_SIM_JUDGE_API_KEY ? { apiKey: e.AGENS_SIM_JUDGE_API_KEY } : {}),
        ...(e.AGENS_SIM_JUDGE_PROVIDER_TYPE
          ? { type: e.AGENS_SIM_JUDGE_PROVIDER_TYPE as ProviderFile["type"] }
          : {}),
        ...((e.AGENS_SIM_JUDGE_WIRE_API ?? e.AGENS_WIRE_API)
          ? { wireApi: (e.AGENS_SIM_JUDGE_WIRE_API ?? e.AGENS_WIRE_API) as ProviderFile["wireApi"] }
          : {}),
      },
    };
  }
  if (e.AGENS_SIM_JUDGE_BASE_URL_ADDITION) {
    if (!e.AGENS_SIM_JUDGE_API_KEY_ADDITION) {
      throw new Error(
        "AGENS_SIM_JUDGE_API_KEY_ADDITION is required when AGENS_SIM_JUDGE_BASE_URL_ADDITION is set.",
      );
    }
    const ratioText = e.AGENS_SIM_JUDGE_ROUTING_RATIO?.trim();
    const ratio = !ratioText
      ? 1
      : Number(ratioText);
    if (!Number.isFinite(ratio) || ratio < 0) {
      throw new Error(
        "AGENS_SIM_JUDGE_ROUTING_RATIO must be a finite non-negative number.",
      );
    }
    out.userSimulation = {
      ...(out.userSimulation ?? {}),
      judgeRouting: {
        additionalProvider: {
          baseUrl: e.AGENS_SIM_JUDGE_BASE_URL_ADDITION,
          apiKey: e.AGENS_SIM_JUDGE_API_KEY_ADDITION,
          type: "openai",
          wireApi: "completions",
        },
        ratio,
      },
    };
  }
  if (e.AGENS_SIM_PERSONA_PATH) {
    out.userSimulation = { ...(out.userSimulation ?? {}), personaPath: e.AGENS_SIM_PERSONA_PATH };
  }
  if (e.AGENS_SIM_METRICS_PATH) {
    out.userSimulation = { ...(out.userSimulation ?? {}), psychometricPath: e.AGENS_SIM_METRICS_PATH };
  }
  if (e.AGENS_SIM_OUTPUT_DIR) {
    out.userSimulation = { ...(out.userSimulation ?? {}), outputDir: e.AGENS_SIM_OUTPUT_DIR };
  }
  if (e.AGENS_SIM_STEP_MS) {
    out.userSimulation = { ...(out.userSimulation ?? {}), stepMs: Number(e.AGENS_SIM_STEP_MS) };
  }
  if (e.AGENS_SIM_FRAME_MS) {
    out.userSimulation = { ...(out.userSimulation ?? {}), frameMs: Number(e.AGENS_SIM_FRAME_MS) };
  }
  if (e.AGENS_SIM_TURN_LIMIT) {
    out.userSimulation = { ...(out.userSimulation ?? {}), turnLimit: Number(e.AGENS_SIM_TURN_LIMIT) };
  }
  if (e.AGENS_SIM_RECORD_VIDEO) {
    out.userSimulation = {
      ...(out.userSimulation ?? {}),
      recordVideo: e.AGENS_SIM_RECORD_VIDEO !== "false",
    };
  }
  if (e.AGENS_SIM_SAVE_FULL_RECORD) {
    out.userSimulation = {
      ...(out.userSimulation ?? {}),
      saveFullRecord: e.AGENS_SIM_SAVE_FULL_RECORD === "true",
    };
  }
  if (e.AGENS_SIM_EXTRA_GLOBAL_JUDGE_ENABLED || e.AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL) {
    out.userSimulation = {
      ...(out.userSimulation ?? {}),
      extraGlobalJudger: {
        ...(out.userSimulation?.extraGlobalJudger ?? {}),
        ...(e.AGENS_SIM_EXTRA_GLOBAL_JUDGE_ENABLED
          ? { enabled: e.AGENS_SIM_EXTRA_GLOBAL_JUDGE_ENABLED === "true" }
          : {}),
        ...(e.AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL
          ? { model: e.AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL }
          : {}),
      },
    };
  }
  // Clean up an empty provider object so we don't pass one to the SDK.
  if (out.provider && Object.keys(out.provider).length === 0) delete out.provider;
  return out;
}

/**
 * Resolve the effective judge model + provider, applying the fallback chain
 * judge.model → main model, judge.provider → main provider. Returns the
 * pair the critique tool actually needs.
 */
export function resolveJudge(cfg: UserConfig): { model: string; provider?: ProviderFile } {
  const model = cfg.judge?.model ?? cfg.model ?? "gpt-5";
  const provider = cfg.judge?.provider?.baseUrl ? cfg.judge.provider : cfg.provider;
  return { model, provider };
}

export function resolveUserSimulation(cfg: UserConfig): ResolvedUserSimulationConfig {
  validateRemovedSimulationConfig(cfg);
  const sim = cfg.userSimulation ?? {};
  const { judgeRouting, ...resolvedSimulation } = sim;
  const baseJudge = resolveJudge(cfg);
  const turnJudge: ResolvedSimulationJudge = {
    ...baseJudge,
    ...(judgeRouting ? { routing: judgeRouting } : {}),
  };
  const decisionJudge = {
    model: sim.extraGlobalJudger?.model ?? turnJudge.model,
    provider: turnJudge.provider,
    ...(turnJudge.routing ? { routing: turnJudge.routing } : {}),
  };
  return {
    ...resolvedSimulation,
    turnJudge,
    decisionJudge,
    extraGlobalJudger: {
      enabled: sim.extraGlobalJudger?.enabled === true,
      judge: {
        model: decisionJudge.model,
        provider: decisionJudge.provider,
      },
    },
    outputDir: sim.outputDir ?? ".agens-sim",
    stepMs: sim.stepMs ?? 300,
    frameMs: sim.frameMs ?? 50,
    recordVideo: sim.recordVideo !== false,
    saveFullRecord: sim.saveFullRecord === true,
    turnLimit: sim.turnLimit ?? sim.defaultSteps ?? 20,
    defaultSteps: sim.defaultSteps,
  };
}

function validateRemovedSimulationConfig(cfg: UserConfig): void {
  const sim = cfg.userSimulation as (UserSimulationConfig & {
    judge?: unknown;
    screenshotJudger?: unknown;
  }) | undefined;
  if (sim?.judge !== undefined) {
    throw new Error(
      "userSimulation.judge was removed; use judge.model for per-turn evaluation and " +
      "userSimulation.extraGlobalJudger.model for decisions/global evaluation.",
    );
  }
  if (sim?.screenshotJudger !== undefined) {
    throw new Error(
      "userSimulation.screenshotJudger was removed; the combined transition/screenshot judge now runs for every turn.",
    );
  }
}

function validateRemovedSimulationEnv(env: NodeJS.ProcessEnv): void {
  const removed = [
    "AGENS_SIM_SCREENSHOT_JUDGE_ENABLED",
    "AGENS_SIM_SCREENSHOT_JUDGE_MODEL",
    "AGENS_SIM_SCREENSHOT_JUDGE_BASE_URL",
    "AGENS_SIM_SCREENSHOT_JUDGE_API_KEY",
    "AGENS_SIM_SCREENSHOT_JUDGE_PROVIDER_TYPE",
  ].filter((name) => env[name] !== undefined);
  if (removed.length) {
    throw new Error(
      `${removed.join(", ")} ${removed.length === 1 ? "was" : "were"} removed; ` +
      "use AGENS_SIM_JUDGE_MODEL and AGENS_SIM_JUDGE_BASE_URL/API_KEY for per-turn evaluation.",
    );
  }
}

/**
 * Tiny dot-path set/get so `agens config set provider.baseUrl X` Just Works
 * without us hand-coding every field.
 */
export function setByPath(obj: Record<string, any>, path: string, value: unknown): void {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i];
    if (typeof cur[k] !== "object" || cur[k] === null) cur[k] = {};
    cur = cur[k];
  }
  cur[parts[parts.length - 1]] = value;
}

export function getByPath(obj: Record<string, any>, path: string): unknown {
  return path.split(".").reduce<any>((acc, k) => (acc == null ? acc : acc[k]), obj);
}

/**
 * Redact secrets for `agens config show`. Keep first/last 4 chars so users
 * can tell which key is loaded without leaking it.
 */
export function redact(cfg: UserConfig): UserConfig {
  const mask = (s?: string) => (s && s.length > 8 ? `${s.slice(0, 4)}…${s.slice(-4)}` : s ? "***" : s);
  if (!cfg.provider && !cfg.judge && !cfg.userSimulation) return cfg;
  return {
    ...cfg,
    provider: cfg.provider && {
      ...cfg.provider,
      apiKey: mask(cfg.provider.apiKey),
      bearerToken: mask(cfg.provider.bearerToken),
    },
    judge: cfg.judge && {
      ...cfg.judge,
      provider: cfg.judge.provider && {
        ...cfg.judge.provider,
        apiKey: mask(cfg.judge.provider.apiKey),
        bearerToken: mask(cfg.judge.provider.bearerToken),
      },
    },
    userSimulation: cfg.userSimulation && {
      ...cfg.userSimulation,
      judgeRouting: cfg.userSimulation.judgeRouting && {
        ...cfg.userSimulation.judgeRouting,
        additionalProvider: {
          ...cfg.userSimulation.judgeRouting.additionalProvider,
          apiKey: mask(cfg.userSimulation.judgeRouting.additionalProvider.apiKey),
          bearerToken: mask(cfg.userSimulation.judgeRouting.additionalProvider.bearerToken),
        },
      },
    },
  };
}
