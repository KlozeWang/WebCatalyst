/**
 * Shared session-builder config and the single place where we wire up:
 *   - the built-in Playwright MCP server
 *   - resolved skill directories
 *   - permission handling
 *   - BYOK provider (custom OpenAI-compatible / Azure / Anthropic endpoint)
 */
import chalk from "chalk";
import { CopilotClient, approveAll } from "@github/copilot-sdk";
import type { SessionConfig } from "@github/copilot-sdk";
import { resolveSkillDirs } from "./skills.js";
import type { ProviderFile, ResolvedUserSimulationConfig } from "./userconfig.js";
import { makeCritiqueTool } from "./tools/critique.js";
import { makeUserSimulationTool } from "./tools/user_simulation.js";
import { a11yTool } from "./tools/a11y.js";
import { pageDiagnosticsTool } from "./tools/page_diagnostics.js";
import { resolveBrowser } from "./browser.js";
import { runtimeFromResolvedConfig } from "./simulation/engine.js";

export interface HarnessConfig {
  model: string;
  skills: string[];
  enablePlaywright: boolean;
  cwd: string;
  provider?: ProviderFile;
  /** Independent model + provider for the critique tool. If omitted,
   *  critique reuses the builder agent's `model` and `provider`. */
  judge?: { model: string; provider?: ProviderFile };
  /** Browser channel for the Playwright MCP server: 'auto' | 'msedge' | 'chrome' | 'chromium' | 'firefox'. */
  browser?: string;
  /** When true, run the browser visibly (default false/headless). */
  headed?: boolean;
  /** Config for the user simulation engine/tool. */
  userSimulation: ResolvedUserSimulationConfig;
}

export function buildSessionConfig(cfg: HarnessConfig): SessionConfig {
  const mcpServers: NonNullable<SessionConfig["mcpServers"]> = {};

  if (cfg.enablePlaywright) {
    const browser = resolveBrowser(cfg.browser);
    const headless = !cfg.headed;
    process.stderr.write(
      chalk.gray(`· Using browser: ${browser.label}${browser.path ? ` (${browser.path})` : ""} (${headless ? "headless" : "headed"})\n`),
    );
    // --headless: never pop a window. Default in playwright-mcp is HEADED.
    // --isolated: use an in-memory profile so we don't fight Edge/Chrome user sessions.
    const pwArgs = [
      "-y",
      "@playwright/mcp@latest",
      "--browser", browser.channel,
      "--isolated",
    ];
    if (headless) pwArgs.push("--headless");
    if (browser.path && (browser.channel === "msedge" || browser.channel === "chrome")) {
      pwArgs.push("--executable-path", browser.path);
    }
    mcpServers["playwright"] = {
      type: "local",
      command: "npx",
      args: pwArgs,
      tools: ["*"],
      timeout: 60_000,
    };
  }

  const skillDirectories = resolveSkillDirs(cfg.skills);

  const sessionConfig: SessionConfig = {
    model: cfg.model,
    streaming: true,
    onPermissionRequest: approveAll,
    mcpServers,
    skillDirectories,
    // Critique tool runs as an INDEPENDENT model call. If a judge config is
    // set (e.g. Gemini for vision), use it; otherwise fall back to the
    // builder's own provider/model. This is how agens becomes multi-model.
    tools: [
      makeCritiqueTool(
        cfg.judge?.provider ?? cfg.provider,
        cfg.judge?.model ?? cfg.model,
      ),
      makeUserSimulationTool(
        runtimeFromResolvedConfig(cfg.cwd, cfg.userSimulation, cfg.browser, cfg.headed),
      ),
      a11yTool,
      pageDiagnosticsTool,
    ],
  };

  // BYOK: route LLM traffic through the user's own endpoint instead of
  // GitHub Copilot's hosted models. baseUrl is the minimum required field.
  if (cfg.provider?.baseUrl) {
    sessionConfig.provider = {
      type: cfg.provider.type ?? "openai",
      baseUrl: cfg.provider.baseUrl,
      ...(cfg.provider.apiKey ? { apiKey: cfg.provider.apiKey } : {}),
      ...(cfg.provider.bearerToken ? { bearerToken: cfg.provider.bearerToken } : {}),
      ...(cfg.provider.wireApi ? { wireApi: cfg.provider.wireApi } : {}),
    };
  }

  return sessionConfig;
}

export function createClient(cfg: HarnessConfig): CopilotClient {
  return new CopilotClient({ workingDirectory: cfg.cwd });
}
