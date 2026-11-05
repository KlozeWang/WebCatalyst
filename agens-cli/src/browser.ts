/**
 * Browser detection for the Playwright MCP server.
 *
 * The default `@playwright/mcp` setup assumes a bundled Chromium that's been
 * `npx playwright install`-ed. On a fresh Windows machine that bundled
 * Chromium isn't present, so it crashes with:
 *
 *   Chromium distribution 'chrome' is not found at
 *   C:\Users\<u>\AppData\Local\Google\Chrome\Application\chrome.exe
 *
 * Instead we probe the system for an already-installed Chromium-family
 * browser (Edge first on Windows since it ships with the OS, then Chrome),
 * and pass `--browser <channel>` to playwright-mcp so it reuses what's
 * already there. If nothing is found we fall back to bundled Chromium and
 * trigger an install up-front so the agent never has to do it mid-task.
 */
import { existsSync, readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir, platform } from "node:os";
import { join } from "node:path";
import chalk from "chalk";

/** Playwright MCP `--browser` channels that map to a system install. */
export type BrowserChannel = "msedge" | "chrome" | "firefox" | "chromium";

export interface BrowserChoice {
  /** Value to pass to `--browser`. `chromium` means use Playwright's bundled build. */
  channel: BrowserChannel;
  /** Friendly name for logging. */
  label: string;
  /** Absolute path we detected (undefined for bundled chromium). */
  path?: string;
}

function fileFirst(paths: string[]): string | undefined {
  for (const p of paths) if (p && existsSync(p)) return p;
  return undefined;
}

function detectWindows(): BrowserChoice | undefined {
  const pf = process.env["ProgramFiles"] ?? "C:\\Program Files";
  const pf86 = process.env["ProgramFiles(x86)"] ?? "C:\\Program Files (x86)";
  const lad = process.env["LOCALAPPDATA"] ?? join(homedir(), "AppData", "Local");

  // Edge ships with Windows 10+ — preferred default.
  const edge = fileFirst([
    join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
    join(pf86, "Microsoft", "Edge", "Application", "msedge.exe"),
    join(lad, "Microsoft", "Edge", "Application", "msedge.exe"),
  ]);
  if (edge) return { channel: "msedge", label: "Microsoft Edge", path: edge };

  const chrome = fileFirst([
    join(pf, "Google", "Chrome", "Application", "chrome.exe"),
    join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
    join(lad, "Google", "Chrome", "Application", "chrome.exe"),
  ]);
  if (chrome) return { channel: "chrome", label: "Google Chrome", path: chrome };

  return undefined;
}

function detectMac(): BrowserChoice | undefined {
  const chrome = fileFirst([
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    join(homedir(), "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
  ]);
  if (chrome) return { channel: "chrome", label: "Google Chrome", path: chrome };

  const edge = fileFirst([
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    join(homedir(), "Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
  ]);
  if (edge) return { channel: "msedge", label: "Microsoft Edge", path: edge };

  return undefined;
}

function detectLinux(): BrowserChoice | undefined {
  // Use `which` so we honour $PATH and snap/flatpak shims.
  const which = (cmd: string): string | undefined => {
    const r = spawnSync("which", [cmd], { encoding: "utf8" });
    const p = r.stdout?.trim();
    return p && existsSync(p) ? p : undefined;
  };
  for (const [cmd, channel, label] of [
    ["google-chrome", "chrome", "Google Chrome"],
    ["google-chrome-stable", "chrome", "Google Chrome"],
    ["chromium", "chromium", "Chromium"],
    ["chromium-browser", "chromium", "Chromium"],
    ["microsoft-edge", "msedge", "Microsoft Edge"],
  ] as const) {
    const p = which(cmd);
    if (p) return { channel, label, path: p };
  }
  return undefined;
}

/**
 * Detect a usable Chromium-family browser on the host system.
 * Returns `undefined` if nothing is found (caller should fall back to
 * Playwright's bundled chromium and likely trigger an install).
 */
export function detectSystemBrowser(): BrowserChoice | undefined {
  const p = platform();
  if (p === "win32") return detectWindows();
  if (p === "darwin") return detectMac();
  return detectLinux();
}

/**
 * Check whether Playwright already has its bundled Chromium installed.
 * We look in `ms-playwright` cache (the default `PLAYWRIGHT_BROWSERS_PATH`).
 */
export function hasBundledChromium(): boolean {
  const cacheRoot =
    process.env.PLAYWRIGHT_BROWSERS_PATH ||
    (platform() === "win32"
      ? join(process.env["LOCALAPPDATA"] ?? join(homedir(), "AppData", "Local"), "ms-playwright")
      : platform() === "darwin"
        ? join(homedir(), "Library", "Caches", "ms-playwright")
        : join(homedir(), ".cache", "ms-playwright"));
  if (!existsSync(cacheRoot)) return false;
  // Any chromium-* directory counts.
  try {
    return readdirSync(cacheRoot).some((n) => n.startsWith("chromium"));
  } catch {
    return false;
  }
}

/**
 * Run `npx playwright install chromium` synchronously, streaming progress to
 * stderr so the user sees what's happening. Returns true on success.
 */
export function installBundledChromium(): boolean {
  process.stderr.write(
    chalk.yellow("⏳ Installing Playwright Chromium (one-time, ~150 MB)…\n"),
  );
  const r = spawnSync("npx", ["-y", "playwright", "install", "chromium"], {
    stdio: "inherit",
    shell: platform() === "win32",
  });
  if (r.status === 0) {
    process.stderr.write(chalk.green("✓ Chromium installed.\n"));
    return true;
  }
  process.stderr.write(chalk.red("✗ Chromium install failed.\n"));
  return false;
}

/**
 * Resolve which browser to hand to playwright-mcp. Honours:
 *   - explicit override (CLI flag / config / env)
 *   - system browser detection (msedge > chrome)
 *   - bundled chromium (auto-installs if missing)
 */
export function resolveBrowser(override?: string): BrowserChoice {
  const env = process.env.AGENS_BROWSER;
  const wanted = (override ?? env ?? "auto").toLowerCase();

  if (wanted !== "auto") {
    if (wanted === "chromium") {
      if (!hasBundledChromium()) installBundledChromium();
      return { channel: "chromium", label: "Playwright Chromium (bundled)" };
    }
    if (wanted === "msedge" || wanted === "edge") {
      const sys = detectSystemBrowser();
      if (sys?.channel === "msedge") return sys;
      return { channel: "msedge", label: "Microsoft Edge (channel)" };
    }
    if (wanted === "chrome") {
      const sys = detectSystemBrowser();
      if (sys?.channel === "chrome") return sys;
      return { channel: "chrome", label: "Google Chrome (channel)" };
    }
    if (wanted === "firefox") return { channel: "firefox", label: "Firefox" };
  }

  const sys = detectSystemBrowser();
  if (sys) return sys;

  if (!hasBundledChromium()) installBundledChromium();
  return { channel: "chromium", label: "Playwright Chromium (bundled)" };
}
