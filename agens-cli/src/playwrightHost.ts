/**
 * Singleton Playwright Browser kept alive for the whole agens process.
 *
 * Why: our `page_diagnostics` / `a11y_audit` tools used to spawn `npx
 * playwright …` per call, which boots a brand-new Chromium each time
 * (~2–8s of overhead per loop iteration on top of the Playwright MCP
 * server that's *already* running its own browser). With a singleton we
 * pay the cold-start once; every subsequent call gets a fresh
 * BrowserContext (cheap, isolates cookies/storage) and a fresh Page.
 *
 * Reuses the channel + executable path picked by `src/browser.ts`, so on
 * Windows we hit Edge instead of demanding bundled Chromium.
 */
import type { Browser, BrowserContext, Page } from "playwright";
import { chromium, firefox } from "playwright";
import { resolveBrowser, type BrowserChoice } from "./browser.js";

let browserP: Promise<{ browser: Browser; choice: BrowserChoice }> | null = null;

async function launch(): Promise<{ browser: Browser; choice: BrowserChoice }> {
  const choice = resolveBrowser();
  if (choice.channel === "firefox") {
    const browser = await firefox.launch({ headless: true });
    return { browser, choice };
  }
  // chrome / msedge / chromium all come from the `chromium` launcher.
  // Passing channel="chromium" to playwright is invalid — only system
  // channels are accepted there. So map our "bundled chromium" case to
  // "no channel option".
  const launchOpts: Parameters<typeof chromium.launch>[0] = { headless: true };
  if (choice.channel === "chrome" || choice.channel === "msedge") {
    launchOpts.channel = choice.channel;
  }
  if (choice.path && (choice.channel === "chrome" || choice.channel === "msedge")) {
    launchOpts.executablePath = choice.path;
  }
  const browser = await chromium.launch(launchOpts);

  // Auto-close on process exit so we don't leak Chromium processes.
  const close = () => {
    browser.close().catch(() => undefined);
  };
  process.once("exit", close);
  process.once("SIGINT", close);
  process.once("SIGTERM", close);

  return { browser, choice };
}

export async function getBrowser(): Promise<Browser> {
  if (!browserP) browserP = launch();
  return (await browserP).browser;
}

/**
 * Open a fresh, isolated page. Caller MUST `await page.context().close()`
 * (or `await closePage(page)`) when done — that's what tears down the
 * one-shot BrowserContext while keeping the underlying Browser warm.
 */
export async function newPage(): Promise<Page> {
  const browser = await getBrowser();
  const context: BrowserContext = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  return context.newPage();
}

export async function closePage(page: Page): Promise<void> {
  const ctx = page.context();
  try { await page.close(); } catch { /* ignore */ }
  try { await ctx.close(); } catch { /* ignore */ }
}

export async function disposeBrowser(): Promise<void> {
  if (!browserP) return;
  try {
    const { browser } = await browserP;
    await browser.close();
  } catch { /* ignore */ }
  browserP = null;
}
