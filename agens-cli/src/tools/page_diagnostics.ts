/**
 * `page_diagnostics` tool — load a URL with Playwright and capture every
 * runtime error in one shot:
 *   - JS exceptions on the page
 *   - Console errors (level=error)
 *   - Failed network requests (4xx/5xx/abort)
 *
 * Backed by the singleton Browser in `playwrightHost.ts` so each call
 * only pays for a fresh BrowserContext (~50ms), not a full Chromium
 * cold-start.
 */
import { defineTool } from "@github/copilot-sdk";
import { z } from "zod";
import { newPage, closePage } from "../playwrightHost.js";

const parameters = z.object({
  url: z.string().describe("URL to load. For local artifacts use file:///abs/path/index.html."),
  waitMs: z.number().int().min(0).max(10_000).optional().describe(
    "Extra wait after networkidle to catch errors thrown by deferred scripts (Three.js init, etc). Default 500.",
  ),
});

interface JsException { message: string; stack?: string }
interface ConsoleError { text: string; location?: string }
interface NetworkFailure { url: string; status?: number; failure?: string }

export const pageDiagnosticsTool = defineTool("page_diagnostics", {
  description:
    "Load a URL in a headless browser and capture all runtime errors: JS exceptions, console errors, " +
    "and failed network requests. Use after every render in the iterate loop — if totalIssues > 0, fix " +
    "the listed issues before calling `critique`. This catches problems the screenshot can't see.",
  parameters,
  skipPermission: true,
  handler: async (args) => {
    const wait = args.waitMs ?? 500;
    const jsExceptions: JsException[] = [];
    const consoleErrors: ConsoleError[] = [];
    const networkFailures: NetworkFailure[] = [];
    let loaded = false;

    let page;
    try {
      page = await newPage();
    } catch (e) {
      return `ERROR: failed to launch browser: ${(e as Error).message}`;
    }

    page.on("pageerror", (err) => {
      jsExceptions.push({
        message: err.message,
        stack: err.stack?.split("\n").slice(0, 5).join("\n"),
      });
    });
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        const loc = msg.location();
        consoleErrors.push({
          text: msg.text(),
          location: loc?.url ? `${loc.url}:${loc.lineNumber}:${loc.columnNumber}` : undefined,
        });
      }
    });
    page.on("requestfailed", (req) => {
      networkFailures.push({ url: req.url(), failure: req.failure()?.errorText });
    });
    page.on("response", (resp) => {
      const status = resp.status();
      if (status >= 400) networkFailures.push({ url: resp.url(), status });
    });

    try {
      await page.goto(args.url, { waitUntil: "networkidle", timeout: 30_000 });
      if (wait > 0) await page.waitForTimeout(wait);
      loaded = true;
    } catch (e) {
      jsExceptions.push({ message: "navigation failed: " + ((e as Error)?.message ?? String(e)) });
    } finally {
      await closePage(page);
    }

    const totalIssues = jsExceptions.length + consoleErrors.length + networkFailures.length;
    return {
      url: args.url,
      loaded,
      totalIssues,
      passed: totalIssues === 0,
      jsExceptions,
      consoleErrors,
      networkFailures,
    };
  },
});
