import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { detectSystemBrowser, hasBundledChromium } from "../src/browser.js";
import {
  mergeDepthRanges,
  OnlineActionTraceCollector,
  reduceMaximalCallStacks,
} from "../src/simulation/actionTrace.js";
import { launchBrowser } from "../src/simulation/playwright.js";
import type { UserSimulationTurn } from "../src/simulation/types.js";

test("maximal call stacks remove shallower function-entry suffixes", () => {
  assert.deepEqual(
    reduceMaximalCallStacks([
      ["handler"],
      ["middle", "handler"],
      ["leaf", "middle", "handler"],
      ["leaf", "middle", "handler"],
      ["other", "handler"],
      ["recursive", "recursive", "handler"],
    ]),
    [["leaf", "middle", "handler"], ["recursive", "recursive", "handler"], ["other", "handler"]],
  );
});

test("source ranges merge by minimum depth and emit UTF-8 byte offsets", () => {
  const artifact = "😀abcdef";

  assert.deepEqual(mergeDepthRanges([
    { utf16Start: 2, utf16End: 5, depth: 2 },
    { utf16Start: 3, utf16End: 6, depth: 0 },
  ], artifact), [
    { utf8ByteStart: 4, utf8ByteEnd: 5, depth: 2 },
    { utf8ByteStart: 5, utf8ByteEnd: 8, depth: 0 },
  ]);
});

test("online trace records leaf-to-caller depths in the simulation browser", { timeout: 30_000 }, async (t) => {
  const systemBrowser = detectSystemBrowser();
  if (!systemBrowser && !hasBundledChromium()) {
    t.skip("Chromium is not installed");
    return;
  }
  const browserName = systemBrowser ? "auto" : "chromium";
  let browser;
  try {
    browser = await launchBrowser(browserName, false);
  } catch (error) {
    t.skip(`Chromium cannot launch: ${error instanceof Error ? error.message.split("\n", 1)[0] : String(error)}`);
    return;
  }

  const fixtureRoot = await mkdtemp(path.join(tmpdir(), "agens-online-trace-"));
  const artifactPath = path.join(fixtureRoot, "index.html");
  const artifact = `<!doctype html>
<button id="go">go</button><div id="state"></div>
<script>
function leaf() { document.querySelector('#state').textContent = 'done'; }
function middle() { leaf(); }
function handler() { middle(); }
document.querySelector('#go').addEventListener('click', handler);
</script>`;
  await writeFile(artifactPath, artifact, "utf8");
  const artifactUrl = pathToFileURL(artifactPath).href;
  const context = await browser.newContext();
  const page = await context.newPage();
  const collector = await OnlineActionTraceCollector.create(
    page,
    artifactUrl,
    await readFile(artifactPath, "utf8"),
    { enabled: true, threshold: 70, maxPauseEventsPerAction: 100 },
  );
  try {
    await collector.initialize();
    await page.goto(artifactUrl, { waitUntil: "networkidle" });
    await collector.prepare();
    assert.equal(await collector.startTurn(0, { type: "click", selector: "#go" }), true);
    await page.click("#go");
    await collector.finishTurn();
    await collector.dispose();

    const attribution = collector.buildAttribution([{
      turnIndex: 0,
      action: { type: "click", selector: "#go" },
      transitionEvaluation: { score: 70 },
    } as UserSimulationTurn]);
    assert.equal(attribution.status, "complete");
    assert.equal(attribution.selectedActionCount, 1);
    assert.equal(attribution.attributedActionCount, 1);
    assert.equal(depthAt(attribution.runs, artifact, artifact.indexOf("function leaf")), 0);
    assert.equal(depthAt(attribution.runs, artifact, artifact.indexOf("function middle")), 1);
    assert.equal(depthAt(attribution.runs, artifact, artifact.indexOf("function handler")), 2);
  } finally {
    await collector.dispose().catch(() => undefined);
    await context.close().catch(() => undefined);
    await browser.close().catch(() => undefined);
    await rm(fixtureRoot, { recursive: true, force: true });
  }
});

function depthAt(
  runs: Array<{ utf8ByteStart: number; utf8ByteEnd: number; depth: number }>,
  artifact: string,
  utf16Offset: number,
): number | undefined {
  const byteOffset = Buffer.byteLength(artifact.slice(0, utf16Offset), "utf8");
  return runs.find((run) => run.utf8ByteStart <= byteOffset && byteOffset < run.utf8ByteEnd)?.depth;
}
