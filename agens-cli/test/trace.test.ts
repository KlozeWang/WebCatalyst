import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { detectSystemBrowser, hasBundledChromium } from "../src/browser.js";
import { launchBrowser } from "../src/simulation/playwright.js";
import {
  mapArtifactCoverage,
  rgbSimilarity,
  runTraceAnalysis,
  TraceAnalysisRunError,
  traceSkipReasons,
  type RawScriptCoverage,
} from "../src/simulation/trace.js";

test("trace selection requires both an eligible input action and the score threshold", () => {
  assert.deepEqual(traceSkipReasons(70, { type: "click", selector: "#start" }, 70), []);
  assert.deepEqual(traceSkipReasons(69.9, { type: "press", key: "Enter" }, 70), ["below-threshold"]);
  assert.deepEqual(traceSkipReasons(90, { type: "wait", ms: 50 }, 70), ["unsupported-action"]);
  assert.deepEqual(traceSkipReasons(10, { type: "type", selector: "#name", text: "Revy" }, 70), [
    "below-threshold",
    "unsupported-action",
  ]);
});

test("coverage maps only exact inline artifact scripts to UTF-16 and UTF-8 HTML ranges", () => {
  const source = `function traced(flag) {\n  if (flag) return 1;\n${" ".repeat(600)}}\n`;
  const artifactText = `<!doctype html>\n<!-- 😀 -->\n<script>${source}</script>`;
  const artifactUrl = "file:///tmp/artifact/index.html";
  const blockStart = source.indexOf("if (flag)");
  const coverage: RawScriptCoverage[] = [
    {
      scriptId: "inline",
      url: artifactUrl,
      functions: [{
        functionName: "traced",
        isBlockCoverage: true,
        ranges: [
          { startOffset: 0, endOffset: source.length, count: 3 },
          { startOffset: blockStart, endOffset: blockStart + "if (flag) return 1;".length, count: 2 },
        ],
      }],
    },
    {
      scriptId: "external",
      url: "https://cdn.example/app.js",
      functions: [{
        functionName: "external",
        isBlockCoverage: false,
        ranges: [{ startOffset: 0, endOffset: 1, count: 1 }],
      }],
    },
    {
      scriptId: "missing-source",
      url: artifactUrl,
      functions: [{
        functionName: "missing",
        isBlockCoverage: false,
        ranges: [{ startOffset: 0, endOffset: 1, count: 1 }],
      }],
    },
    {
      scriptId: "iframe",
      url: artifactUrl,
      functions: [{
        functionName: "iframeCopy",
        isBlockCoverage: false,
        ranges: [{ startOffset: 0, endOffset: source.length, count: 1 }],
      }],
    },
  ];

  const mapped = mapArtifactCoverage({
    artifactText,
    artifactUrl,
    coverage,
    scripts: new Map([
      ["inline", { scriptId: "inline", url: artifactUrl, startLine: 2, startColumn: "<script>".length }],
      ["iframe", {
        scriptId: "iframe",
        url: artifactUrl,
        startLine: 2,
        startColumn: "<script>".length,
        isMainFrameDefaultWorld: false,
      }],
    ]),
    sources: new Map([["inline", source], ["iframe", source]]),
  });

  assert.deepEqual(mapped.summary, {
    executedScriptCount: 4,
    mappedScriptCount: 1,
    excludedScriptCount: 2,
    unmappedArtifactScriptCount: 1,
    executedFunctionCount: 1,
    executedRangeCount: 2,
  });
  assert.deepEqual(mapped.warnings, ["unmapped-artifact-script: scriptId=missing-source"]);
  assert.equal(mapped.functions.length, 1);
  const traced = mapped.functions[0]!;
  assert.equal(traced.functionName, "traced");
  assert.equal(traced.executionCount, 3);
  assert.equal(traced.range.start.line, 3);
  assert.equal(traced.range.start.column, 9);
  assert.equal(traced.range.utf16Start, artifactText.indexOf(source));
  assert.equal(traced.range.utf8ByteStart, Buffer.byteLength(artifactText.slice(0, artifactText.indexOf(source)), "utf8"));
  assert.equal(traced.range.utf8ByteStart, traced.range.utf16Start + 2);
  assert.equal(traced.range.snippet.length, 512);
  assert.equal(traced.range.snippetTruncated, true);
  assert.equal(traced.blocks[0]!.executionCount, 2);
  assert.equal(traced.blocks[0]!.range.snippet, "if (flag) return 1;");
});

test("RGB similarity is normalized between exact equality and maximum difference", () => {
  const black = new Uint8Array([0, 0, 0, 255, 0, 0, 0, 255]);
  const blackCopy = new Uint8Array(black);
  const white = new Uint8Array([255, 255, 255, 255, 255, 255, 255, 255]);
  assert.equal(rgbSimilarity(2, 1, black, blackCopy), 1);
  assert.equal(rgbSimilarity(2, 1, black, white), 0);
  assert.equal(rgbSimilarity(2, 1, black, new Uint8Array(4)), 0);
});

test("trace replays every turn while saving DOM and code only for selected turns", { timeout: 30_000 }, async (t) => {
  const systemBrowser = detectSystemBrowser();
  if (!systemBrowser && !hasBundledChromium()) {
    t.skip("Chromium is not installed");
    return;
  }
  const browserName = systemBrowser ? "auto" : "chromium";
  let browserProbe;
  try {
    browserProbe = await launchBrowser(browserName, false);
  } catch (error) {
    t.skip(`Chromium cannot launch in this environment: ${error instanceof Error ? error.message.split("\n", 1)[0] : String(error)}`);
    return;
  } finally {
    await browserProbe?.close().catch(() => undefined);
  }

  const fixtureRoot = await mkdtemp(path.join(tmpdir(), "agens-trace-"));
  try {
    const artifactDir = path.join(fixtureRoot, "artifact");
    const traceDir = path.join(fixtureRoot, "trace", "trace-1");
    const outputDir = path.join(fixtureRoot, "trace-analysis", "trace-1");
    await Promise.all([
      mkdir(artifactDir, { recursive: true }),
      mkdir(traceDir, { recursive: true }),
      mkdir(outputDir, { recursive: true }),
    ]);
    await writeFile(path.join(outputDir, "stale.txt"), "old output", "utf8");
    await writeFile(path.join(artifactDir, "index.html"), `<!doctype html>
<html><body>
<button id="start">start</button><div id="state">0</div>
<script>
let value = 0;
let started = false;
function render() { document.querySelector('#state').textContent = String(value); }
function add(delta) { value += delta; render(); }
document.querySelector('#start').addEventListener('click', () => {
  started = true;
  add(1);
  setTimeout(() => add(2), 60);
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'a') add(4);
});
document.addEventListener('mousemove', (event) => {
  if (started && event.clientX > 720) add(8);
});
</script>
</body></html>`, "utf8");

    const actions = [
      { type: "click", selector: "#start" },
      { type: "wait", ms: 100 },
      { type: "press", key: "a" },
      { type: "mouseMove", dx: 10, dy: 0, steps: 1 },
    ] as const;
    const scores = [80, 90, 60, 70];
    const simulatedAdvance = [50, 150, 50, 50];
    const actionDurations = [0, 100, 0, 0];
    const turns = [];
    for (let turnIndex = 0; turnIndex < actions.length; turnIndex++) {
      const label = `turn-${String(turnIndex).padStart(3, "0")}`;
      const action = actions[turnIndex]!;
      turns.push({
        turnIndex,
        action,
        simulatedAdvanceMs: simulatedAdvance[turnIndex],
        terminated: false,
        actionExecution: {
          actionType: action.type,
          success: true,
          durationMs: actionDurations[turnIndex],
        },
      });
      await writeJson(path.join(traceDir, `${label}-decision.json`), { action });
      await writeJson(path.join(traceDir, `${label}-transition-evaluation.json`), {
        turnIndex,
        score: scores[turnIndex],
      });
    }
    await writeJson(path.join(traceDir, "result.json"), { turns });

    const result = await runTraceAnalysis(
      { inputPath: fixtureRoot, threshold: 70 },
      { cwd: "/", browser: browserName },
    );

    assert.equal(result.ok, true);
    assert.equal(result.totalTurnCount, 4);
    assert.equal(result.processedTurnCount, 4);
    assert.equal(result.selectedTurnCount, 2);
    assert.deepEqual(result.turns.map((turn) => turn.selected), [true, false, false, true]);
    assert.deepEqual(result.turns.map((turn) => turn.status), ["replayed", "replayed", "replayed", "replayed"]);
    assert.equal(result.turns[2]!.replayExecution?.durationMs, 0);
    assert.equal(result.turns[3]!.replayExecution?.durationMs, 0);

    const firstDom = await readFile(path.join(outputDir, "turn-000-dom.html"), "utf8");
    const lastDom = await readFile(path.join(outputDir, "turn-003-dom.html"), "utf8");
    assert.match(firstDom, /<div id="state">1<\/div>/);
    assert.match(lastDom, /<div id="state">15<\/div>/);
    await assert.rejects(access(path.join(outputDir, "turn-001-dom.html")));
    await assert.rejects(access(path.join(outputDir, "turn-002-dom.html")));
    await assert.rejects(access(path.join(outputDir, "stale.txt")));

    const codeTrace = JSON.parse(await readFile(path.join(outputDir, "turn-003-code-trace.json"), "utf8"));
    assert.equal(codeTrace.turnIndex, 3);
    assert.equal(codeTrace.transitionScore, 70);
    assert.ok(codeTrace.functions.length > 0);
    assert.ok(codeTrace.coverageSummary.mappedScriptCount > 0);
    assert.ok(codeTrace.functions.some((fn: { range?: { snippet?: string } }) => fn.range?.snippet?.includes("add(8)")));

    const manifest = JSON.parse(await readFile(path.join(outputDir, "trace-result.json"), "utf8"));
    assert.equal(manifest.ok, true);
    assert.equal(manifest.processedTurnCount, 4);

    const failingTurns = turns.map((turn, turnIndex) => turnIndex === 1
      ? {
          ...turn,
          action: { type: "click", selector: "#missing" },
          actionExecution: { actionType: "click", success: false },
        }
      : turn);
    await writeJson(path.join(traceDir, "turn-001-decision.json"), {
      action: { type: "click", selector: "#missing" },
    });
    await writeJson(path.join(traceDir, "result.json"), { turns: failingTurns });
    const failure = await runTraceAnalysis(
      { inputPath: fixtureRoot, threshold: 70 },
      { cwd: "/", browser: browserName },
    ).then(
      () => undefined,
      (error: unknown) => error,
    );
    assert.ok(failure instanceof TraceAnalysisRunError);
    assert.equal(failure.result.ok, false);
    assert.equal(failure.result.processedTurnCount, 1);
    assert.equal(failure.result.turns[0]!.status, "replayed");
    assert.equal(failure.result.turns[1]!.status, "failed");
    assert.equal(failure.result.turns[2]!.status, "pending");
    const partialManifest = JSON.parse(await readFile(path.join(outputDir, "trace-result.json"), "utf8"));
    assert.equal(partialManifest.ok, false);
    assert.equal(partialManifest.processedTurnCount, 1);
    await access(path.join(outputDir, "turn-000-dom.html"));
    await assert.rejects(access(path.join(outputDir, "turn-003-dom.html")));
  } finally {
    await rm(fixtureRoot, { recursive: true, force: true });
  }
});

async function writeJson(filePath: string, value: unknown): Promise<void> {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
