import assert from "node:assert/strict";
import test from "node:test";
import { JudgeTaskQueue } from "../src/simulation/judgeTaskQueue.js";

test("JudgeTaskQueue runs one task at a time in FIFO order", async () => {
  const queue = new JudgeTaskQueue();
  const events: string[] = [];
  let releaseFirst!: () => void;
  const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve; });

  const first = queue.enqueue("turn-000 combined", async () => {
    events.push("combined-000:start");
    await firstGate;
    events.push("combined-000:end");
    return "first-result";
  });
  events.push("decision-001");
  const second = queue.enqueue("turn-001 combined", async () => {
    events.push("combined-001:start");
    await Promise.resolve();
    events.push("combined-001:end");
    return "second-result";
  });

  assert.deepEqual(events, ["combined-000:start", "decision-001"]);
  releaseFirst();
  assert.equal(await first, "first-result");
  assert.equal(await second, "second-result");
  await queue.drain();
  assert.deepEqual(events, [
    "combined-000:start",
    "decision-001",
    "combined-000:end",
    "combined-001:start",
    "combined-001:end",
  ]);
});

test("JudgeTaskQueue drains accepted work and rejects new work after a failure", async () => {
  const queue = new JudgeTaskQueue();
  const events: string[] = [];
  const failed = queue.enqueue("turn-000 combined", async () => {
    events.push("combined-000");
    await Promise.resolve();
    throw new Error("judge unavailable");
  });
  const acceptedBeforeFailure = queue.enqueue("turn-001 combined", async () => {
    events.push("combined-001");
  });

  await assert.rejects(failed, /turn-000 combined: judge unavailable/);
  assert.throws(
    () => queue.enqueue("turn-002 combined", async () => undefined),
    /Turn judge queue failed/,
  );
  await acceptedBeforeFailure;
  await assert.rejects(queue.drain(), /turn-000 combined/);
  assert.deepEqual(events, ["combined-000", "combined-001"]);
});
