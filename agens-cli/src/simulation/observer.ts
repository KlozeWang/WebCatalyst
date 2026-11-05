import type { BrowserContext, Page } from "playwright";
import { advanceClock } from "./playwright.js";
import type { SimulationAction, VideoSegment } from "./types.js";

export async function observeAfterAction(args: {
  page: Page;
  context: BrowserContext;
  action: SimulationAction;
  frameMs: number;
  maxAdvanceMs: number;
  turnIndex: number;
  videoSegments?: VideoSegment[];
}): Promise<{ reachedGoal: boolean; totalWaitMs: number }> {
  const durationMs = getActionDurationMs(args.action, args.frameMs);
  if (durationMs > 0) return { reachedGoal: false, totalWaitMs: 0 };

  const defaultAfterDelayMs = Math.min(50, args.maxAdvanceMs);
  if (defaultAfterDelayMs <= 0) return { reachedGoal: false, totalWaitMs: 0 };

  const segmentStart = Date.now();
  await advanceClock(args.context, args.page, defaultAfterDelayMs, args.frameMs);
  args.videoSegments?.push({
    label: `turn-${args.turnIndex}-default-after-delay`,
    startMs: segmentStart,
    endMs: Date.now(),
  });
  return { reachedGoal: false, totalWaitMs: defaultAfterDelayMs };
}

function getActionDurationMs(action: SimulationAction, frameMs: number): number {
  if (action.type === "click" || action.type === "press" || action.type === "combo" || action.type === "mouseMove") {
    if (!Number.isFinite(action.durationMs) || action.durationMs! < 0) return 0;
    return Math.max(Math.round(action.durationMs!), Math.round(frameMs * 2 + 25));
  }
  return 0;
}
