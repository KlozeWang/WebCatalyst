import { chromium, firefox, type Browser, type BrowserContext, type Locator, type Page } from "playwright";
import { resolveBrowser } from "../browser.js";
import type { ActionExecutionMetadata, ActionHint, Point, SimulationAction } from "./types.js";

export interface HeldActionCheckpoint {
  checkpointMs: number;
  deltaMs: number;
  index: number;
}

export interface PerformActionOptions {
  onHeldActionCheckpoint?: (checkpoint: HeldActionCheckpoint) => Promise<void>;
}

type TargetResolution = NonNullable<ActionExecutionMetadata["targetResolution"]>;
type TargetResolutionMethod = TargetResolution["method"];

interface ResolvedActionTarget {
  locator?: Locator;
  point?: Point;
  method: TargetResolutionMethod;
}

export async function launchBrowser(browserName: string | undefined, headed: boolean | undefined): Promise<Browser> {
  const choice = resolveBrowser(browserName);
  const headless = !headed;
  if (choice.channel === "firefox") {
    return firefox.launch({ headless });
  }
  const launchOpts: Parameters<typeof chromium.launch>[0] = { headless };
  if (choice.channel === "chrome" || choice.channel === "msedge") launchOpts.channel = choice.channel;
  if (choice.path) {
    launchOpts.executablePath = choice.path;
  }
  return chromium.launch(launchOpts);
}

export async function pauseClockNow(context: BrowserContext, page: Page): Promise<void> {
  const now = await page.evaluate(() => Date.now()).catch(() => Date.now());
  await context.clock.pauseAt(now + 50);
}

export async function moveMouseToViewportCenter(page: Page): Promise<Point> {
  const viewport = page.viewportSize() ?? { width: 1440, height: 900 };
  const point = { x: viewport.width / 2, y: viewport.height / 2 };
  await page.mouse.move(point.x, point.y);
  await rememberMousePoint(page, point);
  return point;
}

export async function performAction(
  page: Page,
  context: BrowserContext,
  action: SimulationAction,
  frameMs: number,
  options: PerformActionOptions = {},
): Promise<{ advanceMs: number; metadata: ActionExecutionMetadata }> {
  const metadata: ActionExecutionMetadata = { actionType: action.type };

  try {
    switch (action.type) {
    case "click": {
      const ignoredCoordinates = await shouldIgnoreCanvasOriginCoordinates(page, action);
      const effectiveAction = ignoredCoordinates ? stripActionCoordinates(action) : action;
      if (ignoredCoordinates) {
        if (Number.isFinite(action.x) && Number.isFinite(action.y)) {
          metadata.ignoredCoordinates = { x: action.x!, y: action.y! };
        }
        metadata.activationMethod = "pointer-lock-canvas-click-current-mouse";
        metadata.targetResolution = {
          method: "pointer-lock-canvas-current-mouse",
          selector: action.selector,
        };
      }
      const target = ignoredCoordinates ? undefined : await resolveActionTarget(page, effectiveAction, metadata);
      const point = target?.point;
      if (point) metadata.hitTarget = await hitTest(page, point);
      if (!ignoredCoordinates && await isCanvasSelectorClickWithoutCoordinates(page, effectiveAction)) {
        addMetadataWarning(
          metadata,
          "canvas-selector-click-without-coordinates: selector-only canvas clicks hit the canvas center; use selector plus page x/y for 2D canvas internal targets.",
        );
      }
      const shouldCenterAfterClick = await isNativeButtonActionTarget(page, effectiveAction, target);
      const duration = normalizeDurationMs(effectiveAction.durationMs, frameMs, metadata);
      if (ignoredCoordinates) {
        metadata.durationMs = duration;
        await dispatchPointerLockMouseButton(page, "mousedown");
        try {
          if (duration > 0) {
            await advanceHeldActionClock(context, page, duration, frameMs, options.onHeldActionCheckpoint);
          }
        } finally {
          await dispatchPointerLockMouseButton(page, "mouseup");
        }
      } else if (duration > 0) {
        if (point) {
          await page.mouse.move(point.x, point.y);
          await rememberMousePoint(page, point);
        }
        metadata.durationMs = duration;
        await page.mouse.down();
        try {
          await advanceHeldActionClock(context, page, duration, frameMs, options.onHeldActionCheckpoint);
        } finally {
          await page.mouse.up();
        }
      } else if (typeof effectiveAction.x === "number" && typeof effectiveAction.y === "number") {
        await page.mouse.click(effectiveAction.x, effectiveAction.y);
        await rememberMousePoint(page, { x: effectiveAction.x, y: effectiveAction.y });
      }
      else if (target?.locator) await target.locator.click();
      else throw new Error("click action requires selector, text, or x/y coordinates.");
      if (shouldCenterAfterClick) {
        metadata.activationMethod = "native-button-click-then-center-mouse";
        metadata.mouseCenterPoint = await moveMouseToViewportCenter(page).catch(() => undefined);
      }
      return completeAction(metadata, metadata.durationMs ?? 0);
    }
    case "mouseMove": {
      const current = await currentMousePoint(page);
      const dx = Number.isFinite(action.dx) ? action.dx : 0;
      const dy = Number.isFinite(action.dy) ? action.dy : 0;
      const duration = normalizeDurationMs(action.durationMs, frameMs, metadata);
      const steps = clampInteger(action.steps ?? stepsForMouseMove(duration, frameMs), 1, 60);
      const target = { x: current.x + dx, y: current.y + dy };
      metadata.durationMs = duration;
      metadata.mouseMove = {
        from: current,
        to: target,
        delta: { x: dx, y: dy },
        steps,
      };
      if (await hasPointerLock(page)) {
        await dispatchPointerLockMouseMove(page, context, { x: dx, y: dy }, steps, metadata.durationMs, frameMs);
        await rememberMousePoint(page, current);
      } else if (metadata.durationMs > 0) {
        await moveMouseSmoothly(page, context, current, target, steps, metadata.durationMs, frameMs);
      } else {
        await page.mouse.move(target.x, target.y, { steps });
        await rememberMousePoint(page, target);
      }
      return completeAction(metadata, metadata.durationMs);
    }
    case "hover": {
      const target = await resolveActionTarget(page, action, metadata);
      const point = target?.point;
      if (point) metadata.hitTarget = await hitTest(page, point);
      if (typeof action.x === "number" && typeof action.y === "number") {
        await page.mouse.move(action.x, action.y);
        await rememberMousePoint(page, { x: action.x, y: action.y });
      }
      else if (target?.locator) await target.locator.hover();
      else if (point) {
        await page.mouse.move(point.x, point.y);
        await rememberMousePoint(page, point);
      }
      else throw new Error("hover action requires selector, text, or x/y coordinates.");
      return completeAction(metadata, 0);
    }
    case "press": {
      if (!action.key) throw new Error("press action requires key.");
      const duration = normalizeDurationMs(action.durationMs, frameMs, metadata);
      metadata.durationMs = duration;
      if (duration > 0) {
        await page.keyboard.down(action.key);
        try {
          await advanceHeldActionClock(context, page, duration, frameMs, options.onHeldActionCheckpoint);
        } finally {
          await page.keyboard.up(action.key);
        }
      } else {
        await page.keyboard.press(action.key);
      }
      return completeAction(metadata, duration);
    }
    case "combo": {
      const keys = normalizeComboKeysForExecution(action.keys);
      const duration = normalizeDurationMs(action.durationMs, frameMs, metadata);
      metadata.durationMs = duration;
      for (const key of keys) await page.keyboard.down(key);
      try {
        if (duration > 0) {
          await advanceHeldActionClock(context, page, duration, frameMs, options.onHeldActionCheckpoint);
        }
      } finally {
        for (const key of [...keys].reverse()) {
          await page.keyboard.up(key).catch(() => undefined);
        }
      }
      return completeAction(metadata, duration);
    }
    case "type":
      if (!action.selector) throw new Error("type action requires selector.");
      await page.fill(action.selector, action.text);
      return completeAction(metadata, 0);
    case "wait": {
      const waitMs = action.ms ?? 0;
      metadata.durationMs = waitMs;
      await advanceClock(context, page, waitMs, frameMs);
      return completeAction(metadata, waitMs);
    }
    case "terminate":
      return completeAction(metadata, 0);
    }
  } catch (err) {
    return failAction(metadata, err);
  }
}

function completeAction(
  metadata: ActionExecutionMetadata,
  advanceMs: number,
): { advanceMs: number; metadata: ActionExecutionMetadata } {
  metadata.success = true;
  return { advanceMs, metadata };
}

function failAction(
  metadata: ActionExecutionMetadata,
  err: unknown,
): { advanceMs: number; metadata: ActionExecutionMetadata } {
  metadata.success = false;
  metadata.executionError = serializeActionExecutionError(err);
  addMetadataWarning(metadata, `action-execution-error: ${firstLine(metadata.executionError.message)}`);
  return { advanceMs: metadata.durationMs ?? 0, metadata };
}

function serializeActionExecutionError(err: unknown): NonNullable<ActionExecutionMetadata["executionError"]> {
  const name = err instanceof Error ? err.name : undefined;
  const message = err instanceof Error ? err.message : String(err);
  return {
    name,
    message: message.slice(0, 2_000),
  };
}

function firstLine(text: string): string {
  return text.split(/\r?\n/, 1)[0] ?? text;
}

function normalizeComboKeysForExecution(keys: string[]): string[] {
  const normalized = keys
    .filter((key): key is string => typeof key === "string" && key.trim().length > 0)
    .map((key) => key.trim())
    .filter((key, index, all) => all.indexOf(key) === index);
  if (normalized.length < 1 || normalized.length > 2) {
    throw new Error("combo action requires 1 or 2 unique keyboard keys.");
  }
  return normalized;
}

async function moveMouseSmoothly(
  page: Page,
  context: BrowserContext,
  from: Point,
  to: Point,
  steps: number,
  durationMs: number,
  frameMs: number,
): Promise<void> {
  let previousElapsed = 0;
  for (let i = 1; i <= steps; i++) {
    const ratio = i / steps;
    const point = {
      x: from.x + (to.x - from.x) * ratio,
      y: from.y + (to.y - from.y) * ratio,
    };
    await page.mouse.move(point.x, point.y);
    await rememberMousePoint(page, point);
    const elapsed = Math.round(durationMs * ratio);
    const delta = elapsed - previousElapsed;
    if (delta > 0) await advanceClock(context, page, delta, frameMs);
    previousElapsed = elapsed;
  }
}

async function dispatchPointerLockMouseMove(
  page: Page,
  context: BrowserContext,
  delta: Point,
  steps: number,
  durationMs: number,
  frameMs: number,
): Promise<void> {
  let previousElapsed = 0;
  for (let i = 1; i <= steps; i++) {
    const stepDelta = {
      x: delta.x / steps,
      y: delta.y / steps,
    };
    await page.evaluate(({ dx, dy }) => {
      const target = document.pointerLockElement ?? document;
      target.dispatchEvent(new MouseEvent("mousemove", {
        bubbles: true,
        cancelable: true,
        movementX: dx,
        movementY: dy,
        buttons: 0,
      }));
    }, { dx: stepDelta.x, dy: stepDelta.y });
    const elapsed = durationMs > 0 ? Math.round(durationMs * (i / steps)) : 0;
    const timeDelta = elapsed - previousElapsed;
    if (timeDelta > 0) await advanceClock(context, page, timeDelta, frameMs);
    previousElapsed = elapsed;
  }
}

async function dispatchPointerLockMouseButton(page: Page, type: "mousedown" | "mouseup"): Promise<void> {
  await page.evaluate((eventType) => {
    const target = document.pointerLockElement ?? document;
    target.dispatchEvent(new MouseEvent(eventType, {
      bubbles: true,
      cancelable: true,
      button: 0,
      buttons: eventType === "mousedown" ? 1 : 0,
    }));
  }, type);
}

async function hasPointerLock(page: Page): Promise<boolean> {
  return page.evaluate(() => Boolean(document.pointerLockElement)).catch(() => false);
}

async function currentMousePoint(page: Page): Promise<Point> {
  const viewport = page.viewportSize() ?? { width: 1440, height: 900 };
  const point = await page.evaluate(() => {
    const w = window as typeof window & { __agensMouse?: { x: number; y: number } };
    return w.__agensMouse;
  }).catch(() => undefined);
  if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) return point;
  return { x: viewport.width / 2, y: viewport.height / 2 };
}

async function rememberMousePoint(page: Page, point: Point): Promise<void> {
  await page.evaluate(({ x, y }) => {
    const w = window as typeof window & { __agensMouse?: { x: number; y: number } };
    w.__agensMouse = { x, y };
  }, point).catch(() => undefined);
}

function stepsForMouseMove(durationMs: number | undefined, frameMs: number): number {
  if (!durationMs || durationMs <= 0) return 8;
  return Math.max(1, Math.ceil(durationMs / Math.max(1, frameMs)));
}

function clampInteger(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.round(value)));
}

function normalizeDurationMs(
  requestedDurationMs: number | undefined,
  frameMs: number,
  metadata: ActionExecutionMetadata,
): number {
  if (!Number.isFinite(requestedDurationMs) || requestedDurationMs! < 0) return 0;
  const requested = Math.max(0, Math.round(requestedDurationMs!));
  const minimum = minimumExplicitDurationMs(frameMs);
  const duration = Math.max(requested, minimum);
  if (duration !== requested) {
    metadata.requestedDurationMs = requested;
  }
  return duration;
}

function minimumExplicitDurationMs(frameMs: number): number {
  return Math.max(1, Math.round(frameMs * 2 + 25));
}

function stripActionCoordinates(action: Extract<SimulationAction, { type: "click" }>): Extract<SimulationAction, { type: "click" }> {
  const { x: _x, y: _y, ...rest } = action;
  return rest;
}

function addMetadataWarning(metadata: ActionExecutionMetadata, warning: string): void {
  metadata.warnings = [...(metadata.warnings ?? []), warning];
}

async function isCanvasSelectorClickWithoutCoordinates(
  page: Page,
  action: Extract<SimulationAction, { type: "click" }>,
): Promise<boolean> {
  if (!action.selector) return false;
  if (action.text) return false;
  if (Number.isFinite(action.x) || Number.isFinite(action.y)) return false;
  return page.locator(action.selector).first().evaluate((el) => el instanceof HTMLCanvasElement).catch(() => false);
}

async function shouldIgnoreCanvasOriginCoordinates(
  page: Page,
  action: Extract<SimulationAction, { type: "click" }>,
): Promise<boolean> {
  if (action.selector !== "#game") return false;
  const hasCoordinates = Number.isFinite(action.x) || Number.isFinite(action.y);
  const hasOnlyOriginCoordinates = action.x === 0 && action.y === 0;
  if (hasCoordinates && !hasOnlyOriginCoordinates) return false;
  return page.evaluate(() => {
    const el = document.querySelector("#game");
    return Boolean(el && el instanceof HTMLCanvasElement && document.pointerLockElement === el);
  }).catch(() => false);
}

async function advanceHeldActionClock(
  context: BrowserContext,
  page: Page,
  durationMs: number,
  frameMs: number,
  onCheckpoint?: (checkpoint: HeldActionCheckpoint) => Promise<void>,
): Promise<void> {
  let elapsedMs = 0;
  let checkpointMs = 50;
  let index = 0;

  while (checkpointMs <= durationMs) {
    const deltaMs = checkpointMs - elapsedMs;
    if (deltaMs > 0) {
      await advanceClock(context, page, deltaMs, frameMs);
      elapsedMs = checkpointMs;
    }
    await onCheckpoint?.({ checkpointMs, deltaMs, index });
    checkpointMs *= 2;
    index += 1;
  }

  if (elapsedMs < durationMs) {
    await advanceClock(context, page, durationMs - elapsedMs, frameMs);
  }
}

async function resolveActionTarget(
  page: Page,
  action: SimulationAction,
  metadata: ActionExecutionMetadata,
): Promise<ResolvedActionTarget | undefined> {
  if (action.type !== "click" && action.type !== "hover") return undefined;
  if (typeof action.x === "number" && typeof action.y === "number") {
    metadata.targetResolution = {
      method: "coordinates",
      selector: action.selector,
      text: action.text,
    };
    return {
      method: "coordinates",
      point: { x: action.x, y: action.y },
    };
  }

  if (action.selector && action.text) {
    const scopedLocator = page.locator(action.selector).filter({ hasText: action.text });
    const matchCount = await scopedLocator.count();
    if (matchCount > 0) {
      const locator = scopedLocator.first();
      metadata.targetResolution = {
        method: "selector-text",
        selector: action.selector,
        text: action.text,
        matchCount,
      };
      return {
        method: "selector-text",
        locator,
        point: await locatorCenter(locator),
      };
    }

    addMetadataWarning(
      metadata,
      `selector-text-no-match: selector ${JSON.stringify(action.selector)} had no element containing text ${JSON.stringify(action.text)}; falling back to selector-only.`,
    );
    const selectorLocator = page.locator(action.selector);
    const selectorMatchCount = await selectorLocator.count();
    const locator = selectorLocator.first();
    metadata.targetResolution = {
      method: "selector",
      selector: action.selector,
      text: action.text,
      fallback: "selector-text-no-match",
      matchCount: selectorMatchCount,
    };
    if (selectorMatchCount === 0) {
      throw new Error(`No element matched selector ${JSON.stringify(action.selector)}.`);
    }
    return {
      method: "selector",
      locator,
      point: await locatorCenter(locator),
    };
  }

  if (action.selector) {
    const selectorLocator = page.locator(action.selector);
    const matchCount = await selectorLocator.count();
    const locator = selectorLocator.first();
    metadata.targetResolution = {
      method: "selector",
      selector: action.selector,
      matchCount,
    };
    if (matchCount === 0) {
      throw new Error(`No element matched selector ${JSON.stringify(action.selector)}.`);
    }
    return {
      method: "selector",
      locator,
      point: await locatorCenter(locator),
    };
  }

  if (action.text) {
    const textLocator = page.getByText(action.text);
    const matchCount = await textLocator.count();
    const locator = textLocator.first();
    metadata.targetResolution = {
      method: "text",
      text: action.text,
      matchCount,
    };
    if (matchCount === 0) {
      throw new Error(`No element matched text ${JSON.stringify(action.text)}.`);
    }
    return {
      method: "text",
      locator,
      point: await locatorCenter(locator),
    };
  }

  return undefined;
}

async function locatorCenter(locator: Locator): Promise<Point | undefined> {
  const box = await locator.boundingBox().catch(() => null);
  if (!box) return undefined;
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

async function isNativeButtonActionTarget(
  page: Page,
  action: SimulationAction,
  target?: ResolvedActionTarget,
): Promise<boolean> {
  if (action.type !== "click") return false;
  if (target?.locator) {
    return target.locator.evaluate(isNativeButtonElement).catch(() => false);
  }
  const point = target?.point;
  if (point) {
    return page.evaluate(({ x, y }) => {
      const el = document.elementFromPoint(x, y);
      if (!el) return false;
      const tag = el.tagName.toLowerCase();
      const input = el as HTMLInputElement;
      return tag === "button" || (tag === "input" && ["button", "submit", "reset"].includes((input.type || "").toLowerCase()));
    }, point).catch(() => false);
  }
  return false;
}

function isNativeButtonElement(el: Element): boolean {
  const tag = el.tagName.toLowerCase();
  const input = el as HTMLInputElement;
  return tag === "button" || (tag === "input" && ["button", "submit", "reset"].includes((input.type || "").toLowerCase()));
}

async function hitTest(page: Page, point: Point): Promise<ActionExecutionMetadata["hitTarget"]> {
  return page.evaluate(({ x, y }) => {
    const el = document.elementFromPoint(x, y);
    if (!el) return { pagePoint: { x, y } };
    const rect = (el as HTMLElement).getBoundingClientRect();
    const localPoint = { x: x - rect.left, y: y - rect.top };
    const canvas = el instanceof HTMLCanvasElement ? el : undefined;
    return {
      pagePoint: { x, y },
      hitElement: {
        tag: el.tagName.toLowerCase(),
        id: (el as HTMLElement).id || undefined,
        className: typeof (el as HTMLElement).className === "string" ? (el as HTMLElement).className || undefined : undefined,
        text: ((el as HTMLElement).innerText || (el as HTMLInputElement).value || "").trim().slice(0, 120) || undefined,
      },
      elementRect: {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      },
      localPoint,
      canvasSize: canvas ? { width: canvas.width, height: canvas.height } : undefined,
    };
  }, point);
}

export async function advanceClock(
  context: BrowserContext,
  page: Page,
  totalMs: number,
  frameMs: number,
): Promise<void> {
  let remaining = totalMs;
  while (remaining > 0) {
    const delta = Math.min(frameMs, remaining);
    await context.clock.runFor(delta);
    remaining -= delta;
    await page.waitForTimeout(Math.min(25, Math.max(5, delta)));
  }
}

export async function readActionHints(page: Page): Promise<ActionHint[]> {
  return page.evaluate(() => {
    function selectorFor(el: Element): string | undefined {
      const html = el as HTMLElement;
      if (html.id) return `#${CSS.escape(html.id)}`;
      const aria = html.getAttribute("aria-label");
      if (aria) return `[aria-label="${CSS.escape(aria)}"]`;
      return undefined;
    }

    return Array.from(document.querySelectorAll("button, [role=button], input, textarea, select, a"))
      .slice(0, 40)
      .map((el) => {
        const rect = (el as HTMLElement).getBoundingClientRect();
        return {
          tag: el.tagName.toLowerCase(),
          selectorHint: selectorFor(el),
          text: ((el as HTMLElement).innerText || (el as HTMLInputElement).value || "").trim().slice(0, 120),
          rect: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          },
        };
      });
  });
}
