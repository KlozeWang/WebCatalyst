import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type { LoadedText, UserSimulationTurn } from "./types.js";

export async function createRunDir(cwd: string, outputDir: string): Promise<string> {
  const base = path.resolve(cwd, outputDir);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runDir = path.join(base, stamp);
  await mkdir(runDir, { recursive: true });
  return runDir;
}

export function normalizeUrl(raw: string, cwd: string): string {
  if (/^[a-zA-Z][a-zA-Z\d+.-]*:/.test(raw)) return raw;
  return pathToFileURL(path.resolve(cwd, raw)).href;
}

export async function loadOptionalText(
  label: string,
  maybePath: string | undefined,
  cwd: string,
  maxChars = 20_000,
  required = true,
): Promise<LoadedText | undefined> {
  if (!maybePath) return undefined;
  const resolved = path.resolve(cwd, maybePath);
  try {
    const src = await readFile(resolved, "utf8");
    const truncated = src.length > maxChars;
    return {
      path: resolved,
      text: truncated ? src.slice(0, maxChars) + "\n...[truncated]" : src,
      truncated,
    };
  } catch (e) {
    if (!required) return undefined;
    throw new Error(`Failed to read ${label} file at ${resolved}: ${(e as Error).message}`);
  }
}

export async function writeJson(filePath: string, data: unknown): Promise<void> {
  await writeFile(filePath, JSON.stringify(data, null, 2) + "\n", "utf8");
}

export function summarizeTurns(turns: UserSimulationTurn[]): Array<Record<string, unknown>> {
  return turns.slice(-10).map((turn) => ({
    turnIndex: turn.turnIndex,
    turnLabel: turnLabel(turn.turnIndex),
    summary: turn.decision.summary,
    action: turn.action,
    reachedGoal: turn.reachedGoal,
    finalObservation: turn.finalObservation?.observation,
    afterLabel: labelFromPath(turn.afterScreenshotPath),
    terminated: turn.terminated,
    terminationReason: turn.terminationReason,
  }));
}

export function turnLabel(turnIndex: number): string {
  return `turn-${String(turnIndex).padStart(3, "0")}`;
}

function labelFromPath(filePath: string): string {
  return path.basename(filePath, path.extname(filePath));
}

export function normalizeMaybeMs(msValue: unknown, legacySecondsValue?: unknown): number | undefined {
  if (Number.isFinite(msValue)) return Math.round(Number(msValue));
  if (Number.isFinite(legacySecondsValue)) return Math.round(clampSeconds(legacySecondsValue, 0, 10) * 1000);
  return undefined;
}

export function clampSeconds(value: unknown, min: number, max: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

export function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

export function clampMs(value: number, min: number, max: number, name: string): number {
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${name} must be between ${min} and ${max} milliseconds.`);
  }
  return Math.floor(value);
}

export function clampCount(value: number, min: number, max: number, name: string): number {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer between ${min} and ${max}.`);
  }
  return value;
}
