import { execFile } from "node:child_process";
import { unlink } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import type { VideoSegment } from "./types.js";

const execFileAsync = promisify(execFile);

export async function processSimulationVideos(args: {
  sourceVideoPath: string;
  saveFullRecord: boolean;
  runDir: string;
  videoStartedAtMs?: number;
  activeSegments?: VideoSegment[];
}): Promise<{ fullRecordPath?: string; fullRecordError?: string; errors: string[] }> {
  const errors: string[] = [];

  let fullRecordPath: string | undefined;
  let fullRecordError: string | undefined;
  if (args.saveFullRecord) {
    const systemFfmpeg = await findFfmpeg();
    if (!systemFfmpeg) {
      fullRecordError = "save_full_record requires a system ffmpeg with MP4 support; none was found on PATH or FFMPEG_PATH.";
    } else {
      fullRecordPath = path.join(args.runDir, "full-record.mp4");
      try {
        await convertPlaywrightVideoToMp4(systemFfmpeg, {
          inputPath: args.sourceVideoPath,
          outputPath: fullRecordPath,
          videoStartedAtMs: args.videoStartedAtMs,
          activeSegments: args.activeSegments,
        });
      } catch (err) {
        fullRecordError = `Failed to convert full recording to mp4: ${(err as Error).message}`;
        fullRecordPath = undefined;
      }
    }
  }

  try {
    await unlink(args.sourceVideoPath);
  } catch (err) {
    errors.push(`Failed to remove temporary Playwright recording: ${(err as Error).message}`);
  }

  return { fullRecordPath, fullRecordError, errors };
}

async function convertPlaywrightVideoToMp4(ffmpegPath: string, args: {
  inputPath: string;
  outputPath: string;
  videoStartedAtMs?: number;
  activeSegments?: VideoSegment[];
}): Promise<void> {
  const segments = normalizeSegments(args.activeSegments ?? [], args.videoStartedAtMs);
  if (!segments.length) {
    await runFfmpeg(ffmpegPath, [
      "-i", args.inputPath,
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      "-y",
      args.outputPath,
    ]);
    return;
  }

  const filterParts = segments.map((segment, index) =>
    `[0:v]trim=start=${formatSeconds(segment.startSec)}:end=${formatSeconds(segment.endSec)},setpts=PTS-STARTPTS[v${index}]`);
  const concatInputs = segments.map((_, index) => `[v${index}]`).join("");
  const filter = `${filterParts.join(";")};${concatInputs}concat=n=${segments.length}:v=1:a=0[outv]`;

  await runFfmpeg(ffmpegPath, [
    "-i", args.inputPath,
    "-filter_complex", filter,
    "-map", "[outv]",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-y",
    args.outputPath,
  ]);
}

function normalizeSegments(
  segments: VideoSegment[],
  videoStartedAtMs: number | undefined,
): Array<{ startSec: number; endSec: number }> {
  if (!videoStartedAtMs) return [];
  const defaultBufferMs = 600;
  const defaultMinDurationMs = 250;
  const normalized = segments
    .filter((segment) => Number.isFinite(segment.startMs) && Number.isFinite(segment.endMs))
    .map((segment) => {
      const preBufferMs = segment.preBufferMs ?? defaultBufferMs;
      const postBufferMs = segment.postBufferMs ?? defaultBufferMs;
      const minDurationMs = segment.minDurationMs ?? defaultMinDurationMs;
      return {
        startMs: Math.max(videoStartedAtMs, segment.startMs - preBufferMs),
        endMs: Math.max(segment.startMs + minDurationMs, segment.endMs + postBufferMs),
      };
    })
    .filter((segment) => segment.endMs > videoStartedAtMs && segment.endMs > segment.startMs)
    .sort((a, b) => a.startMs - b.startMs);

  const merged: Array<{ startMs: number; endMs: number }> = [];
  for (const segment of normalized) {
    const previous = merged.at(-1);
    if (previous && segment.startMs <= previous.endMs + 100) {
      previous.endMs = Math.max(previous.endMs, segment.endMs);
    } else {
      merged.push({ ...segment });
    }
  }

  return merged.map((segment) => ({
    startSec: Math.max(0, (segment.startMs - videoStartedAtMs) / 1000),
    endSec: Math.max(0.001, (segment.endMs - videoStartedAtMs) / 1000),
  })).filter((segment) => segment.endSec > segment.startSec);
}

function formatSeconds(value: number): string {
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, ".0");
}

async function runFfmpeg(ffmpegPath: string, args: string[]): Promise<void> {
  try {
    await execFileAsync(ffmpegPath, args, { maxBuffer: 2_000_000 });
  } catch (err: any) {
    const stderr = typeof err?.stderr === "string" ? err.stderr.trim() : "";
    throw new Error(stderr || err?.message || "ffmpeg failed");
  }
}

async function findFfmpeg(): Promise<string | undefined> {
  const candidates = [
    process.env.FFMPEG_PATH,
    "ffmpeg",
  ].filter((x): x is string => !!x);

  for (const candidate of candidates) {
    try {
      await execFileAsync(candidate, ["-version"], { maxBuffer: 200_000 });
      return candidate;
    } catch {
      // Try the next candidate.
    }
  }
  return undefined;
}
