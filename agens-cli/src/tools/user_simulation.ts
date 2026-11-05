import { defineTool } from "@github/copilot-sdk";
import { z } from "zod";
import { runUserSimulation, type UserSimulationRuntime } from "../simulation/engine.js";

const parameters = z.object({
  url: z.string().describe("URL to load. For local artifacts use file:///abs/path/index.html."),
  taskPrompt: z.string().optional().describe("Task or scenario for the simulated user."),
  artifactPath: z.string().optional().describe("Optional artifact source path to include in the simulation judge context."),
  includeCode: z.boolean().optional().describe("Include local artifact source code in simulation judge context. Uses artifactPath when provided, otherwise resolves local file URLs/paths."),
  personaPath: z.string().optional().describe("Optional Markdown persona file. Overrides config."),
  psychometricPath: z.string().optional().describe("Optional Markdown psychometric metrics/guidelines file. Overrides config."),
  outputDir: z.string().optional().describe("Directory where screenshots, optional MP4 recording, decision, and trace files should be written."),
  stepMs: z.number().int().min(0).max(60_000).optional().describe("Default timing budget used when maxAdvanceMs is omitted."),
  frameMs: z.number().int().min(1).max(1_000).optional().describe("Clock chunk size used while advancing held actions and deterministic observations."),
  turnLimit: z.number().int().min(1).max(1_000).optional().describe("Max simulated user turns before stopping."),
  recordVideo: z.boolean().optional().describe("Whether to record Playwright video when saveFullRecord is enabled."),
  saveFullRecord: z.boolean().optional().describe("Whether to also save a full-run MP4 recording."),
  save_full_record: z.boolean().optional().describe("Alias for saveFullRecord."),
  maxAdvanceMs: z.number().int().min(0).max(120_000).optional().describe("Max simulated milliseconds for the default after-action delay."),
});

export function makeUserSimulationTool(runtime: UserSimulationRuntime) {
  return defineTool("user_simulation", {
    description:
      "Simulate a multi-step user play episode on a rendered page. The tool freezes Playwright Clock while a " +
      "simulation judge chooses each action, executes actions through Playwright, advances page time with " +
      "a default 50ms after-action delay plus deterministic in-action observations for held actions, and repeats until turnLimit or a terminate action. It returns " +
      "per-turn screenshots, decisions, observations, a state trace, and optionally " +
      "a full-run mp4 recording. TransitionJudger evaluates the before/observe/after image sequence; " +
      "when explicitly enabled, ScreenshotJudger also evaluates each turn's after screenshot. When configured, " +
      "ExtraGlobalJudger summarizes the full run from structured transition/screenshot evaluations. FPS games can use " +
      "relative mouseMove actions for smooth view control. Use this for " +
      "timing-sensitive artifacts such as canvas games.",
    parameters,
    skipPermission: true,
    handler: async (args) => {
      try {
        return await runUserSimulation({
          ...args,
          saveFullRecord: args.saveFullRecord ?? args.save_full_record,
        }, runtime);
      } catch (e) {
        return `ERROR: user_simulation failed: ${(e as Error).message}`;
      }
    },
  });
}
