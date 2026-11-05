/**
 * One-shot mode: `agens "do X"` — send a single prompt, stream the response,
 * then exit. No REPL, no history.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import chalk from "chalk";
import { buildSessionConfig, createClient, type HarnessConfig } from "./config.js";
import { attachLogger } from "./logger.js";
import { disposeBrowser } from "./playwrightHost.js";

export interface OneShotOptions {
  /** When set, write a JSON trajectory (prompt, response, events, base64 images) to this file. */
  trajectoryPath?: string;
  /** Print the model's chain-of-thought (`assistant.reasoning`) inline. */
  showReasoning?: boolean;
}

export async function runOneShot(
  prompt: string,
  cfg: HarnessConfig,
  opts: OneShotOptions = {},
): Promise<void> {
  const client = createClient(cfg);
  await client.start();
  const startedAt = new Date().toISOString();
  let responseText = "";
  try {
    const session = await client.createSession(buildSessionConfig(cfg));
    process.stderr.write(
      chalk.bold.cyan("agens » ") +
        chalk.white(prompt.length > 200 ? prompt.slice(0, 199) + "…" : prompt) +
        "\n" +
        chalk.gray(
          `(model=${cfg.model}, playwright=${cfg.enablePlaywright ? "on" : "off"}` +
            (cfg.skills.length ? `, skills=${cfg.skills.join(",")}` : "") +
            `)\n\n`,
        ),
    );
    const logger = attachLogger(session, { showReasoning: opts.showReasoning });
    try {
      // Capture response text for trajectory writing without double-rendering.
      session.on("assistant.message_delta", (e: any) => {
        responseText += e?.data?.deltaContent ?? "";
      });
      session.on("assistant.message", (e: any) => {
        const content = e?.data?.content;
        if (typeof content === "string") responseText += content;
        else if (Array.isArray(content)) {
          for (const part of content) {
            if (typeof part === "string") responseText += part;
            else if (part?.text) responseText += part.text;
          }
        }
      });

      await new Promise<void>((resolve, reject) => {
        session.on("session.idle", () => resolve());
        session.on("error" as any, (e: any) => reject(e?.data ?? e));
        session.send({ prompt }).catch(reject);
      });
      logger.endLine();
      logger.printSummary();

      if (opts.trajectoryPath) {
        let events: unknown[] = [];
        try {
          events = await session.getEvents();
        } catch (err) {
          console.error(
            chalk.yellow("warning: failed to retrieve session events for --trajectory:"),
            err instanceof Error ? err.message : err,
          );
        }
        const payload = {
          schemaVersion: 1,
          startedAt,
          finishedAt: new Date().toISOString(),
          sessionId: (session as any).sessionId,
          config: {
            model: cfg.model,
            skills: cfg.skills,
            enablePlaywright: cfg.enablePlaywright,
            cwd: cfg.cwd,
            provider: cfg.provider?.baseUrl
              ? { type: cfg.provider.type ?? "openai", baseUrl: cfg.provider.baseUrl }
              : undefined,
            judge: cfg.judge
              ? {
                  model: cfg.judge.model,
                  provider: cfg.judge.provider?.baseUrl
                    ? {
                        type: cfg.judge.provider.type ?? "openai",
                        baseUrl: cfg.judge.provider.baseUrl,
                      }
                    : undefined,
                }
              : undefined,
            userSimulation: cfg.userSimulation
              ? {
                  turnJudge: {
                    model: cfg.userSimulation.turnJudge.model,
                    provider: cfg.userSimulation.turnJudge.provider?.baseUrl
                      ? {
                          type: cfg.userSimulation.turnJudge.provider.type ?? "openai",
                          baseUrl: cfg.userSimulation.turnJudge.provider.baseUrl,
                        }
                      : undefined,
                  },
                  decisionJudge: {
                    model: cfg.userSimulation.decisionJudge.model,
                    provider: cfg.userSimulation.decisionJudge.provider?.baseUrl
                      ? {
                          type: cfg.userSimulation.decisionJudge.provider.type ?? "openai",
                          baseUrl: cfg.userSimulation.decisionJudge.provider.baseUrl,
                        }
                      : undefined,
                  },
                  personaPath: cfg.userSimulation.personaPath,
                  psychometricPath: cfg.userSimulation.psychometricPath,
                  outputDir: cfg.userSimulation.outputDir,
                  stepMs: cfg.userSimulation.stepMs,
                  frameMs: cfg.userSimulation.frameMs,
                  turnLimit: cfg.userSimulation.turnLimit,
                  recordVideo: cfg.userSimulation.recordVideo,
                  saveFullRecord: cfg.userSimulation.saveFullRecord,
                }
              : undefined,
          },
          prompt,
          response: responseText,
          events,
        };
        // Resolve relative to the user's shell cwd, NOT cfg.cwd — the latter
        // is the agent's workspace and shouldn't capture user-facing paths.
        const outPath = path.resolve(process.cwd(), opts.trajectoryPath);
        await fs.mkdir(path.dirname(outPath), { recursive: true });
        await fs.writeFile(outPath, JSON.stringify(payload, null, 2), "utf8");
        console.error(chalk.gray(`\n[trajectory written to ${outPath}]`));
      }
    } finally {
      logger.dispose();
      await session.disconnect();
    }
  } catch (err) {
    console.error(chalk.red("Session error:"), err);
    process.exitCode = 1;
  } finally {
    await client.stop();
    await disposeBrowser();
  }
}
