/**
 * Interactive REPL. One persistent session for the whole conversation, so
 * the model keeps full context across turns. Type `:exit` (or Ctrl+C) to quit.
 *
 * Slash-style meta commands:
 *   :exit | :quit       Leave the REPL.
 *   :reset              Drop current session and start a new one.
 *   :skills             Show loaded skills.
 *   :help               Show available commands.
 */
import readline from "node:readline";
import chalk from "chalk";
import { buildSessionConfig, createClient, type HarnessConfig } from "./config.js";
import { attachLogger, type LoggerHandle } from "./logger.js";
import { disposeBrowser } from "./playwrightHost.js";

export async function startRepl(cfg: HarnessConfig): Promise<void> {
  console.log(chalk.bold.cyan("agens"), chalk.gray(`(model=${cfg.model}, playwright=${cfg.enablePlaywright ? "on" : "off"})`));
  console.log(chalk.gray("Type :help for commands, Ctrl+C to exit.\n"));

  const client = createClient(cfg);
  await client.start();

  let session = await client.createSession(buildSessionConfig(cfg));
  let logger: LoggerHandle = attachLogger(session);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: chalk.green("agens> "),
  });

  const cleanup = async () => {
    try { logger.dispose(); } catch { /* ignore */ }
    try { await session.disconnect(); } catch { /* ignore */ }
    try { await client.stop(); } catch { /* ignore */ }
    try { await disposeBrowser(); } catch { /* ignore */ }
  };

  rl.on("SIGINT", async () => {
    console.log(chalk.gray("\nbye."));
    await cleanup();
    process.exit(0);
  });

  rl.prompt();

  for await (const raw of rl) {
    const line = raw.trim();
    if (!line) { rl.prompt(); continue; }

    if (line.startsWith(":")) {
      const cmd = line.slice(1).toLowerCase();
      if (cmd === "exit" || cmd === "quit") break;
      if (cmd === "help") { printHelp(); rl.prompt(); continue; }
      if (cmd === "skills") {
        console.log(chalk.gray("Loaded skills:"), cfg.skills.length ? cfg.skills.join(", ") : "(none)");
        rl.prompt();
        continue;
      }
      if (cmd === "reset") {
        logger.dispose();
        await session.disconnect();
        session = await client.createSession(buildSessionConfig(cfg));
        logger = attachLogger(session);
        console.log(chalk.gray("(session reset)"));
        rl.prompt();
        continue;
      }
      console.log(chalk.yellow(`Unknown command: ${line}. Try :help`));
      rl.prompt();
      continue;
    }

    // Send the prompt and wait for the session to go idle before re-prompting.
    await new Promise<void>((resolve) => {
      const onIdle = () => {
        logger.endLine();
        offIdle();
        resolve();
      };
      const offIdle = session.on("session.idle", onIdle);
      session.send({ prompt: line }).catch((err: any) => {
        console.error(chalk.red("\nsend error:"), err);
        offIdle();
        resolve();
      });
    });

    rl.prompt();
  }

  await cleanup();
}

function printHelp(): void {
  console.log(`
${chalk.bold("Commands")}
  :help         Show this help
  :skills       List skills loaded in this session
  :reset        Discard the current session and start fresh
  :exit / :quit Leave agens

Anything else is sent as a prompt to the agent.
`);
}
