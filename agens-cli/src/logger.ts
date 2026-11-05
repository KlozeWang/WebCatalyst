/**
 * Pretty event logger for agens.
 *
 * Goal: make a streaming agent session feel *legible*. The user should be able
 * to glance at the terminal and know exactly what the agent is doing right
 * now (which tool, with what argument, for how long) without drowning in raw
 * SDK events.
 *
 * Format conventions:
 *   ◆  intent / plan / reasoning (cyan)
 *   →  tool call in flight       (blue)
 *   ✔  tool call ok              (green, with duration)
 *   ✗  tool call failed          (red, with error)
 *   ·  ambient info              (gray)
 *
 * Assistant text streams inline with no prefix so it reads like prose. We
 * track whether the cursor is mid-line so tool lines never collide with
 * assistant text mid-token.
 */
import chalk from "chalk";

type AnySession = {
  on(eventType: string, handler: (e: any) => void): () => void;
};

export interface LoggerOptions {
  /** Stream that receives assistant text. Defaults to stdout. */
  out?: NodeJS.WriteStream;
  /** Stream that receives status/tool lines. Defaults to stderr. */
  status?: NodeJS.WriteStream;
  /** When true, also show the model's chain-of-thought (`assistant.reasoning`). */
  showReasoning?: boolean;
}

export interface LoggerHandle {
  /** Force a newline if we're mid-stream (e.g. before printing a final summary). */
  endLine(): void;
  /** Print a one-line summary of the run (duration, tool count, turn count). */
  printSummary(): void;
  /** Detach all event handlers. */
  dispose(): void;
}

interface ActiveTool {
  name: string;
  display: string;
  startedAt: number;
}

const ICON = {
  intent: "◆",
  toolRun: "→",
  toolOk: "✔",
  toolErr: "✗",
  info: "·",
};

export function attachLogger(session: AnySession, opts: LoggerOptions = {}): LoggerHandle {
  const out = opts.out ?? process.stdout;
  const status = opts.status ?? process.stderr;
  const isTTY = !!(status as any).isTTY;
  const startedAt = Date.now();

  let midLine = false; // are we in the middle of an assistant text line?
  let sawAnyDelta = false;
  const active = new Map<string, ActiveTool>();
  let toolCount = 0;
  let turnCount = 0;
  let lastIntent = "";

  function ensureNewline(stream: NodeJS.WriteStream): void {
    if (midLine) {
      stream.write("\n");
      midLine = false;
    }
  }

  function writeStatus(line: string): void {
    ensureNewline(status);
    status.write(line + "\n");
  }

  const offs: Array<() => void> = [];
  const on = (type: string, fn: (e: any) => void) => {
    offs.push(session.on(type, fn));
  };

  // --- assistant text streaming ---
  on("assistant.message_delta", (e: any) => {
    const delta: string = e?.data?.deltaContent ?? "";
    if (!delta) return;
    sawAnyDelta = true;
    out.write(delta);
    midLine = !delta.endsWith("\n");
  });

  // Fallback for providers that don't stream: render the final consolidated
  // message only when no delta arrived.
  on("assistant.message", (e: any) => {
    if (sawAnyDelta) return;
    const content = e?.data?.content;
    if (typeof content === "string") {
      out.write(content);
      midLine = !content.endsWith("\n");
    } else if (Array.isArray(content)) {
      for (const part of content) {
        const text = typeof part === "string" ? part : part?.text;
        if (text) {
          out.write(text);
          midLine = !text.endsWith("\n");
        }
      }
    }
  });

  // --- turn boundaries ---
  on("assistant.turn_start", () => {
    turnCount += 1;
    if (turnCount > 1 && isTTY) {
      writeStatus(chalk.gray(`── turn ${turnCount} ──`));
    }
  });

  // --- model's stated intent (concise, human-friendly) ---
  on("assistant.intent", (e: any) => {
    const intent: string = (e?.data?.intent ?? "").trim();
    if (!intent || intent === lastIntent) return;
    lastIntent = intent;
    writeStatus(chalk.cyan(`${ICON.intent} ${truncate(intent, 200)}`));
  });

  // --- optional chain-of-thought ---
  if (opts.showReasoning) {
    on("assistant.reasoning", (e: any) => {
      const text: string = (e?.data?.content ?? "").trim();
      if (!text) return;
      writeStatus(chalk.gray.italic(`${ICON.info} ${truncate(text, 400)}`));
    });
  }

  // --- tool lifecycle ---
  on("tool.execution_start", (e: any) => {
    const d = e?.data ?? {};
    const toolCallId: string = d.toolCallId ?? `${Math.random()}`;
    const name = displayToolName(d);
    const summary = summarizeArgs(name, d.arguments);
    const display = summary ? `${chalk.bold(name)} ${chalk.gray(summary)}` : chalk.bold(name);
    active.set(toolCallId, { name, display, startedAt: Date.now() });
    toolCount += 1;
    writeStatus(`${chalk.blue(ICON.toolRun)} ${display}`);
  });

  on("tool.execution_progress", (e: any) => {
    const d = e?.data ?? {};
    const msg: string = (d.progressMessage ?? "").trim();
    if (!msg) return;
    const callId: string | undefined = d.toolCallId;
    const tool = callId ? active.get(callId) : undefined;
    const prefix = tool ? `${tool.name}: ` : "";
    writeStatus(chalk.gray(`  ${ICON.info} ${prefix}${truncate(msg, 160)}`));
  });

  on("tool.execution_complete", (e: any) => {
    const d = e?.data ?? {};
    const toolCallId: string = d.toolCallId;
    const tool = active.get(toolCallId);
    active.delete(toolCallId);

    const ms = tool ? Date.now() - tool.startedAt : undefined;
    const dur = ms !== undefined ? chalk.gray(` ${formatDuration(ms)}`) : "";
    const display = tool?.display ?? chalk.bold(displayToolName(d));

    if (d.success === false || d.error) {
      const errMsg = d.error?.message ? truncate(String(d.error.message), 200) : "failed";
      writeStatus(`${chalk.red(ICON.toolErr)} ${display}${dur} ${chalk.red(errMsg)}`);
      return;
    }
    writeStatus(`${chalk.green(ICON.toolOk)} ${display}${dur}`);
  });

  // --- session-level info/warning/error ---
  on("session.warning", (e: any) => {
    const msg = String(e?.data?.message ?? "").trim();
    if (msg) writeStatus(chalk.yellow(`! ${truncate(msg, 240)}`));
  });
  on("session.error", (e: any) => {
    const msg = String(e?.data?.message ?? e?.data?.errorCode ?? "session error").trim();
    writeStatus(chalk.red(`✗ ${truncate(msg, 240)}`));
  });

  return {
    endLine() {
      ensureNewline(out);
    },
    printSummary() {
      ensureNewline(out);
      const elapsed = formatDuration(Date.now() - startedAt);
      const parts = [
        chalk.gray(`Done in ${elapsed}`),
        chalk.gray(`${toolCount} tool${toolCount === 1 ? "" : "s"}`),
        chalk.gray(`${turnCount || 1} turn${(turnCount || 1) === 1 ? "" : "s"}`),
      ];
      status.write(parts.join(chalk.gray(" · ")) + "\n");
    },
    dispose() {
      for (const off of offs) {
        try { off(); } catch { /* ignore */ }
      }
      offs.length = 0;
    },
  };
}

// ---------- helpers ----------

function displayToolName(d: any): string {
  const mcp = d?.mcpServerName;
  const mcpTool = d?.mcpToolName ?? d?.toolName;
  if (mcp) return `${mcp}.${mcpTool}`;
  return d?.toolName ?? "tool";
}

/**
 * Pull the most-interesting arg(s) out of a tool invocation so the user
 * sees `create index.html` instead of just `create`.
 */
function summarizeArgs(toolName: string, args: Record<string, unknown> | undefined): string {
  if (!args || typeof args !== "object") return "";
  const lname = toolName.toLowerCase();

  // shell/terminal style tools
  if (/(powershell|bash|shell|terminal|run.?in.?terminal|exec)/.test(lname)) {
    const cmd = pickString(args, ["command", "cmd", "script", "input"]);
    if (cmd) return oneLine(cmd, 100);
  }

  // file write/edit/read
  if (/(create|write|edit|update|replace|read|view|delete|cat)/.test(lname)) {
    const p = pickString(args, [
      "filePath", "file_path", "path", "file", "filename",
      "target_file", "uri",
    ]);
    if (p) {
      const size = typeof args.content === "string" ? ` (${humanBytes(Buffer.byteLength(args.content))})` : "";
      return shortPath(p) + size;
    }
  }

  // browser / playwright tools
  if (/(navigate|goto|open.?browser|browser_navigate|fetch)/.test(lname)) {
    const url = pickString(args, ["url", "uri", "href"]);
    if (url) return truncate(url, 100);
  }
  if (/(click|hover|type|fill|press)/.test(lname)) {
    const target = pickString(args, ["selector", "element", "text", "ref"]);
    if (target) return truncate(target, 80);
  }
  if (/screenshot/.test(lname)) {
    const p = pickString(args, ["path", "filename", "filePath"]);
    return p ? shortPath(p) : "";
  }

  // intent / planning tools
  if (/(report_intent|intent|plan)/.test(lname)) {
    const t = pickString(args, ["intent", "plan", "summary", "message", "text"]);
    if (t) return oneLine(t, 140);
  }

  // skill loader
  if (/skill/.test(lname)) {
    const t = pickString(args, ["name", "skill", "skillName", "id"]);
    if (t) return t;
  }

  // search / grep
  if (/(grep|search|find)/.test(lname)) {
    const q = pickString(args, ["query", "pattern", "q", "search"]);
    if (q) return `"${oneLine(q, 80)}"`;
  }

  // Fallback: first short string-valued arg.
  for (const [k, v] of Object.entries(args)) {
    if (typeof v === "string" && v.length > 0 && v.length <= 120) {
      return `${k}=${oneLine(v, 100)}`;
    }
  }
  return "";
}

function pickString(args: Record<string, unknown>, keys: string[]): string | undefined {
  for (const k of keys) {
    const v = args[k];
    if (typeof v === "string" && v.trim()) return v;
  }
  return undefined;
}

function oneLine(s: string, max: number): string {
  return truncate(s.replace(/\s+/g, " ").trim(), max);
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, Math.max(1, max - 1)) + "…";
}

function shortPath(p: string): string {
  // Strip common cwd-ish prefixes for readability.
  const norm = p.replace(/\\/g, "/");
  const cwd = process.cwd().replace(/\\/g, "/");
  if (norm.startsWith(cwd + "/")) return norm.slice(cwd.length + 1);
  return norm;
}

function humanBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s - m * 60);
  return `${m}m${r.toString().padStart(2, "0")}s`;
}
