/**
 * `agens update` — self-update via npm. Reuses whatever package manager
 * agens is currently installed with (global npm by default). Honest about
 * what it can and cannot do:
 *
 *   - If installed globally (`npm i -g agens`): runs `npm install -g agens@latest`.
 *   - If installed from a local tarball / npm link: prints instructions
 *     and exits, rather than silently doing the wrong thing.
 *
 * We deliberately do NOT auto-update on every run — that's a footgun.
 * Users opt in by typing the command.
 */
import { spawn } from "node:child_process";
import { readFileSync, existsSync, realpathSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import chalk from "chalk";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PKG_ROOT = resolve(__dirname, "..");

interface PackageJson { version: string; name: string }

function readPkg(): PackageJson {
  return JSON.parse(readFileSync(join(PKG_ROOT, "package.json"), "utf8"));
}

/** Detect how agens got onto the machine, so we can act accordingly. */
function detectInstall(): "global-npm" | "linked" | "tarball" | "unknown" {
  try {
    // npm link installs leave a symlink in the global node_modules pointing
    // back at the source repo. realpath reveals it.
    const real = realpathSync(PKG_ROOT);
    if (real !== PKG_ROOT) return "linked";
  } catch { /* ignore */ }

  // A clean `npm i -g` install has package.json with no .git nearby.
  if (existsSync(join(PKG_ROOT, ".git"))) return "linked";

  // Heuristic: global npm prefix contains "node_modules/agens".
  if (PKG_ROOT.includes(`${"node_modules"}${"/"}agens`)) return "global-npm";

  return "unknown";
}

export async function runUpdate(): Promise<void> {
  const pkg = readPkg();
  const kind = detectInstall();

  console.log(chalk.bold(`agens ${pkg.version}`));
  console.log(chalk.gray(`Install type: ${kind}`));
  console.log("");

  if (kind === "linked") {
    console.log(chalk.yellow("This looks like a dev / linked install (npm link or local clone)."));
    console.log("Update by pulling and rebuilding:");
    console.log(chalk.cyan(`  cd ${PKG_ROOT} && git pull && npm install && npm run build`));
    return;
  }

  if (kind === "unknown") {
    console.log(chalk.yellow("Couldn't determine install method automatically."));
    console.log("If you installed from a tarball, redownload the latest .tgz and run:");
    console.log(chalk.cyan(`  npm install -g <path-to-new-agens.tgz>`));
    console.log("If you installed globally from npm, run:");
    console.log(chalk.cyan(`  npm install -g agens@latest`));
    return;
  }

  // global-npm: run the update
  console.log(chalk.cyan("→ npm install -g agens@latest"));
  await new Promise<void>((resolveP) => {
    const proc = spawn("npm", ["install", "-g", "agens@latest"], { stdio: "inherit" });
    proc.on("exit", (code) => {
      if (code === 0) {
        const after = readPkg().version;
        console.log("");
        console.log(chalk.green(`✓ Updated. ${pkg.version} → ${after}`));
      } else {
        console.log(chalk.red(`✗ npm exited with code ${code}`));
      }
      resolveP();
    });
  });
}
