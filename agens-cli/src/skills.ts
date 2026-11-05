/**
 * Skill resolution. Skills are folders containing a SKILL.md file. We look
 * for them in two locations, in this order:
 *
 *   1. <package-root>/skills/<name>       (built-in, shipped with agens)
 *   2. $AGENS_SKILLS_DIR/<name>           (user override, optional)
 *
 * `agens --skill foo` then resolves to the matching directory, which is
 * passed through to the SDK's `skillDirectories` option.
 */
import { existsSync, readdirSync, statSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
// dist/skills.js -> package root is one up from dist/
const PACKAGE_ROOT = resolve(__dirname, "..");
const BUILTIN_SKILLS_DIR = join(PACKAGE_ROOT, "skills");
const USER_SKILLS_DIR = process.env.AGENS_SKILLS_DIR;

export interface SkillInfo {
  name: string;
  path: string;
}

function isSkillDir(p: string): boolean {
  try {
    return statSync(p).isDirectory() && existsSync(join(p, "SKILL.md"));
  } catch {
    return false;
  }
}

function scan(dir: string): SkillInfo[] {
  if (!existsSync(dir)) return [];
  const out: SkillInfo[] = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (isSkillDir(full)) out.push({ name, path: full });
  }
  return out;
}

export function listSkills(): SkillInfo[] {
  const seen = new Map<string, SkillInfo>();
  // User skills win (so users can shadow built-ins)
  for (const s of scan(BUILTIN_SKILLS_DIR)) seen.set(s.name, s);
  if (USER_SKILLS_DIR) {
    for (const s of scan(USER_SKILLS_DIR)) seen.set(s.name, s);
  }
  return [...seen.values()].sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Turn a list of requested skill names into absolute directory paths,
 * throwing if any name is unknown so the user finds out at startup
 * instead of mid-session.
 */
export function resolveSkillDirs(names: string[]): string[] {
  if (names.length === 0) return [];
  const all = new Map(listSkills().map((s) => [s.name, s.path]));
  const out: string[] = [];
  for (const n of names) {
    const p = all.get(n);
    if (!p) {
      throw new Error(
        `Unknown skill: "${n}". Run \`agens skills\` to see what's available.`
      );
    }
    out.push(p);
  }
  return out;
}
