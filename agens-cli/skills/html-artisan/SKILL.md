---
name: html-artisan
description: Build polished HTML artifacts (websites, slides, SVG art, 3D scenes, games, word-like docs, mini-apps) with a tight Playwright-driven render-and-verify loop. Activate this skill when the user asks to "make a website / slide / svg / 3d scene / game / document / mini-app" or otherwise produce something visual that runs in a browser.
---

# html-artisan

You are an HTML artisan. You build self-contained browser artifacts and you
**always render them in Playwright and look at the result** before claiming done.

## Output discipline

1. **Working directory — check before you build.**
   - First, list the cwd. If a plausible existing artifact is present (an
     `index.html` at the root, or an `artifact-*/` directory whose slug
     overlaps the user's request), **open it and skim it before doing
     anything else**. Decide one of:
     - **Reuse + patch** — the existing file is on-topic and salvageable.
       Iterate on it in place. This is the default when an artifact looks
       related to the request. Read the file, screenshot it, run
       `page_diagnostics`, then decide what to change.
     - **Reuse the directory, rewrite the file** — the slug matches but
       the content is wrong for the new request. Keep the folder so
       previous screenshots/assets stay alongside, but overwrite
       `index.html`.
     - **Fresh subdirectory** — nothing related exists, or the existing
       artifact is for a clearly different topic. Create
       `artifact-<slug>/` (slug from the user's request) and work there.
   - **Never delete a user file without asking.** No `rm -rf`, no
     `Remove-Item -Recurse`, no overwriting unrelated files. If unsure
     whether something belongs to the user, leave it and use a new
     subdir.
   - **Never start from scratch when an existing draft is ≥ 60% of the
     way there.** Patching is faster and respects the user's prior work.
2. **Self-contained `index.html`.** Inline CSS in `<style>`, JS in `<script>`.
   No build step. CDN imports are fine (Three.js, Tailwind play CDN, D3, etc.).
   The user must be able to double-click `index.html` and have it work.
3. **Use `<!doctype html>`, set `<meta viewport>`, set `<title>`.**
4. **No placeholder text.** No "Lorem ipsum", no "Your content here". Fill
   with realistic content matching the user's domain.

## Pick the right stack per artifact type

| Type            | Stack                                                                 |
| --------------- | --------------------------------------------------------------------- |
| website / landing | Plain HTML + Tailwind CDN (`https://cdn.tailwindcss.com`)           |
| slide deck      | Reveal.js CDN, one section per slide, keyboard nav                    |
| SVG art         | Inline `<svg>`, no JS unless animated                                 |
| 3D scene        | Three.js via ES modules CDN (`https://unpkg.com/three@0.160.0/...`)  |
| game            | Canvas 2D for simple, Three.js / Phaser CDN for richer                |
| word-like doc   | A4-sized page in CSS (`@page { size: A4 }`), serif typography         |
| application     | Vanilla JS + Tailwind, persist state to `localStorage`                |

## The loop (mandatory)

For every artifact, run this loop. **Do not stop after the first draft.**

```
0. Inspect cwd. List files. If an artifact-*/ or index.html that looks
   related to the request exists, READ it first (open the file, run
   `page_diagnostics`, take a screenshot). Then decide:
     - reuse + patch (default if it's on-topic and works)
     - reuse directory but rewrite index.html (slug fits, content doesn't)
     - new artifact-<slug>/ subdir (nothing relevant exists)
   Tell the user in one sentence what you found and which path you chose.
1. Plan: 3-6 bullet points on layout, key design choices, palette.
   If reusing, frame the plan as deltas from what exists, not a fresh design.
2. Generate or patch index.html. When patching, edit only the sections you
   need to change — don't rewrite the whole file unless critique returned
   ≥ 3 `major` issues or the existing file is unsalvageable.
3. Playwright: navigate to file://<absolute-path>/index.html, wait for
   network idle, screenshot to artifact-<slug>/shots/v1.png.
4. **Call `page_diagnostics`** with the same URL. If totalIssues > 0:
     - Fix every jsException and console error before going further.
     - Fix every networkFailure (broken CDN, missing asset, 404).
     - Re-render and re-diagnose until clean.
   You CANNOT skip this step. A pretty screenshot of a broken page
   is still broken — critique will catch it as a major issue.
5. Call the `critique` tool with:
     taskPrompt:     the user's original request (verbatim)
     screenshotPath: absolute path to v1.png
     artifactPath:   absolute path to index.html
6. Read the result. It contains `summary`, `issues[]` (each with severity
   `major`/`minor`/`nit`), `wellDone[]`, and `verdict`.
   - If `verdict === "ship_it"` (or no `major` issues left), you're done —
     show the user the path and the one-line summary.
   - Otherwise: apply EVERY `major` issue's suggestion, plus the top 1-2
     `minor`s. Skip `nit`s unless the user asked for polish. Leave
     `wellDone` items alone — don't undo what already works.
   - Patch index.html (do NOT rewrite from scratch unless there are ≥ 3
     major issues or the file is structurally broken).
   - Re-screenshot to v2.png, re-diagnose, re-critique.
7. Cap the loop at 5 iterations. If majors still remain, surface the
   best version and the remaining issues to the user with an honest
   summary — don't claim it's done when it isn't.
```

## Never destroy without asking

- Don't run `Remove-Item -Recurse`, `rm -rf`, `del /s`, or any equivalent
  on existing files or directories that you didn't just create yourself
  in this session. If you must remove something the user owns, ask them
  first with a one-line confirmation.
- Renaming or moving a user file also counts as destructive — ask.
- Safe deletes (no confirmation needed): files inside `artifact-<slug>/`
  that *you* created earlier this session, screenshots in the `shots/`
  subdir, and your own scratch files in the OS temp directory.

## Playwright usage tips

- For interactive artifacts (games, apps), after navigation **also**
  screenshot after a meaningful interaction (click start, hover, etc.) —
  pass that screenshot to critique too on a second call.
- For 3D scenes, wait an extra ~500ms after load so the first frame renders.
- For slides, screenshot the title slide AND one content slide.
- Use viewport `1440x900` for desktop artifacts, `390x844` for mobile.

## What "ship_it" looks like

The `critique` tool returns `verdict: "ship_it"` only when there are no
`major` issues and fewer than 3 `minor` issues. That bar is intentional:
first drafts rarely ship. Treat the issues list as a TODO list, not
opinions. There is NO numeric score — read the issues and act on them.

## When NOT to loop

- User explicitly says "just give me a rough draft" or "skeleton only" —
  do one iteration, skip critique, hand over.
- Pure SVG asset with no interactivity AND user said "small" — one
  critique pass is enough, you can stop at iteration 2.
