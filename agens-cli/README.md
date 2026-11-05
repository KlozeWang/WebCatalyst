# agens

A multi-model, multi-modal agent harness built on the
[GitHub Copilot SDK](https://github.com/github/copilot-sdk).

agens generates HTML artifacts (websites, slides, SVG art, 3D scenes,
games, word-like docs, mini-applications), renders them with the built-in
[Playwright MCP](https://github.com/microsoft/playwright-mcp) server, and
runs a self-critique loop using an independent vision model so the
artifact actually gets *good* — not just generated.

## Install

**One-line installer** (handles Node version):

```bash
bash install.sh
```

This downloads a private Node ≥ 24 into `~/.local/node` if your system
Node is older, then installs the bundled `agens-*.tgz` globally.

**Manual** (you already have Node ≥ 24):

```bash
npm install -g ./agens-0.4.0.tgz
```

**From npm** (when published):

```bash
npm install -g agens
```

**Dev mode** (working on agens itself):

```bash
bash install.sh --link
```

## Usage

```bash
agens                              # interactive REPL
agens "build a smiling-sun SVG landing page"
agens simulate file:///abs/path/artifact-snake-game/index.html --turn-limit 20
agens trace /abs/path/to/reward-run --threshold 70
agens --skill three-d-scene "solar system, draggable camera"
agens skills                       # list built-in skills
agens config show                  # see effective config
agens --no-playwright              # disable browser MCP
agens -m gpt-5.2                   # override model for this run
```

## Built-in skills

| Skill           | Trigger keywords                            |
| --------------- | ------------------------------------------- |
| `html-artisan`  | website, landing, page, app — the umbrella |
| `slide-deck-pro`| slide, deck, pitch, presentation           |
| `three-d-scene` | 3d, scene, three.js, WebGL, shader         |
| `game-builder`  | game, arcade, puzzle, platformer           |
| `critic`        | rubric, score, evaluate, feedback           |

Skills live in `skills/<name>/SKILL.md`. Drop a new folder and it's
auto-discovered. Set `AGENS_SKILLS_DIR=/your/dir` to add a second root.

## Built-in tools (injected into every session)

- **`critique`** — sends prompt + screenshot to an independent judge model
  (`gpt-5.2` by default), returns structured rubric (`rubrics[]`,
  `overallScore`, `passed`, `suggestions[]`). Used by the iterate loop.
- **`a11y_audit`** — runs axe-core against any rendered URL, returns a
  list of accessibility violations with selectors and fix hints.
- **`user_simulation`** — freezes Playwright Clock, asks an independent
  simulation judge for the next user action, executes it with Playwright,
  waits 50ms before the after screenshot for ordinary actions, and records
  deterministic in-action screenshots for held actions. It repeats until
  `turnLimit` or the judge returns `terminate`.

## Configuration

Precedence (highest wins): **CLI flag → env var → `~/.agens/config.json` → built-in defaults**.

| What                 | env                       | flag              | config path                  |
| -------------------- | ------------------------- | ----------------- | ---------------------------- |
| Builder model        | `AGENS_MODEL`             | `-m, --model`     | `model`                      |
| Builder endpoint     | `AGENS_BASE_URL`          | `--base-url`      | `provider.baseUrl`           |
| Builder key          | `AGENS_API_KEY`           | `--api-key`       | `provider.apiKey`            |
| Turn judge model     | `AGENS_SIM_JUDGE_MODEL`   | `--judge-model`   | `judge.model`                |
| Shared simulation endpoint | `AGENS_SIM_JUDGE_BASE_URL` | `--sim-judge-base-url` | `judge.provider.baseUrl` |
| Shared simulation key | `AGENS_SIM_JUDGE_API_KEY` | `--sim-judge-api-key` | `judge.provider.apiKey`  |
| Additional simulation endpoint | `AGENS_SIM_JUDGE_BASE_URL_ADDITION` | — | — |
| Additional simulation key | `AGENS_SIM_JUDGE_API_KEY_ADDITION` | — | — |
| Additional-to-base routing weight | `AGENS_SIM_JUDGE_ROUTING_RATIO` | — | — |
| Disable simulation judge temperature | `AGENS_SIM_JUDGE_DISABLE_TEMPERATURE=true` | — | — |
| Simulation persona   | `AGENS_SIM_PERSONA_PATH`  | `simulate --persona` | `userSimulation.personaPath` |
| Simulation metrics   | `AGENS_SIM_METRICS_PATH`  | `simulate --metrics` | `userSimulation.psychometricPath` |
| Simulation default timing budget | `AGENS_SIM_STEP_MS` | `simulate --step-ms` | `userSimulation.stepMs` |
| Simulation turn limit| `AGENS_SIM_TURN_LIMIT`    | `simulate --turn-limit` | `userSimulation.turnLimit` |
| Full simulation MP4  | `AGENS_SIM_SAVE_FULL_RECORD` | `simulate --save-full-record` | `userSimulation.saveFullRecord` |
| Global eval          | `AGENS_SIM_EXTRA_GLOBAL_JUDGE_ENABLED` | `simulate --extra-global-judge` | `userSimulation.extraGlobalJudger.enabled` |
| Decision/global model | `AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL` | `--extra-global-judge-model` | `userSimulation.extraGlobalJudger.model` |

Within simulation, `AGENS_SIM_JUDGE_MODEL` is used for one combined
TransitionJudger/ScreenshotJudger call per turn, not action decisions.
`AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL` is used for both
action decisions and the final GlobalJudger call. If the latter is omitted,
decisions and global evaluation fall back to `AGENS_SIM_JUDGE_MODEL`.

When `AGENS_SIM_JUDGE_BASE_URL_ADDITION` is set, action decisions and combined
turn evaluations are randomly distributed between it and the base endpoint.
`AGENS_SIM_JUDGE_API_KEY_ADDITION` is required. The routing ratio is expressed
as additional:base weight, defaults to `1` (a 1:1 split), and accepts `0` to
disable the additional route. The base endpoint retains its configured wire API;
the additional endpoint always uses `/chat/completions`. Final global evaluation
and the separate critique tool always use their original, non-additional endpoint.
Image requests are sent unmodified normally. If an endpoint returns HTTP 400,
Agens uses Pillow (`PYTHON_BIN`, or `python3` by default) to resize and encode
the attached images as JPEG, then retries that endpoint once. Original screenshot
files are not changed. Install Pillow with `uv pip install Pillow` outside the
provided container image.
Set `AGENS_SIM_JUDGE_DISABLE_TEMPERATURE=true` to omit the `temperature`
parameter from all simulation judge requests. This is useful for models that
only accept their default temperature.

Each rollout creates a single-consumer FIFO for turn-level judge calls. After
the action decision and execution, the turn enqueues one combined transition
and screenshot task. The next decision does not receive or wait for transition
feedback, so it can overlap queued evaluation. Before GlobalJudger runs, the
queue must drain successfully. Any queued judge failure stops new work, drains
already accepted tasks, skips GlobalJudger, and fails the rollout.

The removed `--sim-judge-model`, all `--sim-screenshot-judge*` switches,
`AGENS_SIM_SCREENSHOT_JUDGE_*` variables, `userSimulation.judge`, and
`userSimulation.screenshotJudger` are no longer accepted. Use the two model
groups above instead.

```bash
agens config init                                # template at ~/.agens/config.json
agens config set provider.baseUrl https://your-provider.example/v1
agens config set provider.apiKey  YOUR_API_KEY
agens config set model            YOUR_BUILDER_MODEL
agens config set judge.model      gpt-5.2
agens config set userSimulation.extraGlobalJudger.model gpt-5.2
agens config show                                # secrets redacted
```

Simulation outputs are written under `.agens-sim/<timestamp>/` by default:
`turn-000-before.png`, `turn-000-decision.json`,
optional `turn-000-observe-*.png`, `turn-000-after.png`,
`state_trace.jsonl`, and `result.json`. Use `--turn-limit 1` to reproduce
the old single-action behavior. Add `--save-full-record` to save one
full-run `full-record.mp4` when a system ffmpeg with MP4 support is
available.

Transition and screenshot evaluation always run together in one model request
per turn and write `turn-000-transition-evaluation.json` and
`turn-000-screenshot-evaluation.json`.
Transition rewards are product-quality weighted: visual stability is weighted
highest, app feedback and simulator execution are next, and task progress is
the weakest component.
Screenshot rewards evaluate screenshot usability only: visual stability,
state visibility, readability, and a separate screenshot aesthetic reward,
not task completion. Add `--extra-global-judge` to also write
`global-evaluation.json` after the episode completes. The global evaluator
receives only structured transition
and screenshot evaluation data, action/timing metadata, and an optional
`prompt.txt` loaded from the local artifact directory; it does not receive
screenshots or source code. Its score uses visual stability 0.15, screenshot aesthetics
0.15, feedback clarity 0.10, interaction reliability 0.20, state legibility
0.10, prompt alignment 0.20, and task progress 0.10, then applies usability
caps only for severe defects, missing required content, or clear prompt
mismatch/incompleteness, such as blank pages, non-interactive pages, or a
targeted Start/Play entrypoint that prevents the main experience from
beginning. Ordinary UX weaknesses reduce reward fields but do not set a cap.
The combined turn evaluator always receives exactly two images in order:
before-action and after-action, even when they are identical. Action overlays
and held-action observe screenshots remain local artifacts and are not sent to
the turn judge. Artifact source code is sent only to the action decision model;
it is not sent to transition, screenshot, or global judges.
Held-action observe PNGs are captured while the key/mouse is still down at
50/100/200/400ms checkpoints that do not exceed `durationMs`. Actions without
`durationMs` do not produce observe PNGs; their `after.png` is captured after
a default 50ms simulated delay.
For pointer-lock/FPS games, the simulation judge can use `mouseMove` actions
with `dx/dy`, `durationMs`, and `steps` to smoothly rotate the view before
firing; canvas fire clicks should omit `x/y` so absolute mouse movement does
not disturb the camera.
For platformers and action games, the judge can use keyboard combo actions
such as `{"type":"combo","keys":["ArrowRight","Space"],"durationMs":180}` for
simultaneous input. Combos are keyboard-only and must contain 1 or 2 keys; a
judge response with more than 2 combo keys is rejected and retried.
Its transition JSON still includes `experience` guidance and a concise
`state` description of `before.png` for analysis and output compatibility.
These judge-derived fields are not passed to later action decisions.

### Offline transition code tracing

`agens trace` replays an existing retained reward run and records which
functions and blocks from `artifact/index.html` execute during high-scoring
input transitions:

```bash
agens trace /abs/path/to/run-2026-08-24T08-17-09-795Z \
  --threshold 70 \
  --json
```

The input may be the reward run directory or its concrete
`trace/<timestamp>` directory. Every original action is replayed in order so
later state is preserved. A turn is captured only when its transition score
is greater than or equal to the required threshold and its action is
`click`, `mouseMove`, `press`, or `combo`. Legacy traces replay with a 1440x900
viewport and 50ms clock frames; use `--frame-ms` when the original run used a
different frame size.

Results replace `<run>/trace-analysis/<trace-id>/` atomically and contain
`trace-result.json` plus `turn-000-code-trace.json` and
`turn-000-dom.html` for each selected turn. Code traces include function and
block counts, HTML UTF-16 and UTF-8 offsets, line/column locations, and short
source snippets. Only JavaScript that maps exactly to the main artifact HTML
is included; CDN libraries, eval code, iframes, and workers are excluded.

The command also compares each replayed frame with the original after-action
screenshot and marks attribution as degraded after visible replay drift.
Coverage means that code executed during the transition window; it is not a
strict causal proof or a token reward by itself. DOM snapshots do not contain
canvas pixels, shadow DOM, computed styles, or the JavaScript heap. Chromium,
Chrome, or Edge is required because Firefox does not expose CDP precise
coverage.

### verl reward simulation

`agens simulate-reward` is a training-oriented wrapper around
`agens simulate`. It accepts rollout HTML, writes a retained artifact
directory, forces the global judge on (the combined turn judge is always on),
and emits a
`reward-result.json` that a verl custom reward function can parse directly.
The canonical training reward is `globalEvaluation.score / 100`; transition
and screenshot scores plus their raw evaluation objects are retained for
future reward shaping.

```bash
agens simulate-reward \
  --html-file /tmp/rollout.html \
  --task "play the game and inspect whether it works" \
  --out-root .agens-reward \
  --global-step 12 \
  --prompt-id gsm8k-42 \
  --sample-id 3 \
  --rollout-id gsm8k-42-3 \
  --turn-limit 10 \
  --json
```

For Ray/verl workers, pass a JSON request instead:

```bash
agens simulate-reward --request request.json --json
agens simulate-reward --request-stdin --json < request.json
```

The request object may contain `html` or `htmlFile`, `taskPrompt`,
`outputRoot`, `globalStep`, `promptId`, `sampleId`, `rolloutId`,
`turnLimit`, `stepMs`, `frameMs`, `maxAdvanceMs`, `saveFullRecord`, and
`recordVideo`. Outputs are kept indefinitely under
`<outputRoot>/step-<globalStep>/prompt-<promptId>/sample-<sampleId>/run-<timestamp>/`.
Each run contains `artifact/index.html`, optional `artifact/prompt.txt`,
`trace/<timestamp>/result.json`, per-turn evaluation JSON files,
`global-evaluation.json`, and top-level `reward-result.json`.

`reward-result.json` includes:

```json
{
  "ok": true,
  "score": 0.87,
  "scoreSource": "globalEvaluation.score / 100",
  "globalScore": 87,
  "transitionScore": 82.5,
  "screenshotScore": 79,
  "paths": {
    "rewardResultPath": ".../reward-result.json",
    "runDir": ".../trace/2026-..."
  },
  "transitionEvaluations": [],
  "screenshotEvaluations": [],
  "globalEvaluation": {}
}
```

### Configure your endpoint

Agens does not bundle a service endpoint or API key. `agens config init`
creates a template without either value. Configure your provider before
using a custom builder endpoint:

```bash
agens config init
agens config set provider.baseUrl https://your-provider.example/v1
agens config set provider.apiKey YOUR_API_KEY
agens config set model YOUR_BUILDER_MODEL
agens config set judge.model YOUR_VISION_MODEL
```

The judge reuses the builder endpoint and credentials unless you configure
`judge.provider` separately. Critique and simulation require an explicit,
vision-capable endpoint. With no builder endpoint set, generation uses the
Copilot SDK's normal authentication.

For a separate simulation endpoint, use environment variables:

```bash
export AGENS_SIM_JUDGE_BASE_URL="https://your-judge.example/v1"
export AGENS_SIM_JUDGE_API_KEY="YOUR_JUDGE_API_KEY"
export AGENS_SIM_JUDGE_MODEL="YOUR_VISION_MODEL"
export AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL="YOUR_VISION_MODEL"
export AGENS_SIM_JUDGE_WIRE_API="completions"
```

Use `responses` instead of `completions` if your endpoint exposes the
Responses API. Omit the key only if the endpoint does not require
authentication. Select models available from your provider; the built-in
model names do not provide access to those models.

Configuration precedence is CLI flags, environment variables, config file,
then built-in defaults. Standalone `agens` does not load `.env` automatically;
source it in your shell if you keep settings there. The repository's B200
training and inference launchers load the root `.env` themselves, preserve
configured keys, and require `AGENS_SIM_JUDGE_BASE_URL`. See the
[root configuration guide](../README.md#configure-judge-access-and-local-paths).
Existing `~/.agens/config.json` files are kept by `config init`; update any
old endpoint and credentials with `config set` or edit that file directly.

## Architecture

```
┌─────────────────────────────┐     ┌──────────────────────────┐
│ Builder agent               │     │ Judge (critique tool)    │
│  Copilot SDK runtime        │ ──→ │  Independent model call  │
│  + skills (html-artisan…)   │     │  Vision input + JSON     │
│  + Playwright MCP           │     │  schema output           │
│  + tools (critique, a11y)   │     └──────────────────────────┘
└─────────────────────────────┘
        ↑                                   ↓
        └──── rubric + suggestions ─────────┘
                  iterate ≤ 5×
```

## Repo layout

```
agens/
├── src/
│   ├── cli.ts             # commander entrypoint + subcommands
│   ├── config.ts          # SessionConfig builder (Playwright MCP, tools, skills)
│   ├── userconfig.ts      # 3-layer config: file + env + defaults
│   ├── simulation/        # user simulation engine (Clock, actions, screenshots, optional MP4)
│   ├── repl.ts            # interactive mode
│   ├── oneshot.ts         # `agens "prompt"` one-shot mode
│   ├── skills.ts          # skill discovery + resolution
│   └── tools/
│       ├── critique.ts    # screenshot → judge model → rubric
│       ├── user_simulation.ts # multi-turn user simulation + transition capture
│       └── a11y.ts        # axe-core via Playwright
├── skills/
│   ├── html-artisan/
│   ├── critic/
│   ├── slide-deck-pro/
│   ├── three-d-scene/
│   └── game-builder/
├── install.sh             # one-line installer (Node + tarball)
├── package.json
└── README.md
```

## Requirements

- Node ≥ 24 (Copilot SDK runtime uses `node:sqlite`). Use `install.sh`
  if your distro is behind.
- Chromium (auto-installed by Playwright MCP on first run, ~120 MB).

## License

ISC
