# WebCatalyst Interactive Demo

This directory provides a read-only gallery for browsing and comparing
interactive HTML games generated from 100 game designing prompts.

## Included Models

| Directory | Model |
| --- | --- |
| `WebCatalyst/` | WebCatalyst |
| `Qwen3.6-35B-A3B/` | Qwen3.6-35B-A3B baseline |
| `claude-opus-4-7/` | Claude Opus 4.7 |
| `gemini-3.1-pro-preview/` | Gemini 3.1 Pro Preview |
| `gpt-5.3-codex/` | GPT-5.3 Codex |
| `gpt-5.4-mini/` | GPT-5.4 Mini |

Each directory contains a `results.jsonl` file with 100 prompts and their
generated HTML. WebCatalyst is the fixed reference shown on the left side of
the comparison viewer.

## Run Locally

Python 3.9 or later is required. The server uses only the Python standard
library.

From the repository root:

```bash
python3 demo/server.py
```

Open <http://127.0.0.1:8765>. The game preview server starts automatically on
<http://127.0.0.1:8766>.

To use different ports:

```bash
python3 demo/server.py --port 8800 --preview-port 8801
```

The preview port defaults to the viewer port plus one. Both ports must be
available and different.

## Browse the Gallery

- Search prompts by text, ID, or sequence number.
- Select a prompt from the dropdown or use the previous and next buttons.
- Compare WebCatalyst with the other five models.
- Click inside a preview to interact with the generated game.
- Use `⤢` to open WebCatalyst and the selected model side by side.
- Switch among desktop, widescreen, and mobile viewport sizes.
- Adjust enlarged previews from 50% to 150%, fit them to the window, or enter
  full screen.
- Use `↻` to restart a game and `Esc` to close the enlarged view.

The browser stores the last selected prompt as a local navigation preference.

## Data Loading

The server reads `results.jsonl` from each model directory and ignores
`results.legacy.jsonl`. It pairs outputs by the full prompt text and preserves
the prompt order from WebCatalyst.

The viewer uses each row's `html` field. If that field is absent, it extracts
HTML from `response`. Missing outputs are displayed as unavailable previews.
Duplicate prompts stop loading with an explicit error to prevent incorrect
pairing.

The current dataset contains:

- 100 shared prompts
- 6 models
- 600 result records
- 500 WebCatalyst-versus-comparison pairs

## Isolation and Network Access

The viewer and generated games use separate local origins. Games run in
sandboxed iframes with script, storage, and pointer-lock access, while generated
scripts cannot access the viewer.

Original game HTML is served unchanged. Games that reference external fonts,
scripts, images, or other assets require network access.

Both servers are read-only. Requests that attempt to create, modify, or delete
data return HTTP 405.

## Validation

Run the checks from the `demo/` directory:

```bash
python3 -B -m unittest -v test_server.py
node --check app.js
```

The tests cover prompt pairing, duplicate detection, HTML extraction, dataset
and preview endpoints, model ordering, and rejection of write requests.
