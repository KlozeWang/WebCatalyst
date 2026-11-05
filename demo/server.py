#!/usr/bin/env python3
"""Dependency-free, local HTML comparison viewer. Run: python3 server.py."""

import argparse
import hashlib
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
REFERENCE = "WebCatalyst"



def extract_html(row):
    html = row.get("html")
    if isinstance(html, str) and html.strip():
        return html
    response = row.get("response") or ""
    response = response.rsplit("</think>", 1)[-1]
    start = re.search(r"<!doctype\s+html|<html\b", response, re.I)
    if not start:
        return ""
    html = response[start.start():]
    end = re.search(r"</html\s*>", html, re.I)
    return html[:end.end()] if end else html.rstrip().removesuffix("```").rstrip()


class Dataset:
    def __init__(self, root):
        files = sorted(root.glob("*/results.jsonl"))
        models = [p.parent.name for p in files]
        if REFERENCE not in models:
            raise ValueError(f"Missing reference data: {REFERENCE}/results.jsonl")
        self.models = [REFERENCE] + [m for m in models if m != REFERENCE]
        self.rows = {}
        prompts = {}
        digest = hashlib.sha256()
        for model in self.models:
            path = root / model / "results.jsonl"
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Missing prompt at {path}:{line_number}")
                key = hashlib.sha256(prompt.encode()).hexdigest()[:24]
                if (key, model) in self.rows:
                    raise ValueError(f"Duplicate prompt at {path}:{line_number}; resolve it before comparison")
                html = extract_html(row)
                self.rows[key, model] = {
                    "html": html, "status": row.get("status", "unknown"),
                    "error": str(row.get("error") or ""),
                }
                digest.update(json.dumps([model, prompt, html], ensure_ascii=False).encode())
                prompts.setdefault(key, {
                    "key": key, "id": row.get("id"), "index": len(prompts),
                    "prompt": prompt, "task": row.get("task", ""), "models": [],
                })
                if html:
                    prompts[key]["models"].append(model)
        self.prompts = list(prompts.values())
        self.by_key = prompts
        self.fingerprint = digest.hexdigest()


def handler_for(dataset, preview_port=None, games_only=False):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if not games_only:
                self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if games_only:
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "games":
                    row = dataset.rows.get((parts[1], parts[2]))
                    if row and row["html"]:
                        self.send(200, row["html"].encode(), "text/html; charset=utf-8")
                        return
                self.send(404, {"error": "Game not found"})
                return
            if path == "/api/dataset":
                self.send(200, {
                    "reference_model": REFERENCE, "models": dataset.models,
                    "dataset_id": dataset.fingerprint, "prompts": dataset.prompts,
                    "preview_port": preview_port,
                })
            elif path.startswith("/api/prompts/"):
                key = path.removeprefix("/api/prompts/")
                if key not in dataset.by_key:
                    self.send(404, {"error": "Prompt not found"})
                    return
                self.send(200, {model: {
                    "available": bool(dataset.rows.get((key, model), {}).get("html")),
                    "status": dataset.rows.get((key, model), {}).get("status", "missing"),
                    "error": dataset.rows.get((key, model), {}).get("error", ""),
                } for model in dataset.models})
            elif path in {"/", "/index.html", "/app.js", "/styles.css"}:
                file = ROOT / ("index.html" if path == "/" else path[1:])
                self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] + "; charset=utf-8")
            elif path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            else:
                self.send(404, {"error": "Not found"})

        def read_only(self):
            self.send(405, {"error": "This viewer is read-only"})

        do_PUT = read_only
        do_POST = read_only
        do_DELETE = read_only
        do_PATCH = read_only

        def log_message(self, format, *args):
            if args and str(args[1]) not in {"200", "204"}:
                super().log_message(format, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preview-port", type=int, help="Separate game origin; defaults to --port + 1")
    args = parser.parse_args()
    dataset = Dataset(ROOT)
    preview = ThreadingHTTPServer((args.host, args.preview_port if args.preview_port is not None else args.port + 1),
                                 handler_for(dataset, games_only=True))
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler_for(dataset, preview.server_port))
    except OSError:
        preview.server_close()
        raise
    threading.Thread(target=preview.serve_forever, daemon=True).start()
    print(f"Viewer:    http://{args.host}:{server.server_port}", flush=True)
    print(f"Games:     http://{args.host}:{preview.server_port} (isolated origin)", flush=True)
    print(f"Dataset:   {len(dataset.prompts)} prompts / {len(dataset.models)} models", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        preview.shutdown()
        preview.server_close()


if __name__ == "__main__":
    main()
