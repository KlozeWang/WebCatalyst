# Copyright 2026 WebCatalyst Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core viewer checks: python3 -B -m unittest -v test_server.py."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from server import REFERENCE, Dataset, extract_html, handler_for


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Row order and IDs differ; only prompt text can pair the games correctly.
        for model, prompts in [(REFERENCE, ["A", "B"]), ("opponent", ["B", "A"])]:
            folder = self.root / model
            folder.mkdir()
            rows = [
                {"prompt": prompt, "id": index, "html": f"<html>{model}-{prompt}</html>"}
                for index, prompt in enumerate(prompts)
            ]
            (folder / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            (folder / "results.legacy.jsonl").write_text("invalid legacy data")
        self.dataset = Dataset(self.root)
        self.key = self.dataset.prompts[0]["key"]

    def request(self, path, method="GET", games_only=False):
        # Exercise the real HTTP handler without opening a listening socket.
        handler = handler_for(self.dataset, preview_port=8766, games_only=games_only)
        instance = object.__new__(handler)
        instance.path, instance.command = path, method
        instance.request_version = "HTTP/1.1"
        instance.requestline = f"{method} {path} HTTP/1.1"
        instance.client_address = ("127.0.0.1", 12345)
        instance.wfile = io.BytesIO()
        instance.log_message = lambda *args: None
        getattr(instance, f"do_{method}")()
        headers, body = instance.wfile.getvalue().split(b"\r\n\r\n", 1)
        return int(headers.split()[1]), body

    def test_prompt_alignment_ignores_row_order_and_legacy_files(self):
        self.assertEqual(len(self.dataset.prompts), 2)
        self.assertEqual(self.dataset.rows[self.key, "opponent"]["html"], "<html>opponent-A</html>")
        self.assertEqual(self.dataset.models[0], REFERENCE)

    def test_dataset_and_prompt_endpoints(self):
        status, body = self.request("/api/dataset")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["preview_port"], 8766)
        self.assertEqual(data["models"], [REFERENCE, "opponent"])
        self.assertNotIn("annotation_file", data)
        status, body = self.request(f"/api/prompts/{self.key}")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["opponent"]["available"])
        self.assertEqual(self.request("/api/prompts/unknown")[0], 404)

    def test_preview_origin_only_serves_games(self):
        path = f"/games/{self.key}/opponent"
        status, body = self.request(path, games_only=True)
        self.assertEqual(status, 200)
        self.assertEqual(body.decode(), "<html>opponent-A</html>")
        self.assertEqual(self.request(path)[0], 404)
        self.assertEqual(self.request("/api/dataset", games_only=True)[0], 404)

    def test_removed_endpoints_and_writes_are_unavailable(self):
        before = sorted(p.relative_to(self.root) for p in self.root.rglob("*"))
        for path in ["/api/annotations", "/api/import", "/annotations.json"]:
            self.assertEqual(self.request(path)[0], 404)
            for method in ["PUT", "POST", "DELETE", "PATCH"]:
                for games_only in [False, True]:
                    with self.subTest(path=path, method=method, games_only=games_only):
                        self.assertEqual(self.request(path, method, games_only)[0], 405)
        self.assertEqual(before, sorted(p.relative_to(self.root) for p in self.root.rglob("*")))

    def test_duplicate_prompts_fail_explicitly(self):
        source = self.root / "opponent" / "results.jsonl"
        with source.open("a") as stream:
            stream.write('\n{"prompt":"A","html":"<html>duplicate</html>"}')
        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            Dataset(self.root)

    def test_response_fallback_removes_thinking_and_fences(self):
        row = {"response": "reasoning <html>wrong</html></think>\n```html\n<!DOCTYPE html><html>right</html>\n```"}
        self.assertEqual(extract_html(row), "<!DOCTYPE html><html>right</html>")


if __name__ == "__main__":
    unittest.main()
