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

import json
from pathlib import Path

import pytest

from verl.utils.reward_score.agens_pre_rollout import (
    PreRolloutReferenceError,
    resolve_pre_rollout_reference,
)


def _write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _root(tmp_path: Path, prompt: str = "Build a game.") -> tuple[Path, Path]:
    root = tmp_path / "results"
    _write(root / "state" / "run-config-1-test.json", json.dumps({"row_count": 1}))
    run_dir = root / "agens" / "step-inference" / "prompt-000000" / "sample-0" / "run-1"
    _write(run_dir / "artifact" / "prompt.txt", prompt)
    return root, run_dir


def test_resolver_selects_before_screenshot_for_highest_evaluation_score(tmp_path: Path) -> None:
    root, run_dir = _root(tmp_path)
    trace = run_dir / "trace" / "trace-1"
    for turn, score in [(0, 75), (1, 93)]:
        _write(trace / f"turn-{turn:03d}-before.png", f"before-{turn}".encode())
        _write(
            trace / f"turn-{turn:03d}-screenshot-evaluation.json",
            json.dumps({"turnIndex": turn, "score": score}),
        )

    reference = resolve_pre_rollout_reference(root, 0, "Build a game.")

    assert reference.screenshot_path == (trace / "turn-001-before.png").resolve()
    assert reference.screenshot_score == 93
    assert reference.turn_index == 1
    assert reference.selection == "highest-screenshot-evaluation-score"


def test_resolver_uses_turn_zero_only_when_no_valid_evaluation_exists(tmp_path: Path) -> None:
    root, run_dir = _root(tmp_path)
    before = run_dir / "trace" / "trace-1" / "turn-000-before.png"
    _write(before, b"before")

    reference = resolve_pre_rollout_reference(root, 0, "Build a game.")

    assert reference.screenshot_path == before.resolve()
    assert reference.screenshot_score is None
    assert reference.selection == "turn-000-fallback"


def test_resolver_rejects_same_index_with_different_prompt(tmp_path: Path) -> None:
    root, run_dir = _root(tmp_path, prompt="A different game.")
    _write(run_dir / "trace" / "trace-1" / "turn-000-before.png", b"before")

    with pytest.raises(PreRolloutReferenceError, match="matching the training prompt exactly"):
        resolve_pre_rollout_reference(root, 0, "Build a game.")
