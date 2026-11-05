"""Resolve prompt-matched screenshots from retained Agens inference runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


class PreRolloutReferenceError(ValueError):
    """Raised when a dataset row has no safe, prompt-matched reference."""


@dataclass(frozen=True)
class PreRolloutReference:
    """A pre-rollout screenshot selected for one original dataset row."""

    prompt_id: str
    screenshot_path: Path
    evaluation_path: Path | None
    screenshot_score: float | None
    turn_index: int
    selection: str

    def as_request(self) -> dict[str, Any]:
        return {
            "promptId": self.prompt_id,
            "screenshotPath": str(self.screenshot_path),
            "evaluationPath": str(self.evaluation_path) if self.evaluation_path is not None else None,
            "screenshotScore": self.screenshot_score,
            "turnIndex": self.turn_index,
            "selection": self.selection,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


@lru_cache(maxsize=16)
def load_pre_rollout_run_config(results_root: str) -> dict[str, Any]:
    """Load the newest complete inference run config for a results root."""
    root = Path(results_root).expanduser().resolve()
    state_dir = root / "state"
    for path in sorted(state_dir.glob("run-config-*.json"), reverse=True):
        try:
            config = _read_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        row_count = config.get("row_count")
        if not isinstance(row_count, int) or row_count < 1:
            continue
        return config
    raise PreRolloutReferenceError(f"No valid inference run config found under {state_dir}")


def validate_results_dataset(results_root: Path, dataset_path: Path, row_count: int) -> dict[str, Any]:
    """Ensure the retained inference root was produced from the exact dataset."""
    root = results_root.expanduser().resolve()
    dataset = dataset_path.expanduser().resolve()
    config = load_pre_rollout_run_config(str(root))
    errors = []
    if config.get("row_count") != row_count:
        errors.append(f"row count differs: results={config.get('row_count')} dataset={row_count}")
    expected_hash = config.get("input_sha256")
    actual_hash = sha256_file(dataset)
    if expected_hash != actual_hash:
        errors.append(f"SHA-256 differs: results={expected_hash!r} dataset={actual_hash!r}")
    if errors:
        raise PreRolloutReferenceError("Pre-rollout dataset mismatch: " + "; ".join(errors))
    return config


def _safe_existing_path(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise PreRolloutReferenceError(f"Reference path escapes the results root: {resolved}")
    return resolved


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def _evaluation_candidate(path: Path, root: Path) -> tuple[float, int, Path] | None:
    try:
        evaluation = _read_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    score = _numeric_score(evaluation.get("score"))
    turn_index = evaluation.get("turnIndex")
    if score is None or isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        return None
    before_path = path.parent / f"turn-{turn_index:03d}-before.png"
    if not before_path.is_file():
        return None
    return score, turn_index, _safe_existing_path(before_path, root)


@lru_cache(maxsize=8192)
def _resolve_cached(results_root: str, index: int, task_prompt: str) -> PreRolloutReference:
    root = Path(results_root).expanduser().resolve()
    config = load_pre_rollout_run_config(str(root))
    row_count = int(config["row_count"])
    if index < 0 or index >= row_count:
        raise PreRolloutReferenceError(f"Dataset index {index} is outside the pre-rollout row count {row_count}")

    width = max(6, len(str(row_count - 1)))
    prompt_id = f"{index:0{width}d}"
    prompt_dir = root / "agens" / "step-inference" / f"prompt-{prompt_id}"
    if not prompt_dir.is_dir():
        raise PreRolloutReferenceError(f"prompt {prompt_id}: missing directory {prompt_dir}")

    matching_runs: list[Path] = []
    for run_dir in sorted(prompt_dir.glob("sample-*/run-*"), reverse=True):
        prompt_path = run_dir / "artifact" / "prompt.txt"
        try:
            retained_prompt = prompt_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if retained_prompt == task_prompt:
            matching_runs.append(run_dir)
    if not matching_runs:
        raise PreRolloutReferenceError(
            f"prompt {prompt_id}: no run has artifact/prompt.txt matching the training prompt exactly"
        )

    evaluated: list[tuple[float, str, int, Path, Path]] = []
    for run_dir in matching_runs:
        for evaluation_path in run_dir.glob("trace/*/turn-*-screenshot-evaluation.json"):
            candidate = _evaluation_candidate(evaluation_path, root)
            if candidate is None:
                continue
            score, turn_index, before_path = candidate
            evaluated.append((score, str(evaluation_path.parent), turn_index, before_path, evaluation_path))

    if evaluated:
        score, _, turn_index, before_path, evaluation_path = max(evaluated, key=lambda item: (item[0], item[1]))
        return PreRolloutReference(
            prompt_id=prompt_id,
            screenshot_path=before_path,
            evaluation_path=_safe_existing_path(evaluation_path, root),
            screenshot_score=score,
            turn_index=turn_index,
            selection="highest-screenshot-evaluation-score",
        )

    turn_zero_paths = sorted(
        (path for run_dir in matching_runs for path in run_dir.glob("trace/*/turn-000-before.png")),
        reverse=True,
    )
    if not turn_zero_paths:
        raise PreRolloutReferenceError(
            f"prompt {prompt_id}: no usable screenshot evaluation and no fallback turn-000-before.png"
        )
    return PreRolloutReference(
        prompt_id=prompt_id,
        screenshot_path=_safe_existing_path(turn_zero_paths[0], root),
        evaluation_path=None,
        screenshot_score=None,
        turn_index=0,
        selection="turn-000-fallback",
    )


def resolve_pre_rollout_reference(results_root: str | Path, index: int, task_prompt: str) -> PreRolloutReference:
    """Resolve a reference while enforcing index and exact-prompt identity."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise PreRolloutReferenceError(f"Pre-rollout dataset index must be an integer, got {index!r}")
    if not isinstance(task_prompt, str) or not task_prompt:
        raise PreRolloutReferenceError("Pre-rollout task prompt must be a non-empty string")
    return _resolve_cached(str(Path(results_root).expanduser().resolve()), index, task_prompt)
