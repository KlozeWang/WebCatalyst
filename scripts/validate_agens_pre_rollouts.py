#!/usr/bin/env python3
"""Validate a training dataset against retained Agens pre-rollout screenshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.reward_score.agens_pre_rollout import (  # noqa: E402
    PreRolloutReferenceError,
    resolve_pre_rollout_reference,
    validate_results_dataset,
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        )
    return ""


def _task_prompt(prompt: Any) -> str:
    if isinstance(prompt, str) and prompt:
        return prompt
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                text = _content_text(message.get("content"))
                if text:
                    return text
    raise ValueError("prompt must be a non-empty string or contain a non-empty user message")


def load_task_prompts(path: Path, prompt_key: str) -> list[str]:
    suffix = path.suffix.lower()
    values: list[Any]
    if suffix == ".jsonl":
        values = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or prompt_key not in row:
                    raise ValueError(f"line {line_number} does not contain prompt key {prompt_key!r}")
                values.append(row[prompt_key])
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=[prompt_key])
        values = frame[prompt_key].tolist()
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    return [_task_prompt(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--error-preview", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prompts = load_task_prompts(args.dataset, args.prompt_key)
        validate_results_dataset(args.results_root, args.dataset, len(prompts))
    except Exception as error:
        print(f"pre-rollout validation failed: {error}", file=sys.stderr)
        return 1

    warnings: list[str] = []
    evaluated = 0
    fallback = 0
    for index, prompt in enumerate(prompts):
        try:
            reference = resolve_pre_rollout_reference(args.results_root, index, prompt)
        except PreRolloutReferenceError as error:
            warnings.append(str(error))
            continue
        if reference.screenshot_score is None:
            fallback += 1
        else:
            evaluated += 1

    if warnings:
        preview = warnings[: max(0, args.error_preview)]
        print(
            f"pre-rollout validation warning: {len(warnings)} of {len(prompts)} training prompts have no "
            "usable screenshot; their Stage-1 penalty will be 0",
            file=sys.stderr,
        )
        for warning in preview:
            print(f"  - {warning}", file=sys.stderr)
        omitted = len(warnings) - len(preview)
        if omitted:
            print(f"  - ... {omitted} more omitted", file=sys.stderr)

    print(
        f"Validated pre-rollout dataset identity: {evaluated + fallback} usable references "
        f"({evaluated} score-selected, {fallback} turn-000 fallbacks), "
        f"{len(warnings)} unavailable with Stage-1 penalty 0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
