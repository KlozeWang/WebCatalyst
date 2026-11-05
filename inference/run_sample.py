#!/usr/bin/env python3
"""Generate DesignV3 HTML samples for a base model and every checkpoint step."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import infer_and_simulate as shared  # noqa: E402
from scripts.html_game_dataset import HTML_GAME_SYSTEM_PROMPT  # noqa: E402
from verl.utils.reward_score.agens_sim_reward import _extract_html  # noqa: E402


DEFAULT_INPUT = Path("")
DEFAULT_CHECKPOINT_ROOT = Path("/mnt/nfs/webcatalyst/checkpoints")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "inference" / "results"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
RESULT_FIELDS = {"prompt", "id", "category", "html"}
STEP_PATTERN = re.compile(r"global_step_(\d+)")


@dataclass(frozen=True)
class InputRow:
    """A validated DesignV3 row paired with its normalized chat sample."""

    prompt: Any
    source_id: str | int
    category: str
    sample: shared.Sample


@dataclass(frozen=True)
class ModelSpec:
    """One output step and the model path used to generate it."""

    name: str
    step: int
    model_path: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_folder",
        help="Single checkpoint folder name under --checkpoint-root",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="DesignV3 JSONL input")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL)
    shared.add_generation_arguments(parser)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> Path:
    folder_path = Path(args.checkpoint_folder)
    if folder_path.name != args.checkpoint_folder or args.checkpoint_folder in {"", ".", ".."}:
        raise ValueError("checkpoint_folder must be a single directory name, not a path")
    if args.input.suffix.lower() != ".jsonl" or not args.input.is_file():
        raise FileNotFoundError(f"DesignV3 input JSONL does not exist: {args.input}")
    if not args.checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {args.checkpoint_root}")
    checkpoint_dir = args.checkpoint_root / args.checkpoint_folder
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint folder does not exist: {checkpoint_dir}")
    shared.validate_generation_args(args)
    return checkpoint_dir


def load_rows(path: Path) -> list[InputRow]:
    raw_rows: list[tuple[int, dict[str, Any]]] = []
    seen_ids: set[tuple[type, str | int]] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            missing = [key for key in ("prompt", "id", "category") if key not in row]
            if missing:
                raise ValueError(f"JSONL line {line_number} is missing fields: {', '.join(missing)}")
            source_id = row["id"]
            if isinstance(source_id, bool) or not isinstance(source_id, (str, int)):
                raise TypeError(f"JSONL line {line_number} id must be a string or integer")
            if isinstance(source_id, str) and not source_id:
                raise ValueError(f"JSONL line {line_number} id cannot be empty")
            id_key = (type(source_id), source_id)
            if id_key in seen_ids:
                raise ValueError(f"JSONL line {line_number} has duplicate id {source_id!r}")
            seen_ids.add(id_key)
            if not isinstance(row["category"], str) or not row["category"]:
                raise ValueError(f"JSONL line {line_number} category must be a non-empty string")
            raw_rows.append((line_number, row))

    if not raw_rows:
        raise ValueError(f"Input dataset is empty: {path}")

    width = max(6, len(str(len(raw_rows) - 1)))
    rows = []
    for index, (line_number, row) in enumerate(raw_rows):
        try:
            sample = shared.normalize_sample(index, row["prompt"], width)
        except Exception as error:
            raise ValueError(f"Invalid prompt on JSONL line {line_number}: {error}") from error
        rows.append(
            InputRow(
                prompt=row["prompt"],
                source_id=row["id"],
                category=row["category"],
                sample=sample,
            )
        )
    return rows


def _validate_hf_checkpoint(path: Path) -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hugging Face checkpoint has no config.json: {path}")

    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            with index_path.open(encoding="utf-8") as source:
                weight_map = json.load(source).get("weight_map")
        except (AttributeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid safetensors index: {index_path}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors index has no weight map: {index_path}")
        shard_values = list(weight_map.values())
        if any(not isinstance(name, str) for name in shard_values):
            raise ValueError(f"Safetensors index contains invalid shard names: {index_path}")
        shard_names = set(shard_values)
        missing = sorted(name for name in shard_names if not (path / name).is_file())
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(f"Checkpoint {path} is missing {len(missing)} shards; first: {preview}")
    elif not any(path.glob("*.safetensors")):
        raise FileNotFoundError(f"Hugging Face checkpoint has no safetensors weights: {path}")


def resolve_hf_checkpoint(step_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in step_dir.rglob("huggingface")
        if path.is_dir()
        and (path / "config.json").is_file()
        and ((path / "model.safetensors.index.json").is_file() or any(path.glob("*.safetensors")))
    )
    if not candidates:
        raise FileNotFoundError(f"No Hugging Face checkpoint found under {step_dir}")

    preferred_paths = [
        step_dir / "actor" / "model" / "huggingface",
        step_dir / "model" / "huggingface",
    ]
    selected = next((path for path in preferred_paths if path in candidates), None)
    if selected is None:
        if len(candidates) != 1:
            formatted = "\n  ".join(str(path) for path in candidates)
            raise ValueError(f"Multiple Hugging Face checkpoints found under {step_dir}:\n  {formatted}")
        selected = candidates[0]

    _validate_hf_checkpoint(selected)
    return selected.resolve()


def discover_models(checkpoint_dir: Path, base_model_path: str) -> list[ModelSpec]:
    step_dirs: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        match = STEP_PATTERN.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        step = int(match.group(1))
        if step == 0:
            raise ValueError(f"{path} conflicts with the base model's reserved global_step_0 output")
        step_dirs.append((step, path))

    if not step_dirs:
        raise FileNotFoundError(f"No global_step_* checkpoint directories found under {checkpoint_dir}")

    models = [ModelSpec(name="global_step_0", step=0, model_path=base_model_path)]
    seen_steps = {0}
    for step, step_dir in sorted(step_dirs):
        if step in seen_steps:
            raise ValueError(f"Duplicate numeric checkpoint step {step} under {checkpoint_dir}")
        seen_steps.add(step)
        models.append(
            ModelSpec(
                name=f"global_step_{step}",
                step=step,
                model_path=str(resolve_hf_checkpoint(step_dir)),
            )
        )
    return models


def validate_result_file(path: Path, rows: list[InputRow]) -> None:
    results = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of existing result {path}: {error}") from error
            if not isinstance(result, dict) or set(result) != RESULT_FIELDS:
                raise ValueError(f"Existing result line {line_number} must contain exactly {sorted(RESULT_FIELDS)}")
            results.append(result)

    if len(results) != len(rows):
        raise ValueError(f"Existing result {path} has {len(results)} rows; expected {len(rows)}")
    for line_number, (result, row) in enumerate(zip(results, rows, strict=True), start=1):
        if type(result["id"]) is not type(row.source_id) or result["id"] != row.source_id:
            raise ValueError(f"Existing result {path} has a mismatched id on line {line_number}")
        if result["prompt"] != row.prompt or result["category"] != row.category:
            raise ValueError(f"Existing result {path} does not match the input on line {line_number}")
        if not isinstance(result["html"], str) or not result["html"].strip():
            raise ValueError(f"Existing result {path} has empty HTML on line {line_number}")


def _step_config(args: argparse.Namespace, model: ModelSpec, row_count: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "input_sha256": shared.sha256_file(args.input),
        "row_count": row_count,
        "checkpoint_folder": args.checkpoint_folder,
        "step": model.step,
        "model_path": model.model_path,
        "generation": {
            "n": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "enable_thinking": True,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "rollout_tp": args.rollout_tp,
            "rollout_gpu_memory_utilization": args.rollout_gpu_memory_utilization,
            "rollout_max_model_len": args.rollout_max_model_len,
            "rollout_max_num_batched_tokens": args.rollout_max_num_batched_tokens,
            "rollout_max_num_seqs": args.rollout_max_num_seqs,
            "rollout_moe_backend": args.rollout_moe_backend,
            "generation_concurrency": args.generation_concurrency,
            "nnodes": args.nnodes,
            "gpus_per_node": args.gpus_per_node,
        },
        "system_prompt_sha256": hashlib.sha256(HTML_GAME_SYSTEM_PROMPT.encode()).hexdigest(),
    }


def _generation_html(state_root: Path, row: InputRow) -> str | None:
    generation = shared.latest_event(state_root, row.sample.prompt_id, "generation-success")
    if generation is None:
        return None
    response = generation.get("response")
    if not isinstance(response, str):
        return None
    html = _extract_html(response)
    return html if html.strip() else None


def _pending_samples(rows: list[InputRow], state_root: Path) -> list[shared.Sample]:
    return [
        row.sample
        for row in rows
        if _generation_html(state_root, row) is None
    ]


def _generate_pending(
    pending: list[shared.Sample],
    model: ModelSpec,
    args: argparse.Namespace,
    state_root: Path,
) -> None:
    import ray

    model_args = argparse.Namespace(**{**vars(args), "model_path": model.model_path})
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
            }
        }
    )
    try:
        config = shared.build_rollout_config(model_args)

        async def generation_phase() -> None:
            server_handles, addresses = await shared.start_rollout_servers(config)
            await shared.generate_samples(
                pending,
                addresses,
                model_args,
                state_root,
                progress_desc=model.name,
            )
            assert server_handles

        asyncio.run(generation_phase())
    finally:
        ray.shutdown()


def materialize_results(path: Path, rows: list[InputRow], state_root: Path) -> None:
    records = []
    missing_ids = []
    for row in rows:
        html = _generation_html(state_root, row)
        if html is None:
            missing_ids.append(row.source_id)
            continue
        records.append(
            {
                "prompt": row.prompt,
                "id": row.source_id,
                "category": row.category,
                "html": html,
            }
        )

    if missing_ids:
        preview = ", ".join(repr(value) for value in missing_ids[:20])
        raise RuntimeError(f"Generation failed for {len(missing_ids)} inputs; first ids: {preview}")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                destination.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> int:
    checkpoint_dir = validate_args(args)
    rows = load_rows(args.input)
    models = discover_models(checkpoint_dir, args.base_model_path)
    result_dir = args.output_root / args.checkpoint_folder
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Inference model order:", flush=True)
    for model in models:
        print(f"  {model.name}: {model.model_path}", flush=True)

    completed = 0
    for model in models:
        output_path = result_dir / f"{model.name}.jsonl"
        if output_path.exists():
            validate_result_file(output_path, rows)
            print(f"[{model.name}] complete result already exists; skipping", flush=True)
            completed += 1
            continue

        state_root = result_dir / ".state" / model.name
        shared.initialize_state(
            state_root,
            _step_config(args, model, len(rows)),
            resume=state_root.exists(),
        )
        pending = _pending_samples(rows, state_root)
        print(f"[{model.name}] {len(pending)}/{len(rows)} samples pending", flush=True)
        if pending:
            _generate_pending(pending, model, args, state_root)
        materialize_results(output_path, rows, state_root)
        validate_result_file(output_path, rows)
        print(f"[{model.name}] wrote {output_path}", flush=True)
        completed += 1

    print(f"Completed {completed}/{len(models)} model steps.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume completed samples.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"sample inference failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
