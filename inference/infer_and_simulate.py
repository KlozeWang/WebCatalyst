#!/usr/bin/env python3
"""Generate one HTML response per prompt and evaluate it with Agens."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.html_game_dataset import HTML_GAME_SYSTEM_PROMPT  # noqa: E402
from verl.utils.reward_score.agens_sim_reward import _extract_html, compute_score  # noqa: E402


STATE_SCHEMA_VERSION = 1
DEFAULT_MODEL_PATH = "Qwen/Qwen3.6-35B-A3B"
FAILED_PROMPT_PREVIEW_LIMIT = 20


@dataclass(frozen=True)
class Sample:
    """A normalized input row."""

    index: int
    prompt_id: str
    messages: list[dict[str, Any]]
    task_prompt: str


@dataclass(frozen=True)
class GenerationResult:
    """The useful subset of an OpenAI-compatible chat response."""

    response: str
    finish_reason: str | None
    usage: dict[str, Any] | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input Parquet or JSONL dataset")
    parser.add_argument("--output-root", required=True, type=Path, help="Retained output and resume-state directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Hugging Face model path or repository ID")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run in the same output root")

    add_generation_arguments(parser)

    parser.add_argument("--agens-cli", type=Path, default=REPO_ROOT / "agens-cli" / "dist" / "cli.js")
    parser.add_argument("--agens-cwd", type=Path, default=REPO_ROOT / "agens-cli")
    parser.add_argument("--agens-turn-limit", type=int, default=8)
    parser.add_argument("--agens-concurrency", type=int, default=1)
    parser.add_argument("--agens-attempt-timeout", type=int, default=300)
    parser.add_argument("--agens-total-timeout", type=int, default=330)
    parser.add_argument("--agens-max-retries", type=int, default=0)
    return parser.parse_args(argv)


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the rollout and sampling arguments shared by inference entrypoints."""
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--rollout-tp", type=int, default=1)
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=32768)
    parser.add_argument("--rollout-max-model-len", type=int, default=34816)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=34816)
    parser.add_argument("--rollout-max-num-seqs", type=int, default=16)
    parser.add_argument("--rollout-moe-backend", default="triton")
    parser.add_argument(
        "--generation-concurrency",
        type=int,
        default=0,
        help="Maximum in-flight requests; 0 uses replicas multiplied by max_num_seqs",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)


def validate_args(args: argparse.Namespace) -> None:
    if args.input.suffix.lower() not in {".parquet", ".pq", ".jsonl"}:
        raise ValueError(f"Input must be a Parquet or JSONL file: {args.input}")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input}")
    validate_generation_args(args)
    positive_fields = {
        "agens-turn-limit": args.agens_turn_limit,
        "agens-concurrency": args.agens_concurrency,
        "agens-attempt-timeout": args.agens_attempt_timeout,
        "agens-total-timeout": args.agens_total_timeout,
    }
    invalid = [name for name, value in positive_fields.items() if value < 1]
    if invalid:
        raise ValueError(f"These values must be positive: {', '.join(invalid)}")
    if args.agens_max_retries < 0:
        raise ValueError("agens-max-retries cannot be negative")
    if args.agens_total_timeout <= args.agens_attempt_timeout:
        raise ValueError("agens-total-timeout must be greater than agens-attempt-timeout")
    if not args.agens_cli.is_file():
        raise FileNotFoundError(f"Agens CLI does not exist: {args.agens_cli}")
    if not args.agens_cwd.is_dir():
        raise FileNotFoundError(f"Agens working directory does not exist: {args.agens_cwd}")


def validate_generation_args(args: argparse.Namespace) -> None:
    """Validate the rollout and sampling arguments shared by inference entrypoints."""
    if args.nnodes < 1 or args.gpus_per_node < 1 or args.rollout_tp < 1:
        raise ValueError("nnodes, gpus-per-node, and rollout-tp must be positive")
    world_size = args.nnodes * args.gpus_per_node
    if world_size % args.rollout_tp:
        raise ValueError(f"World size {world_size} must be divisible by rollout TP {args.rollout_tp}")
    positive_fields = {
        "max-prompt-length": args.max_prompt_length,
        "max-response-length": args.max_response_length,
        "rollout-max-model-len": args.rollout_max_model_len,
        "rollout-max-num-batched-tokens": args.rollout_max_num_batched_tokens,
        "rollout-max-num-seqs": args.rollout_max_num_seqs,
    }
    invalid = [name for name, value in positive_fields.items() if value < 1]
    if invalid:
        raise ValueError(f"These values must be positive: {', '.join(invalid)}")
    if args.generation_concurrency < 0:
        raise ValueError("generation-concurrency cannot be negative")
    if not 0 < args.rollout_gpu_memory_utilization <= 1:
        raise ValueError("rollout-gpu-memory-utilization must be in (0, 1]")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be non-negative and top-p must be in (0, 1]")
    if args.max_prompt_length + args.max_response_length > args.rollout_max_model_len:
        raise ValueError("max-prompt-length plus max-response-length exceeds rollout-max-model-len")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if hasattr(content, "tolist"):
        content = content.tolist()
    if isinstance(content, list):
        parts = []
        for item in content:
            item = _jsonable(item)
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def normalize_sample(index: int, prompt: Any, width: int = 6) -> Sample:
    """Normalize the two prompt representations supported by the training data."""
    prompt_id = f"{index:0{width}d}"
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("prompt is empty")
        return Sample(
            index=index,
            prompt_id=prompt_id,
            messages=[
                {"role": "system", "content": HTML_GAME_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            task_prompt=prompt,
        )

    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise TypeError("prompt must be a non-empty string or chat-message list")

    messages: list[dict[str, Any]] = []
    for message in prompt:
        message = _jsonable(message)
        if not isinstance(message, dict):
            raise TypeError("every chat message must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("every chat message must have a role")
        if "content" not in message:
            raise ValueError("every chat message must have content")
        messages.append(dict(message))

    if not any(message.get("role") == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": HTML_GAME_SYSTEM_PROMPT})

    task_prompt = next(
        (_content_text(message.get("content")) for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    if not task_prompt:
        raise ValueError("chat prompt must contain a non-empty user message")
    return Sample(index=index, prompt_id=prompt_id, messages=messages, task_prompt=task_prompt)


def load_prompts(path: Path, prompt_key: str) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        prompts = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}: {error}") from error
                if not isinstance(row, dict) or prompt_key not in row:
                    raise ValueError(f"JSONL line {line_number} does not contain prompt key {prompt_key!r}")
                prompts.append(row[prompt_key])
        return prompts

    dataframe = pd.read_parquet(path)
    if prompt_key not in dataframe.columns:
        raise ValueError(f"Parquet dataset does not contain prompt column {prompt_key!r}")
    return dataframe[prompt_key].tolist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_config(args: argparse.Namespace, row_count: int) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "row_count": row_count,
        "prompt_key": args.prompt_key,
        "model_path": args.model_path,
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
            "nnodes": args.nnodes,
            "gpus_per_node": args.gpus_per_node,
        },
        "simulation": {
            "agens_cli": str(args.agens_cli.resolve()),
            "agens_cwd": str(args.agens_cwd.resolve()),
            "turn_limit": args.agens_turn_limit,
            "attempt_timeout": args.agens_attempt_timeout,
            "total_timeout": args.agens_total_timeout,
            "max_retries": args.agens_max_retries,
            "judge_model": os.environ.get("AGENS_SIM_JUDGE_MODEL"),
            "extra_global_judge_model": os.environ.get("AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL"),
            "judge_wire_api": os.environ.get("AGENS_SIM_JUDGE_WIRE_API", os.environ.get("AGENS_WIRE_API")),
        },
        "system_prompt_sha256": hashlib.sha256(HTML_GAME_SYSTEM_PROMPT.encode()).hexdigest(),
    }


def initialize_state(output_root: Path, config: dict[str, Any], resume: bool) -> Path:
    output_root = output_root.resolve()
    state_dir = output_root / "state"
    if resume:
        existing = latest_valid_json(state_dir.glob("run-config-*.json"))
        if existing is None:
            raise FileNotFoundError(f"Cannot resume without a valid run config under {state_dir}")
        if existing != config:
            raise ValueError("Resume configuration does not match the existing dataset/model/settings")
        return output_root

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty; use --resume or choose another directory: {output_root}")
    config_path = state_dir / f"run-config-{time.time_ns()}-{uuid.uuid4().hex}.json"
    write_json_exclusive(config_path, config)
    return output_root


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Write a new JSON file without replacing or deleting any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, default=str)
        destination.write("\n")


def write_text_exclusive(path: Path, value: str) -> None:
    """Write a new text file without replacing or deleting an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        destination.write(value)


def _has_retained_agens_artifacts(result: dict[str, Any], task_prompt: str) -> bool:
    run_dir_value = result.get("agens_run_dir")
    reward_result_value = result.get("agens_reward_result")
    if not isinstance(run_dir_value, str) or not isinstance(reward_result_value, str):
        return False

    run_dir = Path(run_dir_value)
    required = [
        run_dir / "artifact" / "index.html",
        run_dir / "trace",
        Path(reward_result_value),
    ]
    if task_prompt:
        required.append(run_dir / "artifact" / "prompt.txt")
    return all(path.exists() for path in required)


def retain_simulation_fallback(
    output_root: Path,
    sample: Sample,
    generation: dict[str, Any],
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Retain inputs and a failure trace when Agens exits before creating a run."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3] + "Z"
    run_dir = (
        output_root
        / "agens"
        / "step-inference"
        / f"prompt-{sample.prompt_id}"
        / "sample-0"
        / f"run-{timestamp}-retained-{uuid.uuid4().hex[:8]}"
    )
    artifact_dir = run_dir / "artifact"
    trace_dir = run_dir / "trace" / timestamp
    artifact_path = artifact_dir / "index.html"
    prompt_path = artifact_dir / "prompt.txt"
    failure_trace_path = trace_dir / "simulation-failure.json"
    reward_result_path = run_dir / "reward-result.json"

    errors: list[str] = []
    if error:
        errors.append(error)
    if result is not None and result.get("agens_errors"):
        errors.append(str(result["agens_errors"]))
    if not errors:
        errors.append("Agens returned without retained simulation artifacts.")

    metadata = {
        "globalStep": "inference",
        "promptId": sample.prompt_id,
        "sampleId": "0",
        "rolloutId": f"inference-{sample.prompt_id}-0",
    }
    paths = {
        "sampleDir": str(run_dir),
        "artifactDir": str(artifact_dir),
        "artifactPath": str(artifact_path),
        "stateTracePath": str(failure_trace_path),
        "rewardResultPath": str(reward_result_path),
    }
    fallback_result = {
        "ok": result.get("agens_ok") is True if result is not None else False,
        "score": result.get("score", 0.0) if result is not None else 0.0,
        "scoreSource": "globalEvaluation.score / 100",
        "metadata": metadata,
        "paths": paths,
        "transitionEvaluations": [],
        "screenshotEvaluations": [],
        "errors": errors,
        "retainedByInferenceFallback": True,
    }

    write_text_exclusive(artifact_path, _extract_html(generation["response"]))
    write_text_exclusive(prompt_path, sample.task_prompt)
    write_json_exclusive(
        failure_trace_path,
        {
            "event": "simulation-failure",
            "timestamp_ns": time.time_ns(),
            "metadata": metadata,
            "errors": errors,
        },
    )
    write_json_exclusive(reward_result_path, fallback_result)
    return {
        "agens_run_dir": str(run_dir),
        "agens_reward_result": str(reward_result_path),
        "agens_artifacts_fallback": True,
    }


def latest_valid_json(paths) -> dict[str, Any] | None:
    """Return the newest complete JSON object from append-only state files."""
    for path in sorted(paths, reverse=True):
        try:
            return read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def write_event(output_root: Path, prompt_id: str, event: str, value: dict[str, Any]) -> Path:
    event_id = f"{time.time_ns()}-{uuid.uuid4().hex}"
    path = output_root / "state" / f"prompt-{prompt_id}" / f"{event}-{event_id}.json"
    write_json_exclusive(path, {"event": event, "prompt_id": prompt_id, "timestamp_ns": time.time_ns(), **value})
    return path


def latest_event(output_root: Path, prompt_id: str, event: str) -> dict[str, Any] | None:
    event_dir = output_root / "state" / f"prompt-{prompt_id}"
    return latest_valid_json(event_dir.glob(f"{event}-*.json"))


def build_rollout_config(args: argparse.Namespace):
    """Compose the normal VERL trainer config and retain only rollout-relevant settings."""
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_dir = str(REPO_ROOT / "verl" / "trainer" / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="ppo_trainer")

    with open_dict(config):
        config.trainer.nnodes = args.nnodes
        config.trainer.n_gpus_per_node = args.gpus_per_node
        config.actor_rollout_ref.model.path = args.model_path
        config.actor_rollout_ref.model.trust_remote_code = True

        rollout = config.actor_rollout_ref.rollout
        rollout.name = "vllm"
        rollout.tensor_model_parallel_size = args.rollout_tp
        rollout.gpu_memory_utilization = args.rollout_gpu_memory_utilization
        rollout.max_model_len = args.rollout_max_model_len
        rollout.max_num_batched_tokens = args.rollout_max_num_batched_tokens
        rollout.max_num_seqs = args.rollout_max_num_seqs
        rollout.prompt_length = args.max_prompt_length
        rollout.response_length = args.max_response_length
        rollout.dtype = "bfloat16"
        rollout.n = 1
        rollout.temperature = args.temperature
        rollout.top_p = args.top_p
        rollout.top_k = args.top_k
        rollout.do_sample = True
        rollout.engine_kwargs.vllm["moe_backend"] = args.rollout_moe_backend
        rollout.engine_kwargs.vllm["language_model_only"] = True
    return config


async def start_rollout_servers(config) -> tuple[list[Any], list[str]]:
    """Start the same standalone vLLM replicas used by VERL generation."""
    from verl.workers.rollout.replica import get_rollout_replica_class

    rollout = config.actor_rollout_ref.rollout
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    replica_count = world_size // rollout.tensor_model_parallel_size
    replica_class = get_rollout_replica_class(rollout.name)
    replicas = [
        replica_class(
            replica_rank=rank,
            config=rollout,
            model_config=config.actor_rollout_ref.model,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for rank in range(replica_count)
    ]
    await asyncio.gather(*(replica.init_standalone() for replica in replicas))
    server_handles = [replica.server_handle for replica in replicas]
    server_addresses = [replica.server_address for replica in replicas]
    if not all(server_handles) or not all(server_addresses):
        raise RuntimeError("A vLLM rollout replica started without a server handle or address")
    return server_handles, server_addresses


def chat_request(sample: Sample, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model_path,
        "messages": sample.messages,
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_response_length,
        "chat_template_kwargs": {"enable_thinking": True},
    }


async def submit_generation(
    session: aiohttp.ClientSession,
    server_address: str,
    sample: Sample,
    args: argparse.Namespace,
) -> GenerationResult:
    async with session.post(
        f"http://{server_address}/v1/chat/completions",
        headers={"Authorization": "Bearer token-abc123"},
        json=chat_request(sample, args),
    ) as response:
        text = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"vLLM returned HTTP {response.status}: {text[-2000:]}")
        try:
            payload = json.loads(text)
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid vLLM chat response: {text[-2000:]}") from error
        if not isinstance(content, str) or not content:
            raise RuntimeError("vLLM returned an empty response")
        usage = payload.get("usage")
        return GenerationResult(
            response=content,
            finish_reason=choice.get("finish_reason"),
            usage=usage if isinstance(usage, dict) else None,
        )


async def generate_samples(
    samples: list[Sample],
    server_addresses: list[str],
    args: argparse.Namespace,
    output_root: Path,
    progress_desc: str = "generation",
) -> None:
    concurrency = args.generation_concurrency or len(server_addresses) * args.rollout_max_num_seqs
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def generate_one(position: int, sample: Sample) -> tuple[Sample, GenerationResult | Exception]:
            async with semaphore:
                try:
                    result = await submit_generation(
                        session,
                        server_addresses[position % len(server_addresses)],
                        sample,
                        args,
                    )
                    return sample, result
                except Exception as error:
                    return sample, error

        tasks = [asyncio.create_task(generate_one(position, sample)) for position, sample in enumerate(samples)]
        with tqdm(total=len(tasks), desc=progress_desc, unit="sample", dynamic_ncols=True) as progress:
            for completed in asyncio.as_completed(tasks):
                sample, result = await completed
                if isinstance(result, Exception):
                    write_event(output_root, sample.prompt_id, "generation-failure", {"error": str(result)})
                    progress.write(f"[generation] prompt {sample.prompt_id} failed: {result}", file=sys.stderr)
                else:
                    write_event(output_root, sample.prompt_id, "generation-success", asdict(result))
                    progress.write(f"[generation] prompt {sample.prompt_id} complete")
                progress.update()


async def simulate_samples(samples: list[Sample], args: argparse.Namespace, output_root: Path) -> None:
    semaphore = asyncio.Semaphore(args.agens_concurrency)
    agens_output_root = output_root / "agens"

    async def simulate_one(sample: Sample, generation: dict[str, Any]) -> None:
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    compute_score(
                        data_source="html_game",
                        solution_str=generation["response"],
                        ground_truth="",
                        extra_info={
                            "global_steps": "inference",
                            "group_uid": sample.prompt_id,
                            "session_id": 0,
                            "uid": f"inference-{sample.prompt_id}-0",
                            "task_prompt": sample.task_prompt,
                            "turn_limit": args.agens_turn_limit,
                        },
                        agens_cli=str(args.agens_cli.resolve()),
                        agens_cwd=str(args.agens_cwd.resolve()),
                        output_root=str(agens_output_root),
                        turn_limit=args.agens_turn_limit,
                        attempt_timeout=args.agens_attempt_timeout,
                        max_retries=args.agens_max_retries,
                    ),
                    timeout=args.agens_total_timeout,
                )
            except Exception as error:
                fallback = retain_simulation_fallback(
                    output_root,
                    sample,
                    generation,
                    error=str(error),
                )
                write_event(
                    output_root,
                    sample.prompt_id,
                    "simulation-failure",
                    {
                        "error": str(error),
                        "result": {
                            "score": 0.0,
                            "agens_ok": False,
                            "agens_skipped": True,
                            **fallback,
                        },
                    },
                )
                tqdm.write(f"[simulation] prompt {sample.prompt_id} failed: {error}", file=sys.stderr)
                return

            if not _has_retained_agens_artifacts(result, sample.task_prompt):
                result = {
                    **result,
                    **retain_simulation_fallback(
                        output_root,
                        sample,
                        generation,
                        result=result,
                    ),
                }
            event = "simulation-success" if result.get("agens_ok") is True else "simulation-failure"
            write_event(output_root, sample.prompt_id, event, {"result": result})
            tqdm.write(f"[simulation] prompt {sample.prompt_id} {event.removeprefix('simulation-')}")

    tasks = []
    for sample in samples:
        if latest_event(output_root, sample.prompt_id, "simulation-success") is not None:
            continue
        generation = latest_event(output_root, sample.prompt_id, "generation-success")
        if generation is not None:
            tasks.append(asyncio.create_task(simulate_one(sample, generation)))
    if tasks:
        with tqdm(total=len(tasks), desc="simulation", unit="sample", dynamic_ncols=True) as progress:
            for completed in asyncio.as_completed(tasks):
                await completed
                progress.update()


def summarize(samples: list[Sample], invalid_ids: list[str], output_root: Path) -> dict[str, Any]:
    generated = 0
    succeeded = 0
    failed_ids = list(invalid_ids)
    for sample in samples:
        generation = latest_event(output_root, sample.prompt_id, "generation-success")
        simulation = latest_event(output_root, sample.prompt_id, "simulation-success")
        generated += generation is not None
        succeeded += simulation is not None
        if simulation is None:
            failed_ids.append(sample.prompt_id)
    failed_ids = sorted(failed_ids)
    failed_count = len(failed_ids)
    summary = {
        "total": len(samples) + len(invalid_ids),
        "generated": generated,
        "succeeded": succeeded,
        "failed": failed_count,
        "failed_prompt_ids_preview": failed_ids[:FAILED_PROMPT_PREVIEW_LIMIT],
        "failed_prompt_ids_omitted": max(0, failed_count - FAILED_PROMPT_PREVIEW_LIMIT),
    }
    summary_path = output_root / "state" / f"run-summary-{time.time_ns()}-{uuid.uuid4().hex}.json"
    write_json_exclusive(summary_path, summary)
    return summary


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    prompts = load_prompts(args.input, args.prompt_key)
    output_root = initialize_state(args.output_root, run_config(args, len(prompts)), args.resume)
    width = max(6, len(str(max(len(prompts) - 1, 0))))

    samples = []
    invalid_ids = []
    for index, prompt in enumerate(prompts):
        prompt_id = f"{index:0{width}d}"
        try:
            samples.append(normalize_sample(index, prompt, width))
        except Exception as error:
            invalid_ids.append(prompt_id)
            write_event(output_root, prompt_id, "generation-failure", {"error": f"Invalid prompt: {error}"})
            print(f"[input] prompt {prompt_id} invalid: {error}", file=sys.stderr, flush=True)

    pending_generation = [
        sample
        for sample in samples
        if latest_event(output_root, sample.prompt_id, "generation-success") is None
        and latest_event(output_root, sample.prompt_id, "simulation-success") is None
    ]

    if pending_generation:
        import ray

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
            config = build_rollout_config(args)

            async def generation_phase() -> None:
                # Ray actors are reference counted. Retain their handles until
                # every HTTP generation request has completed.
                server_handles, addresses = await start_rollout_servers(config)
                await generate_samples(pending_generation, addresses, args, output_root)
                assert server_handles

            asyncio.run(generation_phase())
        finally:
            ray.shutdown()

    asyncio.run(simulate_samples(samples, args, output_root))
    summary = summarize(samples, invalid_ids, output_root)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted; rerun with --resume to continue without repeating successful samples.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"inference failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
