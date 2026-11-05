# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Custom reward function that evaluates rollout HTML with agens simulate-reward."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import signal
from pathlib import Path
from typing import Any

from verl.utils.reward_score.agens_pre_rollout import PreRolloutReferenceError, resolve_pre_rollout_reference

DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 450
DEFAULT_MAX_RETRIES = 2
DEFAULT_REFERENCE_PENALTY_LAMBDA = 0.5
PROCESS_CLEANUP_TIMEOUT_SECONDS = 5
logger = logging.getLogger(__name__)


def _finite_score_component(result: dict[str, Any], key: str) -> float | None:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _score_component(result: dict[str, Any], key: str) -> float:
    value = _finite_score_component(result, key)
    return value if value is not None else 0.0


def _pre_rollout_score_components(result: dict[str, Any]) -> dict[str, float]:
    components = {}
    for source, destination in (
        ("unpenalizedGlobalScore", "unpenalized_global_score"),
        ("preRolloutVisualPenalty", "pre_rollout_visual_penalty"),
    ):
        value = _finite_score_component(result, source)
        if value is not None:
            components[destination] = value
    return components


def _validate_reference_penalty_lambda(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("reference_penalty_lambda must be a finite number between 0 and 1")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("reference_penalty_lambda must be a finite number between 0 and 1")
    return value


def _training_scores(
    result: dict[str, Any], reference_penalty_lambda: float
) -> tuple[float, float, float | None] | None:
    score = _finite_score_component(result, "score")
    if score is None:
        return None

    global_score = _score_component(result, "globalScore")
    unpenalized_score = _finite_score_component(result, "unpenalizedGlobalScore")
    reference_penalty = _finite_score_component(result, "preRolloutVisualPenalty")
    if unpenalized_score is None or reference_penalty is None:
        return score, global_score, None

    applied_reference_penalty = reference_penalty_lambda * reference_penalty
    adjusted_global_score = unpenalized_score - applied_reference_penalty
    return adjusted_global_score / 100, adjusted_global_score, applied_reference_penalty


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    *,
    agens_cli: str | None = None,
    agens_cwd: str | None = None,
    node_bin: str = "node",
    output_root: str = "run_log",
    turn_limit: int = 10,
    attempt_timeout: int = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    reference_penalty_lambda: float = DEFAULT_REFERENCE_PENALTY_LAMBDA,
    pre_rollout_results_root: str | None = None,
) -> dict[str, Any]:
    """Compute reward by running agens simulate-reward on the generated HTML."""
    extra_info = dict(extra_info or {})
    reference_penalty_lambda = _validate_reference_penalty_lambda(reference_penalty_lambda)
    repo_root = Path(__file__).resolve().parents[3]
    agens_cli_path = Path(agens_cli) if agens_cli else repo_root / "agens-cli" / "dist" / "cli.js"
    agens_cli_path = agens_cli_path if agens_cli_path.is_absolute() else (repo_root / agens_cli_path).resolve()
    agens_cwd_path = Path(agens_cwd) if agens_cwd else agens_cli_path.parents[1]
    agens_cwd_path = agens_cwd_path if agens_cwd_path.is_absolute() else (repo_root / agens_cwd_path).resolve()

    html, html_char_start, html_char_end = _extract_html_with_span(solution_str)
    task_prompt = _task_prompt(extra_info, ground_truth)
    request = {
        "html": html,
        "taskPrompt": task_prompt,
        "outputRoot": output_root,
        "globalStep": extra_info.get("global_steps", extra_info.get("global_step", "unknown")),
        "promptId": extra_info.get("group_uid", extra_info.get("prompt_id", extra_info.get("index", "unknown"))),
        "sampleId": extra_info.get("session_id", extra_info.get("sample_id", "unknown")),
        "rolloutId": extra_info.get("uid", extra_info.get("rollout_id", "unknown")),
        "turnLimit": int(extra_info.get("turn_limit", turn_limit)),
    }
    trace_config = extra_info.get("agens_trace_advantage")
    trace_requested = (
        isinstance(trace_config, dict)
        and bool(trace_config.get("enabled", False))
        and not bool(extra_info.get("validate", False))
    )
    if trace_requested:
        request["actionTrace"] = {
            "enabled": True,
            "threshold": trace_config.get("threshold", 70),
            "maxPauseEventsPerAction": trace_config.get("max_pause_events_per_action", 4096),
        }
    if pre_rollout_results_root and not bool(extra_info.get("validate", False)):
        raw_index = extra_info.get("index")
        if hasattr(raw_index, "item"):
            raw_index = raw_index.item()
        try:
            reference = resolve_pre_rollout_reference(pre_rollout_results_root, raw_index, task_prompt)
        except PreRolloutReferenceError as error:
            logger.warning(
                "[AgensReward] no usable pre-rollout screenshot for dataset index=%r; "
                "skipping Stage-1 comparison with penalty=0: %s",
                raw_index,
                error,
            )
            request["skipPreRolloutVisualComparison"] = True
        else:
            request["preRolloutReference"] = reference.as_request()

    errors: list[str] = []
    last_result: dict[str, Any] | None = None
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        result, error = await _run_agens_attempt(
            node_bin=node_bin,
            agens_cli=agens_cli_path,
            agens_cwd=agens_cwd_path,
            verl_cwd=repo_root,
            request=request,
            timeout=attempt_timeout,
        )
        if result is not None:
            last_result = result
            score = result.get("score")
            training_scores = _training_scores(result, reference_penalty_lambda)
            if result.get("ok") is True and training_scores is not None:
                training_score, global_score, applied_reference_penalty = training_scores
                reward = {
                    "score": training_score,
                    "global_score": global_score,
                    **_pre_rollout_score_components(result),
                    "transition_score": _score_component(result, "transitionScore"),
                    "screenshot_score": _score_component(result, "screenshotScore"),
                    "agens_ok": True,
                    "agens_skipped": False,
                    "agens_attempts": attempt,
                    "agens_run_dir": (result.get("paths") or {}).get("sampleDir"),
                    "agens_reward_result": (result.get("paths") or {}).get("rewardResultPath"),
                }
                if applied_reference_penalty is not None:
                    reward["applied_reference_penalty"] = applied_reference_penalty
                if trace_requested:
                    try:
                        reward["agens_trace"] = _build_trace_envelope(
                            result,
                            solution_str=solution_str,
                            html=html,
                            html_char_start=html_char_start,
                            html_char_end=html_char_end,
                        )
                    except Exception as error:
                        logger.warning("[AgensReward] action trace metadata failed; keeping scalar reward: %s", error)
                        reward["agens_trace"] = {
                            "schema_version": 1,
                            "status": "failed",
                            "runs": [],
                            "errors": [str(error)],
                        }
                return reward
            errors.append(f"attempt {attempt}: agens returned ok={result.get('ok')} score={score}")
            result_errors = _compact(result.get("errors")) if result.get("errors") else "<none>"
            if result.get("errors"):
                errors.append(f"attempt {attempt} errors: {result_errors}")
            logger.warning(
                "[AgensReward] attempt %d/%d failed for prompt_id=%s sample_id=%s: ok=%r score=%r errors=%s",
                attempt,
                attempts,
                request["promptId"],
                request["sampleId"],
                result.get("ok"),
                score,
                result_errors,
            )
        else:
            errors.append(f"attempt {attempt}: {error}")
            logger.warning(
                "[AgensReward] attempt %d/%d failed for prompt_id=%s sample_id=%s: %s",
                attempt,
                attempts,
                request["promptId"],
                request["sampleId"],
                error,
            )

    logger.warning(
        "[AgensReward] all %d attempts failed for prompt_id=%s sample_id=%s; returning score=0.0",
        attempts,
        request["promptId"],
        request["sampleId"],
    )
    last_result = last_result or {}
    return {
        "score": 0.0,
        "global_score": _score_component(last_result, "globalScore"),
        **_pre_rollout_score_components(last_result),
        "transition_score": _score_component(last_result, "transitionScore"),
        "screenshot_score": _score_component(last_result, "screenshotScore"),
        "agens_ok": False,
        "agens_skipped": True,
        "agens_attempts": attempts,
        "agens_skip_reason": "agens simulate-reward failed after retries",
        "agens_errors": _compact(errors),
        "agens_run_dir": (last_result.get("paths") or {}).get("sampleDir"),
        "agens_reward_result": (last_result.get("paths") or {}).get("rewardResultPath"),
    }


async def _run_agens_attempt(
    *,
    node_bin: str,
    agens_cli: Path,
    agens_cwd: Path,
    verl_cwd: Path,
    request: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any] | None, str | None]:
    cmd = [
        node_bin,
        str(agens_cli),
        "simulate-reward",
        "--request-stdin",
        "--json",
        "--cwd",
        str(verl_cwd),
    ]
    env = os.environ.copy()
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(agens_cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(json.dumps(request).encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        await _kill_process_group(proc)
        return None, f"timeout after {timeout}s"
    except asyncio.CancelledError:
        await _kill_process_group(proc)
        raise
    except Exception as exc:
        await _kill_process_group(proc)
        return None, str(exc)

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        return None, f"exit={proc.returncode} stderr={_compact(stderr_text)}"
    try:
        return _parse_json_object(stdout_text), None
    except Exception as exc:
        return None, f"failed to parse stdout: {exc}; stderr={_compact(stderr_text)}"


async def _kill_process_group(proc: asyncio.subprocess.Process | None) -> None:
    """Kill an agens subprocess and every descendant in its process group."""
    if proc is None:
        return

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        # Fall back to killing the direct child if process-group signaling is
        # unavailable. The subprocess is normally its own group leader because
        # it is created with start_new_session=True.
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    try:
        await asyncio.wait_for(proc.wait(), timeout=PROCESS_CLEANUP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("[AgensReward] timed out waiting for subprocess pid=%d to exit", proc.pid)
    except Exception:
        logger.warning("[AgensReward] failed waiting for subprocess pid=%d to exit", proc.pid, exc_info=True)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("agens stdout JSON must be an object")
    return parsed


def _extract_html_with_span(solution: str) -> tuple[str, int, int]:
    # Thinking-capable Qwen models emit the private reasoning and final answer
    # in one decoded response. Keep the original response tokens untouched for
    # GRPO, but expose only the post-thinking answer to artifact storage and
    # agens evaluation. Use the last closing tag defensively in case reasoning
    # text itself mentions the marker.
    start = 0
    think_ends = list(re.finditer(r"</think\s*>", solution, flags=re.IGNORECASE))
    if think_ends:
        start = think_ends[-1].end()
    while start < len(solution) and solution[start].isspace():
        start += 1

    fenced = re.search(r"```(?:html)?\s*(.*?)```", solution[start:], flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        html_start = start + fenced.start(1)
        html_end = start + fenced.end(1)
        while html_start < html_end and solution[html_start].isspace():
            html_start += 1
        while html_end > html_start and solution[html_end - 1].isspace():
            html_end -= 1
        return solution[html_start:html_end], html_start, html_end
    return solution[start:], start, len(solution)


def _extract_html(solution: str) -> str:
    return _extract_html_with_span(solution)[0]


def _build_trace_envelope(
    result: dict[str, Any],
    *,
    solution_str: str,
    html: str,
    html_char_start: int,
    html_char_end: int,
) -> dict[str, Any]:
    """Validate artifact-byte attribution and shift it into decoded-response byte coordinates."""
    response_bytes = solution_str.encode("utf-8")
    html_bytes = html.encode("utf-8")
    byte_start = len(solution_str[:html_char_start].encode("utf-8"))
    byte_end = len(solution_str[:html_char_end].encode("utf-8"))
    base = {
        "schema_version": 1,
        "status": "failed",
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "response_utf8_byte_length": len(response_bytes),
        "artifact_sha256": hashlib.sha256(html_bytes).hexdigest(),
        "artifact_utf8_byte_length": len(html_bytes),
        "artifact_response_span": {"start": byte_start, "end": byte_end},
        "runs": [],
    }
    errors: list[str] = []
    if byte_end - byte_start != len(html_bytes) or response_bytes[byte_start:byte_end] != html_bytes:
        errors.append("extracted HTML does not match its decoded-response UTF-8 byte span")

    attribution = result.get("actionTraceAttribution")
    if not isinstance(attribution, dict):
        errors.append("Agens result did not contain actionTraceAttribution")
    else:
        if attribution.get("schemaVersion") != 1:
            errors.append("unsupported actionTraceAttribution schemaVersion")
        if attribution.get("offsetEncoding") != "utf-8-bytes":
            errors.append("action trace offsets are not UTF-8 byte offsets")
        if attribution.get("artifactSha256") != base["artifact_sha256"]:
            errors.append("action trace artifact SHA-256 does not match extracted HTML")
        if attribution.get("artifactUtf8ByteLength") != len(html_bytes):
            errors.append("action trace artifact byte length does not match extracted HTML")

        shifted_runs = []
        previous_end = 0
        raw_runs = attribution.get("runs")
        if not isinstance(raw_runs, list):
            errors.append("action trace runs must be a list")
        else:
            for index, run in enumerate(raw_runs):
                if not isinstance(run, dict):
                    errors.append(f"action trace run {index} must be an object")
                    break
                start = run.get("utf8ByteStart")
                end = run.get("utf8ByteEnd")
                depth = run.get("depth")
                valid_ints = all(
                    isinstance(value, int) and not isinstance(value, bool) for value in (start, end, depth)
                )
                if not valid_ints or start < previous_end or end <= start or end > len(html_bytes) or depth < 0:
                    errors.append(f"action trace run {index} has invalid or overlapping offsets")
                    break
                shifted_runs.append({"start": byte_start + start, "end": byte_start + end, "depth": depth})
                previous_end = end
            else:
                base["runs"] = shifted_runs

        status = attribution.get("status")
        if status in {"complete", "partial", "empty", "failed"}:
            base["status"] = status
        else:
            errors.append("action trace attribution has an invalid status")
        for source, destination in (
            ("selectedActionCount", "selected_action_count"),
            ("attributedActionCount", "attributed_action_count"),
        ):
            value = attribution.get(source)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                base[destination] = value
        if attribution.get("warnings"):
            base["warnings"] = attribution["warnings"]
        if attribution.get("errors"):
            base["trace_errors"] = attribution["errors"]

    if errors:
        base["status"] = "failed"
        base["runs"] = []
        base["errors"] = errors
    return base


def _task_prompt(extra_info: dict[str, Any], ground_truth: Any) -> str:
    for key in ("task", "task_prompt", "prompt", "question"):
        value = extra_info.get(key)
        if isinstance(value, str) and value:
            return value
    if isinstance(ground_truth, dict):
        for key in ("task", "task_prompt", "prompt", "question"):
            value = ground_truth.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _compact(value: Any, limit: int = 2000) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    value = value.replace("\x00", "")
    return value[-limit:]
