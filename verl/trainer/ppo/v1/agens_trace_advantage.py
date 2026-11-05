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

"""Map Agens source traces to original rollout tokens and redistribute GRPO credit."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)


def apply_agens_trace_token_advantages(
    *,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_lengths: Sequence[int],
    extra_fields: Sequence[Any],
    batch_keys: Sequence[str],
    tokenizer: Any,
    beta: float,
    gamma: float,
) -> dict[str, float]:
    """Mutate final-session advantages using trace depth, preserving each row's total credit."""
    final_indices = _final_session_indices(batch_keys)
    stats = {
        "training/agens_trace_advantage/applied_rollouts": 0.0,
        "training/agens_trace_advantage/partial_rollouts": 0.0,
        "training/agens_trace_advantage/trace_fallback_rollouts": 0.0,
        "training/agens_trace_advantage/alignment_fallback_rollouts": 0.0,
        "training/agens_trace_advantage/reweighted_tokens": 0.0,
        "training/agens_trace_advantage/eligible_tokens": 0.0,
    }
    failure_reasons: dict[str, int] = {}
    byte_piece_cache: dict[int, bytes | None] = {}

    for row in final_indices:
        mask_indices = torch.nonzero(response_mask[row] > 0, as_tuple=False).flatten()
        token_count = int(mask_indices.numel())
        stats["training/agens_trace_advantage/eligible_tokens"] += token_count
        if token_count == 0:
            _count_reason(failure_reasons, "empty response mask")
            stats["training/agens_trace_advantage/alignment_fallback_rollouts"] += 1
            continue

        trace = _trace_from_extra_fields(extra_fields[row] if row < len(extra_fields) else None)
        if not isinstance(trace, dict) or trace.get("status") not in {"complete", "partial"}:
            status = trace.get("status") if isinstance(trace, dict) else "missing"
            _count_reason(failure_reasons, f"trace status {status}")
            stats["training/agens_trace_advantage/trace_fallback_rollouts"] += 1
            continue
        if trace.get("status") == "partial":
            stats["training/agens_trace_advantage/partial_rollouts"] += 1
        if not trace.get("runs"):
            _count_reason(failure_reasons, "trace contains no attributed source runs")
            stats["training/agens_trace_advantage/trace_fallback_rollouts"] += 1
            continue

        response_length = int(response_lengths[row])
        ids = [int(token_id) for token_id in response_ids[row, :response_length].tolist()]
        try:
            spans = original_token_utf8_spans(
                tokenizer,
                ids,
                expected_sha256=trace.get("response_sha256"),
                expected_byte_length=trace.get("response_utf8_byte_length"),
                byte_piece_cache=byte_piece_cache,
            )
            runs = _validated_trace_runs(trace, spans[-1][1] if spans else 0)
            levels = token_trace_depths(spans, runs)
        except (TypeError, ValueError) as error:
            stats["training/agens_trace_advantage/alignment_fallback_rollouts"] += 1
            _count_reason(failure_reasons, str(error))
            continue

        scalar_advantage = float(advantages[row, int(mask_indices[0])].item())
        selected_levels = [levels[int(index)] for index in mask_indices.tolist()]
        raw_weights = torch.ones(token_count, dtype=advantages.dtype, device=advantages.device)
        finite_count = 0
        if scalar_advantage != 0:
            for local_index, depth in enumerate(selected_levels):
                if depth is None:
                    continue
                finite_count += 1
                trace_term = beta * (gamma**depth)
                raw_weights[local_index] = 1.0 + trace_term if scalar_advantage > 0 else max(1.0 - trace_term, 0.0)
            weight_sum = float(raw_weights.sum().item())
            if not torch.isfinite(raw_weights).all() or weight_sum <= 0:
                stats["training/agens_trace_advantage/trace_fallback_rollouts"] += 1
                _count_reason(failure_reasons, "trace weights have a non-positive or non-finite sum")
                continue
            raw_weights *= token_count / weight_sum
            weighted_advantage = raw_weights * scalar_advantage
            advantages[row, mask_indices] = weighted_advantage
            returns[row, mask_indices] = weighted_advantage

        stats["training/agens_trace_advantage/applied_rollouts"] += 1
        stats["training/agens_trace_advantage/reweighted_tokens"] += finite_count

    eligible = stats["training/agens_trace_advantage/eligible_tokens"]
    stats["training/agens_trace_advantage/reweighted_token_fraction"] = (
        stats["training/agens_trace_advantage/reweighted_tokens"] / eligible if eligible else 0.0
    )
    if failure_reasons:
        summary = ", ".join(f"{reason} ({count})" for reason, count in sorted(failure_reasons.items()))
        logger.warning("Agens trace advantage used uniform fallback: %s", summary)
    return stats


def original_token_utf8_spans(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    expected_sha256: Any,
    expected_byte_length: Any,
    byte_piece_cache: dict[int, bytes | None] | None = None,
) -> list[tuple[int, int]]:
    """Return UTF-8 byte spans from the original IDs without changing tokenization."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("trace response SHA-256 is missing or invalid")
    if isinstance(expected_byte_length, bool) or not isinstance(expected_byte_length, int) or expected_byte_length < 0:
        raise ValueError("trace response byte length is missing or invalid")

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    cache = byte_piece_cache if byte_piece_cache is not None else {}
    byte_spans: list[tuple[int, int]] = []
    pieces: list[bytes] = []
    cursor = 0
    byte_level_succeeded = True
    for token_id in token_ids:
        if token_id in special_ids:
            byte_spans.append((cursor, cursor))
            continue
        if token_id not in cache:
            token = tokenizer.convert_ids_to_tokens(token_id)
            cache[token_id] = _byte_level_piece(token) if isinstance(token, str) else None
        piece = cache[token_id]
        if piece is None:
            byte_level_succeeded = False
            break
        start = cursor
        cursor += len(piece)
        pieces.append(piece)
        byte_spans.append((start, cursor))

    if byte_level_succeeded:
        response_bytes = b"".join(pieces)
        try:
            _verify_response_bytes(response_bytes, expected_sha256, expected_byte_length)
        except ValueError:
            # ASCII-only token strings can superficially look byte-level even
            # for another tokenizer family. Require the decoded response hash,
            # then try the exact-ID fast-tokenizer path below.
            pass
        else:
            return byte_spans

    return _fast_tokenizer_exact_fallback(
        tokenizer,
        token_ids,
        special_ids=special_ids,
        expected_sha256=expected_sha256,
        expected_byte_length=expected_byte_length,
    )


def token_trace_depths(
    token_spans: Sequence[tuple[int, int]],
    runs: Sequence[tuple[int, int, int]],
) -> list[int | None]:
    """O(tokens + runs) conversion; any unmarked byte makes the token unmarked (INF)."""
    levels: list[int | None] = []
    run_index = 0
    for token_start, token_end in token_spans:
        if token_end <= token_start:
            levels.append(None)
            continue
        while run_index < len(runs) and runs[run_index][1] <= token_start:
            run_index += 1
        cursor = token_start
        max_depth = -1
        scan = run_index
        while scan < len(runs) and runs[scan][0] < token_end:
            run_start, run_end, depth = runs[scan]
            if run_start > cursor:
                break
            if run_end > cursor:
                cursor = min(token_end, run_end)
                max_depth = max(max_depth, depth)
            if cursor >= token_end:
                break
            scan += 1
        levels.append(max_depth if cursor >= token_end and max_depth >= 0 else None)
    return levels


def _validated_trace_runs(trace: dict[str, Any], response_byte_length: int) -> list[tuple[int, int, int]]:
    if trace.get("schema_version") != 1:
        raise ValueError("unsupported trace envelope schema")
    if trace.get("response_utf8_byte_length") != response_byte_length:
        raise ValueError("trace response length does not match original token bytes")
    raw_runs = trace.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("trace runs are missing")
    runs: list[tuple[int, int, int]] = []
    previous_end = 0
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise ValueError("trace run is not an object")
        start, end, depth = raw.get("start"), raw.get("end"), raw.get("depth")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end, depth)):
            raise ValueError("trace run offsets and depth must be integers")
        if start < previous_end or end <= start or end > response_byte_length or depth < 0:
            raise ValueError("trace runs are invalid or overlap")
        runs.append((start, end, depth))
        previous_end = end
    return runs


def _fast_tokenizer_exact_fallback(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    special_ids: set[int],
    expected_sha256: str,
    expected_byte_length: int,
) -> list[tuple[int, int]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("tokenizer is neither byte-level nor a fast tokenizer with exact offsets")
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    response_bytes = decoded.encode("utf-8")
    _verify_response_bytes(response_bytes, expected_sha256, expected_byte_length)
    encoded = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
    encoded_ids = list(encoded["input_ids"])
    non_special_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    if encoded_ids != non_special_ids:
        raise ValueError("re-tokenized response IDs do not exactly match the original non-special IDs")
    offsets = list(encoded["offset_mapping"])
    char_boundaries = sorted({value for pair in offsets for value in pair})
    byte_boundaries = _character_to_utf8_offsets(decoded, char_boundaries)
    spans: list[tuple[int, int]] = []
    offset_index = 0
    cursor = 0
    for token_id in token_ids:
        if token_id in special_ids:
            spans.append((cursor, cursor))
            continue
        char_start, char_end = offsets[offset_index]
        start, end = byte_boundaries[char_start], byte_boundaries[char_end]
        if start < cursor or end < start:
            raise ValueError("fast-tokenizer offsets are not monotonic")
        spans.append((start, end))
        cursor = end
        offset_index += 1
    return spans


def _byte_level_piece(token: str) -> bytes | None:
    output = bytearray()
    for character in token:
        value = _BYTE_LEVEL_UNICODE_TO_BYTE.get(character)
        if value is None:
            return None
        output.append(value)
    return bytes(output)


def _build_byte_level_inverse() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, unicode_values, strict=True)}


_BYTE_LEVEL_UNICODE_TO_BYTE = _build_byte_level_inverse()


def _verify_response_bytes(response_bytes: bytes, expected_sha256: str, expected_byte_length: int) -> None:
    if len(response_bytes) != expected_byte_length:
        raise ValueError("original token bytes do not match trace response byte length")
    if hashlib.sha256(response_bytes).hexdigest() != expected_sha256:
        raise ValueError("original token bytes do not match trace response SHA-256")


def _character_to_utf8_offsets(text: str, positions: Sequence[int]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    prior_character = 0
    prior_byte = 0
    for position in positions:
        prior_byte += len(text[prior_character:position].encode("utf-8"))
        offsets[position] = prior_byte
        prior_character = position
    return offsets


def _trace_from_extra_fields(extra_field: Any) -> Any:
    if not isinstance(extra_field, dict):
        return None
    reward_extra_info = extra_field.get("reward_extra_info")
    return reward_extra_info.get("agens_trace") if isinstance(reward_extra_info, dict) else None


def _final_session_indices(batch_keys: Sequence[str]) -> list[int]:
    final: dict[str, tuple[int, int]] = {}
    for row, key in enumerate(batch_keys):
        fields = key.rsplit("_", 2)
        if len(fields) != 3:
            raise ValueError(f"Unexpected TransferQueue key format: {key}")
        session_key = f"{fields[0]}_{fields[1]}"
        output_index = int(fields[2])
        if session_key not in final or output_index > final[session_key][0]:
            final[session_key] = (output_index, row)
    return [row for _, row in final.values()]


def _count_reason(reasons: dict[str, int], reason: str) -> None:
    reasons[reason] = reasons.get(reason, 0) + 1
