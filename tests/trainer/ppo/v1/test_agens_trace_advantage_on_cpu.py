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

import hashlib

import pytest
import torch

from verl.trainer.ppo.v1.agens_trace_advantage import (
    apply_agens_trace_token_advantages,
    original_token_utf8_spans,
    token_trace_depths,
)


class _ByteLevelTokenizer:
    all_special_ids = [99]
    is_fast = False
    pad_token_id = 0
    pieces = {1: "<think>", 2: "abc", 3: "def", 99: "<eos>"}

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[token_id]


class _FastNonByteTokenizer:
    all_special_ids = []
    is_fast = True

    def convert_ids_to_tokens(self, token_id):
        return {1: "hello", 2: "world"}[token_id]

    def decode(self, token_ids, skip_special_tokens=True):
        return "hello world"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        assert text == "hello world"
        return {"input_ids": [1, 2], "offset_mapping": [(0, 5), (6, 11)]}


def _trace(response: bytes, runs: list[dict], status: str = "complete") -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "response_utf8_byte_length": len(response),
        "runs": runs,
    }


def test_original_token_spans_preserve_special_tokens_as_empty_spans() -> None:
    response = b"<think>abcdef"
    spans = original_token_utf8_spans(
        _ByteLevelTokenizer(),
        [1, 2, 99, 3],
        expected_sha256=hashlib.sha256(response).hexdigest(),
        expected_byte_length=len(response),
    )

    assert spans == [(0, 7), (7, 10), (10, 10), (10, 13)]


def test_ascii_non_byte_tokenizer_uses_exact_fast_tokenizer_fallback() -> None:
    response = b"hello world"

    spans = original_token_utf8_spans(
        _FastNonByteTokenizer(),
        [1, 2],
        expected_sha256=hashlib.sha256(response).hexdigest(),
        expected_byte_length=len(response),
    )

    assert spans == [(0, 5), (6, 11)]


def test_token_depth_requires_every_byte_to_be_marked_and_takes_maximum_depth() -> None:
    spans = [(0, 3), (3, 6), (6, 9)]

    assert token_trace_depths(spans, [(0, 1, 0), (1, 3, 2), (3, 5, 0), (6, 9, 1)]) == [2, None, 1]


@pytest.mark.parametrize(
    ("advantage", "expected_raw"),
    [
        (2.0, [1.0, 1.5, 1.25]),
        (-2.0, [1.0, 0.5, 0.75]),
    ],
)
def test_trace_advantage_reweights_only_final_output_and_preserves_credit(advantage, expected_raw) -> None:
    response = b"<think>abcdef"
    trace = _trace(
        response,
        [
            {"start": 7, "end": 10, "depth": 0},
            {"start": 10, "end": 13, "depth": 1},
        ],
    )
    advantages = torch.full((2, 3), advantage)
    returns = advantages.clone()
    mask = torch.ones((2, 3), dtype=torch.long)
    ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
    extra_fields = [
        {"reward_extra_info": {"agens_trace": trace}},
        {"reward_extra_info": {"agens_trace": trace}},
    ]

    metrics = apply_agens_trace_token_advantages(
        advantages=advantages,
        returns=returns,
        response_mask=mask,
        response_ids=ids,
        response_lengths=[3, 3],
        extra_fields=extra_fields,
        batch_keys=["prompt_0_0", "prompt_0_1"],
        tokenizer=_ByteLevelTokenizer(),
        beta=0.5,
        gamma=0.5,
    )

    expected = torch.tensor(expected_raw) * (3 / sum(expected_raw)) * advantage
    assert torch.equal(advantages[0], torch.full((3,), advantage))
    assert torch.allclose(advantages[1], expected)
    assert torch.equal(returns, advantages)
    assert advantages[1].sum().item() == pytest.approx(3 * advantage)
    assert metrics["training/agens_trace_advantage/applied_rollouts"] == 1
    assert metrics["training/agens_trace_advantage/reweighted_tokens"] == 2


def test_alignment_failure_keeps_uniform_advantage() -> None:
    response = b"<think>abcdef"
    trace = _trace(response, [{"start": 7, "end": 10, "depth": 0}])
    trace["response_sha256"] = "0" * 64
    advantages = torch.full((1, 3), 2.0)
    returns = advantages.clone()

    metrics = apply_agens_trace_token_advantages(
        advantages=advantages,
        returns=returns,
        response_mask=torch.ones((1, 3), dtype=torch.long),
        response_ids=torch.tensor([[1, 2, 3]]),
        response_lengths=[3],
        extra_fields=[{"reward_extra_info": {"agens_trace": trace}}],
        batch_keys=["prompt_0_0"],
        tokenizer=_ByteLevelTokenizer(),
        beta=0.5,
        gamma=0.5,
    )

    assert torch.equal(advantages, torch.full((1, 3), 2.0))
    assert torch.equal(returns, advantages)
    assert metrics["training/agens_trace_advantage/alignment_fallback_rollouts"] == 1
