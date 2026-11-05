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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.trainer.ppo.v1.agens_reward_metrics import AGENS_REWARD_COMPONENTS_TAG
from verl.trainer.ppo.v1.agent_loop_tq import (
    AgentLoopWorkerTQ,
    _agens_failure_filter_enabled,
    _agens_reward_failure_reason,
    _agens_trace_advantage_config,
    _is_agens_reward_failure,
    _settle_session_tasks,
)


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


@pytest.mark.parametrize(
    "reward_extra_info",
    [
        {"agens_ok": False, "agens_skipped": True},
        {"timeout": True, "acc": 0.0},
        {"error": "judge API failed", "acc": 0.0},
    ],
)
def test_agens_reward_failure_recognizes_final_failure_modes(reward_extra_info):
    assert _is_agens_reward_failure(reward_extra_info)


@pytest.mark.parametrize(
    "reward_extra_info",
    [
        {"agens_ok": True, "score": 0.0},
        {"agens_ok": True, "agens_trace": {"status": "failed"}, "score": 0.5},
        {"agens_ok": True, "agens_attempts": 2, "score": 0.5},
        {},
        None,
    ],
)
def test_agens_reward_failure_keeps_valid_or_unmarked_rewards(reward_extra_info):
    assert not _is_agens_reward_failure(reward_extra_info)


def test_agens_failure_filter_is_opt_in():
    disabled = OmegaConf.create({"reward": {}})
    enabled = OmegaConf.create({"reward": {"agens_failure_filter": {"enabled": True}}})

    assert not _agens_failure_filter_enabled(disabled)
    assert _agens_failure_filter_enabled(enabled)


def test_agens_trace_advantage_forwards_only_capture_parameters():
    config = OmegaConf.create(
        {
            "reward": {
                "agens_trace_advantage": {
                    "enabled": True,
                    "threshold": 75,
                    "beta": 0.7,
                    "gamma": 0.8,
                    "max_pause_events_per_action": 123,
                }
            }
        }
    )

    assert _agens_trace_advantage_config(config) == {
        "enabled": True,
        "threshold": 75,
        "max_pause_events_per_action": 123,
    }


def test_agens_reward_failure_reason_prefers_manager_failure_details():
    assert _agens_reward_failure_reason({"timeout": True, "agens_errors": "later"}) == "reward computation timed out"
    assert _agens_reward_failure_reason({"error": "connection reset"}) == "connection reset"


def test_postprocess_drops_final_reward_failure_before_materialization():
    worker_cls = AgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = worker_cls.__new__(worker_cls)
    worker.config = OmegaConf.create({"reward": {"agens_failure_filter": {"enabled": True}}})
    output = SimpleNamespace(extra_fields={})

    async def compute_score(outputs, kwargs):
        outputs[-1].extra_fields["reward_extra_info"] = {
            "agens_ok": False,
            "agens_skipped": True,
            "agens_errors": "judge API unavailable",
        }

    worker._compute_score = AsyncMock(side_effect=compute_score)
    worker._compute_teacher_logprobs = AsyncMock()

    asyncio.run(
        worker._agent_loop_postprocess(
            output,
            validate=False,
            uid="prompt-1",
            session_id=2,
        )
    )

    worker._compute_score.assert_awaited_once()
    worker._compute_teacher_logprobs.assert_not_awaited()


def test_postprocess_attaches_reward_components_only_to_final_training_output():
    worker_cls = AgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = worker_cls.__new__(worker_cls)
    worker.config = OmegaConf.create({"reward": {}})
    outputs = [
        AgentLoopOutput(
            prompt_ids=[1], response_ids=[2], response_mask=[1], metrics=AgentLoopMetrics(), extra_fields={}
        ),
        AgentLoopOutput(
            prompt_ids=[1], response_ids=[3], response_mask=[1], metrics=AgentLoopMetrics(), extra_fields={}
        ),
    ]

    async def compute_score(scored_outputs, kwargs):
        scored_outputs[-1].reward_score = 0.8
        scored_outputs[-1].extra_fields["reward_extra_info"] = {
            "agens_ok": True,
            "transition_score": 81,
            "screenshot_score": 73,
            "unpenalized_global_score": 77,
            "pre_rollout_visual_penalty": 5,
            "applied_reference_penalty": 2.5,
        }

    worker._compute_score = AsyncMock(side_effect=compute_score)
    worker._compute_teacher_logprobs = AsyncMock()
    worker._compute_multi_modal_inputs = MagicMock(return_value={})
    worker._compute_position_ids = MagicMock(return_value=torch.tensor([[0, 1]]))
    batch_put = AsyncMock()

    with (
        patch("verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_batch_put", batch_put),
        patch("verl.trainer.ppo.v1.agent_loop_tq.list_of_dict_to_tensordict", return_value=MagicMock()),
    ):
        asyncio.run(
            worker._agent_loop_postprocess(
                outputs,
                validate=False,
                uid="prompt-1",
                session_id=2,
                global_steps=7,
            )
        )

    tags = batch_put.await_args.kwargs["tags"]
    assert AGENS_REWARD_COMPONENTS_TAG not in tags[0]
    assert tags[1][AGENS_REWARD_COMPONENTS_TAG] == {
        "transition_score": 81.0,
        "screenshot_score": 73.0,
        "unpenalized_global_score": 77.0,
        "reference_penalty": 5.0,
        "applied_reference_penalty": 2.5,
    }
