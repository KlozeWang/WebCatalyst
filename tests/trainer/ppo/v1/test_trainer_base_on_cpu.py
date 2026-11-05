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

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.parameter_sync_step = 1
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo", "filter_groups": filter_groups},
            "actor_rollout_ref": {"rollout": {"n": 5, "temperature": 1.0}},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {
                "reward_manager": {"name": "rate_limited"},
                "custom_reward_function": {"path": "/repo/verl/utils/reward_score/agens_sim_reward.py"},
                "reward_model": {"enable": False, "enable_resource_pool": False},
            },
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_agens_failure_filter_is_forwarded_to_builtin_sync_sampler():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.reward.agens_failure_filter = {"enabled": True, "min_valid_rollouts": 4}

    sampler = trainer._build_replay_buffer()

    assert sampler.agens_min_valid_rollouts == 4


def test_agens_reward_metrics_are_enabled_only_for_agens_reward_function():
    trainer = _trainer_with_filter_groups({"enable": False})

    assert trainer._uses_agens_sim_reward()

    trainer.config.reward.custom_reward_function.path = "/repo/verl/utils/reward_score/other_reward.py"
    assert not trainer._uses_agens_sim_reward()


def test_agens_failure_filter_rejects_threshold_above_rollout_n():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.reward.agens_failure_filter = {"enabled": True, "min_valid_rollouts": 6}

    with pytest.raises(ValueError, match="cannot exceed"):
        trainer._build_replay_buffer()


def test_agens_failure_filter_rejects_custom_sampler():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.reward.agens_failure_filter = {"enabled": True, "min_valid_rollouts": 4}
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with pytest.raises(ValueError, match="built-in V1 replay buffer"):
        trainer._build_replay_buffer()


def test_agens_failure_pause_threshold_is_validated_and_resolved():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.reward.agens_failure_filter = {
        "enabled": True,
        "min_valid_rollouts": 4,
        "pause_after_dropped_samples": 10,
    }

    assert trainer._resolve_agens_failure_pause_threshold() == 10


@pytest.mark.parametrize("threshold", [True, 0, -1, 1.5, "10"])
def test_agens_failure_pause_rejects_invalid_threshold(threshold):
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.reward.agens_failure_filter = {
        "enabled": True,
        "min_valid_rollouts": 4,
        "pause_after_dropped_samples": threshold,
    }

    with pytest.raises(ValueError, match="must be a positive integer"):
        trainer._resolve_agens_failure_pause_threshold()


def test_agens_failure_pause_requires_failure_filter():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.reward.agens_failure_filter = {
        "enabled": False,
        "min_valid_rollouts": 4,
        "pause_after_dropped_samples": 10,
    }

    with pytest.raises(ValueError, match="enabled=True"):
        trainer._resolve_agens_failure_pause_threshold()


def test_agens_trace_advantage_config_is_validated_and_resolved():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.reward.agens_trace_advantage = {
        "enabled": True,
        "threshold": 75,
        "beta": 0.4,
        "gamma": 0.8,
        "max_pause_events_per_action": 321,
    }

    assert trainer._resolve_agens_trace_advantage() == {
        "threshold": 75.0,
        "beta": 0.4,
        "gamma": 0.8,
        "max_pause_events_per_action": 321,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold", 101, "between 0 and 100"),
        ("beta", -0.1, "non-negative"),
        ("gamma", 1.1, "between 0 and 1"),
        ("max_pause_events_per_action", 0, "positive integer"),
    ],
)
def test_agens_trace_advantage_rejects_invalid_hyperparameters(field, value, message):
    trainer = _trainer_with_filter_groups({"enable": False})
    config = {
        "enabled": True,
        "threshold": 70,
        "beta": 0.5,
        "gamma": 0.9,
        "max_pause_events_per_action": 4096,
    }
    config[field] = value
    trainer.config.reward.agens_trace_advantage = config

    with pytest.raises(ValueError, match=message):
        trainer._resolve_agens_trace_advantage()


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_step_once_returns_before_training_when_agens_filter_empties_batch():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.global_steps = 3
    empty_batch = KVBatchMeta(partition_id="train", keys=[], tags=[])
    trainer.replay_buffer = MagicMock()
    trainer.replay_buffer.sample.return_value = (
        empty_batch,
        {"training/agens_reward_filter/skipped_batches": 1},
    )
    trainer.on_sample_begin = MagicMock()
    trainer.on_sample_end = MagicMock()
    trainer._balance_batch = MagicMock(side_effect=AssertionError("must not balance an empty batch"))

    metrics = {}
    result = trainer._step_once(metrics=metrics, timing_raw={}, sample_batch_size=64)

    assert result is empty_batch
    assert metrics["training/agens_reward_filter/skipped_batches"] == 1
    trainer.on_sample_begin.assert_called_once_with()
    trainer.on_sample_end.assert_called_once_with()
    trainer._balance_batch.assert_not_called()


@pytest.mark.parametrize(
    ("dropped_samples", "effective_samples", "all_samples_dropped"),
    [
        (10, 6, False),
        (16, 0, True),
    ],
)
def test_agens_failure_pause_blocks_at_threshold_and_covers_all_dropped_batch(
    dropped_samples, effective_samples, all_samples_dropped
):
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.agens_failure_pause_threshold = 10
    trainer.global_steps = 7
    trainer._agens_failure_pause_request_queue = MagicMock()
    trainer._agens_failure_pause_resume_queue = MagicMock()
    trainer._agens_failure_pause_resume_queue.get.side_effect = lambda: (
        trainer._agens_failure_pause_request_queue.put.call_args.args[0]["request_id"]
    )

    trainer._pause_after_agens_failure_drops(
        {
            "training/agens_reward_filter/dropped_samples": dropped_samples,
            "training/agens_reward_filter/effective_samples": effective_samples,
        },
        is_last_step=False,
    )

    request = trainer._agens_failure_pause_request_queue.put.call_args.args[0]
    assert request["step"] == 7
    assert request["dropped_samples"] == dropped_samples
    assert request["threshold"] == 10
    assert request["all_samples_dropped"] is all_samples_dropped
    trainer._agens_failure_pause_resume_queue.get.assert_called_once_with()


@pytest.mark.parametrize(
    ("dropped_samples", "is_last_step"),
    [
        (9, False),
        (10, True),
    ],
)
def test_agens_failure_pause_does_not_block_below_threshold_or_on_last_step(dropped_samples, is_last_step):
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.agens_failure_pause_threshold = 10
    trainer.global_steps = 7
    trainer._agens_failure_pause_request_queue = MagicMock()
    trainer._agens_failure_pause_resume_queue = MagicMock()

    trainer._pause_after_agens_failure_drops(
        {"training/agens_reward_filter/dropped_samples": dropped_samples},
        is_last_step=is_last_step,
    )

    trainer._agens_failure_pause_request_queue.put.assert_not_called()
    trainer._agens_failure_pause_resume_queue.get.assert_not_called()


def test_agens_failure_pause_ignores_validation_drop_metrics():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.agens_failure_pause_threshold = 10
    trainer.global_steps = 7
    trainer._agens_failure_pause_request_queue = MagicMock()
    trainer._agens_failure_pause_resume_queue = MagicMock()

    trainer._pause_after_agens_failure_drops(
        {"validation/agens_reward_filter/dropped_samples": 16},
        is_last_step=False,
    )

    trainer._agens_failure_pause_request_queue.put.assert_not_called()
    trainer._agens_failure_pause_resume_queue.get.assert_not_called()


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )
