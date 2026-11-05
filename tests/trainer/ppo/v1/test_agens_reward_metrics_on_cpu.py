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

import pytest

from verl.trainer.ppo.v1.agens_reward_metrics import (
    AGENS_REWARD_COMPONENTS_TAG,
    compute_agens_reward_component_metrics,
    extract_agens_reward_components,
)


def test_extract_agens_reward_components_maps_reference_penalty_and_rejects_nonfinite_values():
    components = extract_agens_reward_components(
        {
            "transition_score": 80,
            "screenshot_score": 72.5,
            "unpenalized_global_score": float("nan"),
            "pre_rollout_visual_penalty": 6,
            "applied_reference_penalty": 3,
        }
    )

    assert components == {
        "transition_score": 80.0,
        "screenshot_score": 72.5,
        "reference_penalty": 6.0,
        "applied_reference_penalty": 3.0,
    }


def test_compute_agens_reward_component_metrics_averages_available_final_rollouts_only():
    tags = [
        {},  # Earlier output from the first multi-output rollout.
        {
            AGENS_REWARD_COMPONENTS_TAG: {
                "transition_score": 10.0,
                "screenshot_score": 20.0,
                "unpenalized_global_score": 30.0,
                "reference_penalty": 4.0,
                "applied_reference_penalty": 2.0,
            }
        },
        {
            AGENS_REWARD_COMPONENTS_TAG: {
                "transition_score": 30.0,
                "screenshot_score": 40.0,
                "unpenalized_global_score": 50.0,
                "reference_penalty": 6.0,
                "applied_reference_penalty": 3.0,
            }
        },
        {
            "is_padding": True,
            AGENS_REWARD_COMPONENTS_TAG: {
                "transition_score": 100.0,
                "screenshot_score": 100.0,
                "unpenalized_global_score": 100.0,
                "reference_penalty": 100.0,
                "applied_reference_penalty": 50.0,
            },
        },
    ]

    assert compute_agens_reward_component_metrics(tags) == {
        "training/agens_reward/transition_score/mean": 20.0,
        "training/agens_reward/screenshot_score/mean": 30.0,
        "training/agens_reward/unpenalized_global_score/mean": 40.0,
        "training/agens_reward/reference_penalty/mean": 5.0,
        "training/agens_reward/applied_reference_penalty/mean": 2.5,
    }


def test_compute_agens_reward_component_metrics_uses_per_component_population():
    tags = [
        {AGENS_REWARD_COMPONENTS_TAG: {"transition_score": 20.0}},
        {
            AGENS_REWARD_COMPONENTS_TAG: {
                "transition_score": 40.0,
                "screenshot_score": 70.0,
                "unpenalized_global_score": float("inf"),
            }
        },
    ]

    metrics = compute_agens_reward_component_metrics(tags)

    assert metrics["training/agens_reward/transition_score/mean"] == 30.0
    assert metrics["training/agens_reward/screenshot_score/mean"] == 70.0
    assert metrics["training/agens_reward/unpenalized_global_score/mean"] == 0.0
    assert metrics["training/agens_reward/reference_penalty/mean"] == 0.0
    assert metrics["training/agens_reward/applied_reference_penalty/mean"] == 0.0


@pytest.mark.parametrize("tags", [[], [{"is_padding": True}]])
def test_compute_agens_reward_component_metrics_returns_zero_for_empty_step(tags):
    assert compute_agens_reward_component_metrics(tags) == {
        "training/agens_reward/transition_score/mean": 0.0,
        "training/agens_reward/screenshot_score/mean": 0.0,
        "training/agens_reward/unpenalized_global_score/mean": 0.0,
        "training/agens_reward/reference_penalty/mean": 0.0,
        "training/agens_reward/applied_reference_penalty/mean": 0.0,
    }
