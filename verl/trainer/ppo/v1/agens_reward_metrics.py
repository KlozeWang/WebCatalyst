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

"""Lightweight Agens reward-component metadata and training metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

AGENS_REWARD_COMPONENTS_TAG = "agens_reward_components"

_REWARD_COMPONENT_SOURCES = (
    ("transition_score", "transition_score"),
    ("screenshot_score", "screenshot_score"),
    ("unpenalized_global_score", "unpenalized_global_score"),
    ("reference_penalty", "pre_rollout_visual_penalty"),
    ("applied_reference_penalty", "applied_reference_penalty"),
)
_METRIC_PREFIX = "training/agens_reward"


def extract_agens_reward_components(reward_extra_info: Any) -> dict[str, float]:
    """Copy finite scalar reward components into a small rollout-tag payload."""
    if not isinstance(reward_extra_info, Mapping):
        return {}

    components = {}
    for metric_name, source_name in _REWARD_COMPONENT_SOURCES:
        value = _finite_real(reward_extra_info.get(source_name))
        if value is not None:
            components[metric_name] = value
    return components


def compute_agens_reward_component_metrics(tags: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Average each available component over final, non-padding rollout tags."""
    values: dict[str, list[float]] = {metric_name: [] for metric_name, _ in _REWARD_COMPONENT_SOURCES}
    for tag in tags:
        if not isinstance(tag, Mapping) or tag.get("is_padding", False):
            continue
        components = tag.get(AGENS_REWARD_COMPONENTS_TAG)
        if not isinstance(components, Mapping):
            continue
        for metric_name in values:
            value = _finite_real(components.get(metric_name))
            if value is not None:
                values[metric_name].append(value)

    return {
        f"{_METRIC_PREFIX}/{metric_name}/mean": math.fsum(component_values) / len(component_values)
        if component_values
        else 0.0
        for metric_name, component_values in values.items()
    }


def _finite_real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None
