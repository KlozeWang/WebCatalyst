# Copyright 2026 WebCatalyst Authors
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

"""Dataset adapter for JSONL records with a string ``prompt`` field."""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

HTML_GAME_SYSTEM_PROMPT = (
    "You are an expert HTML5 game developer. "
    "Respond with only the complete, self-contained HTML source code for the requested game. "
    "Start your response with <!DOCTYPE html> and end it with </html>. "
    "Do not include Markdown code fences, explanations, or any text outside the HTML document. "
    "Include all CSS and JavaScript in the HTML file."
)


class HtmlGameDataset(RLHFDataset):
    """Adapt compact HTML-game records to VERL's RLHF runtime schema."""

    _SOURCE_INDEX_KEY = "_html_game_source_index"

    def prepare_dataframe_before_sample_selection(self, dataframe):
        """Retain original row indices across sampling, filtering, and shuffling."""
        dataframe = super().prepare_dataframe_before_sample_selection(dataframe)
        if self._SOURCE_INDEX_KEY not in dataframe.column_names:
            dataframe = dataframe.add_column(self._SOURCE_INDEX_KEY, list(range(len(dataframe))))
        return dataframe

    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        """Declare this dataset text-only even when the model exposes a processor."""
        del cls, messages, image_patch_size, config
        return None, None, None

    @classmethod
    async def process_multi_modal_info(cls, messages, image_patch_size, config):
        """Text-only AgentLoop counterpart of :meth:`_process_multi_modal_info`."""
        return cls._process_multi_modal_info(messages, image_patch_size, config)

    def _build_messages(self, example: dict[str, Any], key: str) -> list[dict[str, str]]:
        prompt = example[key]
        if isinstance(prompt, str):
            example = dict(example)
            example[key] = [{"role": "user", "content": prompt}]

        messages = RLHFDataset._build_messages(self, example, key)
        if not any(message.get("role") == "system" for message in messages):
            messages = [{"role": "system", "content": HTML_GAME_SYSTEM_PROMPT}, *messages]
        return messages

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        source_index = int(row.pop(self._SOURCE_INDEX_KEY, item))
        raw_prompt = row["raw_prompt"]
        task_prompt = raw_prompt[-1]["content"]

        row["data_source"] = row.get("data_source") or "html_game"
        row["reward_model"] = row.get("reward_model") or {
            "style": "rule",
            "ground_truth": "",
        }

        extra_info = dict(row.get("extra_info") or {})
        extra_info["index"] = source_index
        extra_info["task_prompt"] = task_prompt
        row["extra_info"] = extra_info
        row["index"] = source_index

        # The original curation record is useful offline but not during
        # rollout/reward execution, where transferring it adds avoidable cost.
        row.pop("metadata", None)
        return row
