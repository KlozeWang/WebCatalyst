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

from ray.util.queue import Empty

from verl.trainer import main_ppo


class _Stdin:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def _pause_request() -> dict:
    return {
        "request_id": "7:request",
        "step": 7,
        "dropped_samples": 16,
        "effective_samples": 0,
        "threshold": 10,
        "all_samples_dropped": True,
    }


def test_confirm_agens_failure_pause_reads_from_interactive_terminal():
    input_fn = MagicMock()

    waited = main_ppo._confirm_agens_failure_pause(
        _pause_request(),
        stdin=_Stdin(is_tty=True),
        input_fn=input_fn,
    )

    assert waited is True
    prompt = input_fn.call_args.args[0]
    assert "Step 7 is complete" in prompt
    assert "All samples in the step were dropped" in prompt
    assert "Press Enter" in prompt


def test_confirm_agens_failure_pause_continues_without_tty():
    input_fn = MagicMock(side_effect=AssertionError("must not read non-interactive stdin"))

    with patch.object(main_ppo.logger, "warning") as warning:
        waited = main_ppo._confirm_agens_failure_pause(
            _pause_request(),
            stdin=_Stdin(is_tty=False),
            input_fn=input_fn,
        )

    assert waited is False
    input_fn.assert_not_called()
    warning.assert_called_once()


def test_confirm_agens_failure_pause_continues_on_eof():
    input_fn = MagicMock(side_effect=EOFError)

    with patch.object(main_ppo.logger, "warning") as warning:
        waited = main_ppo._confirm_agens_failure_pause(
            _pause_request(),
            stdin=_Stdin(is_tty=True),
            input_fn=input_fn,
        )

    assert waited is False
    warning.assert_called_once()


def test_driver_relays_pause_request_and_matching_resume_id():
    run_ref = object()
    request = _pause_request()
    request_queue = MagicMock()
    request_queue.get_nowait.side_effect = [request, Empty()]
    resume_queue = MagicMock()

    with (
        patch.object(main_ppo.ray, "wait", side_effect=[([], []), ([run_ref], [])]),
        patch.object(main_ppo.ray, "get") as ray_get,
        patch.object(main_ppo, "_confirm_agens_failure_pause") as confirm,
    ):
        main_ppo._wait_for_task_runner_with_agens_failure_pause(run_ref, request_queue, resume_queue)

    confirm.assert_called_once_with(request)
    resume_queue.put.assert_called_once_with(request["request_id"])
    ray_get.assert_called_once_with(run_ref)
