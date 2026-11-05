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
import hashlib
import sys
from pathlib import Path

import pytest

from verl.utils.reward_score import agens_sim_reward
from verl.utils.reward_score.agens_sim_reward import _extract_html, _extract_html_with_span, _run_agens_attempt


def test_extract_html_removes_thinking_and_leading_blank_lines() -> None:
    solution = "<think>private reasoning\nwith details</think>\n\n   <!DOCTYPE html>\n<html></html>"

    assert _extract_html(solution) == "<!DOCTYPE html>\n<html></html>"


def test_extract_html_handles_fenced_answer_after_thinking() -> None:
    solution = "<think>reasoning that mentions </think></think>\n\n```html\n<!DOCTYPE html>\n<html></html>\n```"

    assert _extract_html(solution) == "<!DOCTYPE html>\n<html></html>"


def test_extract_html_preserves_non_thinking_answer_except_leading_whitespace() -> None:
    solution = "\n\n  <!DOCTYPE html>\n<html></html>\n"

    assert _extract_html(solution) == "<!DOCTYPE html>\n<html></html>\n"


def test_extract_html_span_is_exact_with_unicode_and_fences() -> None:
    solution = "<think>推理</think>\n```html\n  😀<html></html> \n``` trailing"

    html, start, end = _extract_html_with_span(solution)

    assert html == "😀<html></html>"
    assert solution[start:end] == html


def test_compute_score_uses_zero_for_missing_failed_components(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failed_attempt(**kwargs):
        return {
            "ok": False,
            "score": 0,
            "errors": ["Missing globalEvaluation.score"],
            "paths": {},
        }, None

    monkeypatch.setattr(agens_sim_reward, "_run_agens_attempt", failed_attempt)

    result = asyncio.run(
        agens_sim_reward.compute_score(
            data_source="test",
            solution_str="<html></html>",
            ground_truth=None,
            max_retries=0,
        )
    )

    assert result["score"] == 0.0
    assert result["global_score"] == 0.0
    assert result["transition_score"] == 0.0
    assert result["screenshot_score"] == 0.0
    assert result["agens_ok"] is False
    assert result["agens_skipped"] is True


def test_compute_score_keeps_reward_when_retry_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    async def recovering_attempt(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return None, "judge API unavailable"
        return {
            "ok": True,
            "score": 0.0,
            "globalScore": 0,
            "paths": {},
        }, None

    monkeypatch.setattr(agens_sim_reward, "_run_agens_attempt", recovering_attempt)

    result = asyncio.run(
        agens_sim_reward.compute_score(
            data_source="test",
            solution_str="<html></html>",
            ground_truth=None,
            max_retries=1,
        )
    )

    assert result["score"] == 0.0
    assert result["agens_ok"] is True
    assert result["agens_skipped"] is False
    assert result["agens_attempts"] == 2


def test_compute_score_shifts_action_trace_to_decoded_response_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    solution = "<think>推理</think>\n```html\n😀<script>go()</script>\n```"
    html = "😀<script>go()</script>"
    html_bytes = html.encode()
    function_start = html_bytes.index(b"go")

    async def successful_attempt(**kwargs):
        requests.append(kwargs["request"])
        return {
            "ok": True,
            "score": 0.75,
            "globalScore": 75,
            "paths": {},
            "actionTraceAttribution": {
                "schemaVersion": 1,
                "status": "complete",
                "offsetEncoding": "utf-8-bytes",
                "artifactSha256": hashlib.sha256(html_bytes).hexdigest(),
                "artifactUtf8ByteLength": len(html_bytes),
                "selectedActionCount": 1,
                "attributedActionCount": 1,
                "runs": [{"utf8ByteStart": function_start, "utf8ByteEnd": function_start + 2, "depth": 0}],
            },
        }, None

    monkeypatch.setattr(agens_sim_reward, "_run_agens_attempt", successful_attempt)
    result = asyncio.run(
        agens_sim_reward.compute_score(
            data_source="test",
            solution_str=solution,
            ground_truth=None,
            extra_info={
                "validate": False,
                "agens_trace_advantage": {
                    "enabled": True,
                    "threshold": 70,
                    "max_pause_events_per_action": 123,
                },
            },
            max_retries=0,
        )
    )

    assert requests[0]["actionTrace"] == {
        "enabled": True,
        "threshold": 70,
        "maxPauseEventsPerAction": 123,
    }
    trace = result["agens_trace"]
    span = trace["artifact_response_span"]
    assert solution.encode()[span["start"] : span["end"]] == html_bytes
    assert trace["runs"] == [
        {"start": span["start"] + function_start, "end": span["start"] + function_start + 2, "depth": 0}
    ]
    assert trace["response_sha256"] == hashlib.sha256(solution.encode()).hexdigest()


def test_compute_score_passes_reference_for_training_but_not_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    resolver_calls = []

    class Reference:
        def as_request(self):
            return {"screenshotPath": "/reference/before.png"}

    def fake_resolver(root, index, task_prompt):
        resolver_calls.append((root, index, task_prompt))
        return Reference()

    async def successful_attempt(**kwargs):
        requests.append(kwargs["request"])
        result = {
            "ok": True,
            "score": 0.5,
            "globalScore": 50,
            "paths": {},
        }
        if "preRolloutReference" in kwargs["request"]:
            result.update({"unpenalizedGlobalScore": 75, "preRolloutVisualPenalty": 25})
        return result, None

    monkeypatch.setattr(agens_sim_reward, "resolve_pre_rollout_reference", fake_resolver)
    monkeypatch.setattr(agens_sim_reward, "_run_agens_attempt", successful_attempt)

    training = asyncio.run(
        agens_sim_reward.compute_score(
            data_source="html_game",
            solution_str="<html></html>",
            ground_truth=None,
            extra_info={"index": 4, "task_prompt": "Build a game.", "validate": False},
            pre_rollout_results_root="/results",
            max_retries=0,
        )
    )
    validation = asyncio.run(
        agens_sim_reward.compute_score(
            data_source="html_game",
            solution_str="<html></html>",
            ground_truth=None,
            extra_info={"index": 4, "task_prompt": "Build a game.", "validate": True},
            pre_rollout_results_root="/results",
            max_retries=0,
        )
    )

    assert resolver_calls == [("/results", 4, "Build a game.")]
    assert requests[0]["preRolloutReference"] == {"screenshotPath": "/reference/before.png"}
    assert "referencePenaltyLambda" not in requests[0]
    assert "preRolloutReference" not in requests[1]
    assert training["score"] == 0.625
    assert training["global_score"] == 62.5
    assert training["unpenalized_global_score"] == 75
    assert training["pre_rollout_visual_penalty"] == 25
    assert training["applied_reference_penalty"] == 12.5
    assert validation["score"] == 0.5
    assert validation["global_score"] == 50
    assert "unpenalized_global_score" not in validation
    assert "pre_rollout_visual_penalty" not in validation
    assert "applied_reference_penalty" not in validation


@pytest.mark.parametrize(
    ("reference_penalty_lambda", "expected_global_score", "expected_applied_penalty"),
    [(0.0, 75.0, 0.0), (1.0, 50.0, 25.0)],
)
def test_compute_score_applies_reference_penalty_lambda_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    reference_penalty_lambda: float,
    expected_global_score: float,
    expected_applied_penalty: float,
) -> None:
    async def successful_attempt(**kwargs):
        return {
            "ok": True,
            "score": 0.5,
            "globalScore": 50,
            "unpenalizedGlobalScore": 75,
            "preRolloutVisualPenalty": 25,
            "paths": {},
        }, None

    monkeypatch.setattr(agens_sim_reward, "_run_agens_attempt", successful_attempt)

    result = asyncio.run(
        agens_sim_reward.compute_score(
            data_source="html_game",
            solution_str="<html></html>",
            ground_truth=None,
            reference_penalty_lambda=reference_penalty_lambda,
            max_retries=0,
        )
    )

    assert result["score"] == expected_global_score / 100
    assert result["global_score"] == expected_global_score
    assert result["pre_rollout_visual_penalty"] == 25
    assert result["applied_reference_penalty"] == expected_applied_penalty


@pytest.mark.parametrize("reference_penalty_lambda", [-0.01, 1.01, float("nan"), float("inf"), float("-inf")])
def test_compute_score_rejects_invalid_reference_penalty_lambda(reference_penalty_lambda: float) -> None:
    with pytest.raises(ValueError, match="reference_penalty_lambda must be a finite number between 0 and 1"):
        asyncio.run(
            agens_sim_reward.compute_score(
                data_source="html_game",
                solution_str="<html></html>",
                ground_truth=None,
                reference_penalty_lambda=reference_penalty_lambda,
                max_retries=0,
            )
        )


def test_compute_score_skips_missing_training_reference_with_zero_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def missing_resolver(root, index, task_prompt):
        raise agens_sim_reward.PreRolloutReferenceError("prompt has no usable screenshot")

    async def successful_attempt(**kwargs):
        requests.append(kwargs["request"])
        return {
            "ok": True,
            "score": 0.64,
            "globalScore": 64,
            "unpenalizedGlobalScore": 64,
            "preRolloutVisualPenalty": 0,
            "paths": {},
        }, None

    monkeypatch.setattr(agens_sim_reward, "resolve_pre_rollout_reference", missing_resolver)
    monkeypatch.setattr(agens_sim_reward, "_run_agens_attempt", successful_attempt)

    result = asyncio.run(
        agens_sim_reward.compute_score(
            data_source="html_game",
            solution_str="<html></html>",
            ground_truth=None,
            extra_info={"index": 11, "task_prompt": "Build a game.", "validate": False},
            pre_rollout_results_root="/results",
            max_retries=0,
        )
    )

    assert len(requests) == 1
    assert requests[0]["promptId"] == 11
    assert requests[0]["skipPreRolloutVisualComparison"] is True
    assert "preRolloutReference" not in requests[0]
    assert result["score"] == 0.64
    assert result["global_score"] == 64
    assert result["unpenalized_global_score"] == 64
    assert result["pre_rollout_visual_penalty"] == 0
    assert result["applied_reference_penalty"] == 0


def _process_is_running(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        state = stat_path.read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


async def _wait_for_file(path: Path, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.01)


async def _wait_for_process_exit(pid: int, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while _process_is_running(pid):
            await asyncio.sleep(0.01)


def test_cancellation_kills_agens_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        pid_file = tmp_path / "pids"
        script = tmp_path / "fake_agens.py"
        script.write_text(
            """
import os
from pathlib import Path
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(os.environ["AGENS_TEST_PID_FILE"]).write_text(f"{os.getpid()} {child.pid}")
sys.stdin.buffer.read()
time.sleep(60)
"""
        )
        monkeypatch.setenv("AGENS_TEST_PID_FILE", str(pid_file))

        task = asyncio.create_task(
            _run_agens_attempt(
                node_bin=sys.executable,
                agens_cli=script,
                agens_cwd=tmp_path,
                verl_cwd=tmp_path,
                request={},
                timeout=60,
            )
        )
        await _wait_for_file(pid_file)
        parent_pid, child_pid = (int(pid) for pid in pid_file.read_text().split())

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await _wait_for_process_exit(parent_pid)
        await _wait_for_process_exit(child_pid)

    asyncio.run(run_test())
