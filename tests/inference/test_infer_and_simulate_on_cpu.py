import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from inference import infer_and_simulate as runner
from scripts.html_game_dataset import HTML_GAME_SYSTEM_PROMPT


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = vars(
        runner.parse_args(
            [
                "--input",
                str(tmp_path / "input.parquet"),
                "--output-root",
                str(tmp_path / "output"),
            ]
        )
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_normalize_string_prompt_and_build_training_style_request(tmp_path: Path) -> None:
    sample = runner.normalize_sample(7, "Build a maze game.")
    request = runner.chat_request(sample, _args(tmp_path))

    assert sample.prompt_id == "000007"
    assert sample.messages == [
        {"role": "system", "content": HTML_GAME_SYSTEM_PROMPT},
        {"role": "user", "content": "Build a maze game."},
    ]
    assert request["n"] == 1
    assert request["temperature"] == 0.6
    assert request["top_p"] == 0.95
    assert request["top_k"] == 20
    assert request["chat_template_kwargs"] == {"enable_thinking": True}


def test_normalize_chat_prompt_preserves_system_and_extracts_last_user() -> None:
    messages = [
        {"role": "system", "content": "Existing system prompt."},
        {"role": "user", "content": "First request."},
        {"role": "assistant", "content": "Question."},
        {"role": "user", "content": [{"type": "text", "text": "Final request."}]},
    ]

    sample = runner.normalize_sample(0, messages)

    assert sample.messages == messages
    assert sample.task_prompt == "Final request."


def test_load_parquet_chat_prompt_round_trips_into_normalizer(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    pd.DataFrame({"prompt": [[{"role": "user", "content": "Build a runner game."}]]}).to_parquet(input_path)

    prompts = runner.load_prompts(input_path, "prompt")
    sample = runner.normalize_sample(0, prompts[0])

    assert sample.messages[0] == {"role": "system", "content": HTML_GAME_SYSTEM_PROMPT}
    assert sample.task_prompt == "Build a runner game."


def test_load_training_jsonl_prompt(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"prompt":"Build a quiz game.","metadata":{"source":"test"}}\n', encoding="utf-8")

    prompts = runner.load_prompts(input_path, "prompt")

    assert prompts == ["Build a quiz game."]


def test_initialize_state_requires_matching_config_for_resume(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    config = {"schema_version": 1, "model_path": "model-a"}

    assert runner.initialize_state(output_root, config, resume=False) == output_root.resolve()
    assert runner.initialize_state(output_root, config, resume=True) == output_root.resolve()

    with pytest.raises(ValueError, match="does not match"):
        runner.initialize_state(output_root, {**config, "model_path": "model-b"}, resume=True)
    with pytest.raises(FileExistsError, match="not empty"):
        runner.initialize_state(output_root, config, resume=False)


def test_generation_records_each_result_and_isolates_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [runner.normalize_sample(0, "First"), runner.normalize_sample(1, "Second")]
    args = _args(tmp_path, generation_concurrency=2)

    async def fake_submit(session, server_address, sample, actual_args):
        assert server_address in {"server-0", "server-1"}
        assert actual_args is args
        if sample.index == 1:
            raise RuntimeError("generation broke")
        return runner.GenerationResult(response="<html>ok</html>", finish_reason="stop", usage={"total_tokens": 4})

    monkeypatch.setattr(runner, "submit_generation", fake_submit)

    asyncio.run(runner.generate_samples(samples, ["server-0", "server-1"], args, tmp_path))

    success = runner.latest_event(tmp_path, "000000", "generation-success")
    failure = runner.latest_event(tmp_path, "000001", "generation-failure")
    assert success["response"] == "<html>ok</html>"
    assert failure["error"] == "generation broke"


def test_start_rollout_servers_returns_actor_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.workers.rollout import replica as replica_module

    created = []

    class FakeReplica:
        def __init__(self, replica_rank, **kwargs):
            self.server_handle = object()
            self.server_address = f"node:{3000 + replica_rank}"
            self.started = False
            created.append(self)

        async def init_standalone(self):
            self.started = True

    monkeypatch.setattr(replica_module, "get_rollout_replica_class", lambda name: FakeReplica)
    config = SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=2, nnodes=1),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(name="vllm", tensor_model_parallel_size=1),
            model=object(),
        ),
    )

    handles, addresses = asyncio.run(runner.start_rollout_servers(config))

    assert handles == [replica.server_handle for replica in created]
    assert addresses == ["node:3000", "node:3001"]
    assert all(replica.started for replica in created)


def test_simulation_resume_skips_success_and_retries_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [runner.normalize_sample(0, "First"), runner.normalize_sample(1, "Second")]
    args = _args(tmp_path, agens_concurrency=2)
    for sample in samples:
        runner.write_event(tmp_path, sample.prompt_id, "generation-success", {"response": "<html></html>"})

    calls = []

    async def first_compute_score(**kwargs):
        prompt_id = kwargs["extra_info"]["group_uid"]
        calls.append(prompt_id)
        assert kwargs["extra_info"]["session_id"] == 0
        assert kwargs["turn_limit"] == 8
        assert kwargs["max_retries"] == 0
        assert kwargs["output_root"] == str(tmp_path / "agens")
        if prompt_id == "000001":
            raise RuntimeError("simulation broke")
        return {"score": 0.8, "agens_ok": True}

    monkeypatch.setattr(runner, "compute_score", first_compute_score)
    asyncio.run(runner.simulate_samples(samples, args, tmp_path))

    assert sorted(calls) == ["000000", "000001"]
    assert runner.latest_event(tmp_path, "000000", "simulation-success") is not None
    failure = runner.latest_event(tmp_path, "000001", "simulation-failure")
    assert failure is not None
    failed_run_dir = Path(failure["result"]["agens_run_dir"])
    assert (failed_run_dir / "artifact" / "index.html").read_text(encoding="utf-8") == "<html></html>"
    assert (failed_run_dir / "artifact" / "prompt.txt").read_text(encoding="utf-8") == "Second"
    assert list((failed_run_dir / "trace").glob("*/simulation-failure.json"))
    assert (failed_run_dir / "reward-result.json").is_file()

    retry_calls = []

    async def retry_compute_score(**kwargs):
        retry_calls.append(kwargs["extra_info"]["group_uid"])
        return {"score": 0.9, "agens_ok": True}

    monkeypatch.setattr(runner, "compute_score", retry_compute_score)
    asyncio.run(runner.simulate_samples(samples, args, tmp_path))

    assert retry_calls == ["000001"]
    assert runner.latest_event(tmp_path, "000001", "simulation-success") is not None


def test_simulation_retains_artifacts_when_agens_returns_before_creating_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = runner.normalize_sample(11, "Build sample eleven.")
    args = _args(tmp_path)
    runner.write_event(
        tmp_path,
        sample.prompt_id,
        "generation-success",
        {"response": "<think>hidden</think>```html\n<html>eleven</html>\n```"},
    )

    async def failed_compute_score(**kwargs):
        return {
            "score": 0.0,
            "agens_ok": False,
            "agens_skipped": True,
            "agens_errors": "attempt 1: exit=1 stderr=EAGAIN",
            "agens_run_dir": None,
            "agens_reward_result": None,
        }

    monkeypatch.setattr(runner, "compute_score", failed_compute_score)
    asyncio.run(runner.simulate_samples([sample], args, tmp_path))

    failure = runner.latest_event(tmp_path, sample.prompt_id, "simulation-failure")
    result = failure["result"]
    run_dir = Path(result["agens_run_dir"])
    assert result["agens_artifacts_fallback"] is True
    assert (run_dir / "artifact" / "index.html").read_text(encoding="utf-8") == "<html>eleven</html>"
    assert (run_dir / "artifact" / "prompt.txt").read_text(encoding="utf-8") == "Build sample eleven."
    trace = runner.read_json(next((run_dir / "trace").glob("*/simulation-failure.json")))
    assert trace["errors"] == ["attempt 1: exit=1 stderr=EAGAIN"]
    reward_result = runner.read_json(run_dir / "reward-result.json")
    assert reward_result["ok"] is False
    assert reward_result["paths"]["artifactPath"] == str(run_dir / "artifact" / "index.html")


def test_rollout_config_uses_b200_defaults(tmp_path: Path) -> None:
    config = runner.build_rollout_config(_args(tmp_path))
    rollout = config.actor_rollout_ref.rollout

    assert config.trainer.n_gpus_per_node == 8
    assert rollout.name == "vllm"
    assert rollout.tensor_model_parallel_size == 1
    assert rollout.max_model_len == 34816
    assert rollout.max_num_batched_tokens == 34816
    assert rollout.max_num_seqs == 16
    assert rollout.engine_kwargs.vllm.moe_backend == "triton"
    assert rollout.engine_kwargs.vllm.language_model_only is True


def test_summary_limits_failed_prompt_preview(tmp_path: Path) -> None:
    samples = [runner.normalize_sample(index, f"Prompt {index}") for index in range(25)]

    summary = runner.summarize(samples, [], tmp_path)

    assert summary["failed"] == 25
    assert summary["failed_prompt_ids_preview"] == [f"{index:06d}" for index in range(20)]
    assert summary["failed_prompt_ids_omitted"] == 5
    assert "failed_prompt_ids" not in summary
