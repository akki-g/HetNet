"""Small real-domain audits of the new synchronous learner and resumability."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import io
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from softrole.config import Config
from softrole.model import SoftRoleNet
from softrole.train import (append_json, apply_gradients, log_episode_batch,
                            summarize, train, worker_collect)


def tiny_config(**overrides):
    settings = dict(task="pcp", num_p=2, num_a=1, dim=5, vision=1, max_steps=3,
                    experts=2, pre_dim=8, hidden_dim=8, heads=1, head_dim=4,
                    msg_dim=4, seed=23, epochs=1, updates_per_epoch=3,
                    batch_steps=2, nprocesses=1, save_every=1, detach_gap=2)
    settings.update(overrides)
    return Config(**settings)


@pytest.mark.parametrize("clip", [1., 1000.])
def test_unequal_collector_episode_counts_are_averaged_once_before_one_clip(monkeypatch, clip):
    model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    batches = [{"num_episodes": 1, "gradients": [np.array([[100., 0.]]), np.array([0.])]},
               {"num_episodes": 3, "gradients": [np.array([[-96., 8.]]), np.array([12.])]}]
    # The global episode mean is (1,2,3); local means or locally clipped
    # gradients would produce a different direction, not merely a scale.
    observed = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        observed.append(torch.cat([parameter.grad.flatten().clone() for parameter in parameters]))
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    norm = apply_gradients(model, optimizer, batches, clip)
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], torch.tensor([1., 2., 3.], dtype=torch.float64))
    assert norm == pytest.approx(np.sqrt(14.))
    expected = -0.5 * np.array([1., 2., 3.]) * min(1., clip / np.sqrt(14.))
    actual = torch.cat([model.weight.detach().flatten(), model.bias.detach()]).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_spawned_collectors_match_serial_frozen_snapshot_collections():
    config = tiny_config(nprocesses=2)
    torch.manual_seed(config.seed)
    model = SoftRoleNet(**config.model_kwargs()).double()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    serial = [worker_collect(state, config.model_kwargs(), config.to_dict(), rank, 7)
              for rank in range(2)]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        pending = [pool.submit(worker_collect, state, config.model_kwargs(), config.to_dict(), rank, 7)
                   for rank in range(2)]
        spawned = [future.result(timeout=60) for future in pending]
    for expected, actual in zip(serial, spawned):
        assert actual.keys() == expected.keys()
        for key in expected.keys() - {"gradients"}:
            assert actual[key] == expected[key]
        for actual_gradient, expected_gradient in zip(actual["gradients"], expected["gradients"]):
            np.testing.assert_array_equal(actual_gradient, expected_gradient)


def assert_state_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_state_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_state_equal(left, right)
    else:
        assert first == second


def test_partial_epoch_resume_matches_uninterrupted_model_and_optimizer(tmp_path):
    config = tiny_config()
    uninterrupted_path = train(config, tmp_path / "uninterrupted")
    partial_path = train(replace(config, total_steps=1), tmp_path / "partial")
    partial = torch.load(partial_path, map_location="cpu", weights_only=False)
    assert partial["updates"] == 1
    assert partial["completed_epochs"] == 0
    assert partial["updates_in_partial_epoch"] == 1
    resumed_path = train(config, tmp_path / "resumed", resume=partial_path, episode_log="stdout")
    uninterrupted = torch.load(uninterrupted_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    for field in ("updates", "total_steps", "total_episodes", "completed_epochs", "updates_in_partial_epoch"):
        assert uninterrupted[field] == resumed[field]
    assert resumed["updates"] == 3
    assert_state_equal(uninterrupted["model_state"], resumed["model_state"])
    assert_state_equal(uninterrupted["optimizer_state"], resumed["optimizer_state"])
    metadata = json.loads((tmp_path / "resumed" / "run.json").read_text())
    assert metadata["parent_source_sha256"] == partial["source_sha256"]
    assert metadata["episode_log"] == "stdout"
    assert not (tmp_path / "resumed" / "episodes.jsonl").exists()
    with pytest.raises(FileExistsError):
        train(config, tmp_path / "uninterrupted")


@pytest.mark.parametrize("override", [
    {"seed": 0.5}, {"vision": 1.5}, {"total_steps": 1.5},
    {"failure_window": (1,)}, {"failure_window": (1.5, 2)},
    {"failure_window": (-1, 2)}, {"failure_window": (3, 2)},
])
def test_malformed_programmatic_config_is_rejected_before_collection(override):
    with pytest.raises(ValueError):
        tiny_config(**override)


def test_batched_episode_file_retains_legacy_bytes_and_order_with_one_open(tmp_path, monkeypatch):
    records = [{"scenario_id": "second", "update": 7, "collector": 1},
               {"scenario_id": "first", "update": 7, "collector": 0}]
    legacy = tmp_path / "legacy.jsonl"
    for record in records:
        append_json(legacy, record)
    destination = tmp_path / "episodes.jsonl"
    opened = []
    original_open = Path.open

    def observe_open(path, mode="r", *args, **kwargs):
        if path == destination and mode == "a":
            opened.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe_open)
    log_episode_batch(destination, records, update=7)
    expected = (b'{"collector": 1, "scenario_id": "second", "update": 7}\n'
                b'{"collector": 0, "scenario_id": "first", "update": 7}\n')
    assert destination.read_bytes() == legacy.read_bytes() == expected
    assert opened == [destination]


def test_stdout_episode_batch_is_one_tagged_object_with_one_flush(tmp_path, monkeypatch):
    class RecordedStream(io.StringIO):
        flush_count = 0

        def flush(self):
            self.flush_count += 1
            super().flush()

    stream = RecordedStream()
    monkeypatch.setattr(sys, "stdout", stream)
    records = [{"scenario_id": "a", "collector": 0, "update": 4},
               {"scenario_id": "b", "collector": 1, "update": 4}]
    path = tmp_path / "episodes.jsonl"
    log_episode_batch(path, records, update=4, mode="stdout")
    assert len(stream.getvalue().splitlines()) == 1
    assert json.loads(stream.getvalue()) == {
        "record_type": "softrole_episode_batch", "schema_version": 1,
        "update": 4, "episodes": records}
    assert stream.flush_count == 1
    assert not path.exists()


@pytest.mark.parametrize("include_new_field", [False, True])
def test_mean_agent_return_weights_episodes_equally_with_variable_rosters(include_new_field):
    common = {"steps": 1, "success": False, "event_exposed": False}
    episodes = [dict(common, team_return=-6., num_agents=2),
                dict(common, team_return=-4., num_agents=4)]
    if include_new_field:
        episodes[0]["mean_agent_return"] = -3.
        episodes[1]["mean_agent_return"] = -1.
    result = summarize(episodes)
    # An episode's typical agent receives -3 or -1; equal episodes yield -2.
    # Pooling all six agents instead would give -10/6, a different estimand.
    assert result["mean_agent_return"] == -2.
    assert result["team_return"] == -5.
    assert result["mean_agent_return"] != result["team_return"] / result["num_agents"]


@pytest.mark.parametrize("nprocesses", [1, 2])
def test_logging_destination_preserves_training_and_complete_episode_ledger(tmp_path, capsys, monkeypatch,
                                                                         nprocesses):
    config = tiny_config(nprocesses=nprocesses, compositions=((1, 1), (2, 1), (2, 2)))
    opened = []
    original_open = Path.open

    def observe_open(path, mode="r", *args, **kwargs):
        if path.name == "episodes.jsonl" and mode == "a":
            opened.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe_open)
    checkpoints, outputs = {}, {}
    for mode in ("file", "stdout"):
        checkpoint = train(config, tmp_path / mode, episode_log=mode)
        checkpoints[mode] = torch.load(checkpoint, map_location="cpu", weights_only=False)
        outputs[mode] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        metadata = json.loads((tmp_path / mode / "run.json").read_text())
        assert metadata["episode_log"] == mode
        assert "episode_log" not in checkpoints[mode]["config"]
    for key in ("model_state", "optimizer_state", "config", "model_config", "updates",
                "total_steps", "total_episodes", "signature"):
        assert_state_equal(checkpoints["file"][key], checkpoints["stdout"][key])
    episodes = [json.loads(line) for line in (tmp_path / "file" / "episodes.jsonl").read_text().splitlines()]
    batches = [row for row in outputs["stdout"] if row.get("record_type") == "softrole_episode_batch"]
    assert [row["update"] for row in batches] == [1, 2, 3]
    assert [record for row in batches for record in row["episodes"]] == episodes
    assert len(episodes) == checkpoints["file"]["total_episodes"]
    assert sum(row["steps"] for row in episodes) == checkpoints["file"]["total_steps"]
    assert set(row["collector"] for row in episodes) == set(range(nprocesses))
    assert len(opened) == checkpoints["file"]["updates"]
    assert not (tmp_path / "stdout" / "episodes.jsonl").exists()
    for row in episodes:
        assert row["mean_agent_return"] == pytest.approx(sum(row["agent_returns"]) / row["num_agents"])
    metrics = {}
    for mode in ("file", "stdout"):
        metrics[mode] = [json.loads(line) for line in (tmp_path / mode / "metrics.jsonl").read_text().splitlines()]
        for row in metrics[mode]:
            row.pop("wall_time_seconds")
    assert metrics["file"] == metrics["stdout"]
    assert metrics["file"][0]["mean_agent_return"] == pytest.approx(
        sum(row["mean_agent_return"] for row in episodes) / len(episodes))
    assert (tmp_path / "file" / "updates.jsonl").read_bytes() == (tmp_path / "stdout" / "updates.jsonl").read_bytes()


def test_invalid_logging_mode_is_rejected_before_creating_run(tmp_path):
    path = tmp_path / "invalid"
    with pytest.raises(ValueError, match="episode_log"):
        train(tiny_config(), path, episode_log="none")
    assert not path.exists()


def test_cli_logging_option_is_runtime_only(tmp_path, monkeypatch, capsys):
    from softrole.__main__ import main

    calls = []
    monkeypatch.setattr("softrole.train.train", lambda *args, **kwargs: calls.append((args, kwargs)))
    main(["train", "--output", str(tmp_path / "stdout"), "--episode-log", "stdout"])
    assert calls[0][1] == {"episode_log": "stdout"}
    assert "episode_log" not in calls[0][0][0].to_dict()
    main(["train", "--output", str(tmp_path / "dry"), "--episode-log", "stdout", "--dry-run"])
    assert "episode_log" not in json.loads(capsys.readouterr().out)
    assert not (tmp_path / "dry").exists()
