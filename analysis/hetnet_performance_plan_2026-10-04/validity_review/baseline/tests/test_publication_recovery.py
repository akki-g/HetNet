"""Small actual CPU updates: recovery, random-neutral saves and buffered logs.

Run the isolated runtime in subprocesses so its historical import names cannot
replace root/SoftRole modules inside pytest.  No research budget or scheduler.
"""
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "publication_reconstruction/runtime"


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def without_time(value, field="wall_time_seconds"):
    return [{key: item for key, item in row.items() if key != field} for row in value]


def invoke(base, name, *, task="pcp", binary=False, collectors=1, spec="supplement-v1",
           resume=None, wall=0, milestones=(1,), episode_log="file", extra=()):
    output = base / name
    output.mkdir()
    manifest = base / "source_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"purpose": "isolated recovery test", "files": {
            p.relative_to(RUNTIME).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in RUNTIME.rglob("*.py")}}))
    command = [sys.executable, "-u", str(RUNTIME / "main.py"),
        "--publication_env_version", "corrected-v1", "--model_spec", spec,
        "--source_manifest", str(manifest), "--env_name", "fire_commander" if task == "fc" else "predator_capture",
        "--nfriendly_P", "2", "--nfriendly_A", "1", "--nagents", "3", "--hetgat", "--hetgat_a2c",
        "--seed", "7", "--nprocesses", str(collectors), "--num_epochs", "2", "--epoch_size", "3",
        "--batch_size", "4", "--max_steps", "3", "--detach_gap", "5", "--dim", "5",
        "--vision", "1" if task == "fc" else "2", "--hid_size", "128", "--lrate", "0.001" if spec == "supplement-v1" else "0.0001",
        "--save_every", "1", "--save_dir", str(output / "checkpoints"),
        "--metrics_file", str(output / "metrics.jsonl"), "--experiment_name", "test",
        "--episode_log", episode_log, "--wall_seconds", str(wall)]
    if task == "fc":
        command += ["--nfires", "1", "--reward_type", "3"]
    if binary:
        command += ["--use_binary", "--msg_dim", "16"]
    if milestones:
        command += ["--milestones", *map(str, milestones)]
    if resume:
        command += ["--resume_checkpoint", str(resume)]
    command += list(extra)
    environment = {**os.environ, "DGLBACKEND": "pytorch", "OPENBLAS_NUM_THREADS": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                            capture_output=True, timeout=100)
    (output / "console.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    status = json.loads((output / "run_status.json").read_text())
    checkpoint = torch.load(status["checkpoint"], map_location="cpu")
    return output, checkpoint, status


@pytest.fixture(scope="module", params=[
    ("pcp", False, 1, "public-code-v1"),
    ("pcp", True, 1, "supplement-v1"),
    ("pcp", True, 4, "supplement-v1"),
    ("fc", False, 4, "supplement-v1"),
])
def continued(request, tmp_path_factory):
    base = tmp_path_factory.mktemp("publication-recovery")
    task, binary, collectors, spec = request.param
    options = dict(task=task, binary=binary, collectors=collectors, spec=spec)
    full, expected, status = invoke(base, "uninterrupted", **options)
    milestone = next((full / "checkpoints").rglob("*_steps1.pt"))
    initial_bytes = milestone.read_bytes()
    resumed, actual, resumed_status = invoke(base, "continued", **options,
        resume=milestone, episode_log="stdout")
    assert milestone.read_bytes() == initial_bytes
    return base, options, full, expected, resumed, actual, status, resumed_status


def test_resume_restores_model_optimizer_counts_and_every_collector_rng(continued):
    _, options, full, expected, resumed, actual, status, resumed_status = continued
    for key in ["policy_net", "trainer", "log"]:
        same(expected[key], actual[key])
    for key in ["counts", "rng_states", "recorder_state", "milestones_reached"]:
        same(expected["recovery"][key], actual["recovery"][key])
    assert len(actual["recovery"]["rng_states"]) == options["collectors"]
    assert status["counts"] == resumed_status["counts"]
    assert status["counts"]["updates"] == 6
    assert resumed_status["stop_reason"] == "epoch_cap_completed"
    assert 0 < expected["recovery"]["active_time_seconds"] <= status["active_time_seconds"]
    assert status["active_time_seconds"] == status["segment_wall_time_seconds"]
    milestone = next((full / "checkpoints").rglob("*_steps1.pt"))
    previous_active = torch.load(milestone, map_location="cpu")["recovery"]["active_time_seconds"]
    assert previous_active < actual["recovery"]["active_time_seconds"] <= resumed_status["active_time_seconds"]
    assert resumed_status["active_time_seconds"] == previous_active + resumed_status["segment_wall_time_seconds"]
    optimizer_state = actual["trainer"]["state"]
    assert optimizer_state
    buffer = "exp_avg" if options["spec"] == "supplement-v1" else "square_avg"
    assert all(buffer in state for state in optimizer_state.values())
    for state in optimizer_state.values():
        assert int(state["step"]) == status["counts"]["updates"]
        assert all(not isinstance(v, torch.Tensor) or bool(torch.isfinite(v).all()) for v in state.values())


def test_partial_epoch_metrics_and_complete_episode_order_survive_resume(continued):
    _, _, full, expected, resumed, actual, _, _ = continued
    assert without_time(rows(full / "metrics.jsonl")) == without_time(rows(resumed / "metrics.jsonl"))
    assert without_time(rows(full / "updates.jsonl")[1:]) == without_time(rows(resumed / "updates.jsonl"))
    recorded = rows(full / "episodes.jsonl")
    tagged = [json.loads(line) for line in (resumed / "console.log").read_text().splitlines()
              if line.startswith('{"episodes":')]
    tagged = [row for row in tagged if row.get("record_type") == "publication_episode_batch"]
    assert len(tagged) == 5
    assert all(r["record_type"] == "publication_episode_batch" for r in tagged)
    replayed = [episode for batch in tagged for episode in batch["episodes"]]
    assert without_time(replayed, "rollout_wall_time_seconds") == without_time(
        [r for r in recorded if r["update"] > 1], "rollout_wall_time_seconds")
    assert not (resumed / "episodes.jsonl").exists()
    assert sum(r["steps"] for r in recorded) == expected["recovery"]["counts"]["env_steps"]
    assert len(recorded) == expected["recovery"]["counts"]["episodes"]
    for r in recorded:
        assert r["mean_agent_return"] == pytest.approx(sum(r["reward_per_agent"]) / r["num_agents"])
    for row in rows(full / "checkpoint_records.jsonl"):
        checkpoint = Path(row["path"])
        assert "update%08d" % row["update"] in checkpoint.name
        assert row["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert row["counts"]["updates"] == row["update"]


def test_checkpoint_saving_is_random_neutral_and_wall_pause_continues(continued):
    base, options, full, expected, _, _, _, _ = continued
    if options != dict(task="pcp", binary=True, collectors=1, spec="supplement-v1"):
        return
    _, without_milestone, _ = invoke(base, "without-milestone", **options, milestones=())
    same(expected["policy_net"], without_milestone["policy_net"])
    same(expected["trainer"], without_milestone["trainer"])
    same(expected["recovery"]["rng_states"], without_milestone["recovery"]["rng_states"])
    paused, checkpoint, paused_status = invoke(base, "paused", **options, wall=1e-9)
    assert paused_status["stop_reason"] == "paused_wall_time"
    assert not paused_status["scientific_budget_completed"]
    assert checkpoint["recovery"]["updates_in_epoch"] == 1
    assert rows(paused / "metrics.jsonl") == []
    _, final, _ = invoke(base, "unpaused", **options, resume=Path(paused_status["checkpoint"]))
    same(expected["policy_net"], final["policy_net"])
    same(expected["trainer"], final["trainer"])


def test_seed_streams_and_cached_normal_rng_restore():
    from publication_reconstruction.runtime.hetnet_ext.recovery import capture_rng, restore_rng, seed_stream
    previous = capture_rng()
    try:
        fingerprints = set()
        for seed in range(3):
            for collector in (None, 0, 1, 2, 3):
                seed_stream(seed, collector)
                state = capture_rng()
                digest = hashlib.sha256(state["numpy"][1].tobytes() + state["torch"].numpy().tobytes()).hexdigest()
                assert digest not in fingerprints
                fingerprints.add(digest)
        seed_stream(7, 2)
        np.random.normal()  # deliberately retain MT19937's cached normal value
        state = capture_rng()
        expected = (random.random(), np.random.normal(size=7), torch.rand(7))
        restore_rng(state)
        actual = (random.random(), np.random.normal(size=7), torch.rand(7))
        same(expected, actual)
    finally:
        restore_rng(previous)


def test_atomic_checkpoint_does_not_publish_partial_or_overwrite(tmp_path, monkeypatch):
    from publication_reconstruction.runtime.hetnet_ext.recovery import atomic_checkpoint
    path = tmp_path / "model_update00000001.pt"
    with monkeypatch.context() as patched:
        def interrupted_save(value, stream):
            stream.write(b"incomplete")
            raise OSError("simulated interrupted write")
        patched.setattr(torch, "save", interrupted_save)
        with pytest.raises(OSError, match="interrupted"):
            atomic_checkpoint(path, {})
    assert list(tmp_path.iterdir()) == []
    atomic_checkpoint(path, {"model": torch.ones(2)})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, {"model": torch.zeros(2)})
    assert path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_episode_file_bytes_equal_stdout_array_and_one_open_per_update(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from publication_reconstruction.runtime.hetnet_ext.recording import TrainingRecorder
    model = torch.nn.Linear(1, 1)
    file_recorder = TrainingRecorder(Namespace(metrics_file=str(tmp_path / "file/metrics.jsonl"),
                                               episode_log="file"), model)
    stdout_recorder = TrainingRecorder(Namespace(metrics_file=str(tmp_path / "stdout/metrics.jsonl"),
                                                 episode_log="stdout"), model)
    episodes = [dict(collector=0, collector_episode=i, steps=2, success=False,
                     num_agents=3, reward_per_agent=[-1.0, -2.0, -3.0],
                     team_return=-6.0, mean_agent_return=-2.0,
                     rollout_wall_time_seconds=0.125 * (i + 1)) for i in range(3)]
    fresh = dict(num_steps=6, num_episodes=3, success=0, reward=np.array([-3., -6., -9.]),
                 action_loss=1., value_loss=2.)
    counts = dict(env_steps=6, episodes=3, updates=1, epoch=1)
    opened = []
    original_open = Path.open
    def counted_open(path, *args, **kwargs):
        if path.name == "episodes.jsonl":
            opened.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counted_open)
    file_recorder.record_update(fresh, episodes, counts, 1, 1, 0.1)
    stdout_recorder.record_update(fresh, episodes, counts, 1, 1, 0.1)
    assert opened == [tmp_path / "file/episodes.jsonl"]
    tagged = json.loads(capsys.readouterr().out)
    assert [row["rollout_wall_time_seconds"] for row in tagged["episodes"]] == [0.125, 0.25, 0.375]
    expected_bytes = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                             for row in tagged["episodes"]).encode()
    assert (tmp_path / "file/episodes.jsonl").read_bytes() == expected_bytes


def test_step_cap_closes_partial_epoch_and_records_resource_units(tmp_path):
    output, checkpoint, status = invoke(tmp_path, "budget", extra=("--max_env_steps", "10"))
    assert status["stop_reason"] == "budget_completed"
    assert status["scientific_budget_completed"]
    assert status["counts"] == dict(env_steps=12, episodes=4, updates=2, epoch=1)
    assert checkpoint["recovery"]["completed_epochs"] == 0
    assert checkpoint["recovery"]["updates_in_epoch"] == 2
    metric, = rows(output / "metrics.jsonl")
    assert metric["updates"] == 2 and metric["updates_in_epoch"] == 2
    collector, = status["resources"]["collectors"]
    assert collector["max_rss"] > 0 and collector["max_rss_unit"] in {"bytes", "KiB"}
    assert collector["torch_threads"] == 1
    assert status["training_update_seconds"] > 0 and status["checkpoint_seconds"] > 0


def test_multiple_milestones_in_one_update_keep_same_crossing_after_resume(tmp_path):
    full, expected, _ = invoke(tmp_path, "crossed", milestones=(1, 2))
    first = next((full / "checkpoints").rglob("*_steps1.pt"))
    saved = torch.load(first, map_location="cpu")
    assert saved["recovery"]["counts"]["updates"] == 1
    assert saved["recovery"]["milestones_reached"] == [1, 2]
    resumed, actual, _ = invoke(tmp_path, "resumed", milestones=(1, 2), resume=first)
    same(expected["policy_net"], actual["policy_net"])
    same(expected["trainer"], actual["trainer"])
    assert not list((resumed / "checkpoints").rglob("*_steps2.pt"))
    assert actual["recovery"]["milestones_reached"] == [1, 2]
