"""Local orchestration checks with tiny subprocesses; never start training."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap

import pytest


@pytest.fixture
def launcher(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/publication_preflight_local.py"
    spec = importlib.util.spec_from_file_location("publication_preflight_local", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "validate_source_origins", lambda: None)
    return module


@pytest.fixture
def fake_job(tmp_path):
    script = tmp_path / "fake_preflight.py"
    script.write_text(textwrap.dedent('''\
        import json
        import os
        from pathlib import Path
        import signal
        import subprocess
        import sys
        import time

        root, name, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
        grandchild = None
        def terminate(signum, frame):
            (root / (name + ".terminated")).write_text("terminated")
            if grandchild is not None:
                grandchild.wait(timeout=2)
            raise SystemExit(15)
        signal.signal(signal.SIGTERM, terminate)
        started = time.monotonic()
        (root / (name + ".ready")).write_text(json.dumps({
            "pid": os.getpid(), "started": started,
            "environment": {key: os.environ.get(key) for key in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "PYTHONUNBUFFERED", "DGLBACKEND")}}))
        if mode == "wait":
            grandchild = subprocess.Popen([sys.executable, "-u", __file__, str(root),
                                           name + "_probe", "grandchild"])
        if mode in ("wait", "grandchild"):
            while True:
                time.sleep(.01)
        if mode == "fail":
            print("deliberate child failure", flush=True)
            raise SystemExit(7)
        # All four jobs must be alive together before any can complete.
        deadline = started + 8
        while len(list(root.glob("*.ready"))) != 4:
            if time.monotonic() > deadline:
                raise RuntimeError("four jobs did not overlap")
            time.sleep(.01)
        print("ordinary child output", flush=True)
        print(json.dumps({"record_type": "unrelated"}), flush=True)
        print(json.dumps({"episodes": [
            {"episode": 0, "collector": 0, "steps": 4, "success": True, "mean_agent_return": -2.,
             "rollout_wall_time_seconds": .125},
            {"episode": 1, "collector": 1, "steps": 8, "success": False, "mean_agent_return": 1.,
             "rollout_wall_time_seconds": .375}],
            "record_type": "publication_episode_batch", "update": 1},
            sort_keys=True), flush=True)
        report = root / name / "seed991/preflight.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"counts": {"updates": 100, "episodes": 2,
            "steps": 12}, "measured_update_steps_per_second": 24.}))
        (root / (name + ".finished")).write_text(str(time.monotonic()))
    '''))

    def job(root, name, mode="success"):
        return {"workload": name,
                "command": [sys.executable, "-u", str(script), str(root), name, mode]}
    return job


def test_commands_preserve_four_locked_preflight_workloads(launcher, tmp_path):
    jobs = launcher.commands(tmp_path)
    assert [(job["index"], job["workload"]) for job in jobs] == [
        (0, "pp_real"), (1, "pcp_real"), (2, "fc_real"), (3, "pcp_binary")]
    for index, job in enumerate(jobs):
        assert job["command"] == [sys.executable, "-u", "-m", "publication_reconstruction",
                                  "run-index", "--preflight", "--index", str(index),
                                  "--run-root", str(tmp_path)]


def test_four_jobs_overlap_and_save_episode_metrics(launcher, fake_job, tmp_path, monkeypatch, capsys):
    root = tmp_path / "runs"
    names = ["pp_real", "pcp_real", "fc_real", "pcp_binary"]
    monkeypatch.setattr(launcher, "commands", lambda path: [fake_job(path, name) for name in names])
    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    assert launcher.main(["--run-root", str(root)]) == 0

    starts = [json.loads((root / f"{name}.ready").read_text()) for name in names]
    finishes = [float((root / f"{name}.finished").read_text()) for name in names]
    assert max(row["started"] for row in starts) < min(finishes)
    expected_environment = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                            "PYTHONUNBUFFERED": "1", "DGLBACKEND": "pytorch"}
    assert all(row["environment"] == expected_environment for row in starts)
    manifest = json.loads((root / "local_launch.json").read_text())
    assert manifest["concurrency"] == 4
    assert manifest["environment_overrides"] == expected_environment
    assert (root / "local_launcher.py").read_bytes() == Path(launcher.__file__).read_bytes()
    summary = json.loads((root / "local_summary.json").read_text())
    for name in names:
        row = summary["jobs"][name]
        assert row["state"] == "completed" and row["exit_code"] == 0
        assert row["counts"] == {"updates": 100, "episodes": 2, "steps": 12}
        assert row["steps_per_second"] == 24.
        assert row["elapsed_seconds"] > 0
        assert "ordinary child output" in (root / "logs" / f"{name}.log").read_text()
        episodes = [json.loads(line) for line in
                    (root / "logs" / f"{name}.episodes.jsonl").read_text().splitlines()]
        assert [episode["rollout_wall_time_seconds"] for episode in episodes] == [.125, .375]
        progress = json.loads((root / "logs" / f"{name}.progress.jsonl").read_text())
        assert progress["update"] == 1 and progress["episodes"] == 2 and progress["steps"] == 12
        assert progress["success_rate"] == .5 and progress["mean_agent_return"] == -.5
        assert progress["mean_episode_steps"] == 6.
        assert progress["mean_rollout_wall_time_seconds"] == .25
        assert progress["min_rollout_wall_time_seconds"] == .125
        assert progress["max_rollout_wall_time_seconds"] == .375
    printed = capsys.readouterr().out
    assert "rollout/episode=0.250s" in printed
    for name in names:
        assert (f"[{name}] episode=0 collector=0 steps=4 success=1 "
                "return/agent=-2.0000 rollout=0.125s") in printed
        assert (f"[{name}] episode=1 collector=1 steps=8 success=0 "
                "return/agent=1.0000 rollout=0.375s") in printed


def test_child_failure_propagates_to_exit_and_summary(launcher, fake_job, tmp_path, monkeypatch):
    root = tmp_path / "failed"
    monkeypatch.setattr(launcher, "commands", lambda path: [fake_job(path, "failed_job", "fail")])
    assert launcher.main(["--run-root", str(root)]) == 1
    row = json.loads((root / "local_summary.json").read_text())["jobs"]["failed_job"]
    assert row["state"] == "failed" and row["exit_code"] == 7
    assert "deliberate child failure" in (root / "logs/failed_job.log").read_text()


def test_dry_run_prints_commands_without_output_or_children(launcher, tmp_path, monkeypatch, capsys):
    root = tmp_path / "dry_run"
    monkeypatch.setattr(launcher, "run_all", lambda *args: pytest.fail("dry run started children"))
    assert launcher.main(["--run-root", str(root), "--dry-run"]) == 0
    assert not root.exists()
    stdout = capsys.readouterr().out
    assert stdout.count("run-index --preflight --index") == 4
    assert "Concurrency: 4 jobs, 4 collectors each" in stdout


def test_existing_output_is_refused_without_mutation(launcher, tmp_path, monkeypatch):
    marker = tmp_path / "existing.txt"
    marker.write_text("preserve this")
    monkeypatch.setattr(launcher, "run_all", lambda *args: pytest.fail("existing output started children"))
    with pytest.raises(FileExistsError):
        launcher.main(["--run-root", str(tmp_path)])
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == "preserve this"


def test_cancel_terminates_and_reaps_child_and_probe(launcher, fake_job, tmp_path):
    root = tmp_path / "interrupted"
    (root / "logs").mkdir(parents=True)

    async def exercise():
        task = asyncio.create_task(launcher.run_all(root, [fake_job(root, "waiting", "wait")], 1))
        try:
            for _ in range(500):
                if (root / "waiting_probe.ready").exists():
                    break
                await asyncio.sleep(.01)
            else:
                pytest.fail("mock child did not start within five seconds")
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(exercise())
    for name in ("waiting", "waiting_probe"):
        child = json.loads((root / f"{name}.ready").read_text())
        assert (root / f"{name}.terminated").read_text() == "terminated"
        with pytest.raises(ProcessLookupError):
            os.kill(child["pid"], 0)
    row = json.loads((root / "local_summary.json").read_text())["jobs"]["waiting"]
    assert row["state"] == "interrupted" and row["exit_code"] == 15
