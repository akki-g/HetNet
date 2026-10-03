"""Locked workloads, compute-node commands and pre-outcome panel selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from publication_reconstruction import study
from publication_reconstruction.__main__ import parser, resolve


def test_committed_study_matches_resolved_scientific_protocols(tmp_path):
    manifest = json.loads((study.ROOT / "publication_reconstruction/STUDY.json").read_text())
    resolved = study.resolved_study(tmp_path)["protocols"]
    assert len(manifest["runs"]) == len(resolved) == 12
    for entry, protocol in zip(manifest["runs"], resolved):
        for left, right in (("array_index", "array_index"), ("task", "task"), ("communication", "variant"),
                            ("seed", "seed"), ("target_steps", "max_env_steps"), ("milestones", "milestones"),
                            ("epoch_cap", "epochs"), ("horizon", "episode_horizon")):
            assert entry[left] == protocol[right]


@pytest.mark.parametrize("index", range(12))
def test_locked_array(index, tmp_path):
    p = resolve(parser().parse_args(study.train_arguments(index, tmp_path)))
    task, variant, target = study.WORKLOADS[index // 3]
    assert (p["task"], p["variant"], p["seed"]) == (task, variant, index % 3)
    assert p["model_spec"] == "supplement-v1" and p["optimizer"]["name"] == "Adam"
    assert p["collectors"] == 4 and p["batch_step_floor_per_collector"] == 500
    assert p["max_env_steps"] == p["epoch_budget_min_steps_without_step_cap"] == target
    assert p["max_overshoot_steps"] == (3195 if task == "fc" else 2315)
    assert p["milestones"] == ([10_000_000, 20_000_000, 28_000_000] if task == "fc"
                               else [10_000_000, 20_000_000, 30_000_000, 40_000_000])
    assert p["wall_seconds"] == 46 * 3600 and p["episode_log"] == "stdout"
    assert p["command"][p["command"].index("--env_name") + 1] == (
        "fire_commander" if task == "fc" else "predator_capture")


@pytest.mark.parametrize("index", range(4))
def test_preflight_has_100_production_updates(index, tmp_path):
    p = resolve(parser().parse_args(study.train_arguments(index, tmp_path, True)))
    assert p["epochs"] * p["updates_per_epoch"] == 100
    assert p["max_env_steps"] is None and p["milestones"] == []
    assert p["collectors"] == 4 and p["batch_step_floor_per_collector"] == 500
    assert p["episode_horizon"] == (300 if index == 2 else 80)


@pytest.mark.parametrize("index,preflight", [(-1, False), (12, False), (4, True), (True, False)])
def test_reject_bad_array(index, preflight, tmp_path):
    with pytest.raises(ValueError):
        study.train_arguments(index, tmp_path, preflight)


def test_budget_floor_rejected():
    args = parser().parse_args(["train", "--task", "pcp", "--seed", "0", "--epochs", "1",
                               "--max-env-steps", "40000000"])
    with pytest.raises(ValueError, match="Epoch cap"):
        resolve(args)


def test_generated_shell_preserves_arguments(tmp_path):
    unusual = str(tmp_path / "folder with ' quote; and $x")
    job = {"run_dir": unusual, "path": unusual + "/model.pt", "scenarios": unusual + "/panel.json",
           "protocol": unusual + "/protocol.json", "output": unusual + "/result.json", "condition": "sham"}
    path = tmp_path / "submit.sbatch"
    path.write_text(study.slurm_array([job]))
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "#SBATCH --array=0-0%3" in text and "--sham" in text and "--trace" in text
    # Execute mocked shell builtins: no Slurm submission or actual evaluator.
    harness = 'module() { :; }; srun() { printf "%s\\n" "$@"; }; export -f module srun; '
    result = subprocess.run(["bash", "-c", harness + 'source "$1"', "bash", str(path)],
                            env={"PATH": "/usr/bin:/bin", "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_ARRAY_TASK_ID": "0"},
                            text=True, capture_output=True)
    # exec cannot execute a shell function; replace exec solely in this mocked harness.
    path.write_text(path.read_text().replace("exec srun", "srun"))
    result = subprocess.run(["bash", "-c", harness + 'source "$1"', "bash", str(path)],
                            env={"PATH": "/usr/bin:/bin", "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_ARRAY_TASK_ID": "0"},
                            text=True, capture_output=True, check=True)
    assert unusual in result.stdout.splitlines()
    assert job["scenarios"] in result.stdout.splitlines()


def test_preparation_reuses_exact_pcp_bytes_and_locks_panels(tmp_path, monkeypatch):
    panels = tmp_path / "previous"
    panels.mkdir()
    for name, teams, n, seed, failure in (
            ("nominal", [(2, 1), (1, 2), (2, 2), (3, 1), (3, 2)], 500, 2700, 0),
            ("failure", [(2, 1)], 100, 2701, 1)):
        (panels / f"scenarios_{name}.json").write_text(json.dumps(study.panel("pcp", teams, n, seed, failure)))
    def select(run, expected, target):
        return {"run_dir": str(run), "path": str(run / "checkpoint.pt"), "task": expected["task"],
                "variant": expected["variant"], "training_seed": expected["seed"], "selection_target": target}
    monkeypatch.setattr(study, "select_checkpoint", select)
    args = argparse.Namespace(run_root=tmp_path / "runs", pcp_panel_dir=panels, run_map=None, output=tmp_path / "prepared")
    result = study.prepare_evaluation(args)
    assert result["policies"] == 12 and result["jobs"] == 24 and not result["submitted"]
    for name in ("nominal", "failure"):
        assert (args.output / f"pcp_{name}/scenarios.json").read_bytes() == (panels / f"scenarios_{name}.json").read_bytes()
    for task, seed in (("pp", 2702), ("fc", 2703)):
        protocol = json.loads((args.output / f"{task}_nominal/protocol.json").read_text())
        assert protocol["distribution"]["scenario_seed"] == seed
        assert len(json.loads((args.output / f"{task}_nominal/scenarios.json").read_text())) == 500
    subprocess.run(["bash", "-n", str(args.output / "submit.sbatch")], check=True)
    with pytest.raises(FileExistsError):
        study.prepare_evaluation(args)


def test_select_first_saved_not_best_performance(tmp_path, monkeypatch):
    from publication_reconstruction import artifacts
    expected = resolve(parser().parse_args(study.train_arguments(3, tmp_path)))
    rows = [{"path": "early", "counts": {"updates": 4, "env_steps": 30_000_004}, "run_dir": "r", "checkpoint_sha256": "a"},
            {"path": "later", "counts": {"updates": 5, "env_steps": 30_002_030}, "run_dir": "r", "checkpoint_sha256": "b"}]
    monkeypatch.setattr(artifacts, "verify_run", lambda run: (expected, {}))
    monkeypatch.setattr(study, "checkpoint_candidates", lambda run: rows)
    monkeypatch.setattr(artifacts, "digest", lambda path: "a")
    monkeypatch.setattr(artifacts, "load_checkpoint", lambda run, path: {"reconstruction": {"counts": rows[0]["counts"]}})
    selected = study.select_checkpoint(tmp_path, expected, 30_000_000)
    assert selected["path"] == "early"


def test_preparation_rejects_dangling_output_symlink(tmp_path):
    output = tmp_path / "output"
    output.symlink_to(tmp_path / "missing")
    args = argparse.Namespace(output=output)
    with pytest.raises(FileExistsError):
        study.prepare_evaluation(args)
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("updates", [99, 100])
def test_preflight_summary_requires_full_panel_and_separates_time_costs(tmp_path, monkeypatch, updates):
    from publication_reconstruction import artifacts, evaluation
    status = {"checkpoint": str(tmp_path / "checkpoint.pt"), "counts": {"updates": updates},
              "resources": {"slurm_cpus_per_task": "4"}, "startup_to_training_seconds": 5,
              "segment_wall_time_seconds": 300}
    (tmp_path / "run_status.json").write_text(json.dumps(status))
    (tmp_path / "updates.jsonl").write_text((json.dumps({"steps": 2000, "wall_time_seconds": 2}) + "\n") * updates)
    (tmp_path / "checkpoint_records.jsonl").write_text(json.dumps({"wall_time_seconds": .2}) + "\n")
    (tmp_path / "protocol.json").write_text(json.dumps({"task": "pp", "variant": "real"}))
    (tmp_path / "environment.json").write_text(json.dumps({"packages": {"torch": "2.2.1"}}))
    monkeypatch.setattr(artifacts, "load_checkpoint", lambda *args: {})
    validated = []
    monkeypatch.setattr(artifacts, "_validate_optimizer", lambda *args: validated.append(args))
    probes = []
    def probe(args):
        probes.append(args)
        assert len(json.loads(args.scenarios.read_text())) == 1
        assert args.checkpoint == Path(status["checkpoint"])
        return {"episodes": 1, "parameters_unchanged": True}
    monkeypatch.setattr(evaluation, "evaluate", probe)
    if updates == 99:
        with pytest.raises(ValueError, match="100 updates"):
            study.preflight_summary(tmp_path)
        assert not probes and not (tmp_path / "preflight.json").exists()
    else:
        report = study.preflight_summary(tmp_path)
        assert validated and len(probes) == 1
        assert report["measured_update_steps_per_second"] == 1000
        assert report["projected_update_work_hours_excluding_startup_logging_and_checkpoints"] == pytest.approx(40_000 / 3600)
        assert report["projected_segment_hours_including_logging_and_checkpoints_excluding_startup"] == pytest.approx(60_000 / 3600)
        assert report["resources"]["slurm_cpus_per_task"] == "4"
