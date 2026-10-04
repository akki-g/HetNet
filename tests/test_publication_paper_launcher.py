"""Paper preset integrity and matched benchmark/study launch contracts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from publication_reconstruction.__main__ import parser, resolve
from publication_reconstruction.benchmark import benchmark_plan, compare_runs
from publication_reconstruction.paper_study import study_plan


@pytest.mark.parametrize("task,variant,p,a", [("pp", "real", 5, 6), ("pcp", "binary", 5, 6), ("fc", "real", 4, 5)])
@pytest.mark.parametrize("backend", ["dgl", "torch-v1"])
def test_paper_preset_is_complete_and_declares_bandwidth(task, variant, p, a, backend):
    args = parser().parse_args(["train", "--reconstruction-spec", "paper-v1", "--task", task,
                               "--variant", variant, "--message-backend", backend, "--seed", "0"])
    value = resolve(args)
    assert value["model_spec"] == value["env_version"] == "paper-v1"
    assert value["learner_spec"] == "paper-equations-v1"
    assert value["message_backend"] == backend
    assert value["optimizer"]["name"] == "Adam" and value["optimizer"]["lr"] == .001
    assert value["action_dimensions"] == {"P": p, "A": a}
    assert value["binary_bandwidth"]["bits_per_sender_per_round"] == 256
    assert value["collectors"] == 4 and value["rng_scheme"] == "seedsequence-v1"
    assert not value["profile_phases"] and "--profile_phases" not in value["command"]
    if variant == "binary":
        assert value["command"][value["command"].index("--msg_dim") + 1] == "64"


@pytest.mark.parametrize("extra", [
    ["--message-backend", "torch-v1"], ["--model-spec", "paper-v1"], ["--env-version", "paper-v1"],
    ["--reconstruction-spec", "paper-v1", "--model-spec", "supplement-v1"],
    ["--reconstruction-spec", "paper-v1", "--env-version", "corrected-v1"]])
def test_incompatible_component_mixtures_rejected(extra):
    with pytest.raises(ValueError):
        resolve(parser().parse_args(["train", "--task", "pcp", "--seed", "0", *extra]))


def test_paper_single_step_no_longer_uses_undefined_sample_std():
    value = resolve(parser().parse_args(["train", "--reconstruction-spec", "paper-v1", "--task", "fc",
                                        "--seed", "0", "--horizon", "1", "--batch-steps", "1"]))
    assert value["episode_horizon"] == 1


def test_benchmark_is_sequential_paired_production_work_and_profiles_separately(tmp_path):
    plan = benchmark_plan(tmp_path / "fresh", True)
    assert plan["throughput_runs"] == 24 and plan["throughput_updates"] == 480
    assert plan["diagnostic_runs"] == 8 and len(plan["jobs"]) == 32
    for job in plan["jobs"]:
        p = resolve(parser().parse_args(job["command"][4:]))
        assert p["collectors"] == 4 and p["batch_step_floor_per_collector"] == 500
        assert p["episode_horizon"] == (300 if job["task"] == "fc" else 80)
        assert p["seed"] == 991 and p["epochs"] * p["updates_per_epoch"] == job["updates"]
        assert p["profile_phases"] == job["profiled"]
        assert p["max_env_steps"] is None
    assert [job["backend"] for job in plan["jobs"][:2]] == ["dgl", "torch-v1"]
    assert [job["backend"] for job in plan["jobs"][8:10]] == ["torch-v1", "dgl"]
    assert [job["backend"] for job in plan["jobs"][16:18]] == ["dgl", "torch-v1"]
    assert not (tmp_path / "fresh").exists()


def fake_rows(tmp_path):
    rows = []
    for job in benchmark_plan(tmp_path)["jobs"]:
        fast = job["backend"] == "torch-v1"
        rows.append({**job, "initial_identity": {"model": "m", "optimizer": "o", "rng": "r"},
                     "source_manifest_sha256": "source", "collector_intervals": [{"affinity": [0, 1, 2, 3]}] * 4,
                     "post_first_update_steps_per_second": 120 if fast else 100,
                     "segment_steps_per_second": 110 if fast else 90,
                     "episode_ledger_identity": "different" if fast else "baseline",
                     "trajectory_identity_complete": False})
    return rows


def test_comparison_reports_all_pairs_and_requires_replay_for_changed_work(tmp_path):
    result = compare_runs(fake_rows(tmp_path))
    assert len(result) == 4
    assert all(r["throughput_gate"] and r["fixed_rollout_compute_replay_required"] for r in result)
    assert all(p["update_throughput_ratio"] == 1.2 for r in result for p in r["pairs"])


@pytest.mark.parametrize("field", ["initial_identity", "source_manifest_sha256", "collector_intervals"])
def test_comparison_rejects_unmatched_pairs(tmp_path, field):
    rows = fake_rows(tmp_path)
    if field == "collector_intervals":
        rows[1][field] = [{"affinity": [4, 5, 6, 7]}] * 4
    else:
        rows[1][field] = "mismatch"
    with pytest.raises(ValueError):
        compare_runs(rows)


def test_matched_fc_study_preserves_softrole_method_and_uses_one_environment(tmp_path):
    plan = study_plan(tmp_path, "torch-v1")
    assert len(plan["hetnet"]) == 12 and len(plan["softrole_fc"]) == 6
    assert len(plan["fc_scenarios"]) == 500 and not plan["submitted"]
    for row in plan["softrole_fc"]:
        cfg = row["config"]
        assert cfg["task"] == "fc" and cfg["env_version"] == "paper-v1"
        assert cfg["total_steps"] == 28_000_000 and cfg["actor_coeff"] == 50
        assert cfg["lr"] == 1e-4
    for row in plan["hetnet"]:
        assert row["message_backend"] == "torch-v1" and row["model_spec"] == "paper-v1"


def test_cli_benchmark_dry_run_creates_no_files(tmp_path):
    output = tmp_path / "fresh"
    result = subprocess.run([sys.executable, "-m", "publication_reconstruction", "benchmark",
                             "--output", str(output), "--dry-run"], check=True, text=True, capture_output=True)
    assert json.loads(result.stdout)["throughput_runs"] == 24
    assert not output.exists()


def test_new_slurm_scripts_have_fixed_resources_and_no_nested_submission():
    root = Path(__file__).resolve().parents[1]
    for name in ("publication_backend_benchmark", "publication_paper_preflight", "publication_paper_train", "softrole_paper_fc"):
        path = root / "slurm" / (name + ".sbatch")
        subprocess.run(["bash", "-n", str(path)], check=True)
        value = path.read_text()
        assert "#SBATCH --cpus-per-task=4" in value and "#SBATCH --ntasks=1" in value
        assert "exec srun --cpu-bind=cores" in value
        assert "sbatch " not in value
    value = (root / "slurm/publication_backend_benchmark.sbatch").read_text()
    assert "#SBATCH --array" not in value and "#SBATCH --time=24:00:00" in value
