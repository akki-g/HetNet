"""Budget and evidence boundaries for the opt-in historical reconstruction."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from publication_reconstruction.__main__ import archive_source, parser, resolve


@pytest.mark.parametrize("task,recipe,collectors,epochs,horizon,floor", [
    ("pp", "june-2022", 1, 2000, 80, 10_000_000),
    ("pcp", "june-2022", 1, 2000, 80, 10_000_000),
    ("fc", "june-2022", 4, 1400, 300, 28_000_000),
    ("pp", "october-2022", 4, 2000, 80, 40_000_000),
    ("pcp", "october-2022", 4, 2000, 80, 40_000_000),
    ("fc", "october-2022", 4, 1400, 300, 28_000_000),
])
def test_dated_recipes_record_samples_and_updates(task, recipe, collectors, epochs, horizon, floor, tmp_path):
    args = parser().parse_args(["train", "--task", task, "--seed", "0", "--recipe", recipe,
                                "--output", str(tmp_path / "new")])
    plan = resolve(args)
    assert (plan["collectors"], plan["epochs"], plan["episode_horizon"]) == (collectors, epochs, horizon)
    assert plan["epoch_budget_min_steps_without_step_cap"] == floor
    assert plan["max_env_steps"] is None
    assert plan["updates_per_epoch"] * plan["epochs"] == (14000 if task == "fc" else 20000)
    command = plan["command"]
    value = lambda flag: command[command.index(flag) + 1]
    assert value("--nagents") == "3"  # repairs the June PP recipe's omitted physical count
    assert Path(value("--metrics_file")).parent == tmp_path / "new"
    assert not (tmp_path / "new").exists()


def test_dry_run_is_side_effect_free_and_versions_are_distinct(tmp_path):
    for version in ["historical-2022", "corrected-v1"]:
        result = subprocess.run([sys.executable, "-m", "publication_reconstruction", "train",
            "--task", "pcp", "--seed", "2", "--env-version", version, "--max-env-steps", "10000000",
            "--output", str(tmp_path / version), "--dry-run"], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        assert plan["env_version"] == version and plan["max_env_steps"] == 10_000_000
        assert plan["max_overshoot_steps"] == 578
    assert list(tmp_path.iterdir()) == []


def test_archive_hashes_match_executed_copy(tmp_path):
    archive_source(tmp_path)
    manifest = json.loads((tmp_path / "source_manifest.json").read_text())
    assert "runtime/main.py" in manifest["files"]
    assert "runtime/envs/ic3net_envs/fire_commander_env.py" in manifest["files"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((tmp_path / "source" / name).read_bytes()).hexdigest() == digest
    assert (tmp_path / "uv.lock").is_file()


def test_existing_evidence_is_never_overwritten(tmp_path):
    evidence = tmp_path / "stdout.log"
    evidence.write_text("previous evidence")
    result = subprocess.run([sys.executable, "-m", "publication_reconstruction", "train", "--task", "pp",
        "--seed", "0", "--output", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert evidence.read_text() == "previous evidence"
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize("flag,value", [("--epochs", "0"), ("--collectors", "0"),
    ("--batch-steps", "-1"), ("--max-env-steps", "0"), ("--horizon", "0"), ("--seed", "-1")])
def test_invalid_budgets_rejected_before_run_creation(flag, value):
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--task", "pp", "--seed", "0", flag, value])


@pytest.mark.parametrize("task", ["pcp", "fc"])
@pytest.mark.parametrize("model_spec", ["public-code-v1", "supplement-v1"])
@pytest.mark.parametrize("collectors", [1, 4])
def test_singleton_a_advantage_rejected_before_run_creation(tmp_path, task, model_spec, collectors):
    output = tmp_path / "invalid"
    args = parser().parse_args(["train", "--task", task, "--seed", "0",
        "--model-spec", model_spec, "--collectors", str(collectors),
        "--horizon", "1", "--batch-steps", "1", "--output", str(output)])
    with pytest.raises(ValueError, match="per-class advantage normalization"):
        resolve(args)
    assert not output.exists()


@pytest.mark.parametrize("task,horizon,batch_steps", [
    ("pp", 1, 1), ("pcp", 2, 1), ("pcp", 1, 2), ("fc", 2, 1), ("fc", 1, 2)])
def test_small_configs_with_multiple_class_advantages_remain_available(task, horizon, batch_steps):
    args = parser().parse_args(["train", "--task", task, "--seed", "0",
        "--horizon", str(horizon), "--batch-steps", str(batch_steps)])
    assert resolve(args)["episode_horizon"] == horizon
