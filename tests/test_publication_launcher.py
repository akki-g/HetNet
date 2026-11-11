"""Budget and evidence boundaries for the opt-in historical reconstruction."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from publication_reconstruction import __main__ as launcher
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


# --- Opt-in 6x6 map for the paper-v1 PCP workload ---------------------------------

def plan(tmp_path, *extra, task="pcp", spec="paper-v1"):
    return resolve(parser().parse_args(["train", "--task", task, "--reconstruction-spec", spec, "--seed", "0",
                                        "--output", str(tmp_path / "new"), *extra]))


def command_value(command, flag):
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("task,spec", [("pcp", "paper-v1"), ("pp", "paper-v1"), ("fc", "paper-v1"),
                                       ("pcp", "legacy"), ("pp", "legacy"), ("fc", "legacy")])
def test_default_map_dimension_is_five_and_leaves_the_resolved_protocol_unchanged(tmp_path, task, spec):
    default, explicit = plan(tmp_path, task=task, spec=spec), plan(tmp_path, "--dim", "5", task=task, spec=spec)
    assert default == explicit and "dim" not in default
    assert command_value(default["command"], "--dim") == "5" and default["command"].count("--dim") == 1
    assert json.dumps(default, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    # A namespace that predates the option (as the archive validator builds) resolves to the same 5x5 protocol.
    namespace = parser().parse_args(["train", "--task", task, "--reconstruction-spec", spec, "--seed", "0",
                                     "--output", str(tmp_path / "new")])
    del namespace.dim
    assert resolve(namespace) == default


def test_dimension_six_changes_only_the_command_item_and_adds_the_dim_key(tmp_path):
    five, six = plan(tmp_path), plan(tmp_path, "--dim", "6")
    assert six["dim"] == 6 and "dim" not in five
    assert command_value(six["command"], "--dim") == "6" and six["command"].count("--dim") == 1
    index = six["command"].index("--dim") + 1
    assert six["command"][:index] == five["command"][:index] and six["command"][index + 1:] == five["command"][index + 1:]
    assert {key for key in set(five) | set(six) if five.get(key) != six.get(key)} == {"command", "dim"}
    # The scientific recipe is otherwise the 5x5 one.
    for key in ("model_spec", "env_version", "learner_spec", "optimizer", "collectors", "epochs", "episode_horizon",
               "rng_scheme", "action_dimensions", "binary_bandwidth"):
        assert six[key] == five[key]


@pytest.mark.parametrize("task,spec", [("pp", "paper-v1"), ("fc", "paper-v1"), ("pcp", "legacy"),
                                       ("pp", "legacy"), ("fc", "legacy")])
def test_dimension_six_is_rejected_outside_the_paper_v1_pcp_workload(tmp_path, task, spec):
    with pytest.raises(ValueError, match="Map dimension 6"):
        plan(tmp_path, "--dim", "6", task=task, spec=spec)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("value", ["4", "7", "0", "-5", "5.5", "six"])
def test_only_dimensions_five_and_six_parse(value):
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--task", "pcp", "--seed", "0", "--dim", value])
    # Resolution refuses any other value even when the parser is bypassed.
    namespace = parser().parse_args(["train", "--task", "pcp", "--reconstruction-spec", "paper-v1", "--seed", "0"])
    for bad in (7, 6.0, True, "6"):
        namespace.dim = bad
        with pytest.raises(ValueError, match="dimension must be 5 or 6"):
            resolve(namespace)


def test_dimension_six_dry_run_is_visible_and_side_effect_free(tmp_path):
    result = subprocess.run([sys.executable, "-m", "publication_reconstruction", "train", "--task", "pcp",
                             "--reconstruction-spec", "paper-v1", "--message-backend", "torch-v1", "--seed", "2",
                             "--dim", "6", "--output", str(tmp_path / "run"), "--dry-run"],
                            capture_output=True, text=True, check=True)
    protocol = json.loads(result.stdout)
    assert protocol["dim"] == 6 and command_value(protocol["command"], "--dim") == "6"
    assert not (tmp_path / "run").exists()


def test_resume_carries_the_stored_dimension_in_the_stored_command(tmp_path, monkeypatch):
    # Validation of the archive and checkpoint is stubbed here; this checks only what resolve_resume copies.
    from publication_reconstruction import artifacts
    original = plan(tmp_path / "parent", "--dim", "6")
    monkeypatch.setattr(artifacts, "verify_run", lambda run: (original, {}))
    monkeypatch.setattr(artifacts, "digest", lambda path: "0" * 64)
    monkeypatch.setattr(artifacts, "load_checkpoint", lambda run, path, require_recovery=False: {
        "reconstruction": {"counts": {"env_steps": 12, "episodes": 2, "updates": 1, "epoch": 1}},
        "recovery": {"completed_epochs": 0}})
    arguments = parser().parse_args(["resume", "--run-dir", str(tmp_path / "parent/new"),
                                     "--checkpoint", str(tmp_path / "checkpoint.pt"),
                                     "--output", str(tmp_path / "continued")])
    resumed = launcher.resolve_resume(arguments)
    assert resumed["dim"] == 6 and command_value(resumed["command"], "--dim") == "6"
    assert command_value(resumed["command"], "--resume_checkpoint") == str((tmp_path / "checkpoint.pt").resolve())


def test_a_tiny_six_by_six_paper_v1_pcp_run_trains_on_a_six_by_six_map(tmp_path):
    import torch
    output = tmp_path / "run"
    code = launcher.main(["train", "--task", "pcp", "--reconstruction-spec", "paper-v1",
                          "--message-backend", "torch-v1", "--seed", "0", "--dim", "6", "--collectors", "1",
                          "--epochs", "1", "--updates-per-epoch", "2", "--batch-steps", "6", "--horizon", "6",
                          "--save-every", "1", "--episode-log", "file", "--output", str(output)])
    assert code == 0 and (output / "exit_code.txt").read_text().strip() == "0"
    status = json.loads((output / "run_status.json").read_text())
    assert status["scientific_budget_completed"] is True and status["counts"]["updates"] == 2
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["dim"] == 6 and command_value(protocol["command"], "--dim") == "6"
    resolved = json.loads((output / "resolved_args.json").read_text())
    # The environment's own observation width: 25 vision squares x (dim^2 position cells + 4 status entries).
    # The same tiny run on the default 5x5 map records 725.
    assert resolved["dim"] == 6 and resolved["num_inputs"] == 25 * (6 ** 2 + 4) == 1000
    saved = torch.load(status["checkpoint"], map_location="cpu", weights_only=False)
    assert saved["reconstruction"]["resolved_args"]["dim"] == 6
    # The network's position-status layers are sized from the map: 6*6*25, not 5*5*25.
    assert tuple(saved["policy_net"]["prepro_stat.P.weight"].shape) == (900, 900)
    assert all(torch.isfinite(value).all() for value in saved["policy_net"].values())
    # The checkpoint validator rebuilds the expected command from the protocol, so it must carry the map size.
    from publication_reconstruction.artifacts import load_checkpoint
    loaded = load_checkpoint(output, Path(status["checkpoint"]))
    assert loaded["reconstruction"]["resolved_args"]["dim"] == 6
