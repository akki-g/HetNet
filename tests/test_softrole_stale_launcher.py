"""Locked gate configuration, provenance and Slurm wiring without training."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from softrole import stale_capability as gate
from softrole.programme import CAPCON_PROFILE, programme_config
from test_softrole_graded_pilot_launcher import (
    assert_environment, calls, run_slurm, setup_environment as pilot_environment,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "slurm/softrole_stale_prepare.sbatch"
SLURM = ROOT / "slurm/softrole_stale_train.sbatch"


def setup_environment(tmp_path, **kwargs):
    root, manifest, run_root, env = pilot_environment(tmp_path, **kwargs)
    python = Path(env.get("HETNET_PYTHON", root / ".venv/bin/python"))
    python.write_text(python.read_text().replace("softrole.graded_pilot", "softrole.stale_capability")
                      .replace("if operation == 'prepare':", "if operation == 'prepare' and '--dry-run' not in args:")
                      .replace("else:\n    assert pathlib", "elif operation != 'prepare':\n    assert pathlib"))
    for old, new in (("SOFTROLE_GRADED_PILOT_MANIFEST", "SOFTROLE_STALE_MANIFEST"),
                     ("SOFTROLE_GRADED_PILOT_RUN_ROOT", "SOFTROLE_STALE_RUN_ROOT")):
        env.pop(new, None)
        if old in env:
            env[new] = env.pop(old)
    if not kwargs.get("defaults", False):
        container = Path(env["HETNET_RUN_DIR"])
        contained_manifest = container / "locked manifest.json"
        if manifest.exists():
            manifest.rename(contained_manifest)
        manifest, run_root = contained_manifest, container / "training runs"
        env.update(SOFTROLE_STALE_MANIFEST=str(manifest), SOFTROLE_STALE_RUN_ROOT=str(run_root))
    return root, manifest, run_root, env


def launch_args(manifest, index, run_root, dry_run=True):
    result = ["-u", "-m", "softrole.stale_capability", "launch", "--manifest", str(manifest),
              "--index", str(index), "--run-root", str(run_root)]
    return result + (["--dry-run"] if dry_run else [])


def forbid_training(*args, **kwargs):
    pytest.fail("preparation and dry-runs must not train")


@pytest.fixture
def manifest_path(tmp_path, monkeypatch):
    source = gate.source_snapshot()
    monkeypatch.setattr(gate, "source_snapshot", lambda: deepcopy(source))
    path = tmp_path / "manifest.json"
    gate.prepare(path)
    return path


@pytest.mark.parametrize("coefficient", gate.GATE_COEFFICIENTS)
@pytest.mark.parametrize("seed", gate.GATE_SEEDS)
def test_gate_changes_only_declared_fields_and_preserves_execution_contract(coefficient, seed):
    primary = programme_config("l1b_decomposed_beta1", seed, 1e-3, profile=CAPCON_PROFILE)
    config = gate.gate_config(coefficient, seed)
    expected = dict(asdict(primary), capabilities="graded", descriptor="capability",
                    compositions=((1, 1), (2, 1)), sensing_p=(1, 2, 3), sensing_a=(0,),
                    held_out_sensing_p=(), held_out_sensing_a=(), failure_prob=0.0,
                    env_version="graded-observation-v2", actor_capabilities="explicit",
                    entropy_coeff=coefficient, total_steps=2_000_000)
    assert asdict(config) == expected
    assert config.model_kwargs() == primary.model_kwargs()
    assert (config.nprocesses, config.batch_steps, config.max_steps) == (4, 500, 80)
    assert (config.optimizer, config.lr, config.critic_mode, config.credit_beta) == (
        "adam", 0.001, "decomposed", 1.0)
    assert (gate.PROTOCOL["device"], gate.PROTOCOL["dtype"], gate.PROTOCOL["threads_per_collector"]) == (
        "cpu", "float64", 1)


def test_changed_field_guard_refuses_unrelated_settings(monkeypatch):
    def unauthorized(primary, **settings):
        return replace(primary, **settings, lr=0.002)
    monkeypatch.setattr(gate, "replace", unauthorized)
    with pytest.raises(ValueError, match="beyond declared"):
        gate.gate_config(0.003, 0)


@pytest.mark.parametrize("coefficient,seed", [(0, 0), (0.1, 0), (True, 0), ("0.003", 0),
                                               (0.003, True), (0.003, 0.0), (0.003, 3)])
def test_config_rejects_unconfirmed_parameters(coefficient, seed):
    with pytest.raises(ValueError):
        gate.gate_config(coefficient, seed)


def test_preparation_is_fresh_immutable_and_dry_run_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "train", forbid_training)
    output = tmp_path / "new" / "manifest.json"
    preview = gate.prepare(output, stage="gate", dry_run=True)
    assert not output.parent.exists()
    manifest = gate.prepare(output)
    assert manifest["entries"] == preview["entries"]
    assert json.loads(output.read_text()) == manifest
    assert gate.validate_manifest(manifest) is manifest
    entries = manifest["entries"]
    assert [(row["coefficient"], row["seed"]) for row in entries] == [
        (coefficient, seed) for coefficient in gate.GATE_COEFFICIENTS for seed in gate.GATE_SEEDS]
    assert len({row["relative_output"] for row in entries}) == 6
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        gate.prepare(output)
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="Only"):
        gate.prepare(tmp_path / "main.json", stage="main")


@pytest.mark.parametrize("change", ["protocol", "config", "order", "output", "stage", "numeric_type",
                                     "source_hash", "source_path", "version", "extra", "hash"])
def test_manifest_refuses_tampering_even_with_recomputed_outer_hash(manifest_path, change):
    manifest = json.loads(manifest_path.read_text())
    if change == "protocol":
        manifest["protocol"]["collapse_rule"]["success_below"] = 0.1
    elif change == "config":
        manifest["entries"][0]["config"]["lr"] = 0.002
    elif change == "order":
        manifest["entries"].reverse()
    elif change == "output":
        manifest["entries"][0]["relative_output"] = "../escape"
    elif change == "stage":
        manifest["stage"] = "main"
    elif change == "numeric_type":
        manifest["entries"][0]["seed"] = False
    elif change == "source_hash":
        manifest["source"]["sha256"] = "0" * 64
    elif change == "source_path":
        manifest["source"]["files"]["../escape.py"] = "0" * 64
        manifest["source"]["sha256"] = gate.digest(manifest["source"]["files"])
    elif change == "version":
        manifest["version"] = True
    elif change == "extra":
        manifest["override"] = True
    manifest["manifest_sha256"] = gate.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if change == "hash":
        manifest["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        gate.validate_manifest(manifest)


@pytest.mark.parametrize("index", range(6))
def test_launch_dry_run_does_not_train_or_create_output(manifest_path, tmp_path, monkeypatch, index):
    monkeypatch.setattr(gate, "train", forbid_training)
    record = gate.launch(manifest_path, index, tmp_path / "training", dry_run=True)
    entry = gate.expected_entries()[index]
    assert record["config"] == entry["config"]
    assert record["coefficient"] == entry["coefficient"]
    assert record["dry_run"] is True
    assert not Path(record["output"]).exists() and not (tmp_path / "training").exists()


def test_launch_source_changes_and_existing_output_are_refused(manifest_path, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "train", forbid_training)
    entry = gate.expected_entries()[0]
    root = tmp_path / "training"
    output = root / entry["relative_output"]
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        gate.launch(manifest_path, 0, root, dry_run=True)
    source = gate.source_snapshot()
    source["sha256"] = "0" * 64
    monkeypatch.setattr(gate, "source_snapshot", lambda: source)
    with pytest.raises(ValueError, match="Source changed"):
        gate.launch(manifest_path, 1, root, dry_run=True)


@pytest.mark.parametrize("source_changes", [False, True])
def test_training_receipt_requires_post_training_source_lock(manifest_path, tmp_path, monkeypatch, source_changes):
    observed = []
    def fake_train(config, output, **settings):
        observed.append((config, output, settings))
        output.mkdir(parents=True)
        if source_changes:
            source = gate.source_snapshot()
            source["sha256"] = "0" * 64
            monkeypatch.setattr(gate, "source_snapshot", lambda: source)
        return output / "final.pt"
    monkeypatch.setattr(gate, "train", fake_train)
    if source_changes:
        with pytest.raises(RuntimeError, match="Source changed"):
            gate.launch(manifest_path, 5, tmp_path / "training")
    else:
        record = gate.launch(manifest_path, 5, tmp_path / "training")
        assert record["checkpoint"].endswith("final.pt")
    config, output, settings = observed[0]
    assert config == gate.gate_config(0.01, 2)
    assert settings == dict(episode_log="chunked-gzip", checkpoint_commits=True)
    assert (output / "stale_capability.json").exists() is not source_changes


@pytest.mark.parametrize("args", [
    ["prepare", "--stage", "main", "--output", "unused.json"],
    ["prepare", "--stage", "gate", "--output", "unused.json", "--entropy-coeff", "0.1"],
    ["launch", "--manifest", "missing.json", "--index", "0", "--run-root", "unused", "--seed", "8"],
    ["launch", "--manifest", "missing.json", "--index", "6", "--run-root", "unused"],
    ["launch", "--manifest", "missing.json", "--index", "0", "--run-root", "unused", "--dry"],
])
def test_cli_refuses_unimplemented_stages_overrides_and_abbreviations(args, monkeypatch):
    monkeypatch.setattr(gate, "train", forbid_training)
    with pytest.raises(SystemExit) as error:
        gate.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("defaults", [False, True])
def test_preparation_shell_creates_then_previews_all_six_runs(tmp_path, defaults):
    root, manifest, run_root, env = setup_environment(tmp_path, defaults=defaults)
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)
    assert [row["args"] for row in observed] == [
        ["-u", "-m", "softrole.stale_capability", "prepare", "--stage", "gate", "--output", str(manifest)],
        *[launch_args(manifest, i, run_root) for i in range(6)]]
    for row in observed:
        assert_environment(row, root)
    assert manifest.exists() and not run_root.exists()


def test_preparation_shell_dry_run_does_not_create_manifest(tmp_path):
    _, manifest, _, env = setup_environment(tmp_path)
    result = run_slurm(PREPARE, ["--dry-run"], tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert len(calls(env)) == 1 and calls(env)[0]["args"][-1] == "--dry-run"
    assert not manifest.exists()


def test_existing_manifest_is_never_rewritten_and_failed_preview_stops(tmp_path):
    _, manifest, run_root, env = setup_environment(tmp_path, existing=True)
    before = manifest.read_bytes()
    env["MOCK_FAIL"] = "launch:2"
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 17
    assert [row["args"] for row in calls(env)] == [launch_args(manifest, i, run_root) for i in range(3)]
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("dry_run", [False, True])
def test_training_shell_passes_exact_manifest_entry_and_execution_contract(tmp_path, index, dry_run):
    root, manifest, run_root, env = setup_environment(tmp_path, index=str(index), existing=True, defaults=True)
    result = run_slurm(SLURM, ["--dry-run"] if dry_run else [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)
    assert len(observed) == 1 and observed[0]["args"] == launch_args(manifest, index, run_root, dry_run)
    assert_environment(observed[0], root)
    assert not run_root.exists()


@pytest.mark.parametrize("index", ["-1", "6", "00", "5.0", "x"])
def test_training_shell_refuses_bad_index_before_execution(tmp_path, index):
    _, _, _, env = setup_environment(tmp_path, index=index, existing=True)
    result = run_slurm(SLURM, [], tmp_path, env)
    assert result.returncode == 2 and not calls(env)


@pytest.mark.parametrize("script", [SLURM, PREPARE])
@pytest.mark.parametrize("args", [["--stage", "main"], ["--entropy-coeff", "1"], ["--dry"],
                                   ["--run-root", "elsewhere"], ["--total-steps", "1"]])
def test_shell_refuses_protocol_overrides(script, args, tmp_path):
    _, _, _, env = setup_environment(tmp_path, existing=True)
    result = run_slurm(script, args, tmp_path, env)
    assert result.returncode == 2 and not calls(env)


@pytest.mark.parametrize("script", [SLURM, PREPARE])
@pytest.mark.parametrize("variable", ["SOFTROLE_STALE_MANIFEST", "SOFTROLE_STALE_RUN_ROOT"])
@pytest.mark.parametrize("symlink", [False, True])
def test_shell_refuses_outputs_outside_run_container_before_execution(script, variable, symlink, tmp_path):
    _, manifest, _, env = setup_environment(tmp_path, existing=True)
    outside = tmp_path / "outside run"
    outside.mkdir()
    (outside / "manifest.json").write_text('{"outside": "preserve"}\n')
    route = outside
    if symlink:
        route = Path(env["HETNET_RUN_DIR"]) / "escaped output"
        route.symlink_to(outside, target_is_directory=True)
    env[variable] = str(route / ("manifest.json" if variable.endswith("MANIFEST") else "training"))
    before = manifest.read_bytes()
    result = run_slurm(script, ["--dry-run"], tmp_path, env)
    assert result.returncode != 0 and "must stay inside" in result.stderr
    assert not calls(env) and not Path(env["MODULE_CAPTURE"]).exists()
    assert manifest.read_bytes() == before
    assert (outside / "manifest.json").read_text() == '{"outside": "preserve"}\n'
    assert not (outside / "training").exists()


def test_source_inventory_includes_study_module_and_sbatch_files():
    files = gate.source_snapshot()["files"]
    for name in ("src/softrole/stale_capability.py", "slurm/softrole_stale_prepare.sbatch",
                 "slurm/softrole_stale_train.sbatch"):
        assert name in files


@pytest.mark.parametrize("script", [PREPARE, SLURM])
def test_submission_helper_dry_run_routes_all_outputs_without_writes(script, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("stale_submit_test", ROOT / "scripts/submit_stokes.py")
    submit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submit)
    result = submit.main(["--title", "stale capability gate check", "--purpose", "Engineering dry-run",
                          "--dry-run", "--", str(script)])
    assert result == 0
    preview = json.loads(capsys.readouterr().out)
    run = Path(preview["run_dir"])
    assert not run.exists()
    assert preview["output_environment"]["SOFTROLE_STALE_MANIFEST"] == str(run / "manifest.json")
    assert preview["output_environment"]["SOFTROLE_STALE_RUN_ROOT"] == str(run / "training")
    assert script.name in submit.RUN_ROOT_OUTPUT_JOBS
    with pytest.raises(ValueError, match="inside"):
        submit.route_outputs(script, [], run, {"SOFTROLE_STALE_MANIFEST": str(tmp_path / "escape")})


def test_shell_syntax_and_resources():
    for script in (PREPARE, SLURM):
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "unmeasured starting values" in script.read_text()
    assert "#SBATCH --array=0-5%2" in SLURM.read_text()
    assert "#SBATCH --cpus-per-task=4" in SLURM.read_text()
