"""Locked 10-run capcon_hyper supplement to the capability-mechanism study: configuration, manifest, dispatch and launchers; no training."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import submit_stokes as submit
from softrole import mechanism_study as parent
from softrole import mechanism_supplement as study
from softrole.baselines import BASELINE_CLASSES
from softrole.config import Config
from softrole.factory import build_model
from stokes_test_support import configure_stokes_run

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/softrole_mechanism_supp_train.sbatch"
PREPARE = ROOT / "slurm/softrole_mechanism_supp_prepare.sbatch"
RUNBOOK = ROOT / "src/softrole/MECHANISM_SUPPLEMENT.md"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CONDITIONS = (("s5", 5), ("s6", 6))
ARMS = (("capcon_hyper", "capcon_hyper", "capability"),)
INTERFACE = ("VERSION", "STUDY", "CONDITIONS", "ARMS", "SEEDS", "TARGET_STEPS", "COUNT", "PROTOCOL", "require_index",
             "locate", "study_config", "expected_entries", "digest", "validate_manifest", "prepare", "launch", "main")
DECLARED = ("2026-11-10, after the parent study's training array was launched and before any frozen "
            "evaluation was run")


def expected_rows():
    return [(5 * ci + seed, condition, "capcon_hyper", seed, f"{condition}/capcon_hyper/seed{seed}")
            for ci, (condition, _) in enumerate(CONDITIONS) for seed in range(5)]


def parent_index(index):
    return 35 * (index // 5) + index % 5  # the parent study's s?/capcon/seed? entry


def test_locked_constants_and_the_public_interface_of_a_study():
    assert (study.VERSION, study.STUDY) == (1, "capability_mechanism_supplement_v1")
    assert study.CONDITIONS == CONDITIONS == parent.CONDITIONS and study.ARMS == ARMS
    assert study.SEEDS == (0, 1, 2, 3, 4) == parent.SEEDS
    assert study.TARGET_STEPS == 10_000_000 == parent.TARGET_STEPS and study.COUNT == 10
    for name in INTERFACE:
        assert hasattr(study, name) and hasattr(parent, name), name
    for name in ("require_index", "locate", "study_config", "expected_entries", "digest", "validate_manifest",
                 "prepare", "launch", "main"):
        assert callable(getattr(study, name))
    assert inspect.signature(study.launch) == inspect.signature(parent.launch)
    assert inspect.signature(study.prepare) == inspect.signature(parent.prepare)
    assert study.digest({"a": 1}) == parent.digest({"a": 1})


def test_ten_entries_with_unique_indices_mapping_and_outputs():
    rows = study.expected_entries()
    assert len(rows) == 10
    assert [(r["index"], r["condition"], r["arm"], r["seed"], r["relative_output"]) for r in rows] == expected_rows()
    assert [r["index"] for r in rows] == list(range(10))
    assert len({r["relative_output"] for r in rows}) == 10
    # index = condition_i * 5 + seed_i
    assert rows[0]["relative_output"] == "s5/capcon_hyper/seed0" and rows[4]["relative_output"] == "s5/capcon_hyper/seed4"
    assert rows[5]["relative_output"] == "s6/capcon_hyper/seed0" and rows[9]["relative_output"] == "s6/capcon_hyper/seed4"
    for row in rows:
        assert row["config"] == json.loads(json.dumps(study.study_config(row["index"]).to_dict()))
        assert row["config"]["total_steps"] == 10_000_000 and row["config"]["model"] == "capcon_hyper"
    assert study.locate(7) == ("s6", 6, "capcon_hyper", "capcon_hyper", "capability", 2)


@pytest.mark.parametrize("index", [-1, 10, 11, 69, True, False, 0., 1.5, "0", None])
def test_study_config_rejects_noninteger_and_out_of_scope_indices(index):
    for function in (study.study_config, study.locate, study.require_index):
        with pytest.raises(ValueError, match="index"):
            function(index)


def test_every_config_equals_the_parent_capcon_config_except_for_the_model():
    for index, condition, arm, seed, _ in expected_rows():
        config, capcon = study.study_config(index), parent.study_config(parent_index(index))
        assert parent.locate(parent_index(index)) == (condition, dict(CONDITIONS)[condition], "capcon", "shared",
                                                      "capability", seed)
        assert isinstance(config, Config) and capcon.model == "shared"
        differing = {key for key, value in asdict(capcon).items() if asdict(config)[key] != value}
        assert differing == {"model"} and config.model == "capcon_hyper"
        assert asdict(config) == dict(asdict(capcon), model="capcon_hyper")
        # The normalised manifest entry differs from the parent's capcon entry in `model` only.
        entry, reference = study.expected_entries()[index]["config"], parent.expected_entries()[parent_index(index)]["config"]
        assert {key for key in set(entry) | set(reference) if entry.get(key) != reference.get(key)} == {"model"}
        # Fields the parent study fixes for every arm.
        assert (config.communication, config.hidden_dim, config.hyper_dim, config.descriptor) == (
            "real", 64, 64, "capability")
        assert (config.optimizer, config.lr, config.critic_mode, config.credit_beta) == ("adam", 1e-3, "decomposed", 1.)
        assert (config.entropy_coeff, config.null_degree_correction, config.total_steps) == (0.01, True, 10_000_000)
        assert (config.max_steps, config.nprocesses, config.batch_steps) == (80, 4, 500)
        assert config.dim == dict(CONDITIONS)[condition] and config.seed == seed and config.vision == 2


def test_every_config_validates_and_builds_the_capcon_hyper_actor():
    for index in range(10):
        config = study.study_config(index)
        config.validate()
        model = build_model(config.model_kwargs())
        assert type(model) is BASELINE_CLASSES["capcon_hyper"] and model.hyper_dim == 64
        assert model.obs_dim == (702 if index < 5 else 977)
        assert sum(p.numel() for name, p in model.named_parameters() if not name.startswith("critic")) == (
            340_466 if index < 5 else 410_866)


def test_a_config_that_differs_beyond_the_model_is_refused(monkeypatch):
    real = study.replace
    monkeypatch.setattr(study, "replace", lambda config, **changes: real(config, hidden_dim=32, **changes))
    with pytest.raises(ValueError, match="beyond the model"):
        study.study_config(0)
    monkeypatch.setattr(study, "replace", real)
    monkeypatch.setattr(study, "ARMS", (("capcon_hyper", "capcon_hyper", "type"),))
    with pytest.raises(ValueError, match="beyond the model"):
        study.study_config(0)


def test_protocol_is_the_parents_with_this_arm_list_and_the_declaration():
    protocol, reference = study.PROTOCOL, parent.PROTOCOL
    assert {key for key in set(protocol) | set(reference) if protocol.get(key) != reference.get(key)} == {
        "arms", "parent_study", "declared", "purpose"}
    assert protocol["purpose"].startswith("Supplementary arm") and "onfirmatory" not in protocol["purpose"]
    assert protocol["arms"] == [{"name": "capcon_hyper", "model": "capcon_hyper", "descriptor": "capability"}]
    assert protocol["parent_study"] == "capability_mechanism_study_v2" == parent.STUDY
    assert protocol["declared"] == DECLARED
    assert [(c["name"], c["dim"]) for c in protocol["conditions"]] == list(CONDITIONS)
    assert protocol["seeds"] == [0, 1, 2, 3, 4] and protocol["target_joint_environment_steps_per_run"] == 10_000_000
    assert protocol["primary_arm"] == "l1b_decomposed_beta1" and protocol["primary_profile"] == reference["primary_profile"]
    assert (protocol["episode_log"], protocol["checkpoint_commits"]) == ("chunked-gzip", True)
    assert (protocol["device"], protocol["dtype"], protocol["threads_per_collector"]) == ("cpu", "float64", 1)
    # The parent study's protocol object is a separate copy and is untouched.
    assert "parent_study" not in reference and reference["arms"] != protocol["arms"]
    assert parent.STUDY == "capability_mechanism_study_v2" and parent.COUNT == 70 and parent.VERSION == 2


@pytest.fixture
def manifest_path(tmp_path, monkeypatch):
    source = study.source_snapshot()
    monkeypatch.setattr(study, "source_snapshot", lambda: deepcopy(source))
    path = tmp_path / "manifest.json"
    study.prepare(path)
    return path


def resign(manifest):
    """A semantically invalid manifest remains invalid with a correct outer hash."""
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()


def forbid_training(*args, **kwargs):
    pytest.fail("supplement preparation/validation/dry-run must not train")


def test_prepare_validate_round_trip_without_training_or_other_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "train", forbid_training)
    monkeypatch.setattr("softrole.train.make_env", forbid_training)
    path = tmp_path / "manifest.json"
    manifest = study.prepare(path)
    loaded = json.loads(path.read_text())
    assert loaded == manifest and study.validate_manifest(loaded) is loaded
    assert manifest.keys() == {"version", "study", "protocol", "entries", "source", "manifest_sha256"}
    assert manifest["version"] == 1 and manifest["study"] == "capability_mechanism_supplement_v1"
    assert manifest["entries"] == study.expected_entries() and len(manifest["entries"]) == 10
    assert manifest["protocol"] == json.loads(json.dumps(study.PROTOCOL))
    assert manifest["source"]["files"]
    assert manifest["source"]["sha256"] == hashlib.sha256(
        json.dumps(manifest["source"]["files"], sort_keys=True).encode()).hexdigest()
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert list(tmp_path.iterdir()) == [path]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        study.prepare(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("corruption", [
    "version", "study", "extra_key", "protocol", "protocol_steps", "parent_study", "declared", "arms_as_parent",
    "missing_entry", "duplicate_entry", "order", "arm", "condition", "seed", "index", "output", "budget", "model",
    "descriptor", "dim", "training_support", "source_hash", "source_file_hash", "boolean_index", "boolean_seed",
    "float_index", "float_seed", "float_budget", "numeric_protocol_flag",
])
def test_resigned_manifest_tampering_is_rejected_before_any_output(manifest_path, tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(study, "train", forbid_training)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["entries"]
    if corruption == "version":
        manifest["version"] = 2
    elif corruption == "study":
        manifest["study"] = parent.STUDY
    elif corruption == "extra_key":
        manifest["results"] = {"winner": "capcon_hyper"}
    elif corruption == "protocol":
        manifest["protocol"]["held_out_evaluation_completed"] = True
    elif corruption == "protocol_steps":
        manifest["protocol"]["target_joint_environment_steps_per_run"] = 2_000_000
    elif corruption == "parent_study":
        manifest["protocol"]["parent_study"] = "capability_mechanism_study_v1"
    elif corruption == "declared":
        manifest["protocol"]["declared"] = "2026-11-11, after the parent study's evaluation"
    elif corruption == "arms_as_parent":
        manifest["protocol"]["arms"] = [{"name": name, "model": model, "descriptor": descriptor}
                                        for name, model, descriptor in parent.ARMS]
    elif corruption == "missing_entry":
        entries.pop()
    elif corruption == "duplicate_entry":
        entries[1] = deepcopy(entries[0])
    elif corruption == "order":
        entries.reverse()
    elif corruption in ("boolean_index", "boolean_seed", "float_index", "float_seed"):
        key = "index" if corruption.endswith("index") else "seed"
        entries[0][key] = False if corruption.startswith("boolean") else 0.
    elif corruption == "float_budget":
        entries[0]["config"]["total_steps"] = 10_000_000.
    elif corruption == "numeric_protocol_flag":
        manifest["protocol"]["checkpoint_commits"] = 1
    elif corruption in ("arm", "condition", "seed", "index", "output"):
        key, value = {"arm": ("arm", "capcon"), "condition": ("condition", "s6"), "seed": ("seed", 4),
                      "index": ("index", 0), "output": ("relative_output", "../escape")}[corruption]
        entries[3][key] = value
    elif corruption in ("budget", "model", "descriptor", "dim", "training_support"):
        key, value = {"budget": ("total_steps", 40_000_000), "model": ("model", "shared"),
                      "descriptor": ("descriptor", "type"),
                      "dim": ("dim", 7), "training_support": ("sensing_p", [3])}[corruption]
        entries[8]["config"][key] = value
    elif corruption == "source_hash":
        manifest["source"]["sha256"] = "0" * 64
    else:
        name = next(iter(manifest["source"]["files"]))
        manifest["source"]["files"][name] = "not-a-hash"
    resign(manifest)
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        study.validate_manifest(manifest)
    with pytest.raises(ValueError):
        study.launch(corrupted, 0, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


def test_manifest_digest_and_current_checkout_are_both_required(manifest_path, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "train", forbid_training)
    manifest = json.loads(manifest_path.read_text())
    corrupt_digest = deepcopy(manifest)
    corrupt_digest["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        study.validate_manifest(corrupt_digest)
    changed_source = deepcopy(manifest["source"])
    changed_source["sha256"] = "f" * 64
    monkeypatch.setattr(study, "source_snapshot", lambda: changed_source)
    with pytest.raises(ValueError, match="[Ss]ource|[Cc]heckout"):
        study.launch(manifest_path, 0, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


def test_a_parent_study_manifest_is_rejected_here_and_a_supplement_manifest_by_the_parent(
        manifest_path, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "train", forbid_training)
    monkeypatch.setattr(parent, "train", forbid_training)
    source = json.loads(manifest_path.read_text())["source"]
    monkeypatch.setattr(parent, "source_snapshot", lambda: deepcopy(source))
    parent_path = tmp_path / "parent manifest.json"
    parent_manifest = parent.prepare(parent_path)
    assert parent_manifest["version"] == 2 and len(parent_manifest["entries"]) == 70
    supplement_manifest = json.loads(manifest_path.read_text())
    with pytest.raises(ValueError, match="mechanism supplement"):
        study.validate_manifest(parent_manifest)
    with pytest.raises(ValueError):
        parent.validate_manifest(supplement_manifest)
    for index in (0, 5, 9):
        with pytest.raises(ValueError):
            study.launch(parent_path, index, tmp_path / "runs", dry_run=True)
        with pytest.raises(ValueError):
            parent.launch(manifest_path, index, tmp_path / "runs", dry_run=True)
    # Each launcher still accepts its own manifest.
    assert study.launch(manifest_path, 0, tmp_path / "runs", dry_run=True)["study"] == study.STUDY
    assert parent.launch(parent_path, 0, tmp_path / "runs", dry_run=True)["study"] == parent.STUDY
    # Re-signing the parent manifest under the supplement's name and version still fails: its protocol and
    # entries are the parent's.
    forged = deepcopy(parent_manifest)
    forged["version"], forged["study"] = study.VERSION, study.STUDY
    resign(forged)
    with pytest.raises(ValueError):
        study.validate_manifest(forged)
    assert not (tmp_path / "runs").exists()


def test_dry_run_returns_exact_dispatch_for_all_ten_indices_and_writes_nothing(manifest_path, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "train", forbid_training)
    before = manifest_path.read_bytes()
    manifest = json.loads(before)
    root = tmp_path / "runs with spaces ' $(unused)"
    rows = study.expected_entries()
    for index in range(10):
        result = study.launch(manifest_path, index, root, dry_run=True)
        row = rows[index]
        assert result["study"] == "capability_mechanism_supplement_v1"
        assert (result["index"], result["condition"], result["arm"], result["seed"]) == (
            index, row["condition"], "capcon_hyper", row["seed"])
        assert result["config"] == row["config"]
        assert result["output"] == str((root / row["relative_output"]).resolve())
        assert result["source_sha256"] == manifest["source"]["sha256"]
        assert result["manifest_sha256"] == manifest["manifest_sha256"]
        assert result["dry_run"] is True
    assert not root.exists()
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("index", [-1, 10, 70, True, 0., "0", None])
def test_launch_rejects_bad_index_without_outputs(manifest_path, tmp_path, monkeypatch, index):
    monkeypatch.setattr(study, "train", forbid_training)
    with pytest.raises(ValueError, match="index"):
        study.launch(manifest_path, index, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_existing_run_is_refused_and_preserved_even_in_dry_run(manifest_path, tmp_path, monkeypatch, dry_run):
    monkeypatch.setattr(study, "train", forbid_training)
    output = tmp_path / "runs/s5/capcon_hyper/seed0"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("existing experiment\n")
    with pytest.raises(FileExistsError):
        study.launch(manifest_path, 0, tmp_path / "runs", dry_run=dry_run)
    assert sentinel.read_text() == "existing experiment\n"


@pytest.mark.parametrize("index", [0, 4, 5, 9])
def test_real_dispatch_uses_fixed_training_config_and_committed_compressed_ledgers(
        manifest_path, tmp_path, monkeypatch, index):
    calls = []

    def record_training(config, output, **kwargs):
        calls.append((config, Path(output), kwargs))
        Path(output).mkdir(parents=True)
        return Path(output) / "checkpoints/epoch0010.pt"

    monkeypatch.setattr(study, "train", record_training)
    original = manifest_path.read_bytes()
    result = study.launch(manifest_path, index, tmp_path / "runs")
    row = study.expected_entries()[index]
    output = (tmp_path / "runs" / row["relative_output"]).resolve()
    assert len(calls) == 1
    config, actual_output, settings = calls[0]
    assert config == study.study_config(index) and config.model == "capcon_hyper"
    assert actual_output == output
    assert settings == {"episode_log": "chunked-gzip", "checkpoint_commits": True}
    assert result["dry_run"] is False
    assert result["checkpoint"] == str(output / "checkpoints/epoch0010.pt")
    assert json.loads((output / "study.json").read_text()) == json.loads(json.dumps(result))
    assert manifest_path.read_bytes() == original


def test_source_change_during_training_prevents_success_record(manifest_path, tmp_path, monkeypatch):
    source = deepcopy(json.loads(manifest_path.read_text())["source"])
    monkeypatch.setattr(study, "source_snapshot", lambda: deepcopy(source))
    calls = []

    def changed_during_training(config, output, **kwargs):
        output = Path(output)
        output.mkdir(parents=True)
        calls.append(output)
        source["sha256"] = "f" * 64
        return output / "checkpoint.pt"

    monkeypatch.setattr(study, "train", changed_during_training)
    before = manifest_path.read_bytes()
    with pytest.raises(RuntimeError, match="Source changed during"):
        study.launch(manifest_path, 0, tmp_path / "runs")
    assert calls == [tmp_path / "runs/s5/capcon_hyper/seed0"]
    assert not (calls[0] / "study.json").exists()
    assert manifest_path.read_bytes() == before


def test_cli_prepare_and_dry_runs_are_reviewable_without_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(study, "train", forbid_training)
    manifest = tmp_path / "manifest.json"
    study.main(["prepare", "--output", str(manifest)])
    assert json.loads(capsys.readouterr().out)["study"] == "capability_mechanism_supplement_v1"
    for index in (0, 4, 5, 9):
        study.main(["launch", "--manifest", str(manifest), "--index", str(index),
                    "--run-root", str(tmp_path / "runs"), "--dry-run"])
        result = json.loads(capsys.readouterr().out)
        assert result["index"] == index and result["dry_run"] is True
        assert result["config"] == study.expected_entries()[index]["config"]
    assert not (tmp_path / "runs").exists()
    # A parent-study manifest is an argument error for this CLI, not a launch.
    with pytest.raises(SystemExit) as error:
        study.main(["prepare", "--output", str(manifest)])
    assert error.value.code == 2


@pytest.mark.parametrize("extra", [
    ["--resume"], ["--profile", "capcon_l1_40m_v1"], ["--seed", "1"], ["--total-steps", "1"],
    ["--max-steps", "3"], ["--nprocesses", "1"], ["--dim", "7"], ["--model", "cash"],
    ["--condition", "s6"], ["--episode-log", "stdout"],
])
def test_cli_has_no_resume_or_scientific_runtime_overrides(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(study, "train", forbid_training)
    with pytest.raises(SystemExit) as error:
        study.main(["launch", "--manifest", str(tmp_path / "not-read.json"), "--index", "0",
                    "--run-root", str(tmp_path / "runs"), "--dry-run", *extra])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", ["-1", "10", "70", "1.5", "true", ""])
def test_cli_rejects_bad_indices_before_launch(manifest_path, tmp_path, monkeypatch, index):
    monkeypatch.setattr(study, "train", forbid_training)
    with pytest.raises(SystemExit) as error:
        study.main(["launch", "--manifest", str(manifest_path), "--index", index,
                    "--run-root", str(tmp_path / "runs"), "--dry-run"])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


# --- submit_stokes.py routing -------------------------------------------------

@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "slurm").mkdir(parents=True)
    for path in (SLURM, PREPARE):
        (root / "slurm" / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(submit, "ROOT", root)
    monkeypatch.chdir(root)
    for name in set(key for routes in submit.ENV_ROUTES.values() for key in routes):
        monkeypatch.delenv(name, raising=False)
    return root


@pytest.mark.parametrize("job,extra", [("softrole_mechanism_supp_prepare.sbatch", []),
                                       ("softrole_mechanism_supp_train.sbatch", ["--array=0-9%10"])])
def test_submit_stokes_dry_run_accepts_both_batch_scripts(checkout, capsys, job, extra):
    args = ["--dry-run", "--title", "Capability mechanism supplement training",
            "--purpose", "Preview launcher routing only.", "--", *extra, "slurm/" + job]
    assert submit.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    run_dir = Path(plan["run_dir"])
    assert plan["output_environment"]["SOFTROLE_MECHANISM_SUPP_MANIFEST"] == str(run_dir / "manifest.json")
    assert plan["output_environment"]["SOFTROLE_MECHANISM_SUPP_RUN_ROOT"] == str(run_dir / "training")
    assert "SOFTROLE_MECHANISM_MANIFEST" not in plan["output_environment"]
    assert plan["command"][-1] == str(checkout / "slurm" / job)
    assert not (checkout / "runs").exists()


def test_both_scripts_are_registered_like_the_mechanism_training_pair():
    for name in ("softrole_mechanism_supp_train.sbatch", "softrole_mechanism_supp_prepare.sbatch"):
        assert set(submit.ENV_ROUTES[name]) == {"SOFTROLE_MECHANISM_SUPP_RUN_ROOT", "SOFTROLE_MECHANISM_SUPP_MANIFEST"}
    assert "softrole_mechanism_supp_train.sbatch" in submit.RUN_ROOT_OUTPUT_JOBS
    assert "softrole_mechanism_supp_prepare.sbatch" not in submit.RUN_ROOT_OUTPUT_JOBS  # as for the parent pair
    assert set(submit.ENV_ROUTES["softrole_mechanism_train.sbatch"]) == {
        "SOFTROLE_MECHANISM_RUN_ROOT", "SOFTROLE_MECHANISM_MANIFEST"}


def test_submit_stokes_train_cannot_route_output_outside_the_run(checkout):
    args = ["--title", "Capability mechanism supplement training", "--purpose", "Containment check only.",
            "--", "slurm/softrole_mechanism_supp_train.sbatch"]
    assert submit.main(args + ["--run-root", "/tmp/outside"]) == 2
    assert submit.main(args + ["--run-ro=/tmp/outside"]) == 2
    assert not (checkout / "runs").exists()


@pytest.mark.parametrize("variable", ["SOFTROLE_MECHANISM_SUPP_MANIFEST", "SOFTROLE_MECHANISM_SUPP_RUN_ROOT"])
@pytest.mark.parametrize("job", ["softrole_mechanism_supp_prepare.sbatch", "softrole_mechanism_supp_train.sbatch"])
def test_submit_stokes_rejects_environment_routes_outside_the_run(checkout, monkeypatch, variable, job):
    monkeypatch.setenv(variable, "/tmp/outside")
    args = ["--title", "Capability mechanism supplement training", "--purpose", "Containment check only.",
            "--", "slurm/" + job]
    assert submit.main(args) == 2
    assert not (checkout / "runs").exists()


# --- Slurm scripts with mocked srun/module/python ---------------------------

def capture_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"names = {(*THREADS, 'PYTHONPATH', 'PYTHONUNBUFFERED', 'DGLBACKEND')!r}\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['PYTHON_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': args, 'cwd': os.getcwd(), 'python': sys.argv[0], "
        "'environment': {name: os.environ.get(name) for name in names}}) + '\\n')\n"
        "assert args[:3] == ['-u', '-m', 'softrole.mechanism_supplement']\n"
        "operation = args[3]\n"
        "key = operation if operation == 'prepare' else 'launch:' + args[args.index('--index') + 1]\n"
        "if os.environ.get('MOCK_FAIL') == key: sys.exit(17)\n"
        "if operation == 'prepare':\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with output.open('x') as stream: stream.write('{\"mock_manifest\": true}\\n')\n"
        "else:\n"
        "    assert pathlib.Path(args[args.index('--manifest') + 1]).is_file()\n"
    )
    path.chmod(0o755)
    return path


def setup_environment(tmp_path, *, index="0", existing=False, defaults=False):
    root = tmp_path / "checkout"
    root.mkdir()
    tools = tmp_path / "mock-tools"
    tools.mkdir()
    for name, code in {
        "module": '#!/bin/bash\nprintf "%s\\n" "$*" > "$MODULE_CAPTURE"\n',
        "srun": '#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n',
    }.items():
        file = tools / name
        file.write_text(code)
        file.chmod(0o755)
    env = dict(os.environ)
    for name in ("SOFTROLE_MECHANISM_SUPP_MANIFEST", "SOFTROLE_MECHANISM_SUPP_RUN_ROOT", "SOFTROLE_MECHANISM_MANIFEST",
                 "SOFTROLE_MECHANISM_RUN_ROOT", "HETNET_PYTHON", "MOCK_FAIL"):
        env.pop(name, None)
    env.update({name: "32" for name in THREADS})
    env.update(PATH=f"{tools}:{env['PATH']}", PYTHONPATH="inherited-path", PYTHONUNBUFFERED="0", DGLBACKEND="other",
               MODULE_CAPTURE=str(tmp_path / "module.txt"), PYTHON_CAPTURE=str(tmp_path / "python.jsonl"),
               SLURM_SUBMIT_DIR=str(root), SLURM_ARRAY_TASK_ID=index, SLURM_ARRAY_JOB_ID="4242")
    env = configure_stokes_run(env, tmp_path)
    if defaults:
        capture_python(root / ".venv/bin/python")
        manifest, run_root = Path(env["HETNET_RUN_DIR"]) / "manifest.json", Path(env["HETNET_RUN_DIR"]) / "training"
    else:
        env["HETNET_PYTHON"] = str(capture_python(tmp_path / "capture python"))
        container = Path(env["HETNET_RUN_DIR"])
        manifest, run_root = container / "locked manifest.json", container / "training runs"
        env.update(SOFTROLE_MECHANISM_SUPP_MANIFEST=str(manifest), SOFTROLE_MECHANISM_SUPP_RUN_ROOT=str(run_root))
    if existing:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"existing": "preserve these bytes"}\n')
    return root, manifest, run_root, env


def run_slurm(script, arguments, tmp_path, env):
    return subprocess.run(["bash", str(script), *arguments], cwd=tmp_path, env=env,
                          text=True, capture_output=True)


def calls(env):
    path = Path(env["PYTHON_CAPTURE"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_environment(call, root):
    assert call["cwd"] == str(root)
    assert all(call["environment"][name] == "1" for name in THREADS)
    assert call["environment"]["PYTHONPATH"] == f"{root}/src:{root}/src/envs:inherited-path"
    assert call["environment"]["PYTHONUNBUFFERED"] == "1"
    assert call["environment"]["DGLBACKEND"] == "pytorch"


def launch_args(manifest, index, run_root, *, dry_run=False):
    result = ["-u", "-m", "softrole.mechanism_supplement", "launch", "--manifest", str(manifest),
              "--index", str(index), "--run-root", str(run_root)]
    return result + (["--dry-run"] if dry_run else [])


@pytest.mark.parametrize("index", [0, 4, 5, 9])
@pytest.mark.parametrize("dry_run", [False, True])
def test_training_slurm_invokes_python_directly_with_locked_index(tmp_path, index, dry_run):
    root, manifest, run_root, env = setup_environment(tmp_path, index=str(index), existing=True)
    result = run_slurm(SLURM, ["--dry-run"] if dry_run else [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)
    assert len(observed) == 1
    assert observed[0]["args"] == launch_args(manifest, index, run_root, dry_run=dry_run)
    assert_environment(observed[0], root)
    assert Path(env["MODULE_CAPTURE"]).read_text() == "load anaconda/anaconda-2024.10\n"
    assert not run_root.exists()


def test_training_slurm_default_paths_and_interpreter(tmp_path):
    root, manifest, run_root, env = setup_environment(tmp_path, existing=True, defaults=True)
    result = run_slurm(SLURM, ["--dry-run"], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)[0]
    assert observed["args"] == launch_args(manifest, 0, run_root, dry_run=True)
    assert observed["python"] == str(root / ".venv/bin/python")
    assert_environment(observed, root)


@pytest.mark.parametrize("index", ["10", "69", "70", "-1", "00", "01", "x", "9a", "100", "1 2"])
def test_training_slurm_rejects_invalid_array_indices_before_execution(tmp_path, index):
    _, _, _, env = setup_environment(tmp_path, index=index, existing=True)
    result = run_slurm(SLURM, [], tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


@pytest.mark.parametrize("args", [
    ["--dry"], ["--dry-run=true"], ["--dry-run", "--dry-run"], ["--resume", "checkpoint.pt"],
    ["--total-steps", "1"], ["--epochs", "1"], ["--nprocesses", "1"], ["--max-steps", "4"],
    ["--seed", "1"], ["--model", "cash"], ["--dim", "7"], ["--manifest", "other.json"],
    ["--run-root", "other"], ["--episode-log", "stdout"],
])
def test_training_slurm_rejects_all_config_and_runtime_overrides_before_execution(tmp_path, args):
    _, _, _, env = setup_environment(tmp_path, existing=True)
    result = run_slurm(SLURM, args, tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


def test_training_slurm_requires_prepared_manifest(tmp_path):
    _, _, _, env = setup_environment(tmp_path)
    result = run_slurm(SLURM, ["--dry-run"], tmp_path, env)
    assert result.returncode == 2 and "not found" in result.stderr and not calls(env)


@pytest.mark.parametrize("defaults", [False, True])
def test_preparation_creates_manifest_then_previews_all_ten_entries_without_training(tmp_path, defaults):
    root, manifest, run_root, env = setup_environment(tmp_path, defaults=defaults)
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    expected = [["-u", "-m", "softrole.mechanism_supplement", "prepare", "--output", str(manifest)]]
    expected += [launch_args(manifest, index, run_root, dry_run=True) for index in range(10)]
    observed = calls(env)
    assert [call["args"] for call in observed] == expected
    for call in observed:
        assert_environment(call, root)
    assert manifest.is_file() and not run_root.exists()


def test_existing_manifest_is_validated_by_all_previews_and_never_rewritten(tmp_path):
    _, manifest, run_root, env = setup_environment(tmp_path, existing=True)
    before = manifest.read_bytes()
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert [call["args"] for call in calls(env)] == [launch_args(manifest, i, run_root, dry_run=True)
                                                     for i in range(10)]
    assert manifest.read_bytes() == before and not run_root.exists()


@pytest.mark.parametrize("failure, expected_count", [("prepare", 1), ("launch:0", 2), ("launch:9", 11)])
def test_preparation_stops_on_first_failed_step(tmp_path, failure, expected_count):
    _, _, run_root, env = setup_environment(tmp_path)
    env["MOCK_FAIL"] = failure
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 17 and len(calls(env)) == expected_count
    assert not run_root.exists()


@pytest.mark.parametrize("args", [["--dry-run"], ["--output", "other.json"], ["--total-steps", "1"]])
def test_preparation_rejects_arguments_before_execution(tmp_path, args):
    _, _, _, env = setup_environment(tmp_path)
    result = run_slurm(PREPARE, args, tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


@pytest.mark.parametrize("variable", ["SOFTROLE_MECHANISM_SUPP_MANIFEST", "SOFTROLE_MECHANISM_SUPP_RUN_ROOT"])
@pytest.mark.parametrize("symlink", [False, True])
def test_training_slurm_refuses_outputs_outside_run_container_before_execution(variable, symlink, tmp_path):
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
    result = run_slurm(SLURM, ["--dry-run"], tmp_path, env)
    assert result.returncode != 0 and "must stay inside" in result.stderr
    assert not calls(env) and not Path(env["MODULE_CAPTURE"]).exists()
    assert manifest.read_bytes() == before
    assert (outside / "manifest.json").read_text() == '{"outside": "preserve"}\n'
    assert not (outside / "training").exists()


def test_parent_study_variables_do_not_redirect_the_supplement(tmp_path):
    _, manifest, run_root, env = setup_environment(tmp_path, existing=True)
    env.update(SOFTROLE_MECHANISM_MANIFEST=str(tmp_path / "parent manifest.json"),
               SOFTROLE_MECHANISM_RUN_ROOT=str(tmp_path / "parent training"))
    result = run_slurm(SLURM, ["--dry-run"], tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert [call["args"] for call in calls(env)] == [launch_args(manifest, 0, run_root, dry_run=True)]


def test_training_slurm_sources_the_containment_helper_before_any_python():
    text = SLURM.read_text()
    assert 'stokes_run_output "$SOFTROLE_MECHANISM_SUPP_MANIFEST" "$run_root"' in text
    assert text.index("stokes_run_output") < text.index("module load") < text.index("srun")


PARENT_TO_SUPPLEMENT = (
    ("softrole_mechanism_train", "softrole_mechanism_supp_train"),
    ("softrole_mechanism_prepare", "softrole_mechanism_supp_prepare"),
    ("SOFTROLE_MECHANISM_MANIFEST", "SOFTROLE_MECHANISM_SUPP_MANIFEST"),
    ("SOFTROLE_MECHANISM_RUN_ROOT", "SOFTROLE_MECHANISM_SUPP_RUN_ROOT"),
    ("softrole.mechanism_study", "softrole.mechanism_supplement"),
    ("{0..69}", "{0..9}"), ("--array=0-69%10", "--array=0-9%10"),
    ("^([0-9]|[1-6][0-9])$", "^[0-9]$"), ("0 through 69", "0 through 9"),
)


def executable_lines(text):
    """Every line except descriptive comments and the job name (the helper overrides it at submission)."""
    return [line for line in text.splitlines()
            if not line.startswith("#") or (line.startswith("#SBATCH") and "--job-name" not in line)]


@pytest.mark.parametrize("parent_script,script", [("softrole_mechanism_prepare.sbatch", PREPARE),
                                                  ("softrole_mechanism_train.sbatch", SLURM)])
def test_each_script_is_its_mechanism_study_counterpart_with_names_and_the_index_range_swapped(parent_script, script):
    text = (ROOT / "slurm" / parent_script).read_text()
    for old, new in PARENT_TO_SUPPLEMENT:
        text = text.replace(old, new)
    assert executable_lines(text) == executable_lines(script.read_text())


# --- launch(): output containment and symlinks -----------------------------

def test_launch_refuses_an_output_path_that_escapes_the_run_root_through_a_symlink(
        manifest_path, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "train", forbid_training)
    root, outside = tmp_path / "runs", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "s5").symlink_to(outside, target_is_directory=True)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="inside the run root"):
            study.launch(manifest_path, 0, root, dry_run=dry_run)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("target,error", [("dangling inside", FileExistsError), ("existing inside", FileExistsError),
                                          ("outside", ValueError)])
@pytest.mark.parametrize("dry_run", [False, True])
def test_launch_refuses_a_symlink_at_the_output_leaf(manifest_path, tmp_path, monkeypatch, target, error, dry_run):
    monkeypatch.setattr(study, "train", forbid_training)
    root = tmp_path / "runs"
    leaf = root / "s5/capcon_hyper/seed0"
    leaf.parent.mkdir(parents=True)
    destination = tmp_path / "elsewhere" if target == "outside" else root / "destination"
    if target != "dangling inside":
        destination.mkdir()
    leaf.symlink_to(destination, target_is_directory=True)
    with pytest.raises(error):
        study.launch(manifest_path, 0, root, dry_run=dry_run)
    assert leaf.is_symlink() and (not destination.exists() or not any(destination.iterdir()))


def test_launch_accepts_a_run_root_reached_through_a_symlink(manifest_path, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "train", forbid_training)
    real, link = tmp_path / "real runs", tmp_path / "linked runs"
    real.mkdir()
    link.symlink_to(real, target_is_directory=True)
    record = study.launch(manifest_path, 0, link, dry_run=True)
    assert record["output"] == str(real.resolve() / "s5/capcon_hyper/seed0")
    assert list(real.iterdir()) == []


def test_slurm_shell_syntax_and_declared_resources():
    result = subprocess.run(["bash", "-n", str(SLURM), str(PREPARE)], capture_output=True)
    assert result.returncode == 0, result.stderr
    for path, extra in ((SLURM, ("--cpus-per-task=4", "--mem=16G", "--time=12:00:00", "--array=0-9%10")),
                        (PREPARE, ("--cpus-per-task=1", "--mem=2G", "--time=00:10:00"))):
        text = path.read_text()
        for setting in ("--account=cenyioha", "--partition=normal", "--constraint=sapphirerapids", *extra):
            assert f"#SBATCH {setting}\n" in text
        assert "softrole.mechanism_supplement" in text and "SOFTROLE_MECHANISM_SUPP_" in text
        assert "softrole.mechanism_study" not in text
        assert "graded_pilot" not in text.lower() and "GRADED_PILOT" not in text


# --- Runbook ------------------------------------------------------------------

def test_runbook_exists_is_listed_in_gitignore_and_states_the_arm_counts_declaration_and_commands():
    text = RUNBOOK.read_text()
    assert "!/src/softrole/MECHANISM_SUPPLEMENT.md" in (ROOT / ".gitignore").read_text()
    assert DECLARED.split(", ", 1)[1] in " ".join(text.split()) and "2026-11-10" in text
    for needle in ("softrole_mechanism_supp_prepare.sbatch", "softrole_mechanism_supp_train.sbatch", "afterok",
                   "capability_mechanism_supplement_v1", "capability_mechanism_study_v2", "--array=0-9%10",
                   "SOFTROLE_MECHANISM_SUPP_MANIFEST", "SOFTROLE_MECHANISM_SUPP_RUN_ROOT", "--init-only",
                   "340,466", "410,866", "226,444", "261,644", "59,936", "77,536", "54,086", "71,686",
                   "index = condition_i * 5 + seed_i", "capcon", "cash", "hyper_dim"):
        assert needle in text, needle
    assert "hypothes" not in text.lower() and "outperform" not in text.lower()
