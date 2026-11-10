"""Locked 70-run capability-mechanism study: configuration, manifest, dispatch and launchers; no training."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import submit_stokes as submit
from softrole import mechanism_study as study
from softrole import programme
from softrole.config import COMPOSITION_TEST, COMPOSITION_TRAIN, EXTRAPOLATION_TEST, Config
from softrole.factory import build_model
from stokes_test_support import configure_stokes_run

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/softrole_mechanism_train.sbatch"
PREPARE = ROOT / "slurm/softrole_mechanism_prepare.sbatch"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CONDITIONS = (("s5", 5), ("s6", 6))
ARMS = (("capcon", "shared", "capability"), ("capcon_class_input", "shared", "type"),
        ("class_typed", "class_typed", "type"), ("ca_mlp", "ca_mlp", "capability"),
        ("ca_cc_gnn", "ca_cc_gnn", "capability"), ("rnn_exp", "rnn_exp", "capability"),
        ("cash", "cash", "capability"))
ARM_NAMES = [name for name, _, _ in ARMS]
ALLOWED = {"model", "descriptor", "dim", "vision", "capabilities", "env_version", "sensing_p", "held_out_sensing_p",
           "compositions", "held_out", "entropy_coeff", "null_degree_correction", "total_steps"}


def primary(seed):
    return programme.programme_config("l1b_decomposed_beta1", seed, 1e-3, profile=programme.CAPCON_PROFILE)


def expected_rows():
    return [(35 * ci + 5 * ai + seed, condition, arm, seed, f"{condition}/{arm}/seed{seed}")
            for ci, (condition, _) in enumerate(CONDITIONS) for ai, (arm, _, _) in enumerate(ARMS)
            for seed in range(5)]


def test_locked_constants():
    assert (study.VERSION, study.STUDY) == (2, "capability_mechanism_study_v2")
    assert study.CONDITIONS == CONDITIONS and study.ARMS == ARMS
    assert study.SEEDS == (0, 1, 2, 3, 4) and study.TARGET_STEPS == 10_000_000


def test_seventy_entries_with_unique_indices_mapping_and_outputs():
    rows = study.expected_entries()
    assert len(rows) == 70
    assert [(r["index"], r["condition"], r["arm"], r["seed"], r["relative_output"]) for r in rows] == expected_rows()
    assert [r["index"] for r in rows] == list(range(70))
    assert len({r["relative_output"] for r in rows}) == 70
    # index = condition_i * 35 + arm_i * 5 + seed_i
    assert rows[7]["relative_output"] == "s5/capcon_class_input/seed2"
    assert rows[12]["relative_output"] == "s5/class_typed/seed2"
    assert rows[35]["relative_output"] == "s6/capcon/seed0" and rows[69]["relative_output"] == "s6/cash/seed4"
    for row in rows:
        assert row["config"] == json.loads(json.dumps(study.study_config(row["index"]).to_dict()))
        assert row["config"]["total_steps"] == 10_000_000


@pytest.mark.parametrize("index", [-1, 70, 71, True, False, 0., 1.5, "0", None])
def test_study_config_rejects_noninteger_and_out_of_scope_indices(index):
    with pytest.raises(ValueError, match="index"):
        study.study_config(index)


def test_every_config_differs_from_the_primary_arm_only_in_allowed_fields():
    models = {name: model for name, model, _ in ARMS}
    descriptors = {name: descriptor for name, _, descriptor in ARMS}
    for index, condition, arm, seed, _ in expected_rows():
        config = study.study_config(index)
        base = primary(seed)
        expected = dict(asdict(base), model=models[arm], descriptor=descriptors[arm],
                        dim=dict(CONDITIONS)[condition], vision=2,
                        capabilities="graded", env_version="graded-observation-v2",
                        sensing_p=(1, 3), held_out_sensing_p=(2,), compositions=COMPOSITION_TRAIN,
                        held_out=COMPOSITION_TEST + EXTRAPOLATION_TEST, entropy_coeff=0.01,
                        null_degree_correction=True, total_steps=10_000_000)
        assert isinstance(config, Config) and asdict(config) == expected
        changed = {key for key in asdict(base) if asdict(base)[key] != asdict(config)[key]}
        assert changed <= ALLOWED
        # Fields the spec leaves to the primary arm.
        assert (config.optimizer, config.lr, config.critic_mode, config.credit_beta) == ("adam", 1e-3, "decomposed", 1.)
        assert (config.communication, config.hidden_dim, config.hyper_dim) == ("real", 64, 64)
        assert (config.max_steps, config.nprocesses, config.batch_steps) == (80, 4, 500)
        assert (config.epochs, config.updates_per_epoch, config.save_every) == (2000, 10, 10)
        assert config.seed == seed and config.failure_prob == 0 and config.descriptor == descriptors[arm]


def test_arms_within_condition_and_seed_differ_only_in_model_and_descriptor():
    for ci, _ in enumerate(CONDITIONS):
        for seed in range(5):
            configs = [asdict(study.study_config(35 * ci + 5 * ai + seed)) for ai in range(len(ARMS))]
            for (_, model, descriptor), other in zip(ARMS[1:], configs[1:]):
                differing = {k for k in configs[0] if configs[0][k] != other[k]}
                assert differing == ({"model"} if model != ARMS[0][1] else set()) | (
                    {"descriptor"} if descriptor != ARMS[0][2] else set())
            assert [c["model"] for c in configs] == [model for _, model, _ in ARMS]
            assert [c["descriptor"] for c in configs] == [descriptor for _, _, descriptor in ARMS]
    # The two class-typed arms and the class-input arm are the only ones using the class descriptor.
    assert [name for name, _, descriptor in ARMS if descriptor == "type"] == ["capcon_class_input", "class_typed"]


def test_every_config_validates_and_builds_its_model():
    for index in range(70):
        config = study.study_config(index)
        config.validate()
        model = build_model(config.model_kwargs())
        assert model is not None
        assert (type(model).__name__ == "ClassTypedNet") == (config.model == "class_typed")


def test_study_does_not_change_historical_programme_profiles():
    before = {profile: programme.profile_definition(profile) for profile in programme.PROGRAMMES}
    snapshot = deepcopy(before)
    study.expected_entries()
    assert programme.PROGRAMMES == (programme.PROGRAMME, programme.CAPCON_PROFILE)
    assert programme.SEEDS == (0, 1, 2)
    assert len(programme.entries(1e-3, profile=programme.CAPCON_PROFILE)) == 12
    assert snapshot == {profile: programme.profile_definition(profile) for profile in programme.PROGRAMMES}


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
    pytest.fail("study preparation/validation/dry-run must not train")


def test_prepare_validate_round_trip_without_training_or_other_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "train", forbid_training)
    monkeypatch.setattr("softrole.train.make_env", forbid_training)
    path = tmp_path / "manifest.json"
    manifest = study.prepare(path)
    loaded = json.loads(path.read_text())
    assert loaded == manifest and study.validate_manifest(loaded) is loaded
    assert manifest.keys() == {"version", "study", "protocol", "entries", "source", "manifest_sha256"}
    assert manifest["version"] == 2 and manifest["study"] == "capability_mechanism_study_v2"
    assert manifest["entries"] == study.expected_entries() and len(manifest["entries"]) == 70
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


def test_protocol_is_the_locked_declaration():
    protocol = study.PROTOCOL
    assert protocol["primary_arm"] == "l1b_decomposed_beta1" and protocol["primary_profile"] == programme.CAPCON_PROFILE
    assert [(c["name"], c["dim"]) for c in protocol["conditions"]] == list(CONDITIONS)
    assert [(a["name"], a["model"], a["descriptor"]) for a in protocol["arms"]] == list(ARMS)
    assert protocol["seeds"] == [0, 1, 2, 3, 4] and protocol["target_joint_environment_steps_per_run"] == 10_000_000
    assert protocol["stopping"] == "first complete collector batch reaching the target; record actual counts"
    assert protocol["checkpoint_rule"] == "first committed checkpoint at or above the target; never best epoch"
    assert (protocol["episode_log"], protocol["checkpoint_commits"]) == ("chunked-gzip", True)
    assert (protocol["device"], protocol["dtype"], protocol["threads_per_collector"]) == ("cpu", "float64", 1)


@pytest.mark.parametrize("corruption", [
    "version", "study", "extra_key", "protocol", "protocol_steps", "missing_entry", "duplicate_entry", "order",
    "arm", "condition", "seed", "index", "output", "budget", "model", "descriptor", "dim", "training_support",
    "source_hash", "source_file_hash", "boolean_index", "boolean_seed", "float_index", "float_seed",
    "float_budget", "numeric_protocol_flag",
])
def test_resigned_manifest_tampering_is_rejected_before_any_output(manifest_path, tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(study, "train", forbid_training)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["entries"]
    if corruption == "version":
        manifest["version"] = 1  # a v1 manifest must not be accepted by the v2 launcher
    elif corruption == "study":
        manifest["study"] = programme.CAPCON_PROFILE
    elif corruption == "extra_key":
        manifest["results"] = {"winner": "capcon"}
    elif corruption == "protocol":
        manifest["protocol"]["held_out_evaluation_completed"] = True
    elif corruption == "protocol_steps":
        manifest["protocol"]["target_joint_environment_steps_per_run"] = 2_000_000
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
        entries[7][key] = value
    elif corruption in ("budget", "model", "descriptor", "dim", "training_support"):
        key, value = {"budget": ("total_steps", 40_000_000), "model": ("model", "shared"),
                      "descriptor": ("descriptor", "capability"),
                      "dim": ("dim", 7), "training_support": ("sensing_p", [3])}[corruption]
        # Index 10 is s5/class_typed/seed0: the model and the class descriptor both matter.
        entries[10 if corruption in ("model", "descriptor") else 30]["config"][key] = value
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


def test_dry_run_returns_exact_dispatch_for_all_seventy_indices_and_writes_nothing(manifest_path, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "train", forbid_training)
    before = manifest_path.read_bytes()
    manifest = json.loads(before)
    root = tmp_path / "runs with spaces ' $(unused)"
    rows = study.expected_entries()
    for index in range(70):
        result = study.launch(manifest_path, index, root, dry_run=True)
        row = rows[index]
        assert result["study"] == "capability_mechanism_study_v2"
        assert (result["index"], result["condition"], result["arm"], result["seed"]) == (
            index, row["condition"], row["arm"], row["seed"])
        assert result["config"] == row["config"]
        assert result["output"] == str((root / row["relative_output"]).resolve())
        assert result["source_sha256"] == manifest["source"]["sha256"]
        assert result["manifest_sha256"] == manifest["manifest_sha256"]
        assert result["dry_run"] is True
    assert not root.exists()
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("index", [-1, 70, True, 0., "0", None])
def test_launch_rejects_bad_index_without_outputs(manifest_path, tmp_path, monkeypatch, index):
    monkeypatch.setattr(study, "train", forbid_training)
    with pytest.raises(ValueError, match="index"):
        study.launch(manifest_path, index, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_existing_run_is_refused_and_preserved_even_in_dry_run(manifest_path, tmp_path, monkeypatch, dry_run):
    monkeypatch.setattr(study, "train", forbid_training)
    output = tmp_path / "runs/s5/capcon/seed0"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("existing experiment\n")
    with pytest.raises(FileExistsError):
        study.launch(manifest_path, 0, tmp_path / "runs", dry_run=dry_run)
    assert sentinel.read_text() == "existing experiment\n"


@pytest.mark.parametrize("index", [0, 7, 12, 35, 69])
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
    assert config == study.study_config(index)
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
    assert calls == [tmp_path / "runs/s5/capcon/seed0"]
    assert not (calls[0] / "study.json").exists()
    assert manifest_path.read_bytes() == before


def test_cli_prepare_and_dry_runs_are_reviewable_without_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(study, "train", forbid_training)
    manifest = tmp_path / "manifest.json"
    study.main(["prepare", "--output", str(manifest)])
    assert json.loads(capsys.readouterr().out)["study"] == "capability_mechanism_study_v2"
    for index in (0, 12, 34, 35, 69):
        study.main(["launch", "--manifest", str(manifest), "--index", str(index),
                    "--run-root", str(tmp_path / "runs"), "--dry-run"])
        result = json.loads(capsys.readouterr().out)
        assert result["index"] == index and result["dry_run"] is True
        assert result["config"] == study.expected_entries()[index]["config"]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("extra", [
    ["--resume"], ["--profile", programme.CAPCON_PROFILE], ["--seed", "1"], ["--total-steps", "1"],
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


@pytest.mark.parametrize("index", ["-1", "70", "1.5", "true", ""])
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


@pytest.mark.parametrize("job,extra", [("softrole_mechanism_prepare.sbatch", []),
                                       ("softrole_mechanism_train.sbatch", ["--array=0-69%10"])])
def test_submit_stokes_dry_run_accepts_both_batch_scripts(checkout, capsys, job, extra):
    args = ["--dry-run", "--title", "Capability mechanism study training",
            "--purpose", "Preview launcher routing only.", "--", *extra, "slurm/" + job]
    assert submit.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    run_dir = Path(plan["run_dir"])
    assert plan["output_environment"]["SOFTROLE_MECHANISM_MANIFEST"] == str(run_dir / "manifest.json")
    assert plan["output_environment"]["SOFTROLE_MECHANISM_RUN_ROOT"] == str(run_dir / "training")
    assert plan["command"][-1] == str(checkout / "slurm" / job)
    assert not (checkout / "runs").exists()


def test_submit_stokes_train_cannot_route_output_outside_the_run(checkout):
    args = ["--title", "Capability mechanism study training", "--purpose", "Containment check only.",
            "--", "slurm/softrole_mechanism_train.sbatch"]
    assert submit.main(args + ["--run-root", "/tmp/outside"]) == 2
    assert submit.main(args + ["--run-ro=/tmp/outside"]) == 2
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
        "assert args[:3] == ['-u', '-m', 'softrole.mechanism_study']\n"
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
    for name in ("SOFTROLE_MECHANISM_MANIFEST", "SOFTROLE_MECHANISM_RUN_ROOT", "HETNET_PYTHON", "MOCK_FAIL"):
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
        env.update(SOFTROLE_MECHANISM_MANIFEST=str(manifest), SOFTROLE_MECHANISM_RUN_ROOT=str(run_root))
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
    result = ["-u", "-m", "softrole.mechanism_study", "launch", "--manifest", str(manifest),
              "--index", str(index), "--run-root", str(run_root)]
    return result + (["--dry-run"] if dry_run else [])


@pytest.mark.parametrize("index", [0, 9, 10, 34, 35, 69])
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


@pytest.mark.parametrize("index", ["70", "-1", "00", "01", "x", "6a", "100", "6 9"])
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
def test_preparation_creates_manifest_then_previews_all_seventy_entries_without_training(tmp_path, defaults):
    root, manifest, run_root, env = setup_environment(tmp_path, defaults=defaults)
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    expected = [["-u", "-m", "softrole.mechanism_study", "prepare", "--output", str(manifest)]]
    expected += [launch_args(manifest, index, run_root, dry_run=True) for index in range(70)]
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
                                                     for i in range(70)]
    assert manifest.read_bytes() == before and not run_root.exists()


@pytest.mark.parametrize("failure, expected_count", [("prepare", 1), ("launch:0", 2), ("launch:69", 71)])
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


@pytest.mark.parametrize("variable", ["SOFTROLE_MECHANISM_MANIFEST", "SOFTROLE_MECHANISM_RUN_ROOT"])
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


def test_training_slurm_sources_the_containment_helper_before_any_python():
    text = SLURM.read_text()
    assert 'stokes_run_output "$SOFTROLE_MECHANISM_MANIFEST" "$run_root"' in text
    assert text.index("stokes_run_output") < text.index("module load") < text.index("srun")


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
    leaf = root / "s5/capcon/seed0"
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
    assert record["output"] == str(real.resolve() / "s5/capcon/seed0")
    assert list(real.iterdir()) == []


def test_slurm_shell_syntax_and_declared_resources():
    result = subprocess.run(["bash", "-n", str(SLURM), str(PREPARE)], capture_output=True)
    assert result.returncode == 0, result.stderr
    for path, extra in ((SLURM, ("--cpus-per-task=4", "--mem=16G", "--time=12:00:00", "--array=0-69%10")),
                        (PREPARE, ("--cpus-per-task=1", "--mem=2G", "--time=00:10:00"))):
        text = path.read_text()
        for setting in ("--account=cenyioha", "--partition=normal", "--constraint=sapphirerapids", *extra):
            assert f"#SBATCH {setting}\n" in text
        assert "softrole.mechanism_study" in text and "SOFTROLE_MECHANISM_" in text
        assert "graded_pilot" not in text.lower() and "GRADED_PILOT" not in text
