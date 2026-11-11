"""Locked 12-run 6x6 anchor (original HetNet and shared-body arms on plain PCP): configuration, manifest, dispatch and launchers; no study-scale training."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from publication_reconstruction.study import train_arguments
from scripts import submit_stokes as submit
from softrole import anchor_study as study
from softrole import mechanism_evaluation
from softrole import mechanism_study as parent
from softrole import mechanism_supplement as supplement
from softrole import pcp_additional_seeds
from softrole.config import Config
from softrole.factory import build_model
from stokes_test_support import configure_stokes_run

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/softrole_anchor_train.sbatch"
PREPARE = ROOT / "slurm/softrole_anchor_prepare.sbatch"
RUNBOOK = ROOT / "src/softrole/ANCHOR_STUDY.md"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
ARMS = ("hetnet_original", "capcon", "capcon_class_input", "class_typed")
INTERFACE = ("VERSION", "STUDY", "ARMS", "SEEDS", "TARGET_STEPS", "COUNT", "PROTOCOL", "require_index", "locate",
             "expected_entries", "digest", "validate_manifest", "prepare", "launch", "main")
DECLARED = ("2026-11-10, after the parent study's training array was launched and before any frozen "
            "evaluation was run")
PURPOSE = ("External anchor: the original HetNet and the shared-body arms on plain 6x6 "
           "Predator-Capture-Prey with one fixed team; compared on training logs")
HETNET_INDICES, SOFTROLE_INDICES = (0, 1, 2), tuple(range(3, 12))
PARENT_ARM_POSITION = {"capcon": 0, "capcon_class_input": 1, "class_typed": 2}
S6_FIRST_PARENT_INDEX = 35  # 7 arms x 5 seeds of the s5 condition precede the s6 entries


def expected_rows():
    return [(3 * arm_i + seed, arm, "hetnet" if arm == "hetnet_original" else "softrole", seed, f"{arm}/seed{seed}")
            for arm_i, arm in enumerate(ARMS) for seed in range(3)]


def parent_index(index):
    """The parent study's s6 entry of the same arm and seed, computed independently of the module under test."""
    arm = ARMS[index // 3]
    return S6_FIRST_PARENT_INDEX + 5 * PARENT_ARM_POSITION[arm] + index % 3


def test_locked_constants_and_the_public_interface_of_a_study():
    assert (study.VERSION, study.STUDY) == (1, "pcp_6x6_anchor_v1")
    assert tuple(arm[0] for arm in study.ARMS) == ARMS and study.SEEDS == (0, 1, 2)
    assert study.TARGET_STEPS == 10_000_000 and study.COUNT == 12
    for name in INTERFACE:
        assert hasattr(study, name) and hasattr(supplement, name), name
    for name in ("require_index", "locate", "expected_entries", "digest", "validate_manifest", "prepare", "launch",
                 "main"):
        assert callable(getattr(study, name))
    assert inspect.signature(study.launch) == inspect.signature(supplement.launch)
    assert inspect.signature(study.prepare) == inspect.signature(supplement.prepare)
    assert study.digest({"a": 1}) == supplement.digest({"a": 1}) == parent.digest({"a": 1})
    # Helpers are imported, not duplicated.
    assert study.digest is parent.digest and study.run_entry is pcp_additional_seeds.run_entry
    assert study.source_snapshot.__module__ == "softrole.train" and study.train.__module__ == "softrole.train"


def test_twelve_entries_with_unique_indices_mapping_and_outputs():
    rows = study.expected_entries()
    assert len(rows) == 12
    assert [(r["index"], r["arm"], r["kind"], r["seed"], r["relative_output"]) for r in rows] == expected_rows()
    assert [r["index"] for r in rows] == list(range(12)) and len({r["relative_output"] for r in rows}) == 12
    # index = arm_i * 3 + seed_i
    assert rows[0]["relative_output"] == "hetnet_original/seed0" and rows[2]["relative_output"] == "hetnet_original/seed2"
    assert rows[4]["relative_output"] == "capcon/seed1" and rows[8]["relative_output"] == "capcon_class_input/seed2"
    assert rows[9]["relative_output"] == "class_typed/seed0" and rows[11]["relative_output"] == "class_typed/seed2"
    assert study.locate(0) == ("hetnet_original", "hetnet", "paper-v1", None, 0)
    assert study.locate(4) == ("capcon", "softrole", "shared", "capability", 1)
    assert study.locate(8) == ("capcon_class_input", "softrole", "shared", "type", 2)
    assert study.locate(11) == ("class_typed", "softrole", "class_typed", "type", 2)
    for row in rows:
        if row["kind"] == "softrole":
            assert row["config"] == json.loads(json.dumps(study.study_config(row["index"]).to_dict()))
            assert set(row) == {"index", "arm", "kind", "seed", "relative_output", "config"}
            assert row["config"]["total_steps"] == 10_000_000 and row["config"]["dim"] == 6
        else:
            assert set(row) == {"index", "arm", "kind", "seed", "relative_output", "arguments", "resolved"}
    assert json.loads(json.dumps(rows)) == rows  # the manifest round-trips exactly


@pytest.mark.parametrize("index", [-1, 12, 13, 70, True, False, 0., 1.5, "0", None])
def test_locate_and_config_reject_noninteger_and_out_of_scope_indices(index):
    for function in (study.locate, study.require_index):
        with pytest.raises(ValueError, match="index"):
            function(index)


def test_a_hetnet_index_has_no_shared_body_configuration():
    for index in HETNET_INDICES:
        with pytest.raises(ValueError, match="no parent-study configuration"):
            study.study_config(index)


def test_every_shared_body_config_equals_the_parent_s6_config_except_for_the_team_and_sensing_fields():
    for index in SOFTROLE_INDICES:
        arm, seed = ARMS[index // 3], index % 3
        config, reference = study.study_config(index), parent.study_config(parent_index(index))
        assert parent.locate(parent_index(index))[:3] == ("s6", 6, arm) and parent.locate(parent_index(index))[5] == seed
        assert isinstance(config, Config)
        differing = {key for key, value in asdict(reference).items() if asdict(config)[key] != value}
        assert differing == {"compositions", "sensing_p", "held_out_sensing_p"}
        assert (config.compositions, config.sensing_p, config.held_out_sensing_p) == (((2, 1),), (3,), ())
        assert (reference.compositions, reference.sensing_p, reference.held_out_sensing_p) == (
            ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1)), (1, 3), (2,))
        assert asdict(config) == dict(asdict(reference), compositions=((2, 1),), sensing_p=(3,), held_out_sensing_p=())
        # The normalised manifest entry differs from the parent's in these fields only.
        entry, other = study.expected_entries()[index]["config"], parent.expected_entries()[parent_index(index)]["config"]
        assert {key for key in set(entry) | set(other) if entry.get(key) != other.get(key)} <= {
            "compositions", "sensing_p", "held_out_sensing_p"}
        # Fields the parent study fixes for every arm; full sensing is level vision + 1.
        assert (config.dim, config.vision, config.capabilities, config.env_version) == (
            6, 2, "graded", "graded-observation-v2")
        assert (config.optimizer, config.lr, config.critic_mode, config.credit_beta) == ("adam", 1e-3, "decomposed", 1.)
        assert (config.entropy_coeff, config.null_degree_correction, config.total_steps) == (0.01, True, 10_000_000)
        assert (config.max_steps, config.nprocesses, config.batch_steps, config.seed) == (80, 4, 500, seed)
        assert (config.model, config.descriptor) == (reference.model, reference.descriptor)
        assert config.training_compositions == ((2, 1),) and config.sensing_p == (config.vision + 1,)


def test_every_shared_body_config_validates_and_builds_the_parents_actor():
    def parameters(model):
        return sum(p.numel() for name, p in model.named_parameters() if not name.startswith("critic"))

    for index in SOFTROLE_INDICES:
        config = study.study_config(index)
        config.validate()
        model = build_model(config.model_kwargs())
        reference = build_model(parent.study_config(parent_index(index)).model_kwargs())
        assert type(model) is type(reference) and model.obs_dim == reference.obs_dim == 977
        assert parameters(model) == parameters(reference)


def test_a_config_that_differs_beyond_the_three_fields_is_refused(monkeypatch):
    real = study.replace
    monkeypatch.setattr(study, "replace", lambda config, **changes: real(config, hidden_dim=32, **changes))
    with pytest.raises(ValueError, match="beyond"):
        study.study_config(3)
    monkeypatch.setattr(study, "replace", real)
    monkeypatch.setattr(study, "ARMS", (study.ARMS[0], ("capcon", "softrole", "shared", "type")) + study.ARMS[2:])
    with pytest.raises(ValueError, match="beyond"):
        study.study_config(3)
    monkeypatch.undo()
    monkeypatch.setattr(study, "SENSING_P", (1, 3))  # a sensing level the parent already has
    with pytest.raises(ValueError, match="beyond"):
        study.study_config(3)


def flags(arguments):
    """{flag: [values]} of a publication_reconstruction argument list."""
    result, current = {}, None
    for item in arguments[1:]:
        if item.startswith("--"):
            current = item
            result[current] = []
        else:
            result[current].append(item)
    return result


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hetnet_arguments_differ_from_the_5x5_workload_in_exactly_the_declared_items(tmp_path, seed):
    output = tmp_path / "run" / "hetnet_original" / f"seed{seed}" / "training"
    anchor = study.hetnet_arguments(seed, output)
    reference = train_arguments(3, tmp_path / "root", reconstruction_spec="paper-v1", message_backend="torch-v1")
    assert reference[reference.index("--task") + 1] == "pcp" and reference[reference.index("--variant") + 1] == "real"
    a, r = flags(anchor), flags(reference)
    assert anchor[0] == reference[0] == "train"
    assert set(a) - set(r) == {"--dim"} and set(r) == set(a) - {"--dim"}
    assert {flag for flag in r if a[flag] != r[flag]} == {"--output", "--max-env-steps", "--milestones"} | (
        {"--seed"} if seed else set())
    assert a["--dim"] == ["6"] and a["--seed"] == [str(seed)] and a["--output"] == [str(output)]
    assert a["--max-env-steps"] == ["10000000"] and a["--milestones"] == ["10000000"]
    assert r["--max-env-steps"] == ["40000000"] and r["--milestones"] == ["10000000", "20000000", "30000000", "40000000"]
    for kept in ("--collectors", "--batch-steps", "--updates-per-epoch", "--horizon", "--epochs", "--save-every",
                 "--episode-log", "--model-spec", "--env-version", "--recipe", "--reconstruction-spec",
                 "--message-backend", "--wall-seconds", "--task", "--variant"):
        assert a[kept] == r[kept], kept
    assert (a["--reconstruction-spec"], a["--message-backend"], a["--model-spec"]) == (
        ["paper-v1"], ["torch-v1"], ["paper-v1"])
    # Applying the five edits to the reference list reproduces the anchor list token for token.
    edited = list(reference)
    edited[edited.index("--seed") + 1] = str(seed)
    edited[edited.index("--output") + 1] = str(output)
    edited[edited.index("--max-env-steps") + 1] = "10000000"
    start = edited.index("--milestones") + 1
    edited[start:start + 4] = ["10000000"]
    edited[edited.index("--seed") + 2:edited.index("--seed") + 2] = ["--dim", "6"]
    assert anchor == edited


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hetnet_protocol_is_the_5x5_paper_v1_pcp_protocol_with_the_declared_changes(tmp_path, seed):
    from publication_reconstruction.__main__ import parser, resolve
    resolved = study.hetnet_protocol(study.hetnet_arguments(seed, tmp_path / "anchor"))
    reference = resolve(parser().parse_args(
        train_arguments(3, tmp_path, reconstruction_spec="paper-v1", message_backend="torch-v1")))
    assert {key for key in set(resolved) | set(reference) if resolved.get(key) != reference.get(key)} == {
        "command", "dim", "max_env_steps", "milestones", "output"} | ({"seed"} if seed else set())
    assert resolved["dim"] == 6 and "dim" not in reference and resolved["seed"] == seed
    assert (resolved["task"], resolved["variant"], resolved["reconstruction_spec"], resolved["message_backend"]) == (
        "pcp", "real", "paper-v1", "torch-v1")
    assert (resolved["max_env_steps"], resolved["milestones"]) == (10_000_000, [10_000_000])
    assert resolved["collectors"] == 4 and resolved["episode_horizon"] == 80 and resolved["wall_seconds"] == 165600
    command = resolved["command"]
    assert command[command.index("--dim") + 1] == "6" and command.count("--dim") == 1


def test_manifest_hetnet_entries_equal_the_protocol_resolved_for_a_real_run_root(tmp_path):
    root = (tmp_path / "a root with ' quote").resolve()
    for row in study.expected_entries()[:3]:
        assert row["resolved"]["command"][0] == study.PLACEHOLDER_PYTHON
        assert row["arguments"][row["arguments"].index("--output") + 1] == (
            f"{study.PLACEHOLDER_ROOT}/{row['relative_output']}/training")
        actual = study.hetnet_protocol(study.hetnet_arguments(row["seed"], root / row["relative_output"] / "training"))
        text = json.dumps(study.locked_protocol(actual)).replace(json.dumps(str(root))[1:-1], study.PLACEHOLDER_ROOT)
        assert json.loads(text) == row["resolved"]
        assert actual["command"][0] == sys.executable


def test_protocol_records_the_declared_scope_and_the_differences():
    protocol = study.PROTOCOL
    assert protocol["study"] == "pcp_6x6_anchor_v1" and protocol["purpose"] == PURPOSE
    assert [arm["name"] for arm in protocol["arms"]] == list(ARMS)
    assert [arm["kind"] for arm in protocol["arms"]] == ["hetnet", "softrole", "softrole", "softrole"]
    assert protocol["seeds"] == [0, 1, 2] and protocol["target_joint_environment_steps_per_run"] == 10_000_000
    assert protocol["map"] == {"dim": 6} and protocol["team"] == {"perception": 2, "capture": 1}
    assert protocol["sensing"]["level"] == "full" and protocol["sensing"]["sensing_p"] == [3]
    assert protocol["parent_study"] == "capability_mechanism_study_v2" == parent.STUDY
    assert protocol["declared"] == DECLARED
    assert protocol["hetnet_workload"]["changed_from_5x5_workload"] == [
        "--seed", "--output", "--dim", "--max-env-steps", "--milestones"]
    for key, value in protocol["softrole_training"].items():
        assert value == parent.PROTOCOL[key], key
    differences = " ".join(protocol["differences"])
    assert any("original HetNet system's own learner, observation and critics" in line
               and "not a controlled comparison" in line for line in protocol["differences"])
    assert "compositions=((2, 1),), sensing_p=(3,) and held_out_sensing_p=()" in differences
    assert "--dim 6" in differences and "--max-env-steps 10000000" in differences
    assert "hypothes" not in json.dumps(protocol).lower()


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


def forbid(*args, **kwargs):
    pytest.fail("anchor preparation/validation/dry-run must not train or launch")


def forbid_everything(monkeypatch):
    import publication_reconstruction.__main__ as reconstruction
    monkeypatch.setattr(study, "train", forbid)
    monkeypatch.setattr(study, "run_entry", forbid)
    monkeypatch.setattr(reconstruction, "main", forbid)


def test_prepare_validate_round_trip_without_training_or_other_artifacts(tmp_path, monkeypatch):
    forbid_everything(monkeypatch)
    monkeypatch.setattr("softrole.train.make_env", forbid)
    path = tmp_path / "manifest.json"
    manifest = study.prepare(path)
    loaded = json.loads(path.read_text())
    assert loaded == manifest and study.validate_manifest(loaded) is loaded
    assert manifest.keys() == {"version", "study", "protocol", "entries", "source", "manifest_sha256"}
    assert manifest["version"] == 1 and manifest["study"] == "pcp_6x6_anchor_v1"
    assert manifest["entries"] == study.expected_entries() and len(manifest["entries"]) == 12
    assert manifest["protocol"] == json.loads(json.dumps(study.PROTOCOL))
    assert manifest["source"]["files"]
    assert manifest["source"]["sha256"] == hashlib.sha256(
        json.dumps(manifest["source"]["files"], sort_keys=True).encode()).hexdigest()
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert any(name.endswith("softrole/anchor_study.py") for name in manifest["source"]["files"])
    assert any(name.endswith("slurm/softrole_anchor_train.sbatch") for name in manifest["source"]["files"])
    assert list(tmp_path.iterdir()) == [path]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        study.prepare(path)
    assert path.read_bytes() == original


def test_the_manifest_does_not_depend_on_the_interpreter_or_the_working_directory(tmp_path, monkeypatch):
    before = study.expected_entries()
    monkeypatch.setattr(sys, "executable", "/some/other/venv/bin/python")
    monkeypatch.chdir(tmp_path)
    assert study.expected_entries() == before and "/some/other" not in json.dumps(before)


@pytest.mark.parametrize("corruption", [
    "version", "study", "extra_key", "protocol", "protocol_steps", "parent_study", "declared", "differences",
    "map", "team", "missing_entry", "duplicate_entry", "order", "arm", "kind", "seed", "index", "output", "budget",
    "model", "descriptor", "dim", "team_composition", "sensing", "held_out_sensing", "hetnet_dim", "hetnet_steps",
    "hetnet_milestones", "hetnet_output", "hetnet_spec", "hetnet_resolved", "source_hash", "source_file_hash",
    "boolean_index", "boolean_seed", "float_index", "float_seed", "float_budget", "numeric_protocol_flag",
])
def test_resigned_manifest_tampering_is_rejected_before_any_output(manifest_path, tmp_path, monkeypatch, corruption):
    forbid_everything(monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["entries"]

    def argument(entry, flag, value, position=1):
        entry["arguments"][entry["arguments"].index(flag) + position] = value

    if corruption == "version":
        manifest["version"] = 2
    elif corruption == "study":
        manifest["study"] = parent.STUDY
    elif corruption == "extra_key":
        manifest["results"] = {"winner": "capcon"}
    elif corruption == "protocol":
        manifest["protocol"]["held_out_evaluation_completed"] = True
    elif corruption == "protocol_steps":
        manifest["protocol"]["target_joint_environment_steps_per_run"] = 2_000_000
    elif corruption == "parent_study":
        manifest["protocol"]["parent_study"] = "capability_mechanism_study_v1"
    elif corruption == "declared":
        manifest["protocol"]["declared"] = "2026-11-11, after the parent study's evaluation"
    elif corruption == "differences":
        manifest["protocol"]["differences"] = manifest["protocol"]["differences"][1:]
    elif corruption == "map":
        manifest["protocol"]["map"]["dim"] = 5
    elif corruption == "team":
        manifest["protocol"]["team"] = {"perception": 3, "capture": 1}
    elif corruption == "missing_entry":
        entries.pop()
    elif corruption == "duplicate_entry":
        entries[4] = deepcopy(entries[3])
    elif corruption == "order":
        entries.reverse()
    elif corruption in ("boolean_index", "boolean_seed", "float_index", "float_seed"):
        key = "index" if corruption.endswith("index") else "seed"
        entries[3][key] = False if corruption.startswith("boolean") else 0.
    elif corruption == "float_budget":
        entries[3]["config"]["total_steps"] = 10_000_000.
    elif corruption == "numeric_protocol_flag":
        manifest["protocol"]["softrole_training"]["checkpoint_commits"] = 1
    elif corruption in ("arm", "kind", "seed", "index", "output"):
        key, value = {"arm": ("arm", "capcon_class_input"), "kind": ("kind", "softrole"), "seed": ("seed", 2),
                      "index": ("index", 0), "output": ("relative_output", "../escape")}[corruption]
        entries[4 if corruption != "kind" else 0][key] = value
    elif corruption in ("budget", "model", "descriptor", "dim", "team_composition", "sensing", "held_out_sensing"):
        key, value = {"budget": ("total_steps", 40_000_000), "model": ("model", "capcon_hyper"),
                      "descriptor": ("descriptor", "capability"), "dim": ("dim", 5),
                      "team_composition": ("compositions", [[1, 1], [1, 2], [2, 1], [2, 2], [3, 1]]),
                      "sensing": ("sensing_p", [1, 3]), "held_out_sensing": ("held_out_sensing_p", [2])}[corruption]
        entries[8]["config"][key] = value
    elif corruption == "hetnet_dim":
        argument(entries[1], "--dim", "5")
    elif corruption == "hetnet_steps":
        argument(entries[1], "--max-env-steps", "40000000")
    elif corruption == "hetnet_milestones":
        argument(entries[1], "--milestones", "20000000")
    elif corruption == "hetnet_output":
        argument(entries[1], "--output", "/elsewhere/hetnet_original/seed1/training")
    elif corruption == "hetnet_spec":
        argument(entries[1], "--message-backend", "dgl")
    elif corruption == "hetnet_resolved":
        entries[1]["resolved"]["dim"] = 5
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
    forbid_everything(monkeypatch)
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


def test_other_studies_manifests_are_rejected_here_and_an_anchor_manifest_by_the_others(
        manifest_path, tmp_path, monkeypatch):
    forbid_everything(monkeypatch)
    monkeypatch.setattr(parent, "train", forbid)
    monkeypatch.setattr(supplement, "train", forbid)
    source = json.loads(manifest_path.read_text())["source"]
    monkeypatch.setattr(parent, "source_snapshot", lambda: deepcopy(source))
    monkeypatch.setattr(supplement, "source_snapshot", lambda: deepcopy(source))
    parent_path, supplement_path = tmp_path / "parent manifest.json", tmp_path / "supplement manifest.json"
    parent_manifest, supplement_manifest = parent.prepare(parent_path), supplement.prepare(supplement_path)
    anchor_manifest = json.loads(manifest_path.read_text())
    assert (len(parent_manifest["entries"]), len(supplement_manifest["entries"])) == (70, 10)
    for foreign, path in ((parent_manifest, parent_path), (supplement_manifest, supplement_path)):
        with pytest.raises(ValueError, match="anchor"):
            study.validate_manifest(foreign)
        for index in (0, 3, 11):
            with pytest.raises(ValueError):
                study.launch(path, index, tmp_path / "runs", dry_run=True)
    with pytest.raises(ValueError):
        parent.validate_manifest(anchor_manifest)
    with pytest.raises(ValueError):
        supplement.validate_manifest(anchor_manifest)
    for index in (0, 5, 9):
        with pytest.raises(ValueError):
            parent.launch(manifest_path, index, tmp_path / "runs", dry_run=True)
        with pytest.raises(ValueError):
            supplement.launch(manifest_path, index, tmp_path / "runs", dry_run=True)
    # Each launcher still accepts its own manifest.
    assert study.launch(manifest_path, 0, tmp_path / "runs", dry_run=True)["study"] == study.STUDY
    assert parent.launch(parent_path, 0, tmp_path / "runs", dry_run=True)["study"] == parent.STUDY
    assert supplement.launch(supplement_path, 0, tmp_path / "runs", dry_run=True)["study"] == supplement.STUDY
    # Re-signing a foreign manifest under this study's name and version still fails: its entries are not ours.
    for foreign in (parent_manifest, supplement_manifest):
        forged = deepcopy(foreign)
        forged["version"], forged["study"] = study.VERSION, study.STUDY
        resign(forged)
        with pytest.raises(ValueError):
            study.validate_manifest(forged)
    assert not (tmp_path / "runs").exists()
    # The frozen evaluation of the parent study does not list the anchor.
    assert study.STUDY not in mechanism_evaluation.STUDIES


def test_dry_run_returns_the_exact_dispatch_for_all_twelve_indices_and_writes_nothing(
        manifest_path, tmp_path, monkeypatch):
    forbid_everything(monkeypatch)
    before = manifest_path.read_bytes()
    manifest = json.loads(before)
    root = tmp_path / "runs with spaces ' $(unused)"
    rows = study.expected_entries()
    for index in range(12):
        result = study.launch(manifest_path, index, root, dry_run=True)
        row, output = rows[index], (root / rows[index]["relative_output"]).resolve()
        assert result["study"] == "pcp_6x6_anchor_v1" and result["dry_run"] is True
        assert (result["index"], result["arm"], result["kind"], result["seed"]) == (
            index, row["arm"], row["kind"], row["seed"])
        assert result["output"] == str(output)
        assert result["source_sha256"] == manifest["source"]["sha256"]
        assert result["manifest_sha256"] == manifest["manifest_sha256"]
        assert "exit_code" not in result and "checkpoint" not in result
        if row["kind"] == "hetnet":
            assert result["arguments"] == study.hetnet_arguments(row["seed"], output / "training")
            assert result["resolved"] == study.hetnet_protocol(result["arguments"])
            assert result["resolved"]["dim"] == 6 and result["resolved"]["output"] == str(output / "training")
            assert "config" not in result
        else:
            assert result["config"] == row["config"] and "arguments" not in result
    assert not root.exists()
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("index", [-1, 12, 70, True, 0., "0", None])
def test_launch_rejects_bad_index_without_outputs(manifest_path, tmp_path, monkeypatch, index):
    forbid_everything(monkeypatch)
    with pytest.raises(ValueError, match="index"):
        study.launch(manifest_path, index, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", [0, 11])
@pytest.mark.parametrize("dry_run", [False, True])
def test_existing_run_is_refused_and_preserved_even_in_dry_run(manifest_path, tmp_path, monkeypatch, index, dry_run):
    forbid_everything(monkeypatch)
    output = tmp_path / "runs" / study.expected_entries()[index]["relative_output"]
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("existing experiment\n")
    with pytest.raises(FileExistsError):
        study.launch(manifest_path, index, tmp_path / "runs", dry_run=dry_run)
    assert sentinel.read_text() == "existing experiment\n"


@pytest.mark.parametrize("index", [3, 6, 11])
def test_real_softrole_dispatch_uses_the_fixed_training_config_and_committed_compressed_ledgers(
        manifest_path, tmp_path, monkeypatch, index):
    calls = []
    forbid_everything(monkeypatch)

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
    assert config == study.study_config(index) and actual_output == output
    assert (config.compositions, config.sensing_p, config.dim) == (((2, 1),), (3,), 6)
    assert settings == {"episode_log": "chunked-gzip", "checkpoint_commits": True}
    assert result["dry_run"] is False and result["checkpoint"] == str(output / "checkpoints/epoch0010.pt")
    assert json.loads((output / "study.json").read_text()) == json.loads(json.dumps(result))
    assert manifest_path.read_bytes() == original


def test_source_change_during_softrole_training_prevents_success_record(manifest_path, tmp_path, monkeypatch):
    forbid_everything(monkeypatch)
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
        study.launch(manifest_path, 4, tmp_path / "runs")
    assert calls == [tmp_path / "runs/capcon/seed1"] and not (calls[0] / "study.json").exists()
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("index", [0, 2])
def test_real_hetnet_dispatch_runs_the_locked_arguments_and_writes_the_launch_records(
        manifest_path, tmp_path, monkeypatch, index):
    import publication_reconstruction.__main__ as reconstruction
    forbid_everything(monkeypatch)
    calls = []
    monkeypatch.setattr(reconstruction, "main", lambda arguments: calls.append(arguments) or 0)
    monkeypatch.setattr(study, "run_entry", pcp_additional_seeds.run_entry)
    manifest = json.loads(manifest_path.read_text())
    original = manifest_path.read_bytes()
    result = study.launch(manifest_path, index, tmp_path / "runs")
    row = study.expected_entries()[index]
    seed_dir = (tmp_path / "runs" / row["relative_output"]).resolve()
    expected = study.hetnet_arguments(row["seed"], seed_dir / "training")
    assert calls == [expected] and result["exit_code"] == 0 and result["arguments"] == expected
    assert result["output"] == str(seed_dir) and result["kind"] == "hetnet" and "checkpoint" not in result
    launch = json.loads((seed_dir / "launch.json").read_text())
    assert launch["kind"] == "hetnet" and launch["arguments"] == expected
    assert launch["resolved"] == study.hetnet_protocol(expected) and launch["resolved"]["dim"] == 6
    assert launch["output"] == str(seed_dir / "training") and launch["arm"] == "hetnet_original"
    assert launch["manifest"] == str(manifest_path.resolve())
    assert launch["manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert launch["anchor_manifest_sha256"] == manifest["manifest_sha256"] and launch["study"] == study.STUDY
    assert launch["source_sha256"] == manifest["source"]["sha256"]
    assert launch["environment"]["executable"] == sys.executable and set(launch["slurm"]) >= {"SLURM_JOB_ID"}
    assert json.loads((seed_dir / "launcher_exit.json").read_text()) == {"exit_code": 0}
    assert not (seed_dir / "training").exists()  # created by the (mocked) trainer, never by the anchor launcher
    assert manifest_path.read_bytes() == original
    with pytest.raises(FileExistsError):
        study.launch(manifest_path, index, tmp_path / "runs")
    assert len(calls) == 1


@pytest.mark.parametrize("code", [1, 3])
def test_hetnet_launcher_exit_code_is_recorded_and_becomes_the_process_exit_status(
        manifest_path, tmp_path, monkeypatch, capsys, code):
    import publication_reconstruction.__main__ as reconstruction
    forbid_everything(monkeypatch)
    monkeypatch.setattr(reconstruction, "main", lambda arguments: code)
    monkeypatch.setattr(study, "run_entry", pcp_additional_seeds.run_entry)
    status = study.main(["launch", "--manifest", str(manifest_path), "--index", "1",
                         "--run-root", str(tmp_path / "runs")])
    assert status == code
    assert '"exit_code": %d' % code in capsys.readouterr().out
    assert json.loads((tmp_path / "runs/hetnet_original/seed1/launcher_exit.json").read_text()) == {"exit_code": code}


def test_a_failing_hetnet_launcher_leaves_an_exit_record_and_the_error_propagates(manifest_path, tmp_path, monkeypatch):
    import publication_reconstruction.__main__ as reconstruction
    forbid_everything(monkeypatch)

    def fail(arguments):
        raise RuntimeError("deliberate launcher failure")

    monkeypatch.setattr(reconstruction, "main", fail)
    monkeypatch.setattr(study, "run_entry", pcp_additional_seeds.run_entry)
    with pytest.raises(RuntimeError, match="deliberate"):
        study.launch(manifest_path, 0, tmp_path / "runs")
    assert json.loads((tmp_path / "runs/hetnet_original/seed0/launcher_exit.json").read_text()) == {"exit_code": 1}


def test_hetnet_source_check_precedes_the_launcher(manifest_path, tmp_path, monkeypatch):
    import publication_reconstruction.__main__ as reconstruction
    import softrole.train as training
    forbid_everything(monkeypatch)
    monkeypatch.setattr(reconstruction, "main", forbid)
    monkeypatch.setattr(study, "run_entry", pcp_additional_seeds.run_entry)
    changed = deepcopy(study.source_snapshot())
    changed["sha256"] = "e" * 64
    monkeypatch.setattr(training, "source_snapshot", lambda: changed)  # the check run_entry itself performs
    with pytest.raises(ValueError, match="Checkout changed"):
        study.launch(manifest_path, 0, tmp_path / "runs")
    assert not (tmp_path / "runs/hetnet_original/seed0/launcher_exit.json").exists()


def test_cli_prepare_and_dry_runs_for_both_kinds_are_reviewable_without_training(tmp_path, monkeypatch, capsys):
    forbid_everything(monkeypatch)
    manifest = tmp_path / "manifest.json"
    assert study.main(["prepare", "--output", str(manifest)]) == 0
    assert json.loads(capsys.readouterr().out)["study"] == "pcp_6x6_anchor_v1"
    for index in (0, 3, 6, 11):
        assert study.main(["launch", "--manifest", str(manifest), "--index", str(index),
                           "--run-root", str(tmp_path / "runs"), "--dry-run"]) == 0
        result = json.loads(capsys.readouterr().out)
        entry = study.expected_entries()[index]
        assert result["index"] == index and result["dry_run"] is True and result["arm"] == entry["arm"]
        assert ("arguments" in result) == (entry["kind"] == "hetnet") and ("config" in result) == (
            entry["kind"] == "softrole")
        if entry["kind"] == "hetnet":
            assert result["arguments"][result["arguments"].index("--dim") + 1] == "6"
        else:
            assert result["config"] == entry["config"]
    assert not (tmp_path / "runs").exists()
    with pytest.raises(SystemExit) as error:
        study.main(["prepare", "--output", str(manifest)])
    assert error.value.code == 2


@pytest.mark.parametrize("extra", [
    ["--resume"], ["--profile", "capcon_l1_40m_v1"], ["--seed", "1"], ["--total-steps", "1"],
    ["--max-steps", "3"], ["--nprocesses", "1"], ["--dim", "5"], ["--model", "cash"], ["--max-env-steps", "1"],
    ["--condition", "s6"], ["--episode-log", "stdout"], ["--milestones", "1"],
])
def test_cli_has_no_resume_or_scientific_runtime_overrides(tmp_path, monkeypatch, extra):
    forbid_everything(monkeypatch)
    with pytest.raises(SystemExit) as error:
        study.main(["launch", "--manifest", str(tmp_path / "not-read.json"), "--index", "0",
                    "--run-root", str(tmp_path / "runs"), "--dry-run", *extra])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", ["-1", "12", "70", "1.5", "true", ""])
def test_cli_rejects_bad_indices_before_launch(manifest_path, tmp_path, monkeypatch, index):
    forbid_everything(monkeypatch)
    with pytest.raises(SystemExit) as error:
        study.main(["launch", "--manifest", str(manifest_path), "--index", index,
                    "--run-root", str(tmp_path / "runs"), "--dry-run"])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


# --- launch(): output containment and symlinks -----------------------------

@pytest.mark.parametrize("index,arm", [(0, "hetnet_original"), (3, "capcon")])
def test_launch_refuses_an_output_path_that_escapes_the_run_root_through_a_symlink(
        manifest_path, tmp_path, monkeypatch, index, arm):
    forbid_everything(monkeypatch)
    root, outside = tmp_path / "runs", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / arm).symlink_to(outside, target_is_directory=True)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="inside the run root"):
            study.launch(manifest_path, index, root, dry_run=dry_run)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("index,arm", [(0, "hetnet_original"), (3, "capcon")])
@pytest.mark.parametrize("target,error", [("dangling inside", FileExistsError), ("existing inside", FileExistsError),
                                          ("outside", ValueError)])
@pytest.mark.parametrize("dry_run", [False, True])
def test_launch_refuses_a_symlink_at_the_output_leaf(
        manifest_path, tmp_path, monkeypatch, index, arm, target, error, dry_run):
    forbid_everything(monkeypatch)
    root = tmp_path / "runs"
    leaf = root / arm / "seed0"
    leaf.parent.mkdir(parents=True)
    destination = tmp_path / "elsewhere" if target == "outside" else root / "destination"
    if target != "dangling inside":
        destination.mkdir()
    leaf.symlink_to(destination, target_is_directory=True)
    with pytest.raises(error):
        study.launch(manifest_path, index, root, dry_run=dry_run)
    assert leaf.is_symlink() and (not destination.exists() or not any(destination.iterdir()))


def test_launch_accepts_a_run_root_reached_through_a_symlink(manifest_path, tmp_path, monkeypatch):
    forbid_everything(monkeypatch)
    real, link = tmp_path / "real runs", tmp_path / "linked runs"
    real.mkdir()
    link.symlink_to(real, target_is_directory=True)
    for index, relative in ((0, "hetnet_original/seed0"), (3, "capcon/seed0")):
        assert study.launch(manifest_path, index, link, dry_run=True)["output"] == str(real.resolve() / relative)
    assert list(real.iterdir()) == []


# --- tiny end-to-end launches ------------------------------------------------

REAL_STUDY_CONFIG, REAL_HETNET_ARGUMENTS = study.study_config, study.hetnet_arguments


def tiny_study_config(index):
    return replace(REAL_STUDY_CONFIG(index), max_steps=10, batch_steps=20, nprocesses=1, epochs=3,
                   updates_per_epoch=1, save_every=1, hidden_dim=16, pre_dim=16, heads=2, head_dim=4, msg_dim=8,
                   total_steps=40)


def tiny_hetnet_arguments(seed, output):
    arguments = REAL_HETNET_ARGUMENTS(seed, output)
    for flag, value in (("--collectors", "1"), ("--epochs", "1"), ("--updates-per-epoch", "2"),
                        ("--batch-steps", "6"), ("--horizon", "6"), ("--save-every", "1"),
                        ("--max-env-steps", "12"), ("--milestones", "12")):
        arguments[arguments.index(flag) + 1] = value
    return arguments


def test_tiny_end_to_end_launch_of_a_shared_body_index(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "study_config", tiny_study_config)
    manifest = tmp_path / "manifest.json"
    study.prepare(manifest)
    result = study.launch(manifest, 3, tmp_path / "training")
    output = tmp_path / "training/capcon/seed0"
    assert Path(result["checkpoint"]).is_file() and Path(result["checkpoint"]).is_relative_to(output.resolve())
    assert json.loads((output / "study.json").read_text())["arm"] == "capcon"
    saved = json.loads((output / "config.json").read_text())
    assert (saved["dim"], saved["compositions"], saved["sensing_p"], saved["model"]) == (6, [[2, 1]], [3], "shared")
    assert (output / "finished.json").is_file() and (output / "metrics.jsonl").is_file()
    with pytest.raises(FileExistsError):
        study.launch(manifest, 3, tmp_path / "training")


def test_tiny_end_to_end_launch_of_the_hetnet_index_trains_on_a_six_by_six_map(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(study, "hetnet_arguments", tiny_hetnet_arguments)
    manifest = tmp_path / "manifest.json"
    study.prepare(manifest)
    result = study.launch(manifest, 1, tmp_path / "training")
    seed_dir, run = tmp_path / "training/hetnet_original/seed1", tmp_path / "training/hetnet_original/seed1/training"
    assert result["exit_code"] == 0 and result["kind"] == "hetnet"
    assert json.loads((seed_dir / "launcher_exit.json").read_text()) == {"exit_code": 0}
    launch = json.loads((seed_dir / "launch.json").read_text())
    assert launch["kind"] == "hetnet" and launch["arguments"] == result["arguments"]
    assert launch["resolved"]["dim"] == 6 and launch["output"] == str(run.resolve())
    assert (run / "exit_code.txt").read_text().strip() == "0"
    status = json.loads((run / "run_status.json").read_text())
    assert status["scientific_budget_completed"] is True
    protocol = json.loads((run / "protocol.json").read_text())
    assert protocol["dim"] == 6 and protocol["seed"] == 1 and protocol["reconstruction_spec"] == "paper-v1"
    assert protocol["message_backend"] == "torch-v1" and protocol["task"] == "pcp" and protocol["variant"] == "real"
    resolved = json.loads((run / "resolved_args.json").read_text())
    assert resolved["dim"] == 6 and resolved["num_inputs"] == 25 * (6 ** 2 + 4)  # 725 on the 5x5 map
    assert (resolved["nfriendly_P"], resolved["nfriendly_A"], resolved["vision"]) == (2, 1, 2)
    saved = torch.load(status["checkpoint"], map_location="cpu", weights_only=False)
    assert tuple(saved["policy_net"]["prepro_stat.P.weight"].shape) == (900, 900)  # 6*6 cells x 25 vision squares
    with pytest.raises(FileExistsError):
        study.launch(manifest, 1, tmp_path / "training")


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


@pytest.mark.parametrize("job,extra", [("softrole_anchor_prepare.sbatch", []),
                                       ("softrole_anchor_train.sbatch", ["--array=0-11%12"])])
def test_submit_stokes_dry_run_accepts_both_batch_scripts(checkout, capsys, job, extra):
    args = ["--dry-run", "--title", "PCP 6x6 anchor training", "--purpose", "Preview launcher routing only.",
            "--", *extra, "slurm/" + job]
    assert submit.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    run_dir = Path(plan["run_dir"])
    assert plan["output_environment"]["SOFTROLE_ANCHOR_MANIFEST"] == str(run_dir / "manifest.json")
    assert plan["output_environment"]["SOFTROLE_ANCHOR_RUN_ROOT"] == str(run_dir / "training")
    assert "SOFTROLE_MECHANISM_SUPP_MANIFEST" not in plan["output_environment"]
    assert plan["command"][-1] == str(checkout / "slurm" / job)
    assert not (checkout / "runs").exists()


def test_both_scripts_are_registered_like_the_supplement_pair():
    for name in ("softrole_anchor_train.sbatch", "softrole_anchor_prepare.sbatch"):
        assert set(submit.ENV_ROUTES[name]) == {"SOFTROLE_ANCHOR_RUN_ROOT", "SOFTROLE_ANCHOR_MANIFEST"}
    assert "softrole_anchor_train.sbatch" in submit.RUN_ROOT_OUTPUT_JOBS
    assert "softrole_anchor_prepare.sbatch" not in submit.RUN_ROOT_OUTPUT_JOBS  # as for the supplement pair
    assert set(submit.ENV_ROUTES["softrole_mechanism_supp_train.sbatch"]) == {
        "SOFTROLE_MECHANISM_SUPP_RUN_ROOT", "SOFTROLE_MECHANISM_SUPP_MANIFEST"}


def test_submit_stokes_train_cannot_route_output_outside_the_run(checkout):
    args = ["--title", "PCP 6x6 anchor training", "--purpose", "Containment check only.",
            "--", "slurm/softrole_anchor_train.sbatch"]
    assert submit.main(args + ["--run-root", "/tmp/outside"]) == 2
    assert submit.main(args + ["--run-ro=/tmp/outside"]) == 2
    assert not (checkout / "runs").exists()


@pytest.mark.parametrize("variable", ["SOFTROLE_ANCHOR_MANIFEST", "SOFTROLE_ANCHOR_RUN_ROOT"])
@pytest.mark.parametrize("job", ["softrole_anchor_prepare.sbatch", "softrole_anchor_train.sbatch"])
def test_submit_stokes_rejects_environment_routes_outside_the_run(checkout, monkeypatch, variable, job):
    monkeypatch.setenv(variable, "/tmp/outside")
    args = ["--title", "PCP 6x6 anchor training", "--purpose", "Containment check only.", "--", "slurm/" + job]
    assert submit.main(args) == 2
    assert not (checkout / "runs").exists()


# --- Slurm scripts with mocked srun/module/python ---------------------------

def capture_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"names = {(*THREADS, 'PYTHONPATH', 'PYTHONUNBUFFERED', 'DGLBACKEND', 'PYTHONNOUSERSITE')!r}\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['PYTHON_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': args, 'cwd': os.getcwd(), 'python': sys.argv[0], "
        "'environment': {name: os.environ.get(name) for name in names}}) + '\\n')\n"
        "assert args[:3] == ['-u', '-m', 'softrole.anchor_study']\n"
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
    for name in ("SOFTROLE_ANCHOR_MANIFEST", "SOFTROLE_ANCHOR_RUN_ROOT", "SOFTROLE_MECHANISM_SUPP_MANIFEST",
                 "SOFTROLE_MECHANISM_SUPP_RUN_ROOT", "SOFTROLE_MECHANISM_MANIFEST", "SOFTROLE_MECHANISM_RUN_ROOT",
                 "HETNET_PYTHON", "MOCK_FAIL"):
        env.pop(name, None)
    env.update({name: "32" for name in THREADS})
    env.update(PATH=f"{tools}:{env['PATH']}", PYTHONPATH="inherited-path", PYTHONUNBUFFERED="0", DGLBACKEND="other",
               PYTHONNOUSERSITE="0", MODULE_CAPTURE=str(tmp_path / "module.txt"),
               PYTHON_CAPTURE=str(tmp_path / "python.jsonl"), SLURM_SUBMIT_DIR=str(root),
               SLURM_ARRAY_TASK_ID=index, SLURM_ARRAY_JOB_ID="4242")
    env = configure_stokes_run(env, tmp_path)
    if defaults:
        capture_python(root / ".venv/bin/python")
        manifest, run_root = Path(env["HETNET_RUN_DIR"]) / "manifest.json", Path(env["HETNET_RUN_DIR"]) / "training"
    else:
        env["HETNET_PYTHON"] = str(capture_python(tmp_path / "capture python"))
        container = Path(env["HETNET_RUN_DIR"])
        manifest, run_root = container / "locked manifest.json", container / "training runs"
        env.update(SOFTROLE_ANCHOR_MANIFEST=str(manifest), SOFTROLE_ANCHOR_RUN_ROOT=str(run_root))
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
    assert call["environment"]["PYTHONNOUSERSITE"] == "1"  # as the original-HetNet additional-seed jobs export


def launch_args(manifest, index, run_root, *, dry_run=False):
    result = ["-u", "-m", "softrole.anchor_study", "launch", "--manifest", str(manifest),
              "--index", str(index), "--run-root", str(run_root)]
    return result + (["--dry-run"] if dry_run else [])


@pytest.mark.parametrize("index", [0, 2, 3, 9, 10, 11])
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


@pytest.mark.parametrize("index", ["12", "13", "69", "70", "-1", "00", "01", "012", "x", "9a", "100", "1 2", "11 "])
def test_training_slurm_rejects_invalid_array_indices_before_execution(tmp_path, index):
    _, _, _, env = setup_environment(tmp_path, index=index, existing=True)
    result = run_slurm(SLURM, [], tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


@pytest.mark.parametrize("args", [
    ["--dry"], ["--dry-run=true"], ["--dry-run", "--dry-run"], ["--resume", "checkpoint.pt"],
    ["--total-steps", "1"], ["--epochs", "1"], ["--nprocesses", "1"], ["--max-steps", "4"],
    ["--seed", "1"], ["--model", "cash"], ["--dim", "5"], ["--manifest", "other.json"],
    ["--run-root", "other"], ["--episode-log", "stdout"], ["--max-env-steps", "1"],
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
def test_preparation_creates_manifest_then_previews_all_twelve_entries_without_training(tmp_path, defaults):
    root, manifest, run_root, env = setup_environment(tmp_path, defaults=defaults)
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    expected = [["-u", "-m", "softrole.anchor_study", "prepare", "--output", str(manifest)]]
    expected += [launch_args(manifest, index, run_root, dry_run=True) for index in range(12)]
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
                                                     for i in range(12)]
    assert manifest.read_bytes() == before and not run_root.exists()


@pytest.mark.parametrize("failure, expected_count", [("prepare", 1), ("launch:0", 2), ("launch:11", 13)])
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


@pytest.mark.parametrize("variable", ["SOFTROLE_ANCHOR_MANIFEST", "SOFTROLE_ANCHOR_RUN_ROOT"])
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


def test_other_studies_variables_do_not_redirect_the_anchor(tmp_path):
    _, manifest, run_root, env = setup_environment(tmp_path, existing=True)
    env.update(SOFTROLE_MECHANISM_MANIFEST=str(tmp_path / "parent manifest.json"),
               SOFTROLE_MECHANISM_RUN_ROOT=str(tmp_path / "parent training"),
               SOFTROLE_MECHANISM_SUPP_MANIFEST=str(tmp_path / "supplement manifest.json"),
               SOFTROLE_MECHANISM_SUPP_RUN_ROOT=str(tmp_path / "supplement training"))
    result = run_slurm(SLURM, ["--dry-run"], tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert [call["args"] for call in calls(env)] == [launch_args(manifest, 0, run_root, dry_run=True)]


def test_training_slurm_sources_the_containment_helper_before_any_python():
    text = SLURM.read_text()
    assert 'stokes_run_output "$SOFTROLE_ANCHOR_MANIFEST" "$run_root"' in text
    assert text.index("stokes_run_output") < text.index("module load") < text.index("srun")


SUPPLEMENT_TO_ANCHOR = (
    ("softrole_mechanism_supp_train", "softrole_anchor_train"),
    ("softrole_mechanism_supp_prepare", "softrole_anchor_prepare"),
    ("SOFTROLE_MECHANISM_SUPP_MANIFEST", "SOFTROLE_ANCHOR_MANIFEST"),
    ("SOFTROLE_MECHANISM_SUPP_RUN_ROOT", "SOFTROLE_ANCHOR_RUN_ROOT"),
    ("softrole.mechanism_supplement", "softrole.anchor_study"),
    ("{0..9}", "{0..11}"), ("--array=0-9%10", "--array=0-11%12"), ("--time=12:00:00", "--time=48:00:00"),
    ("^[0-9]$", "^([0-9]|1[01])$"), ("0 through 9", "0 through 11"),
    ("export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch\n", "export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch PYTHONNOUSERSITE=1\n"),
)


def executable_lines(text):
    """Every line except descriptive comments and the job name (the helper overrides it at submission)."""
    return [line for line in text.splitlines()
            if not line.startswith("#") or (line.startswith("#SBATCH") and "--job-name" not in line)]


@pytest.mark.parametrize("supplement_script,script", [("softrole_mechanism_supp_prepare.sbatch", PREPARE),
                                                      ("softrole_mechanism_supp_train.sbatch", SLURM)])
def test_each_script_is_its_supplement_counterpart_with_names_range_time_and_the_hetnet_export_swapped(
        supplement_script, script):
    text = (ROOT / "slurm" / supplement_script).read_text()
    for old, new in SUPPLEMENT_TO_ANCHOR:
        text = text.replace(old, new)
    assert executable_lines(text) == executable_lines(script.read_text())


def test_scripts_export_what_the_original_hetnet_additional_seed_script_exports_and_the_supplement_scripts_do_not():
    reference = (ROOT / "slurm/softrole_pcp_additional_seeds.sbatch").read_text()
    exports = {name for line in reference.splitlines() if line.startswith("export ")
               for name in (token.split("=")[0] for token in line.split()[1:] if "=" in token)}
    assert {"DGLBACKEND", "PYTHONNOUSERSITE"} <= exports
    for path in (SLURM, PREPARE):
        text = path.read_text()
        assert "DGLBACKEND=pytorch" in text and "PYTHONNOUSERSITE=1" in text


def test_slurm_shell_syntax_and_declared_resources():
    result = subprocess.run(["bash", "-n", str(SLURM), str(PREPARE)], capture_output=True)
    assert result.returncode == 0, result.stderr
    for path, extra in ((SLURM, ("--cpus-per-task=4", "--mem=16G", "--time=48:00:00", "--array=0-11%12",
                                 "--job-name=pcp6-anchor")),
                        (PREPARE, ("--cpus-per-task=1", "--mem=2G", "--time=00:10:00"))):
        text = path.read_text()
        for setting in ("--account=cenyioha", "--partition=normal", "--constraint=sapphirerapids", *extra):
            assert f"#SBATCH {setting}\n" in text
        assert "softrole.anchor_study" in text and "SOFTROLE_ANCHOR_" in text
        assert "mechanism_study" not in text and "mechanism_supplement" not in text
        assert "graded_pilot" not in text.lower() and "GRADED_PILOT" not in text


# --- Runbook ------------------------------------------------------------------

def test_runbook_exists_is_listed_in_gitignore_and_states_the_runs_differences_declaration_and_commands():
    text = RUNBOOK.read_text()
    flat = " ".join(text.split())
    assert "!/src/softrole/ANCHOR_STUDY.md" in (ROOT / ".gitignore").read_text()
    assert DECLARED in flat and "2026-11-10" in text
    for needle in ("softrole_anchor_prepare.sbatch", "softrole_anchor_train.sbatch", "afterok", "pcp_6x6_anchor_v1",
                   "capability_mechanism_study_v2", "--array=0-11%12", "SOFTROLE_ANCHOR_MANIFEST",
                   "SOFTROLE_ANCHOR_RUN_ROOT", "--init-only", "index = arm_i * 3 + seed_i", "hetnet_original",
                   "capcon_class_input", "class_typed", "`compositions`", "`((2, 1),)`", "`(3,)`", "`()`",
                   "((1, 1), (1, 2), (2, 1), (2, 2), (3, 1))", "`sensing_p`", "`held_out_sensing_p`",
                   "train_arguments(3", "--dim", "--max-env-steps", "--milestones", "40000000", "10000000",
                   "launch.json", "launcher_exit.json", "DGLBACKEND=pytorch", "PYTHONNOUSERSITE=1",
                   "48 hours", "PCP 6x6 anchor training"):
        assert needle in text or needle in flat, needle
    assert ("use the original HetNet system's own learner, observation and critics, not the study's, so they are "
            "not a controlled comparison with the shared-body rows") in flat
    assert "hypothes" not in text.lower() and "outperform" not in text.lower()
