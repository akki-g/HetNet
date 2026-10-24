"""Locked two-arm graded pilot preparation and dispatch; no policy training."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from softrole import graded_pilot as pilot
from softrole import programme
from softrole.config import Config


@pytest.mark.parametrize("index,descriptor", [(0, "capability"), (1, "type")])
def test_pilot_configuration_changes_only_declared_sensing_descriptor_and_budget(index, descriptor):
    baseline = programme.programme_config("l1b_decomposed_beta1", 0, 1e-3,
                                          profile=programme.CAPCON_PROFILE)
    expected = dict(asdict(baseline), capabilities="graded", descriptor=descriptor,
                    sensing_p=(1, 3), sensing_a=(0,), held_out_sensing_p=(2,),
                    held_out_sensing_a=(1, 2, 3), total_steps=2_000_000)
    actual = pilot.pilot_config(index)
    assert isinstance(actual, Config)
    assert asdict(actual) == expected
    assert (actual.max_steps, actual.nprocesses, actual.batch_steps) == (80, 4, 500)
    assert (actual.epochs, actual.updates_per_epoch, actual.save_every) == (2000, 10, 10)
    assert actual.model_kwargs() == baseline.model_kwargs()
    assert (actual.optimizer, actual.lr, actual.critic_mode, actual.credit_beta) == (
        "adam", 1e-3, "decomposed", 1.)
    assert actual.failure_prob == 0 and actual.compositions == actual.held_out == ()


def test_pilot_does_not_change_historical_programme_profiles():
    before = {profile: deepcopy(programme.profile_definition(profile)) for profile in programme.PROGRAMMES}
    pilot.pilot_config(0)
    pilot.pilot_config(1)
    pilot.expected_entries()
    assert programme.PROGRAMMES == (programme.PROGRAMME, programme.CAPCON_PROFILE)
    assert programme.SEEDS == (0, 1, 2)
    assert len(programme.entries(1e-3)) == 63
    assert len(programme.entries(1e-3, profile=programme.CAPCON_PROFILE)) == 12
    assert programme.programme_config("l1b_decomposed_beta1", 0, 1e-3).total_steps == 20_000_000
    assert programme.programme_config("l1b_decomposed_beta1", 0, 1e-3,
                                      profile=programme.CAPCON_PROFILE).total_steps == 40_000_000
    assert before == {profile: programme.profile_definition(profile) for profile in programme.PROGRAMMES}


@pytest.mark.parametrize("index", [-1, 2, 3, True, False, 0., 1.5, "0", "1", None])
def test_pilot_configuration_rejects_noninteger_and_out_of_scope_indices(index):
    with pytest.raises(ValueError, match="index"):
        pilot.pilot_config(index)


def test_exact_two_arm_mapping_uses_disjoint_seed_zero_outputs():
    rows = pilot.expected_entries()
    assert [(row["index"], row["arm"], row["seed"], row["relative_output"]) for row in rows] == [
        (0, "g_cap", 0, "g_cap/seed0"), (1, "g_type", 0, "g_type/seed0")]
    for row in rows:
        assert row["config"] == json.loads(json.dumps(pilot.pilot_config(row["index"]).to_dict()))
    assert sum(row["config"]["total_steps"] for row in rows) == 4_000_000


@pytest.fixture
def manifest_path(tmp_path, monkeypatch):
    source = pilot.source_snapshot()
    monkeypatch.setattr(pilot, "source_snapshot", lambda: deepcopy(source))
    path = tmp_path / "manifest.json"
    pilot.prepare(path)
    return path


def resign(manifest):
    """A semantically invalid manifest remains invalid with a correct outer hash."""
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()


def forbid_training(*args, **kwargs):
    pytest.fail("pilot preparation/validation/dry-run must not train")


def test_prepare_records_locked_configs_and_source_without_training_or_other_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "train", forbid_training)
    monkeypatch.setattr("softrole.train.make_env", forbid_training)
    path = tmp_path / "manifest.json"
    manifest = pilot.prepare(path)
    assert json.loads(path.read_text()) == manifest
    assert pilot.validate_manifest(manifest) is manifest
    assert manifest.keys() == {"version", "pilot", "protocol", "entries", "source", "manifest_sha256"}
    assert manifest["version"] == 1 and manifest["pilot"] == "graded_sensing_pilot_2m_v1"
    assert manifest["entries"] == pilot.expected_entries()
    assert manifest["source"]["files"]
    assert manifest["source"]["sha256"] == hashlib.sha256(
        json.dumps(manifest["source"]["files"], sort_keys=True).encode()).hexdigest()
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert list(tmp_path.iterdir()) == [path]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        pilot.prepare(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("corruption", [
    "version", "pilot", "extra_key", "protocol", "missing_entry", "duplicate_entry", "order",
    "arm", "seed", "index", "output", "budget", "descriptor", "training_support", "source_hash", "source_file_hash",
    "boolean_index", "boolean_seed", "float_index", "float_seed", "float_budget", "numeric_protocol_flag",
])
def test_resigned_manifest_tampering_is_rejected_before_any_output(manifest_path, tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(pilot, "train", forbid_training)
    manifest = json.loads(manifest_path.read_text())
    if corruption == "version":
        manifest["version"] = 2
    elif corruption == "pilot":
        manifest["pilot"] = programme.CAPCON_PROFILE
    elif corruption == "extra_key":
        manifest["results"] = {"winner": "g_cap"}
    elif corruption == "protocol":
        manifest["protocol"]["held_out_evaluation_completed"] = True
    elif corruption == "missing_entry":
        manifest["entries"].pop()
    elif corruption == "duplicate_entry":
        manifest["entries"][1] = deepcopy(manifest["entries"][0])
    elif corruption == "order":
        manifest["entries"].reverse()
    elif corruption in ("boolean_index", "boolean_seed", "float_index", "float_seed"):
        key = "index" if corruption.endswith("index") else "seed"
        manifest["entries"][0][key] = False if corruption.startswith("boolean") else 0.
    elif corruption == "float_budget":
        manifest["entries"][0]["config"]["total_steps"] = 2_000_000.
    elif corruption == "numeric_protocol_flag":
        manifest["protocol"]["checkpoint_commits"] = 1
    elif corruption in ("arm", "seed", "index", "output"):
        key, value = {"arm": ("arm", "g_class"), "seed": ("seed", 1), "index": ("index", 0),
                      "output": ("relative_output", "../escape")}[corruption]
        manifest["entries"][1][key] = value
    elif corruption in ("budget", "descriptor", "training_support"):
        key, value = {"budget": ("total_steps", 40_000_000), "descriptor": ("descriptor", "capability"),
                      "training_support": ("sensing_p", [3])}[corruption]
        manifest["entries"][1]["config"][key] = value
    elif corruption == "source_hash":
        manifest["source"]["sha256"] = "0" * 64
    else:
        name = next(iter(manifest["source"]["files"]))
        manifest["source"]["files"][name] = "not-a-hash"
    resign(manifest)
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        pilot.validate_manifest(manifest)
    with pytest.raises(ValueError):
        pilot.launch(corrupted, 0, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


def test_manifest_digest_and_current_checkout_are_both_required(manifest_path, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "train", forbid_training)
    manifest = json.loads(manifest_path.read_text())
    corrupt_digest = deepcopy(manifest)
    corrupt_digest["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        pilot.validate_manifest(corrupt_digest)
    changed_source = deepcopy(manifest["source"])
    changed_source["sha256"] = "f" * 64
    monkeypatch.setattr(pilot, "source_snapshot", lambda: changed_source)
    with pytest.raises(ValueError, match="[Ss]ource|[Cc]heckout"):
        pilot.launch(manifest_path, 0, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", [0, 1])
def test_dry_run_returns_exact_dispatch_without_writing_or_training(manifest_path, tmp_path, monkeypatch, index):
    monkeypatch.setattr(pilot, "train", forbid_training)
    before = manifest_path.read_bytes()
    root = tmp_path / "runs with spaces ' $(unused)"
    result = pilot.launch(manifest_path, index, root, dry_run=True)
    row = pilot.expected_entries()[index]
    manifest = json.loads(before)
    assert result["pilot"] == "graded_sensing_pilot_2m_v1"
    assert (result["index"], result["arm"], result["seed"]) == (index, row["arm"], 0)
    assert result["config"] == row["config"]
    assert result["output"] == str((root / row["relative_output"]).resolve())
    assert result["source_sha256"] == manifest["source"]["sha256"]
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert result["dry_run"] is True
    assert not root.exists()
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("index", [-1, 2, True, 0., "0", None])
def test_launch_rejects_bad_index_without_outputs(manifest_path, tmp_path, monkeypatch, index):
    monkeypatch.setattr(pilot, "train", forbid_training)
    with pytest.raises(ValueError, match="index"):
        pilot.launch(manifest_path, index, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_existing_run_is_refused_and_preserved_even_in_dry_run(manifest_path, tmp_path, monkeypatch, dry_run):
    monkeypatch.setattr(pilot, "train", forbid_training)
    output = tmp_path / "runs/g_cap/seed0"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("existing experiment\n")
    with pytest.raises(FileExistsError):
        pilot.launch(manifest_path, 0, tmp_path / "runs", dry_run=dry_run)
    assert sentinel.read_text() == "existing experiment\n"


@pytest.mark.parametrize("index", [0, 1])
def test_real_dispatch_uses_fixed_training_config_and_committed_compressed_ledgers(manifest_path, tmp_path, monkeypatch, index):
    calls = []

    def record_training(config, output, **kwargs):
        calls.append((config, Path(output), kwargs))
        Path(output).mkdir(parents=True)
        return Path(output) / "checkpoints/epoch0010.pt"

    monkeypatch.setattr(pilot, "train", record_training)
    original = manifest_path.read_bytes()
    result = pilot.launch(manifest_path, index, tmp_path / "runs")
    row = pilot.expected_entries()[index]
    output = (tmp_path / "runs" / row["relative_output"]).resolve()
    assert len(calls) == 1
    config, actual_output, settings = calls[0]
    assert config == pilot.pilot_config(index)
    assert actual_output == output
    assert settings == {"episode_log": "chunked-gzip", "checkpoint_commits": True}
    assert result["dry_run"] is False
    assert result["checkpoint"] == str(output / "checkpoints/epoch0010.pt")
    assert manifest_path.read_bytes() == original


def test_source_change_during_training_prevents_success_record(manifest_path, tmp_path, monkeypatch):
    source = deepcopy(json.loads(manifest_path.read_text())["source"])
    monkeypatch.setattr(pilot, "source_snapshot", lambda: deepcopy(source))
    calls = []

    def changed_during_training(config, output, **kwargs):
        output = Path(output)
        output.mkdir(parents=True)
        calls.append(output)
        source["sha256"] = "f" * 64
        return output / "checkpoint.pt"

    monkeypatch.setattr(pilot, "train", changed_during_training)
    before = manifest_path.read_bytes()
    with pytest.raises(RuntimeError, match="Source changed during"):
        pilot.launch(manifest_path, 0, tmp_path / "runs")
    assert calls == [tmp_path / "runs/g_cap/seed0"]
    assert not (calls[0] / "pilot.json").exists()
    assert manifest_path.read_bytes() == before


def test_cli_prepare_and_both_dry_runs_are_reviewable_without_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pilot, "train", forbid_training)
    manifest = tmp_path / "manifest.json"
    pilot.main(["prepare", "--output", str(manifest)])
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["pilot"] == "graded_sensing_pilot_2m_v1"
    for index in (0, 1):
        pilot.main(["launch", "--manifest", str(manifest), "--index", str(index),
                    "--run-root", str(tmp_path / "runs"), "--dry-run"])
        result = json.loads(capsys.readouterr().out)
        assert result["index"] == index and result["dry_run"] is True
        assert result["config"] == pilot.expected_entries()[index]["config"]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("extra", [
    ["--resume"], ["--profile", programme.CAPCON_PROFILE], ["--seed", "1"], ["--total-steps", "1"],
    ["--max-steps", "3"], ["--nprocesses", "1"], ["--sensing-p", "3"], ["--descriptor", "type"],
    ["--diagnostics-every", "1"], ["--episode-log", "stdout"],
])
def test_cli_has_no_resume_or_scientific_runtime_overrides(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(pilot, "train", forbid_training)
    with pytest.raises(SystemExit) as error:
        pilot.main(["launch", "--manifest", str(tmp_path / "not-read.json"), "--index", "0",
                    "--run-root", str(tmp_path / "runs"), "--dry-run", *extra])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", ["-1", "2", "1.5", "true", ""])
def test_cli_rejects_bad_indices_before_launch(manifest_path, tmp_path, monkeypatch, index):
    monkeypatch.setattr(pilot, "train", forbid_training)
    with pytest.raises(SystemExit) as error:
        pilot.main(["launch", "--manifest", str(manifest_path), "--index", index,
                    "--run-root", str(tmp_path / "runs"), "--dry-run"])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()
