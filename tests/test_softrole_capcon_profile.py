"""The 40M critic study is a separate protocol; historical S1 remains 20M."""
import copy
import json
import subprocess

import pytest
import torch

from softrole.config import Config
from softrole.programme import (
    CAPCON_ARMS, CAPCON_PROFILE, PROGRAMME, S1_WINDOW, arm_table, entries, launch, main,
    prepare, programme_config, validate_manifest,
)
from test_softrole_programme import synthetic_selection
from test_softrole_programme_launcher import launch as shell_launch, slurm_env, SLURM


@pytest.fixture(scope="module")
def profile_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("capcon-profile")
    _, selection = synthetic_selection(root, completions=((6, 6, 6), (5, 5, 5)))
    path = root / "manifest.json"
    prepare(selection, path, bounded=True, profile=CAPCON_PROFILE)
    return path


def test_exact_four_arm_mapping_and_scientific_configuration():
    rows = entries(.001, profile=CAPCON_PROFILE)
    assert len(rows) == 12
    assert [(r["arm"], r["seed"]) for r in rows] == [(a, s) for a in CAPCON_ARMS for s in (0, 1, 2)]
    assert [r["index"] for r in rows] == list(range(12))
    assert len({r["relative_output"] for r in rows}) == 12
    assert sum(r["config"]["total_steps"] for r in rows) == 480_000_000
    for row in rows:
        config = Config(**row["config"])
        historical = programme_config(row["arm"], row["seed"], .001).to_dict()
        assert config.to_dict() == dict(historical, total_steps=40_000_000)
        assert (config.model, config.communication, config.msg_dim) == ("shared", "real", 64)
        assert (config.comm_rounds, config.heads, config.independent_heads) == (3, 4, True)
        assert (config.task, config.num_p, config.num_a, config.max_steps) == ("pcp", 2, 1, 80)
        assert (config.optimizer, config.lr, config.nprocesses) == ("adam", .001, 4)
        assert config.compositions == () and config.held_out == () and config.failure_prob == 0
        assert config.epochs * config.updates_per_epoch * config.nprocesses * config.batch_steps >= config.total_steps
    assert [(r["config"]["critic_mode"], r["config"]["credit_beta"]) for r in rows[::3]] == [
        ("team", 1.), ("decomposed", 1.), ("decomposed", .5), ("decomposed", 0.),
    ]
    assert [r["control"] for r in rows[::3]] == [None, "c1_real64", "l1b_decomposed_beta1", "l1b_decomposed_beta1"]


def test_historical_defaults_and_s1_recipe_remain_separate():
    assert arm_table()["programme"] == PROGRAMME
    assert len(entries(.0003)) == 63
    assert programme_config("c1_real64", 0, .0003).total_steps == 20_000_000
    assert arm_table(CAPCON_PROFILE)["runs"] == 12
    assert arm_table(CAPCON_PROFILE)["stages"] == {"C1": 1, "L1": 3}
    for kwargs in ({"lr": .0003}, {"arm_name": "banks_failure_banked"}, {"profile": "unknown"}):
        settings = dict(arm_name="c1_real64", seed=0, lr=.001, profile=CAPCON_PROFILE)
        settings.update(kwargs)
        with pytest.raises(ValueError):
            programme_config(**settings)


def test_new_manifest_locks_checkpoint_evaluation_and_certification(profile_manifest):
    manifest = json.loads(profile_manifest.read_text())
    assert validate_manifest(manifest) is manifest
    assert manifest["programme"] == CAPCON_PROFILE
    assert manifest["lr"] == manifest["lr_selection"]["selected_lr"] == .001
    assert manifest["checkpoint_policy"] == {"rule": "first_saved_at_or_above", "target_steps": 40_000_000}
    policy = manifest["evaluation_policy"]
    assert policy == {
        "phase": "nominal-selection", "compositions": [[2, 1]], "conditions": ["nominal"],
        "panel_seed": 2704, "episodes_per_composition": 500,
        "selection_metric": "completion_steps_horizon_capped", "weighting": "equal_training_seed_means",
        "tie_order": list(CAPCON_ARMS),
    }
    assert manifest["presets"].keys() == {"fixed"}


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "order", "output", "config", "arm", "budget", "panel", "profile", "bounded"])
def test_entire_manifest_mapping_is_validated_before_any_run(profile_manifest, tmp_path, corruption):
    manifest = json.loads(profile_manifest.read_text())
    if corruption == "missing":
        manifest["entries"].pop()
    elif corruption == "duplicate":
        manifest["entries"][-1] = copy.deepcopy(manifest["entries"][-2])
    elif corruption == "order":
        manifest["entries"][-2:] = reversed(manifest["entries"][-2:])
    elif corruption == "output":
        manifest["entries"][-1]["relative_output"] = manifest["entries"][0]["relative_output"]
    elif corruption == "config":
        manifest["entries"][-1]["config"]["credit_beta"] = .5
    elif corruption == "arm":
        manifest["arms"][-1]["control"] = "c1_real64"
    elif corruption == "budget":
        manifest["checkpoint_policy"]["target_steps"] = 20_000_000
    elif corruption == "panel":
        manifest["evaluation_policy"]["compositions"] = [[3, 2]]
    elif corruption == "profile":
        manifest["programme"] = PROGRAMME
    else:
        manifest["bounded_check"] = False
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        launch(bad, 0, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


def test_prepare_refuses_wrong_rate_and_bounded_research_evidence(tmp_path):
    _, selection = synthetic_selection(tmp_path)
    with pytest.raises(ValueError, match="certified Adam 0.001"):
        prepare(selection, tmp_path / "wrong-rate.json", bounded=True, profile=CAPCON_PROFILE)
    with pytest.raises(ValueError, match="bounded"):
        prepare(selection, tmp_path / "research.json", profile=CAPCON_PROFILE)
    assert not (tmp_path / "wrong-rate.json").exists()
    assert not (tmp_path / "research.json").exists()


@pytest.mark.parametrize("corruption", ["stripped", "forged_research", "receipt_hash", "missing_marker", "string_marker", "integer_marker"])
def test_new_manifest_revalidates_embedded_certification_before_launch(profile_manifest, tmp_path, corruption):
    manifest = json.loads(profile_manifest.read_text())
    selection = manifest["lr_selection"]
    if corruption == "stripped":
        manifest["lr_selection"] = {key: selection[key] for key in ("selected_lr", "rule", "bounded_check")}
    elif corruption == "forged_research":
        manifest["bounded_check"] = selection["bounded_check"] = False
        selection["window"] = list(S1_WINDOW)
    elif corruption == "receipt_hash":
        selection["runs"][0]["audit_sha256"] = "0" * 64
    elif corruption == "missing_marker":
        del selection["bounded_check"]
    elif corruption == "string_marker":
        selection["bounded_check"] = "false"
    else:
        selection["bounded_check"] = 1
    bad = tmp_path / "bad-certification.json"
    bad.write_text(json.dumps(manifest))
    # Source/configuration remain exactly as prepared; evidence alone is invalid.
    with pytest.raises(ValueError):
        launch(bad, 0, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


def test_all_twelve_shell_entries_resolve_and_out_of_profile_indices_fail(profile_manifest, tmp_path):
    expected = entries(.001, profile=CAPCON_PROFILE)
    for index in range(12):
        result = shell_launch(tmp_path, profile_manifest, str(index), "--dry-run")
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == expected[index]["config"]
    result = shell_launch(tmp_path, profile_manifest, "12", "--dry-run")
    assert result.returncode == 2 and "outside the manifest" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", [0, 3, 11, 12])
def test_profile_slurm_dispatch_preserves_manifest_boundaries(profile_manifest, tmp_path, index):
    result = subprocess.run(["bash", str(SLURM), "--dry-run"], env=slurm_env(tmp_path, profile_manifest, index),
                            text=True, capture_output=True)
    if index < 12:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == entries(.001, profile=CAPCON_PROFILE)[index]["config"]
    else:
        assert result.returncode == 2 and "outside the manifest" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", [0, 3, 6, 9])
def test_all_four_learners_execute_with_new_profile_provenance(profile_manifest, tmp_path, index):
    overrides = dict(epochs=1, updates_per_epoch=1, batch_steps=1, nprocesses=1, total_steps=1, save_every=1)
    config = launch(profile_manifest, index, tmp_path / "runs", overrides=overrides, diagnostics_every=1)
    manifest = json.loads(profile_manifest.read_text())
    output = tmp_path / "runs" / manifest["entries"][index]["relative_output"]
    record = json.loads(output.with_suffix(".programme.json").read_text())
    checkpoint = torch.load(output / "checkpoints/epoch0001.pt", weights_only=False)
    assert record["programme"] == CAPCON_PROFILE and record["bounded_check"] is True
    assert checkpoint["updates"] == 1 and checkpoint["total_steps"] >= 1
    assert all(torch.isfinite(value).all() for value in checkpoint["model_state"].values())
    assert checkpoint["config"]["critic_mode"] == config.critic_mode
    assert checkpoint["config"]["credit_beta"] == config.credit_beta


def test_profile_cli_is_explicit_and_default_is_legacy(capsys):
    main(["arms", "--profile", CAPCON_PROFILE])
    assert json.loads(capsys.readouterr().out)["runs"] == 12
    main(["arms"])
    assert json.loads(capsys.readouterr().out)["runs"] == 63


def test_prepare_cli_selects_the_requested_profile(profile_manifest, tmp_path, capsys):
    selection = profile_manifest.parent / "lr_selection.json"
    output = tmp_path / "manifest.json"
    main(["prepare", "--profile", CAPCON_PROFILE, "--lr-selection", str(selection),
          "--output", str(output), "--bounded"])
    assert json.loads(capsys.readouterr().out)["entries"] == 12
    assert json.loads(output.read_text())["programme"] == CAPCON_PROFILE


def test_new_profile_refuses_stale_source_before_outputs(profile_manifest, tmp_path, monkeypatch):
    import softrole.train as training
    monkeypatch.setattr(training, "source_snapshot", lambda: {"sha256": "0" * 64})
    with pytest.raises(ValueError, match="Checkout differs"):
        launch(profile_manifest, 0, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()
