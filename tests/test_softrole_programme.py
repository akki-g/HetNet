"""Locked programme: arm table, S1 selection rule, manifests and launch provenance."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from softrole.config import COMPOSITION_TRAIN, FAILURE_TRAIN, Config
from softrole.programme import (ARMS, ARM_INDEX, RUNTIME_KEYS, S1_RATES, S1_RULE, SEEDS, arm_table, entries,
                                launch, main, normalized, prepare, programme_config, select_learning_rate, sha,
                                window_statistics)

ROOT = Path(__file__).resolve().parents[1]


def test_arm_table_counts_controls_and_declared_factors():
    table = arm_table()
    assert len(ARMS) == 21 and table["runs"] == 63 and table["seeds"] == [0, 1, 2]
    assert table["stages"] == {"C1": 6, "A1": 2, "L1": 3, "O1": 1, "BANKS": 7, "L1C": 2}
    assert len({arm["name"] for arm in ARMS}) == 21
    preset_keys = {"compositions", "held_out", "failure_prob", "failure_window"}
    for row in table["arms"]:
        if row["control"] is None:
            assert row["name"] in ("c1_real64", "a1_composition_original_null")
            continue
        control = ARM_INDEX[row["control"]]
        left = programme_config(row["name"], 0, 3e-4).to_dict()
        right = programme_config(control["name"], 0, 3e-4).to_dict()
        differing = {key for key in left if left[key] != right[key]}
        allowed = set(row["changed_from_control"]) | (preset_keys if row["preset_differs_from_control"] else set())
        assert differing <= allowed, row["name"]
        assert set(row["changed_from_control"]) <= differing, row["name"]
    # Shared/composition/team is reused from A1 rather than trained again.
    assert ARM_INDEX["banks_composition_banked"]["control"] == "a1_composition_original_null"
    assert not any(arm["name"] == "banks_composition_shared" for arm in ARMS)
    assert {arm["model"] if False else arm["factors"]["model"] for arm in ARMS if arm["stage"] == "BANKS"} == {"shared", "banked", "capability", "constant"}


def test_every_entry_configuration_validates_and_matches_the_common_protocol():
    rows = entries(1e-3)
    assert [row["index"] for row in rows] == list(range(63))
    assert [row["seed"] for row in rows] == [0, 1, 2] * 21
    assert len({row["relative_output"] for row in rows}) == 63
    for row in rows:
        config = Config(**row["config"])
        arm = ARM_INDEX[row["arm"]]
        assert (config.lr, config.total_steps, config.nprocesses, config.batch_steps) == (1e-3, 20_000_000, 4, 500)
        assert (config.updates_per_epoch, config.save_every, config.epochs, config.optimizer) == (10, 10, 2000, "adam")
        assert (config.comm_rounds, config.independent_heads, config.heads, config.head_dim) == (3, True, 4, 16)
        assert (config.task, config.max_steps, config.vision, config.dim, config.detach_gap) == ("pcp", 80, 2, 5, 5)
        assert (config.gamma, config.gae_lambda, config.actor_coeff, config.value_coeff, config.max_grad_norm) == (1., .95, 50., 1., .75)
        if arm["preset"] == "failure":
            assert config.failure_prob == .5 and config.failure_window == (1, 1) and config.compositions == FAILURE_TRAIN
        elif arm["preset"] == "composition":
            assert config.compositions == COMPOSITION_TRAIN and config.failure_prob == 0
        else:
            assert config.compositions == () and (config.num_p, config.num_a) == (2, 1) and config.failure_prob == 0
        for key, value in arm["factors"].items():
            assert getattr(config, key) == value
        assert row["relative_output"] == f"{arm['stage']}/{arm['name']}/seed{row['seed']}"


def test_c1_real64_control_is_exactly_the_s1_run_family(tmp_path):
    result = subprocess.run(["bash", str(ROOT / "scripts/softrole_lr_pcp.sh"), "0", "--dry-run"], cwd=tmp_path,
                            text=True, capture_output=True,
                            env=dict(os.environ, HETNET_PYTHON=sys.executable, SOFTROLE_LR_PCP_RUN_ROOT=str(tmp_path / "runs")))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == normalized(programme_config("c1_real64", 0, 3e-4))


def synthetic_run(root, lr, seed, completion, last_steps=12_500_000, config_overrides=None,
                  update_steps=500_000, success=True):
    """Ledgers with one episode per update; updates end every ``update_steps`` joint steps."""
    run = Path(root) / f"lr{lr:g}" / f"seed{seed}"
    run.mkdir(parents=True)
    config = programme_config("c1_real64", seed, lr).to_dict()
    config.update(config_overrides or {})
    (run / "config.json").write_text(json.dumps(config))
    with (run / "updates.jsonl").open("w") as updates, (run / "episodes.jsonl").open("w") as episodes:
        for update in range(1, last_steps // update_steps + 1):
            updates.write(json.dumps({"update": update, "total_steps": update * update_steps, "episodes": 1,
                                      "steps": update_steps, "gradient_norm_before_clip": 1.}) + "\n")
            episodes.write(json.dumps({"update": update, "collector": 0, "steps": completion, "success": success,
                                       "team_return": -.15 * completion, "num_agents": 3,
                                       "mean_agent_return": -.05 * completion, "event_exposed": False}) + "\n")
    return run


def synthetic_selection(root, completions=((6, 6, 6), (5, 5, 8))):
    """Six S1-shaped runs; by default the rates tie at 6 and the lower rate wins."""
    runs = [synthetic_run(root / "s1", lr, seed, c) for lr, values in zip(S1_RATES, completions) for seed, c in zip(SEEDS, values)]
    return select_learning_rate(runs, root / "lr_selection.json"), root / "lr_selection.json"


def test_window_statistics_cap_failures_at_the_horizon(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "updates.jsonl").write_text('{"update": 1, "total_steps": 3}\n{"update": 2, "total_steps": 7}\n')
    (run / "episodes.jsonl").write_text(
        '{"update": 1, "steps": 3, "success": true, "team_return": -0.3, "num_agents": 3, "event_exposed": false}\n'
        '{"update": 2, "steps": 4, "success": false, "team_return": -0.6, "num_agents": 3, "event_exposed": true}\n'
        '{"update": 2, "steps": 2, "success": true, "team_return": -0.2, "num_agents": 2, "mean_agent_return": -0.1}\n')
    stats, last = window_statistics(run, [(0, 3), (3, 7), (3, 9)], horizon=80)
    assert last == 7
    assert stats["0-3"]["episodes"] == 1 and stats["0-3"]["capped_completion"] == 3 and stats["0-3"]["complete"]
    row = stats["3-7"]
    assert row["episodes"] == 2 and row["capped_completion"] == 41 and row["success_rate"] == .5
    assert row["mean_agent_return"] == pytest.approx((-.2 - .1) / 2) and row["event_exposed_episodes"] == 1
    assert stats["3-9"]["complete"] is False
    with pytest.raises(ValueError, match="start < end"):
        window_statistics(run, [(3, 3)], 80)


def test_selection_rule_prefers_lower_completion_and_breaks_ties_downward(tmp_path):
    record, path = synthetic_selection(tmp_path / "tie")
    assert record["selected_lr"] == 3e-4 and record["rule"] == S1_RULE and not record["bounded_check"]
    assert record["equal_seed_means"] == {"0.0003": 6., "0.001": 6.}
    assert all(r["window"]["complete"] and r["window"]["episodes"] == 8 for r in record["runs"])
    assert all(set(r["inputs"]) == {str(Path(r["run"]) / n) for n in ("config.json", "updates.jsonl", "episodes.jsonl")} for r in record["runs"])
    faster, _ = synthetic_selection(tmp_path / "faster", completions=((6, 6, 6), (5, 5, 5)))
    assert faster["selected_lr"] == 1e-3
    with pytest.raises(FileExistsError):
        synthetic_selection(tmp_path / "tie")


def test_selection_refuses_partial_windows_missing_seeds_and_foreign_runs(tmp_path):
    runs = [synthetic_run(tmp_path / "a", lr, seed, 6) for lr in S1_RATES for seed in SEEDS]
    partial = synthetic_run(tmp_path / "b", 3e-4, 0, 6, last_steps=11_000_000)
    with pytest.raises(ValueError, match="partial comparisons"):
        select_learning_rate([partial] + runs[1:], tmp_path / "partial.json")
    with pytest.raises(ValueError, match="all three seeds"):
        select_learning_rate(runs[:-1], tmp_path / "missing.json")
    foreign = synthetic_run(tmp_path / "c", 3e-4, 0, 6, config_overrides={"communication": "binary"})
    with pytest.raises(ValueError, match="S1 Real64 protocol"):
        select_learning_rate([foreign] + runs[1:], tmp_path / "foreign.json")
    reduced = synthetic_run(tmp_path / "d", 3e-4, 0, 6, config_overrides={"total_steps": 1})
    with pytest.raises(ValueError, match="S1 Real64 protocol"):
        select_learning_rate([reduced] + runs[1:], tmp_path / "reduced.json")
    with pytest.raises(ValueError, match="fixed"):
        select_learning_rate(runs, tmp_path / "window.json", window=(0, 1_000_000))
    bounded = select_learning_rate([reduced] + runs[1:], tmp_path / "bounded.json", window=(0, 1_000_000), bounded=True)
    assert bounded["bounded_check"] and bounded["window"] == [0, 1_000_000]
    assert not any(p.exists() for p in (tmp_path / "partial.json", tmp_path / "missing.json", tmp_path / "foreign.json"))


def test_prepare_locks_entries_source_and_selection(tmp_path):
    from softrole.train import source_snapshot
    _, selection = synthetic_selection(tmp_path)
    manifest = prepare(selection, tmp_path / "manifest.json")
    assert manifest["lr"] == 3e-4 and manifest["entries"] == entries(3e-4) and not manifest["bounded_check"]
    assert manifest["source_sha256"] == source_snapshot()["sha256"]
    assert manifest["lr_selection_sha256"] == sha(selection) and manifest["lr_selection"]["selected_lr"] == 3e-4
    assert manifest["diagnostics_every"] == 50 and manifest["runtime_override_keys"] == list(RUNTIME_KEYS)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        prepare(selection, tmp_path / "manifest.json")
    tampered = json.loads(selection.read_text())
    tampered["rule"] = "best looking curve"
    (tmp_path / "tampered.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fixed S1 rule"):
        prepare(tmp_path / "tampered.json", tmp_path / "m2.json")
    bounded = json.loads(selection.read_text())
    bounded["bounded_check"] = True
    (tmp_path / "bounded.json").write_text(json.dumps(bounded))
    with pytest.raises(ValueError, match="bounded"):
        prepare(tmp_path / "bounded.json", tmp_path / "m3.json")
    assert prepare(tmp_path / "bounded.json", tmp_path / "m3.json", bounded=True)["bounded_check"] is True
    assert not (tmp_path / "m2.json").exists()


def test_launch_dry_run_resolves_entries_without_output(tmp_path):
    _, selection = synthetic_selection(tmp_path)
    prepare(selection, tmp_path / "manifest.json")
    config = launch(tmp_path / "manifest.json", 5, tmp_path / "runs", dry_run=True)
    assert config == Config(**entries(3e-4)[5]["config"])
    changed = launch(tmp_path / "manifest.json", 5, tmp_path / "runs", dry_run=True,
                     overrides={"epochs": 1, "total_steps": 7, "nprocesses": None})
    assert {k for k in config.to_dict() if config.to_dict()[k] != changed.to_dict()[k]} == {"epochs", "total_steps"}
    for overrides, message in (({"communication": "binary"}, "not a runtime option"), ({"epochs": 0}, "positive"),
                               ({"epochs": True}, "positive")):
        with pytest.raises(ValueError, match=message):
            launch(tmp_path / "manifest.json", 0, tmp_path / "runs", dry_run=True, overrides=overrides)
    with pytest.raises(ValueError, match="outside the manifest"):
        launch(tmp_path / "manifest.json", 63, tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


def test_launch_refuses_a_changed_checkout_before_writing(tmp_path, monkeypatch):
    import softrole.train as training
    _, selection = synthetic_selection(tmp_path)
    prepare(selection, tmp_path / "manifest.json")
    original = training.source_snapshot
    monkeypatch.setattr(training, "source_snapshot", lambda output=None: {**original(output), "sha256": "0" * 64})
    with pytest.raises(ValueError, match="Checkout differs"):
        launch(tmp_path / "manifest.json", 0, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", [0, 30, 49, 62])
def test_launch_executes_bounded_entries_with_provenance(tmp_path, index):
    _, selection = synthetic_selection(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare(selection, manifest_path)
    overrides = {"epochs": 1, "updates_per_epoch": 1, "batch_steps": 1, "nprocesses": 1, "total_steps": 1, "save_every": 1}
    config = launch(manifest_path, index, tmp_path / "runs", overrides=overrides, diagnostics_every=1)
    entry = manifest["entries"][index]
    output = tmp_path / "runs" / entry["relative_output"]
    assert json.loads((output / "config.json").read_text()) == dict(entry["config"], **overrides) == normalized(config)
    record = json.loads((output.parent / (output.name + ".programme.json")).read_text())
    assert record["manifest_sha256"] == sha(manifest_path) and record["arm"] == entry["arm"] and record["index"] == index
    assert record["bounded_check"] is True and record["runtime_overrides"] == overrides and record["lr"] == 3e-4
    run = json.loads((output / "run.json").read_text())
    assert run["source_sha256"] == manifest["source_sha256"] == record["source_sha256"]
    assert run["learner"] == {key: entry["config"][key] for key in run["learner"]}
    checkpoint = torch.load(output / "checkpoints" / "epoch0001.pt", weights_only=False)
    assert checkpoint["updates"] == 1 and normalized(Config(**checkpoint["config"])) == normalized(config)
    assert (output / "diagnostics.jsonl").exists()
    with pytest.raises(FileExistsError):
        launch(manifest_path, index, tmp_path / "runs", overrides=overrides)


def test_cli_prints_tables_and_rejects_scientific_flags(tmp_path, capsys):
    main(["arms"])
    assert json.loads(capsys.readouterr().out)["runs"] == 63
    _, selection = synthetic_selection(tmp_path)
    main(["prepare", "--lr-selection", str(selection), "--output", str(tmp_path / "manifest.json")])
    assert json.loads(capsys.readouterr().out)["entries"] == 63
    main(["launch", "--manifest", str(tmp_path / "manifest.json"), "--index", "12", "--run-root", str(tmp_path / "runs"), "--dry-run"])
    assert json.loads(capsys.readouterr().out) == entries(3e-4)[12]["config"]
    for extra in (["--communication", "binary"], ["--lr", "0.1"], ["--epochs", "0"], ["--seed", "1"]):
        with pytest.raises(SystemExit) as error:
            main(["launch", "--manifest", str(tmp_path / "manifest.json"), "--index", "0", "--run-root", str(tmp_path / "runs"), "--dry-run", *extra])
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["prepare", "--lr-selection", str(selection), "--output", str(tmp_path / "manifest.json")])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()
