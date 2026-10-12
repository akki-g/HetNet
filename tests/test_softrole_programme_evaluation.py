"""Programme evaluation: panels, selection, structural freeze certification, end-to-end bounded flow."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from softrole.config import Config
from softrole.env import make_env
from softrole.model import SoftRoleNet
from softrole.programme import entries, launch, prepare
from softrole.programme_evaluation import (CONDITIONS, FINAL_COMPOSITIONS, main, panels, prepare_evaluation,
                                           protocol, run_task, select_checkpoint, summarize_evaluation)
from softrole.report import validate_evaluation_protocol
from softrole.rollout import run_episode
from softrole.scenarios import Scenario, make_scenarios
from softrole.train import train
from test_softrole_programme import synthetic_selection


def test_panels_are_reproducible_and_failure_panels_share_random_streams():
    first, second = panels(2704, 5), panels(2704, 5)
    assert first == second and set(first) == set(FINAL_COMPOSITIONS)
    seeds = set()
    for composition, panel in first.items():
        seeds.add(panel["seed"])
        assert len(panel["nominal"]) == len(panel["failure"]) == 5
        for nominal, failure in zip(panel["nominal"], panel["failure"]):
            assert (nominal.num_p, nominal.num_a) == composition and nominal.event_step == -1 and nominal.victim == -1
            assert failure.event_step == 1 and 0 <= failure.victim < composition[0]
            assert (failure.env_seed, failure.action_seed, failure.message_seed) == (nominal.env_seed, nominal.action_seed, nominal.message_seed)
    assert len(seeds) == 3
    assert panels(2705, 5) != first
    with pytest.raises(ValueError, match="two sensing agents"):
        panels(2704, 5, compositions=((1, 2),))


@pytest.mark.parametrize("kind", ["nominal", "failure"])
def test_protocol_sidecars_validate_against_their_panels(kind):
    panel = panels(2704, 4)[(3, 2)]
    scenarios = [asdict(s) for s in panel[kind]]
    declared = protocol(20_000_000, (3, 2), kind, panel["seed"], 4, "abc")
    validate_evaluation_protocol(declared, {"total_steps": 20_000_000}, scenarios, "abc")
    with pytest.raises(ValueError, match="Scenario file hash"):
        validate_evaluation_protocol(declared, {"total_steps": 20_000_000}, scenarios, "other")
    with pytest.raises(ValueError, match="step budget"):
        validate_evaluation_protocol(declared, {"total_steps": 19_999_999}, scenarios, "abc")
    other = [asdict(s) for s in panel["failure" if kind == "nominal" else "nominal"]]
    with pytest.raises(ValueError, match="differs from the declared distribution"):
        validate_evaluation_protocol(declared, {"total_steps": 20_000_000}, other, "abc")


def freeze_config(model):
    return Config(model=model, communication="real", comm_rounds=2, pre_dim=8, hidden_dim=8, heads=2, head_dim=4,
                  msg_dim=4, max_steps=6, vision=1, compositions=((2, 1),))


def freeze_traces(model_name):
    config = freeze_config(model_name)
    torch.manual_seed(3)
    model = SoftRoleNet(**config.model_kwargs()).double()
    scenario = replace(make_scenarios(5, 1, ((2, 1),))[0], event_step=1, victim=0)
    results = {}
    for intervention in ("none", "freeze_affected", "freeze_all"):
        episode = run_episode(model, make_env(config), config, scenario, training=False,
                              intervention=intervention, trace=True)
        results[intervention] = episode
    return results


def strip(metrics):
    return {k: v for k, v in metrics.items() if not k.startswith("intervention")}


@pytest.mark.parametrize("model", ["shared", "constant"])
def test_gate_freezes_are_structural_no_ops_for_shared_and_constant_actors(model):
    results = freeze_traces(model)
    for intervention in ("freeze_affected", "freeze_all"):
        assert results[intervention].traces == results["none"].traces
        assert strip(results[intervention].metrics) == strip(results["none"].metrics)
        assert results[intervention].metrics["intervention_exposed"]


def test_capability_gate_freeze_changes_the_victim_gate_after_sensor_loss():
    results = freeze_traces("capability")
    live, frozen = results["none"].traces, results["freeze_affected"].traces
    assert live[0] == frozen[0]
    assert frozen[1]["gate"][0] == frozen[0]["gate"][0] and live[1]["gate"][0] != live[0]["gate"][0]
    assert frozen[1]["gate"][0] != live[1]["gate"][0]
    assert all(step["gate"][1:] == live_step["gate"][1:] for step, live_step in zip(frozen[:2], live[:2]))


def test_select_checkpoint_takes_the_first_saved_at_or_above_and_verifies_the_ledger(tmp_path):
    config = Config(model="shared", optimizer="adam", pre_dim=8, hidden_dim=8, heads=1, head_dim=4, msg_dim=4,
                    max_steps=3, vision=1, epochs=4, updates_per_epoch=1, batch_steps=1, nprocesses=1, save_every=2)
    train(config, tmp_path / "run")
    metrics = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    assert sorted(p.name for p in (tmp_path / "run" / "checkpoints").glob("*.pt")) == ["epoch0002.pt", "epoch0004.pt"]
    late = select_checkpoint(tmp_path / "run", config, metrics[2]["total_steps"])
    assert late["epoch"] == 4 and late["preceding_saved_epoch"] == 2 and late["preceding_total_steps"] == metrics[1]["total_steps"]
    early = select_checkpoint(tmp_path / "run", config, 1)
    assert early["epoch"] == 2 and early["preceding_saved_epoch"] is None and early["total_steps"] == metrics[1]["total_steps"]
    with pytest.raises(ValueError, match="no saved checkpoint reaches"):
        select_checkpoint(tmp_path / "run", config, metrics[3]["total_steps"] + 1)
    ledger = tmp_path / "run" / "metrics.jsonl"
    original = ledger.read_bytes()
    metrics[3]["total_steps"] += 1
    ledger.write_text("".join(json.dumps(row) + "\n" for row in metrics))
    with pytest.raises(ValueError, match="differs from the metrics ledger"):
        select_checkpoint(tmp_path / "run", config, 1 + metrics[1]["total_steps"])
    ledger.write_bytes(original)
    (tmp_path / "run" / "finished.json").rename(tmp_path / "finished.json")
    with pytest.raises(ValueError, match="has not finished"):
        select_checkpoint(tmp_path / "run", config, 1)


BOUNDED = {"epochs": 1, "updates_per_epoch": 1, "batch_steps": 1, "nprocesses": 1, "total_steps": 1, "save_every": 1}
ENTRIES = (0, 1, 24, 25, 18, 36)  # c1_real64 s0/s1, l1b_decomposed_beta1 s0/s1, a1 original-null s0, banked composition s0


@pytest.fixture(scope="module")
def bounded_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("bounded")
    _, selection = synthetic_selection(root)
    manifest = root / "manifest.json"
    prepare(selection, manifest)
    for index in ENTRIES:
        launch(manifest, index, root / "runs", overrides=dict(BOUNDED), diagnostics_every=1)
    return root, manifest


def test_research_preparation_refuses_bounded_runs_and_panel_changes(bounded_runs, tmp_path):
    root, manifest = bounded_runs
    with pytest.raises(ValueError, match="bounded engineering run"):
        prepare_evaluation(manifest, root / "runs", tmp_path / "research")
    with pytest.raises(ValueError, match="locked panel seed"):
        prepare_evaluation(manifest, root / "runs", tmp_path / "research", episodes=3)
    with pytest.raises(ValueError, match="locked panel seed"):
        prepare_evaluation(manifest, root / "runs", tmp_path / "research", entry_indices=[0])
    assert not (tmp_path / "research").exists()


def test_bounded_evaluation_prepares_runs_and_summarizes_end_to_end(bounded_runs, tmp_path):
    root, manifest = bounded_runs
    evaluation = prepare_evaluation(manifest, root / "runs", tmp_path / "eval", panel_seed=2704, episodes=1,
                                    target_steps=1, bounded=True, entry_indices=list(ENTRIES))
    assert evaluation["bounded_check"] and evaluation["tasks"] == 6
    gated = [r for r in evaluation["runs"] if r["gated"]]
    assert [r["arm"] for r in gated] == ["banks_composition_banked"]
    assert evaluation["jobs"] == 5 * 3 * 6 + 1 * 3 * 10
    for run in evaluation["runs"]:
        assert run["selection"]["epoch"] == 1 and run["selection"]["target_steps"] == 1
        assert set(run["development"]) == {"4000000-8000000", "8000000-12000000", "18000000-20000000"}
        assert not any(w["complete"] for w in run["development"].values())
    development = json.loads((tmp_path / "eval" / "development.json").read_text())
    assert not development["per_arm"]["c1_real64"]["8000000-12000000"]["complete_for_all_seeds"]
    panel_dir = tmp_path / "eval" / "panels"
    assert sorted(p.name for p in panel_dir.glob("*.json")) == sorted(
        f"final_{p}_{a}_{kind}{suffix}" for p, a in FINAL_COMPOSITIONS for kind in ("nominal", "failure")
        for suffix in (".json", ".protocol.json"))
    sbatch = (tmp_path / "eval" / "evaluate.sbatch").read_text()
    assert "#SBATCH --array=0-5%6" in sbatch and "softrole.programme_evaluation run" in sbatch
    tasks = tmp_path / "eval" / "tasks.json"
    plan = run_task(tasks, 0, dry_run=True)
    assert plan["executed"] == 0 and plan["skipped_existing"] == 0 and not (tmp_path / "eval" / "reports").exists()
    with pytest.raises(ValueError, match="missing"):
        summarize_evaluation(tmp_path / "eval", tmp_path / "early")
    results = [run_task(tasks, index) for index in range(evaluation["tasks"])]
    assert [r["executed"] for r in results] == [18, 18, 18, 18, 18, 30]
    assert run_task(tasks, 5)["skipped_existing"] == 30
    reports = sorted((tmp_path / "eval" / "reports").rglob("*.json"))
    assert len(reports) == evaluation["jobs"]
    overview = summarize_evaluation(tmp_path / "eval", tmp_path / "summary", bootstrap_samples=50)
    assert overview["missing_reports"] == [] and set(overview["compositions"]) == {"3_2", "4_1", "4_2"}
    composition = overview["compositions"]["3_2"]
    arms = {row["arm"] for row in composition["strata"]}
    assert arms == {"c1_real64", "l1b_decomposed_beta1", "a1_composition_original_null", "banks_composition_banked"}
    interventions = {(row["arm"], row["intervention"]) for row in composition["difference_in_differences"]}
    assert ("c1_real64", "comm_off") in interventions and ("banks_composition_banked", "freeze_all") in interventions
    assert ("c1_real64", "freeze_all") not in interventions
    by_contrast = {}
    for row in composition["contrasts"]["absolute"]:
        by_contrast.setdefault(row["contrast"], set()).add(row["training_seeds"])
    assert by_contrast["l1b_decomposed_beta1_minus_c1_real64"] == {2}
    assert by_contrast["banks_composition_banked_minus_a1_composition_original_null"] == {1}
    assert any(row["contrast"] == "l1b_decomposed_beta1_minus_c1_real64" for row in composition["contrasts"]["difference_in_differences"])
    detail = json.loads((tmp_path / "summary" / "summary_3_2.json").read_text())
    paired = detail["summary"]["paired_groups"]
    assert all(g["contrast"] == "failure_minus_sham" for g in paired) and len(paired) == 4 * 2 + 1 * 2
    assert all(row["sham"] is False or row["intervention"] in ("none", "comm_off", "freeze_affected", "freeze_all")
               for row in composition["strata"])
    with pytest.raises(FileExistsError):
        summarize_evaluation(tmp_path / "eval", tmp_path / "summary")


def test_cli_entry_points(bounded_runs, tmp_path, capsys):
    root, manifest = bounded_runs
    main(["prepare", "--manifest", str(manifest), "--run-root", str(root / "runs"), "--output", str(tmp_path / "eval"),
          "--episodes", "1", "--target-steps", "1", "--bounded", "--entries", "0"])
    assert json.loads(capsys.readouterr().out)["tasks"] == 1
    main(["run", "--tasks", str(tmp_path / "eval" / "tasks.json"), "--index", "0", "--dry-run"])
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    with pytest.raises(SystemExit) as error:
        main(["run", "--tasks", str(tmp_path / "eval" / "tasks.json"), "--index", "1", "--dry-run"])
    assert error.value.code == 2
