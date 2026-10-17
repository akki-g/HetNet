"""The 40M learner profile has a separate, complete nominal selection contract."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

import pytest
import torch

from softrole import CHECKPOINT_VERSION, EVALUATION_VERSION
from softrole.config import Config
from softrole.programme import (CAPCON_ARMS, CAPCON_PROFILE, MANIFEST_VERSION, RUNTIME_KEYS, S1_RULE,
                                SEEDS, arm_table, entries, launch, prepare, profile_definition, read_json,
                                programme_config, sha, write_json)
from softrole.programme_evaluation import (canonical_digest, composition_seed, nominal_selection,
                                           prepare_evaluation, protocol, rank_nominal_completion, run_task,
                                           select_checkpoint, selected_entries, summarize_evaluation)
from softrole.model import SoftRoleNet
from softrole.scenarios import make_scenarios
from test_softrole_programme import synthetic_selection


def score_rows(value=40):
    return [{"arm": arm, "seed": seed, "episodes": 1, "capped_completion_sum": value}
            for arm in CAPCON_ARMS for seed in SEEDS]


def test_exact_ranking_uses_seed_means_and_declared_ties_without_float_tolerance():
    rows = score_rows()
    tied = rank_nominal_completion(rows)
    assert [row["arm"] for row in tied] == list(CAPCON_ARMS)
    assert all(row["tied_best"] and row["mean_exact"] == [40, 1] for row in tied)
    rows[3].update(episodes=10**14, capped_completion_sum=40 * 10**14 - 1)
    close = rank_nominal_completion(rows)
    assert close[0]["arm"] == CAPCON_ARMS[1]
    assert Fraction(*close[0]["mean_exact"]) == 40 - Fraction(1, 3 * 10**14)
    # Pooling these episodes would prefer the control; equal seed weighting prefers beta1.
    rows = score_rows(60)
    rows[0].update(episodes=100, capped_completion_sum=100)
    rows[1]["capped_completion_sum"] = rows[2]["capped_completion_sum"] = 80
    for row in rows[3:6]:
        row["capped_completion_sum"] = 50
    ranked = rank_nominal_completion(rows)
    assert ranked[0]["arm"] == CAPCON_ARMS[1]
    assert ranked[1]["mean_exact"] == [161, 3]
    with pytest.raises(ValueError, match="every seed"):
        rank_nominal_completion(rows[:-1])
    with pytest.raises(ValueError, match="unique"):
        rank_nominal_completion(rows + [rows[0]])


def test_capcon_phase_is_nominal_only_and_requires_all_seeds():
    manifest = {"programme": CAPCON_PROFILE, "entries": entries(1e-3, profile=CAPCON_PROFILE)}
    chosen, stages = selected_entries(manifest, "nominal-selection", None, None, False)
    assert len(chosen) == 12 and stages == ["C1", "L1"]
    for phase, selected in (("development", None), ("final", None), ("nominal-selection", ["L1"])):
        with pytest.raises(ValueError, match="nominal-selection only"):
            selected_entries(manifest, phase, selected, None, False)
    manifest["entries"].pop()
    with pytest.raises(ValueError, match="every seed"):
        selected_entries(manifest, "nominal-selection", None, None, False)


def test_40m_selector_uses_accepted_continuation_records_not_orphan_files(tmp_path, monkeypatch):
    # Ledger/continuation auditing has real execution coverage in the recovery tests.
    # This isolates the scientific 40M selector from expensive 40M policy execution.
    config = programme_config(CAPCON_ARMS[0], 0, 1e-3, profile=CAPCON_PROFILE)
    model = SoftRoleNet(**config.model_kwargs()).double()
    records = []
    for index, steps in enumerate((39_900_000, 40_000_007, 40_100_000)):
        checkpoint = tmp_path / f"attempt{index}" / "epoch1900.pt"
        checkpoint.parent.mkdir()
        torch.save({"config": config.to_dict(), "model_state": model.state_dict(),
                    "signature": {"sha256": "synthetic-model"}}, checkpoint)
        records.append({"updates": 18900 + index * 100, "epoch": 1890 + index * 10,
                        "total_steps": steps, "total_episodes": 900000 + index,
                        "checkpoint": str(checkpoint), "sha256": sha(checkpoint), "source_sha256": "synthetic-source"})
    (tmp_path / "orphan_at_40m.pt").write_bytes(b"not an accepted checkpoint")
    called = []
    def accepted(run):
        called.append(run)
        return records
    monkeypatch.setattr("softrole.programme.accepted_checkpoints", accepted)
    selected = select_checkpoint(tmp_path, config, 40_000_000)
    assert called == [tmp_path]
    assert selected["checkpoint"] == records[1]["checkpoint"] and selected["target_steps"] == 40_000_000
    assert selected["preceding_total_steps"] == 39_900_000
    assert selected["preceding_checkpoint_sha256"] == records[0]["sha256"]


@pytest.fixture(scope="module")
def bounded_capcon(tmp_path_factory):
    root = tmp_path_factory.mktemp("capcon_nominal")
    _, selection = synthetic_selection(root / "s1", completions=((6, 6, 6), (5, 5, 5)))
    manifest = root / "manifest.json"
    prepare(selection, manifest, bounded=True, diagnostics_every=None, profile=CAPCON_PROFILE)
    overrides = dict(epochs=1, updates_per_epoch=1, batch_steps=1, nprocesses=1, total_steps=1, save_every=1)
    for index in range(12):
        launch(manifest, index, root / "runs", overrides=overrides)
    evaluation = prepare_evaluation(manifest, root / "runs", root / "evaluation", bounded=True,
                                    target_steps=1, episodes=1)
    for index in range(12):
        run_task(root / "evaluation/tasks.json", index)
    return root, manifest, evaluation


def test_bounded_profile_end_to_end_has_only_common_nominal_jobs_and_no_scientific_winner(bounded_capcon, tmp_path):
    root, _, evaluation = bounded_capcon
    assert evaluation["phase"] == "nominal-selection" and evaluation["compositions"] == [[2, 1]]
    assert evaluation["tasks"] == evaluation["jobs"] == 12
    assert evaluation["development_windows"] == [] and evaluation["final_compositions"] == []
    assert sorted(p.name for p in (root / "evaluation/panels").iterdir()) == [
        "nominal-selection_2_1_nominal.json", "nominal-selection_2_1_nominal.protocol.json"]
    tasks = read_json(root / "evaluation/tasks.json")
    jobs = [task["jobs"][0] for task in tasks["tasks"]]
    assert len({job["scenarios_sha256"] for job in jobs}) == 1
    assert all(job["condition"] == "nominal" and not job["trace"] and not job["sham"] for job in jobs)
    assert run_task(root / "evaluation/tasks.json", 0)["skipped_existing"] == 1
    summary = summarize_evaluation(root / "evaluation", tmp_path / "summary", bootstrap_samples=10)
    selection = read_json(tmp_path / "summary/selection.json")
    assert summary["configuration_selection"]["selected_arm"] is None
    assert selection["status"] == "engineering_check" and selection["complete"]
    assert len(selection["ranking"]) == 4 and len(selection["evidence"]) == 12
    assert selection["evaluation_manifest_sha256"] == sha(root / "evaluation/evaluation_manifest.json")
    assert selection["tasks_file_sha256"] == sha(root / "evaluation/tasks.json")
    assert selection["evaluator_source_sha256"] == evaluation["evaluator_source_sha256"]
    assert selection["source_sha256"] == read_json(root / "manifest.json")["source_sha256"]
    assert set(summary["compositions"]) == {"2_1"}


def test_research_preparation_rejects_bounded_and_changed_protocol(bounded_capcon, tmp_path):
    root, manifest, _ = bounded_capcon
    with pytest.raises(ValueError, match="bounded manifest"):
        prepare_evaluation(manifest, root / "runs", tmp_path / "research")
    for kwargs in ({"target_steps": 20_000_000}, {"panel_seed": 99}, {"episodes": 2}, {"entry_indices": [0]}):
        with pytest.raises(ValueError, match="locked panel seed"):
            prepare_evaluation(manifest, root / "runs", tmp_path / "research", **kwargs)
    assert not (tmp_path / "research").exists()


@pytest.fixture
def synthetic_research_evaluation(tmp_path, monkeypatch):
    """Synthetic outcomes test selection validation; these are not executed research runs."""
    # Certification itself is tested in programme tests; this fixture replaces only
    # that boundary, preserving all profile, panel, report and ranking validation.
    monkeypatch.setattr("softrole.programme.validate_selection", lambda selection: selection)
    definition = profile_definition(CAPCON_PROFILE)
    manifest = {"version": MANIFEST_VERSION, "programme": CAPCON_PROFILE, "bounded_check": False,
                "lr": 1e-3, "lr_selection": {"selected_lr": 1e-3, "rule": S1_RULE, "bounded_check": False},
                "source_sha256": "synthetic-source", "common": definition["common"],
                "presets": json.loads(json.dumps(definition["presets"])),
                "arms": arm_table(CAPCON_PROFILE)["arms"], "entries": entries(1e-3, profile=CAPCON_PROFILE),
                "runtime_override_keys": list(RUNTIME_KEYS),
                "checkpoint_policy": definition["checkpoint_policy"], "evaluation_policy": definition["evaluation_policy"]}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    panel_seed = composition_seed(2704, (2, 1))
    scenarios = [asdict(s) for s in make_scenarios(panel_seed, 500, ((2, 1),))]
    panel_path, protocol_path = tmp_path / "panel.json", tmp_path / "protocol.json"
    write_json(panel_path, scenarios)
    declared = protocol(40_000_000, (2, 1), "nominal", panel_seed, 500, sha(panel_path), "nominal-selection")
    write_json(protocol_path, declared)
    evaluation = {"programme": CAPCON_PROFILE, "phase": "nominal-selection", "bounded_check": False,
                  "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                  "target_steps": 40_000_000, "panel_seed": 2704, "episodes_per_composition": 500,
                  "evaluation_policy": definition["evaluation_policy"], "checkpoint_policy": definition["checkpoint_policy"],
                  "evaluator_source_sha256": "synthetic-evaluator"}
    tasks = {"tasks": []}
    for entry in manifest["entries"]:
        cfg = entry["config"]
        job = {"arm": entry["arm"], "seed": entry["seed"], "composition": [2, 1], "condition": "nominal",
               "panel_kind": "nominal", "intervention": "none", "intervention_step": None, "sham": False, "trace": False,
               "source_sha256": manifest["source_sha256"], "config": cfg, "scenarios": str(panel_path),
               "scenarios_sha256": sha(panel_path), "protocol": str(protocol_path), "protocol_sha256": sha(protocol_path),
               "panel_sha256": canonical_digest(sorted(scenarios, key=lambda s: str(s["scenario_id"]))),
               "checkpoint": f"synthetic-checkpoint-{entry['index']}", "checkpoint_sha256": f"checkpoint-{entry['index']}",
               "model_signature": f"model-{entry['index']}",
               "checkpoint_progress": {"epoch": 1900, "updates": 19000, "total_steps": 40_000_007, "total_episodes": 900000},
               "output": str(tmp_path / f"report{entry['index']}.json")}
        report = {key: job[key] for key in ("checkpoint_sha256", "checkpoint_progress", "model_signature", "config",
                                          "source_sha256", "intervention", "sham", "scenarios_sha256")}
        report.update(evaluation_version=EVALUATION_VERSION, format_version=CHECKPOINT_VERSION,
                      training_seed=entry["seed"], environment_version=cfg["env_version"], evaluation_protocol=declared,
                      evaluator={"source": {"sha256": evaluation["evaluator_source_sha256"]}},
                      model_config=json.loads(json.dumps(Config(**cfg).model_kwargs())), scenarios=scenarios, episodes=500,
                      per_episode=[dict(s, steps=20, success=True, team_return=-3., composition=[2, 1],
                                        intervention="none", intervention_step=-1, sham=False) for s in scenarios])
        write_json(job["output"], report)
        tasks["tasks"].append({"arm": entry["arm"], "seed": entry["seed"], "jobs": [job]})
    evaluation["tasks_sha256"] = canonical_digest(tasks["tasks"])
    return evaluation, tasks


def test_complete_scientific_selection_caps_failures_and_records_exact_ties(synthetic_research_evaluation):
    evaluation, tasks = synthetic_research_evaluation
    selection = nominal_selection(evaluation, tasks)
    assert selection["selected_arm"] == CAPCON_ARMS[0] and selection["status"] == "selected"
    assert all(row["mean_exact"] == [20, 1] and row["tied_best"] for row in selection["ranking"])
    job = tasks["tasks"][0]["jobs"][0]
    report = read_json(job["output"])
    report["per_episode"][0].update(success=False, steps=1, team_return=1e6)
    Path(job["output"]).write_text(json.dumps(report))
    changed = nominal_selection(evaluation, tasks)
    assert changed["selected_arm"] == CAPCON_ARMS[1]
    control = next(row for row in changed["ranking"] if row["arm"] == CAPCON_ARMS[0])
    assert Fraction(*control["mean_exact"]) == 20 + Fraction(60, 1500)
    assert len(changed["evidence"]) == 12 and changed["manifest_sha256"] == evaluation["manifest_sha256"]


def test_research_preparation_resolves_40m_and_common_500_scenario_nominal_panel(
        synthetic_research_evaluation, tmp_path, monkeypatch):
    evaluation, _ = synthetic_research_evaluation
    manifest = read_json(evaluation["manifest"])
    root = tmp_path / "research-shaped-runs"
    for entry in manifest["entries"]:
        run = root / entry["relative_output"]
        write_json(run / "config.json", entry["config"])
        write_json(run / "source_manifest.json", {"sha256": manifest["source_sha256"]})
        write_json(run / "run.json", {"source_sha256": manifest["source_sha256"]})
        write_json(run.parent / (run.name + ".programme.json"),
                   {"manifest_sha256": evaluation["manifest_sha256"], "index": entry["index"], "bounded_check": False})
    # Only checkpoint/ledger lookup is substituted; profile and preparation validation run normally.
    monkeypatch.setattr("softrole.programme.accepted_checkpoints", lambda run, return_audit: ([], {}))
    calls = []
    def select(run, config, target_steps, accepted):
        calls.append((config.total_steps, target_steps))
        return {"checkpoint": str(run / "synthetic.pt"), "sha256": "synthetic-checkpoint",
                "source_sha256": manifest["source_sha256"], "evaluation_model_signature": "synthetic-model",
                "epoch": 1900, "updates": 19000, "total_steps": 40_000_007, "total_episodes": 900000}
    monkeypatch.setattr("softrole.programme_evaluation.select_checkpoint", select)
    monkeypatch.setattr("softrole.programme_evaluation.window_statistics", lambda *args, **kwargs: ({}, 40_000_007))
    result = prepare_evaluation(evaluation["manifest"], root, tmp_path / "prepared")
    assert calls == [(40_000_000, 40_000_000)] * 12
    assert result["bounded_check"] is False and result["phase"] == "nominal-selection"
    assert result["tasks"] == result["jobs"] == 12 and result["episodes_per_composition"] == 500
    tasks = read_json(tmp_path / "prepared/tasks.json")
    assert len({task["jobs"][0]["scenarios_sha256"] for task in tasks["tasks"]}) == 1
    assert all(len(read_json(task["jobs"][0]["scenarios"])) == 500 for task in tasks["tasks"])


def test_missing_engineering_or_malformed_evidence_cannot_select(synthetic_research_evaluation):
    evaluation, tasks = synthetic_research_evaluation
    bounded = deepcopy(evaluation)
    bounded["bounded_check"] = True
    assert nominal_selection(bounded, tasks)["selected_arm"] is None
    job = tasks["tasks"][0]["jobs"][0]
    report_path = Path(job["output"])
    original = report_path.read_text()
    report_path.unlink()
    with pytest.raises(ValueError, match="missing"):
        nominal_selection(evaluation, tasks)
    missing = nominal_selection(evaluation, tasks, allow_missing=True)
    assert missing["status"] == "incomplete" and missing["selected_arm"] is None and not missing["ranking"]
    report_path.write_text(original)
    report = json.loads(original)
    report["checkpoint_progress"]["total_steps"] -= 1
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="checkpoint_progress"):
        nominal_selection(evaluation, tasks)
    report_path.write_text(original)
    duplicate = deepcopy(tasks)
    duplicate["tasks"][-1] = duplicate["tasks"][0]
    with pytest.raises(ValueError, match="unique"):
        nominal_selection(evaluation, duplicate)
    bad = deepcopy(tasks)
    bad["tasks"][0]["jobs"][0]["sham"] = True
    with pytest.raises(ValueError, match="locked condition"):
        nominal_selection(evaluation, bad)
