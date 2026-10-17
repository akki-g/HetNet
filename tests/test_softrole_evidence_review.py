"""Independent checks of programme evidence binding and stage panel coverage."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

import pytest

from softrole import CHECKPOINT_VERSION, EVALUATION_VERSION
from softrole.config import Config
from softrole.programme import entries, normalized, sha
from softrole.programme_evaluation import (development_conditions, protocol, selected_entries,
                                          validate_report_job, run_task, select_checkpoint)
from softrole.report import canonical_digest, panel_digest
from softrole.scenarios import make_scenarios


def prepared_report(tmp_path):
    config = Config(model="shared")
    scenarios = [asdict(replace(make_scenarios(35, 1, [(2, 1)])[0], event_step=1, victim=0))]
    panel = tmp_path / "scenarios.json"
    panel.write_text(json.dumps(scenarios))
    declared = protocol(20_000_000, (2, 1), "failure", 35, 1, sha(panel), phase="development")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(declared))
    job = {"protocol": str(protocol_path), "protocol_sha256": sha(protocol_path),
           "scenarios": str(panel), "scenarios_sha256": sha(panel),
           "panel_sha256": panel_digest({s["scenario_id"]: s for s in scenarios}),
           "checkpoint_sha256": "a" * 64, "model_signature": "b" * 64,
           "checkpoint_progress": {"epoch": 100, "updates": 1000, "total_steps": 20_000_100,
                                   "total_episodes": 300_000},
           "config": normalized(config), "source_sha256": "c" * 64, "seed": config.seed,
           "intervention": "none", "intervention_step": None, "sham": False,
           "composition": [2, 1], "trace": True}
    evaluator = {"evaluator_source_sha256": "d" * 64}
    outcome = dict(scenarios[0], intervention="none", intervention_step=1, sham=False,
                   composition=[2, 1], steps=2, success=False, team_return=-1.,
                   trace=[dict(step=t, actions=[0, 0, 0], gate=[[1.]] * 3,
                               alpha_null=[[.25]] * 3, team_reward=-.5) for t in range(2)])
    report = {key: job[key] for key in ("checkpoint_sha256", "checkpoint_progress", "model_signature",
                                       "config", "source_sha256", "intervention", "sham", "scenarios_sha256")}
    report.update(evaluation_version=EVALUATION_VERSION, format_version=CHECKPOINT_VERSION,
                  training_seed=config.seed, environment_version=config.env_version,
                  evaluation_protocol=declared, evaluator={"source": {"sha256": "d" * 64}},
                  model_config=config.model_kwargs(), scenarios=scenarios, episodes=1,
                  per_episode=[outcome])
    return report, job, evaluator


@pytest.mark.parametrize("indices", [[0, 0], [1, 2], [0, 2]])
def test_complete_trace_requires_every_actual_step_once(tmp_path, indices):
    report, job, evaluation = prepared_report(tmp_path)
    validate_report_job(report, job, evaluation)
    damaged = deepcopy(report)
    for row, step in zip(damaged["per_episode"][0]["trace"], indices):
        row["step"] = step
    with pytest.raises(ValueError, match="trace|Trace"):
        validate_report_job(damaged, job, evaluation)


def test_missing_trace_is_rejected_for_a_prepared_failure_job(tmp_path):
    report, job, evaluation = prepared_report(tmp_path)
    del report["per_episode"][0]["trace"]
    with pytest.raises(ValueError, match="trace"):
        validate_report_job(report, job, evaluation)


def test_l1c_selection_includes_the_full_recursive_control_chain():
    selected, stages = selected_entries({"entries": entries(.001)}, "development", ["L1C"], None, False)
    assert stages == ["L1C"]
    assert {row["arm"] for row in selected} == {
        "l1c_composition_mean_agent", "l1c_failure_mean_agent",
        "banks_failure_shared", "a1_composition_original_null"}
    assert len(selected) == 12
    for row in selected:
        names = {condition[0] for condition in development_conditions(row["arm"], (2, 1), stages)}
        assert {"nominal", "comm_off_nominal", "failure", "sham", "failure_comm_off",
                "sham_comm_off", "historical_failure", "historical_sham"} <= names


def test_development_o1_controls_remain_paired_when_selected_with_other_stages():
    manifest = {"entries": entries(.001)}
    selected, stages = selected_entries(manifest, "development", ["O1", "L1C"], None, False)
    assert {"c1_real64", "o1_physical_only"} <= {row["arm"] for row in selected}
    for composition in ((2, 1), (1, 2), (2, 2), (3, 1)):
        left = development_conditions("o1_physical_only", composition, stages)
        right = development_conditions("c1_real64", composition, stages)
        assert left == right
    assert development_conditions("o1_physical_only", (1, 1), stages) == []


def test_different_complete_panels_cannot_pair_even_when_scenario_ids_match(tmp_path):
    from softrole.report import _paired_difference
    row = dict(training_seed=0, episodes=1, scenario_ids=["same-id"], panel_sha256="panel-one",
               checkpoint={"checkpoint_sha256": "checkpoint", "model_signature": "model",
                           "progress": {"epoch": 1, "updates": 1, "total_steps": 20_000_100,
                                        "total_episodes": 1}},
               success_rate=0., team_return=-1., completion_steps_horizon_capped=80., mean_agent_return=-1/3)
    other = dict(row, panel_sha256="same-id-different-random-streams")
    with pytest.raises(ValueError, match="complete panels"):
        _paired_difference([row], [other], "Cross-arm contrast")


@pytest.mark.parametrize("dry_run", [False, True])
def test_changed_protocol_is_rejected_before_creating_any_report(tmp_path, monkeypatch, dry_run):
    from softrole import programme_evaluation
    import softrole.evaluate
    import softrole.train
    _, job, evaluation = prepared_report(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"read-only fixture checkpoint identity")
    job.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
               output=str(tmp_path / "must_not_be_created.json"))
    tasks = [{"task": 0, "arm": "c1_real64", "seed": 0, "jobs": [job]}]
    evaluation.update(version=programme_evaluation.EVALUATION_VERSION, tasks_sha256=canonical_digest(tasks))
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation))
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps({"evaluation": str(evaluation_path), "tasks": tasks}))
    monkeypatch.setattr(softrole.train, "source_snapshot", lambda: {"sha256": "d" * 64})
    calls = []
    monkeypatch.setattr(softrole.evaluate, "evaluate_checkpoint", lambda *a, **kw: calls.append(a))
    with (tmp_path / "protocol.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="[Pp]rotocol"):
        run_task(tasks_path, 0, dry_run=dry_run)
    assert not calls and not (tmp_path / "must_not_be_created.json").exists()


def test_checkpoint_selection_ignores_abandoned_saved_ancestor_tail(tmp_path, monkeypatch):
    import torch
    import softrole.train as trainer
    from softrole.programme import accepted_checkpoints

    def source_snapshot(output=None):
        identity = {"sha256": "c" * 64, "files": {}, "git_commit": "fixture"}
        if output is not None:
            (output / "source_manifest.json").write_text(json.dumps(identity))
        return identity

    # Freeze provenance explicitly so concurrent development edits cannot race
    # this numerical lineage test. Every checkpoint still comes from real training.
    monkeypatch.setattr(trainer, "source_snapshot", source_snapshot)
    cfg = Config(model="shared", pre_dim=4, hidden_dim=4, heads=1, head_dim=2, msg_dim=2,
                 max_steps=2, nprocesses=1, batch_steps=1, epochs=4, updates_per_epoch=1, save_every=1)
    root = tmp_path / "run"
    trainer.train(cfg, root, episode_log="chunked-gzip")
    (tmp_path / "run.programme.json").write_text(json.dumps(
        {"manifest_sha256": "e" * 64, "source_sha256": "c" * 64}))
    parent = root / "checkpoints" / "epoch0002.pt"
    payload = torch.load(parent, weights_only=False)
    child = root / "attempts" / "0001"
    metadata = {"version": 1, "attempt": 1, "logical_run": str(root.resolve()),
                "manifest_sha256": "e" * 64, "source_sha256": "c" * 64,
                "config_sha256": hashlib.sha256(json.dumps(normalized(cfg), sort_keys=True).encode()).hexdigest(),
                "parent_checkpoint": str(parent.resolve()), "parent_checkpoint_sha256": sha(parent),
                "accepted_updates": payload["updates"], "accepted_total_steps": payload["total_steps"],
                "accepted_total_episodes": payload["total_episodes"]}
    trainer.train(cfg, child, resume=parent, episode_log="chunked-gzip", attempt=metadata)
    (child / "finished.json").unlink()
    accepted = accepted_checkpoints(root)
    assert [row["updates"] for row in accepted] == [1, 2, 3, 4]
    assert all(str(child) in row["checkpoint"] for row in accepted[2:])
    early = select_checkpoint(root, cfg, payload["total_steps"])
    assert early["checkpoint"] == str(parent.resolve())
    later = select_checkpoint(root, cfg, payload["total_steps"] + 1)
    assert later["checkpoint"] == str((child / "checkpoints" / "epoch0003.pt").resolve())
    assert later["preceding_checkpoint_sha256"] == sha(parent)
