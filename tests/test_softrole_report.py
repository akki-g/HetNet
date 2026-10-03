"""Inference uses independently trained seeds, not correlated rollout counts."""

import json

import pytest

from softrole.report import summarize_reports


def report_file(tmp_path, seed, episodes, name=None, **overrides):
    report = {"config": {"seed": seed, "model": "banked", "task": "pcp", "max_steps": 80},
              "environment_version": "corrected-observation-v1", "source_sha256": "source1",
              "checkpoint_progress": {"epoch": 100, "updates": 1000},
              "model_signature": f"seed{seed}", "intervention": "none", "per_episode": episodes}
    report.update(overrides)
    path = tmp_path / (name or f"seed{seed}.json")
    path.write_text(json.dumps(report))
    return path


def episode(scenario, success, steps=10, reward=1., composition=(2, 1), **overrides):
    record = {"scenario_id": str(scenario), "success": success, "steps": steps,
              "team_return": reward, "composition": list(composition), "event_step": -1,
              "event_exposed": False, "intervention": "none", "intervention_step": -1}
    record.update(overrides)
    return record


def test_seed_means_receive_equal_weight_despite_unequal_episode_counts(tmp_path):
    first = report_file(tmp_path, 0, [episode(0, True, reward=10.)])
    second = report_file(tmp_path, 1, [episode(i, False, reward=0.) for i in range(9)])
    group = summarize_reports([first, second], bootstrap_samples=500, seed=3)["groups"][0]
    assert group["training_seeds"] == 2 and group["episodes"] == 10
    assert group["metrics"]["success_rate"]["mean"] == 0.5
    assert group["metrics"]["team_return"]["mean"] == 5.0
    assert group["metrics"]["mean_agent_return"]["mean"] == pytest.approx(5.0 / 3)
    assert group["metrics"]["success_rate"]["ci95"] == [0., 1.]
    assert group["metrics"]["completion_steps_horizon_capped"]["mean"] == 45.


def test_one_training_seed_has_no_between_seed_confidence_interval(tmp_path):
    source = report_file(tmp_path, 0, [episode(i, i % 2 == 0) for i in range(100)])
    group = summarize_reports([source], bootstrap_samples=100)["groups"][0]
    assert all(metric["ci95"] is None for metric in group["metrics"].values())
    assert group["per_seed"][0]["censored_completions"] == 50


def test_compositions_and_scheduled_failures_are_separate_strata(tmp_path):
    source = report_file(tmp_path, 0, [episode(0, True), episode(1, True, composition=(3, 2)),
                                     episode(2, True, event_step=20, event_exposed=True)])
    result = summarize_reports(source, bootstrap_samples=100)
    assert len(result["groups"]) == 3
    assert sum(group["episodes"] for group in result["groups"]) == 3


def test_agent_return_uses_each_compositions_roster_and_old_reports(tmp_path):
    # Neither num_agents nor the newly logged diagnostic is required in old reports.
    source = report_file(tmp_path, 0, [episode(0, True, reward=-3., composition=(2, 1)),
                                     episode(1, True, reward=-10., composition=(3, 2))])
    groups = summarize_reports(source, bootstrap_samples=100)["groups"]
    means = {tuple(g["composition"]): g["metrics"]["mean_agent_return"]["mean"] for g in groups}
    assert means == {(2, 1): -1., (3, 2): -2.}


@pytest.mark.parametrize("change", ["horizon", "environment", "checkpoint", "source", "model",
                                    "evaluator", "evaluation_version"])
def test_incompatible_experiments_are_not_pooled(tmp_path, change):
    first = report_file(tmp_path, 0, [episode(0, True)])
    overrides = {}
    if change == "horizon":
        overrides["config"] = {"seed": 1, "model": "banked", "task": "pcp", "max_steps": 100}
    elif change == "environment":
        overrides["environment_version"] = "another-observation-version"
    elif change == "checkpoint":
        overrides["checkpoint_progress"] = {"epoch": 200, "updates": 2000}
    elif change == "source":
        overrides["source_sha256"] = "source2"
    elif change == "model":
        overrides["config"] = {"seed": 1, "model": "shared", "task": "pcp", "max_steps": 80}
    elif change == "evaluator":
        overrides["evaluator"] = {"source": {"sha256": "new-evaluator"}}
    elif change == "evaluation_version":
        overrides["evaluation_version"] = 2
    second = report_file(tmp_path, 1, [episode(0, False)], **overrides)
    assert len(summarize_reports([first, second], bootstrap_samples=100)["groups"]) == 2


def test_duplicate_scenarios_and_conflicting_checkpoints_are_rejected(tmp_path):
    first = report_file(tmp_path, 0, [episode(0, True)])
    with pytest.raises(ValueError, match="Duplicate scenario"):
        summarize_reports([first, first], bootstrap_samples=100)
    second = report_file(tmp_path, 0, [episode(1, False)], name="other.json", model_signature="changed")
    with pytest.raises(ValueError, match="Conflicting checkpoints"):
        summarize_reports([first, second], bootstrap_samples=100)


def test_bootstrap_is_reproducible_and_outputs_do_not_overwrite(tmp_path):
    files = [report_file(tmp_path, i, [episode(0, bool(i % 2), reward=float(i))]) for i in range(5)]
    output = tmp_path / "summary.json"
    first = summarize_reports(files, output=output, bootstrap_samples=200, seed=19)
    second = summarize_reports(files, bootstrap_samples=200, seed=19)
    assert first == second == json.loads(output.read_text())
    with pytest.raises(FileExistsError):
        summarize_reports(files, output=output, bootstrap_samples=200, seed=19)


def test_event_exposure_does_not_discard_pre_event_completions(tmp_path):
    source = report_file(tmp_path, 0, [episode(0, True, steps=5, event_step=20),
                                     episode(1, False, steps=80, event_step=20, event_exposed=True)])
    group = summarize_reports(source, bootstrap_samples=100)["groups"][0]
    assert group["metrics"]["success_rate"]["mean"] == 0.5
    assert group["per_seed"][0]["event_exposed_episodes"] == 1
    assert group["per_seed"][0]["episodes"] == 2


def test_malformed_or_unknown_environment_reports_fail_explicitly(tmp_path):
    source = report_file(tmp_path, 0, [episode(0, True)], environment_version=None)
    with pytest.raises(ValueError, match="environment version"):
        summarize_reports(source, bootstrap_samples=100)


def test_sham_and_physical_failure_are_separate_matched_strata(tmp_path):
    scheduled = episode(0, False, event_step=20, event_exposed=True)
    actual = report_file(tmp_path, 0, [scheduled], name="failure.json", sham=False)
    sham_episode = dict(scheduled, event_exposed=False, sham=True)
    sham = report_file(tmp_path, 0, [sham_episode], name="sham.json", sham=True)
    groups = summarize_reports([actual, sham], bootstrap_samples=100)["groups"]
    assert len(groups) == 2
    assert {group["sham"] for group in groups} == {False, True}
    assert all(group["scheduled_sensor_failure"] for group in groups)


def protocol_panel():
    from softrole.report import protocol_identity
    from softrole.scenarios import make_scenarios
    from dataclasses import asdict
    import hashlib
    scenarios = [asdict(s) for s in make_scenarios(31, 2, [(2, 1)], 1., (1, 3))]
    panel_bytes = json.dumps(scenarios).encode()
    selection = {"rule": "first_saved_at_or_above", "target_steps": 1000}
    distribution = {"task": "pcp", "compositions": [[2, 1]], "episodes_per_composition": 2,
                    "scenario_seed": 31, "failure_probability": 1., "failure_window": [1, 3],
                    "victim_rule": "uniform_sensing_agent"}
    protocol = {"selection": selection, "distribution": distribution,
                "protocol_id": protocol_identity(selection, distribution),
                "scenario_sha256": hashlib.sha256(panel_bytes).hexdigest()}
    return scenarios, panel_bytes, protocol


def protocol_report(tmp_path, seed, epoch, sham=False, rewards=(1., 3.), **overrides):
    scenarios, _, protocol = protocol_panel()
    outcomes = [episode(s["scenario_id"], bool(i), reward=rewards[i],
                        **{k: v for k, v in s.items() if k != "scenario_id"})
                for i, s in enumerate(scenarios)]
    kwargs = dict(checkpoint_progress={"epoch": epoch, "updates": epoch * 10,
                                      "total_steps": 1000 + epoch, "total_episodes": 100},
                  scenarios=scenarios, scenarios_sha256=protocol["scenario_sha256"],
                  evaluation_protocol=protocol, sham=sham, checkpoint_sha256=f"checkpoint{seed}")
    kwargs.update(overrides)
    return report_file(tmp_path, seed, outcomes, name=f"seed{seed}-{epoch}-{sham}.json", **kwargs)


def test_explicit_budget_groups_different_epochs_and_retains_actual_progress(tmp_path):
    files = [protocol_report(tmp_path, seed, epoch) for seed, epoch in enumerate((100, 150, 200))]
    result = summarize_reports(files, bootstrap_samples=100)
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["training_seeds"] == 3
    assert "checkpoint_epoch" not in group
    assert [s["checkpoint"]["progress"]["epoch"] for s in group["per_seed"]] == [100, 150, 200]


@pytest.mark.parametrize("damage", ["budget", "identity", "panel", "count", "episode"])
def test_protocol_rejects_false_budget_and_panel_claims(tmp_path, damage):
    path = protocol_report(tmp_path, 0, 100)
    data = json.loads(path.read_text())
    if damage == "budget":
        data["checkpoint_progress"]["total_steps"] = 999
    elif damage == "identity":
        data["evaluation_protocol"]["protocol_id"] = "invented"
    elif damage == "panel":
        data["scenarios_sha256"] = "other"
    elif damage == "count":
        data["scenarios"].pop()
    else:
        data["per_episode"][0]["env_seed"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summarize_reports(path, bootstrap_samples=50)


def test_paired_failure_minus_sham_uses_complete_panel_and_seed_intervals(tmp_path):
    files = []
    for seed, fail, sham in [(0, (3., 7.), (1., 3.)), (1, (-3., -1.), (1., 3.))]:
        files += [protocol_report(tmp_path, seed, 100 + seed, False, fail),
                  protocol_report(tmp_path, seed, 100 + seed, True, sham)]
    group = summarize_reports(files, bootstrap_samples=500, seed=4)["paired_groups"][0]
    assert group["training_seeds"] == 2
    assert group["episodes"] == 4
    assert [r["team_return"] for r in group["per_seed"]] == [3., -4.]
    assert group["metrics"]["team_return"] == {"mean": -.5, "ci95": [-4., 3.]}
    assert group["metrics"]["mean_agent_return"]["mean"] == pytest.approx(-1 / 6)


@pytest.mark.parametrize("damage", ["streams", "checkpoint", "prefix", "seed_set"])
def test_pairing_requires_matched_streams_checkpoint_prefix_and_seeds(tmp_path, damage):
    failure = protocol_report(tmp_path, 0, 100, False)
    sham = protocol_report(tmp_path, 0, 100, True)
    data = json.loads(sham.read_text())
    if damage == "streams":
        data["scenarios"][0]["message_seed"] += 1
        data["per_episode"][0]["message_seed"] += 1
    elif damage == "checkpoint":
        data["checkpoint_sha256"] = "other"
    elif damage == "prefix":
        data["per_episode"][0]["pre_event_victim_reached"] = True
    else:
        data["config"]["seed"] = 1
    sham.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pairs|pre-event"):
        summarize_reports([failure, sham], bootstrap_samples=50)


def test_legacy_sidecar_requires_exact_companion_panel_and_keeps_inputs_immutable(tmp_path):
    source = protocol_report(tmp_path, 0, 100)
    data = json.loads(source.read_text())
    data.pop("evaluation_protocol")
    data.pop("scenarios_sha256")
    source.write_text(json.dumps(data))
    original = source.read_bytes()
    _, panel_bytes, protocol = protocol_panel()
    panel_path = tmp_path / "scenarios.json"
    panel_path.write_bytes(panel_bytes)
    sidecar = tmp_path / "protocol.json"
    sidecar.write_text(json.dumps(protocol))
    result = summarize_reports(source, bootstrap_samples=50, protocol=sidecar)
    assert result["groups"][0]["evaluation_protocol"]["protocol_id"] == protocol["protocol_id"]
    assert source.read_bytes() == original
    panel_path.write_bytes(panel_bytes + b" ")
    with pytest.raises(ValueError, match="hash"):
        summarize_reports(source, bootstrap_samples=50, protocol=sidecar)
