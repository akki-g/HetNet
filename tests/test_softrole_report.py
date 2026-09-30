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


@pytest.mark.parametrize("change", ["horizon", "environment", "checkpoint", "source", "model"])
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
