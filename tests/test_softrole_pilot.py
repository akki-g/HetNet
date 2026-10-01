"""Assigned-panel pairing and replay of the narrow native-PCP failure pilot."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from softrole import CHECKPOINT_VERSION, ENVIRONMENT_VERSION, EVALUATION_VERSION
from softrole.config import recipe
from softrole.model import SoftRoleNet
from softrole.pilot import DIAGNOSTICS, paired_summary, run_pilot


def paired_reports():
    scenarios = [{"scenario_id": str(i), "env_seed": 100 + i,
                  "action_seed": 200 + i, "message_seed": 300 + i,
                  "num_p": 2, "num_a": 1, "event_step": 10, "victim": 0}
                 for i in range(4)]
    base = {"scenarios": scenarios, "checkpoint": "synthetic.pt",
            "checkpoint_sha256": "checkpoint", "model_signature": "model",
            "source_sha256": "training-source", "training_seed": 0,
            "checkpoint_progress": {"epoch": 1, "updates": 10,
                                    "total_steps": 1000, "total_episodes": 50},
            "config": {"model": "shared", "max_steps": 80},
            "evaluation_version": EVALUATION_VERSION, "evaluator": {"source": "evaluator"},
            "intervention": "none"}
    reports = []
    for sham in (False, True):
        rows = []
        success = [True, not sham, sham, False]
        steps = [5, 80 if sham else 20, 25 if sham else 80, 80]
        returns = [-4, -80 if sham else -10, -20 if sham else -90, -70 if sham else -100]
        for i, scenario in enumerate(scenarios):
            exposed = i != 0
            diagnostic = [(None, None, None), (False, True, True),
                          (True, True, True), (False, False, False)][i]
            rows.append({**scenario, "composition": [2, 1], "success": success[i],
                         "steps": steps[i], "team_return": returns[i],
                         "scheduled_event_exposed": exposed,
                         "event_exposed": exposed and not sham,
                         "pre_event_success": i == 0,
                         "scheduled_censored": exposed and not success[i],
                         **dict(zip(DIAGNOSTICS, diagnostic)),
                         "trace": [{"step": t, "actions": [4, 4, 4]} for t in range(steps[i])]})
        reports.append({**deepcopy(base), "sham": sham, "per_episode": rows})
    return reports


def test_paired_summary_keeps_early_completions_and_discordant_full_panel_outcomes():
    failure, sham = paired_reports()
    # Pair by scenario identity even if the outcome rows arrive in another order.
    sham["per_episode"].reverse()
    result = paired_summary(failure, sham)
    for name in ("failure", "sham"):
        outcome = result[name]
        assert outcome["episodes"] == 4 and outcome["success_rate"] == .5
        assert outcome["pre_event_successes"] == 1
        assert outcome["scheduled_event_exposed"] == outcome["diagnostic_denominator"] == 3
        assert outcome["censored_completions"] == outcome["post_schedule_censored"] == 2
        assert outcome["diagnostic_true_counts"] == dict(zip(DIAGNOSTICS, [1, 2, 2]))
    assert result["failure"]["actual_event_exposed"] == 3
    assert result["sham"]["actual_event_exposed"] == 0
    assert result["failure"]["team_return"] == -51
    assert result["sham"]["team_return"] == -43.5
    assert result["failure"]["completion_steps_horizon_capped"] == 46.25
    assert result["sham"]["completion_steps_horizon_capped"] == 47.5
    assert result["failure_minus_sham"] == {
        "success_rate": 0., "team_return": -7.5, "completion_steps_horizon_capped": -1.25}
    assert result["failure_only_successes"] == result["sham_only_successes"] == 1
    assert result["paired_prefixes_and_diagnostics_verified"]


@pytest.mark.parametrize("corruption", ["assigned_scenario", "duplicate_outcome", "missing_outcome",
                                         "outcome_seed", "prefix", "diagnostic", "exposure"])
def test_pairing_rejects_incompatible_records(corruption):
    failure, sham = paired_reports()
    if corruption == "assigned_scenario":
        sham["scenarios"][1]["victim"] = 1
    elif corruption == "duplicate_outcome":
        sham["per_episode"][1] = deepcopy(sham["per_episode"][0])
    elif corruption == "missing_outcome":
        sham["per_episode"].pop()
    elif corruption == "outcome_seed":
        # Both sides can agree with each other and still not belong to the panel.
        for report in (failure, sham):
            report["per_episode"][1]["env_seed"] += 1000
    elif corruption == "prefix":
        sham["per_episode"][1]["trace"][9]["actions"] = [1, 4, 4]
    elif corruption == "diagnostic":
        sham["per_episode"][1][DIAGNOSTICS[1]] = False
    else:
        sham["per_episode"][1]["scheduled_event_exposed"] = False
    with pytest.raises(ValueError):
        paired_summary(failure, sham)


def save_checkpoint(path, model="shared", **overrides):
    config = recipe("pcp", model=model, pre_dim=8, hidden_dim=8, experts=2,
                    heads=1, head_dim=4, msg_dim=2, **overrides)
    with torch.random.fork_rng():
        torch.manual_seed(123)
        net = SoftRoleNet(**config.model_kwargs()).double()
    torch.save({"format_version": CHECKPOINT_VERSION, "environment_version": ENVIRONMENT_VERSION,
                "config": config.to_dict(), "model_config": config.model_kwargs(),
                "model_state": net.state_dict(), "source_sha256": "test-training-source",
                "epoch": 1, "updates": 10, "total_steps": 1000, "total_episodes": 50}, path)
    return path


@pytest.mark.parametrize("case,match", [
    ("missing_pair", "matched shared and banked"),
    ("non_native", "nominal shared/banked PCP"),
    ("failure_trained", "nominal shared/banked PCP"),
    ("config_mismatch", "configurations differ"),
    ("source_mismatch", "source"),
    ("missing_progress", "progress"),
])
def test_pilot_rejects_ineligible_comparisons_before_creating_output(tmp_path, case, match):
    shared_options = {"num_p": 3} if case == "non_native" else {}
    if case == "failure_trained":
        shared_options["failure_prob"] = .5
    paths = [save_checkpoint(tmp_path / "shared.pt", **shared_options)]
    if case != "missing_pair":
        banked_options = {"comm_range": 1.} if case == "config_mismatch" else {}
        paths.append(save_checkpoint(tmp_path / "banked.pt", model="banked", **banked_options))
    if case in ("source_mismatch", "missing_progress"):
        saved = torch.load(paths[-1], map_location="cpu", weights_only=False)
        if case == "source_mismatch":
            saved["source_sha256"] = "another-training-source"
        else:
            del saved["total_steps"]
        torch.save(saved, paths[-1])
    output = tmp_path / "pilot"
    with pytest.raises(ValueError, match=match):
        run_pilot(paths, output, "All synthetic checkpoint pairs at epoch 1", episodes=2)
    assert not output.exists()


def test_real_pilot_archives_panel_and_source_replays_and_refuses_overwrite(tmp_path):
    paths = [save_checkpoint(tmp_path / f"{model}.pt", model=model)
             for model in ("shared", "banked")]
    checkpoint_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    output = tmp_path / "pilot"
    rule = "All synthetic shared/banked seed 0 checkpoints at epoch 1; no outcome selection"
    summary = run_pilot(paths, output, rule, episodes=2, seed=1700)
    manifest = json.loads((output / "pilot.json").read_text())
    source = json.loads((output / "source_manifest.json").read_text())
    scenarios = json.loads((output / "scenarios.json").read_text())
    assert manifest["checkpoint_rule"] == rule
    assert manifest["composition"] == [2, 1] and manifest["failure_window"] == [10, 30]
    assert manifest["scenarios_sha256"] == hashlib.sha256((output / "scenarios.json").read_bytes()).hexdigest()
    assert manifest["evaluator_source_sha256"] == source["sha256"]
    assert len(scenarios) == 2
    assert all((row["num_p"], row["num_a"]) == (2, 1)
               and 10 <= row["event_step"] <= 30 and row["victim"] in (0, 1) for row in scenarios)
    assert "softrole/pilot.py" in source["files"]
    for name, digest in source["files"].items():
        assert hashlib.sha256((output / "source" / name).read_bytes()).hexdigest() == digest
    reports = {}
    for model in ("shared", "banked"):
        for condition in ("failure", "sham"):
            filename = f"{model}_seed0_{condition}.json"
            report = json.loads((output / filename).read_text())
            reports[filename] = report
            assert report["scenarios"] == scenarios
            assert report["episodes"] == 2 and report["config"]["max_steps"] == 80
            assert report["evaluator"]["source"]["sha256"] == source["sha256"]
            assert report["source_sha256"] == "test-training-source"
            assert all("trace" in row and all(key in row for key in DIAGNOSTICS)
                       for row in report["per_episode"])
    assert len(summary["policies"]) == 2
    assert all(row["paired_prefixes_and_diagnostics_verified"] for row in summary["policies"])
    replay_output = tmp_path / "replay"
    assert run_pilot(paths, replay_output, rule, episodes=2, seed=1700) == summary
    for filename, report in reports.items():
        assert json.loads((replay_output / filename).read_text()) == report
    with pytest.raises(FileExistsError):
        run_pilot(paths, output, rule, episodes=2, seed=1700)
    assert json.loads((output / "summary.json").read_text()) == summary
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == checkpoint_hashes
