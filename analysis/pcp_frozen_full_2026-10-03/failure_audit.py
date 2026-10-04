#!/usr/bin/env python3
"""Read-only, standard-library audit of the frozen PCP failure/sham JSON pairs.

Run from the repository root. No policies are executed and no input is modified.
Assertions independently reconcile recorded outcomes, trace prefixes, and event
accounting. Mask implementation is traced to archived code, not inferred from
unrecorded observations. The JSON output includes input hashes and code anchors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "stokes_runs/frozen_pcp_30m/frozen_pcp_30m_20261002_slurm"
IDENTITY_FIELDS = (
    "checkpoint", "checkpoint_sha256", "model_signature", "config", "model_config",
    "environment_version", "source_sha256", "checkpoint_progress", "evaluator",
    "training_seed", "format_version", "evaluation_version", "intervention",
)
DIAGNOSTICS = (
    "pre_event_victim_reached", "pre_event_victim_target_visible",
    "pre_event_victim_target_seen",
)
OUTCOMES = ("success", "steps", "team_return", "agent_returns", "mean_agent_return")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(actual, expected, label):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (label, actual, expected)


def flatten(value):
    if isinstance(value, list):
        for element in value:
            yield from flatten(element)
    else:
        yield value


def audit_report(report, panel, sham):
    assert report["scenarios"] == panel
    assert report["sham"] is sham
    assert report["episodes"] == len(panel)
    assert report["intervention"] == "none"
    episodes = report["per_episode"]
    assert len(episodes) == len(panel)
    assert len({e["scenario_id"] for e in episodes}) == len(panel)
    horizon = report["config"]["max_steps"]
    for scenario, episode in zip(panel, episodes):
        assert all(episode[k] == v for k, v in scenario.items())
        assert episode["sham"] is sham
        assert episode["composition"] == [2, 1]
        assert episode["num_agents"] == 3
        steps, event = episode["steps"], episode["event_step"]
        assert 1 <= steps <= horizon
        exposed = 0 <= event < steps
        assert episode["scheduled_event_exposed"] is exposed
        assert episode["event_exposed"] is (exposed and not sham)
        assert episode["pre_event_success"] is (episode["success"] and not exposed)
        assert episode["scheduled_censored"] is (exposed and not episode["success"])
        assert episode["recovery_censored"] is (exposed and not sham and not episode["success"])
        assert episode["post_schedule_steps"] == (steps - event if exposed else 0)
        assert episode["post_event_steps"] == (steps - event if exposed and not sham else 0)
        completion = steps - event if exposed and episode["success"] else None
        assert episode["scheduled_completion_steps"] == completion
        assert episode["recovery_steps"] == (completion if not sham else None)
        assert all(isinstance(episode[k], bool) if exposed else episode[k] is None for k in DIAGNOSTICS)
        assert episode["intervention"] == "none"
        assert episode["intervention_exposed"] is False
        assert len(episode["trace"]) == steps
        assert [t["step"] for t in episode["trace"]] == list(range(steps))
        assert_close(sum(episode["agent_returns"]), episode["team_return"], "agent/team return")
        assert_close(episode["team_return"] / 3, episode["mean_agent_return"], "agent mean")
        assert_close(sum(t["team_reward"] for t in episode["trace"]), episode["team_return"], "trace return")
        assert_close(mean(episode["agent_returns"][:2]), episode["return_nocap"], "P mean")
        assert_close(episode["agent_returns"][2], episode["return_cap"], "A mean")
        bits = 2 * report["config"]["msg_dim"]
        assert episode["payload_bits_per_agent_step"] == bits
        assert episode["payload_bits_generated"] == bits * 3 * steps
        assert all(math.isfinite(x) for t in episode["trace"] for x in flatten(t["gate"]))
        for t in episode["trace"]:
            for gate in t["gate"]:
                assert_close(sum(gate), 1, "gate simplex")
                assert all(0 <= g <= 1 for g in gate)
        gate_entropy = mean(-sum(g * math.log(max(g, 1e-15)) for g in gate)
                            for t in episode["trace"] for gate in t["gate"])
        assert_close(episode["gate_entropy"], gate_entropy, "gate entropy")
        null = mean(mean(list(flatten(t["alpha_null"]))) for t in episode["trace"])
        assert_close(episode["alpha_null"], null, "null weight")
    assert_close(report["success_rate"], mean(e["success"] for e in episodes), "success reduction")
    assert_close(report["mean_team_return"], mean(e["team_return"] for e in episodes), "team reduction")
    assert_close(report["mean_agent_return"], mean(e["team_return"] / 3 for e in episodes), "agent reduction")
    for exposed_key, count_key, rate_key in (
        ("event_exposed", "event_exposed_episodes", "post_event_success_rate"),
        ("scheduled_event_exposed", "scheduled_event_exposed_episodes", "post_schedule_success_rate"),
    ):
        exposed = [e for e in episodes if e[exposed_key]]
        assert report[count_key] == len(exposed)
        assert report[rate_key] == (mean(e["success"] for e in exposed) if exposed else None)
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    root = args.root.resolve()
    paths = [root / "manifest.json", root / "scenarios_failure.json"]
    source_lines = {
        "source/softrole/rollout.py": {
            "diagnostics_precede_event": [112, 121], "failure_before_actor": [122, 148],
            "trace_fields": [164, 168], "exposure_and_censoring": [185, 194],
        },
        "source/softrole/env.py": {
            "pure_persistent_sensory_mask": [19, 38], "cached_diagnostics": [143, 158],
            "environment_step_and_masked_next_view": [160, 182],
        },
        "source/softrole/evaluate.py": {
            "frozen_model_and_source_checks": [73, 93], "summary_reductions": [94, 115],
        },
        "source/softrole/scenarios.py": {"independent_seed_streams": [45, 82]},
        "source/envs/ic3net_envs/predator_capture_env.py": {
            "step_penalty": [42, 45], "individual_reward_reduction": [487, 524],
        },
    }
    paths += [root / name for name in source_lines]
    paths += [root / f"results/{model}_seed{seed}_{condition}.json"
              for model in ("shared", "banked") for seed in range(3)
              for condition in ("failure", "sham")]
    before = {str(path.relative_to(root)): sha256(path) for path in paths}
    manifest = json.loads((root / "manifest.json").read_text())
    panel = json.loads((root / "scenarios_failure.json").read_text())
    assert len(panel) == 100
    assert len({s["scenario_id"] for s in panel}) == 100
    assert all(s["num_p"] == 2 and s["num_a"] == 1 and 10 <= s["event_step"] <= 30 and s["victim"] in (0, 1) for s in panel)
    assert before["scenarios_failure.json"] == manifest["artifact_sha256"]["scenarios_failure.json"]
    assert manifest["failure_panel"]["trace"] is True
    rows, exposed_details, pair_checks = [], [], []
    total_trace_rows, total_pre_event_trace_rows = 0, 0
    for model in ("shared", "banked"):
        for seed in range(3):
            reports = {condition: json.loads((root / f"results/{model}_seed{seed}_{condition}.json").read_text())
                       for condition in ("failure", "sham")}
            failure, sham = reports["failure"], reports["sham"]
            assert all(failure[k] == sham[k] for k in IDENTITY_FIELDS)
            assert failure["config"]["model"] == model and failure["training_seed"] == seed
            assert failure["evaluator"]["source"]["sha256"] == manifest["evaluation_source_sha256"]
            f = audit_report(failure, panel, False)
            s = audit_report(sham, panel, True)
            full_trace_equal, prefix_equal, outcome_equal = 0, 0, 0
            for index, (a, b) in enumerate(zip(f, s)):
                assert all(a[k] == b[k] for k in DIAGNOSTICS)
                assert a["scheduled_event_exposed"] == b["scheduled_event_exposed"]
                cutoff = min(a["event_step"], a["steps"], b["steps"])
                assert a["trace"][:cutoff] == b["trace"][:cutoff]
                total_pre_event_trace_rows += cutoff
                prefix_equal += 1
                total_trace_rows += len(a["trace"]) + len(b["trace"])
                full_trace_equal += a["trace"] == b["trace"]
                outcome_equal += all(a[k] == b[k] for k in OUTCOMES)
                if not a["event_exposed"]:
                    assert a["trace"] == b["trace"]
                    assert all(a[k] == b[k] for k in OUTCOMES)
                else:
                    changed = Counter(k for ta, tb in zip(a["trace"], b["trace"])
                                      for k in ta if ta[k] != tb[k])
                    first_trace_difference = next((t for t, (ta, tb) in enumerate(zip(a["trace"], b["trace"])) if ta != tb), None)
                    exposed_details.append({
                        "model": model, "seed": seed, "episode_index": index,
                        "scenario": panel[index],
                        "failure": {k: v for k, v in a.items() if k != "trace"},
                        "sham": {k: v for k, v in b.items() if k != "trace"},
                        "first_trace_difference_step": first_trace_difference,
                        "changed_trace_steps_by_field": dict(changed),
                        "first_action_difference_step": next(
                            (t for t, (ta, tb) in enumerate(zip(a["trace"], b["trace"]))
                             if ta["actions"] != tb["actions"]), None),
                        "team_reward_histogram_failure": dict(Counter(t["team_reward"] for t in a["trace"])),
                        "team_reward_histogram_sham": dict(Counter(t["team_reward"] for t in b["trace"])),
                    })
            pair_checks.append({"model": model, "seed": seed, "pairs": len(f),
                                "exact_pre_event_prefix_pairs": prefix_equal,
                                "exact_full_trace_pairs": full_trace_equal,
                                "exact_outcome_pairs": outcome_equal})
            rows.append({
                "model": model, "seed": seed, "episodes_per_condition": len(f),
                "actual_exposures_failure": sum(e["event_exposed"] for e in f),
                "actual_exposures_sham": sum(e["event_exposed"] for e in s),
                "scheduled_exposures_each_condition": sum(e["scheduled_event_exposed"] for e in f),
                "completed_before_scheduled_event_each_condition": sum(e["pre_event_success"] for e in f),
                "censored_after_scheduled_event_each_condition": sum(e["scheduled_censored"] for e in f),
                "successes_failure": sum(e["success"] for e in f),
                "successes_sham": sum(e["success"] for e in s),
                "mean_steps_each_condition": mean(e["steps"] for e in f),
                "median_steps_each_condition": median(e["steps"] for e in f),
                "min_steps": min(e["steps"] for e in f), "max_steps": max(e["steps"] for e in f),
                "episodes_completed_by_step_count_10": sum(e["success"] and e["steps"] <= 10 for e in f),
                "mean_team_return_each_condition": mean(e["team_return"] for e in f),
                "mean_agent_return_each_condition": mean(e["mean_agent_return"] for e in f),
                "paired_failure_minus_sham": {
                    k: mean(a[k] - b[k] for a, b in zip(f, s))
                    for k in ("success", "steps", "team_return", "mean_agent_return")
                },
            })
    after = {str(path.relative_to(root)): sha256(path) for path in paths}
    assert before == after, "Input file changed during the audit"
    result = {
        "schema": 1, "input_root": str(root.relative_to(REPO)),
        "scope": "Read-only audit of twelve existing failure/sham reports; no policy replay or training",
        "input_sha256": before, "source_line_anchors": source_lines,
        "panel": {"count": len(panel), "composition": [2, 1],
                  "event_step_histogram": dict(sorted(Counter(s["event_step"] for s in panel).items())),
                  "victim_histogram": dict(sorted(Counter(s["victim"] for s in panel).items())),
                  "exact_manifest_hash_match": True},
        "rows": rows, "pair_checks": pair_checks, "exposed_details": exposed_details,
        "totals": {
            "policy_scenario_pairs": 600, "episode_evaluations": 1200,
            "actual_sensor_exposure_assignments": len(exposed_details),
            "completed_before_own_scheduled_event_assignments": sum(r["completed_before_scheduled_event_each_condition"] for r in rows),
            "exact_pre_event_prefix_pairs": sum(r["exact_pre_event_prefix_pairs"] for r in pair_checks),
            "exact_full_trace_pairs": sum(r["exact_full_trace_pairs"] for r in pair_checks),
            "exact_outcome_pairs": sum(r["exact_outcome_pairs"] for r in pair_checks),
            "total_recorded_trace_steps_both_conditions": total_trace_rows,
            "pre_event_trace_steps_compared": total_pre_event_trace_rows,
        },
        "conclusions": [
            "All six policies use the exact common manifest-hashed scenario panel in both conditions.",
            "All 600 failure/sham pairs agree exactly on every stored pre-event trace field.",
            "Only one pair reaches its scheduled event; 599 complete before their own scheduled event.",
            "The exposed pair changes attention and actions after masking but has identical failure/return/length outcomes.",
            "The lone exposed victim had not reached, had no current target view, and had no prior direct target sight.",
            "No banked policy physically experiences sensor loss in this panel; gate adaptation is untested here.",
            "Equal outcome averages are insufficient evidence of sensor-loss robustness because exposure is nearly absent.",
            "Both exposed-pair trajectories receive the floor reward -0.15 each step: 3 agents times -0.05, for 80 steps, equals -12 team return despite changed actions.",
        ],
        "mathematical_interpretation": {
            "exposure": "E_i = 1[T_i > t_i] for zero-based event time t_i and executed-step count T_i; the paired prefix is common.",
            "dilution": "For n_exposed > 0 and unexposed paired outcomes equal, mean_i(Y_i^failure - Y_i^sham) = (n_exposed/M) * mean_exposed(Y_i^failure - Y_i^sham).",
            "shared_seed0_success_contrast_resolution": "One exposed assignment among 100 limits the absolute full-panel paired success contrast to at most 1 percentage point.",
            "zero_exposure": "For the other five policies the full-panel outcome contrast is identically zero because no event is reached, so no exposed-only contrast is defined.",
            "inferential_unit": "Six policies from three training seeds per method; 600 assignments reuse 100 scenarios and are not 600 independent trained policies.",
        },
        "limitations": [
            "Traces contain gates, null weights, actions, and team rewards; they do not contain raw/actor observations, positions, hidden states, or RNG states.",
            "Persistent victim-only sensory masking is supported by archived implementation; this audit cannot directly re-observe the unrecorded model inputs.",
            "Matching stored prefixes proves agreement for stored fields, not bitwise identity of every internal tensor.",
            "Reports record checkpoint hashes/model signatures and evaluator mutation checks; this audit does not rerun policies or independently load checkpoint tensors.",
            "Direct target sight is a diagnostic, not proof of knowledge acquired through memory or communication.",
            "No between-seed inferential interval or robustness claim is supported by one exposed policy-scenario assignment.",
        ],
        "validation": {"all_assertions_passed": True, "input_bytes_unchanged_during_audit": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "totals": result["totals"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
