"""Independent stdlib audit of archived PCP nominal evaluation outcomes."""
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "stokes_runs/frozen_pcp_30m/frozen_pcp_30m_20261002_slurm"


def mean(values):
    return statistics.fmean(values)


def metrics(rows):
    successes = sum(r["success"] for r in rows)
    succ = [r for r in rows if r["success"]]
    return dict(
        episodes=len(rows), successes=successes, failures=len(rows)-successes,
        success_rate=successes/len(rows),
        mean_team_return=mean(r["team_return"] for r in rows),
        mean_agent_return=mean(r["team_return"]/r["num_agents"] for r in rows),
        mean_steps=mean(r["steps"] for r in rows),
        horizon_capped_completion_steps=mean(r["steps"] if r["success"] else 80 for r in rows),
        successful_only_steps=mean(r["steps"] for r in succ) if succ else None,
        mean_sensing_agent_return=mean(r["return_nocap"] for r in rows),
        mean_actuating_agent_return=mean(r["return_cap"] for r in rows),
        gate_entropy_episode_mean=mean(r["gate_entropy"] for r in rows),
        alpha_null_episode_mean=mean(r["alpha_null"] for r in rows),
        unsuccessful_steps=dict(Counter(r["steps"] for r in rows if not r["success"])),
    )


def unresolved_from_rewards(rows):
    """With -.05 until completion then zero, R=-.05H implies no completion."""
    failed = [r for r in rows if not r["success"]]
    unresolved_p, unresolved_a = Counter(), Counter()
    both_complete = 0
    for row in failed:
        assert row["steps"] == 80
        p = row["num_p"]
        np = sum(math.isclose(value, -4, abs_tol=1e-10) for value in row["agent_returns"][:p])
        na = sum(math.isclose(value, -4, abs_tol=1e-10) for value in row["agent_returns"][p:])
        unresolved_p[np] += 1
        unresolved_a[na] += 1
        both_complete += np == 0 and na == 0
    assert both_complete == 0
    return dict(failed_episodes=len(failed), unfinished_sensing_agents_histogram=dict(unresolved_p),
                unfinished_actuating_agents_histogram=dict(unresolved_a),
                failed_episode_mean_sensing_agent_return=mean(r["return_nocap"] for r in failed) if failed else None,
                failed_episode_mean_actuating_agent_return=mean(r["return_cap"] for r in failed) if failed else None)


def main():
    panel = json.loads((DATA / "scenarios_nominal.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    results = []
    hashes = {}
    episode_sets = defaultdict(list)
    checks = Counter()
    for path in sorted((DATA / "results").glob("*_nominal.json")):
        payload = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
        report = json.loads(payload)
        rows = report["per_episode"]
        model, seed = report["config"]["model"], report["training_seed"]
        assert report["scenarios"] == panel
        assert len(rows) == len(panel) == 2500
        assert report["episodes"] == len(rows)
        assert report["evaluator"]["source"]["sha256"] == manifest["evaluation_source_sha256"]
        assert report["config"]["max_steps"] == 80
        assert len({r["scenario_id"] for r in rows}) == len(rows)
        for row, scenario in zip(rows, panel):
            assert row["scenario_id"] == scenario["scenario_id"]
            assert row["composition"] == [scenario["num_p"], scenario["num_a"]]
            for key in ("env_seed", "action_seed", "message_seed"):
                assert row[key] == scenario[key]
            assert row["num_agents"] == sum(row["composition"])
            assert len(row["agent_returns"]) == row["num_agents"]
            assert math.isclose(sum(row["agent_returns"]), row["team_return"], abs_tol=1e-12)
            assert math.isclose(row["team_return"]/row["num_agents"], row["mean_agent_return"], abs_tol=1e-12)
            assert 1 <= row["steps"] <= 80
            assert row["success"] or row["steps"] == 80
            # All returns are zero or negative multiples of the declared .05 penalty.
            for value in row["agent_returns"]:
                assert value <= 0 and math.isclose(value / .05, round(value / .05), abs_tol=1e-10)
                assert value >= -.05 * row["steps"] - 1e-12
            checks["episode_invariants"] += 1
        total = metrics(rows)
        for key in ("success_rate", "mean_team_return", "mean_agent_return"):
            assert math.isclose(total[key], report[key], abs_tol=1e-12)
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row["composition"])].append(row)
        by_composition = []
        for composition, group in sorted(groups.items()):
            assert len(group) == 500
            values = metrics(group)
            entry = dict(model=model, seed=seed, composition=list(composition), **values,
                         unresolved_from_rewards=unresolved_from_rewards(group))
            by_composition.append(entry)
            episode_sets[(model, composition)].append(entry)
        results.append(dict(model=model, seed=seed, path=str(path.relative_to(ROOT)),
                            checkpoint_progress=report["checkpoint_progress"],
                            metrics=total, compositions=by_composition,
                            unresolved_from_rewards=unresolved_from_rewards(rows)))
    equal_seed = []
    for (model, composition), entries in sorted(episode_sets.items()):
        assert len(entries) == 3
        equal_seed.append(dict(model=model, composition=list(composition),
                              seeds=[e["seed"] for e in entries],
                              success_counts=[e["successes"] for e in entries],
                              success_rates=[e["success_rate"] for e in entries],
                              metrics={key:mean(e[key] for e in entries) for key in (
                                  "success_rate", "mean_team_return", "mean_agent_return", "mean_steps",
                                  "horizon_capped_completion_steps", "successful_only_steps",
                                  "mean_sensing_agent_return", "mean_actuating_agent_return")},
                              seed_ranges={key:[min(e[key] for e in entries), max(e[key] for e in entries)]
                                           for key in ("success_rate", "mean_team_return", "mean_steps")}))
    evidence = [
        dict(path="source/envs/ic3net_envs/predator_capture_env.py", lines="42-44, 81-87, 416-420, 450-455, 486-545",
             claim="Default -.05 step cost, zero completion reward, no partial capture reward; P remains reached and A remains captured; success requires all P/A reached and all A captured."),
        dict(path="source/softrole/env.py", lines="55-71, 165-182",
             claim="PCP uses default mixed reward, second_reward_scheme false; physical rewards unchanged; horizon80 truncates unsuccessful episodes."),
        dict(path="source/softrole/rollout.py", lines="148-153, 171-184",
             claim="Episode rewards accumulate physical per-agent rewards; returns ordered P then A; steps count environment transitions."),
        dict(path="source/softrole/evaluate.py", lines="108-116",
             claim="Reported agent mean is episode average of team return divided by that episode's roster size."),
        dict(path="source/softrole/report.py", lines="100-111",
             claim="Reporting caps unsuccessful completions at horizon and keeps seed means separate."),
    ]
    for source in evidence:
        source["sha256"] = hashlib.sha256((DATA / source["path"]).read_bytes()).hexdigest()
    output = dict(input_hashes=hashes, checks=dict(checks), per_seed=results,
                  equal_seed_per_composition=equal_seed,
                  code_evidence=evidence,
                  reward_inference="For each agent r_t=-.05 until its post-action completion, then0. Hence return=-.05 times number of unfinished post-action transitions. In an80-step failed episode, return=-4 implies no completion by horizon. For P completion is arrival; for A it is successful capture. A return alone cannot distinguish never arriving from arriving but not capturing.",
                  limitations=[
                      "No new policy evaluation, no causal attribution of failure mechanism.",
                      "Three training seeds; seed ranges are descriptive, not confidence intervals.",
                      "Agent return means average within each episode then across episodes.",
                      "All unsuccessful episodes were observed to reach the 80-step horizon.",
                      "Successful-only completion means condition on success and can conceal failures.",
                  ])
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+"\n")
    for row in equal_seed:
        print(row["model"], row["composition"], row["success_counts"], row["metrics"])


if __name__ == "__main__":
    main()
