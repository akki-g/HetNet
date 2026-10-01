"""Read-only sample-budget analysis of already copied PP/PCP training metrics.

Writes only this script's directory. No policy loading, training, evaluation,
remote scheduler query, or changes to existing inputs occur.
"""
import csv
import hashlib
import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
INPUTS = {}
METRICS = ("steps_taken", "success_rate", "team_return")


def read(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()
    }
    return raw.decode()


def average(rows, key):
    return math.fsum(r[key] * r["episodes"] for r in rows) / sum(r["episodes"] for r in rows)


def window(rows, budget=None):
    selected = (rows if budget is None else [r for r in rows if r["total_steps"] <= budget])[-50:]
    assert len(selected) == 50
    result = {
        "requested_budget": budget,
        "epochs": [selected[0]["epoch"], selected[-1]["epoch"]],
        "updates_at_endpoint": selected[-1]["updates"],
        "steps_start": selected[0]["total_steps"] - selected[0]["steps"],
        "steps_end": selected[-1]["total_steps"],
        "steps": sum(r["steps"] for r in selected),
        "episodes": sum(r["episodes"] for r in selected),
        **{key: average(selected, key) for key in METRICS},
    }
    assert abs(result["steps_taken"] - result["steps"] / result["episodes"]) < 1e-10
    return result


def group(values):
    assert len(values) == 3
    return {
        "n_seeds": len(values),
        "steps_end_min": min(v["steps_end"] for v in values),
        "steps_end_max": max(v["steps_end"] for v in values),
        **{key: {"mean": math.fsum(v[key] for v in values) / len(values),
                 "seed_min": min(v[key] for v in values),
                 "seed_max": max(v[key] for v in values)} for key in METRICS},
    }


inventory = json.loads(read(ROOT / "analysis/results_audit_2026-09-30/training/inventory.json"))
runs = []
for prior in inventory["runs"]:
    if prior["task"] not in ("pp", "pcp") or not prior["model"].startswith("SoftRole"):
        continue
    config = json.loads(read(ROOT / prior["config_source"]))
    rows = [json.loads(line) for line in read(ROOT / prior["source"]).splitlines() if line.strip()]
    assert config == prior["config"]
    assert config["nprocesses"] == 4 and config["batch_steps"] == 500
    assert config["updates_per_epoch"] == 10 and config["max_steps"] == 80
    assert config["failure_prob"] == 0 and config["compositions"] == []
    assert config["total_steps"] is None
    assert len(rows) == prior["completed_epochs"]
    assert rows[-1]["total_steps"] == prior["recorded_total_steps"]
    steps = episodes = 0
    for epoch, row in enumerate(rows, 1):
        steps += row["steps"]
        episodes += row["episodes"]
        assert row["epoch"] == epoch and row["updates"] == epoch * 10
        assert row["total_steps"] == steps and row["total_episodes"] == episodes
        assert row["event_exposed_episodes"] == 0 and not row["partial_epoch"]
        assert abs(row["steps_taken"] * row["episodes"] - row["steps"]) < 1e-7
    runs.append({key: prior[key] for key in ("task", "model", "seed", "source", "config_source")} | {
        "epochs": len(rows), "total_steps": steps, "total_episodes": episodes,
        "at_10m": window(rows, 10_000_000), "at_20m": window(rows, 20_000_000),
        "latest": window(rows), "_rows": rows,
    })
assert len(runs) == 12
groups = []
for task in ("pp", "pcp"):
    task_runs = [r for r in runs if r["task"] == task]
    common_budget = min(r["total_steps"] for r in task_runs) // 500_000 * 500_000
    for run in task_runs:
        run["latest_common_task_budget"] = window(run["_rows"], common_budget)
    for model in ("SoftRole shared", "SoftRole banked"):
        selected = [r for r in task_runs if r["model"] == model]
        assert {r["seed"] for r in selected} == {0, 1, 2}
        stats = {name: group([r[name] for r in selected]) for name in
                 ("at_10m", "at_20m", "latest", "latest_common_task_budget")}
        a, b = stats["at_10m"]["steps_taken"]["mean"], stats["at_20m"]["steps_taken"]["mean"]
        stats["change_10m_to_20m"] = {
            "episode_length_difference": b-a,
            "episode_length_percent_change": 100 * (b-a) / a,
            "all_three_seeds_lower_length": all(r["at_20m"]["steps_taken"] < r["at_10m"]["steps_taken"] for r in selected),
            "seed_differences": [{"seed": r["seed"], "episode_length_difference": r["at_20m"]["steps_taken"]-r["at_10m"]["steps_taken"]} for r in selected],
        }
        groups.append({"task": task, "model": model, "latest_common_task_budget": common_budget, "windows": stats})
for run in runs:
    del run["_rows"]

# Legacy stdout's cumulative sample counters are inflated. Only archived,
# structured ledgers support exact sample-budget windows; do not repair tails
# by dividing their printed counters by a guessed constant.
legacy = []
for prior in inventory["runs"]:
    if prior["task"] not in ("pp", "pcp") or prior["model"] != "HetNet Real":
        continue
    path = ROOT / prior["config_source"]
    rows = [json.loads(line) for line in read(path.parent / "metrics.jsonl").splitlines()]
    for row in rows:
        row["updates"] = row["epoch"] * prior["updates_per_epoch"]
        row["team_return"] = sum(row["reward_per_agent"])
    record = {key: prior[key] for key in ("task", "model", "seed")} | {
        "source": str((path.parent / "metrics.jsonl").relative_to(ROOT)),
        "last_exact_steps": rows[-1]["total_steps"],
        "at_10m": window(rows, 10_000_000) if rows[-1]["total_steps"] >= 10_000_000 else None,
        "at_20m": window(rows, 20_000_000) if rows[-1]["total_steps"] >= 20_000_000 else None,
    }
    legacy.append(record)

result = {
    "scope": "Existing copied local training metrics only; no remote status or frozen-policy evaluation.",
    "estimator": "Last 50 complete epochs ending at or before each budget; episode-weighted within seed, equal weight for each of 3 training seeds. Latest windows have unequal budgets and are descriptive only. Seed ranges are not confidence intervals.",
    "provenance_limit": "SoftRole appended stdout retains the audited configuration/source-prefix linkage but has no new corresponding checkpoint/source archive. Legacy exact steps stop at archived metrics; later stdout cumulative counters are invalid.",
    "budget_arithmetic": {
        "one_collector_2000_epochs": {"collectors": 1, "batch_floor_per_collector": 500, "updates_per_epoch": 10, "epochs": 2000, "optimizer_updates": 20000, "minimum_environment_steps": 10000000, "complete_episode_upper_bound_at_horizon80": 11580000},
        "four_collectors_2000_epochs": {"collectors": 4, "batch_floor_per_collector": 500, "updates_per_epoch": 10, "epochs": 2000, "optimizer_updates": 20000, "minimum_environment_steps": 40000000, "complete_episode_upper_bound_at_horizon80": 46320000},
        "interpretation": "10M is a lower bound implied by June release commands, not an established paper budget or evidence of optimal stopping. Four collectors multiply samples per update, not update count. Modern sample-matched and update-matched comparisons differ.",
    },
    "softrole_runs": runs, "softrole_groups": groups,
    "legacy_exact_runs": legacy,
    "legacy_pcp_at_10m": group([r["at_10m"] for r in legacy if r["task"] == "pcp"]),
    "inputs": INPUTS,
}
for relative, identity in INPUTS.items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == identity["sha256"]
(OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
with (OUT / "windows.csv").open("w", newline="") as stream:
    fields = ["task", "model", "seed", "window", "requested_budget", "steps_start", "steps_end", "steps", "episodes", "updates_at_endpoint", *METRICS]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for run in runs:
        for name in ("at_10m", "at_20m", "latest", "latest_common_task_budget"):
            writer.writerow({key: run[key] for key in ("task", "model", "seed")} | {"window": name} | run[name])
print(json.dumps({"softrole_groups": groups, "legacy_pcp_at_10m": result["legacy_pcp_at_10m"], "input_count": len(INPUTS)}, indent=2))
