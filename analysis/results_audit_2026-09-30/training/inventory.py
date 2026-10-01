"""Independently inventory copied training stdout and recorded configurations.

Reads only source evidence; writes only adjacent inventory.json and progress.csv.
No policy, environment, training or cluster execution occurs.
"""
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASKS = ("pp", "pcp", "fc")
MODELS = ("HetNet Real", "SoftRole shared", "SoftRole banked")
HEADER = re.compile(r"^Epoch (\d+)\s+Reward \[([^]]+)\].*Episodes (\d+), Total Steps (\d+),")
inputs = {}


def read(path):
    blob = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {
        "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()
    }
    return blob.decode()


def avg(rows, key, weighted):
    weights = [row["episodes"] if weighted else 1 for row in rows]
    return math.fsum(row[key] * weight for row, weight in zip(rows, weights)) / sum(weights)


runs = []
previous = json.loads(read(ROOT / "analysis/project_overview_2026-09-30/latest_snapshot/summary.json"))
for path in sorted([*ROOT.glob("logs_1/*.out"), *ROOT.glob("logs_sr/*.out")]):
    text = read(path)
    sr = path.parent.name == "logs_sr"
    if sr:
        index = int(path.stem.rsplit("_", 1)[1])
        task, model, seed = TASKS[index // 6], MODELS[1 + (index // 3) % 2], index % 3
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        raw_config = None
    else:
        call = ast.parse(next(line for line in text.splitlines() if line.startswith("Namespace(")), mode="eval").body
        raw_config = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        task = "fc" if raw_config["env_name"] == "fire_commander" else ("pcp" if raw_config["nfriendly_A"] else "pp")
        model, seed = MODELS[0], raw_config["seed"]
        assert not raw_config["use_binary"]
        rows, row = [], None
        for line in text.splitlines():
            match = HEADER.match(line)
            if match:
                assert row is None
                epoch, reward, episodes, steps = match.groups()
                row = {"epoch": int(epoch), "reward_per_agent": [float(x) for x in reward.split()],
                       "overcounted_stdout_steps": int(steps), "overcounted_stdout_episodes": int(episodes)}
                row["team_return"] = sum(row["reward_per_agent"])
            elif row is not None and line.startswith("Success: "):
                row["success_rate"] = float(line.split(": ")[1])
            elif row is not None and line.startswith("Steps-taken: "):
                row["steps_taken"] = float(line.split(": ")[1])
                rows.append(row)
                row = None
        assert row is None
    directory = ROOT / "stokes_runs/runs" / ("softrole_primary" if sr else "reproduction-fast") / (task + "_" + (model.split()[-1] if sr else "real")) / f"seed{seed}"
    config_path = directory / ("config.json" if sr else "resolved_args.json")
    config = json.loads(read(config_path))
    archived = [json.loads(line) for line in read(directory / "metrics.jsonl").splitlines()]
    assert len(rows) >= len(archived)
    assert [r["epoch"] for r in rows] == list(range(1, len(rows) + 1))
    for fresh, old in zip(rows, archived):
        if sr:
            assert fresh == old
        else:
            for key in ("success_rate", "steps_taken"):
                assert abs(fresh[key] - old[key]) <= .005000001
            assert all(abs(x-y) <= .005000001 for x, y in zip(fresh["reward_per_agent"], old["reward_per_agent"]))
    if sr:
        assert (config["task"], config["model"], config["seed"]) == (task, model.split()[-1], seed)
        assert config["total_steps"] is None and config["failure_prob"] == 0 and config["compositions"] == []
        steps = episodes = 0
        for row in rows:
            steps += row["steps"]
            episodes += row["episodes"]
            assert (row["total_steps"], row["total_episodes"]) == (steps, episodes)
            assert abs(row["steps_taken"] * row["episodes"] - row["steps"]) < 1e-7
            assert row["updates"] == row["epoch"] * config["updates_per_epoch"] and not row["partial_epoch"]
            assert row["event_exposed_episodes"] == 0
    else:
        assert all(raw_config[k] == config[k] for k in raw_config if k in config)
    epochs = config["epochs"] if sr else config["num_epochs"]
    updates = config["updates_per_epoch"] if sr else config["epoch_size"]
    batch = config["batch_steps"] if sr else config["batch_size"]
    collectors, horizon = config["nprocesses"], config["max_steps"]
    old_summary = next(r for r in previous["runs"] if (r["task"], r["model"], r["seed"]) == (task, model, seed))
    metrics = {key: avg(rows[-50:], key, sr) for key in ("success_rate", "steps_taken", "team_return")}
    for key, value in metrics.items():
        assert abs(value - old_summary["latest"][key]) < 1e-10
    assert len(rows) == old_summary["epochs"]
    runs.append({"task": task, "model": model, "seed": seed, "source": str(path.relative_to(ROOT)),
                 "config_source": str(config_path.relative_to(ROOT)), "config": config,
                 "completed_epochs": len(rows), "epoch_cap": epochs, "completed_updates": len(rows) * updates,
                 "episode_horizon": horizon, "updates_per_epoch": updates, "collectors": collectors,
                 "batch_floor_per_collector": batch, "global_training_step_cap": None,
                 "steps_per_epoch_lower_bound": updates * collectors * batch,
                 "steps_per_epoch_upper_bound": updates * collectors * (batch + horizon - 1),
                 "full_epoch_budget_steps_lower_bound": epochs * updates * collectors * batch,
                 "full_epoch_budget_steps_upper_bound": epochs * updates * collectors * (batch + horizon - 1),
                 "recorded_total_steps": rows[-1]["total_steps"] if sr else None,
                 "recorded_total_episodes": rows[-1]["total_episodes"] if sr else None,
                 "archived_epochs": len(archived), "archived_total_steps": archived[-1]["total_steps"],
                 "latest_window_first_epoch": rows[-50]["epoch"], "latest_window_last_epoch": rows[-1]["epoch"],
                 "latest_window_weighting": "episodes" if sr else "equal epochs of rounded stdout",
                 **{"latest_" + key: value for key, value in metrics.items()}})

runs.sort(key=lambda r: (TASKS.index(r["task"]), MODELS.index(r["model"]), r["seed"]))
assert len(runs) == 27
for relative, identity in inputs.items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == identity["sha256"]
result = {"scope": "Copied local logs only; no remote scheduler status. New tails have no new checkpoint/source archive.",
          "legacy_counter_limitation": "stdout adds cumulative-within-epoch counts after every update; no exact new sample totals beyond archived metrics.",
          "runs": runs, "inputs": inputs, "completed_epochs": sum(r["completed_epochs"] for r in runs),
          "softrole_steps": sum(r["recorded_total_steps"] or 0 for r in runs),
          "softrole_episodes": sum(r["recorded_total_episodes"] or 0 for r in runs),
          "at_epoch_cap": sum(r["completed_epochs"] == r["epoch_cap"] for r in runs)}
(HERE / "inventory.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
with (HERE / "progress.csv").open("w", newline="") as stream:
    fields = [key for key in runs[0] if key != "config"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(runs)
print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "inputs")}, indent=2))
for task in TASKS:
    for model in MODELS:
        selected = [r for r in runs if r["task"] == task and r["model"] == model]
        print(task, model, "epochs", [r["completed_epochs"] for r in selected],
              "steps", [r["recorded_total_steps"] for r in selected],
              "latest length", sum(r["latest_steps_taken"] for r in selected) / 3,
              "latest success", sum(r["latest_success_rate"] for r in selected) / 3)
