"""Read-only training progress: python -m hetnet_ext.progress --runs runs/reproduction.

Success, episode length and class rewards are episode-weighted over the window;
losses retain the recorder's joint-step denominator. No learner is imported.
Complete means exit zero and target epochs reached; checkpoints are not verified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def read_jsonl(path):
    """Ignore an unfinished final line; reject malformed committed records."""
    raw = Path(path).read_bytes()
    warnings = []
    if raw and not raw.endswith(b"\n"):
        raw = raw.rsplit(b"\n", 1)[0] if b"\n" in raw else b""
        warnings.append("Ignored unfinished final metrics line; poll again.")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every committed metrics line must be a JSON object")
    return rows, warnings


def validate_metrics(rows, args):
    for key in ("num_epochs", "max_steps", "nfriendly_P", "nfriendly_A"):
        minimum = 0 if key.startswith("nfriendly_") else 1
        if type(args.get(key)) is not int or args[key] < minimum:
            raise ValueError(f"Invalid resolved argument: {key}")
    n_agents = args["nfriendly_P"] + args["nfriendly_A"]
    if not n_agents:
        raise ValueError("No agents in resolved arguments")
    total_steps = total_episodes = 0
    for epoch, row in enumerate(rows, 1):
        if row.get("epoch") != epoch or epoch > args["num_epochs"]:
            raise ValueError("Epochs must be contiguous 1..N within num_epochs")
        for field in ("steps", "episodes"):
            if type(row.get(field)) is not int or row[field] <= 0:
                raise ValueError(f"Invalid {field} at epoch {epoch}")
        reward = row["reward_per_agent"]
        if not isinstance(reward, list) or len(reward) != n_agents:
            raise ValueError(f"Wrong reward vector length at epoch {epoch}")
        numbers = [row[k] for k in ("wall_time_seconds", "success_rate", "steps_taken",
                                   "policy_loss", "value_loss")] + reward
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in numbers):
            raise ValueError(f"Non-finite/non-numeric metric at epoch {epoch}")
        if not 0 <= row["success_rate"] <= 1 or not 0 < row["steps_taken"] <= args["max_steps"]:
            raise ValueError(f"Success or episode length outside task bounds at epoch {epoch}")
        if row["wall_time_seconds"] <= 0:
            raise ValueError(f"Nonpositive epoch time at epoch {epoch}")
        total_steps += row["steps"]
        total_episodes += row["episodes"]
        if row["total_steps"] != total_steps or row["total_episodes"] != total_episodes:
            raise ValueError(f"Inconsistent cumulative counters at epoch {epoch}")


def window_metrics(rows, n_p):
    if not rows:
        return None
    episodes = sum(r["episodes"] for r in rows)
    steps = sum(r["steps"] for r in rows)
    mean = lambda f: sum(f(r) * r["episodes"] for r in rows) / episodes
    result = {"first_epoch": rows[0]["epoch"], "last_epoch": rows[-1]["epoch"],
              "episodes": episodes, "success_rate": mean(lambda r: r["success_rate"]),
              "steps_taken": mean(lambda r: r["steps_taken"]),
              "reward_P": mean(lambda r: statistics.mean(r["reward_per_agent"][:n_p])) if n_p else None,
              "reward_A": mean(lambda r: statistics.mean(r["reward_per_agent"][n_p:]))
              if n_p < len(rows[0]["reward_per_agent"]) else None}
    result.update({k: sum(r[k] * r["steps"] for r in rows) / steps for k in ("policy_loss", "value_loss")})
    return result


def inspect_run(path, root, window):
    result = {"run": str(path.relative_to(root)), "path": str(path),
              "status": "running_or_interrupted", "epoch": 0, "target_epochs": None,
              "total_steps": 0, "total_episodes": 0, "seconds_per_epoch": None,
              "recent": None, "warnings": []}
    try:
        exit_file = path / "exit_code.txt"
        if exit_file.exists():
            result["exit_code"] = int(exit_file.read_text().strip())
            result["status"] = "failed" if result["exit_code"] else "incomplete"
        missing = [name for name in ("metrics.jsonl", "resolved_args.json") if not (path / name).exists()]
        if missing:
            result["warnings"].append(f'Missing {", ".join(missing)}; check stdout.log for startup errors.')
            return result
        args = json.loads((path / "resolved_args.json").read_text())
        if not isinstance(args, dict):
            raise ValueError("resolved_args.json must contain a JSON object")
        rows, warnings = read_jsonl(path / "metrics.jsonl")
        result["warnings"].extend(warnings)
        validate_metrics(rows, args)
        result["target_epochs"] = args["num_epochs"]
        if rows:
            result.update({k: rows[-1][k] for k in ("epoch", "total_steps", "total_episodes")})
            result["recent"] = window_metrics(rows[-window:], args["nfriendly_P"])
            result["seconds_per_epoch"] = statistics.median(r["wall_time_seconds"] for r in rows[-window:])
        if result.get("exit_code") == 0 and result["epoch"] == result["target_epochs"] and not warnings:
            result["status"] = "complete"
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as exc:
        if result["status"] != "failed":
            result["status"] = "invalid"
        result["warnings"].append(str(exc))
    return result


def snapshot(root, window=50):
    if window < 1:
        raise ValueError("window must be positive")
    root = Path(root).resolve()
    directories = {path.parent for name in ("command.txt", "metrics.jsonl") for path in root.rglob(name)}
    runs = [inspect_run(path, root, window) for path in sorted(directories)]
    return {"runs_root": str(root), "window_epochs": window,
            "complete_runs": sum(r["status"] == "complete" for r in runs), "runs": runs}


def render_table(report):
    number = lambda x: "-" if x is None else f"{x:.3g}"
    lines = [f'{report["complete_runs"]}/{len(report["runs"])} discovered runs complete; recent window: {report["window_epochs"]} epochs.',
             "run                      epoch/target  status                  success length    rP    rA joint_steps sec/epoch policy_loss value_loss"]
    for run in report["runs"]:
        recent = run["recent"] or {}
        values = " ".join(f"{number(recent.get(k)):>6}" for k in ("success_rate", "steps_taken", "reward_P", "reward_A"))
        losses = " ".join(number(recent.get(k)) for k in ("policy_loss", "value_loss"))
        lines.append(f'{run["run"]:24} {run["epoch"]}/{run["target_epochs"] or "?":<6} '
                     f'{run["status"]:23} {values} {run["total_steps"]:11} {number(run["seconds_per_epoch"]):>9} {losses}')
        lines.extend(f'  {run["run"]}: {warning}' for warning in run["warnings"])
    lines.append("Absent exit_code.txt cannot distinguish an active job from an interruption; check squeue/sacct.")
    lines.append("Complete means exit zero and target epochs reached; checkpoints are not verified.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/reproduction"))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the table")
    parser.add_argument("--out", type=Path, help="Write summary.csv in this directory")
    args = parser.parse_args(argv)
    if args.window < 1:
        parser.error("--window must be positive")
    report = snapshot(args.runs, args.window)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        fields = ["run", "epoch", "target_epochs", "status", "total_steps", "total_episodes", "seconds_per_epoch",
                  "success_rate", "steps_taken", "reward_P", "reward_A", "policy_loss", "value_loss", "warnings"]
        with (args.out / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for run in report["runs"]:
                writer.writerow({**run, **(run["recent"] or {}), "warnings": "; ".join(run["warnings"])})
    print(json.dumps(report, indent=2, allow_nan=False) if args.json else render_table(report))
    return 1 if any(r["status"] == "invalid" for r in report["runs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
