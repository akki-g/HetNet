"""Independent, read-only audit of copied reproduction metadata and epoch records."""
from pathlib import Path
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "stokes_runs/runs/reproduction-fast"
OUT = Path(__file__).resolve().parent


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary(rows):
    n = sum(row["episodes"] for row in rows)
    result = {"epoch_start": rows[0]["epoch"], "epoch_end": rows[-1]["epoch"],
              "steps_start_exclusive": rows[0]["total_steps"] - rows[0]["steps"],
              "steps_end": rows[-1]["total_steps"], "episodes": n}
    for key in ("success_rate", "steps_taken", "policy_loss", "value_loss", "team_return"):
        result[key + "_epoch_mean"] = statistics.mean(row[key] for row in rows)
        # Policy/value entries are diagnostic means of the recorded quantity,
        # not reconstructed per-episode objective losses.
        result[key + "_episode_weighted"] = sum(row["episodes"] * row[key] for row in rows) / n
    return result


runs = []
for folder in sorted(BASE.glob("*/*")):
    if not folder.is_dir():
        continue
    args = json.loads((folder / "resolved_args.json").read_text())
    rows = read_records(folder / "metrics.jsonl")
    initial = json.loads((folder / "initial_signature.json").read_text())
    signatures = read_records(folder / "epoch_signatures.jsonl")
    signature_map = {row["epoch"]: row["signature"]["sha256"] for row in signatures}
    assert [row["epoch"] for row in rows] == list(range(1, len(rows) + 1))
    assert [row["epoch"] for row in signatures] == list(range(1, len(rows) + 1))
    steps = episodes = 0
    for row in rows:
        steps += row["steps"]
        episodes += row["episodes"]
        row["team_return"] = sum(row["reward_per_agent"])
        assert steps == row["total_steps"] and episodes == row["total_episodes"]
        assert abs(row["steps_taken"] * row["episodes"] - row["steps"]) < 1e-7
        assert abs(row["success_rate"] * row["episodes"] - round(row["success_rate"] * row["episodes"])) < 1e-7
        assert 0 <= row["success_rate"] <= 1
        assert all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
    checkpoints = read_records(folder / "checkpoint_records.jsonl")
    for item in checkpoints:
        checkpoint = next(folder.glob("checkpoints/*/run1/" + Path(item["path"]).name))
        sidecar = json.loads(Path(str(checkpoint) + ".signature.json").read_text())
        assert checkpoint.stat().st_size == item["bytes"]
        assert item["parameter_sha256"] == sidecar["sha256"] == signature_map[item["epoch"]]
    assert len(checkpoints) == len(list(folder.glob("checkpoints/*/run1/*.pt")))
    task = folder.parent.name.split("_")[0]
    old_index = {"pp": 0, "pcp": 3, "fc": 6}[task] + args["seed"]
    old_text = (ROOT / f"logs_1/reproduce-896848_{old_index}.out").read_text()
    new_text = (folder / "stdout.log").read_text()
    # Compare completed epoch blocks, stripping neither numbers nor time.
    pattern = re.compile(r"^Epoch (\d+)\t.*?^Steps-taken: [^\n]+", re.M | re.S)
    old_blocks = {int(match[1]): match[0] for match in pattern.finditer(old_text)}
    new_blocks = {int(match[1]): match[0] for match in pattern.finditer(new_text)}
    assert all(new_blocks[epoch] == block for epoch, block in old_blocks.items())
    printed = re.search(r"Episodes (\d+), Total Steps (\d+)", new_blocks[len(rows)])
    last_params = {item["name"]: item for item in signatures[-1]["signature"]["parameters"]}
    same = [item["name"] for item in initial["parameters"]
            if item["sha256"] == last_params[item["name"]]["sha256"]]
    commits = (folder / "environment.txt").read_text().splitlines()
    budgets = {}
    for budget in (4_000_000, 4_500_000, 5_000_000, 5_500_000, 6_000_000):
        subset = [row for row in rows if row["total_steps"] <= budget]
        if rows[-1]["total_steps"] >= budget and len(subset) >= 50:
            budgets[str(budget)] = summary(subset[-50:])
    runs.append({"path": str(folder.relative_to(ROOT)), "task": task, "seed": args["seed"],
                 "source_commit": commits[2], "source_patch_bytes": (folder / "source.patch").stat().st_size,
                 "recorded_slurm": commits[-1], "epochs": len(rows), "target_epochs": args["num_epochs"],
                 "steps": steps, "episodes": episodes, "old_completed_epochs": len(old_blocks),
                 "old_completed_blocks_match": True, "first50": summary(rows[:50]),
                 "latest50": summary(rows[-50:]), "previous50": summary(rows[-100:-50]),
                 "near_step_budgets": budgets, "epoch_wall_hours": sum(row["wall_time_seconds"] for row in rows) / 3600,
                 "checkpoints": len(checkpoints), "checkpoint_last_epoch": checkpoints[-1]["epoch"],
                 "checkpoint_sidecars_match_epochs_and_sizes": True,
                 "signature_changes": len({initial["sha256"], *signature_map.values()}) - 1,
                 "unchanged_parameter_names": same,
                 "printed_over_true_steps": int(printed[2]) / steps,
                 "printed_over_true_episodes": int(printed[1]) / episodes})

result = {"runs": runs, "totals": {"epochs": sum(run["epochs"] for run in runs),
           "steps": sum(run["steps"] for run in runs), "episodes": sum(run["episodes"] for run in runs),
           "checkpoints": sum(run["checkpoints"] for run in runs)}}
ancillary = []
for group in ("reproduction", "speed-check"):
    base = ROOT / "stokes_runs/runs" / group
    for path in sorted(base.rglob("metrics.jsonl")):
        corresponding = BASE / path.relative_to(base)
        old = read_records(path)
        primary = read_records(corresponding)[:len(old)]
        same_metrics = all({k: v for k, v in a.items() if k != "wall_time_seconds"}
                           == {k: v for k, v in b.items() if k != "wall_time_seconds"}
                           for a, b in zip(old, primary))
        old_signatures = read_records(path.parent / "epoch_signatures.jsonl")
        primary_signatures = read_records(corresponding.parent / "epoch_signatures.jsonl")[:len(old)]
        same_signatures = old_signatures == primary_signatures
        assert same_metrics and same_signatures
        ancillary.append({"path": str(path.parent.relative_to(ROOT)), "epochs": len(old),
                          "metrics_match_primary_prefix_except_wall_time": same_metrics,
                          "all_epoch_signatures_match_primary_prefix": same_signatures,
                          "source_commit": (path.parent / "environment.txt").read_text().splitlines()[2]})
result["ancillary_prefixes"] = ancillary
(OUT / "findings.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["totals"], indent=2))
for run in runs:
    print(run["task"], run["seed"], run["epochs"], run["steps"],
          {key: round(run["latest50"][key + "_episode_weighted"], 6)
           for key in ("success_rate", "team_return", "steps_taken")})
