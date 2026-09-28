"""Read-only Run 1 progress from fresh-batch JSONL; never imports the learner.

Training curves are exploratory diagnostics, not frozen-policy results or a
statistical learning test. No epoch-zero measurement is recorded by training.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
import time

from hetnet_ext.grid import ROOT, load_grid, run_directory


def read_jsonl(path):
    """Ignore an in-flight final line; reject corruption in committed lines."""
    if not path.exists():
        return [], []
    raw = path.read_bytes()
    warnings = []
    if raw and not raw.endswith(b"\n"):
        raw = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
        warnings.append(f"{path.name}: ignored unfinished final line; poll again")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name}: each committed line must be a JSON object")
    return rows, warnings


def validate_metrics(rows, n_agents, expected_epochs):
    total_steps = total_episodes = 0
    for epoch, row in enumerate(rows, 1):
        if row.get("epoch") != epoch or epoch > expected_epochs:
            raise ValueError("epochs are not contiguous 1..N within the recipe")
        for field in ("steps", "episodes"):
            if type(row.get(field)) is not int or row[field] <= 0:
                raise ValueError(f"invalid {field} at epoch {epoch}")
        reward = row["reward_per_agent"]
        if not isinstance(reward, list) or len(reward) != n_agents:
            raise ValueError(f"wrong reward vector length at epoch {epoch}")
        for value in [row[k] for k in ("wall_time_seconds", "success_rate", "steps_taken",
                                      "policy_loss", "value_loss")] + reward:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"non-finite/non-numeric metric at epoch {epoch}")
        if not 0 <= row["success_rate"] <= 1 or not 0 < row["steps_taken"] <= 80:
            raise ValueError(f"success/length outside PCP recipe at epoch {epoch}")
        if row["wall_time_seconds"] <= 0:
            raise ValueError(f"nonpositive epoch time at epoch {epoch}")
        total_steps += row["steps"]
        total_episodes += row["episodes"]
        if row["total_steps"] != total_steps or row["total_episodes"] != total_episodes:
            raise ValueError(f"inconsistent fresh-batch cumulative counters at epoch {epoch}")


def window_metrics(rows, n_p):
    if not rows:
        return None
    episodes = sum(r["episodes"] for r in rows)
    mean = lambda f: sum(f(r) * r["episodes"] for r in rows) / episodes
    return {"first_epoch": rows[0]["epoch"], "last_epoch": rows[-1]["epoch"],
            "episodes": episodes,
            "success_rate": mean(lambda r: r["success_rate"]),
            "steps_taken": mean(lambda r: r["steps_taken"]),
            "reward_P": mean(lambda r: statistics.mean(r["reward_per_agent"][:n_p])),
            "reward_A": mean(lambda r: statistics.mean(r["reward_per_agent"][n_p:]))}


def inspect_run(entry, root, window, now):
    path = run_directory(entry, "train", root)
    result = {"composition": entry.composition, "seed": entry.seed, "index": entry.index,
              "path": str(path), "status": "not_seen", "epoch": 0, "expected_epochs": 2000,
              "warnings": [], "recent": None, "early_review": None, "eta_hours": None}
    rows = []
    if not path.exists():
        return result, rows
    result["status"] = "incomplete_check_slurm"
    try:
        config_path = path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if not isinstance(config, dict):
                raise ValueError("config must be a JSON object")
            if config["entry"] != entry.as_dict() or config.get("num_epochs") != 2000 or config.get("mode") != "train":
                raise ValueError("config does not identify this full Run 1 grid entry")
        else:
            result["warnings"].append("config missing; run identity unverified")
        provenance = {}
        if (path / "provenance.json").exists():
            try:
                provenance = json.loads((path / "provenance.json").read_text())
                if not isinstance(provenance, dict):
                    raise ValueError("provenance must be a JSON object")
            except json.JSONDecodeError:
                result["warnings"].append("provenance being written or corrupt; poll again and check Slurm")
        if provenance.get("status") in ("complete", "failed"):
            result["status"] = provenance["status"]
        if provenance.get("error"):
            result["warnings"].append(provenance["error"])
        rows, warnings = read_jsonl(path / "metrics.jsonl")
        result["warnings"].extend(warnings)
        validate_metrics(rows, entry.nfriendly_P + entry.nfriendly_A, 2000)
        if rows:
            latest = rows[-1]
            result.update(epoch=latest["epoch"], total_steps=latest["total_steps"],
                          total_episodes=latest["total_episodes"], policy_loss=latest["policy_loss"],
                          value_loss=latest["value_loss"], first=window_metrics(rows[:window], entry.nfriendly_P),
                          recent=window_metrics(rows[-window:], entry.nfriendly_P))
            timing = rows[-window:]
            p50 = statistics.median(r["wall_time_seconds"] for r in timing)
            result["seconds_per_epoch_median"] = p50
            result["joint_steps_per_second"] = sum(r["steps"] for r in timing) / sum(r["wall_time_seconds"] for r in timing)
            result["metrics_age_seconds"] = max(0, now - (path / "metrics.jsonl").stat().st_mtime)
            if result["status"] == "incomplete_check_slurm" and result["metrics_age_seconds"] > max(600, 5 * p50):
                result["warnings"].append("metrics stale: check squeue/sacct; this alone does not establish a stalled job")
            checkpoints, warnings = read_jsonl(path / "checkpoint_records.jsonl")
            result["warnings"].extend(warnings)
            result["last_checkpoint_epoch"] = max((r["epoch"] for r in checkpoints), default=None)
            saves = [r["wall_time_seconds"] for r in checkpoints]
            if any(not math.isfinite(x) or x < 0 for x in saves):
                raise ValueError("invalid checkpoint timing")
            checkpoint_seconds = statistics.median(saves) if saves else 0
            remaining_saves = sum(e > latest["epoch"] for e in range(50, 2001, 50))
            if result["status"] != "failed":
                result["eta_hours"] = ((2000 - latest["epoch"]) * p50 + remaining_saves * checkpoint_seconds) / 3600
            if entry.composition == "2P1A" and latest["epoch"] >= 300:
                first = window_metrics(rows[:50], entry.nfriendly_P)
                review = window_metrics(rows[250:300], entry.nfriendly_P)
                delta = review["success_rate"] - first["success_rate"]
                result["early_review"] = {"status": "review_due", "first_50": first, "epochs_251_300": review,
                                          "success_change": delta, "no_success_increase": delta <= 0}
        if result["status"] == "complete" and result["epoch"] != 2000:
            raise ValueError("completion provenance conflicts with epoch records")
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as exc:
        result["status"] = "invalid_evidence"
        result["warnings"].append(str(exc))
        rows = []
    return result, rows


def snapshot(root, window=50, now=None):
    if window < 1:
        raise ValueError("window must be positive")
    records, series = [], []
    for entry in load_grid():
        record, rows = inspect_run(entry, root, window, time.time() if now is None else now)
        records.append(record)
        series.append((entry, rows))
    review = [r for r in records if r["early_review"]]
    return {"schema_version": 1, "created_at_unix": time.time() if now is None else now,
            "runs_root": str(Path(root).resolve()), "window_epochs": window,
            "expected_runs": 21, "complete_runs": sum(r["status"] == "complete" for r in records),
            "source_seeds_at_review": len(review),
            "source_review_due": len(review) >= 2,
            "source_review_note": "Compare every available source seed. Fixed windows 1–50 and 251–300; no epoch-zero baseline. No automatic stopping or statistical learning claim.",
            "eta_note": "Per-run extrapolation from recent median epoch time plus observed checkpoint cost; excludes queues, startup, repeated recorder/signature overhead and unknown future slowdown. Before first save, checkpoint cost is omitted. Use calibrated budget for allocation sizing.",
            "runs": records}, series


def render_table(report):
    lines = ["Run 1: %s/21 complete; missing rows are NOT known to be queued." % report["complete_runs"],
             "team seed epoch status                     success steps    rP     rA   sec/ep ETA(h)"]
    for r in report["runs"]:
        recent = r["recent"]
        values = (f'{recent["success_rate"]:7.3f} {recent["steps_taken"]:5.1f} {recent["reward_P"]:6.2f} {recent["reward_A"]:6.2f}'
                  if recent else "      -     -      -      -")
        timing = f'{r["seconds_per_epoch_median"]:7.1f}' if "seconds_per_epoch_median" in r else "      -"
        eta = f'{r["eta_hours"]:6.1f}' if r["eta_hours"] is not None else "     -"
        lines.append(f'{r["composition"]:4} {r["seed"]:4} {r["epoch"]:5} {r["status"]:26} {values} {timing} {eta}')
        for warning in r["warnings"]:
            lines.append(f'  WARNING {r["composition"]}/seed{r["seed"]}: {warning}')
        if r["early_review"]:
            delta = r["early_review"]["success_change"]
            lines.append(f'  REVIEW epoch 300: success change {delta:+.4f} (epochs 251–300 minus 1–50)')
    lines.extend([report["eta_note"], report["source_review_note"]])
    if report["source_review_due"]:
        lines.append("SOURCE LEARNING REVIEW DUE: inspect success, steps and class rewards across seeds before further compute; flat success requires owner review.")
    return "\n".join(lines)


def curves_svg(series, window):
    """Standalone diagnostic plot with every observed seed; no pooled seed CI."""
    colors = ["#2563eb", "#dc2626", "#15803d", "#9333ea", "#c2410c"]
    groups = list(dict.fromkeys(e.composition for e, _ in series))
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="920" viewBox="0 0 1100 920">',
             '<rect width="1100" height="920" fill="white"/>',
             '<g font-family="sans-serif" font-size="12" fill="#111827">',
             f'<text x="50" y="24">Run 1 training diagnostics: episode-weighted blocks of {window} epochs; all seeds; no frozen evaluation</text>']
    blocks = [(e, [window_metrics(rows[i:i + window], e.nfriendly_P) for i in range(0, len(rows), window)]) for e, rows in series]
    for index, key in enumerate(("success_rate", "steps_taken", "reward_P", "reward_A")):
        x0, y0, width, height = 65 + (index % 2) * 525, 75 + (index // 2) * 315, 440, 240
        values = [p[key] for _, points in blocks for p in points]
        lo, hi = (0, 1) if key == "success_rate" else ((0, 80) if key == "steps_taken" else ((min(values), max(values)) if values else (0, 1)))
        if hi == lo:
            lo, hi = lo - 1, hi + 1
        lines.append(f'<text x="{x0}" y="{y0 - 16}">{key}</text>')
        for fraction in (0, .5, 1):
            y = y0 + height * (1 - fraction)
            lines.append(f'<path d="M{x0},{y}h{width}" stroke="#d1d5db"/><text x="{x0 - 5}" y="{y + 4}" text-anchor="end">{lo + fraction * (hi - lo):.2f}</text>')
        for epoch in (0, 1000, 2000):
            x = x0 + width * epoch / 2000
            lines.append(f'<text x="{x}" y="{y0 + height + 20}" text-anchor="middle">{epoch}</text>')
        for e, points in blocks:
            if not points:
                continue
            coords = " ".join(f'{x0 + width * p["last_epoch"] / 2000:.2f},{y0 + height * (1 - (p[key] - lo) / (hi - lo)):.2f}' for p in points)
            dash = ("none", "8 3", "3 3", "10 3 2 3", "1 3")[e.seed]
            lines.append(f'<polyline points="{coords}" fill="none" stroke="{colors[groups.index(e.composition)]}" stroke-width="1.6" stroke-dasharray="{dash}"><title>{html.escape(e.composition)} seed {e.seed}</title></polyline>')
            # Single blocks otherwise have no visible line.
            if len(points) == 1:
                cx, cy = coords.split(",")
                lines.append(f'<circle cx="{cx}" cy="{cy}" r="2" fill="{colors[groups.index(e.composition)]}"/>')
    for i, (e, _) in enumerate(series):
        x, y = 55 + (i % 5) * 210, 725 + (i // 5) * 30
        dash = ("none", "8 3", "3 3", "10 3 2 3", "1 3")[e.seed]
        lines.append(f'<path d="M{x},{y}h35" stroke="{colors[groups.index(e.composition)]}" stroke-dasharray="{dash}"/><text x="{x + 42}" y="{y + 4}">{e.composition} seed {e.seed}</text>')
    lines.append('<text x="50" y="900">x-axis: completed epochs. Missing/failed runs remain in the companion report; curves are not evidence of checkpoint validity.</text></g></svg>')
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/run1_train")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="Print full report as JSON")
    parser.add_argument("--out", type=Path, help="New snapshot directory: JSON, CSV, text and SVG curves")
    args = parser.parse_args(argv)
    if args.window < 1:
        parser.error("--window must be positive")
    report, series = snapshot(args.runs, args.window)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (args.out / "progress.txt").write_text(render_table(report) + "\n")
        (args.out / "learning_curves.svg").write_text(curves_svg(series, args.window))
        with (args.out / "summary.csv").open("w", newline="") as stream:
            fields = ["composition", "seed", "epoch", "status", "success_rate", "steps_taken", "reward_P", "reward_A", "seconds_per_epoch_median", "eta_hours"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in report["runs"]:
                merged = dict(row, **(row["recent"] or {}))
                writer.writerow({field: merged.get(field) for field in fields})
    print(json.dumps(report, indent=2, allow_nan=False) if args.json else render_table(report))
    return int(any(r["status"] in ("failed", "invalid_evidence") for r in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
