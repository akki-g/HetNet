"""Recompute only identifiable quantities from the copied frozen-evaluation stdout.

No training, checkpoint loading, outcome selection or hidden composition inference.
Run from any directory; input snapshots/hashes retain the evidence used by the PDF.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def analyze():
    records, inputs = [], {}
    snapshots = OUT / "inputs"
    snapshots.mkdir(exist_ok=True)
    def capture(path):
        data = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        inputs[relative] = {"sha256": sha(data), "bytes": len(data), "lines": len(data.splitlines())}
        copy = snapshots / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        if copy.exists() and copy.read_bytes() != data:
            raise ValueError(f"Refusing changed input snapshot: {relative}")
        copy.write_bytes(data)
        return data
    for index in range(18):
        path = ROOT / f"logs_sr/pcp-frozen-902506_{index}.out"
        row = json.loads(capture(path))
        stderr = capture(path.with_suffix(".err")).decode()
        assert not any(word in stderr for word in ("Traceback", "srun: error", "CANCELLED"))
        model = "shared" if index < 9 else "banked"
        seed, condition = (index % 9) // 3, ("nominal", "failure", "sham")[index % 3]
        assert f"pcp_{model}/seed{seed}/" in row["checkpoint"]
        assert row["training_seed"] == seed and row["sham"] == (condition == "sham")
        assert row["episodes"] == (2500 if condition == "nominal" else 100)
        successes = row["episodes"] * row["success_rate"]
        assert abs(successes - round(successes)) < 1e-8
        records.append({"index": index, "model": model, "condition": condition,
                        "successes": round(successes), "failures": row["episodes"] - round(successes),
                        "stdout": str(path.relative_to(ROOT)), **row})
    for name in ("pcp-prepare-902504.out", "pcp-prepare-902504.err"):
        capture(ROOT / "logs_sr" / name)
    preflight = []
    for index in range(4):
        error = ROOT / f"logs_1/hetnet-preflight-902621_{index}.err"
        content = capture(error).decode()
        stdout = capture(error.with_suffix(".out"))
        assert not stdout and "Runtime file inventory differs from ORIGINS.json" in content
        preflight.append({"index": index, "error": str(error.relative_to(ROOT)),
                          "stdout_bytes": len(stdout), "reached_training": False})
    aggregate = {}
    for model in ("shared", "banked"):
        selected = [r for r in records if r["model"] == model and r["condition"] == "nominal"]
        aggregate[model] = {"training_seeds": 3, "episodes": sum(r["episodes"] for r in selected),
                            "failures": sum(r["failures"] for r in selected), "metrics": {}}
        for metric in ("success_rate", "mean_team_return", "mean_agent_return"):
            values = [r[metric] for r in selected]
            aggregate[model]["metrics"][metric] = {"equal_seed_mean": statistics.mean(values),
                "min": min(values), "max": max(values), "sample_sd": statistics.stdev(values),
                "seed_values": values}
    pairs = []
    for model in ("shared", "banked"):
        for seed in range(3):
            left, right = [next(r for r in records if r["model"] == model and r["training_seed"] == seed
                                and r["condition"] == condition) for condition in ("failure", "sham")]
            for field in ("checkpoint_sha256", "model_signature", "checkpoint_progress", "source_sha256"):
                assert left[field] == right[field]
            assert left["scheduled_event_exposed_episodes"] == right["scheduled_event_exposed_episodes"]
            assert right["event_exposed_episodes"] == 0
            pairs.append({"model": model, "seed": seed, "assigned": left["episodes"],
                "exposed": left["event_exposed_episodes"],
                "scheduled_exposed_sham": right["scheduled_event_exposed_episodes"],
                "unexposed": left["episodes"] - left["event_exposed_episodes"],
                "failure_successes": left["successes"], "sham_successes": right["successes"],
                "aggregate_deltas": {m: left[m] - right[m] for m in
                    ("success_rate", "mean_team_return", "mean_agent_return")}})
    checkpoints = [r for r in records if r["condition"] == "nominal"]
    summary = {"schema_version": 1, "scope": "stdout summary audit; full evaluation JSONs unavailable locally",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "records": records, "nominal": aggregate, "failure_sham_pairs": pairs,
        "assigned_failure_episodes": sum(p["assigned"] for p in pairs),
        "actual_failure_exposures": sum(p["exposed"] for p in pairs),
        "preflight": preflight,
        "application_completion_summaries": len(records),
        "total_reported_evaluation_episodes": sum(r["episodes"] for r in records),
        "checkpoint_steps_range": [min(r["checkpoint_progress"]["total_steps"] for r in checkpoints),
                                   max(r["checkpoint_progress"]["total_steps"] for r in checkpoints)],
        "training_source_hashes": sorted({r["source_sha256"] for r in records}),
        "nominal_banked_minus_shared": {m: aggregate["banked"]["metrics"][m]["equal_seed_mean"] -
                                          aggregate["shared"]["metrics"][m]["equal_seed_mean"]
                                          for m in ("success_rate", "mean_team_return", "mean_agent_return")},
        "conditional_panel": {"basis": "repository preparation and array mapping; manifest not synced",
                              "teams": [[2,1],[1,2],[2,2],[3,1],[3,2]], "episodes_each":500,
                              "nominal_seed":2700, "failure_seed":2701, "failure_window":[10,30]},
        "unavailable": ["per-composition outcome rows", "episode completion lengths", "paired traces",
                        "victim diagnostics", "evaluator source hash", "scenario panel identity",
                        "scheduler accounting", "observed Stokes throughput"],
        "input_manifest": inputs}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = ["index", "model", "training_seed", "condition", "episodes", "successes", "failures",
              "success_rate", "mean_team_return", "mean_agent_return", "event_exposed_episodes",
              "scheduled_event_exposed_episodes", "stdout"]
    with (OUT / "summary_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    print(json.dumps({"nominal": aggregate, "exposures": summary["actual_failure_exposures"],
                      "reported_episodes": summary["total_reported_evaluation_episodes"]}, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
