"""Recompute both copied preflight timing reports; never execute training or Slurm.

Run from any directory with the repository's Python. Output is confined to this
analysis directory (or --output); original artifacts are checked for mutation.
Checkpoint/source and episode/learner audits are independent companion scripts.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("905118", "905404")
BUDGETS = {"pp_real": 40_000_000, "pcp_real": 40_000_000,
           "fc_real": 28_000_000, "pcp_binary": 40_000_000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "metrics")
    args = parser.parse_args()
    inputs = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    runs, windows = [], []
    for group in GROUPS:
        for workload, budget in BUDGETS.items():
            run = ROOT / "stokes_runs" / f"hetnet_paper_preflight_{group}" / workload / "seed991"
            updates = read(run / "updates.jsonl", lines=True)
            status = read(run / "run_status.json")
            preflight = read(run / "preflight.json")
            protocol = read(run / "protocol.json")
            assert len(updates) == 100
            assert [u["update"] for u in updates] == list(range(1, 101))
            assert protocol["message_backend"] == "torch-v1"
            assert protocol["reconstruction_spec"] == protocol["env_version"] == "paper-v1"
            assert protocol["learner_spec"] == "paper-equations-v1"
            assert protocol["seed"] == 991 and protocol["collectors"] == 4
            assert protocol["batch_step_floor_per_collector"] == 500
            assert protocol["episode_horizon"] == (300 if workload == "fc_real" else 80)
            assert protocol["profile_phases"] is False
            assert preflight["frozen_checkpoint_probe"]["parameters_unchanged"] is True
            assert status["stop_reason"] == "epoch_cap_completed"
            steps, episodes = sum(u["steps"] for u in updates), sum(u["episodes"] for u in updates)
            assert status["counts"] == preflight["counts"] == {
                "env_steps": steps, "episodes": episodes, "updates": 100, "epoch": 10}
            update_seconds = sum(u["wall_time_seconds"] for u in updates)
            assert [u["wall_time_seconds"] for u in updates] == preflight["update_seconds"]
            assert math.isclose(update_seconds, status["training_update_seconds"], abs_tol=1e-9)
            segment_seconds = status["segment_wall_time_seconds"]
            rate = steps / update_seconds
            projected_hours = budget / (steps / segment_seconds) / 3600
            assert math.isclose(rate, preflight["measured_update_steps_per_second"], rel_tol=1e-12)
            assert math.isclose(projected_hours, preflight[
                "projected_segment_hours_including_logging_and_checkpoints_excluding_startup"], rel_tol=1e-12)
            assert all(math.isfinite(u[k]) for u in updates for k in
                       ("policy_loss", "value_loss", "gradient_norm_preclip", "wall_time_seconds"))
            rows = []
            for first in range(0, 100, 20):
                block = updates[first:first + 20]
                row = {"group": group, "workload": workload,
                       "first_update": first + 1, "last_update": first + 20,
                       "steps": sum(u["steps"] for u in block),
                       "seconds": sum(u["wall_time_seconds"] for u in block)}
                row["steps_per_second"] = row["steps"] / row["seconds"]
                rows.append(row)
                windows.append(row)
            runs.append({"group": group, "workload": workload, "steps": steps,
                         "episodes": episodes, "updates": 100,
                         "update_seconds": update_seconds, "segment_seconds": segment_seconds,
                         "startup_seconds": status["startup_to_training_seconds"],
                         "first_update_seconds": updates[0]["wall_time_seconds"],
                         "update_steps_per_second": rate,
                         "segment_steps_per_second": steps / segment_seconds,
                         "last_20_update_steps_per_second": rows[-1]["steps_per_second"],
                         "update_fraction_of_segment": update_seconds / segment_seconds,
                         "checkpoint_fraction_of_segment": status["checkpoint_seconds"] / segment_seconds,
                         "budget_steps": budget, "projected_segment_hours": projected_hours,
                         "planned_46_hour_segments_at_current_rate": math.ceil(projected_hours / 46),
                         "all_updates_clipped_at_0_75": all(u["gradient_norm_preclip"] > .75 for u in updates),
                         "input_run": run.relative_to(ROOT).as_posix()})
    workloads = []
    for workload, budget in BUDGETS.items():
        pair = [r for r in runs if r["workload"] == workload]
        assert (pair[0]["steps"], pair[0]["episodes"]) == (pair[1]["steps"], pair[1]["episodes"])
        workloads.append({"workload": workload, "budget_steps": budget,
                          "905404_to_905118_update_rate_ratio": pair[1]["update_steps_per_second"] /
                              pair[0]["update_steps_per_second"],
                          "projected_hours_range": sorted(r["projected_segment_hours"] for r in pair)})
    job_hours = {group: 3 * sum(r["projected_segment_hours"] for r in runs if r["group"] == group)
                 for group in GROUPS}
    summary = {"date": "2026-10-05", "groups": list(GROUPS), "backend": "torch-v1",
               "counts": {"updates": 800, "env_steps": sum(r["steps"] for r in runs),
                          "episodes": sum(r["episodes"] for r in runs)},
               "runs": runs, "workloads": workloads,
               "projected_12_hetnet_run_job_hours": job_hours,
               "idealized_days_at_three_hetnet_jobs": {g: h / 72 for g, h in job_hours.items()},
               "interpretation": ["Both arrays are same-seed Torch repeats, not a backend speedup measurement.",
                                  "Ranges are two observed early-rate projections, not confidence intervals.",
                                  "Projections include segment logging/checkpoints but exclude startup, probes and queueing.",
                                  "No preflight node/CPU model or per-phase timing was saved.",
                                  "Engineering readiness does not establish convergence or exact paper reproduction."]}
    for name, checksum in inputs.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum, name
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("input_hashes.json", inputs)):
        (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for name, rows in (("throughput.csv", runs), ("update_windows.csv", windows)):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"validated_runs": len(runs), "counts": summary["counts"],
                      "input_files": len(inputs), "inputs_unchanged": True,
                      "projected_job_hours": job_hours}, indent=2))


if __name__ == "__main__":
    main()
