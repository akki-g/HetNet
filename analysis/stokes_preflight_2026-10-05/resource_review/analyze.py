"""Read-only timing/resource comparison of two supplied paper-v1 preflights."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
WORKLOADS = ("pp_real", "pcp_real", "fc_real", "pcp_binary")
HASHES = {}


def payload(path):
    raw = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw


def read(path):
    return json.loads(payload(path))


def lines(path):
    return [json.loads(line) for line in payload(path).splitlines()]


def window(rows):
    steps = sum(r["steps"] for r in rows)
    seconds = sum(r["wall_time_seconds"] for r in rows)
    return {"first_update": rows[0]["update"], "last_update": rows[-1]["update"],
            "steps": steps, "episodes": sum(r["episodes"] for r in rows),
            "seconds": seconds, "steps_per_second": steps / seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark_root = ROOT / "stokes_runs/hetnet_backend_benchmark_904547"
    benchmark = defaultdict(list)
    for name in WORKLOADS:
        for pair in (1, 2, 3):
            run = benchmark_root / "throughput" / name / f"pair{pair}" / "torch-v1"
            updates = lines(run / "updates.jsonl")
            status = read(run / "run_status.json")
            initial = read(run / "initial_training_identity.json")
            benchmark[name].append({"all20": window(updates), "post_first": window(updates[1:]),
                "segment_rate": status["counts"]["env_steps"] / status["segment_wall_time_seconds"],
                "initial": {k: initial[k] for k in ("model", "optimizer", "rng")},
                "update_steps": [r["steps"] for r in updates]})
    rows, updates_by_key = [], {}
    for job in ("905118", "905404"):
        for index, name in enumerate(WORKLOADS):
            run = ROOT / f"stokes_runs/hetnet_paper_preflight_{job}" / name / "seed991"
            status, initial = read(run / "run_status.json"), read(run / "initial_training_identity.json")
            protocol, preflight = read(run / "protocol.json"), read(run / "preflight.json")
            environment = read(run / "environment.json")
            segment = read(run / "training_segment.json")
            updates = lines(run / "updates.jsonl")
            metrics = lines(run / "metrics.jsonl")
            checkpoint_records = lines(run / "checkpoint_records.jsonl")
            exit_code = int(payload(run / "exit_code.txt"))
            out = payload(ROOT / f"logs_1/hetnet-paper-preflight-{job}_{index}.out")
            err = payload(ROOT / f"logs_1/hetnet-paper-preflight-{job}_{index}.err")
            stdout = payload(run / "stdout.log")
            assert exit_code == 0 and not err
            assert len(updates) == status["counts"]["updates"] == preflight["updates"] == 100
            assert [r["update"] for r in updates] == list(range(1, 101))
            assert sum(r["steps"] for r in updates) == status["counts"]["env_steps"]
            assert sum(r["episodes"] for r in updates) == status["counts"]["episodes"]
            assert preflight["counts"] == status["counts"]
            assert [r["wall_time_seconds"] for r in updates] == preflight["update_seconds"]
            assert protocol["message_backend"] == "torch-v1" and protocol["reconstruction_spec"] == "paper-v1"
            assert protocol["collectors"] == status["resources"]["collector_count"] == 4
            assert protocol["batch_step_floor_per_collector"] == 500
            assert status["resources"]["slurm_cpus_per_task"] == "4"
            assert status["stop_reason"] == "epoch_cap_completed"
            assert preflight["frozen_checkpoint_probe"]["parameters_unchanged"]
            assert all(math.isfinite(r["wall_time_seconds"]) and r["wall_time_seconds"] > 0 for r in updates)
            all_work = window(updates)
            assert math.isclose(all_work["seconds"], status["training_update_seconds"], rel_tol=1e-12)
            assert math.isclose(all_work["steps_per_second"], preflight["measured_update_steps_per_second"], rel_tol=1e-12)
            resources = []
            for collector, (a, b) in enumerate(zip(initial["resources"], status["resources"]["collectors"])):
                assert a["torch_threads"] == b["torch_threads"] == 1
                assert a["cpu_affinity"] == b["cpu_affinity"] and len(b["cpu_affinity"]) == 4
                assert a["max_rss_unit"] == b["max_rss_unit"] == "KiB"
                resources.append({"collector": collector, "cpu_affinity": b["cpu_affinity"],
                    "user_cpu_seconds": b["user_cpu_seconds"] - a["user_cpu_seconds"],
                    "system_cpu_seconds": b["system_cpu_seconds"] - a["system_cpu_seconds"],
                    "initial_max_rss_gib": a["max_rss"] / 2 ** 20, "final_max_rss_gib": b["max_rss"] / 2 ** 20})
            user = sum(r["user_cpu_seconds"] for r in resources)
            system = sum(r["system_cpu_seconds"] for r in resources)
            steps, wall = all_work["steps"], status["segment_wall_time_seconds"]
            rate = steps / wall
            budget = 28_000_000 if name == "fc_real" else 40_000_000
            hours = budget / rate / 3600
            windows = [window(updates[n:n + 20]) for n in range(0, 100, 20)]
            bench = benchmark[name]
            ident = {k: initial[k] for k in ("model", "optimizer", "rng")}
            assert all(ident == b["initial"] for b in bench)
            assert all([r["steps"] for r in updates[:20]] == b["update_steps"] for b in bench)
            row = {"job": job, "workload": name, "run": str(run.relative_to(ROOT)),
                "counts": status["counts"], "all_updates": all_work, "updates_2_to_100": window(updates[1:]),
                "updates_2_to_20": window(updates[1:20]), "updates_21_to_100": window(updates[20:]),
                "windows_20_updates": windows, "last20_to_first20_rate_ratio": windows[-1]["steps_per_second"] / windows[0]["steps_per_second"],
                "startup_seconds": status["startup_to_training_seconds"], "segment_seconds": wall,
                "segment_steps_per_second": rate,
                "startup_plus_segment_seconds": wall + status["startup_to_training_seconds"],
                "startup_plus_segment_steps_per_second": steps / (wall + status["startup_to_training_seconds"]),
                "startup_scope": "Recorded runtime startup+training segment; excludes outer archive creation, checkpoint probe and scheduler/queue time.",
                "checkpoint_seconds_status_scope": status["checkpoint_seconds"],
                "checkpoint_fraction_of_segment": status["checkpoint_seconds"] / wall,
                "non_update_segment_seconds": wall - all_work["seconds"],
                "collector_intervals": resources, "cpu_seconds_per_segment_second": (user + system) / wall,
                "cpu_fraction_of_four_cpu_allocation": (user + system) / (4 * wall),
                "aggregate_cpu_ms_per_joint_environment_step": (user + system) * 1000 / steps,
                "system_cpu_fraction": system / (user + system),
                "sum_individual_lifetime_peak_rss_gib_not_job_peak": sum(r["final_max_rss_gib"] for r in resources),
                "largest_individual_lifetime_peak_rss_gib": max(r["final_max_rss_gib"] for r in resources),
                "environment_fields_present": sorted(environment), "hardware_hostname_or_cpu_family_recorded": False,
                "source_manifest_sha256": segment["source_manifest_sha256"], "initial_identity": ident,
                "benchmark_initial_identities_and_first20_step_counts_match": True,
                "relative_to_benchmark_median_first20_rate": windows[0]["steps_per_second"] / median(b["all20"]["steps_per_second"] for b in bench),
                "relative_to_benchmark_median_updates_2_to_20_rate": window(updates[1:20])["steps_per_second"] / median(b["post_first"]["steps_per_second"] for b in bench),
                "relative_to_benchmark_median_segment_rate": rate / median(b["segment_rate"] for b in bench),
                "budget_steps": budget, "conditional_budget_hours_using_full_segment_rate": hours,
                "conditional_minimum_46h_segments": math.ceil(hours / 46),
                "conditional_steps_first46h_capped_to_budget": min(budget, rate * 46 * 3600),
                "conditional_active_hours_after_first46h": max(0, hours - 46),
                "outer_stdout_starts_with_archived_runtime_stdout": out.startswith(stdout),
                "stderr_empty": not err, "exit_code": exit_code,
                "metrics_have_memory_timeseries": any("memory" in k or "rss" in k for m in metrics for k in m),
                "checkpoint_record_count": len(checkpoint_records)}
            assert math.isclose(hours, preflight["projected_segment_hours_including_logging_and_checkpoints_excluding_startup"], rel_tol=1e-12)
            rows.append(row)
            updates_by_key[(job, name)] = updates
    array_comparisons = []
    for name in WORKLOADS:
        left, right = [next(r for r in rows if (r["job"], r["workload"]) == (job, name)) for job in ("905118", "905404")]
        without_time = lambda items: [{k: v for k, v in item.items() if k != "wall_time_seconds"} for item in items]
        array_comparisons.append({"workload": name, "counts_match": left["counts"] == right["counts"],
            "update_ledger_excluding_time_exact": without_time(updates_by_key[("905118", name)]) == without_time(updates_by_key[("905404", name)]),
            "905404_over_905118_update_rate": right["all_updates"]["steps_per_second"] / left["all_updates"]["steps_per_second"],
            "905404_over_905118_segment_rate": right["segment_steps_per_second"] / left["segment_steps_per_second"],
            "905118_over_905404_cpu_ms_per_step": left["aggregate_cpu_ms_per_joint_environment_step"] / right["aggregate_cpu_ms_per_joint_environment_step"]})
    totals = {job: {"counts": {key: sum(r["counts"][key] for r in rows if r["job"] == job) for key in ("updates", "env_steps", "episodes")},
        "summed_startup_plus_segment_minutes": sum(r["startup_plus_segment_seconds"] for r in rows if r["job"] == job) / 60,
        "conditional_full_12_seed_job_hours": 3 * sum(r["conditional_budget_hours_using_full_segment_rate"] for r in rows if r["job"] == job),
        "conditional_ideal_hours_at_three_continuously_active_runs": sum(r["conditional_budget_hours_using_full_segment_rate"] for r in rows if r["job"] == job)} for job in ("905118", "905404")}
    result = {"schema_version": 1, "rows": rows, "array_comparisons": array_comparisons, "totals": totals,
        "resource_ranges_across_all_eight_runs": {key: {"min": min(row[key] for row in rows), "max": max(row[key] for row in rows)} for key in (
            "cpu_seconds_per_segment_second", "cpu_fraction_of_four_cpu_allocation", "system_cpu_fraction",
            "sum_individual_lifetime_peak_rss_gib_not_job_peak", "largest_individual_lifetime_peak_rss_gib",
            "checkpoint_seconds_status_scope", "checkpoint_fraction_of_segment", "non_update_segment_seconds")},
        "hardware_evidence_limit": "No hostname, CPU model/topology, actual Slurm node/constraint/allocation-memory record or accounting data appears in the supplied environment/status/stdout records. Affinity indices are not portable hardware identities. Four Slurm CPUs and one Torch thread per collector are recorded.",
        "limitations": ["Only CPU/RSS start/end snapshots; no phase timers, per-update CPU, page faults, swap, cgroup memory, frequencies or independent co-tenant load measurements.",
            "Sum of individual lifetime peak RSS is not simultaneous job RSS and can double-count shared pages. Initial-to-final high-water increase does not establish a memory leak.",
            "Similar CPU allocation utilization plus larger CPU-seconds per identical work indicates slower CPU service/greater CPU work, not its physical cause; do not infer CPU family, thermal behavior or scheduler contention.",
            "Window rates are descriptive same-seed early-training observations, not independent-seed uncertainty estimates or a guarantee of long-run stability.",
            "The two arrays and benchmark ran in separate allocations; speed differences are not a randomized hardware experiment.",
            "Budget extrapolations assume fixed observed rates; queueing, startup, probes, transfers, continuation downtime and changing learning workloads are excluded."],
        "input_sha256": HASHES, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(rows), "inputs": len(HASHES), "array_comparisons": array_comparisons, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
