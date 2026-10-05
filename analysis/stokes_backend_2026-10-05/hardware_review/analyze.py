"""Read-only hardware/phase audit of Stokes benchmark 904547.

Writes one fresh JSON result. Does not import or execute archived training code.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "stokes_runs/hetnet_backend_benchmark_904547"
HASHES = {}


def read(path):
    payload = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def lines(path):
    payload = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return [json.loads(line) for line in payload.splitlines()]


def span(values):
    return {"min": min(values), "median": median(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hardware = read(INPUT / "hardware.json")
    assert hardware["affinity"] == [0, 2, 4, 6]
    assert hardware["slurm"]["SLURM_CPUS_PER_TASK"] == "4"
    assert hardware["slurm"]["SLURM_MEM_PER_NODE"] == "16384"
    rows, groups = [], defaultdict(list)
    paths = sorted(INPUT.glob("result_*.json"))
    assert len(paths) == 32
    for path in paths:
        result = read(path)
        run = INPUT / result["output"].split("/" + INPUT.name + "/", 1)[1]
        initial = read(run / "initial_training_identity.json")
        status = read(run / "run_status.json")
        protocol = read(run / "protocol.json")
        updates = lines(run / "updates.jsonl")
        start, end = initial["resources"], status["resources"]["collectors"]
        assert len(start) == len(end) == 4 == protocol["collectors"]
        assert status["resources"]["slurm_cpus_per_task"] == "4"
        assert status["stop_reason"] == "epoch_cap_completed"
        assert result["exit_code"] == 0
        assert result["counts"] == status["counts"]
        assert len(updates) == (3 if result["profiled"] else 20)
        assert sum(row["steps"] for row in updates) == result["counts"]["env_steps"]
        for i, (a, b, interval) in enumerate(zip(start, end, result["collector_intervals"])):
            assert a["cpu_affinity"] == b["cpu_affinity"] == interval["affinity"] == hardware["affinity"]
            assert a["torch_threads"] == b["torch_threads"] == interval["torch_threads"] == 1
            assert b["max_rss_unit"] == interval["max_rss_unit"] == "KiB"
            assert b["max_rss"] == interval["max_rss"]
            for key in ("user_cpu_seconds", "system_cpu_seconds"):
                assert math.isclose(b[key] - a[key], interval[key], rel_tol=1e-12, abs_tol=1e-12)
                assert interval[key] >= 0
        intervals = result["collector_intervals"]
        user = sum(row["user_cpu_seconds"] for row in intervals)
        system = sum(row["system_cpu_seconds"] for row in intervals)
        steps = result["counts"]["env_steps"]
        update_seconds = sum(row["wall_time_seconds"] for row in updates)
        assert math.isclose(update_seconds, sum(result["update_seconds"]), rel_tol=1e-12)
        row = dict(input=str(path.relative_to(ROOT)), run=str(run.relative_to(ROOT)),
            task=result["task"], variant=result["variant"], backend=result["backend"], pair=result["pair"],
            profiled=result["profiled"], steps=steps, episodes=result["counts"]["episodes"],
            update_seconds=update_seconds, segment_seconds=result["segment_seconds"],
            startup_seconds=result["startup_seconds"], process_seconds=result["process_seconds"],
            user_cpu_seconds=user, system_cpu_seconds=system,
            aggregate_cpu_seconds_per_segment_second=(user + system) / result["segment_seconds"],
            four_cpu_allocation_utilization=(user + system) / (4 * result["segment_seconds"]),
            aggregate_cpu_ms_per_joint_environment_step=1000 * (user + system) / steps,
            system_cpu_fraction=system / (user + system),
            sum_individual_peak_rss_gib_not_simultaneous_job_peak=sum(r["max_rss"] for r in intervals) / 1024 ** 2,
            largest_individual_peak_rss_gib=max(r["max_rss"] for r in intervals) / 1024 ** 2,
            checkpoint_seconds=result["checkpoint_seconds"],
            checkpoint_fraction_of_segment=result["checkpoint_seconds"] / result["segment_seconds"],
            non_update_non_checkpoint_segment_seconds=result["segment_seconds"] - update_seconds - result["checkpoint_seconds"],
            post_first_update_steps_per_second=sum(r["steps"] for r in updates[1:]) / sum(r["wall_time_seconds"] for r in updates[1:]))
        if result["profiled"]:
            phases = {key: sum(r["phases"][key] for r in updates) for key in updates[0]["phases"]}
            assert result["phase_seconds_by_update"] == [r["phases"] for r in updates]
            row["phase_seconds"] = phases
            row["phase_ms_per_joint_environment_step"] = {k: 1000 * v / steps for k, v in phases.items()}
            collector_keys = ("phase_environment_seconds", "phase_graph_preparation_seconds", "phase_model_inference_seconds", "phase_loss_backward_seconds")
            total = sum(phases[k] for k in collector_keys)
            row["collector_phase_share_of_measured_collector_phases"] = {k: phases[k] / total for k in collector_keys}
            row["collector_phase_seconds_divided_by_4_update_seconds"] = total / (4 * update_seconds)
            row["parent_phase_fraction_of_update_wall"] = {k: phases[k] / update_seconds for k in phases if k not in collector_keys}
        else:
            groups[(row["task"], row["variant"], row["backend"])].append(row)
        rows.append(row)
    group_rows = []
    for (task, variant, backend), values in sorted(groups.items()):
        group_rows.append(dict(task=task, variant=variant, backend=backend,
            statistics={key: span([row[key] for row in values]) for key in values[0] if isinstance(values[0][key], (float, int)) and key not in ("pair", "profiled")},
            replicates=len(values)))
    diagnostic_pairs = []
    for task, variant in (("pp", "real"), ("pcp", "real"), ("fc", "real"), ("pcp", "binary")):
        left, right = [next(r for r in rows if r["profiled"] and (r["task"], r["variant"], r["backend"]) == (task, variant, backend)) for backend in ("dgl", "torch-v1")]
        assert left["steps"] == right["steps"]
        keys = ("phase_environment_seconds", "phase_graph_preparation_seconds", "phase_model_inference_seconds", "phase_loss_backward_seconds")
        delta = {k: left["phase_ms_per_joint_environment_step"][k] - right["phase_ms_per_joint_environment_step"][k] for k in keys}
        diagnostic_pairs.append(dict(task=task, variant=variant,
            collector_phase_saving_ms_per_joint_environment_step=delta,
            fraction_of_summed_collector_phase_savings={k: v / sum(delta.values()) for k, v in delta.items()},
            caveat="Diagnostic collector wall phases sum over four concurrent collectors; contributions are not an additive decomposition of elapsed speedup."))
    old = {}
    for variant in ("real", "binary"):
        path = ROOT / f"stokes_runs/hetnet_preflight_903502/pcp_{variant}/seed991"
        updates = lines(path / "updates.jsonl")
        protocol = read(path / "protocol.json")
        old[variant] = dict(steps=sum(r["steps"] for r in updates), seconds=sum(r["wall_time_seconds"] for r in updates),
            updates=len(updates), model_spec=protocol["model_spec"], environment_version=protocol["env_version"], learner_spec=protocol["learner_spec"],
            protocol_keys=list(protocol))
    old["binary_relative_to_real_seconds_per_step"] = (old["binary"]["seconds"] / old["binary"]["steps"]) / (old["real"]["seconds"] / old["real"]["steps"])
    new_ratio = {}
    for backend in ("dgl", "torch-v1"):
        def rates(variant):
            return [r["post_first_update_steps_per_second"] for r in groups[("pcp", variant, backend)]]
        new_ratio[backend] = median(rates("real")) / median(rates("binary"))
    source = INPUT / "throughput/pp_real/pair1/dgl/source"
    for relative in ("runtime/multi_processing.py", "runtime/trainer.py", "runtime/hetgat/policy.py", "runtime/hetnet_ext/recovery.py", "runtime/main.py", "benchmark.py"):
        path = source / relative
        HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = dict(schema_version=1, hardware_identity={k: hardware[k] for k in ("hostname", "affinity", "logical_cpus", "thread_environment")},
        checks={"all_32_runs_four_collectors_one_torch_thread_matching_four_core_affinity": True,
                "all_128_resource_intervals_match_raw_start_end_counters": True,
                "slurm_matches_4_cpus_16_gib": True},
        rows=rows, throughput_groups=group_rows, diagnostic_pairs=diagnostic_pairs,
        old_preflight=old, new_binary_relative_to_real_post_first_update_seconds_per_step=new_ratio,
        limitations=["Single node/allocation; identical four-core affinity is an allowed CPU set, not one collector pinned per core.",
                     "Allocation records do not prove node/socket exclusivity or lack of external cache/memory-bandwidth contention.",
                     "Resource CPU intervals span collector samples approximately surrounding training; first/last sampling and housekeeping differ slightly from wall timer boundaries.",
                     "RSS fields are lifetime process high-water marks, not concurrent job/cgroup peaks; sums can double-count shared pages. No swap, fault, memory-pressure or bandwidth samples exist.",
                     "Parent wait starts after collector zero finishes its own rollout/backward; it is residual worker synchronization/receive time, not total communication or idle time.",
                     "Graph/forward/loss/environment are separate collector intervals. Collector intervals overlap across four processes; parent wait/aggregation/optimizer must not be added to their total as one elapsed breakdown.",
                     "Loss/backward includes all loss construction and backpropagation, not just message passing. Forward includes the full model.",
                     "Diagnostics are separate three-update runs, not throughput samples. They show where local cost changed, not formal causal mediation or long-run throughput.",
                     "Old supplement-v1 and new paper-v1 change architecture, learner and Binary head/payload semantics. New benchmark cannot identify cause of old slowdown."],
        inputs_sha256=HASHES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "inputs": len(HASHES), "rows": len(rows), "old_binary_ratio": old["binary_relative_to_real_seconds_per_step"], "new_binary_ratio": new_ratio}, indent=2))


if __name__ == "__main__":
    main()
