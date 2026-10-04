"""Sequential, bounded DGL/Torch comparisons of the same paper-v1 baseline.

Twenty-update runs are diagnostics, never the official 100-update preflight.
Time/resource fields are observations, not claims of publication fidelity.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time

WORKLOADS = (("pp", "real"), ("pcp", "real"), ("fc", "real"), ("pcp", "binary"))
ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {name: "1" for name in
              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
THREAD_ENV.update(DGLBACKEND="pytorch", PYTHONUNBUFFERED="1")


def add_commands(commands):
    cli = commands.add_parser("benchmark", help="24 sequential paper-v1 backend runs; never submits")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--diagnostics", action="store_true", help="separate paired three-update phase diagnostics")
    cli.add_argument("--dry-run", action="store_true")


def benchmark_plan(output, diagnostics=False):
    output = Path(output).resolve()
    jobs = []
    for repetition in range(3):
        order = ("dgl", "torch-v1") if repetition != 1 else ("torch-v1", "dgl")
        for task, variant in WORKLOADS:
            for backend in order:
                name = f"{task}_{variant}/pair{repetition + 1}/{backend}"
                jobs.append(_job(output / "throughput" / name, task, variant, backend, repetition, False))
    if diagnostics:
        for task, variant in WORKLOADS:
            for backend in ("dgl", "torch-v1"):
                jobs.append(_job(output / "diagnostics" / f"{task}_{variant}" / backend,
                                 task, variant, backend, 0, True))
    return {"schema_version": 1, "reconstruction_spec": "paper-v1", "submitted": False,
            "seed": 991, "throughput_runs": 24, "throughput_updates": 480,
            "diagnostic_runs": 8 if diagnostics else 0, "jobs": jobs,
            "conditional_replay": {"trigger": "episode trajectories differ or were not completely recorded",
                "maximum_workloads": 4, "recorded_updates_per_workload": 3, "timing_pairs": 3,
                "collectors": 4, "batch_steps": 500, "semantics": "serial fixed-rollout compute; no environment/IPC timing"},
            "order": "fixed workload order within each repetition; DGL/Torch, Torch/DGL, DGL/Torch",
            "primary_statistic": "sum(steps)/sum(update_seconds), updates 2..20",
            "claim": "bounded engineering comparison; three pairs are not a confidence interval"}


def _job(output, task, variant, backend, repetition, profiled):
    argv = ["train", "--reconstruction-spec", "paper-v1", "--task", task, "--variant", variant,
            "--message-backend", backend, "--recipe", "october-2022", "--seed", "991",
            "--collectors", "4", "--batch-steps", "500", "--horizon", "300" if task == "fc" else "80",
            "--epochs", "1" if profiled else "2", "--updates-per-epoch", "3" if profiled else "10",
            "--save-every", "10", "--episode-log", "stdout", "--output", str(output)]
    if profiled:
        argv += ["--profile-phases"]
    return {"task": task, "variant": variant, "backend": backend, "pair": repetition + 1,
            "profiled": profiled, "updates": 3 if profiled else 20, "output": str(output),
            "command": [sys.executable, "-u", "-m", "publication_reconstruction", *argv]}


def host_identity():
    import importlib.metadata
    commands = [["lscpu"], ["lscpu", "-e"]] if sys.platform.startswith("linux") else [
        ["sysctl", "-n", "machdep.cpu.brand_string"], ["sysctl", "hw.physicalcpu", "hw.logicalcpu"]]
    if os.environ.get("SLURM_JOB_ID"):
        commands.append(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]])
    observed = []
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            observed.append({"command": command, "returncode": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
        except (OSError, subprocess.TimeoutExpired) as exc:
            observed.append({"command": command, "unavailable": str(exc)})
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
            "executable": sys.executable, "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
            "thread_environment": THREAD_ENV, "hardware_commands": observed}


def source_identity():
    import hashlib
    from .__main__ import validate_source_origins
    runtime, origins = validate_source_origins()
    payloads = {"runtime/" + name: data for name, data in runtime.items()}
    payloads["ORIGINS.json"] = origins
    for pattern in ("publication_reconstruction/*.py", "publication_reconstruction/*.md",
                    "slurm/publication_*.sbatch", "slurm/softrole_paper_fc.sbatch",
                    "publication_reconstruction/STUDY.json", "pyproject.toml", "uv.lock"):
        for path in sorted(ROOT.glob(pattern)):
            payloads[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _episodes(run):
    path = run / "episodes.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    result = []
    with (run / "stdout.log").open() as stream:
        for line in stream:
            if line.startswith("{"):
                row = json.loads(line)
                if row.get("record_type") == "publication_episode_batch":
                    result.extend(row["episodes"])
    return result


def validate_episode_ledger(episodes, updates, *, collectors, floor, horizon):
    """Reject missing/duplicated work before making any trajectory claim."""
    if not episodes or len(episodes) != sum(row["episodes"] for row in updates):
        raise ValueError("Benchmark episode ledger is empty or incomplete")
    offset = 0
    complete_hashes = True
    for update in updates:
        batch = episodes[offset:offset + update["episodes"]]
        per_collector = [0] * collectors
        collector_steps = [0] * collectors
        previous_collector = 0
        for index, row in enumerate(batch, start=offset + 1):
            expected = {"episode": index, "update": update["update"], "epoch": update["epoch"]}
            if any(type(row.get(key)) is not int or row[key] != value for key, value in expected.items()):
                raise ValueError("Benchmark episode order disagrees with update ledger")
            collector, steps = row.get("collector"), row.get("steps")
            if (type(collector) is not int or not 0 <= collector < collectors
                    or collector < previous_collector or type(steps) is not int or not 1 <= steps <= horizon):
                raise ValueError("Invalid benchmark collector order or episode length")
            if (type(row.get("collector_episode")) is not int
                    or row["collector_episode"] != per_collector[collector]):
                raise ValueError("Benchmark collector episode order disagrees")
            per_collector[collector] += 1
            collector_steps[collector] += steps
            previous_collector = collector
            if "trajectory_sha256" in row:
                if not isinstance(row["trajectory_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["trajectory_sha256"]):
                    raise ValueError("Invalid benchmark trajectory digest")
            else:
                complete_hashes = False
        if sum(collector_steps) != update["steps"] or any(count < floor for count in collector_steps):
            raise ValueError("Benchmark episode steps disagree with the ledger or collector floor")
        offset += len(batch)
    return complete_hashes


def summarize_run(job, process_seconds):
    from .artifacts import load_checkpoint
    from .runtime.hetnet_ext.signatures import tree_signature
    run = Path(job["output"])
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
    status, protocol = _read_json(run / "run_status.json"), _read_json(run / "protocol.json")
    if len(updates) != job["updates"] or status["counts"]["updates"] != job["updates"]:
        raise ValueError("Incomplete bounded benchmark run")
    if status["stop_reason"] != "epoch_cap_completed" or protocol["message_backend"] != job["backend"]:
        raise ValueError("Benchmark exit/backend differs from its plan")
    if sum(row["steps"] for row in updates) != status["counts"]["env_steps"]:
        raise ValueError("Benchmark sample ledger disagrees")
    if any(job["profiled"] != ("phases" in row) for row in updates):
        raise ValueError("Profiled diagnostic and throughput records must stay separate")
    checkpoint = Path(status["checkpoint"])
    saved = load_checkpoint(run, checkpoint)
    from .artifacts import _validate_optimizer, _validate_recorded_checkpoint
    from .study import _validate_preflight_ledger
    elapsed, steps = _validate_preflight_ledger(updates, status,
        _read_json(run / "training_segment.json"), protocol, saved)
    _validate_recorded_checkpoint(run, checkpoint, checkpoint.read_bytes(), saved, protocol)
    _validate_optimizer(saved, protocol, job["updates"])
    if not math.isfinite(process_seconds) or process_seconds <= 0:
        raise ValueError("Benchmark process time must be finite and positive")
    initial = _read_json(run / "initial_training_identity.json")
    end_resources = status["resources"]["collectors"]
    start_resources = initial["resources"]
    if len(end_resources) != 4 or len(start_resources) != 4:
        raise ValueError("Benchmark requires four collector resource intervals")
    intervals = [{"collector": i,
                  "user_cpu_seconds": end["user_cpu_seconds"] - start["user_cpu_seconds"],
                  "system_cpu_seconds": end["system_cpu_seconds"] - start["system_cpu_seconds"],
                  "max_rss": end["max_rss"], "max_rss_unit": end["max_rss_unit"],
                  "affinity": end.get("cpu_affinity"), "torch_threads": end.get("torch_threads")}
                 for i, (start, end) in enumerate(zip(start_resources, end_resources))]
    if any(row["torch_threads"] != 1 for row in intervals):
        raise ValueError("Collector Torch thread count differs from the benchmark contract")
    selected = updates[1:]
    post = sum(row["steps"] for row in selected) / sum(row["wall_time_seconds"] for row in selected)
    episodes = _episodes(run)
    trajectory_complete = validate_episode_ledger(episodes, updates, collectors=4,
        floor=protocol["batch_step_floor_per_collector"], horizon=protocol["episode_horizon"])
    excluded = {"rollout_wall_time_seconds"}
    numeric_episodes = [{k: v for k, v in row.items() if k not in excluded} for row in episodes]
    return {**job, "counts": status["counts"], "post_first_update_steps_per_second": post,
            "all_update_steps_per_second": steps / elapsed,
            "segment_steps_per_second": steps / status["segment_wall_time_seconds"],
            "process_steps_per_second": steps / process_seconds, "process_seconds": process_seconds,
            "first_update_seconds": updates[0]["wall_time_seconds"],
            "startup_seconds": status["startup_to_training_seconds"],
            "segment_seconds": status["segment_wall_time_seconds"], "checkpoint_seconds": status["checkpoint_seconds"],
            "update_seconds": [row["wall_time_seconds"] for row in updates],
            "phase_seconds_by_update": [row.get("phases") for row in updates] if job["profiled"] else None,
            "phase_semantics": "collector wall intervals sum across concurrent collectors; parent intervals are separate" if job["profiled"] else None,
            "collector_intervals": intervals,
            "initial_identity": {k: initial[k] for k in ("model", "optimizer", "rng")},
            "final_identity": {"model": tree_signature(saved["policy_net"]),
                               "optimizer": tree_signature(saved["trainer"]),
                               "rng": tree_signature(saved["recovery"]["rng_states"])},
            "episode_ledger_identity": tree_signature(numeric_episodes),
            "source_manifest_sha256": saved["reconstruction"]["source_manifest_sha256"],
            "trajectory_identity_complete": trajectory_complete,
            "exit_code": int((run / "exit_code.txt").read_text())}


def compare_runs(rows):
    comparisons = []
    for task, variant in WORKLOADS:
        pairs = []
        segment_rates = {"dgl": [], "torch-v1": []}
        for repetition in range(1, 4):
            selected = {r["backend"]: r for r in rows if not r["profiled"] and
                        (r["task"], r["variant"], r["pair"]) == (task, variant, repetition)}
            if set(selected) != {"dgl", "torch-v1"}:
                raise ValueError("Missing or incomplete benchmark pair")
            a, b = selected["dgl"], selected["torch-v1"]
            for backend, row in selected.items():
                segment_rates[backend].append(row["segment_steps_per_second"])
            if a["initial_identity"] != b["initial_identity"]:
                raise ValueError("Paired backends did not start from identical model/optimizer/RNG values")
            if a["source_manifest_sha256"] != b["source_manifest_sha256"]:
                raise ValueError("Paired backends used different source archives")
            if [r["affinity"] for r in a["collector_intervals"]] != [r["affinity"] for r in b["collector_intervals"]]:
                raise ValueError("Paired CPU affinity differs")
            pairs.append({"pair": repetition,
                          "update_throughput_ratio": b["post_first_update_steps_per_second"] / a["post_first_update_steps_per_second"],
                          "segment_throughput_ratio": b["segment_steps_per_second"] / a["segment_steps_per_second"],
                          "same_episode_ledger": a["episode_ledger_identity"] == b["episode_ledger_identity"],
                          "trajectories_verified_identical": a["trajectory_identity_complete"] and b["trajectory_identity_complete"] and
                              a["episode_ledger_identity"] == b["episode_ledger_identity"]})
        median_segment_ratio = statistics.median(segment_rates["torch-v1"]) / statistics.median(segment_rates["dgl"])
        speed = all(p["update_throughput_ratio"] > 1 for p in pairs) and median_segment_ratio > 1
        comparisons.append({"task": task, "variant": variant, "pairs": pairs,
                            "ratio_of_median_segment_throughput": median_segment_ratio,
                            "throughput_gate": speed,
                            "fixed_rollout_compute_replay_required": not all(p["trajectories_verified_identical"] for p in pairs),
                            "decision": "candidate; requires correctness and resource review" if speed else "retain DGL; slower or inconclusive",
                            "claim": "realized training throughput; differences in trajectories may change the workload"})
    return comparisons


def _run_child(command, log_path):
    with Path(log_path).open("x") as log:
        begin = time.monotonic()
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **THREAD_ENV},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if code:
        raise RuntimeError(f"Benchmark child exited {code}; see {log_path}")
    return time.monotonic() - begin


def run_benchmark(args):
    from .__main__ import write_json
    plan = benchmark_plan(args.output, args.diagnostics)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    source = source_identity()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    write_json(output / "plan.json", plan)
    write_json(output / "hardware.json", host_identity())
    write_json(output / "source_identity.json", source)
    rows = []
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for index, job in enumerate(plan["jobs"]):
            if source_identity() != source:
                raise RuntimeError("Source changed during the paired benchmark")
            print(f"[{index + 1}/{len(plan['jobs'])}] {job['task']} {job['variant']} {job['backend']} pair{job['pair']} profiled={job['profiled']}", flush=True)
            elapsed = _run_child(job["command"], output / "logs" / f"{index:02d}.log")
            row = summarize_run(job, elapsed)
            rows.append(row)
            write_json(output / f"result_{index:02d}.json", row)
        if source_identity() != source:
            raise RuntimeError("Source changed during the paired benchmark")
        comparisons = compare_runs(rows)
        for comparison in comparisons:
            if comparison["fixed_rollout_compute_replay_required"]:
                task, variant = comparison["task"], comparison["variant"]
                replay_output = output / "replay" / f"{task}_{variant}"
                command = [sys.executable, "-u", "-m", "publication_reconstruction.runtime.hetnet_ext.paper_replay",
                           "--output", str(replay_output), "--task", task, "--variant", variant,
                           "--collectors", "4", "--updates", "3", "--batch-steps", "500",
                           "--horizon", "300" if task == "fc" else "80", "--timing-repeats", "3", "--seed", "991"]
                print(f"[fixed-rollout replay] {task} {variant}", flush=True)
                if source_identity() != source:
                    raise RuntimeError("Source changed before fixed-rollout replay")
                seconds = _run_child(command, output / "logs" / f"replay_{task}_{variant}.log")
                replay = _read_json(replay_output / "summary.json")
                comparison["compute_replay"] = {"command": command, "report": str(replay_output / "summary.json"),
                    "process_seconds": seconds, "correctness_passed": replay["correctness_passed"],
                    "speedups": [pair["speedup"] for pair in replay["pairs"]]}
                comparison["compute_gate"] = replay["correctness_passed"] and all(
                    pair["speedup"] > 1 for pair in replay["pairs"])
                if not comparison["compute_gate"]:
                    comparison["decision"] = "retain DGL; fixed-rollout correctness/speed failed or inconclusive"
        if source_identity() != source:
            raise RuntimeError("Source changed during fixed-rollout replay")
        summary = {"schema_version": 1, "completed": True, "runs": len(rows),
                   "comparisons": comparisons, "jobs_submitted": False,
                   "resource_review_required": True,
                   "limitations": "Training readiness also requires independent fidelity/backend/recovery checks and the official100-update preflight. Timers are separate diagnostics."}
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        write_json(output / "failure.json", {"completed": False, "finished_runs": len(rows),
                   "error": str(exc), "error_type": type(exc).__name__})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
