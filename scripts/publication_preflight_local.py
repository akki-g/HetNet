#!/usr/bin/env python3
"""Run the four locked Slurm preflight workloads locally, with separate logs."""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from publication_reconstruction.__main__ import validate_source_origins
from publication_reconstruction.study import WORKLOADS

THREAD_ENV = {name: "1" for name in
              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
THREAD_ENV.update(PYTHONUNBUFFERED="1", DGLBACKEND="pytorch")


def timestamp():
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def commands(run_root):
    return [{"workload": f"{task}_{variant}", "index": index,
             "command": [sys.executable, "-u", "-m", "publication_reconstruction",
                         "run-index", "--preflight", "--index", str(index),
                         "--run-root", str(run_root)]}
            for index, (task, variant, _) in enumerate(WORKLOADS)]


def episode_summary(episodes):
    count = len(episodes)
    durations = [row["rollout_wall_time_seconds"] for row in episodes]
    return {"episodes": count, "steps": sum(row["steps"] for row in episodes),
            "success_rate": sum(row["success"] for row in episodes) / count,
            "mean_agent_return": sum(row["mean_agent_return"] for row in episodes) / count,
            "mean_episode_steps": sum(row["steps"] for row in episodes) / count,
            "mean_rollout_wall_time_seconds": sum(durations) / count,
            "min_rollout_wall_time_seconds": min(durations),
            "max_rollout_wall_time_seconds": max(durations)}


async def run_all(run_root, jobs, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    status = {job["workload"]: {"state": "queued"} for job in jobs}

    def save_status():
        temporary = run_root / "local_summary.json.tmp"
        temporary.write_text(json.dumps({"updated_at": timestamp(), "jobs": status},
                                        indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(run_root / "local_summary.json")

    async def run_one(job):
        async with semaphore:
            name = job["workload"]
            row = status[name]
            row.update(state="running", started_at=timestamp())
            begin = time.monotonic()
            child = None
            save_status()
            print(f"[{name}] starting; log: {run_root / 'logs' / (name + '.log')}", flush=True)
            try:
                child = await asyncio.create_subprocess_exec(
                    *job["command"], cwd=ROOT, env={**os.environ, **THREAD_ENV},
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True, limit=16 * 1024 * 1024)
                row["pid"] = child.pid
                save_status()
                with (run_root / "logs" / f"{name}.log").open("x") as log, \
                     (run_root / "logs" / f"{name}.episodes.jsonl").open("x") as episodes_log, \
                     (run_root / "logs" / f"{name}.progress.jsonl").open("x") as progress_log:
                    async for raw in child.stdout:
                        line = raw.decode("utf-8", errors="replace")
                        log.write(line)
                        log.flush()
                        if not line.startswith('{"episodes":'):
                            continue
                        batch = json.loads(line)
                        if batch.get("record_type") != "publication_episode_batch":
                            continue
                        episodes = batch["episodes"]
                        episodes_log.write("".join(json.dumps(ep, sort_keys=True) + "\n" for ep in episodes))
                        episodes_log.flush()
                        for ep in episodes:
                            print(f"[{name}] episode={ep['episode']} collector={ep['collector']} "
                                  f"steps={ep['steps']} success={int(ep['success'])} "
                                  f"return/agent={ep['mean_agent_return']:.4f} "
                                  f"rollout={ep['rollout_wall_time_seconds']:.3f}s", flush=True)
                        progress = {"update": batch["update"], "elapsed_seconds": time.monotonic() - begin,
                                    **episode_summary(episodes)}
                        progress_log.write(json.dumps(progress, sort_keys=True, allow_nan=False) + "\n")
                        progress_log.flush()
                        row["last_update"] = batch["update"]
                        save_status()
                        print(f"[{name}] {batch['update']:3d}/100 | episodes={progress['episodes']} "
                              f"success={progress['success_rate']:.1%} "
                              f"return/agent={progress['mean_agent_return']:.4f} "
                              f"steps/episode={progress['mean_episode_steps']:.1f} "
                              f"rollout/episode={progress['mean_rollout_wall_time_seconds']:.3f}s "
                              f"elapsed={progress['elapsed_seconds'] / 60:.1f}min", flush=True)
                row["exit_code"] = await child.wait()
                report = run_root / name / "seed991/preflight.json"
                if row["exit_code"] == 0 and report.is_file():
                    result = json.loads(report.read_text())
                    row.update(state="completed", counts=result["counts"],
                               steps_per_second=result["measured_update_steps_per_second"],
                               preflight_report=str(report))
                else:
                    row.update(state="failed", error="See workload log; successful preflight report required.")
            except asyncio.CancelledError:
                row["state"] = "interrupted"
                raise
            except Exception as exc:
                row.update(state="failed", error=str(exc))
            finally:
                if child is not None and child.returncode is None:
                    # The existing launcher handles TERM and cleans up its separately
                    # grouped training process and collectors before exiting.
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(child.wait(), timeout=20)
                    except asyncio.TimeoutError:
                        os.killpg(child.pid, signal.SIGKILL)
                        await child.wait()
                if child is not None:
                    # The frozen checkpoint probe shares the launcher's group;
                    # it must not survive even if its parent exits first.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    row["exit_code"] = child.returncode
                row.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - begin)
                save_status()
                print(f"[{name}] {row['state']} after {row['elapsed_seconds'] / 60:.1f}min", flush=True)

    # asyncio.run handles Ctrl-C; also allow a shell TERM to clean up all jobs.
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, current.cancel)
    save_status()
    try:
        await asyncio.gather(*(run_one(job) for job in jobs))
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        for row in status.values():
            if row["state"] == "queued":
                row["state"] = "not_started"
        save_status()
    return 0 if all(row["state"] == "completed" for row in status.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, help="fresh output directory; defaults to a timestamped runs/ folder")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--dry-run", action="store_true", help="print commands without creating files or starting jobs")
    args = parser.parse_args(argv)
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    run_root = (args.run_root or ROOT / "runs" / f"hetnet_preflight_mac_{stamp}").resolve()
    jobs = commands(run_root)
    validate_source_origins()
    print(f"Output: {run_root}\nConcurrency: {args.concurrency} jobs, 4 collectors each", flush=True)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["command"]))
        return 0
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "logs").mkdir()
    payload = Path(__file__).read_bytes()
    (run_root / "local_launcher.py").write_bytes(payload)
    (run_root / "local_launch.json").write_text(json.dumps({
        "started_at": timestamp(), "concurrency": args.concurrency, "jobs": jobs,
        "environment_overrides": THREAD_ENV, "launcher_sha256": hashlib.sha256(payload).hexdigest(),
        "episode_timing": "Collector elapsed rollout time includes reset, inference, environment and scheduling; "
                          "excludes batch gradients/optimizer/IPC. Overlapping durations are not job elapsed time.",
    }, indent=2, sort_keys=True) + "\n")
    try:
        return asyncio.run(run_all(run_root, jobs, args.concurrency))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Interrupted; local_summary.json records completed and stopped workloads.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
