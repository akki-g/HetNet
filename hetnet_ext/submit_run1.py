"""Review or explicitly submit the existing 21-task Stokes Run 1 array.

Resources come from the confirmed calibration budget. Concurrency is an explicit
operator choice bounded by verified limits, not an assumption that all capacity
is currently free. Dry-run validates evidence but writes nothing and contacts no
scheduler. This module never constructs a policy or imports the ML dependencies.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from .grid import ROOT, code_sha256, file_sha256, load_grid, run_directory
from .train_job import (read_json, utc_now, validate_approval, validate_gate, validate_preflight,
                        within_partition_time_limit, write_json)


def positive_integer(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer from verified evidence")
    return value


def slurm_time(seconds: int) -> str:
    positive_integer(seconds, "Requested time")
    if seconds % 60:
        raise ValueError("Approved time must be in complete minutes")
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"


def prepare(preflight_path: Path, gate_path: Path, approval_path: Path,
            concurrency: int, output_root: Path | None = None) -> dict:
    preflight_path, gate_path, approval_path = (p.resolve() for p in (preflight_path, gate_path, approval_path))
    preflight = read_json(preflight_path)
    validate_preflight(preflight)
    code_hash, lock_hash = code_sha256(), file_sha256(ROOT / "uv.lock")
    validate_gate(read_json(gate_path), code_hash, lock_hash)
    approved = validate_approval(approval_path, code_hash, lock_hash, preflight_path)
    budget = approved["budget"]
    entries = load_grid()
    if len(entries) != 21 or [e.index for e in entries] != list(range(21)):
        raise ValueError("This helper submits only the complete committed 21-task study")
    positive_integer(concurrency, "Array concurrency")
    if concurrency > 21:
        raise ValueError("Array concurrency cannot exceed its 21 tasks")
    limit = preflight["max_concurrent_jobs"]
    if limit != "unlimited" and concurrency > limit:
        raise ValueError("Requested concurrency exceeds the verified user/account limit")
    submit_limit = preflight["max_submit_jobs"]
    if submit_limit != "unlimited" and submit_limit < 21:
        raise ValueError("Verified submission limit cannot accommodate all 21 tasks")

    seconds = positive_integer(budget.get("array_uniform_time_seconds"), "Approved uniform time")
    memory = positive_integer(budget.get("array_uniform_memory_gb"), "Approved uniform memory GiB")
    time_argument = slurm_time(seconds)
    if not within_partition_time_limit(seconds, preflight["max_wall_time_seconds"]):
        raise ValueError("Approved request exceeds the current verified partition time cap")
    counts = Counter(e.composition for e in entries)
    rows = budget.get("rows", [])
    if (budget.get("schema_version") != 1 or len(rows) != len(counts)
            or Counter(row.get("composition") for row in rows) != Counter({name: 1 for name in counts})):
        raise ValueError("Approved budget must contain each of the five training compositions exactly once")
    for row in rows:
        if row.get("seeds") != counts[row["composition"]] or row.get("cpus_per_task") != 4:
            raise ValueError("Approved budget seed counts/CPUs do not match the committed grid")
        positive_integer(row.get("requested_wall_seconds"), "Budget row wall time")
        positive_integer(row.get("memory_gb_with_50pct_margin"), "Budget row memory")
    if (seconds != max(row["requested_wall_seconds"] for row in rows)
            or memory != max(row["memory_gb_with_50pct_margin"] for row in rows)):
        raise ValueError("Approved uniform resources do not equal the budget's maximum row requests")

    output_root = (output_root or ROOT / "runs" / "run1_train").resolve()
    marker = output_root / "run1_submission.json"
    if marker.exists() or marker.is_symlink():
        raise FileExistsError(f"Submission record already exists: {marker}; inspect it before any retry")
    paths = [run_directory(entry, "train", output_root) for entry in entries]
    occupied = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if occupied:
        raise FileExistsError("Refusing existing run directories; preserve prior attempts and choose a new output root: "
                              + ", ".join(occupied))
    logs = ROOT / "logs"
    command = ["sbatch", "--parsable", "--export=ALL", "--partition=normal", "--nodes=1", "--ntasks=1",
               "--cpus-per-task=4", f"--array=0-20%{concurrency}", f"--mem={memory}G", f"--time={time_argument}",
               f"--chdir={ROOT}", f"--output={logs / 'run1-%A_%a.out'}", f"--error={logs / 'run1-%A_%a.err'}"]
    if preflight["account_required"]:
        account = preflight["account"]
        if not isinstance(account, str) or not account.strip() or any(char.isspace() for char in account):
            raise ValueError("Verified account must be a nonempty single account name")
        command.append(f"--account={account}")
    command.append(str(ROOT / "slurm" / "run1_train.sbatch"))
    environment = {"HETNET_STOKES_PREFLIGHT": str(preflight_path), "HETNET_GATE_A": str(gate_path),
                   "HETNET_BUDGET_APPROVAL": str(approval_path), "HETNET_ARRAY_CONCURRENCY": str(concurrency),
                   "HETNET_MEM_GB": str(memory), "HETNET_TIME_SECONDS": str(seconds),
                   "HETNET_OUTPUT_ROOT": str(output_root)}
    evidence = {str(path): file_sha256(path) for path in
                (preflight_path, gate_path, approval_path, Path(approved["budget_file"]))}
    return {"schema_version": 1, "created_at_utc": utc_now(), "cwd": str(ROOT), "command": command,
            "environment": environment, "code_sha256": code_hash, "uv_lock_sha256": lock_hash,
            "evidence_sha256": evidence, "entries": [entry.as_dict() for entry in entries],
            "output_root": str(output_root), "submission_record": str(marker), "logs_directory": str(logs),
            "task_count": 21, "concurrency": concurrency, "cpus_per_task": 4,
            "time_seconds_per_task": seconds, "memory_gib_per_task": memory,
            "projected_total_core_hours": budget["projected_total_core_hours"],
            "percentage_of_remaining_balance": 100 * budget["projected_total_core_hours"] / preflight["remaining_core_hours"]}


def clean_environment(plan: dict) -> dict:
    # SBATCH_* environment options can override script directives or add GPU
    # requests. The reviewed command must be the sole source of sbatch options.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    environment.update(plan["environment"])
    return environment


def shell_command(plan: dict) -> str:
    unset = [item for key in sorted(os.environ) if key.startswith("SBATCH_") for item in ("-u", key)]
    assignments = [f"{key}={value}" for key, value in sorted(plan["environment"].items())]
    return shlex.join(["env", *unset, *assignments, *plan["command"]])


def submit(plan: dict) -> str:
    """Called only by explicit --submit; no scheduler interaction in prepare()."""
    observed = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True, check=True, timeout=15)
    match = re.search(r"^\s*ClusterName\s*=\s*(\S+)", observed.stdout, re.MULTILINE)
    if match is None or match.group(1).lower() != "stokes":
        raise ValueError("Current Slurm configuration is not verified as Stokes; no job submitted")
    if code_sha256() != plan["code_sha256"] or file_sha256(ROOT / "uv.lock") != plan["uv_lock_sha256"]:
        raise ValueError("Code/lock changed since submission review; no job submitted")
    if any(file_sha256(path) != digest for path, digest in plan["evidence_sha256"].items()):
        raise ValueError("Preflight, gate, budget or confirmation changed; no job submitted")
    for entry in load_grid():
        path = run_directory(entry, "train", Path(plan["output_root"]))
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Run directory appeared after review: {path}; no job submitted")
    Path(plan["logs_directory"]).mkdir(parents=True, exist_ok=True)
    marker = Path(plan["submission_record"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    record = {**plan, "status": "submitting", "submission_started_at_utc": utc_now(),
              "observed_cluster": match.group(1), "scontrol_config_sha256": hashlib.sha256(observed.stdout.encode()).hexdigest()}
    # Exclusive creation prevents two simultaneous helper calls from submitting
    # duplicate arrays to the same output root before job directories exist.
    with marker.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        result = subprocess.run(plan["command"], cwd=ROOT, env=clean_environment(plan),
                                capture_output=True, text=True, check=False, timeout=60)
        record.update(sbatch_returncode=result.returncode, sbatch_stdout=result.stdout, sbatch_stderr=result.stderr)
        if result.returncode:
            record["status"] = "submission_failed"
            raise RuntimeError("sbatch returned an error; inspect the preserved submission record before retrying")
        response = re.fullmatch(r"([0-9]+)(?:;([^;\s]+))?", result.stdout.strip())
        if response is None or (response.group(2) and response.group(2).lower() != "stokes"):
            raise RuntimeError("sbatch response is ambiguous; check the queue before any retry")
        record.update(status="submitted", array_job_id=response.group(1))
    except BaseException as exc:
        if record["status"] == "submitting":
            record["status"] = "submission_unknown"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["submission_finished_at_utc"] = utc_now()
        write_json(marker, record)
    return record["array_job_id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--gate-a", type=Path, required=True)
    parser.add_argument("--budget-approval", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True,
                        help="Verified currently available concurrency; accounts for existing jobs")
    parser.add_argument("--output-root", type=Path, help="New Run 1 output root; default runs/run1_train")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and print only (default)")
    mode.add_argument("--submit", action="store_true", help="Actually invoke sbatch after all evidence checks")
    args = parser.parse_args(argv)
    try:
        plan = prepare(args.preflight, args.gate_a, args.budget_approval, args.concurrency, args.output_root)
        print(f"21 tasks; 4 CPUs/task; concurrency {plan['concurrency']}; {plan['memory_gib_per_task']} GiB/task; "
              f"{slurm_time(plan['time_seconds_per_task'])}/task", flush=True)
        print(f"Approved projection: {plan['projected_total_core_hours']:.2f} core-hours "
              f"({plan['percentage_of_remaining_balance']:.2f}% of recorded remaining balance).", flush=True)
        print(shell_command(plan), flush=True)
        if not args.submit:
            print("Dry-run only: no files or jobs created. Recheck live occupancy/balance; add --submit when ready.")
            return 0
        job_id = submit(plan)
        print(f"Submitted array {job_id}; record: {plan['submission_record']}")
        print(f"Monitor: squeue -j {job_id}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Run 1 submission refused or failed: {exc}\nNo automatic retry is performed.\n")


if __name__ == "__main__":
    raise SystemExit(main())
