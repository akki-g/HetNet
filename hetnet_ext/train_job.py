"""Fail-closed Slurm launcher for original HetNet Phase A jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time

from .grid import (ROOT, DEFAULT_GRID, build_command, code_inventory, code_sha256,
                   entries_for_mode, file_sha256, run_directory)


GATE_CHECKS = (
    "environment", "smoke_all_compositions", "shared_weights", "determinism",
    "cross_composition_load", "grid_and_scripts", "gradient_storage",
    "fresh_gradient_aggregation", "clean_lock_environment", "baseline_diff", "unit_contracts",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checked_number(value: object, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} is outside the permitted range")
    return float(value)


def checked_partition_time_limit(value: object) -> float | str:
    """Only a verified partition cap may use the exact JSON sentinel unlimited."""
    if type(value) is str and value == "unlimited":
        return value
    return checked_number(value, "max_wall_time_seconds (positive finite number or 'unlimited')")


def within_partition_time_limit(requested_seconds: object, maximum: object) -> bool:
    requested = checked_number(requested_seconds, "finite job wall-time request")
    limit = checked_partition_time_limit(maximum)
    return limit == "unlimited" or requested <= limit


def validate_preflight(data: dict) -> None:
    if data.get("schema_version") != 1 or data.get("verified") is not True:
        raise ValueError("Stokes preflight must be schema_version 1 with verified=true")
    if data.get("cluster", "").lower() != "stokes" or data.get("partition") != "normal":
        raise ValueError("Only Stokes / normal is authorized; do not switch clusters or queues")
    checked_partition_time_limit(data.get("max_wall_time_seconds"))
    checked_number(data.get("remaining_core_hours"), "remaining_core_hours")
    for name in ("max_concurrent_jobs", "max_submit_jobs"):
        value = data.get(name)
        if value != "unlimited" and (type(value) is not int or value <= 0):
            raise ValueError(f"Preflight {name} must be a verified positive integer or 'unlimited'")
    if type(data.get("account_required")) is not bool:
        raise ValueError("Preflight must explicitly state account_required")
    if data["account_required"] and not data.get("account"):
        raise ValueError("Preflight requires an account, but none is recorded")
    if not data.get("verified_at_utc") or not data.get("evidence_paths"):
        raise ValueError("Preflight must identify timestamp and raw evidence paths")
    if data.get("python_major_minor") != "3.12" or data.get("compute_network_verified") is not True:
        raise ValueError("Confirm Python 3.12 and compute-node network access before launching")


def validate_gate(data: dict, expected_code: str, expected_lock: str) -> None:
    if data.get("schema_version") != 1 or data.get("passed") is not True:
        raise ValueError("Gate A has not passed; cluster jobs must wait")
    failed = [name for name in GATE_CHECKS if data.get("checks", {}).get(name) is not True]
    if failed:
        raise ValueError("Gate A checks are missing or failed: " + ", ".join(failed))
    if data.get("code_sha256") != expected_code or data.get("uv_lock_sha256") != expected_lock:
        raise ValueError("Gate A evidence is stale: code/config/test or uv.lock hash differs")


def validate_approval(path: Path, expected_code: str, expected_lock: str, preflight_path: Path) -> dict:
    approval = read_json(path)
    if approval.get("approved") is not True or not approval.get("approved_by") or not approval.get("approved_at_utc"):
        raise ValueError("Full training requires Akki's explicit recorded budget confirmation")
    budget_path = Path(approval.get("budget_file", ""))
    if not budget_path.is_absolute():
        budget_path = path.parent / budget_path
    if not budget_path.is_file() or file_sha256(budget_path) != approval.get("budget_sha256"):
        raise ValueError("Budget approval must reference the exact existing budget-file hash")
    budget = read_json(budget_path)
    preflight = read_json(preflight_path)
    validate_preflight(preflight)
    if budget.get("preflight_sha256") != file_sha256(preflight_path):
        raise ValueError("Budget approval is stale: current Stokes preflight hash differs")
    if budget.get("code_sha256") != expected_code or budget.get("uv_lock_sha256") != expected_lock:
        raise ValueError("Budget was prepared for different scientific code or dependencies")
    projected = checked_number(budget.get("projected_total_core_hours"), "projected budget")
    balance = checked_number(preflight.get("remaining_core_hours"), "current remaining balance")
    if budget.get("remaining_core_hours") != balance:
        raise ValueError("Budget remaining balance differs from the verified current preflight")
    if (projected > 4000 or projected > 0.1 * balance) and approval.get("threshold_override_approved") is not True:
        raise ValueError("Budget crosses the 4,000 core-hour / 10% gate; explicit override approval required")
    if budget.get("within_partition_cap") is not True:
        raise ValueError("Projected job exceeds the partition cap; resume/recipe decision required")
    return {"approval": approval, "budget": budget, "budget_file": str(budget_path.resolve())}


def parse_slurm_time(value: str) -> int:
    """Parse scontrol's [days-]HH:MM:SS; reject UNLIMITED/unknown values."""
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
        if days < 0:
            raise ValueError("Invalid negative Slurm day value")
    pieces = [int(v) for v in value.split(":")]
    if len(pieces) != 3 or any(v < 0 for v in pieces):
        raise ValueError("Expected Slurm time in [days-]HH:MM:SS")
    hours, minutes, seconds = pieces
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Invalid Slurm minute/second value")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def verify_allocation(preflight: dict, memory_gb: float, time_seconds: int,
                      array_concurrency: int) -> dict:
    """Query the current local Slurm allocation, never submit a remote job."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise ValueError("train_job executes only inside a Slurm allocation; use grid --dry-run locally")
    if os.environ.get("SLURM_CLUSTER_NAME", "").lower() != "stokes":
        raise ValueError("SLURM_CLUSTER_NAME must report Stokes")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4":
        raise ValueError("Phase A requires --cpus-per-task=4 for four collectors")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
    fields = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    if any(fields.get(key) != value for key, value in {"NumCPUs": "4", "NumNodes": "1", "NumTasks": "1"}.items()):
        raise ValueError("Actual allocation must have exactly four total CPUs, one node and one task")
    if (os.environ.get("SLURM_JOB_NUM_NODES") != "1" or os.environ.get("SLURM_NTASKS") != "1"
            or os.environ.get("SLURM_JOB_CPUS_PER_NODE") != "4"):
        raise ValueError("Slurm environment must confirm one node, one task and four allocated CPUs")
    if fields.get("Partition") != "normal":
        raise ValueError("Actual allocation must use normal")
    if type(array_concurrency) is not int or array_concurrency < 1:
        raise ValueError("Declare a positive array concurrency")
    limit = preflight["max_concurrent_jobs"]
    if limit != "unlimited" and array_concurrency > limit:
        raise ValueError("Array concurrency exceeds the verified user/account limit")
    if int(fields.get("ArrayTaskThrottle", "0")) != array_concurrency:
        raise ValueError("Submit with an explicit --array=start-end%concurrency matching the declaration")
    actual_seconds = parse_slurm_time(fields.get("TimeLimit", "unknown"))
    if actual_seconds != time_seconds or not within_partition_time_limit(actual_seconds, preflight["max_wall_time_seconds"]):
        raise ValueError("Actual --time differs from declared resources or exceeds verified partition cap")
    actual_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    if actual_mb != math.ceil(memory_gb * 1024):
        raise ValueError("Actual --mem differs from declared GiB; pass explicit --mem=<GiB>G")
    if preflight["account_required"] and fields.get("Account") != preflight["account"]:
        raise ValueError("Allocated account differs from the verified preflight account")
    return {"scontrol": raw.strip(), "time_seconds": actual_seconds,
            "memory_mib": actual_mb, "cpus_per_task": 4, "array_concurrency": array_concurrency,
            "slurm_stdout": fields.get("StdOut"), "slurm_stderr": fields.get("StdErr")}


def _capture(command: list[str], *, required: bool = False) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if required and result.returncode:
        raise ValueError(f"Provenance command failed: {command!r}: {result.stdout}")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "calibration"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--gate-a", type=Path, required=True)
    parser.add_argument("--budget-approval", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--requested-memory-gb", type=float, required=True)
    parser.add_argument("--requested-time-seconds", type=int, required=True)
    parser.add_argument("--array-concurrency", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.grid.resolve() != DEFAULT_GRID.resolve():
            raise ValueError("Production jobs use the committed configs/run1_grid.json only")
        entries = entries_for_mode(args.mode, args.grid)
        if not 0 <= args.index < len(entries):
            raise ValueError(f"Array index outside 0..{len(entries) - 1}")
        checked_number(args.requested_memory_gb, "requested memory")
        checked_number(args.requested_time_seconds, "requested time")
        if args.requested_time_seconds % 60:
            raise ValueError("Declare time in complete minutes (seconds must be divisible by 60)")
        preflight = read_json(args.preflight)
        validate_preflight(preflight)
        if preflight["max_submit_jobs"] != "unlimited" and len(entries) > preflight["max_submit_jobs"]:
            raise ValueError("Array size exceeds the verified submission limit; ask before splitting the study")
        inventory = code_inventory()
        code_hash, lock_hash = code_sha256(), file_sha256(ROOT / "uv.lock")
        validate_gate(read_json(args.gate_a), code_hash, lock_hash)
        approved = None
        if args.mode == "train":
            if args.budget_approval is None:
                raise ValueError("--budget-approval is required for full training")
            approved = validate_approval(args.budget_approval, code_hash, lock_hash, args.preflight)
        if approved:
            if (args.requested_time_seconds != approved["budget"]["array_uniform_time_seconds"]
                    or args.requested_memory_gb != approved["budget"]["array_uniform_memory_gb"]):
                raise ValueError("Training array resources must match the approved budget's uniform request")
        allocation = verify_allocation(preflight, args.requested_memory_gb, args.requested_time_seconds,
                                       args.array_concurrency)
        entry = entries[args.index]
        directory = run_directory(entry, args.mode, args.output_root)
        # Atomic refusal also protects against two identical array indices racing.
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Preflight refused: {exc}\nNo training started. Resolve the evidence/resource issue first.\n")

    command = build_command(entry, directory, mode=args.mode, path=args.grid)
    recipe = read_json(args.grid)["recipe"]
    target_epochs = recipe["num_epochs"] if args.mode == "train" else read_json(args.grid)["calibration"]["num_epochs"]
    provenance = {
        "schema_version": 1, "status": "starting", "started_at_utc": utc_now(),
        "mode": args.mode, "entry": entry.as_dict(), "command": command,
        "git_commit": _capture(["git", "rev-parse", "HEAD"], required=True),
        "git_status": _capture(["git", "status", "--porcelain"], required=True),
        "code_sha256": code_hash, "code_inventory": inventory, "uv_lock_sha256": lock_hash,
        "python": sys.executable, "python_version": platform.python_version(),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"), "hostname": socket.gethostname(),
        "cpu": _capture(["lscpu"]), "allocation": allocation,
        "bootstrap_started_at_utc": os.environ.get("HETNET_BOOTSTRAP_STARTED_AT"),
        "bootstrap_wall_time_seconds": os.environ.get("HETNET_BOOTSTRAP_SECONDS"),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "preflight_file": str(args.preflight.resolve()), "preflight_sha256": file_sha256(args.preflight),
        "gate_a_file": str(args.gate_a.resolve()), "gate_a_sha256": file_sha256(args.gate_a),
        "budget_approval": approved, "stdout_file": str(directory / "stdout.log"),
        "expected_epochs": target_epochs, "thread_settings": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HETNET_TORCH_THREADS": "1"},
    }
    provenance["git_dirty"] = bool(provenance["git_status"])
    write_json(directory / "config.json", {"entry": entry.as_dict(), "mode": args.mode,
                                          "recipe": recipe, "num_epochs": target_epochs, "command": command})
    write_json(directory / "provenance.json", provenance)
    child = None
    started = time.monotonic()
    interrupted = []

    def stop(signum, _frame):
        interrupted.append(signum)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HETNET_TORCH_THREADS="1",
                   PYTHONUNBUFFERED="1")
        with (directory / "stdout.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            assert child.stdout is not None
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = child.wait()
        if interrupted or returncode:
            raise RuntimeError(f"Training exited {returncode}; signals received {interrupted}")
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
        if [row["epoch"] for row in metrics] != list(range(1, target_epochs + 1)):
            raise RuntimeError("Incomplete or duplicate epoch metrics; preserving failed attempt")
        checkpoints = list((directory / "checkpoints").rglob(f"model_ep{target_epochs}.pt"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Expected one final checkpoint, found {len(checkpoints)}")
        final = checkpoints[0]
        signature_file = Path(str(final) + ".signature.json")
        if not signature_file.is_file() or not (directory / "initial_signature.json").is_file():
            raise RuntimeError("Checkpoint/initial parameter-signature evidence missing")
        if not (directory / "resolved_args.json").is_file():
            raise RuntimeError("Resolved upstream args missing")
        final_signature = read_json(signature_file)
        epoch_signatures = [json.loads(line) for line in (directory / "epoch_signatures.jsonl").read_text().splitlines()
                            if line.strip()]
        if [row.get("epoch") for row in epoch_signatures] != list(range(1, target_epochs + 1)):
            raise RuntimeError("Incomplete or duplicate epoch parameter signatures")
        if final_signature != epoch_signatures[-1].get("signature"):
            raise RuntimeError("Final checkpoint signature differs from the final training epoch")
        if code_sha256() != code_hash or file_sha256(ROOT / "uv.lock") != lock_hash:
            raise RuntimeError("Scientific code or dependency lock changed during execution; run is invalid")
        provenance.update(status="complete", final_checkpoint=str(final),
                          final_checkpoint_sha256=file_sha256(final),
                          final_parameter_signature=final_signature,
                          resolved_args=read_json(directory / "resolved_args.json"),
                          resolved_args_sha256=file_sha256(directory / "resolved_args.json"),
                          initial_parameter_signature=read_json(directory / "initial_signature.json"))
        result = 0
    except Exception as exc:
        provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"Run failed and preserved at {directory}: {exc}", file=sys.stderr)
        result = 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        provenance.update(ended_at_utc=utc_now(), launcher_wall_time_seconds=time.monotonic() - started)
        write_json(directory / "provenance.json", provenance)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
