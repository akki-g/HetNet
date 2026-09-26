"""Project Phase A costs from completed endpoint calibrations and explicit measurements."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from .grid import ROOT, code_sha256, file_sha256, load_grid
from .train_job import checked_number, read_json, utc_now, validate_preflight


ENDPOINTS = ("2P1A", "4P6A")


def summarize_calibration(directory: Path, measurements: dict, expected_code: str,
                          expected_lock: str) -> dict:
    provenance = read_json(directory / "provenance.json")
    if provenance.get("status") != "complete" or provenance.get("mode") != "calibration":
        raise ValueError(f"Calibration is incomplete/failed: {directory}")
    if provenance.get("code_sha256") != expected_code or provenance.get("uv_lock_sha256") != expected_lock:
        raise ValueError(f"Calibration code/lock differs from current study: {directory}")
    config = read_json(directory / "config.json")
    if config.get("num_epochs") != 20:
        raise ValueError("Calibration must use exactly 20 epochs")
    required_recipe = {"epoch_size": 10, "batch_size": 500, "nprocesses": 4,
                       "max_steps": 80, "dim": 5}
    if any(config.get("recipe", {}).get(k) != v for k, v in required_recipe.items()):
        raise ValueError("Calibration does not use the production E/B/P/H/map settings")
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if [r.get("epoch") for r in rows] != list(range(1, 21)):
        raise ValueError("Calibration must have unique complete epochs 1..20")
    durations = [checked_number(r.get("wall_time_seconds"), "epoch duration") for r in rows]
    steps = [checked_number(r.get("steps"), "epoch steps") for r in rows]
    # Predeclared timing convention: report all epochs, plan from epochs 2..20.
    steady = sorted(durations[1:])
    p90 = steady[math.ceil(0.9 * len(steady)) - 1]
    required = ("whole_job_peak_rss_gb", "setup_seconds", "allocation_wall_seconds")
    values = {name: checked_number(measurements.get(name), name,
                                   positive=name in {"whole_job_peak_rss_gb", "allocation_wall_seconds"})
              for name in required}
    if not measurements.get("peak_rss_source") or not measurements.get("accounting_evidence"):
        raise ValueError("Whole-job memory and accounting measurements require source/evidence paths")
    checkpoint = Path(provenance["final_checkpoint"])
    if not checkpoint.is_file():
        # A downloaded artifact tree retains its original cluster provenance paths.
        relocated = list((directory / "checkpoints").rglob(checkpoint.name))
        if len(relocated) == 1:
            checkpoint = relocated[0]
    if not checkpoint.is_file() or file_sha256(checkpoint) != provenance.get("final_checkpoint_sha256"):
        raise ValueError("Calibration final checkpoint/hash missing or changed")
    saves_path = directory / "checkpoint_records.jsonl"
    saves = [json.loads(line) for line in saves_path.read_text().splitlines() if line.strip()]
    if len(saves) != 1 or saves[0].get("epoch") != 20:
        raise ValueError("The 20-epoch calibration must record exactly its final checkpoint save")
    save = saves[0]
    if (Path(save.get("path", "")).name != checkpoint.name
            or save.get("bytes") != checkpoint.stat().st_size
            or not save.get("parameter_sha256")
            or save["parameter_sha256"] != provenance.get("final_parameter_signature", {}).get("sha256")):
        raise ValueError("Recorded checkpoint save does not match the verified final checkpoint")
    values["checkpoint_save_seconds"] = checked_number(save.get("wall_time_seconds"),
                                                       "checkpoint save duration", positive=False)
    if values["allocation_wall_seconds"] < sum(durations) + values["checkpoint_save_seconds"] + values["setup_seconds"]:
        raise ValueError("Allocation wall time cannot be shorter than measured epochs, saves, and setup")
    return {"directory": str(directory.resolve()), "provenance_sha256": file_sha256(directory / "provenance.json"),
            "checkpoint_records_sha256": file_sha256(saves_path), "checkpoint_save_record": save,
            "mean_seconds_per_epoch": statistics.mean(durations[1:]), "p90_seconds_per_epoch": p90,
            "first_epoch_seconds": durations[0], "all_epoch_durations": durations,
            "actual_steps": sum(steps), "seconds_per_actual_step": sum(durations) / sum(steps),
            "seconds_per_update": sum(durations) / 200,
            "checkpoint_bytes": checkpoint.stat().st_size, "measurements": measurements, **values}


def project(calibrations: dict, preflight: dict, *, evaluation_reserve: float,
            other_reserve: float) -> dict:
    """Endpoint p90 slopes; explicitly modeled endpoint envelope elsewhere."""
    validate_preflight(preflight)
    checked_number(evaluation_reserve, "evaluation reserve", positive=False)
    checked_number(other_reserve, "other reserve", positive=False)
    counts = Counter(e.composition for e in load_grid())
    envelope = max(v["p90_seconds_per_epoch"] for v in calibrations.values())
    max_setup = max(v["setup_seconds"] for v in calibrations.values())
    max_save = max(v["checkpoint_save_seconds"] for v in calibrations.values())
    max_memory = max(v["whole_job_peak_rss_gb"] for v in calibrations.values())
    rows = []
    for composition, seeds in counts.items():
        measured = calibrations.get(composition)
        slope = measured["p90_seconds_per_epoch"] if measured else envelope
        setup = measured["setup_seconds"] if measured else max_setup
        save_seconds = measured["checkpoint_save_seconds"] if measured else max_save
        memory = measured["whole_job_peak_rss_gb"] if measured else max_memory
        estimated = setup + 2000 * slope + 40 * save_seconds
        # Use complete minutes so scheduler time-resolution rounding is explicit.
        requested = 60 * math.ceil(1.3 * estimated / 60)
        rows.append({"composition": composition, "seeds": seeds, "cpus_per_task": 4,
                     "slope_source": "measured_endpoint_p90_epochs_2_to_20" if measured else "modeled_max_endpoint_p90",
                     "seconds_per_epoch": slope, "estimated_wall_seconds": estimated,
                     "requested_wall_seconds": requested, "memory_gb_with_50pct_margin": math.ceil(1.5 * memory),
                     "memory_source": "measured_endpoint" if measured else "modeled_max_endpoint",
                     "estimated_core_hours": seeds * 4 * estimated / 3600,
                     "guarded_core_hours": seeds * 4 * requested / 3600})
    calibration_ch = sum(4 * v["allocation_wall_seconds"] / 3600 for v in calibrations.values())
    total = sum(r["guarded_core_hours"] for r in rows) + calibration_ch + evaluation_reserve + other_reserve
    remaining = preflight["remaining_core_hours"]
    return {"schema_version": 1, "created_at_utc": utc_now(), "rows": rows,
            "projection_method": "p90 of epochs 2..20; intermediate rosters use endpoint envelope, not measured slopes",
            "extrapolation_warning": "The endpoint envelope is a planning assumption, not a proved bound; recalibrate if inadequate.",
            "calibration_core_hours": calibration_ch, "evaluation_reserve_core_hours": evaluation_reserve,
            "other_reserve_core_hours": other_reserve,
            "projected_total_core_hours": total, "remaining_core_hours": remaining,
            "percentage_of_remaining_balance": 100 * total / remaining,
            "threshold_exceeded": total > 4000 or total > 0.1 * remaining,
            "within_partition_cap": all(r["requested_wall_seconds"] <= preflight["max_wall_time_seconds"] for r in rows),
            "max_wall_time_seconds": preflight["max_wall_time_seconds"],
            "array_uniform_time_seconds": max(r["requested_wall_seconds"] for r in rows),
            "array_uniform_memory_gb": max(r["memory_gb_with_50pct_margin"] for r in rows),
            "full_array_reserved_core_hours": sum(counts.values()) * 4 * max(r["requested_wall_seconds"] for r in rows) / 3600,
            "calibrations": calibrations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True,
                        help="Explicit whole-job memory, setup/allocation times and reserves; saves come from recorder")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="New directory for budget.json and unapproved approval_template.json")
    args = parser.parse_args(argv)
    try:
        preflight = read_json(args.preflight)
        validate_preflight(preflight)
        measured = read_json(args.measurements)
        current_code, lock = code_sha256(), file_sha256(ROOT / "uv.lock")
        calibrations = {name: summarize_calibration(args.calibration_root / name / "seed0",
                                                   measured["jobs"][name], current_code, lock)
                        for name in ENDPOINTS}
        budget = project(calibrations, preflight,
                         evaluation_reserve=measured["evaluation_reserve_core_hours"],
                         other_reserve=measured["other_reserve_core_hours"])
        budget.update(code_sha256=current_code, uv_lock_sha256=lock,
                      preflight_file=str(args.preflight.resolve()), preflight_sha256=file_sha256(args.preflight),
                      measurements_file=str(args.measurements.resolve()), measurements_sha256=file_sha256(args.measurements))
        args.out.mkdir(parents=True, exist_ok=False)
        budget_path = args.out / "budget.json"
        budget_path.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n")
        approval = {"schema_version": 1, "approved": False, "approved_by": None,
                    "approved_at_utc": None, "threshold_override_approved": False,
                    "budget_file": "budget.json", "budget_sha256": file_sha256(budget_path)}
        (args.out / "approval_template.json").write_text(json.dumps(approval, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Budget refused: {exc}\nProvide measured evidence; unknown values are not defaults.\n")
    print(json.dumps({k: budget[k] for k in ("projected_total_core_hours", "percentage_of_remaining_balance",
                                           "threshold_exceeded", "within_partition_cap")}, indent=2))
    print("Budget created without approval. Akki must confirm before full-array submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
