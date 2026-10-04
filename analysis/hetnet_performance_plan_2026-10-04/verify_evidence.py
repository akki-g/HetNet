"""Read-only checks for PERFORMANCE_PLAN.md; never imports or runs training.

Run from any directory with Python 3. Requires the local copied evidence inputs
listed in evidence.json; these large run archives are not bundled with the plan.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / "publication_reconstruction/PERFORMANCE_PLAN.md"
WORKLOADS = ("pp_real", "pcp_real", "fc_real", "pcp_binary")
RUNS = {
    "mac": "runs/hetnet_preflight_mac_20261003_234455_727228",
    "stokes": "stokes_runs/hetnet_preflight_903502",
}


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measurements():
    results = {}
    for workload in WORKLOADS:
        row = {}
        signatures = {}
        for platform, root in RUNS.items():
            run = ROOT / root / workload / "seed991"
            preflight = read_json(run / "preflight.json")
            status = read_json(run / "run_status.json")
            protocol = read_json(run / "protocol.json")
            updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
            require([u["update"] for u in updates] == list(range(1, 101)), f"{run}: update sequence")
            steps = sum(u["steps"] for u in updates)
            episodes = sum(u["episodes"] for u in updates)
            seconds = sum(u["wall_time_seconds"] for u in updates)
            require((steps, episodes) == (preflight["counts"]["env_steps"], preflight["counts"]["episodes"]), f"{run}: counts")
            require(status["counts"] == preflight["counts"], f"{run}: status counts")
            require(seconds == status["training_update_seconds"], f"{run}: update timing")
            require(preflight["update_seconds"] == [u["wall_time_seconds"] for u in updates], f"{run}: update list")
            rate = steps / seconds
            budget = 28_000_000 if workload == "fc_real" else 40_000_000
            projection = budget * preflight["segment_wall_seconds"] / steps / 3600
            require(rate == preflight["measured_update_steps_per_second"], f"{run}: reported rate")
            require(projection == preflight["projected_segment_hours_including_logging_and_checkpoints_excluding_startup"], f"{run}: projection")
            require(preflight["frozen_checkpoint_probe"]["parameters_unchanged"] is True, f"{run}: recorded frozen probe")
            require((run / "exit_code.txt").read_text().strip() == "0", f"{run}: exit code")
            require(protocol["model_spec"] == "supplement-v1" and protocol["env_version"] == "corrected-v1", f"{run}: selected baseline")
            segment = preflight["segment_wall_seconds"]
            require(segment == status["segment_wall_time_seconds"], f"{run}: segment timing")
            resources = status["resources"]
            collectors = resources["collectors"]
            require(len(collectors) == resources["collector_count"] == 4, f"{run}: collector count")
            require(all(c["torch_threads"] == 1 for c in collectors), f"{run}: Torch threads")
            user_cpu = sum(c["user_cpu_seconds"] for c in collectors)
            system_cpu = sum(c["system_cpu_seconds"] for c in collectors)
            cpu = user_cpu + system_cpu
            unit_scale = {"bytes": 1, "KiB": 1024}
            rss_bytes = sum(c["max_rss"] * unit_scale[c["max_rss_unit"]] for c in collectors)
            checkpoint_records = [json.loads(line) for line in (run / "checkpoint_records.jsonl").read_text().splitlines()]
            checkpoints = list(run.glob("checkpoints/*/run1/*.pt"))
            require(len(checkpoints) == len(checkpoint_records) == 1, f"{run}: checkpoint inventory")
            checkpoint_record = checkpoint_records[0]
            require(sha256(checkpoints[0]) == checkpoint_record["checkpoint_sha256"], f"{run}: checkpoint byte identity")
            probe = read_json(run / "checkpoint_probe/report.json")
            require(probe["parameters_unchanged"] is True, f"{run}: recorded probe immutability")
            require(probe["checkpoint_sha256"] == checkpoint_record["checkpoint_sha256"], f"{run}: probe checkpoint identity")
            require(probe["source_sha256"] == sha256(run / "source_manifest.json"), f"{run}: probe source identity")
            row[platform] = {"steps": steps, "episodes": episodes, "updates": len(updates),
                             "summed_update_seconds": seconds, "update_steps_per_second": rate,
                             "segment_seconds": segment, "startup_seconds": preflight["startup_seconds"],
                             "projected_budget_hours": projection,
                             "update_fraction_of_segment": seconds / segment,
                             "outside_update_seconds": segment - seconds,
                             "outside_update_fraction_of_segment": (segment - seconds) / segment,
                             "full_checkpoint_seconds": status["checkpoint_seconds"],
                             "full_checkpoint_fraction_of_segment": status["checkpoint_seconds"] / segment,
                             "checkpoint_record_seconds": sum(preflight["checkpoint_seconds"]),
                             "remaining_outside_update_seconds": segment - seconds - status["checkpoint_seconds"],
                             "collector_user_cpu_seconds": user_cpu,
                             "collector_system_cpu_seconds": system_cpu,
                             "collector_cpu_seconds_per_joint_step": cpu / steps,
                             "collector_cpu_seconds_over_segment_seconds": cpu / segment,
                             "approximate_fraction_of_four_cpu_capacity": cpu / segment / 4,
                             "system_fraction_of_collector_cpu": system_cpu / cpu,
                             "sum_individual_peak_rss_gib_not_simultaneous_peak": rss_bytes / 2**30,
                             "collector_affinity_masks": [c["cpu_affinity"] for c in collectors],
                             "first20_update_steps_per_second": sum(u["steps"] for u in updates[:20]) / sum(u["wall_time_seconds"] for u in updates[:20]),
                             "last20_update_steps_per_second": sum(u["steps"] for u in updates[-20:]) / sum(u["wall_time_seconds"] for u in updates[-20:]),
                             "source_manifest_sha256": sha256(run / "source_manifest.json"),
                             "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                             "checkpoint_bytes": checkpoints[0].stat().st_size,
                             "recorded_git_head": preflight["runtime"]["git_head"]}
            signatures[platform] = read_json(run / "initial_signature.json")
        left = {p["name"]: p for p in signatures["mac"]["parameters"]}
        right = {p["name"]: p for p in signatures["stokes"]["parameters"]}
        require(left.keys() == right.keys(), f"{workload}: initial parameter names")
        require(all((left[n]["shape"], left[n]["dtype"]) == (right[n]["shape"], right[n]["dtype"]) for n in left), f"{workload}: initial shapes/dtypes")
        different = [n for n in left if left[n]["sha256"] != right[n]["sha256"]]
        require(len(different) == 36 and all(left[n]["dtype"] == "torch.float32" for n in different), f"{workload}: initial hash differences")
        row["initial_differing_float32_parameter_names"] = different
        row["mac_to_stokes_update_rate_ratio"] = row["mac"]["update_steps_per_second"] / row["stokes"]["update_steps_per_second"]
        results[workload] = row
    return results


def launcher_measurements():
    summary = read_json(ROOT / RUNS["mac"] / "local_summary.json")
    jobs = summary["jobs"]
    starts = [datetime.fromisoformat(jobs[w]["started_at"]).timestamp() for w in WORKLOADS]
    finishes = [datetime.fromisoformat(jobs[w]["finished_at"]).timestamp() for w in WORKLOADS]
    require(all(jobs[w]["exit_code"] == 0 for w in WORKLOADS), "Mac launcher exit codes")
    return {"start_spread_seconds_from_iso_timestamps": max(starts) - min(starts),
            "overall_seconds_from_iso_timestamps": max(finishes) - min(starts),
            "four_launcher_overlap_seconds_from_iso_timestamps": min(finishes) - max(starts)}


def evidence_register(plan):
    result = {}
    for reference, description in re.findall(r"^- \*\*([PCEWT]\d+):\*\* (.+)$", plan, flags=re.MULTILINE):
        targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", description)
        result[reference] = {
            "description": description,
            "external_urls": [target for target in targets if "://" in target],
            "local_paths": [str((PLAN.parent / target.split("#", 1)[0]).resolve().relative_to(ROOT))
                            for target in targets if "://" not in target and not target.startswith("#")],
        }
    return result


def verify_revision(inventory):
    """Keep intentional guard edits traceable to the plan's frozen source bytes."""
    if "revision_record" not in inventory:
        return 0
    receipt = read_json(ROOT / inventory["revision_record"])
    prior_path = ROOT / receipt["prior_inventory"]
    require(sha256(prior_path) == receipt["prior_inventory_sha256"], "Changed prior evidence inventory")
    prior = read_json(prior_path)
    require(sha256(ROOT / receipt["change_patch"]) == receipt["change_patch_sha256"], "Changed guard revision patch")
    changes = {name for name, before in prior["local_sha256"].items()
               if inventory["local_sha256"].get(name) != before}
    require(changes == set(receipt["files"]), "Source changes differ from their explicit revision receipt")
    for name, record in receipt["files"].items():
        require(record["before_sha256"] == prior["local_sha256"][name], f"Prior identity: {name}")
        require(record["after_sha256"] == inventory["local_sha256"][name], f"Revised identity: {name}")
        require(sha256(ROOT / record["before_snapshot"]) == record["before_sha256"], f"Frozen source: {name}")
        require(not name.startswith("publication_reconstruction/runtime/"), f"Guard-only revision changed runtime: {name}")
    baseline = read_json(ROOT / receipt["baseline_manifest"])
    for name, expected in baseline["files"].items():
        require(sha256(ROOT / receipt["baseline_root"] / name) == expected, f"Frozen baseline: {name}")
        require(expected == prior["local_sha256"][name], f"Baseline differs from prior inventory: {name}")
    require(inventory["recomputed_measurements"] == prior["recomputed_measurements"], "Guard revision changed timing evidence")
    return len(changes)


def main():
    inventory = read_json(HERE / "evidence.json")
    for relative, expected in inventory["local_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Changed evidence: {relative}")
    revisions = verify_revision(inventory)
    prior_inputs = read_json(ROOT / inventory["archived_run_input_inventory"])
    require(len(prior_inputs) == inventory["archived_run_input_count"], "Archived input inventory size")
    for relative, expected in prior_inputs.items():
        require(sha256(ROOT / relative) == expected, f"Changed archived input: {relative}")
    origins = read_json(ROOT / "publication_reconstruction/ORIGINS.json")
    for name, entry in origins["files"].items():
        relative = Path("publication_reconstruction/runtime") / name
        require(sha256(ROOT / relative) == entry["current_sha256"], f"Runtime provenance: {relative}")
    observed = measurements()
    require(observed == inventory["recomputed_measurements"], "Recomputed timing/resource/signature measurements differ")
    require(launcher_measurements() == inventory["recomputed_launcher_measurements"], "Launcher measurements differ")
    plan = PLAN.read_text()
    require(evidence_register(plan) == inventory["evidence_register"], "Plan evidence register differs from inventory")
    links = 0
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", plan):
        if "://" not in target and not target.startswith("#"):
            require((PLAN.parent / target.split("#", 1)[0]).exists(), f"Missing local link: {target}")
            links += 1
    for label, workload in zip(("PP Real", "PCP Real", "FC Real", "PCP Binary"), WORKLOADS):
        row = observed[workload]
        values = (row["stokes"]["update_steps_per_second"], row["mac"]["update_steps_per_second"],
                  row["mac_to_stokes_update_rate_ratio"], row["stokes"]["projected_budget_hours"])
        require(f"| {label} | " + " | ".join(f"{value:.2f}" for value in values) + " |" in plan, f"Plan table: {label}")
    require(sum(r["mac"]["steps"] for r in observed.values()) == 893647, "Mac total steps")
    require(sum(r["mac"]["episodes"] for r in observed.values()) == 10950, "Mac total episodes")
    require(all(math.isfinite(r[p]["update_steps_per_second"]) for r in observed.values() for p in RUNS), "Finite throughput")
    print(json.dumps({"status": "passed", "hashed_files": len(inventory["local_sha256"]),
                      "guard_source_revisions_with_frozen_predecessors": revisions,
                      "unchanged_archived_run_inputs": len(prior_inputs),
                      "runtime_manifest_files": len(origins["files"]),
                      "local_links": links, "workload_pairs": len(observed),
                      "evidence_references": len(inventory["evidence_register"]),
                      "checkpoint_byte_identities": len(observed) * len(RUNS),
                      "scope": "File identity, local links, ledger/resource arithmetic, launcher timestamps and initial signature metadata; no policy execution or new checkpoint tensor audit."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
