"""Read-only provenance audit of the synced frozen PCP evaluation artifact."""
import collections
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "stokes_runs/frozen_pcp_30m/frozen_pcp_30m_20261002_slurm"
OUT = Path(__file__).with_suffix(".json")
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(BASE / "manifest.json")
    source = read(BASE / "source_manifest.json")
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in BASE.rglob("*") if p.is_file()}
    problems = []

    def check(label, passed):
        if not passed:
            problems.append(label)
        return bool(passed)

    source_files = source["files"]
    actual_source = {str(p.relative_to(BASE / "source")): sha(p)
                     for p in (BASE / "source").rglob("*") if p.is_file()}
    digest = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    source_checks = {
        "file_count": len(source_files),
        "all_source_bytes_match": check("source bytes/inventory", actual_source == source_files),
        "combined_digest_recomputed": digest,
        "combined_digest_matches": check("combined digest", digest == source["sha256"]
                                         == manifest["evaluation_source_sha256"]),
        "preparation_source_matches": check("preparation source",
            sha(BASE / "preparation_source.py") == manifest["preparation_source_sha256"]
            == source_files["scripts/prepare_pcp_frozen.py"]),
        "artifact_hashes_match": check("prepared artifact hashes", all(
            sha(BASE / name) == expected for name, expected in manifest["artifact_sha256"].items())),
    }
    # Regenerate panel assignments from the exact archived scenario implementation.
    path = BASE / "source/softrole/scenarios.py"
    spec = importlib.util.spec_from_file_location("archived_scenarios_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    panels, panel_checks = {}, {}
    for condition in ("nominal", "failure"):
        cfg = manifest[f"{condition}_panel"]
        panel = read(BASE / f"scenarios_{condition}.json")
        compositions = cfg.get("compositions", [cfg.get("composition")])
        count = cfg.get("episodes_per_composition", cfg.get("episodes_per_condition"))
        regenerated = []
        for p, a in compositions:
            seed = int(np.random.SeedSequence([cfg["seed"], p, a]).generate_state(1)[0])
            regenerated.extend(dataclasses.asdict(s) for s in module.make_scenarios(
                seed, count, [(p, a)], cfg.get("failure_probability", 0),
                cfg.get("failure_window", (10, 30))))
        panels[condition] = panel
        panel_checks[condition] = {
            "count": len(panel),
            "compositions": {f"{p}P{a}A": n for (p, a), n in
                sorted(collections.Counter((s["num_p"], s["num_a"]) for s in panel).items())},
            "unique_ids": check(f"{condition} unique IDs", len({s["scenario_id"] for s in panel}) == len(panel)),
            "regenerates_exactly": check(f"{condition} regeneration", regenerated == panel),
            "declared_count_matches": check(f"{condition} count", len(panel) == cfg["count"]),
        }
    policies = {(p["model"], p["training_seed"]): p for p in manifest["policies"]}
    sbatch = (BASE / "submit.sbatch").read_text()
    commands = dict((int(i), shlex.split(line)) for i, line in re.findall(
        r"^  (\d+)\)\n    (exec srun .*?) ;;$", sbatch, flags=re.M))
    reports, report_checks = {}, []
    expected_paths = set()
    for index, job in enumerate(manifest["jobs"]):
        condition = job["condition"]
        key = (job["model"], job["training_seed"])
        policy = policies[key]
        panel = panels["nominal" if condition == "nominal" else "failure"]
        path = BASE / "results" / Path(job["output"]).name
        expected_paths.add(path)
        report = read(path)
        reports[(key, condition)] = report
        metadata_keys = ("checkpoint", "checkpoint_sha256", "training_seed", "config", "model_config",
                         "environment_version", "source_sha256", "checkpoint_progress")
        mismatch = [name for name in metadata_keys if report[name] != policy[name]]
        check(f"report metadata {index}", not mismatch)
        expected_command = ["exec", "srun", "bash",
            "/lustre/fs1/home/ak565492/HetNet/scripts/softrole_evaluate.sh",
            job["checkpoint"], job["output"], "--scenarios", job["scenarios"]]
        if condition != "nominal":
            expected_command += ["--trace"]
        if condition == "sham":
            expected_command += ["--sham"]
        checks = {
            "array_index": index, "path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "model": key[0], "seed": key[1], "condition": condition,
            "metadata_matches_manifest": not mismatch,
            "metadata_mismatches": mismatch,
            "array_command_matches_job": check(f"command {index}", commands[index] == expected_command),
            "scenarios_exact_panel_match": check(f"panel {index}", report["scenarios"] == panel),
            "episodes_match_panel": check(f"episodes {index}", report["episodes"] == len(report["per_episode"]) == len(panel)),
            "episode_scenario_fields_match": check(f"episode assignment {index}", all(
                all(episode[k] == scenario[k] for k in scenario)
                for episode, scenario in zip(report["per_episode"], panel))),
            "sham_label_matches": check(f"sham {index}", report["sham"] == (condition == "sham")),
            "intervention_none": check(f"intervention {index}", report["intervention"] == "none"),
            "evaluator_full_source_matches": check(f"source {index}", report["evaluator"]["source"] == source),
            "trace_presence_matches_command": check(f"trace {index}", all(
                ("trace" in episode) == (condition != "nominal") for episode in report["per_episode"])),
            "format_version": report["format_version"], "evaluation_version": report["evaluation_version"],
        }
        report_checks.append(checks)
    check("result inventory", expected_paths == set((BASE / "results").glob("*.json")))
    runtimes = [r["evaluator"]["runtime"] for r in reports.values()]
    check("matching evaluator runtimes", all(r == runtimes[0] for r in runtimes))
    policy_checks = []
    for key, policy in policies.items():
        report_set = [reports[(key, c)] for c in ("nominal", "failure", "sham")]
        check(f"frozen signature {key}", len({r["model_signature"] for r in report_set}) == 1)
        local_run = ROOT / "stokes_runs/runs/softrole_primary" / f"pcp_{key[0]}" / f"seed{key[1]}"
        local_ckpt = local_run / "checkpoints" / Path(policy["checkpoint"]).name
        local_ledger = local_run / "metrics.jsonl"
        local_rows = [json.loads(line) for line in local_ledger.read_text().splitlines()] if local_ledger.exists() else []
        policy_checks.append({
            "model": key[0], "seed": key[1], "checkpoint_sha256_recorded": policy["checkpoint_sha256"],
            "checkpoint_progress": policy["checkpoint_progress"],
            "all_three_condition_model_signatures_match": len({r["model_signature"] for r in report_set}) == 1,
            "threshold_satisfied_recorded": policy["checkpoint_progress"]["total_steps"] >= manifest["min_steps"],
            "selected_checkpoint_present_locally": local_ckpt.exists(),
            "local_ledger": str(local_ledger.relative_to(ROOT)),
            "local_ledger_sha256": sha(local_ledger) if local_ledger.exists() else None,
            "local_ledger_matches_preparation_hash": sha(local_ledger) == policy["metrics_sha256"] if local_ledger.exists() else False,
            "local_ledger_last_epoch": local_rows[-1]["epoch"] if local_rows else None,
        })
    rehashed = {str(p.relative_to(ROOT)): sha(p) for p in BASE.rglob("*") if p.is_file()}
    check("inputs unchanged during audit", rehashed == inputs)
    audit = {
        "schema_version": 1, "input_root": str(BASE.relative_to(ROOT)),
        "scope": "Read-only provenance and assignment audit; no model execution, training or cluster action.",
        "all_available_checks_pass": not problems, "problems": problems,
        "source": source_checks, "panels": panel_checks, "reports": report_checks,
        "policies": policy_checks, "runtime": runtimes[0], "report_count": len(reports),
        "episode_records": sum(r["episodes"] for r in reports.values()),
        "input_sha256": inputs,
        "limitations": [
            "Selected epoch1500 training checkpoints and contemporaneous full ledgers are absent from this sync. Recorded report/manifest identities agree, but checkpoint bytes/tensors and first-available-threshold selection cannot be independently reverified from these artifacts alone.",
            "Manifest parameter_sha256 and report model_signature use different documented hash algorithms (hetnet_ext/signatures.py:31 versus softrole/evaluate.py:16); inequality is not evidence of mutation. Evaluation code compares its own before/after signatures and aborts on a difference.",
            "Manifest submitted:false describes preparation only, not scheduler state. Completed reports are present; no sacct data was supplied.",
            "The archived evaluator used the live repository at execution; its complete recorded file map matches the supplied source archive. This audit checks recorded source identity, not reconstructed scheduler execution.",
        ],
        "evidence": {
            "selection_implementation": "preparation_source.py:47-115",
            "source_digest_algorithm": "source/softrole/train.py:52-84",
            "scenario_generation": "source/softrole/__main__.py:100-120; source/softrole/scenarios.py:49-88",
            "evaluation_immutability_guards": "source/softrole/evaluate.py:76-95",
            "array_mapping": "submit.sbatch:23-60",
        },
    }
    OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "passed": not problems,
                      "problems": problems, "source_files": len(source_files),
                      "reports": len(reports), "episode_records": audit["episode_records"]}, indent=2))


if __name__ == "__main__":
    main()
