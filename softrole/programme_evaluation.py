"""Frozen evaluation of the locked programme: selection, panels, tasks and summaries.

Development windows are read from each run's training ledgers. The final
frozen panels use the first saved checkpoint at or above 20M joint steps and
common scenario files for the reserved compositions (3,2), (4,1) and (4,2).
Every condition is a fixed combination of panel, intervention and sham flag.
Gate freezes are scheduled only for gated arms (banked, capability): for the
shared and constant actors they are structural no-ops, certified by
``tests/test_softrole_programme_evaluation.py``. Summaries report equal-seed
means with whole-seed bootstrap intervals, within-checkpoint failure-minus-sham
differences, difference-in-differences D_I per intervention, and cross-arm
contrasts against each arm's declared control.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys

if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

import numpy as np
import torch

from softrole import CHECKPOINT_VERSION
from softrole.config import Config
from softrole.programme import (ARM_INDEX, DEVELOPMENT_WINDOWS, MANIFEST_VERSION, PROGRAMME, RUNTIME_KEYS,
                                SEEDS, TOTAL_STEPS, normalized, read_json, sha, window_statistics, write_json)
from softrole.report import contrast_groups, difference_in_differences, protocol_identity, summarize_reports
from softrole.scenarios import Scenario, make_scenarios

EVALUATION_VERSION = 1
FINAL_COMPOSITIONS = ((3, 2), (4, 1), (4, 2))
PANEL_SEED = 2704
EPISODES_PER_COMPOSITION = 500
GATED_MODELS = ("banked", "capability")
SELECTION_RULE = "first_saved_at_or_above"
# name, panel, intervention, intervention_step, sham, gated_only, trace
CONDITIONS = (
    ("nominal", "nominal", "none", None, False, False, False),
    ("comm_off_nominal", "nominal", "comm_off", 0, False, False, False),
    ("failure", "failure", "none", None, False, False, True),
    ("sham", "failure", "none", None, True, False, True),
    ("failure_comm_off", "failure", "comm_off", None, False, False, True),
    ("sham_comm_off", "failure", "comm_off", None, True, False, True),
    ("failure_freeze_affected", "failure", "freeze_affected", None, False, True, True),
    ("sham_freeze_affected", "failure", "freeze_affected", None, True, True, True),
    ("failure_freeze_all", "failure", "freeze_all", None, False, True, True),
    ("sham_freeze_all", "failure", "freeze_all", None, True, True, True),
)
FREEZE_CERTIFICATION = ("Gate freezes pin the gate of the previous step; shared and constant actors have "
                        "input-independent gates, so freeze_affected and freeze_all reproduce the "
                        "unfrozen trajectory exactly (tests/test_softrole_programme_evaluation.py). "
                        "They are scheduled only for banked and capability arms.")
HETNET_ADMISSION = {
    "E2_nominal": "Evaluate each admitted paper-v1 HetNet PCP checkpoint on panels/final_P_A_nominal.json with "
                  "publication_reconstruction/evaluation.py (--run-dir, --checkpoint, --scenarios, --protocol) and "
                  "summarize its reports as a separate stratum labelled by its own configuration.",
    "E5_sensor_loss": "Use panels/final_P_A_failure.json with and without --sham. HetNet has no health input, so "
                      "that evaluator zeroes the victim's typed sensory block: an evaluation-only distribution shift.",
    "labelled_gaps": [
        "typed class inputs and class critics versus label-free capability inputs",
        "per-agent GAE with class-average critic targets versus the team or decomposed learner",
        "Adam 1e-3 and a 40M-step budget versus the programme's selected rate and 20M steps",
        "no health input: sensor loss is an evaluation-only distribution shift for HetNet",
        "DGL/torch-v1 backend tolerance versus exact SoftRole replay",
        "checkpoint selection: first saved at or above HetNet's own budget, not 20M",
    ],
    "status": "documented admission route; this preparer produces no HetNet report",
}


def composition_seed(panel_seed, composition):
    return int(np.random.SeedSequence([int(panel_seed), *map(int, composition)]).generate_state(1)[0])


def panels(panel_seed, episodes, compositions=FINAL_COMPOSITIONS):
    """Common nominal panels and matched step-1 sensor-loss panels (uniform victim)."""
    if type(panel_seed) is not int or panel_seed < 0 or type(episodes) is not int or episodes < 1:
        raise ValueError("panel_seed must be a nonnegative integer and episodes positive")
    result = {}
    for composition in compositions:
        composition = tuple(int(x) for x in composition)
        if composition[0] < 2:
            raise ValueError("Sensor-loss panels need at least two sensing agents")
        seed = composition_seed(panel_seed, composition)
        nominal = make_scenarios(seed, episodes, [composition])
        rng = np.random.default_rng(np.random.SeedSequence([int(panel_seed), *composition, 6006]))
        failure = [replace(s, event_step=1, victim=int(rng.integers(composition[0]))) for s in nominal]
        result[composition] = {"nominal": nominal, "failure": failure, "seed": seed}
    return result


def protocol(target_steps, composition, kind, seed, episodes, scenario_sha256):
    selection = {"rule": SELECTION_RULE, "target_steps": int(target_steps),
                 "lineage": "first saved checkpoint of the programme run at or above the target, verified "
                            "against the preceding saved checkpoint and the metrics ledger"}
    distribution = {"task": "pcp", "compositions": [list(composition)], "episodes_per_composition": int(episodes),
                    "failure_probability": 1 if kind == "failure" else 0, "failure_window": [1, 1],
                    "scenario_seed": int(seed), "victim_rule": "uniform_sensing_agent",
                    "event_step": 1 if kind == "failure" else None}
    return {"protocol_id": protocol_identity(selection, distribution), "selection": selection,
            "distribution": distribution, "scenario_sha256": scenario_sha256}


def select_checkpoint(run, config, target_steps):
    """First saved checkpoint at or above ``target_steps``, verified against the ledger."""
    run = Path(run)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").open()]
    if not metrics or [row["epoch"] for row in metrics] != list(range(1, len(metrics) + 1)):
        raise ValueError(f"{run}: metrics ledger is empty or not contiguous")
    if not (run / "finished.json").exists():
        raise ValueError(f"{run}: training has not finished")
    saved = [row for row in metrics if row["epoch"] % config.save_every == 0 or row["epoch"] == config.epochs
             or (config.total_steps is not None and row["total_steps"] >= config.total_steps)]
    candidates = [row for row in saved if row["total_steps"] >= target_steps]
    if not candidates:
        raise ValueError(f"{run}: no saved checkpoint reaches {target_steps} joint steps")
    chosen = candidates[0]
    earlier = [row for row in saved if row["epoch"] < chosen["epoch"]]

    def load(row):
        path = run / "checkpoints" / f"epoch{row['epoch']:04d}.pt"
        data = path.read_bytes()
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if payload.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError(f"{path}: unsupported checkpoint format")
        if (payload["epoch"] != row["epoch"] or payload["total_steps"] != row["total_steps"]
                or payload["updates"] != row["updates"]):
            raise ValueError(f"{path}: checkpoint progress differs from the metrics ledger")
        if read_json(str(path) + ".signature.json") != payload["signature"]:
            raise ValueError(f"{path}: signature sidecar mismatch")
        if normalized(Config(**payload["config"])) != normalized(config):
            raise ValueError(f"{path}: checkpoint configuration differs from the run configuration")
        return path, payload, hashlib.sha256(data).hexdigest()

    path, payload, digest = load(chosen)
    previous = None
    if earlier:
        _, previous, _ = load(earlier[-1])
        if previous["total_steps"] >= target_steps:
            raise ValueError(f"{run}: an earlier saved checkpoint already reaches the target")
    return {"checkpoint": str(path.resolve()), "sha256": digest, "epoch": chosen["epoch"],
            "total_steps": payload["total_steps"], "updates": payload["updates"],
            "total_episodes": payload["total_episodes"], "source_sha256": payload["source_sha256"],
            "model_signature": payload["signature"]["sha256"], "target_steps": int(target_steps),
            "preceding_saved_epoch": earlier[-1]["epoch"] if earlier else None,
            "preceding_total_steps": previous["total_steps"] if previous else None}


def development_summary(runs):
    """Equal-seed means per arm and window when every seed covers the window."""
    by_arm = {}
    for row in runs:
        by_arm.setdefault(row["arm"], []).append(row)
    summary = {}
    for arm, rows in sorted(by_arm.items()):
        summary[arm] = {}
        for key in rows[0]["development"]:
            windows = [r["development"][key] for r in rows]
            seeds = sorted(r["seed"] for r in rows)
            complete = all(w["complete"] and w["episodes"] for w in windows) and seeds == list(SEEDS)
            entry = {"seeds": seeds, "complete_for_all_seeds": complete, "per_seed": {
                str(r["seed"]): {k: r["development"][key][k] for k in
                                 ("episodes", "complete", "success_rate", "capped_completion", "team_return", "mean_agent_return")}
                for r in rows}}
            if complete:
                for metric in ("success_rate", "capped_completion", "team_return", "mean_agent_return"):
                    values = [w[metric] for w in windows]
                    entry[metric] = {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}
            summary[arm][key] = entry
    return summary


def prepare_evaluation(manifest_path, run_root, output, panel_seed=PANEL_SEED, episodes=EPISODES_PER_COMPOSITION,
                       target_steps=TOTAL_STEPS, bounded=False, entry_indices=None):
    """Select checkpoints, write common panels/protocols, task list and Slurm array."""
    from softrole.train import source_snapshot
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("programme") != PROGRAMME:
        raise ValueError("Unsupported programme manifest")
    if not bounded and (panel_seed != PANEL_SEED or episodes != EPISODES_PER_COMPOSITION
                        or target_steps != TOTAL_STEPS or entry_indices is not None):
        raise ValueError("The research evaluation uses the locked panel seed, 500 scenarios, 20M selection and all entries")
    if manifest.get("bounded_check") and not bounded:
        raise ValueError("A bounded manifest cannot enter the research evaluation")
    if type(target_steps) is not int or target_steps < 1:
        raise ValueError("target_steps must be a positive integer")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    manifest_sha = sha(manifest_path)
    evaluator = source_snapshot()
    selected = manifest["entries"] if entry_indices is None else [manifest["entries"][i] for i in entry_indices]
    if not selected:
        raise ValueError("Select at least one entry")
    runs = []
    for entry in selected:
        run = Path(run_root) / entry["relative_output"]
        record = read_json(run.parent / (run.name + ".programme.json"))
        if record["manifest_sha256"] != manifest_sha or record["index"] != entry["index"]:
            raise ValueError(f"{run}: launched from a different manifest or entry")
        if record["bounded_check"] and not bounded:
            raise ValueError(f"{run}: a bounded engineering run cannot enter the research evaluation")
        config_dict = read_json(run / "config.json")
        expected = entry["config"]
        strip = lambda d: {k: v for k, v in d.items() if k not in RUNTIME_KEYS}
        if (config_dict != expected and not bounded) or strip(config_dict) != strip(expected):
            raise ValueError(f"{run}: configuration differs from the manifest entry")
        config = Config(**config_dict)
        source = read_json(run / "source_manifest.json")
        run_record = read_json(run / "run.json")
        if source["sha256"] != manifest["source_sha256"] or run_record["source_sha256"] != manifest["source_sha256"]:
            raise ValueError(f"{run}: executed source differs from the locked manifest source")
        selection = select_checkpoint(run, config, target_steps)
        if selection["source_sha256"] != manifest["source_sha256"]:
            raise ValueError(f"{run}: checkpoint source differs from the locked manifest source")
        development, last = window_statistics(run, DEVELOPMENT_WINDOWS, config.max_steps)
        runs.append({"index": entry["index"], "arm": entry["arm"], "stage": entry["stage"], "seed": entry["seed"],
                     "control": entry["control"], "run": str(run.resolve()), "model": config.model,
                     "gated": config.model in GATED_MODELS, "selection": selection,
                     "development": development, "last_recorded_steps": last})
    destination.mkdir(parents=True)
    (destination / "panels").mkdir()
    panel_files = {}
    for composition, panel in panels(panel_seed, episodes).items():
        tag = f"final_{composition[0]}_{composition[1]}"
        panel_files[composition] = {}
        for kind in ("nominal", "failure"):
            scenarios_path = destination / "panels" / f"{tag}_{kind}.json"
            write_json(scenarios_path, [asdict(s) for s in panel[kind]])
            scenario_sha = sha(scenarios_path)
            protocol_path = destination / "panels" / f"{tag}_{kind}.protocol.json"
            write_json(protocol_path, protocol(target_steps, composition, kind, panel["seed"], episodes, scenario_sha))
            panel_files[composition][kind] = {"scenarios": str(scenarios_path.resolve()), "scenarios_sha256": scenario_sha,
                                              "protocol": str(protocol_path.resolve())}
    tasks = []
    for row in runs:
        jobs = []
        for composition, files in panel_files.items():
            for name, panel_kind, intervention, step, sham, gated_only, trace in CONDITIONS:
                if gated_only and not row["gated"]:
                    continue
                jobs.append({"arm": row["arm"], "seed": row["seed"], "composition": list(composition),
                             "condition": name, "checkpoint": row["selection"]["checkpoint"],
                             "checkpoint_sha256": row["selection"]["sha256"], **files[panel_kind],
                             "intervention": intervention, "intervention_step": step, "sham": sham, "trace": trace,
                             "output": str((destination / "reports" / row["arm"] / f"seed{row['seed']}" /
                                            f"{composition[0]}_{composition[1]}_{name}.json").resolve())})
        tasks.append({"task": len(tasks), "arm": row["arm"], "seed": row["seed"], "index": row["index"], "jobs": jobs})
    arm_configs = {entry["arm"]: {k: v for k, v in entry["config"].items() if k != "seed"} for entry in manifest["entries"]}
    evaluation = {"version": EVALUATION_VERSION, "programme": PROGRAMME, "manifest": str(manifest_path.resolve()),
                  "manifest_sha256": manifest_sha, "evaluator_source_sha256": evaluator["sha256"],
                  "git_commit": evaluator["git_commit"], "bounded_check": bool(bounded),
                  "target_steps": target_steps, "panel_seed": panel_seed, "episodes_per_composition": episodes,
                  "final_compositions": [list(c) for c in FINAL_COMPOSITIONS],
                  "conditions": [dict(zip(("name", "panel", "intervention", "intervention_step", "sham", "gated_only", "trace"), c))
                                 for c in CONDITIONS],
                  "gated_models": list(GATED_MODELS), "freeze_certification": FREEZE_CERTIFICATION,
                  "development_windows": [list(w) for w in DEVELOPMENT_WINDOWS],
                  "arm_configs": arm_configs, "controls": {arm["name"]: arm["control"] for arm in manifest["arms"]},
                  "runs": runs, "tasks": len(tasks), "jobs": sum(len(t["jobs"]) for t in tasks),
                  "hetnet_admission": HETNET_ADMISSION,
                  "submission": "Preparation never submits. Submit evaluate.sbatch from the repository root after creating logs_sr; keep this checkout fixed until all tasks finish."}
    write_json(destination / "evaluation_manifest.json", evaluation)
    write_json(destination / "tasks.json", {"evaluation": str((destination / "evaluation_manifest.json").resolve()), "tasks": tasks})
    write_json(destination / "development.json", {"windows": [list(w) for w in DEVELOPMENT_WINDOWS],
                                                   "per_run": [{k: r[k] for k in ("index", "arm", "seed", "development", "last_recorded_steps")} for r in runs],
                                                   "per_arm": development_summary(runs),
                                                   "weighting": "episode-weighted within seed; equal seeds across"})
    (destination / "evaluate.sbatch").write_text(slurm_script(destination / "tasks.json", len(tasks)))
    return evaluation


def slurm_script(tasks_path, task_count):
    return "\n".join([
        "#!/bin/bash -l",
        "# Generated frozen-evaluation array for the locked CapCom programme; one task per run.",
        "# Resources are unmeasured starting settings. Submit from the repository root after creating logs_sr.",
        "#SBATCH --job-name=capcom-programme-eval",
        "#SBATCH --account=cenyioha", "#SBATCH --partition=normal", "#SBATCH --constraint=sapphirerapids",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=8G", "#SBATCH --time=12:00:00",
        f"#SBATCH --array=0-{task_count - 1}%6",
        "#SBATCH --output=logs_sr/capcom-programme-eval-%A_%a.out",
        "#SBATCH --error=logs_sr/capcom-programme-eval-%A_%a.err",
        "set -euo pipefail",
        'cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"',
        'index=${SLURM_ARRAY_TASK_ID:?Array index required}',
        "module load anaconda/anaconda-2024.10",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch",
        'export PYTHONPATH="$PWD:$PWD/envs${PYTHONPATH:+:$PYTHONPATH}"',
        'python=${HETNET_PYTHON:-$PWD/.venv/bin/python}',
        f'exec srun --cpu-bind=cores "$python" -u -m softrole.programme_evaluation run --tasks {shlex.quote(str(Path(tasks_path).resolve()))} --index "$index"',
        "",
    ])


def run_task(tasks_path, index, dry_run=False):
    """Evaluate every job of one task; existing identical reports are skipped, never overwritten."""
    from softrole.evaluate import evaluate_checkpoint
    from softrole.train import source_snapshot
    tasks = read_json(tasks_path)
    evaluation = read_json(tasks["evaluation"])
    if type(index) is not int or not 0 <= index < len(tasks["tasks"]):
        raise ValueError("Task index outside the task list")
    if source_snapshot()["sha256"] != evaluation["evaluator_source_sha256"]:
        raise ValueError("Evaluator checkout changed after preparation; restore it or prepare again")
    task = tasks["tasks"][index]
    executed = skipped = 0
    for job in task["jobs"]:
        if sha(job["checkpoint"]) != job["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed since preparation: {job['checkpoint']}")
        data = Path(job["scenarios"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != job["scenarios_sha256"]:
            raise ValueError(f"Scenario panel changed since preparation: {job['scenarios']}")
        declared = read_json(job["protocol"])
        output = Path(job["output"])
        if output.exists():
            existing = read_json(output)
            if (existing.get("checkpoint_sha256") != job["checkpoint_sha256"]
                    or existing.get("scenarios_sha256") != job["scenarios_sha256"]
                    or existing.get("intervention") != job["intervention"] or existing.get("sham") != job["sham"]
                    or existing.get("evaluation_protocol", {}).get("protocol_id") != declared["protocol_id"]):
                raise ValueError(f"Existing report does not match its job: {output}")
            skipped += 1
            continue
        if dry_run:
            continue
        evaluate_checkpoint(job["checkpoint"], [Scenario(**row) for row in json.loads(data)], output,
                            job["intervention"], job["intervention_step"], trace=job["trace"], sham=job["sham"],
                            protocol=declared, scenarios_sha256=job["scenarios_sha256"])
        executed += 1
    return {"task": index, "arm": task["arm"], "seed": task["seed"], "jobs": len(task["jobs"]),
            "executed": executed, "skipped_existing": skipped, "dry_run": bool(dry_run)}


def summarize_evaluation(evaluation_dir, output, bootstrap_samples=10000, seed=0, allow_missing=False):
    """Per-composition strata, failure-minus-sham pairs, D_I and cross-arm contrasts."""
    evaluation_dir = Path(evaluation_dir)
    evaluation = read_json(evaluation_dir / "evaluation_manifest.json")
    tasks = read_json(evaluation_dir / "tasks.json")
    jobs = [job for task in tasks["tasks"] for job in task["jobs"]]
    missing = [job["output"] for job in jobs if not Path(job["output"]).exists()]
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} evaluation reports are missing; summarize after every task finishes")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    # An arm is identified by its scientific configuration; runtime budget keys
    # (bounded engineering checks) do not change the arm. Every seed of an arm
    # must have been evaluated under one identical full configuration.
    strip = lambda config: {k: v for k, v in config.items() if k not in RUNTIME_KEYS}
    scientific_to_arm = {json.dumps(strip(config), sort_keys=True): arm for arm, config in evaluation["arm_configs"].items()}

    def arm_of(config):
        return scientific_to_arm.get(json.dumps(strip(config), sort_keys=True))

    overview = {"version": EVALUATION_VERSION, "programme": PROGRAMME, "bounded_check": evaluation["bounded_check"],
                "missing_reports": missing, "compositions": {}}
    for composition in evaluation["final_compositions"]:
        paths = [job["output"] for job in jobs if job["composition"] == composition and Path(job["output"]).exists()]
        if not paths:
            continue
        summary = summarize_reports(paths, None, bootstrap_samples, seed)
        did = difference_in_differences(summary["paired_groups"], bootstrap_samples, seed)
        full_configs = {}
        for group in summary["groups"]:
            arm = arm_of(group["config"])
            if arm is None:
                raise ValueError("An evaluation report's configuration matches no programme arm")
            if full_configs.setdefault(arm, group["config"]) != group["config"]:
                raise ValueError(f"{arm}: inconsistent run configurations across seeds")
        contrasts = {"absolute": [], "failure_minus_sham": [], "difference_in_differences": []}
        for arm, control in sorted(evaluation["controls"].items()):
            if control is None or arm not in full_configs or control not in full_configs:
                continue
            left, right = full_configs[arm], full_configs[control]
            label = f"{arm}_minus_{control}"
            contrasts["absolute"] += contrast_groups(summary["groups"], left, right, bootstrap_samples, seed, label)
            contrasts["failure_minus_sham"] += contrast_groups(summary["paired_groups"], left, right, bootstrap_samples, seed, label)
            contrasts["difference_in_differences"] += contrast_groups(did, left, right, bootstrap_samples, seed, label)
        key = f"{composition[0]}_{composition[1]}"
        write_json(destination / f"summary_{key}.json",
                   {"composition": composition, "reports": len(paths), "summary": summary,
                    "difference_in_differences": did, "contrasts": contrasts,
                    "method": {"groups": summary["method"], "paired": summary["paired_method"],
                               "difference_in_differences": "D_I = (failure - sham | I) - (failure - sham | none) within seed; equal-seed mean; whole-seed bootstrap",
                               "contrasts": "arm minus declared control within seed on identical strata; equal-seed mean; whole-seed bootstrap"}})
        overview["compositions"][key] = {
            "strata": [{"arm": arm_of(g["config"]), "intervention": g["intervention"],
                        "sham": g["sham"], "scheduled_sensor_failure": g["scheduled_sensor_failure"],
                        "training_seeds": g["training_seeds"], "episodes": g["episodes"], "metrics": g["metrics"]}
                       for g in summary["groups"]],
            "failure_minus_sham": [{"arm": arm_of(g["config"]), "intervention": g["intervention"],
                                    "training_seeds": g["training_seeds"], "metrics": g["metrics"]} for g in summary["paired_groups"]],
            "difference_in_differences": [{"arm": arm_of(g["config"]), "intervention": g["intervention"],
                                           "training_seeds": g["training_seeds"], "metrics": g["metrics"]} for g in did],
            "contrasts": {name: [{"contrast": g["contrast"], "intervention": g.get("intervention"), "sham": g.get("sham"),
                                  "scheduled_sensor_failure": g.get("scheduled_sensor_failure"),
                                  "training_seeds": g["training_seeds"], "metrics": g["metrics"]} for g in rows]
                          for name, rows in contrasts.items()},
        }
    write_json(destination / "overview.json", overview)
    return overview


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="select checkpoints, write panels, tasks and the Slurm array")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--run-root", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--panel-seed", type=int, default=PANEL_SEED)
    prep.add_argument("--episodes", type=positive, default=EPISODES_PER_COMPOSITION)
    prep.add_argument("--target-steps", type=positive, default=TOTAL_STEPS)
    prep.add_argument("--entries", type=int, nargs="+", help="bounded checks only: manifest entry indices")
    prep.add_argument("--bounded", action="store_true", help="engineering checks with reduced budgets/panels")
    run = commands.add_parser("run", help="evaluate one task of the prepared list")
    run.add_argument("--tasks", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--dry-run", action="store_true")
    summary = commands.add_parser("summarize", help="strata, paired differences, D_I and contrasts")
    summary.add_argument("--evaluation", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--bootstrap-samples", type=positive, default=10000)
    summary.add_argument("--seed", type=int, default=0)
    summary.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_evaluation(args.manifest, args.run_root, args.output, args.panel_seed, args.episodes,
                                        args.target_steps, args.bounded, args.entries)
            print(json.dumps({"tasks": result["tasks"], "jobs": result["jobs"], "output": args.output}))
        elif args.command == "run":
            print(json.dumps(run_task(args.tasks, args.index, args.dry_run)))
        else:
            result = summarize_evaluation(args.evaluation, args.output, args.bootstrap_samples, args.seed, args.allow_missing)
            print(json.dumps({"compositions": list(result["compositions"]), "missing_reports": len(result["missing_reports"])}))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
