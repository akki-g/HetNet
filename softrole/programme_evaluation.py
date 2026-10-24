"""Frozen evaluation of the locked programme: selection, panels, tasks and summaries.

The CapCon learner profile selects on one common nominal 2P1A panel at 40M
joint steps. Its exact completion ranking is separate from the legacy programme.
Legacy development windows are read from each run's training ledgers. Its final
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
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import sys

if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

import numpy as np
import torch

from softrole import CHECKPOINT_VERSION, EVALUATION_VERSION as REPORT_VERSION
from softrole.config import Config
from softrole.programme import (ARM_INDEX, CAPCON_ARMS, CAPCON_PROFILE, DEVELOPMENT_WINDOWS, RUNTIME_KEYS,
                                SEEDS, STAGES, normalized, profile_definition, read_json, sha, validate_manifest,
                                window_statistics, write_json)
from softrole.report import (canonical_digest, contrast_groups, difference_in_differences, panel_digest,
                             protocol_identity, summarize_reports, validate_evaluation_protocol)
from softrole.scenarios import Scenario, make_scenarios, scenario_from_record

EVALUATION_VERSION = 2
FINAL_COMPOSITIONS = ((3, 2), (4, 1), (4, 2))
DEVELOPMENT_COMPOSITIONS = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1))
FAILURE_COMPOSITIONS = ((2, 1), (2, 2), (3, 1))
O1_COMPOSITIONS = ((2, 1), (1, 2), (2, 2), (3, 1))
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


def panels(panel_seed, episodes, compositions=FINAL_COMPOSITIONS, *, allow_fully_blind=False):
    """Common nominal panels and matched step-1 sensor-loss panels (uniform victim)."""
    if type(panel_seed) is not int or panel_seed < 0 or type(episodes) is not int or episodes < 1:
        raise ValueError("panel_seed must be a nonnegative integer and episodes positive")
    result = {}
    for composition in compositions:
        composition = tuple(int(x) for x in composition)
        if composition[0] < (1 if allow_fully_blind else 2):
            raise ValueError("Sensor-loss panels need at least two sensing agents unless fully blind panels are explicit")
        seed = composition_seed(panel_seed, composition)
        nominal = make_scenarios(seed, episodes, [composition])
        rng = np.random.default_rng(np.random.SeedSequence([int(panel_seed), *composition, 6006]))
        failure = [replace(s, event_step=1, victim=int(rng.integers(composition[0]))) for s in nominal]
        historical_rng = np.random.default_rng(np.random.SeedSequence([int(panel_seed), *composition, 6010]))
        historical = [replace(s, event_step=int(historical_rng.integers(10, 31)), victim=f.victim)
                      for s, f in zip(nominal, failure)]
        result[composition] = {"nominal": nominal, "failure": failure, "failure_historical": historical, "seed": seed}
    return result


def protocol(target_steps, composition, kind, seed, episodes, scenario_sha256, phase="final"):
    selection = {"rule": SELECTION_RULE, "target_steps": int(target_steps),
                 "lineage": "first saved checkpoint of the programme run at or above the target, verified "
                            "against the preceding accepted checkpoint and independently audited episode/update ledgers"}
    failure = kind.startswith("failure")
    historical = kind == "failure_historical"
    distribution = {"task": "pcp", "phase": phase, "compositions": [list(composition)], "episodes_per_composition": int(episodes),
                    "failure_probability": int(failure), "failure_window": [10, 30] if historical else [1, 1],
                    "scenario_seed": int(seed), "victim_rule": "uniform_sensing_agent",
                    "event_step": None if historical or not failure else 1}
    return {"protocol_id": protocol_identity(selection, distribution), "selection": selection,
            "distribution": distribution, "scenario_sha256": scenario_sha256}


def select_checkpoint(run, config, target_steps, *, accepted=None):
    """First accepted committed checkpoint at the budget, across all attempts."""
    from softrole.programme import accepted_checkpoints
    records = accepted_checkpoints(run) if accepted is None else accepted
    candidates = [row for row in records if row["total_steps"] >= target_steps]
    if not candidates:
        raise ValueError(f"{run}: no saved checkpoint reaches {target_steps} joint steps")
    chosen = candidates[0]
    preceding = [row for row in records if row["updates"] < chosen["updates"]]
    previous = preceding[-1] if preceding else None
    # accepted_checkpoints audits counts/configuration/source against the full
    # logical ledger; an incomplete ancestor tail can never become a candidate.
    payload = torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)
    if normalized(Config(**payload["config"])) != normalized(config):
        raise ValueError("Selected checkpoint configuration differs from the run configuration")
    if previous is not None and previous["total_steps"] >= target_steps:
        raise ValueError("An earlier accepted checkpoint already reaches the target")
    from softrole.model import SoftRoleNet
    from softrole.evaluate import model_signature
    with torch.random.fork_rng():
        model = SoftRoleNet(**config.model_kwargs()).double()
    model.load_state_dict(payload["model_state"], strict=True)
    return {**chosen, "model_signature": payload["signature"]["sha256"],
            "evaluation_model_signature": model_signature(model), "target_steps": int(target_steps),
            "preceding_saved_epoch": previous["epoch"] if previous else None,
            "preceding_total_steps": previous["total_steps"] if previous else None,
            "preceding_checkpoint_sha256": previous["sha256"] if previous else None}


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



def development_conditions(arm, composition, selected_stages):
    """Prespecified training-composition panels, including declared controls."""
    definition = ARM_INDEX[arm]
    diversity = definition["preset"] in ("composition", "failure")
    o1_pair = arm == "o1_physical_only" or (arm == "c1_real64" and "O1" in selected_stages)
    result = []
    if (diversity and composition in DEVELOPMENT_COMPOSITIONS
            or o1_pair and composition in O1_COMPOSITIONS
            or definition["stage"] in ("C1", "L1") and composition == (2, 1)):
        result.append(CONDITIONS[0])
    if diversity:
        # Communication removal is also evaluated without sensor loss.
        result.append(CONDITIONS[1])
    if diversity and composition in FAILURE_COMPOSITIONS:
        result.extend(CONDITIONS[2:])
    elif o1_pair and composition in O1_COMPOSITIONS:
        result.extend(CONDITIONS[2:4])
    if composition == (2, 1) and (diversity or o1_pair):
        source = CONDITIONS[2:] if diversity else CONDITIONS[2:4]
        result.extend(("historical_" + name, "failure_historical", intervention, step, sham, gated, trace)
                      for name, _, intervention, step, sham, gated, trace in source)
    return result


def selected_entries(manifest, phase, stages, entry_indices, bounded):
    capcon = manifest.get("programme") == CAPCON_PROFILE
    if capcon:
        if phase != "nominal-selection" or stages is not None:
            raise ValueError("The CapCon profile supports nominal-selection only, with all four learners")
    elif phase not in ("development", "final"):
        raise ValueError("phase must be development or final for the legacy programme")
    if stages is not None and (not stages or len(stages) != len(set(stages)) or set(stages) - set(STAGES)):
        raise ValueError(f"stages must be distinct members of {STAGES}")
    if phase == "final" and stages is not None:
        raise ValueError("Final preparation requires the complete programme; stages are development only")
    if phase == "development" and stages is None and not bounded:
        raise ValueError("Development preparation requires explicitly selected completed --stages")
    if entry_indices is not None:
        if (len(entry_indices) != len(set(entry_indices))
                or any(type(i) is not int or not 0 <= i < len(manifest["entries"]) for i in entry_indices)):
            raise ValueError("Entry indices must be distinct and in the manifest")
        selected = [manifest["entries"][i] for i in entry_indices]
        return selected, sorted({entry["stage"] for entry in selected})
    wanted = (set(CAPCON_ARMS) if capcon else
              {arm for arm, row in ARM_INDEX.items() if stages is None or row["stage"] in stages})
    for arm in tuple(wanted):
        control = ARM_INDEX[arm]["control"]
        while control is not None:
            wanted.add(control)
            control = ARM_INDEX[control]["control"]
    selected = [entry for entry in manifest["entries"] if entry["arm"] in wanted]
    if not bounded:
        pairs = [(entry["arm"], entry["seed"]) for entry in selected]
        if len(pairs) != len(set(pairs)) or set(pairs) != {(arm, seed) for arm in wanted for seed in SEEDS}:
            raise ValueError("Preparation requires every seed of every selected arm and its controls")
    return selected, list(("C1", "L1") if capcon else stages or STAGES)

def prepare_evaluation(manifest_path, run_root, output, panel_seed=PANEL_SEED, episodes=EPISODES_PER_COMPOSITION,
                       target_steps=None, bounded=False, entry_indices=None, phase=None, stages=None):
    """Select checkpoints, write common panels/protocols, task list and Slurm array."""
    from softrole.train import source_snapshot
    manifest_path = Path(manifest_path)
    manifest = validate_manifest(read_json(manifest_path))
    definition = profile_definition(manifest["programme"])
    capcon = manifest["programme"] == CAPCON_PROFILE
    target_steps = definition["total_steps"] if target_steps is None else target_steps
    phase = ("nominal-selection" if capcon else "development") if phase is None else phase
    if not bounded and (panel_seed != PANEL_SEED or episodes != EPISODES_PER_COMPOSITION
                        or target_steps != definition["total_steps"] or entry_indices is not None):
        raise ValueError("The research evaluation uses the locked panel seed, 500 scenarios, "
                         f"{definition['total_steps']} step selection and all entries")
    if manifest.get("bounded_check") and not bounded:
        raise ValueError("A bounded manifest cannot enter the research evaluation")
    if type(target_steps) is not int or target_steps < 1:
        raise ValueError("target_steps must be a positive integer")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    manifest_sha = sha(manifest_path)
    evaluator = source_snapshot()
    selected, selected_stages = selected_entries(manifest, phase, stages, entry_indices, bounded)
    development_windows = () if capcon else DEVELOPMENT_WINDOWS
    if not selected:
        raise ValueError("Select at least one entry")
    from softrole.programme import accepted_checkpoints
    temporary_audits = tempfile.TemporaryDirectory(prefix="capcom-evaluation-audits-")
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
        accepted, receipt = accepted_checkpoints(run, return_audit=True)
        selection = select_checkpoint(run, config, target_steps, accepted=accepted)
        if selection["source_sha256"] != manifest["source_sha256"]:
            raise ValueError(f"{run}: checkpoint source differs from the locked manifest source")
        development, last = window_statistics(run, development_windows, config.max_steps, receipt=receipt)
        audit_path = Path(temporary_audits.name) / f"entry{entry['index']:03d}.json"
        write_json(audit_path, receipt)
        runs.append({"index": entry["index"], "arm": entry["arm"], "stage": entry["stage"], "seed": entry["seed"],
                     "control": entry["control"], "run": str(run.resolve()), "model": config.model,
                     "config": config_dict,
                     "ledger_audit": {"path": str((destination / "audits" / audit_path.name).resolve()),
                                      "sha256": sha(audit_path)},
                     "gated": config.model in GATED_MODELS, "selection": selection,
                     "development": development, "last_recorded_steps": last})
    destination.mkdir(parents=True)
    shutil.copytree(temporary_audits.name, destination / "audits")
    temporary_audits.cleanup()
    (destination / "panels").mkdir()
    panel_files = {}
    compositions = ((2, 1),) if capcon else FINAL_COMPOSITIONS if phase == "final" else DEVELOPMENT_COMPOSITIONS
    for composition, panel in panels(panel_seed, episodes, compositions, allow_fully_blind=phase == "development").items():
        tag = f"{phase}_{composition[0]}_{composition[1]}"
        panel_files[composition] = {}
        kinds = (("nominal",) if capcon else ("nominal", "failure") if phase == "final" else
                 ("nominal", "failure", "failure_historical"))
        for kind in kinds:
            scenarios_path = destination / "panels" / f"{tag}_{kind}.json"
            write_json(scenarios_path, [asdict(s) for s in panel[kind]])
            scenario_sha = sha(scenarios_path)
            protocol_path = destination / "panels" / f"{tag}_{kind}.protocol.json"
            write_json(protocol_path, protocol(target_steps, composition, kind, panel["seed"], episodes, scenario_sha, phase))
            panel_files[composition][kind] = {"scenarios": str(scenarios_path.resolve()), "scenarios_sha256": scenario_sha,
                                              "protocol": str(protocol_path.resolve()),
                                              "protocol_sha256": sha(protocol_path),
                                              "panel_sha256": canonical_digest([asdict(s) for s in sorted(panel[kind], key=lambda s: str(s.scenario_id))])}
    tasks = []
    for row in runs:
        jobs = []
        for composition, files in panel_files.items():
            conditions = ((CONDITIONS[0],) if capcon else CONDITIONS if phase == "final" else
                          development_conditions(row["arm"], composition, selected_stages))
            for name, panel_kind, intervention, step, sham, gated_only, trace in conditions:
                if gated_only and not row["gated"]:
                    continue
                jobs.append({"arm": row["arm"], "seed": row["seed"], "composition": list(composition),
                             "condition": name, "checkpoint": row["selection"]["checkpoint"],
                             "checkpoint_sha256": row["selection"]["sha256"], **files[panel_kind],
                             "checkpoint_progress": {key: row["selection"][key] for key in
                                                     ("epoch", "updates", "total_steps", "total_episodes")},
                             "model_signature": row["selection"]["evaluation_model_signature"], "config": row["config"],
                             "source_sha256": row["selection"]["source_sha256"],
                             "fully_blind_team_condition": panel_kind.startswith("failure") and composition[0] == 1,
                             "panel_kind": panel_kind,
                             "intervention": intervention, "intervention_step": step, "sham": sham, "trace": trace,
                             "output": str((destination / "reports" / row["arm"] / f"seed{row['seed']}" /
                                            f"{composition[0]}_{composition[1]}_{name}.json").resolve())})
        tasks.append({"task": len(tasks), "arm": row["arm"], "seed": row["seed"], "index": row["index"], "jobs": jobs})
    arm_configs = {entry["arm"]: {k: v for k, v in entry["config"].items() if k != "seed"} for entry in manifest["entries"]}
    evaluation = {"version": EVALUATION_VERSION, "programme": manifest["programme"], "manifest": str(manifest_path.resolve()),
                  "manifest_sha256": manifest_sha, "evaluator_source_sha256": evaluator["sha256"],
                  "git_commit": evaluator["git_commit"], "bounded_check": bool(bounded), "phase": phase,
                  "selected_stages": selected_stages,
                  "target_steps": target_steps, "panel_seed": panel_seed, "episodes_per_composition": episodes,
                  "compositions": [list(c) for c in compositions],
                  "final_compositions": [list(c) for c in FINAL_COMPOSITIONS] if phase == "final" else [],
                  "composition_access": {} if capcon else {"3,2": "historically examined transfer benchmark",
                                         "4,1": "prior access not yet audited", "4,2": "prior access not yet audited"},
                  "conditions": [dict(zip(("name", "panel", "intervention", "intervention_step", "sham", "gated_only", "trace"), c))
                                 for c in ((CONDITIONS[0],) if capcon else CONDITIONS)],
                  "gated_models": list(GATED_MODELS), "freeze_certification": FREEZE_CERTIFICATION,
                  "development_windows": [list(w) for w in development_windows],
                  "arm_configs": arm_configs, "controls": {arm["name"]: arm["control"] for arm in manifest["arms"]},
                  "runs": runs, "tasks": len(tasks), "jobs": sum(len(t["jobs"]) for t in tasks),
                  "tasks_sha256": canonical_digest(tasks),
                  "submission": "Preparation never submits. Submit evaluate.sbatch from the repository root after creating logs_sr; keep this checkout fixed until all tasks finish."}
    if capcon:
        evaluation.update(checkpoint_policy=manifest["checkpoint_policy"], evaluation_policy=manifest["evaluation_policy"])
    else:
        evaluation["hetnet_admission"] = HETNET_ADMISSION
    write_json(destination / "evaluation_manifest.json", evaluation)
    write_json(destination / "tasks.json", {"evaluation": str((destination / "evaluation_manifest.json").resolve()), "tasks": tasks})
    write_json(destination / "development.json", {"windows": [list(w) for w in development_windows],
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



def validate_report_job(report, job, evaluation):
    """Bind reused and summarized outcomes to the exact prepared experiment."""
    declared = read_json(job["protocol"])
    scenarios = read_json(job["scenarios"])
    if (sha(job["protocol"]) != job["protocol_sha256"]
            or sha(job["scenarios"]) != job["scenarios_sha256"]):
        raise ValueError("Prepared protocol or scenario panel changed")
    expected = {"checkpoint_sha256": job["checkpoint_sha256"],
                "checkpoint_progress": job["checkpoint_progress"],
                "model_signature": job["model_signature"], "config": job["config"],
                "source_sha256": job["source_sha256"], "evaluation_version": REPORT_VERSION,
                "format_version": CHECKPOINT_VERSION, "training_seed": job["seed"],
                "environment_version": job["config"]["env_version"],
                "intervention": job["intervention"], "sham": job["sham"],
                "evaluation_protocol": declared, "scenarios_sha256": job["scenarios_sha256"]}
    for key, value in expected.items():
        if canonical_digest(report.get(key)) != canonical_digest(value):
            raise ValueError(f"Report does not match its prepared job: {key}")
    if report.get("evaluator", {}).get("source", {}).get("sha256") != evaluation["evaluator_source_sha256"]:
        raise ValueError("Report does not match its prepared evaluator source")
    if report.get("model_config") != json.loads(json.dumps(Config(**job["config"]).model_kwargs())):
        raise ValueError("Report model configuration differs from its prepared job")
    recorded = report.get("scenarios", [])
    scenario_map = {str(row["scenario_id"]): row for row in scenarios}
    if (len(recorded) != len(scenarios)
            or panel_digest({str(row["scenario_id"]): row for row in recorded}) != job["panel_sha256"]):
        raise ValueError("Report differs from its complete prepared scenario panel")
    validate_evaluation_protocol(declared, report["checkpoint_progress"], recorded, report["scenarios_sha256"])
    outcomes = report.get("per_episode", [])
    identifiers = [str(row["scenario_id"]) for row in outcomes]
    if (len(identifiers) != len(set(identifiers)) or set(identifiers) != set(scenario_map)
            or report.get("episodes") != len(scenarios)):
        raise ValueError("Report outcomes must uniquely cover its complete prepared panel")
    for outcome in outcomes:
        if (type(outcome.get("steps")) is not int or not 1 <= outcome["steps"] <= job["config"]["max_steps"]
                or type(outcome.get("success")) is not bool
                or not isinstance(outcome.get("team_return"), (int, float))
                or not math.isfinite(outcome["team_return"])):
            raise ValueError("Report outcome has invalid steps, success or finite return")
        scenario = scenario_map[str(outcome["scenario_id"])]
        for key, value in scenario.items():
            if outcome.get(key) != value:
                raise ValueError(f"Report outcome differs from its scenario: {key}")
        trigger = scenario["event_step"] if job["intervention_step"] is None else job["intervention_step"]
        if (outcome.get("intervention") != job["intervention"] or outcome.get("intervention_step") != trigger
                or outcome.get("sham") != job["sham"] or outcome.get("composition") != job["composition"]):
            raise ValueError("Report outcome intervention, timing, sham or composition differs from its job")
        trace = outcome.get("trace")
        if job["trace"]:
            if (not isinstance(trace, list) or len(trace) != outcome["steps"]
                    or any(not isinstance(row, dict) for row in trace)
                    or [row.get("step") for row in trace] != list(range(outcome["steps"]))
                    or any(not {"step", "gate", "alpha_null", "actions", "team_reward"}.issubset(row)
                           for row in trace)):
                raise ValueError("Report is missing complete required trace coverage")
            try:
                canonical_digest(trace)
            except ValueError as error:
                raise ValueError("Report trace contains nonfinite values") from error
    return report


def _validate_task_list(tasks, evaluation):
    if (evaluation.get("version") != EVALUATION_VERSION
            or canonical_digest(tasks["tasks"]) != evaluation.get("tasks_sha256")):
        raise ValueError("Task list differs from its prepared evaluation manifest")

def run_task(tasks_path, index, dry_run=False):
    """Evaluate every job of one task; existing identical reports are skipped, never overwritten."""
    from softrole.evaluate import evaluate_checkpoint
    from softrole.train import source_snapshot
    tasks = read_json(tasks_path)
    evaluation = read_json(tasks["evaluation"])
    _validate_task_list(tasks, evaluation)
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
        if sha(job["protocol"]) != job["protocol_sha256"]:
            raise ValueError("Prepared evaluation protocol changed")
        declared = read_json(job["protocol"])
        output = Path(job["output"])
        if output.exists():
            existing = read_json(output)
            validate_report_job(existing, job, evaluation)
            skipped += 1
            continue
        if dry_run:
            continue
        report = evaluate_checkpoint(job["checkpoint"], [scenario_from_record(row) for row in json.loads(data)], output,
                            job["intervention"], job["intervention_step"], trace=job["trace"], sham=job["sham"],
                            protocol=declared, scenarios_sha256=job["scenarios_sha256"])
        validate_report_job(report, job, evaluation)
        executed += 1
    return {"task": index, "arm": task["arm"], "seed": task["seed"], "jobs": len(task["jobs"]),
            "executed": executed, "skipped_existing": skipped, "dry_run": bool(dry_run)}


def rank_nominal_completion(per_seed, tie_order=CAPCON_ARMS):
    """Rank exact equal-seed capped completion means; other metrics never break ties."""
    expected = {(arm, seed) for arm in tie_order for seed in SEEDS}
    indexed = {}
    for row in per_seed:
        key = (row["arm"], row["seed"])
        if key in indexed or key not in expected:
            raise ValueError("Nominal selection requires unique declared arm/seed pairs")
        count, total = row["episodes"], row["capped_completion_sum"]
        if type(count) is not int or count < 1 or type(total) is not int or total < count:
            raise ValueError("Nominal selection requires positive integer episode counts and completion sums")
        indexed[key] = Fraction(total, count)
    if set(indexed) != expected:
        raise ValueError("Nominal selection requires every seed of all four learners")
    scores = {arm: sum((indexed[arm, seed] for seed in SEEDS), Fraction()) / len(SEEDS)
              for arm in tie_order}
    ordered = sorted(tie_order, key=lambda arm: (scores[arm], tie_order.index(arm)))
    return [{"arm": arm, "mean_horizon_capped_completion": float(scores[arm]),
             "mean_exact": [scores[arm].numerator, scores[arm].denominator],
             "tied_best": scores[arm] == scores[ordered[0]]} for arm in ordered]


def nominal_selection(evaluation, tasks, allow_missing=False):
    """Bind the CapCon choice to all twelve validated nominal frozen reports."""
    definition = profile_definition(CAPCON_PROFILE)
    if (evaluation.get("programme") != CAPCON_PROFILE or evaluation.get("phase") != "nominal-selection"
            or evaluation.get("evaluation_policy") != definition["evaluation_policy"]
            or evaluation.get("checkpoint_policy") != definition["checkpoint_policy"]):
        raise ValueError("Nominal selection differs from the locked CapCon policy")
    manifest_path = evaluation["manifest"]
    if sha(manifest_path) != evaluation["manifest_sha256"]:
        raise ValueError("Nominal selection manifest changed after evaluation preparation")
    manifest = validate_manifest(read_json(manifest_path))
    if manifest["programme"] != CAPCON_PROFILE:
        raise ValueError("Nominal selection requires the CapCon profile")
    bounded = evaluation["bounded_check"]
    if not bounded and (manifest["bounded_check"] or evaluation["target_steps"] != definition["total_steps"]
                        or evaluation["panel_seed"] != PANEL_SEED
                        or evaluation["episodes_per_composition"] != EPISODES_PER_COMPOSITION):
        raise ValueError("Research nominal selection cannot use engineering evaluation settings")
    expected = [(arm, seed) for arm in CAPCON_ARMS for seed in SEEDS]
    pairs = [(task["arm"], task["seed"]) for task in tasks["tasks"]]
    if (len(pairs) != len(set(pairs)) or set(pairs) - set(expected)
            or not bounded and pairs != expected):
        raise ValueError("Nominal selection requires all twelve unique declared arm/seed pairs")
    per_seed, evidence, missing, panel_identities = [], [], [], set()
    entries_by_pair = {(entry["arm"], entry["seed"]): entry for entry in manifest["entries"]}
    for task in tasks["tasks"]:
        if len(task["jobs"]) != 1:
            raise ValueError("Nominal selection requires one nominal report per run")
        job = task["jobs"][0]
        entry = entries_by_pair[task["arm"], task["seed"]]
        strip = lambda config: {k: v for k, v in config.items() if k not in RUNTIME_KEYS}
        if (job["arm"] != task["arm"] or job["seed"] != task["seed"]
                or job["composition"] != [2, 1] or job["condition"] != "nominal"
                or job["panel_kind"] != "nominal" or job["intervention"] != "none"
                or job["sham"] or job["trace"] or job["source_sha256"] != manifest["source_sha256"]
                or strip(job["config"]) != strip(entry["config"])
                or not bounded and job["config"] != entry["config"]):
            raise ValueError("Nominal selection job differs from the locked condition")
        declared = protocol(evaluation["target_steps"], (2, 1), "nominal",
                            composition_seed(evaluation["panel_seed"], (2, 1)),
                            evaluation["episodes_per_composition"], job["scenarios_sha256"], "nominal-selection")
        if read_json(job["protocol"]) != declared:
            raise ValueError("Nominal selection protocol differs from the locked evaluation")
        panel_identities.add((job["scenarios_sha256"], job["protocol_sha256"], job["panel_sha256"]))
        path = Path(job["output"])
        if not path.exists():
            missing.append(str(path))
            continue
        report = validate_report_job(read_json(path), job, evaluation)
        outcomes, horizon = report["per_episode"], job["config"]["max_steps"]
        count = len(outcomes)
        total = sum(row["steps"] if row["success"] else horizon for row in outcomes)
        per_seed.append({"arm": task["arm"], "seed": task["seed"], "episodes": count,
                         "capped_completion_sum": total, "completion_mean_exact": [total, count],
                         "completion_steps_horizon_capped": total / count,
                         "success_rate": sum(row["success"] for row in outcomes) / count,
                         "team_return": float(np.mean([row["team_return"] for row in outcomes]))})
        evidence.append({"arm": task["arm"], "seed": task["seed"], "report": str(path),
                         "report_sha256": sha(path), "checkpoint": job["checkpoint"],
                         "checkpoint_sha256": job["checkpoint_sha256"],
                         "checkpoint_progress": job["checkpoint_progress"],
                         "model_signature": job["model_signature"],
                         "scenarios_sha256": job["scenarios_sha256"], "panel_sha256": job["panel_sha256"],
                         "protocol_sha256": job["protocol_sha256"]})
    if len(panel_identities) != 1:
        raise ValueError("Nominal selection requires one common nominal scenario panel and protocol")
    if missing and not allow_missing:
        raise ValueError("Nominal selection reports are missing")
    complete = not missing and set(pairs) == set(expected)
    ranking = rank_nominal_completion(per_seed) if complete else []
    return {"programme": CAPCON_PROFILE, "bounded_check": bounded,
            "status": "engineering_check" if bounded else "selected" if complete else "incomplete",
            "complete": complete, "selected_arm": ranking[0]["arm"] if complete and not bounded else None,
            "rule": "Minimize the exact mean of within-training-seed horizon-capped completion means; "
                    "ties follow the declared arm order. Success and team return are descriptive only.",
            "evaluation_policy": evaluation["evaluation_policy"], "ranking": ranking,
            "per_seed": per_seed, "missing_reports": missing, "evidence": evidence,
            "manifest_sha256": evaluation["manifest_sha256"], "evaluation_identity_sha256": canonical_digest(evaluation),
            "tasks_sha256": evaluation["tasks_sha256"], "source_sha256": manifest["source_sha256"],
            "evaluator_source_sha256": evaluation["evaluator_source_sha256"],
            "interpretation": "Configuration selection on the nominal development panel; no independent final "
                              "HetNet, team-composition or sensor-failure result is established."}


def summarize_evaluation(evaluation_dir, output, bootstrap_samples=10000, seed=0, allow_missing=False):
    """Per-composition strata, failure-minus-sham pairs, D_I and cross-arm contrasts."""
    evaluation_dir = Path(evaluation_dir)
    evaluation = read_json(evaluation_dir / "evaluation_manifest.json")
    tasks = read_json(evaluation_dir / "tasks.json")
    _validate_task_list(tasks, evaluation)
    jobs = [job for task in tasks["tasks"] for job in task["jobs"]]
    missing = [job["output"] for job in jobs if not Path(job["output"]).exists()]
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} evaluation reports are missing; summarize after every task finishes")
    selection = (nominal_selection(evaluation, tasks, allow_missing)
                 if evaluation["programme"] == CAPCON_PROFILE else None)
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

    overview = {"version": EVALUATION_VERSION, "programme": evaluation["programme"], "bounded_check": evaluation["bounded_check"],
                "phase": evaluation["phase"], "missing_reports": missing, "compositions": {}}
    job_by_path = {job["output"]: job for job in jobs}
    if len(job_by_path) != len(jobs):
        raise ValueError("Prepared jobs contain duplicate report outputs")
    for composition in evaluation["compositions"]:
        paths = [job["output"] for job in jobs if job["composition"] == composition and Path(job["output"]).exists()]
        if not paths:
            continue
        summary = summarize_reports(paths, None, bootstrap_samples, seed,
                                    report_validator=lambda report, path: validate_report_job(report, job_by_path[str(path)], evaluation))
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
            "detail": f"summary_{key}.json",
            "strata": [{"arm": arm_of(g["config"]), "intervention": g["intervention"],
                        "sham": g["sham"], "scheduled_sensor_failure": g["scheduled_sensor_failure"],
                        "fully_blind_team_condition": g["scheduled_sensor_failure"] and composition[0] == 1,
                        "physical_full_sensing_loss_scheduled": g["scheduled_sensor_failure"] and composition[0] == 1 and not g["sham"],
                        "evaluation_protocol": g.get("evaluation_protocol"),
                        "training_seeds": g["training_seeds"], "episodes": g["episodes"], "metrics": g["metrics"]}
                       for g in summary["groups"]],
            "failure_minus_sham": [{"arm": arm_of(g["config"]), "intervention": g["intervention"],
                                    "evaluation_protocol": g.get("evaluation_protocol"),
                                    "training_seeds": g["training_seeds"], "metrics": g["metrics"]} for g in summary["paired_groups"]],
            "difference_in_differences": [{"arm": arm_of(g["config"]), "intervention": g["intervention"],
                                           "evaluation_protocol": g.get("evaluation_protocol"),
                                           "training_seeds": g["training_seeds"], "metrics": g["metrics"]} for g in did],
            "contrasts": {name: [{"contrast": g["contrast"], "intervention": g.get("intervention"), "sham": g.get("sham"),
                                  "scheduled_sensor_failure": g.get("scheduled_sensor_failure"),
                                  "evaluation_protocol": g.get("evaluation_protocol"),
                                  "intervention_offset_from_event": g.get("intervention_offset_from_event"),
                                  "fully_blind_team_condition": bool(g.get("scheduled_sensor_failure")) and composition[0] == 1,
                                  "training_seeds": g["training_seeds"], "metrics": g["metrics"]} for g in rows]
                          for name, rows in contrasts.items()},
        }
    if selection is not None:
        selection["evaluation_manifest_sha256"] = sha(evaluation_dir / "evaluation_manifest.json")
        selection["tasks_file_sha256"] = sha(evaluation_dir / "tasks.json")
        write_json(destination / "selection.json", selection)
        overview["configuration_selection"] = {"detail": "selection.json", "status": selection["status"],
                                                "selected_arm": selection["selected_arm"]}
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
    prep.add_argument("--phase", choices=("development", "final", "nominal-selection"),
                      help="defaults to nominal-selection for the CapCon profile, development otherwise")
    prep.add_argument("--stages", nargs="+", choices=STAGES, help="completed development stages; required unless bounded")
    prep.add_argument("--run-root", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--panel-seed", type=int, default=PANEL_SEED)
    prep.add_argument("--episodes", type=positive, default=EPISODES_PER_COMPOSITION)
    prep.add_argument("--target-steps", type=positive, help="defaults to the profile budget; overrides are bounded only")
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
                                        args.target_steps, args.bounded, args.entries, args.phase, args.stages)
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
