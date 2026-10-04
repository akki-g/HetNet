"""Locked first-wave protocols and compute-node preparation (never submits)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = (("pp", "real", 40_000_000), ("pcp", "real", 40_000_000),
             ("fc", "real", 28_000_000), ("pcp", "binary", 40_000_000))


def add_commands(commands):
    run = commands.add_parser("run-index", help="one locked research/preflight array member")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--preflight", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--reconstruction-spec", choices=["legacy", "paper-v1"], default="legacy")
    run.add_argument("--message-backend", choices=["dgl", "torch-v1"], default="dgl")
    plan = commands.add_parser("study-plan", help="write all twelve resolved protocols; no training")
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-evaluation", help="validate checkpoints and prepare a Slurm array")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--pcp-panel-dir", type=Path, required=True,
                         help="existing SoftRole preparation with scenarios_nominal/failure.json")
    prepare.add_argument("--run-map", type=Path,
                         help="optional JSON mapping task_variant/seedN to final continuation run directory")
    prepare.add_argument("--output", type=Path, required=True)


def train_arguments(index, run_root, preflight=False, reconstruction_spec="legacy", message_backend="dgl"):
    if type(index) is not int or not 0 <= index < (4 if preflight else 12):
        raise ValueError("Array index must be 0..3 for preflight or 0..11 for training")
    task, variant, target = WORKLOADS[index if preflight else index // 3]
    seed = 991 if preflight else index % 3
    output = Path(run_root).resolve() / f"{task}_{variant}" / f"seed{seed}"
    args = ["train", "--task", task, "--variant", variant, "--seed", str(seed),
            "--model-spec", "paper-v1" if reconstruction_spec == "paper-v1" else "supplement-v1",
            "--env-version", "paper-v1" if reconstruction_spec == "paper-v1" else "corrected-v1",
            "--recipe", "october-2022", "--collectors", "4", "--batch-steps", "500",
            "--updates-per-epoch", "10", "--horizon", "300" if task == "fc" else "80",
            "--epochs", "10" if preflight else ("1400" if task == "fc" else "2000"),
            "--save-every", "10", "--episode-log", "stdout", "--output", str(output)]
    if reconstruction_spec != "legacy" or message_backend != "dgl":
        args += ["--reconstruction-spec", reconstruction_spec, "--message-backend", message_backend]
    if not preflight:
        milestones = [10_000_000, 20_000_000, target] if task == "fc" else [10_000_000, 20_000_000, 30_000_000, target]
        args += ["--max-env-steps", str(target), "--milestones", *map(str, milestones),
                 "--wall-seconds", "165600"]  # 46 h; Slurm allocation is 48 h.
    return args


def resolved_study(run_root):
    from .__main__ import parser, resolve
    return {"schema_version": 1, "study": "hetnet-supplement-v1-first-wave",
            "submitted": False, "array": "0-11%3", "protocols": [
                {"array_index": i, **resolve(parser().parse_args(train_arguments(i, run_root)))}
                for i in range(12)]}


def preflight_summary(run):
    """Summarize measured update/checkpoint times and observed runtime resources."""
    from .artifacts import load_checkpoint
    run = Path(run)
    status = json.loads((run / "run_status.json").read_text())
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
    records = [json.loads(line) for line in (run / "checkpoint_records.jsonl").read_text().splitlines()]
    if len(updates) != 100:
        raise ValueError("Preflight did not finish its prescribed 100 updates")
    protocol = json.loads((run / "protocol.json").read_text())
    segment = json.loads((run / "training_segment.json").read_text())
    checkpoint = Path(status["checkpoint"])
    saved = load_checkpoint(run, checkpoint)
    elapsed, steps = _validate_preflight_ledger(updates, status, segment, protocol, saved)
    if not records or any(type(row.get("wall_time_seconds")) not in (int, float)
                          or not math.isfinite(row["wall_time_seconds"])
                          or row["wall_time_seconds"] < 0 for row in records):
        raise ValueError("Preflight checkpoint timings must be finite and nonnegative")
    from .artifacts import _validate_optimizer, _validate_recorded_checkpoint
    _validate_recorded_checkpoint(run, checkpoint, checkpoint.read_bytes(), saved, protocol)
    _validate_optimizer(saved, protocol, status["counts"]["updates"])
    target = next(target for task, variant, target in WORKLOADS
                  if (task, variant) == (protocol["task"], protocol["variant"]))
    result = {"schema_version": 1, "usable_checkpoint": str(checkpoint), "updates": len(updates),
              "counts": status["counts"], "update_seconds": [row["wall_time_seconds"] for row in updates],
              "checkpoint_seconds": [row["wall_time_seconds"] for row in records],
              "measured_update_steps_per_second": steps / elapsed,
              "projected_update_work_hours_excluding_startup_logging_and_checkpoints":
                  target * elapsed / steps / 3600,
              "projected_segment_hours_including_logging_and_checkpoints_excluding_startup":
                  target * status["segment_wall_time_seconds"] / steps / 3600,
              "runtime": json.loads((run / "environment.json").read_text()),
              "resources": status["resources"],
              "startup_seconds": status["startup_to_training_seconds"],
              "segment_wall_seconds": status["segment_wall_time_seconds"],
              "limitations": "Timing projection only; convergence and cluster runtime are not guaranteed."}
    from .__main__ import write_json
    # Exercise strict archived loading on the compute node as well as inspecting
    # tensor/optimizer structure. One native episode is an engineering probe.
    probe = run / "checkpoint_probe"
    probe.mkdir(exist_ok=False)
    composition = [(3, 0)] if protocol["task"] == "pp" else [(2, 1)]
    write_json(probe / "scenarios.json", panel(protocol["task"], composition, 1, 3719))
    from .evaluation import evaluate
    evaluated = evaluate(argparse.Namespace(run_dir=run, checkpoint=checkpoint,
        scenarios=probe / "scenarios.json", output=probe / "report.json", protocol=None,
        sham=False, trace=False))
    result["frozen_checkpoint_probe"] = {"report": str(probe / "report.json"),
        "episodes": evaluated["episodes"], "parameters_unchanged": evaluated["parameters_unchanged"]}
    write_json(run / "preflight.json", result)
    return result


def _validate_preflight_ledger(updates, status, segment, protocol, saved):
    """Reconcile one segment's work against its absolute completed-update counts."""
    def counts(value):
        fields = {"env_steps", "episodes", "updates", "epoch"}
        if (not isinstance(value, dict) or set(value) != fields
                or any(type(value[key]) is not int or value[key] < 0 for key in fields)):
            raise ValueError("Preflight progress must contain nonnegative integer counts")
        return dict(value)

    def seconds(value, positive=False):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (positive and value == 0)):
            raise ValueError("Preflight timing must be finite and positive for updates")
        return value

    try:
        initial = counts(segment["starting_counts"])
        total = dict(initial)
        final = counts(status["counts"])
        checkpoint_counts = counts(saved["reconstruction"]["counts"])
        per_epoch = protocol["updates_per_epoch"]
        if type(per_epoch) is not int or per_epoch <= 0:
            raise ValueError("Preflight updates per epoch must be a positive integer")
        if initial["epoch"] != (initial["updates"] + per_epoch - 1) // per_epoch:
            raise ValueError("Preflight starting epoch disagrees with its update count")
        elapsed = 0.0
        for row in updates:
            for key in ("steps", "episodes"):
                if type(row[key]) is not int or row[key] <= 0:
                    raise ValueError("Preflight update step/episode counts must be positive integers")
            if row["episodes"] > row["steps"]:
                raise ValueError("Preflight update has more episodes than environment steps")
            total["updates"] += 1
            total["env_steps"] += row["steps"]
            total["episodes"] += row["episodes"]
            total["epoch"] = (total["updates"] - 1) // per_epoch + 1
            expected = {"update": total["updates"], "epoch": total["epoch"],
                        "update_in_epoch": (total["updates"] - 1) % per_epoch + 1,
                        "total_steps": total["env_steps"], "total_episodes": total["episodes"]}
            if any(type(row[key]) is not int or row[key] != value for key, value in expected.items()):
                raise ValueError("Preflight update order or cumulative counters disagree")
            elapsed += seconds(row["wall_time_seconds"], positive=True)
        if total != final or total != checkpoint_counts:
            raise ValueError("Preflight ledger, status and checkpoint progress disagree")
        recorded_elapsed = seconds(status["training_update_seconds"], positive=True)
        if not math.isclose(elapsed, recorded_elapsed, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("Preflight summed update time disagrees with run status")
        if seconds(status["segment_wall_time_seconds"], positive=True) < elapsed:
            raise ValueError("Preflight segment time is shorter than its update work")
        seconds(status["startup_to_training_seconds"])
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete preflight ledger or progress metadata") from error
    return elapsed, total["env_steps"] - initial["env_steps"]


def panel(task, compositions, episodes, seed, failure=0):
    from softrole.__main__ import scenarios_for
    from softrole.config import recipe
    args = argparse.Namespace(scenarios=None, compositions=compositions, episodes=episodes,
                              seed=seed, failure_prob=failure, failure_window=(10, 30))
    return [asdict(s) for s in scenarios_for(args, recipe(task))]


def checkpoint_candidates(run, seen=None, update_limit=None):
    """Use recorded completed-update checkpoints, trimming abandoned parent suffixes."""
    from .artifacts import verify_run
    seen = set() if seen is None else seen
    run = Path(run).resolve()
    if run in seen:
        raise ValueError("Continuation lineage cycle")
    seen.add(run)
    protocol, _ = verify_run(run)
    candidates = []
    if parent := protocol.get("continuation"):
        retained = parent["counts"]["updates"]
        candidates = checkpoint_candidates(Path(parent["run_dir"]), seen,
                                             retained if update_limit is None else min(retained, update_limit))
        if update_limit is not None and retained >= update_limit:
            return candidates
    with (run / "checkpoint_records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if "counts" not in row:
                raise ValueError("Checkpoint index lacks complete-update counts")
            if update_limit is not None and row["counts"]["updates"] > update_limit:
                break
            recorded = Path(row["path"])
            # Original absolute locations may have changed after a run was copied.
            relative = recorded.relative_to(Path(protocol["output"]))
            row.update(run_dir=str(run), path=str(run / relative))
            candidates.append(row)
            if update_limit is not None and row["counts"]["updates"] == update_limit:
                break  # Do not parse an abandoned or interrupted parent suffix.
    return sorted(candidates, key=lambda row: (row["counts"]["updates"], row["path"]))


def select_checkpoint(run, expected, target):
    from .artifacts import digest, load_checkpoint, verify_run
    def validate_study_run(candidate):
        protocol, _ = verify_run(candidate)
        for field in ("task", "variant", "seed", "model_spec", "env_version", "recipe", "collectors",
                      "updates_per_epoch", "batch_step_floor_per_collector", "episode_horizon", "max_env_steps"):
            if protocol[field] != expected[field]:
                raise ValueError(f"Study run differs from locked {field}: {candidate}")

    validate_study_run(run)
    eligible = [row for row in checkpoint_candidates(run) if row["counts"]["env_steps"] >= target]
    if not eligible:
        raise ValueError(f"No completed-update checkpoint reaches {target}: {run}")
    row = eligible[0]
    # The first eligible checkpoint may belong to a retained parent segment.
    validate_study_run(row["run_dir"])
    if digest(row["path"]) != row["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from its recorded bytes")
    saved = load_checkpoint(row["run_dir"], row["path"])
    if saved.get("seed") != expected["seed"]:
        raise ValueError("Selected checkpoint training seed differs from the locked study")
    if saved["reconstruction"]["counts"] != row["counts"]:
        raise ValueError("Checkpoint index disagrees with checkpoint progress")
    return {**row, "selection_target": target, "task": expected["task"],
            "variant": expected["variant"], "training_seed": expected["seed"]}


def slurm_array(jobs):
    lines = ["#!/bin/bash -l", "# Prepared frozen evaluation; this file never trains.",
             "#SBATCH --job-name=hetnet-frozen", "#SBATCH --account=cenyioha", "#SBATCH --partition=normal",
             "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1", "#SBATCH --mem=4G",
             "#SBATCH --time=04:00:00", f"#SBATCH --array=0-{len(jobs)-1}%3",
             "#SBATCH --output=logs_1/hetnet-frozen-%A_%a.out", "#SBATCH --error=logs_1/hetnet-frozen-%A_%a.err",
             "set -euo pipefail", "cd \"${SLURM_SUBMIT_DIR:?Submit from the repository root}\"",
             "module load anaconda/anaconda-2024.10",
             "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
             "export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch", 'case "${SLURM_ARRAY_TASK_ID:?}" in']
    for index, job in enumerate(jobs):
        command = ["srun", str(ROOT / ".venv/bin/python"), "-m", "publication_reconstruction", "evaluate",
                   "--run-dir", job["run_dir"], "--checkpoint", job["path"],
                   "--scenarios", job["scenarios"], "--protocol", job["protocol"], "--output", job["output"]]
        if job["condition"] != "nominal":
            command.append("--trace")
        if job["condition"] == "sham":
            command.append("--sham")
        lines.append(f"  {index}) exec {shlex.join(command)} ;;")
    lines += ["  *) echo 'Invalid evaluation array index' >&2; exit 2 ;;", "esac", ""]
    return "\n".join(lines)


def prepare_evaluation(args):
    from .__main__ import write_json
    from .artifacts import digest
    from softrole.report import protocol_identity
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing existing preparation directory: {args.output}")
    output = args.output.resolve()
    run_map = json.loads(args.run_map.read_text()) if args.run_map else {}
    study = resolved_study(args.run_root)
    keys = {f"{p['task']}_{p['variant']}/seed{p['seed']}" for p in study["protocols"]}
    if set(run_map) - keys:
        raise ValueError("Run map contains unknown study members")
    policies = []
    for expected in study["protocols"]:
        key = f"{expected['task']}_{expected['variant']}/seed{expected['seed']}"
        run = Path(run_map.get(key, expected["output"]))
        target = 30_000_000 if expected["task"] == "pcp" else expected["max_env_steps"]
        policies.append(select_checkpoint(run, expected, target))
    panels, protocols = {}, {}
    for task, compositions, episodes, seed, failure, target in (
            ("pcp", [(2, 1), (1, 2), (2, 2), (3, 1), (3, 2)], 500, 2700, 0, 30_000_000),
            ("pcp", [(2, 1)], 100, 2701, 1, 30_000_000),
            ("pp", [(3, 0)], 500, 2702, 0, 40_000_000),
            ("fc", [(2, 1)], 500, 2703, 0, 28_000_000)):
        name = f"{task}_{'failure' if failure else 'nominal'}"
        expected_panel = panel(task, compositions, episodes, seed, failure)
        if task == "pcp":
            payload = (args.pcp_panel_dir / f"scenarios_{'failure' if failure else 'nominal'}.json").read_bytes()
            if json.loads(payload) != expected_panel:
                raise ValueError("Existing PCP panel differs from the locked seeds/distribution")
        else:
            payload = (json.dumps(expected_panel, indent=2, sort_keys=True) + "\n").encode()
        panels[name] = payload
        selection = {"rule": "first_saved_at_or_above", "target_steps": target}
        distribution = {"task": task, "compositions": [list(c) for c in compositions],
                        "episodes_per_composition": episodes, "scenario_seed": seed,
                        "failure_probability": failure, "failure_window": [10, 30],
                        "victim_rule": "uniform_sensing_agent"}
        protocols[name] = {"protocol_id": protocol_identity(selection, distribution),
                           "selection": selection, "distribution": distribution,
                           "scenario_sha256": hashlib.sha256(payload).hexdigest()}
    jobs = []
    for policy in policies:
        for condition in (("nominal", "failure", "sham") if policy["task"] == "pcp" else ("nominal",)):
            name = f"{policy['task']}_{'nominal' if condition == 'nominal' else 'failure'}"
            jobs.append({**policy, "condition": condition,
                         "scenarios": str(output / name / "scenarios.json"),
                         "protocol": str(output / name / "protocol.json"),
                         "output": str(output / "results" / f"{policy['task']}_{policy['variant']}_seed{policy['training_seed']}_{condition}.json")})
    # No partial output if any checkpoint or locked input panel is invalid.
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in panels.items():
        (output / name).mkdir()
        (output / name / "scenarios.json").write_bytes(payload)
        write_json(output / name / "protocol.json", protocols[name])
    (output / "submit.sbatch").write_text(slurm_array(jobs))
    manifest = {"schema_version": 1, "submitted": False, "policies": policies, "jobs": jobs,
                "preparation_sha256": digest(__file__), "study": study,
                "artifacts": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
                "execution_requirement": "Keep evaluator checkout and all selected run archives unchanged while queued/running.",
                "fc_comparison_boundary": "Corrected FC must not pool with old-environment SoftRole FC."}
    write_json(output / "manifest.json", manifest)
    return {"output": str(output), "policies": len(policies), "jobs": len(jobs), "submitted": False}


def dispatch(args):
    from .__main__ import main, write_json
    if args.command == "run-index":
        argv = train_arguments(args.index, args.run_root, args.preflight,
                               args.reconstruction_spec, args.message_backend)
        code = main(argv + (["--dry-run"] if args.dry_run else []))
        if args.preflight and not args.dry_run and code == 0:
            preflight_summary(Path(argv[argv.index("--output") + 1]))
        return code
    if args.command == "study-plan":
        result = resolved_study(args.run_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        return {"output": str(args.output.resolve()), "protocols": 12, "submitted": False}
    return prepare_evaluation(args)
