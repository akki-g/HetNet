#!/usr/bin/env python3
"""Prepare a verified six-policy PCP evaluation panel; never submit jobs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# NumPy/OpenBLAS can start threads while Torch is importing. Limit them before
# any numerical imports, including when the login environment requests 64.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import torch

from hetnet_ext.signatures import model_signature
from softrole import CHECKPOINT_VERSION, ENVIRONMENT_VERSION
from softrole.__main__ import scenarios_for
from softrole.config import Config
from softrole.model import SoftRoleNet
from softrole.rollout import validate_scenario
from softrole.train import source_snapshot, write_json


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def read_metrics(path):
    """Require a complete, contiguous ledger and reconcile its true counters."""
    data = path.read_bytes()
    rows, steps, episodes, updates = [], 0, 0, 0
    for line_number, line in enumerate(data.decode().splitlines(), 1):
        row = json.loads(line)
        for field in ("epoch", "steps", "episodes", "total_steps", "total_episodes", "updates"):
            if type(row.get(field)) is not int or row[field] <= 0:
                raise ValueError(f"Invalid {field} in {path}:{line_number}")
        steps += row["steps"]
        episodes += row["episodes"]
        if (row["epoch"] != line_number or row["total_steps"] != steps
                or row["total_episodes"] != episodes or row["updates"] <= updates):
            raise ValueError(f"Noncontiguous or inconsistent counters in {path}:{line_number}")
        updates = row["updates"]
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty metrics ledger: {path}")
    return rows, sha256(data)


def select_checkpoint(run_root, model_name, seed, min_steps):
    run = run_root / f"pcp_{model_name}" / f"seed{seed}"
    metrics_path = run / "metrics.jsonl"
    rows, metrics_hash = read_metrics(metrics_path)
    eligible = [(row, run / "checkpoints" / f"epoch{row['epoch']:04d}.pt")
                for row in rows if row["total_steps"] >= min_steps]
    selected = next(((row, path) for row, path in eligible if path.is_file()), None)
    if selected is None:
        raise ValueError(f"No saved checkpoint reaches {min_steps} steps: {run}")
    row, path = selected
    checkpoint_bytes = path.read_bytes()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    if saved.get("format_version") != CHECKPOINT_VERSION or saved.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError(f"Unsupported checkpoint/environment version: {path}")
    config = Config(**saved["config"])
    if (config.task != "pcp" or config.model != model_name or config.seed != seed
            or (config.num_p, config.num_a) != (2, 1)
            or config.training_compositions != ((2, 1),) or config.failure_prob != 0):
        raise ValueError(f"Expected fixed nominal PCP 2P1A {model_name} seed {seed}: {path}")
    for field in ("epoch", "updates", "total_steps", "total_episodes"):
        if type(saved.get(field)) is not int or saved[field] != row[field]:
            raise ValueError(f"Checkpoint/ledger {field} mismatch: {path}")
    if (saved["updates"] + config.updates_per_epoch - 1) // config.updates_per_epoch != saved["epoch"]:
        raise ValueError(f"Checkpoint epoch/update mismatch: {path}")
    if saved.get("model_config") != config.model_kwargs():
        raise ValueError(f"Checkpoint model configuration mismatch: {path}")
    model = SoftRoleNet(**config.model_kwargs()).double()
    model.load_state_dict(saved["model_state"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError(f"Nonfinite model tensor: {path}")
    signature = model_signature(model)
    sidecar_path = Path(str(path) + ".signature.json")
    sidecar_bytes = sidecar_path.read_bytes()
    if saved.get("signature") != signature or json.loads(sidecar_bytes) != signature:
        raise ValueError(f"Checkpoint tensor signature mismatch: {path}")
    source_id = saved.get("source_sha256")
    if not isinstance(source_id, str) or len(source_id) != 64 or any(c not in "0123456789abcdef" for c in source_id):
        raise ValueError(f"Missing or invalid training source identity: {path}")
    return {
        "model": model_name, "training_seed": seed, "checkpoint": str(path),
        "checkpoint_sha256": sha256(checkpoint_bytes), "parameter_sha256": signature["sha256"],
        "signature_sidecar": str(sidecar_path), "signature_sidecar_sha256": sha256(sidecar_bytes),
        "metrics": str(metrics_path), "metrics_sha256": metrics_hash,
        "checkpoint_progress": {field: saved[field] for field in
                                ("epoch", "updates", "total_steps", "total_episodes")},
        "source_sha256": source_id, "environment_version": saved["environment_version"],
        "config": json.loads(json.dumps(config.to_dict())), "model_config": saved["model_config"],
    }


def panel(config, compositions, episodes, seed, failure_prob):
    args = argparse.Namespace(scenarios=None, compositions=compositions, episodes=episodes,
                              seed=seed, failure_prob=failure_prob, failure_window=(10, 30))
    scenarios = scenarios_for(args, config)
    for scenario in scenarios:
        validate_scenario(config, scenario)
    return [asdict(scenario) for scenario in scenarios]


def slurm_script(jobs):
    """One real Slurm array: each task executes exactly one frozen evaluation."""
    lines = ["#!/bin/bash -l",
        "# Frozen PCP evaluation only. Resource limits are unmeasured starting values.",
        "# Submit from the repository root after creating logs_sr.",
        "#SBATCH --job-name=softrole-pcp-frozen",
        "#SBATCH --account=cenyioha", "#SBATCH --partition=normal",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=4G", "#SBATCH --time=04:00:00",
        f"#SBATCH --array=0-{len(jobs) - 1}%3",
        "#SBATCH --output=logs_sr/pcp-frozen-%A_%a.out",
        "#SBATCH --error=logs_sr/pcp-frozen-%A_%a.err",
        "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
        "index=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch as an array}",
        "module load anaconda/anaconda-2024.10",
        'export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"',
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch",
        'case "$index" in']
    for index, job in enumerate(jobs):
        command = ["srun", "bash", str(ROOT / "scripts/softrole_evaluate.sh"),
                   job["checkpoint"], job["output"], "--scenarios", job["scenarios"]]
        if job["condition"] != "nominal":
            command.append("--trace")
        if job["condition"] == "sham":
            command.append("--sham")
        lines.extend([f"  {index})", "    exec " + shlex.join(command) + " ;;"])
    lines.extend([f"  *) echo 'Expected array index 0..{len(jobs) - 1}' >&2; exit 2 ;;", "esac"])
    return "\n".join(lines) + "\n"


def submission_script(output, jobs):
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
        "# Optional convenience wrapper: submit.sbatch is the actual Slurm job file.",
        "# Keep this evaluator checkout unchanged while jobs are queued or running.",
        f"cd {shlex.quote(str(ROOT))}",
        "mkdir -p logs_sr", "set -o noclobber",
        f"exec 3> {shlex.quote(str(output / 'job_ids.tsv'))}", "set +o noclobber",
        "printf 'model\\tseed\\tcondition\\tjob_id\\n' >&3",
        f"submission=$({shlex.join(['sbatch', '--parsable', str(output / 'submit.sbatch')])})",
        'array_id=${submission%%;*}']
    for index, job in enumerate(jobs):
        fields = shlex.join([job["model"], str(job["training_seed"]), job["condition"]])
        lines.append(f"printf '%s\\t%s\\t%s\\t%s\\n' {fields} \"${{array_id}}_{index}\" >&3")
    return "\n".join(lines) + "\n"


def prepare(run_root, output, min_steps=30_000_000):
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError(f"Refusing existing preparation directory: {output}")
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if type(min_steps) is not int or min_steps <= 0:
        raise ValueError("min_steps must be a positive integer")
    torch.set_num_threads(1)
    policies = [select_checkpoint(run_root, model, seed, min_steps)
                for model in ("shared", "banked") for seed in range(3)]
    common = dict(policies[0]["config"])
    for key in ("model", "seed"):
        common.pop(key)
    for policy in policies:
        comparable = dict(policy["config"])
        for key in ("model", "seed"):
            comparable.pop(key)
        if comparable != common or policy["source_sha256"] != policies[0]["source_sha256"]:
            raise ValueError("Selected policies must share scientific configuration and training source except model/seed")
    config = Config(**policies[0]["config"])
    nominal = panel(config, [(2, 1), (1, 2), (2, 2), (3, 1), (3, 2)], 500, 2700, 0)
    failure = panel(config, [(2, 1)], 100, 2701, 1)
    # Check inputs again before creating output, including checkpoint and ledger mutation.
    for policy in policies:
        for path_key, hash_key in (("checkpoint", "checkpoint_sha256"),
                                   ("metrics", "metrics_sha256"),
                                   ("signature_sidecar", "signature_sidecar_sha256")):
            if sha256(Path(policy[path_key]).read_bytes()) != policy[hash_key]:
                raise ValueError(f"Input changed during preparation: {policy[path_key]}")
    helper_bytes = Path(__file__).read_bytes()
    jobs = []
    for policy in policies:
        for condition in ("nominal", "failure", "sham"):
            jobs.append({"model": policy["model"], "training_seed": policy["training_seed"],
                "condition": condition, "checkpoint": policy["checkpoint"],
                "scenarios": str(output / ("scenarios_nominal.json" if condition == "nominal"
                                           else "scenarios_failure.json")),
                "output": str(output / "results" / f"{policy['model']}_seed{policy['training_seed']}_{condition}.json")})
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_snapshot(output)
    (output / "preparation_source.py").write_bytes(helper_bytes)
    write_json(output / "scenarios_nominal.json", nominal)
    write_json(output / "scenarios_failure.json", failure)
    (output / "submit.sbatch").write_text(slurm_script(jobs))
    script = output / "submit.sh"
    script.write_text(submission_script(output, jobs))
    script.chmod(0o755)
    manifest = {"schema_version": 1, "run_root": str(run_root), "submitted": False,
        "submission_status_scope": "Preparation never submits. Submit submit.sbatch with sbatch; optional submit.sh also records job_ids.tsv. This field is not live scheduler status.",
        "slurm_array": {"tasks": len(jobs), "max_concurrent": 3, "cpus_per_task": 1,
                        "memory": "4G", "time_limit": "04:00:00"},
        "checkpoint_rule": "First available saved checkpoint at or above min_steps in each true metrics.jsonl ledger; no outcome-based selection",
        "min_steps": min_steps, "policies": policies, "jobs": jobs,
        "evaluation_source_sha256": provenance["sha256"],
        "preparation_source_sha256": sha256(helper_bytes),
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in
                            ("scenarios_nominal.json", "scenarios_failure.json", "submit.sbatch", "submit.sh")},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(), "device": "cpu", "dtype": "float64"},
        "nominal_panel": {"compositions": [[2, 1], [1, 2], [2, 2], [3, 1], [3, 2]],
                          "episodes_per_composition": 500, "seed": 2700, "count": len(nominal)},
        "failure_panel": {"composition": [2, 1], "episodes_per_condition": 100, "seed": 2701,
                          "failure_probability": 1, "failure_window": [10, 30], "count": len(failure),
                          "paired_sham_uses_same_scenarios": True, "trace": True},
        "execution_requirement": "submit.sbatch uses the live evaluator checkout. Submit from its repository root after creating logs_sr. Keep the checkout, including AGENTS.md, unchanged while jobs are queued/running. The source archive is provenance, not an alternate execution checkout.",
        "limitations": ["Sample counts can overshoot the threshold and differ across checkpoints",
                        "The 100-scenario native failure/sham panel is an exposure diagnostic",
                        "Early completion can prevent exposure to the reference 10..30 event window"]}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/softrole_primary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-steps", type=positive, default=30_000_000)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.run_root, args.output, args.min_steps)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output.resolve()), "policies": len(manifest["policies"]),
                      "jobs_prepared": len(manifest["jobs"]), "submitted": False}, sort_keys=True))


if __name__ == "__main__":
    main()
