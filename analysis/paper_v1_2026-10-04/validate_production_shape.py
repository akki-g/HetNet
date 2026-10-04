"""Bounded functional CLI validation; deliberately makes no throughput claim.

Default is a no-output dry run. --execute starts eight one-update HetNet runs
and two tiny SoftRole FC runs/evaluations in fresh directories. Other tests may
run concurrently, so archived clocks must not be interpreted as speed evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORKLOADS = (("pp", "real"), ("pcp", "real"), ("fc", "real"), ("pcp", "binary"))
THREADS = {name: "1" for name in
           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
THREADS.update(DGLBACKEND="pytorch", PYTHONUNBUFFERED="1")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    from publication_reconstruction.benchmark import source_identity as publication_identity
    identity = publication_identity()
    for pattern in ("softrole/*.py", "scripts/softrole*.sh"):
        for path in sorted(ROOT.glob(pattern)):
            identity[path.relative_to(ROOT).as_posix()] = digest(path)
    return identity


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def jobs(output):
    result = []
    for task, variant in WORKLOADS:
        for backend in ("dgl", "torch-v1"):
            name = f"{task}_{variant}_{backend}"
            run = output / name
            command = [sys.executable, "-u", "-m", "publication_reconstruction", "train",
                "--reconstruction-spec", "paper-v1", "--task", task, "--variant", variant,
                "--message-backend", backend, "--recipe", "october-2022", "--seed", "991",
                "--epochs", "1", "--updates-per-epoch", "1", "--collectors", "4",
                "--batch-steps", "500", "--horizon", "300" if task == "fc" else "80",
                "--save-every", "10", "--episode-log", "stdout", "--output", str(run)]
            result.append(dict(name=name, run=str(run), task=task, variant=variant,
                               backend=backend, command=command))
    return result


def invoke(command, log):
    with Path(log).open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **THREADS},
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=1200)
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    if code:
        raise RuntimeError(f"CLI exit {code}; inspect {log}")


def finite(value):
    import torch
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def inspect_hetnet(job):
    import random
    import numpy as np
    import torch
    from publication_reconstruction.artifacts import load_checkpoint, _validate_optimizer, _validate_recorded_checkpoint
    from publication_reconstruction.benchmark import _episodes, validate_episode_ledger
    from publication_reconstruction.study import _validate_preflight_ledger
    from publication_reconstruction.runtime.hetnet_ext.signatures import tree_signature
    run = Path(job["run"])
    status, protocol = read(run / "run_status.json"), read(run / "protocol.json")
    assert status["stop_reason"] == "epoch_cap_completed"
    assert status["counts"]["updates"] == 1
    checkpoint = Path(status["checkpoint"])
    saved = load_checkpoint(run, checkpoint)
    _validate_optimizer(saved, protocol, 1)
    _validate_recorded_checkpoint(run, checkpoint, checkpoint.read_bytes(), saved, protocol)
    assert saved["recovery"]["counts"] == status["counts"]
    assert len(saved["recovery"]["rng_states"]) == 4
    for state in saved["recovery"]["rng_states"]:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator().set_state(state["torch"])
    assert finite(saved["policy_net"]) and finite(saved["trainer"])
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
    episodes = _episodes(run)
    assert len(updates) == 1 and updates[0]["update"] == 1
    _validate_preflight_ledger(updates, status, read(run / "training_segment.json"), protocol, saved)
    validate_episode_ledger(episodes, updates, collectors=4, floor=500,
                           horizon=protocol["episode_horizon"])
    assert len(episodes) == status["counts"]["episodes"] == updates[0]["episodes"]
    assert sum(row["steps"] for row in episodes) == status["counts"]["env_steps"] == updates[0]["steps"]
    assert 2000 <= updates[0]["steps"] <= 4 * (500 + protocol["episode_horizon"] - 1)
    assert finite(updates) and finite(episodes) and updates[0]["gradient_norm_preclip"] > 0
    assert protocol["collectors"] == 4 and protocol["batch_step_floor_per_collector"] == 500
    assert all(row["torch_threads"] == 1 for row in status["resources"]["collectors"])
    initial = read(run / "initial_training_identity.json")
    initial_identity = {key: initial[key] for key in ("model", "optimizer", "rng")}
    final_model = tree_signature(saved["policy_net"])
    assert final_model != initial_identity["model"]
    semantic = [{key: value for key, value in row.items() if key != "rollout_wall_time_seconds"}
                for row in episodes]
    trajectories = [{key: row[key] for key in ("collector", "collector_episode", "trajectory_sha256")}
                    for row in episodes if "trajectory_sha256" in row]
    return {**job, "counts": status["counts"], "protocol_sha256": digest(run / "protocol.json"),
            "checkpoint": status["checkpoint"], "checkpoint_sha256": digest(status["checkpoint"]),
            "source_manifest_sha256": saved["reconstruction"]["source_manifest_sha256"],
            "initial_identity": initial_identity,
            "final_identity": {"model": final_model, "optimizer": tree_signature(saved["trainer"]),
                               "rng": tree_signature(saved["recovery"]["rng_states"])},
            "finite_model_optimizer": True, "checkpoint_counts_rng_and_record_binding_valid": True,
            "parameters_changed": True, "gradient_norm_preclip": updates[0]["gradient_norm_preclip"],
            "episode_semantic_sha256": tree_signature(semantic),
            "trajectory_identity_complete": len(trajectories) == len(episodes),
            "trajectory_records": trajectories, "episode_records": semantic,
            "losses": {key: updates[0][key] for key in ("policy_loss", "value_loss", "loss_units")}}


def softrole_check(output, model):
    import torch
    from softrole.publication_env import checkpoint_binding
    from publication_reconstruction.runtime.hetnet_ext.signatures import tree_signature
    run = output / f"softrole_fc_{model}"
    command = [sys.executable, "-u", "-m", "softrole", "train", "--task", "fc",
        "--env-version", "paper-v1", "--model", model, "--seed", "991", "--epochs", "1",
        "--updates-per-epoch", "1", "--batch-steps", "1", "--max-steps", "4",
        "--nprocesses", "2", "--save-every", "1", "--output", str(run)]
    invoke(command, output / f"softrole_fc_{model}.log")
    checkpoint = Path(read(run / "finished.json")["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["updates"] == 1 and saved["environment_version"] == "paper-v1"
    assert saved["model_config"]["allow_stay"] is False
    assert finite(saved["model_state"]) and finite(saved["optimizer_state"])
    checkpoint_binding(checkpoint, saved)
    assert read(run / "initial_signature.json")["sha256"] != saved["signature"]["sha256"]
    before = digest(checkpoint)
    report_path = output / f"softrole_fc_{model}_evaluation.json"
    evaluation_command = [sys.executable, "-u", "-m", "softrole", "evaluate", "--checkpoint",
        str(checkpoint), "--episodes", "1", "--seed", "4711", "--compositions", "2,1",
        "--trace", "--output", str(report_path)]
    invoke(evaluation_command, output / f"softrole_fc_{model}_evaluation.log")
    report = read(report_path)
    assert before == digest(checkpoint) and report["episodes"] == 1
    assert report["environment_version"] == "paper-v1" and report["simulator_source"] == saved["simulator_source"]
    assert finite(report)
    return {"model": model, "command": command, "evaluation_command": evaluation_command,
            "run": str(run), "checkpoint": str(checkpoint), "checkpoint_sha256": before,
            "counts": {key: saved[key] for key in ("updates", "total_steps", "total_episodes")},
            "source_sha256": saved["source_sha256"], "simulator_source": saved["simulator_source"],
            "final_model": tree_signature(saved["model_state"]),
            "final_optimizer": tree_signature(saved["optimizer_state"]),
            "finite_model_optimizer": True, "parameters_changed": True,
            "frozen_checkpoint_bytes_unchanged": True, "evaluation": str(report_path),
            "evaluation_sha256": digest(report_path), "evaluation_episodes": 1}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--summary", type=Path, required=True)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args()
    output, summary_path = args.output.resolve(), args.summary.resolve()
    plan = {"schema_version": 1, "jobs": jobs(output), "production_shape_step_bounds": [16000, 20288],
            "softrole": {"models": ["shared", "banked"], "collectors": 2, "horizon": 4,
                         "updates_each": 1, "batch_floor": 1, "frozen_episodes_each": 1},
            "limitation": "Functional engineering checks only; concurrent test activity prevents speed inference. Not official100-update preflights."}
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if output.exists() or summary_path.exists():
        raise FileExistsError("Use a fresh validation output and summary path")
    expected_source = source_identity()
    output.mkdir(parents=True)
    (output / "validate_production_shape.py").write_bytes(Path(__file__).read_bytes())
    write(output / "plan.json", plan)
    result = {**plan, "created_at": datetime.now(timezone.utc).isoformat(), "passed": False,
              "script_sha256": digest(__file__), "source_identity": expected_source,
              "thread_environment": THREADS, "rows": [], "pairs": [], "softrole_rows": []}
    try:
        for job in plan["jobs"]:
            assert source_identity() == expected_source, "Source changed before functional run"
            invoke(job["command"], output / (job["name"] + ".log"))
            result["rows"].append(inspect_hetnet(job))
            assert source_identity() == expected_source, "Source changed during functional run"
            print("validated", job["name"], result["rows"][-1]["counts"], flush=True)
        for task, variant in WORKLOADS:
            left, right = [row for row in result["rows"] if (row["task"], row["variant"]) == (task, variant)]
            assert left["initial_identity"] == right["initial_identity"], "Paired initial identities differ"
            assert left["source_manifest_sha256"] == right["source_manifest_sha256"], "Paired source manifests differ"
            complete = left["trajectory_identity_complete"] and right["trajectory_identity_complete"]
            result["pairs"].append({"task": task, "variant": variant,
                "initial_identities_equal": True, "source_manifests_equal": True,
                "counts_equal": left["counts"] == right["counts"],
                "episode_semantics_equal": left["episode_semantic_sha256"] == right["episode_semantic_sha256"],
                "trajectory_identity_complete": complete,
                "trajectories_equal": left["trajectory_records"] == right["trajectory_records"] if complete else None,
                "divergent_episode_indices": [i for i, (a, b) in enumerate(zip(left["trajectory_records"], right["trajectory_records"])) if a != b] if complete else None,
                "unpaired_trajectory_records": abs(len(left["trajectory_records"]) - len(right["trajectory_records"])) if complete else None})
        for model in ("shared", "banked"):
            assert source_identity() == expected_source
            result["softrole_rows"].append(softrole_check(output, model))
            assert source_identity() == expected_source
            print("validated", "softrole_fc_" + model, flush=True)
        publication_files = read(Path(result["rows"][0]["run"]) / "source_manifest.json")["files"]
        for row in result["softrole_rows"]:
            assert all(publication_files[name.removeprefix("publication_reconstruction/")] == value
                       for name, value in row["simulator_source"]["files"].items())
        result["matched_simulator_bytes"] = True
        result["passed"] = True
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["final_source_identity"] = source_identity()
        result["source_unchanged"] = result["final_source_identity"] == expected_source
        result["production_shape_steps"] = sum(row["counts"]["env_steps"] for row in result["rows"])
        result["production_shape_updates"] = sum(row["counts"]["updates"] for row in result["rows"])
        write(output / "validation.json", result)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write(summary_path, result)


if __name__ == "__main__":
    main()
