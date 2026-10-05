"""Read-only independent binding/configuration audit of Stokes benchmark 904547.

Run from repository root with .venv/bin/python. Writes only adjacent review.json.
Archived Stokes paths are mapped relative to each protocol's original output.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from publication_reconstruction.artifacts import (
    load_checkpoint, _validate_optimizer, _validate_recorded_checkpoint,
)
from publication_reconstruction.runtime.hetnet_ext.signatures import tree_signature, _entries
from publication_reconstruction.runtime.hetnet_ext.recovery import scientific_arguments
from publication_reconstruction.__main__ import inspect_source_origins

ARCHIVE = ROOT / "stokes_runs/hetnet_backend_benchmark_904547"
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(state):
    value = {"schema_version": 1, "parameters": _entries(state.items()), "buffers": []}
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def main():
    plan = read(ARCHIVE / "plan.json")
    source = read(ARCHIVE / "source_identity.json")
    hardware = read(ARCHIVE / "hardware.json")
    assert len(plan["jobs"]) == 32
    source_differences = []
    for name, expected in source.items():
        current = ROOT / ("publication_reconstruction/" + name if name.startswith("runtime/") or name == "ORIGINS.json" else name)
        if not current.is_file() or sha(current) != expected:
            source_differences.append(name)
    assert not source_differences, source_differences
    _, _, origins = inspect_source_origins()
    assert origins["valid"] and origins["origins_sha256"] == source["ORIGINS.json"]
    installed = {name.lower().replace("_", "-"): version for name, version in hardware["packages"].items()}
    direct_pins = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    for requirement in direct_pins:
        name, version = requirement.split("==")
        assert installed[name.lower().replace("_", "-")] == version
    expected_logs = []
    runs = []
    checkpoints = {}
    archives = Counter()
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in ARCHIVE.glob("*.json")}
    for index, job in enumerate(plan["jobs"]):
        result = read(ARCHIVE / f"result_{index:02d}.json")
        relative = Path(job["output"]).relative_to(Path(plan["jobs"][0]["output"]).parents[3])
        run = ARCHIVE / relative
        protocol = read(run / "protocol.json")
        manifest = read(run / "source_manifest.json")
        initial = read(run / "initial_training_identity.json")
        status = read(run / "run_status.json")
        segment = read(run / "training_segment.json")
        checkpoint = run / Path(status["checkpoint"]).relative_to(protocol["output"])
        saved = load_checkpoint(run, checkpoint)
        _validate_optimizer(saved, protocol, job["updates"])
        _validate_recorded_checkpoint(run, checkpoint, checkpoint.read_bytes(), saved, protocol)
        records = [json.loads(line) for line in (run / "checkpoint_records.jsonl").read_text().splitlines()]
        assert len(records) == 1
        model_signature = signature(saved["policy_net"])
        assert model_signature == read(str(checkpoint) + ".signature.json")
        assert records[0]["parameter_sha256"] == model_signature["sha256"]
        assert status["counts"] == saved["reconstruction"]["counts"] == result["counts"] == saved["recovery"]["counts"]
        assert status["counts"]["updates"] == job["updates"]
        assert segment["starting_counts"] == dict(env_steps=0, episodes=0, epoch=0, updates=0)
        assert segment["resume_checkpoint"] is None
        assert int((run / "exit_code.txt").read_text()) == result["exit_code"] == 0
        assert protocol["model_spec"] == protocol["env_version"] == protocol["reconstruction_spec"] == "paper-v1"
        assert protocol["learner_spec"] == "paper-equations-v1"
        assert protocol["collectors"] == 4 and protocol["batch_step_floor_per_collector"] == 500
        assert protocol["seed"] == 991 and protocol["episode_horizon"] == (300 if job["task"] == "fc" else 80)
        assert protocol["message_backend"] == job["backend"]
        assert protocol["profile_phases"] == job["profiled"]
        assert protocol["rng_scheme"] == "seedsequence-v1"
        resolved = read(run / "resolved_args.json")
        assert resolved.pop("resolved_model") == {"class": "hetgat.paper.PaperNet", "per_class_critic": True, "with_two_state": True}
        assert saved["reconstruction"]["resolved_args"] == resolved
        assert saved["recovery"]["scientific_args"] == scientific_arguments(SimpleNamespace(**resolved))
        assert {k: initial[k] for k in ("model", "optimizer", "rng")} == result["initial_identity"]
        actual_final = {"model": tree_signature(saved["policy_net"]), "optimizer": tree_signature(saved["trainer"]),
                        "rng": tree_signature(saved["recovery"]["rng_states"])}
        assert actual_final == result["final_identity"]
        assert saved["reconstruction"]["source_manifest_sha256"] == result["source_manifest_sha256"] == sha(run / "source_manifest.json")
        environment = read(run / "environment.json")
        assert environment["packages"] == hardware["packages"] and environment["python"] == hardware["python"]
        assert sha(run / "uv.lock") == source["uv.lock"]
        assert len(initial["resources"]) == len(status["resources"]["collectors"]) == 4
        for resource in initial["resources"] + status["resources"]["collectors"]:
            assert resource["torch_threads"] == 1 and resource["cpu_affinity"] == hardware["affinity"]
        for name, digest in manifest["files"].items():
            key = name if name.startswith("runtime/") or name == "ORIGINS.json" else "publication_reconstruction/" + name
            assert source[key] == digest
        archives[sha(run / "source_manifest.json")] += 1
        expected_logs.append(f"[{index+1}/32] {job['task']} {job['variant']} {job['backend']} pair{job['pair']} profiled={job['profiled']}")
        for p in run.rglob("*"):
            if p.is_file():
                inputs[str(p.relative_to(ROOT))] = sha(p)
        dtype_counts = Counter(str(p.dtype) for p in saved["policy_net"].values())
        steps = Counter(float(s["step"]) for s in saved["trainer"]["state"].values())
        runs.append({"index": index, "run": str(relative), "task": job["task"], "variant": job["variant"],
                     "backend": job["backend"], "profiled": job["profiled"], "pair": job["pair"],
                     "counts": status["counts"], "checkpoint_sha256": sha(checkpoint),
                     "parameter_tensors": len(saved["policy_net"]), "parameter_dtypes": dict(dtype_counts),
                     "active_optimizer_parameters": len(saved["trainer"]["state"]), "optimizer_step_counts": dict(steps),
                     "initial_identity": result["initial_identity"], "final_identity": actual_final,
                     "source_manifest_sha256": sha(run / "source_manifest.json"),
                     "source_files": len(manifest["files"]), "git_head": environment["git_head"]})
        checkpoints[(job["task"], job["variant"], job["profiled"], job["pair"], job["backend"])] = saved
    log = ROOT / "logs_1/hetnet-backends-904547.out"
    error_log = ROOT / "logs_1/hetnet-backends-904547.err"
    assert log.read_text().splitlines()[:32] == expected_logs
    assert not error_log.read_text().strip()
    inputs[str(log.relative_to(ROOT))] = sha(log)
    inputs[str(error_log.relative_to(ROOT))] = sha(error_log)
    pairs = []
    for task, variant in (("pp", "real"), ("pcp", "real"), ("fc", "real"), ("pcp", "binary")):
        same_workload = [r for r in runs if (r["task"], r["variant"]) == (task, variant)]
        assert len({json.dumps(r["initial_identity"], sort_keys=True) for r in same_workload}) == 1
        for profiled, repetitions in ((False, range(1, 4)), (True, [1])):
            for pair in repetitions:
                a, b = (checkpoints[(task, variant, profiled, pair, backend)] for backend in ("dgl", "torch-v1"))
                args_a, args_b = (dict(s["recovery"]["scientific_args"]) for s in (a, b))
                assert args_a.pop("message_backend") == "dgl" and args_b.pop("message_backend") == "torch-v1"
                assert args_a == args_b
                assert set(a["policy_net"]) == set(b["policy_net"])
                max_difference = max(float((a["policy_net"][name] - b["policy_net"][name]).abs().max()) for name in a["policy_net"])
                model_by_dtype = defaultdict(float)
                worst_parameter = max(a["policy_net"], key=lambda name: float((a["policy_net"][name] - b["policy_net"][name]).abs().max()))
                for name, value in a["policy_net"].items():
                    dtype = str(value.dtype)
                    model_by_dtype[dtype] = max(model_by_dtype[dtype], float((value - b["policy_net"][name]).abs().max()))
                assert a["trainer"]["param_groups"] == b["trainer"]["param_groups"]
                assert set(a["trainer"]["state"]) == set(b["trainer"]["state"])
                optimizer_difference = defaultdict(float)
                optimizer_by_dtype = defaultdict(float)
                for parameter, state in a["trainer"]["state"].items():
                    assert set(state) == set(b["trainer"]["state"][parameter])
                    for name, value in state.items():
                        other = b["trainer"]["state"][parameter][name]
                        error = float((value - other).abs().max()) if torch.is_tensor(value) else abs(value - other)
                        optimizer_difference[name] = max(optimizer_difference[name], error)
                        if torch.is_tensor(value):
                            key = f"{name}:{value.dtype}"
                            optimizer_by_dtype[key] = max(optimizer_by_dtype[key], error)
                pairs.append({"task": task, "variant": variant, "profiled": profiled, "pair": pair,
                              "only_scientific_argument_difference": "message_backend", "max_final_parameter_abs_difference": max_difference,
                              "max_parameter_difference_by_dtype": dict(model_by_dtype), "worst_parameter": worst_parameter,
                              "max_optimizer_difference": dict(optimizer_difference), "max_optimizer_difference_by_dtype": dict(optimizer_by_dtype),
                              "final_rng_exact": tree_signature(a["recovery"]["rng_states"]) == tree_signature(b["recovery"]["rng_states"])})
        for backend in ("dgl", "torch-v1"):
            repetitions = [r for r in same_workload if r["backend"] == backend and not r["profiled"]]
            assert len({json.dumps(r["final_identity"], sort_keys=True) for r in repetitions}) == 1
    report = {"archive": str(ARCHIVE.relative_to(ROOT)), "checks_passed": True,
              "reviewed_source_identity_files": len(source), "source_differences_from_current_reviewed_tree": source_differences,
              "origins_audit": {key: origins[key] for key in ("valid", "runtime_files", "expected_runtime_files", "origins_sha256", "missing", "unexpected", "changed")},
              "all_direct_dependency_pins_match": direct_pins,
              "unique_source_manifests": dict(archives), "runs": runs, "pairs": pairs,
              "ordering": {"recorded_progress_matches_plan": True, "execution": "Blocking child.wait() before next job in hash-verified benchmark.py",
                           "throughput_pairs": "DGL-first, Torch-first, DGL-first", "diagnostics_after_throughput": True,
                           "limitation": "No independent absolute process start/end timestamps; order supported by harness control flow and stdout"},
              "initial_conditions": "Each workload has identical recorded model, optimizer and all-collector RNG identities across all eight runs",
              "repeatability": "All three throughput repetitions are bitwise identical in final model, optimizer and RNG within each backend/workload",
              "limitations": ["One seed and one compute node; engineering evidence only", "Twenty-update benchmark is not the official 100-update preflight",
                              "Initial tensors are represented by signatures, not separate initial checkpoint files",
                              "No complete action trajectory hashes: episode ledgers cannot establish exact trajectories",
                              "Final cross-backend models need not be exact; fixed-rollout correctness gate is separate",
                              "ORIGINS/architecture fidelity depends on the previously reviewed declared paper-v1 interpretation"]}
    (OUT / "review.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (OUT / "input_hashes.json").write_text(json.dumps(inputs, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"passed": True, "runs": len(runs), "source_identity_files": len(source), "archive_files_hashed": len(inputs),
                      "source_manifests": dict(archives), "paired_final_max_abs_difference": max(p["max_final_parameter_abs_difference"] for p in pairs)}, indent=2))


if __name__ == "__main__":
    main()
