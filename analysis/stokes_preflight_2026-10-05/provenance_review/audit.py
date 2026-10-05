"""Read-only audit of both 100-update paper-v1 Stokes preflight arrays.

Run from repository root: .venv/bin/python analysis/stokes_preflight_2026-10-05/provenance_review/audit.py
Only this fresh analysis directory receives output; recorded Stokes paths are
mapped relative to each immutable run's protocol output.
"""
from collections import Counter
import hashlib
import json
import math
import random
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from publication_reconstruction.artifacts import load_checkpoint, _validate_optimizer, _validate_recorded_checkpoint
from publication_reconstruction.benchmark import _episodes, validate_episode_ledger
from publication_reconstruction.evaluation import verify_evaluator_archive
from publication_reconstruction.runtime.hetnet_ext.signatures import tree_signature, _entries
from publication_reconstruction.runtime.hetnet_ext.recovery import scientific_arguments
from publication_reconstruction.study import _validate_preflight_ledger, panel
from publication_reconstruction.__main__ import inspect_source_origins

OUT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "stokes_runs/hetnet_backend_benchmark_904547"
GROUPS = ("905118", "905404")
WORKLOADS = ("pp_real", "pcp_real", "fc_real", "pcp_binary")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_signature(state):
    value = {"schema_version": 1, "parameters": _entries(state.items()), "buffers": []}
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def evaluation_signature(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def without(rows_, excluded):
    return [{k: v for k, v in row.items() if k not in excluded} for row in rows_]


def finite_json(value):
    if isinstance(value, dict):
        return all(finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_json(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    torch.set_num_threads(1)
    before = {}
    for group in GROUPS:
        for p in sorted((ROOT / f"stokes_runs/hetnet_paper_preflight_{group}").rglob("*")):
            if p.is_file():
                before[str(p.relative_to(ROOT))] = sha(p)
    source = read(BENCHMARK / "source_identity.json")
    benchmark_manifest = read(BENCHMARK / "throughput/pp_real/pair1/torch-v1/source_manifest.json")
    _, _, origins = inspect_source_origins()
    assert origins["valid"]
    run_results, retained = [], {}
    for group in GROUPS:
        for workload in WORKLOADS:
            task, variant = workload.split("_")
            run = ROOT / f"stokes_runs/hetnet_paper_preflight_{group}" / workload / "seed991"
            protocol, status = read(run / "protocol.json"), read(run / "run_status.json")
            manifest, segment = read(run / "source_manifest.json"), read(run / "training_segment.json")
            updates, preflight = rows(run / "updates.jsonl"), read(run / "preflight.json")
            assert manifest == benchmark_manifest
            current_source_differences = []
            for name, expected in manifest["files"].items():
                key = name if name.startswith("runtime/") or name == "ORIGINS.json" else "publication_reconstruction/" + name
                assert expected == source[key]
                current = ROOT / "publication_reconstruction" / name
                if sha(current) != expected:
                    current_source_differences.append(name)
            assert not current_source_differences
            checkpoint = run / Path(status["checkpoint"]).relative_to(protocol["output"])
            saved = load_checkpoint(run, checkpoint)
            assert len(saved["recovery"]["rng_states"]) == 4
            for rng in saved["recovery"]["rng_states"]:
                assert set(rng) == {"python", "numpy", "torch"}
                random.Random().setstate(rng["python"])
                np.random.RandomState().set_state(rng["numpy"])
                torch.Generator().set_state(rng["torch"])
            _validate_optimizer(saved, protocol, 100)
            _validate_recorded_checkpoint(run, checkpoint, checkpoint.read_bytes(), saved, protocol)
            elapsed, steps = _validate_preflight_ledger(updates, status, segment, protocol, saved)
            signature = parameter_signature(saved["policy_net"])
            assert signature == read(str(checkpoint) + ".signature.json")
            checkpoint_records = rows(run / "checkpoint_records.jsonl")
            assert len(checkpoint_records) == 1 and checkpoint_records[0]["parameter_sha256"] == signature["sha256"]
            assert len(updates) == preflight["updates"] == status["counts"]["updates"] == 100
            assert finite_json(updates) and finite_json(rows(run / "metrics.jsonl"))
            assert status["counts"] == preflight["counts"] == saved["recovery"]["counts"]
            assert status["counts"]["epoch"] == protocol["epochs"] == 10
            assert status["scientific_budget_completed"] is True and status["stop_reason"] == "epoch_cap_completed"
            assert protocol["message_backend"] == "torch-v1" and not protocol["profile_phases"]
            assert protocol["model_spec"] == protocol["env_version"] == protocol["reconstruction_spec"] == "paper-v1"
            assert protocol["learner_spec"] == "paper-equations-v1" and protocol["seed"] == 991
            assert protocol["collectors"] == 4 and protocol["batch_step_floor_per_collector"] == 500
            assert protocol["episode_horizon"] == (300 if task == "fc" else 80)
            assert protocol["updates_per_epoch"] == 10 and protocol["max_env_steps"] is None
            assert segment["starting_counts"] == dict(env_steps=0, episodes=0, updates=0, epoch=0)
            assert segment["resume_checkpoint"] is None and int((run / "exit_code.txt").read_text()) == 0
            assert preflight["usable_checkpoint"] == status["checkpoint"]
            assert preflight["update_seconds"] == [u["wall_time_seconds"] for u in updates]
            assert preflight["checkpoint_seconds"] == [r["wall_time_seconds"] for r in checkpoint_records]
            assert preflight["measured_update_steps_per_second"] == steps / elapsed
            assert preflight["segment_wall_seconds"] == status["segment_wall_time_seconds"]
            assert preflight["startup_seconds"] == status["startup_to_training_seconds"]
            assert preflight["resources"] == status["resources"]
            resolved = read(run / "resolved_args.json")
            assert resolved.pop("resolved_model") == {"class": "hetgat.paper.PaperNet", "per_class_critic": True, "with_two_state": True}
            assert resolved == saved["reconstruction"]["resolved_args"]
            assert saved["recovery"]["scientific_args"] == scientific_arguments(SimpleNamespace(**resolved))
            initial = read(run / "initial_training_identity.json")
            environment = read(run / "environment.json")
            assert preflight["runtime"] == environment
            assert environment["packages"] == read(BENCHMARK / "hardware.json")["packages"]
            assert sha(run / "uv.lock") == source["uv.lock"]
            for resource in initial["resources"] + status["resources"]["collectors"]:
                assert resource["torch_threads"] == 1 and len(resource["cpu_affinity"]) == 4
            episodes = _episodes(run)
            complete_trajectories = validate_episode_ledger(episodes, updates, collectors=4, floor=500, horizon=protocol["episode_horizon"])
            # Bind the actual frozen probe to the exact checkpoint and source bytes.
            probe = read(run / "checkpoint_probe/report.json")
            assert finite_json(probe)
            evaluator = probe["evaluator"]["source"]
            sidecar = run / "checkpoint_probe/report.json.sources"
            archive_identity = verify_evaluator_archive(sidecar, evaluator)
            assert archive_identity == probe["evaluator"]["source_archive"]["manifest_sha256"]
            assert evaluator["sha256"] == hashlib.sha256(json.dumps(evaluator["files"], sort_keys=True).encode()).hexdigest()
            for name, expected in evaluator["files"].items():
                assert sha(ROOT / name) == expected
            assert probe["checkpoint"] == status["checkpoint"] and probe["checkpoint_sha256"] == sha(checkpoint)
            assert probe["model_signature"] == evaluation_signature(saved["policy_net"])
            assert probe["parameters_unchanged"] is True and probe["episodes"] == len(probe["per_episode"]) == 1
            assert preflight["frozen_checkpoint_probe"] == {"report": str(Path(protocol["output"]) / "checkpoint_probe/report.json"), "episodes": 1, "parameters_unchanged": True}
            assert probe["source_sha256"] == sha(run / "source_manifest.json") == saved["reconstruction"]["source_manifest_sha256"]
            counts = saved["reconstruction"]["counts"]
            assert probe["checkpoint_progress"] == {"epoch": 10, "updates": 100, "total_steps": counts["env_steps"], "total_episodes": counts["episodes"]}
            assert probe["config"]["model_spec"] == "paper-v1" and probe["config"]["message_backend"] == "torch-v1"
            assert probe["config"]["publication_env_version"] == "paper-v1" and probe["training_seed"] == 991
            scenarios = read(run / "checkpoint_probe/scenarios.json")
            assert scenarios == probe["scenarios"] == panel(task, [(3, 0)] if task == "pp" else [(2, 1)], 1, 3719)
            assert probe["scenarios_sha256"] == sha(run / "checkpoint_probe/scenarios.json")
            assert probe["evaluator"]["runtime"]["packages"] == environment["packages"]
            for name, path in probe["evaluator"]["runtime"]["imported_paths"].items():
                local = run / Path(path).relative_to(protocol["output"])
                assert local.is_relative_to(run / "source/runtime") and local.exists()
            row = probe["per_episode"][0]
            assert 1 <= row["steps"] <= protocol["episode_horizon"]
            assert math.isclose(row["team_return"], sum(row["agent_returns"]), abs_tol=1e-12)
            assert probe["success_rate"] == float(row["success"]) and probe["mean_team_return"] == row["team_return"]
            # Match the common first twenty updates to the reviewed benchmark.
            reference = BENCHMARK / "throughput" / workload / "pair1/torch-v1"
            reference_initial = read(reference / "initial_training_identity.json")
            assert {k: initial[k] for k in ("model", "optimizer", "rng")} == {k: reference_initial[k] for k in ("model", "optimizer", "rng")}
            prefix_updates_exact = without(updates[:20], {"wall_time_seconds"}) == without(rows(reference / "updates.jsonl"), {"wall_time_seconds"})
            prefix_episodes = episodes[:sum(r["episodes"] for r in updates[:20])]
            prefix_episodes_exact = without(prefix_episodes, {"rollout_wall_time_seconds"}) == without(_episodes(reference), {"rollout_wall_time_seconds"})
            epoch_signatures = rows(run / "epoch_signatures.jsonl")
            reference_signature = rows(reference / "epoch_signatures.jsonl")[-1]["signature"]
            prefix_parameters_exact = epoch_signatures[1]["signature"] == reference_signature
            assert prefix_updates_exact and prefix_episodes_exact and prefix_parameters_exact
            assert epoch_signatures[-1]["signature"] == signature
            identities = {"model": tree_signature(saved["policy_net"]), "optimizer": tree_signature(saved["trainer"]), "rng": tree_signature(saved["recovery"]["rng_states"])}
            result = {"group": group, "workload": workload, "backend": "torch-v1", "counts": counts,
                      "source_manifest_sha256": sha(run / "source_manifest.json"), "source_files": len(manifest["files"]),
                      "checkpoint_sha256": sha(checkpoint), "checkpoint_parameter_sha256": signature["sha256"],
                      "final_identity": identities, "model_tensors": len(saved["policy_net"]),
                      "active_adam_parameters": len(saved["trainer"]["state"]), "active_adam_steps": sorted({float(v["step"]) for v in saved["trainer"]["state"].values()}),
                      "initial_identity": {k: initial[k] for k in ("model", "optimizer", "rng")},
                      "first20_benchmark_updates_exact": prefix_updates_exact, "first20_benchmark_episode_summaries_exact": prefix_episodes_exact,
                      "first20_benchmark_parameters_exact": prefix_parameters_exact, "complete_trajectories": complete_trajectories,
                      "frozen_probe": {"parameters_unchanged": True, "episodes": 1, "steps": row["steps"], "success": row["success"],
                                       "team_return": row["team_return"], "report_sha256": sha(run / "checkpoint_probe/report.json"),
                                       "evaluator_source_sha256": evaluator["sha256"], "evaluator_files": len(evaluator["files"]),
                                       "evaluator_manifest_sha256": sha(sidecar / "manifest.json")},
                      "all_integrity_gates_passed": True}
            run_results.append(result)
            retained[(group, workload)] = dict(saved=saved, result=result, updates=updates, episodes=episodes, epoch_signatures=epoch_signatures,
                                              metrics=rows(run / "metrics.jsonl"), probe=probe)
    comparisons = []
    for workload in WORKLOADS:
        a, b = (retained[(g, workload)] for g in GROUPS)
        exact = {
            "scientific_arguments": a["saved"]["recovery"]["scientific_args"] == b["saved"]["recovery"]["scientific_args"],
            "model": a["result"]["final_identity"]["model"] == b["result"]["final_identity"]["model"],
            "optimizer": a["result"]["final_identity"]["optimizer"] == b["result"]["final_identity"]["optimizer"],
            "rng": a["result"]["final_identity"]["rng"] == b["result"]["final_identity"]["rng"],
            "updates_excluding_wall_time": without(a["updates"], {"wall_time_seconds"}) == without(b["updates"], {"wall_time_seconds"}),
            "metrics_excluding_wall_time": without(a["metrics"], {"wall_time_seconds"}) == without(b["metrics"], {"wall_time_seconds"}),
            "episode_summaries_excluding_rollout_time": without(a["episodes"], {"rollout_wall_time_seconds"}) == without(b["episodes"], {"rollout_wall_time_seconds"}),
            "all_epoch_parameter_signatures": a["epoch_signatures"] == b["epoch_signatures"],
            "frozen_probe_episode": a["probe"]["per_episode"] == b["probe"]["per_episode"],
        }
        assert all(exact.values()), (workload, exact)
        comparisons.append({"workload": workload, "exact": exact, "model_max_absolute_difference": 0.0,
                            "checkpoint_bytes_differ": a["result"]["checkpoint_sha256"] != b["result"]["checkpoint_sha256"],
                            "byte_difference_interpretation": "Saved output paths and timing/resource metadata differ; numerical training state is exact"})
    changed = [name for name, expected in before.items() if sha(ROOT / name) != expected]
    assert not changed, changed
    report = {"passed": True, "groups": list(GROUPS), "runs": run_results, "cross_group": comparisons,
              "origins_valid": origins["valid"], "benchmark_reference": str(BENCHMARK.relative_to(ROOT)),
              "source_manifests": dict(Counter(r["source_manifest_sha256"] for r in run_results)),
              "input_files_hashed": len(before), "input_files_changed": changed,
              "excluded_fields": {"updates_and_metrics": ["wall_time_seconds"], "episode_summaries": ["rollout_wall_time_seconds"]},
              "limitations": ["Both groups use torch-v1; their timing difference is not a backend comparison",
                              "Recorded frozen probes were byte/source/signature validated, not rerun locally",
                              "One frozen native episode per checkpoint validates execution/loading, not policy quality",
                              "No complete action trajectory digests were recorded",
                              "Preflight checkpoints exhaust the 10-epoch engineering budget; full research training starts fresh"]}
    (OUT / "review.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (OUT / "input_hashes.json").write_text(json.dumps(before, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"passed": True, "runs": len(run_results), "backend": "torch-v1", "cross_group_all_exact": True,
                      "input_files_hashed": len(before), "source_manifests": report["source_manifests"]}, indent=2))


if __name__ == "__main__":
    main()
