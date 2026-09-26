"""Scientific Gate A contracts; training artifacts are supplied by gate_a.

This module never starts training. Artifact checks skip unless GATE_A_OUTPUT is
set; the gate orchestrator treats absent artifact evidence as NOT_RUN, not PASS.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPOSITIONS = ((2, 1), (3, 3), (4, 6), (3, 1), (2, 2))
TIMING_FIELDS = {"wall_time_seconds"}


def _json(path: Path):
    return json.loads(path.read_text())


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _artifacts():
    value = os.environ.get("GATE_A_OUTPUT")
    if not value:
        pytest.skip("Supply Gate A training artifacts through GATE_A_OUTPUT")
    root = Path(value)
    manifest = _json(root / "smoke_manifest.json")
    return root, {row["id"]: row for row in manifest["runs"]}


def _run_path(root: Path, row: dict) -> Path:
    return root / row["relative_dir"]


def _signature_map(state):
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
               "sha256": hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        for name, tensor in sorted(state.items())
    }


def test_runtime_versions():
    import dgl
    import gym
    import numpy
    import torch
    import torchdata

    assert sys.version_info[:2] == (3, 12), sys.version
    assert torch.__version__.split("+")[0] == "2.2.1"
    assert dgl.__version__ == "2.1.0"
    assert torchdata.__version__.split("+")[0] == "0.7.1"
    assert gym.__version__ == "0.26.2"
    assert int(numpy.__version__.split(".")[0]) < 2


def test_grid_is_requested_bijection():
    from hetnet_ext.grid import load_grid

    entries = load_grid()
    expected = {(p, a, seed) for p, a in COMPOSITIONS
                for seed in range(5 if (p, a) in COMPOSITIONS[:3] else 3)}
    actual = {(e.nfriendly_P, e.nfriendly_A, e.seed) for e in entries}
    assert len(entries) == 21
    assert actual == expected
    assert {e.index for e in entries} == set(range(21))
    assert all(e.nfriendly_P != 1 for e in entries)


def test_singleton_p_is_rejected(tmp_path):
    from hetnet_ext.grid import load_grid

    document = _json(ROOT / "configs" / "run1_grid.json")
    changed = False

    def mutate(value):
        nonlocal changed
        if isinstance(value, dict):
            if "nfriendly_P" in value and not changed:
                value["nfriendly_P"] = 1
                changed = True
            for child in value.values():
                mutate(child)
        elif isinstance(value, list):
            for child in value:
                mutate(child)

    mutate(document)
    assert changed, "Config schema lacks explicit nfriendly_P; update this fixture"
    path = tmp_path / "singleton_p.json"
    path.write_text(json.dumps(document))
    with pytest.raises((ValueError, AssertionError), match=r"(?i)(singleton|nfriendly_P|perception|\bP\b)"):
        load_grid(path)


def test_slurm_scripts_parse():
    scripts = sorted((ROOT / "slurm").glob("*.sbatch"))
    assert scripts, "Gate A needs the cluster scripts before it can pass"
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True)
    shellcheck = shutil.which("shellcheck")
    if shellcheck:
        subprocess.run([shellcheck, *map(str, scripts)], check=True, capture_output=True, text=True)


def test_smoke_artifacts_are_finite_and_complete():
    import torch

    root, runs = _artifacts()
    expected_ids = {f"{p}P{a}A_np{n}" for p, a in COMPOSITIONS for n in (1, 4)}
    expected_ids |= {"2P1A_np1_repeat", "2P1A_np1_other_seed"}
    assert set(runs) == expected_ids
    required = {"epoch", "wall_time_seconds", "steps", "episodes", "success_rate", "steps_taken",
                "reward_per_agent", "policy_loss", "value_loss", "total_steps", "total_episodes"}
    for run_id, row in runs.items():
        directory = _run_path(root, row)
        metrics = _jsonl(directory / "metrics.jsonl")
        signatures = _jsonl(directory / "epoch_signatures.jsonl")
        assert [m["epoch"] for m in metrics] == [1, 2, 3], run_id
        assert [s["epoch"] for s in signatures] == [1, 2, 3], run_id
        assert (directory / "initial_signature.json").is_file(), run_id
        args = _json(directory / "resolved_args.json")
        assert args["num_epochs"] == 3 and args["epoch_size"] == 10, run_id
        assert args["batch_size"] == 40 and args["max_steps"] == 20, run_id
        assert args["nprocesses"] == row["nprocesses"], run_id
        assert args["nfriendly_P"] == row["nfriendly_P"] and args["nfriendly_A"] == row["nfriendly_A"], run_id
        assert args["seed"] == row["seed"] and args["save_every"] == 1, run_id
        total_steps = total_episodes = 0
        for m in metrics:
            assert required <= m.keys(), (run_id, required - m.keys())
            numeric = [m[k] for k in required - {"reward_per_agent"}]
            numeric += m["reward_per_agent"]
            assert all(math.isfinite(float(v)) for v in numeric), run_id
            assert len(m["reward_per_agent"]) == row["nfriendly_P"] + row["nfriendly_A"], run_id
            assert 0 <= m["success_rate"] <= 1 and 1 <= m["steps_taken"] <= 20, run_id
            assert 400 * row["nprocesses"] <= m["steps"] <= 590 * row["nprocesses"], run_id
            assert m["episodes"] > 0, run_id
            total_steps += m["steps"]
            total_episodes += m["episodes"]
            assert m["total_steps"] == total_steps and m["total_episodes"] == total_episodes, run_id
        checkpoint_records = _jsonl(directory / "checkpoint_records.jsonl")
        assert [record["epoch"] for record in checkpoint_records] == [1, 2, 3], run_id
        checkpoints = sorted((directory / "checkpoints").rglob("model_ep*.pt"))
        assert [path.name for path in checkpoints] == ["model_ep1.pt", "model_ep2.pt", "model_ep3.pt"], run_id
        for epoch, checkpoint, record, signature in zip(range(1, 4), checkpoints, checkpoint_records, signatures):
            assert Path(record["path"]).resolve() == checkpoint.resolve(), run_id
            assert record["bytes"] == checkpoint.stat().st_size > 0, run_id
            assert math.isfinite(record["wall_time_seconds"]) and record["wall_time_seconds"] >= 0, run_id
            sidecar = _json(Path(str(checkpoint) + ".signature.json"))
            assert sidecar == signature["signature"], (run_id, epoch)
            assert record["parameter_sha256"] == sidecar["sha256"], (run_id, epoch)
            saved = torch.load(checkpoint, map_location="cpu")
            assert saved["seed"] == row["seed"], run_id
            recorded_state = {value["name"]: {key: value[key] for key in ("shape", "dtype", "sha256")}
                              for value in sidecar["parameters"] + sidecar["buffers"]}
            assert _signature_map(saved["policy_net"]) == recorded_state, (run_id, epoch)


def test_determinism_matches_non_timing_metrics_and_signatures():
    root, runs = _artifacts()
    first = _run_path(root, runs["2P1A_np1"])
    repeat = _run_path(root, runs["2P1A_np1_repeat"])
    other = _run_path(root, runs["2P1A_np1_other_seed"])
    assert _json(first / "initial_signature.json") == _json(repeat / "initial_signature.json")
    assert _json(first / "initial_signature.json") != _json(other / "initial_signature.json")
    first_epochs = _jsonl(first / "epoch_signatures.jsonl")
    repeat_epochs = _jsonl(repeat / "epoch_signatures.jsonl")
    assert first_epochs[0] == repeat_epochs[0], "First-epoch weights differ for the same seed"
    assert first_epochs == repeat_epochs, "Later smoke weights differ for the same seed"
    project = lambda path: [{k: v for k, v in m.items() if k not in TIMING_FIELDS}
                           for m in _jsonl(path / "metrics.jsonl")]
    assert project(first) == project(repeat), "Non-timing scientific metrics differ"
    evidence = {"excluded_metrics_fields": sorted(TIMING_FIELDS),
                "same_seed_initial_and_all_epochs_equal": True, "different_seed_initial_differs": True,
                "first_file_sha256": hashlib.sha256((first / "metrics.jsonl").read_bytes()).hexdigest(),
                "repeat_file_sha256": hashlib.sha256((repeat / "metrics.jsonl").read_bytes()).hexdigest(),
                "projection_sha256": hashlib.sha256(json.dumps(project(first), sort_keys=True).encode()).hexdigest()}
    (root / "determinism_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


def _direct_original_model(args, num_p, num_a):
    """Test-only constructor copying main.py's released A2C model settings.

    It creates the original module directly and constructs no policy optimizer.
    This is a Gate A load fixture, not the production frozen inference runner.
    """
    import torch
    from hetgat.uavnet import UAVNetA2CEasy

    pos = args["dim"] ** 2
    raw = {"vision": args["vision"], "P": pos + 4, "A": pos, "state": 4}
    # main.py uses a DoubleTensor default, while fastreal.py deliberately creates
    # FloatTensor attention parameters. Preserve this released mixed-dtype state:
    # calling model.double() would silently cast checkpoint tensors on strict load.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        return UAVNetA2CEasy(raw, {"P": pos + 4, "A": pos, "state": 4},
                             {"P": 16, "A": 16, "state": 16}, {"P": 5, "A": 6, "state": 8},
                             num_p, num_a, num_heads=4, msg_dim=args["msg_dim"], use_CNN=False,
                             use_real=True, use_tanh=False, per_class_critic=True, per_agent_critic=False,
                             device=torch.device("cpu"), with_two_state=True,
                             obs=(2 * args["vision"] + 1) ** 2, comm_range_P=args["comm_range_P"],
                             comm_range_A=args["comm_range_A"], lossy_comm=args["lossy_comm"],
                             min_comm_loss=args["min_comm_loss"], max_comm_loss=args["max_comm_loss"],
                             tensor_obs=args["tensor_obs"], action_vision=args["A_vision"])
    finally:
        torch.set_default_dtype(previous_dtype)


def test_cross_composition_load_preserves_every_tensor():
    import torch
    from hetnet_ext.signatures import model_signature

    root, runs = _artifacts()
    source = _run_path(root, runs["2P1A_np1"])
    args = _json(source / "resolved_args.json")
    checkpoints = list((source / "checkpoints").rglob("model_ep3.pt"))
    assert len(checkpoints) == 1
    saved = torch.load(checkpoints[0], map_location="cpu")
    state = saved["policy_net"]
    signature = _signature_map(state)
    recorded_signature = _jsonl(source / "epoch_signatures.jsonl")[-1]["signature"]
    results = {}
    for num_p, num_a in COMPOSITIONS:
        model = _direct_original_model(args, num_p, num_a)
        outcome = model.load_state_dict(state, strict=True)
        assert not outcome.missing_keys and not outcome.unexpected_keys
        current = model.state_dict()
        assert _signature_map(current) == signature
        assert all(torch.equal(current[name], tensor) for name, tensor in state.items())
        assert model_signature(model) == recorded_signature, "Checkpoint differs from recorded final epoch"
        results[f"{num_p}P{num_a}A"] = {"strict_load": True, "signature": model_signature(model),
                                        "independent_state_tensors": _signature_map(current)}
    (root / "cross_composition_signatures.json").write_text(json.dumps(
        {"schema_version": 1, "source_checkpoint": str(checkpoints[0]), "compositions": results}, indent=2) + "\n")
