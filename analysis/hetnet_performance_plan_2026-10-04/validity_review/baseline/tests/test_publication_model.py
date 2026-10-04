"""Isolated reconstruction-model contracts; never import its modules in pytest."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "publication_reconstruction/runtime"


@pytest.fixture(scope="module")
def model_probe():
    code = r'''
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

runtime, root = map(Path, sys.argv[1:])
sys.path.insert(0, str(runtime / "envs"))
sys.path.insert(0, str(runtime))
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from hetgat.policy import A2CPolicy
from hetgat.utils import build_hetgraph
from action_utils import select_action
import hetgat.policy as imported_policy
assert Path(imported_policy.__file__).resolve().is_relative_to(runtime)


def policy_for(vision, num_p, num_a, binary):
    return A2CPolicy(
        dict(vision=vision, P=29, A=25, state=4),
        dict(P=29, A=25, state=4),
        dict(P=16, A=16, state=16), dict(P=5, A=6, state=8),
        num_p, num_a, num_heads=4, msg_dim=16, device=torch.device("cpu"),
        gamma=1, lr=1e-4, weight_decay=0, use_real=not binary,
        use_CNN=False, per_class_critic=True, with_two_state=True,
        obs=(2 * vision + 1) ** 2, tensor_obs=False, action_vision=-1)


def graph(num_p, num_a):
    positions = [np.eye(25)[i] for i in range(num_p + num_a)]
    return build_hetgraph(positions, num_P=num_p, num_A=num_a,
                          with_state=True, with_two_state=True)


results = {"extraction": [], "models": [], "layer_preservation": []}
for vision in (1, 2):
    torch.manual_seed(902 + vision)
    model = policy_for(vision, 2, 1, False).model
    cells = (2 * vision + 1) ** 2
    observation = torch.arange(3 * cells * 29).reshape(1, 3, cells * 29).double()
    expected = observation.reshape(1, 3, cells, 29)[0, :2, :, 25:].reshape(2, cells * 4)
    extracted = model.get_obs_features(observation)
    target_cells_retained = 0
    for cell in range(cells):
        impulse = torch.zeros_like(observation)
        impulse[0, 0, cell * 29 + 26] = 1
        extracted_impulse = model.get_obs_features(impulse)
        target_cells_retained += int(extracted_impulse[0, cell * 4 + 1] == 1)
        assert int(torch.count_nonzero(extracted_impulse)) == 1

    empty = torch.zeros_like(observation)
    late_target = empty.clone()
    late_target[0, 0, (cells - 1) * 29 + 26] = 1
    with torch.no_grad():
        before = model([empty, model.init_hidden(1)], graph(2, 1))[-1]["P_o"][0]
        after = model([late_target, model.init_hidden(1)], graph(2, 1))[-1]["P_o"][0]
    results["extraction"].append({
        "vision": vision, "cells": cells,
        "all_channels_exact": torch.equal(extracted, expected),
        "target_cells_retained": target_cells_retained,
        "late_target_changes_sensory_memory": not torch.equal(before[0], after[0]),
        "other_agent_memory_unchanged": torch.equal(before[1], after[1]),
    })

for binary in (False, True):
    for task, num_p, num_a, vision in (("pp", 3, 0, 2), ("pcp", 2, 1, 2), ("fc", 2, 1, 1)):
        torch.manual_seed(731)
        policy = policy_for(vision, num_p, num_a, binary)
        model = policy.model
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        optimizer_steps = []
        def no_step(*args, **kwargs):
            optimizer_steps.append(True)
            raise AssertionError("The policy must return gradients without an Adam update")
        policy.optimizer.step = no_step
        hidden = model.init_hidden(1)
        for step in range(3):
            obs = torch.randn(1, num_p + num_a, 29 * (2 * vision + 1) ** 2)
            logits, values, hidden = policy.batch_select_action_universal([obs, hidden], 0)
            assert logits["P"].shape == (num_p, 5)
            if num_a:
                assert logits["A"].shape == (num_a, 6)
            action = select_action(SimpleNamespace(hetgat=True, nfriendly_P=num_p, nfriendly_A=num_a), logits)
            policy.append_log_probs_properly([action[0, :, 0].numpy()])
            policy.batch_rewards[0].append(np.arange(num_p + num_a) * -.1 - .1)
        losses = policy.batch_finish_per_class(1, sim_time=3, num_P=num_p, num_A=num_a)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        results["models"].append({
            "task": task, "binary": binary,
            "gat_layers": [k for k in dict(model.named_children()) if k.startswith("layer")],
            "finite_losses": all(np.isfinite(float(v)) for v in losses.values()),
            "finite_gradients": bool(gradients) and all(bool(torch.isfinite(v).all()) for v in gradients),
            "nonzero_gradient": any(bool(torch.count_nonzero(v)) for v in gradients),
            "policy_did_not_step": not optimizer_steps,
            "weights_unchanged": all(torch.equal(before[k], p) for k, p in model.named_parameters()),
        })

# The restored observation extractor intentionally changes full actor inputs.
# Check the retained GAT computation independently, using identical arbitrary
# features, identical weights and coupled binary noise, for nonempty A teams.
for binary in (False, True):
    filename = "fastbinary.py" if binary else "fastreal.py"
    spec = importlib.util.spec_from_file_location("original_" + filename[:-3], root / "hetgat/graph" / filename)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original_cls = getattr(original, "MultiHeteroGATLayerBinary" if binary else "MultiHeteroGATLayerReal")
    from hetgat.graph.fastbinary import MultiHeteroGATLayerBinary
    from hetgat.graph.fastreal import MultiHeteroGATLayerReal
    reconstructed_cls = MultiHeteroGATLayerBinary if binary else MultiHeteroGATLayerReal
    in_dim, out_dim = dict(P=29, A=25, state=4), dict(P=16, A=16, state=16)
    extra = (16,) if binary else ()
    left = original_cls(in_dim, out_dim, 4, *extra)
    right = reconstructed_cls(in_dim, out_dim, 4, *extra)
    right.load_state_dict(left.state_dict(), strict=True)
    features = {"P": torch.randn(2, 29), "A": torch.randn(1, 25), "state": torch.randn(2, 4)}
    torch.manual_seed(765)
    expected = left(graph(2, 1), features)
    torch.manual_seed(765)
    actual = right(graph(2, 1), features)
    results["layer_preservation"].append({"binary": binary,
        "outputs_identical": all(torch.equal(expected[k], actual[k]) for k in expected)})
print("PUBLICATION_MODEL_RESULT=" + json.dumps(results))
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(RUNTIME), str(ROOT)],
        cwd=RUNTIME, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    prefix = "PUBLICATION_MODEL_RESULT="
    return json.loads(next(line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)))


@pytest.mark.parametrize("vision,cells", [(1, 9), (2, 25)])
def test_every_sensory_cell_survives_and_reaches_recurrence(model_probe, vision, cells):
    row = next(r for r in model_probe["extraction"] if r["vision"] == vision)
    assert row["all_channels_exact"]
    assert row["target_cells_retained"] == cells
    assert row["late_target_changes_sensory_memory"]
    assert row["other_agent_memory_unchanged"]


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
def test_domain_models_keep_two_layers_and_return_finite_gradients(model_probe, binary, task):
    row = next(r for r in model_probe["models"] if r["binary"] == binary and r["task"] == task)
    assert row["gat_layers"] == ["layer1", "layer2"]
    assert row["finite_losses"] and row["finite_gradients"] and row["nonzero_gradient"]
    assert row["policy_did_not_step"] and row["weights_unchanged"]


@pytest.mark.parametrize("binary", [False, True])
def test_nonempty_action_team_gat_outputs_are_preserved(model_probe, binary):
    row = next(r for r in model_probe["layer_preservation"] if r["binary"] == binary)
    assert row["outputs_identical"]
