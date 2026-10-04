"""Opt-in supplement architecture and immutable public-model compatibility.

The fixture preserves the source before model-spec support: changing or
committing the implementation cannot silently change the compatibility oracle.
Imports run in a fresh process because the two HetNet trees share module names.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "publication_reconstruction/runtime"
PUBLIC_REFERENCE = ROOT / "tests/fixtures/publication_uavnet_public_v1.py"
PUBLIC_REFERENCE_SHA256 = "8db99d5ee4be94836cee626a612977872ffe277a7824a6100afbc758961a01bd"


@pytest.fixture(scope="module")
def spec_probe():
    assert hashlib.sha256(PUBLIC_REFERENCE.read_bytes()).hexdigest() == PUBLIC_REFERENCE_SHA256
    code = r'''
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

runtime, root = map(Path, sys.argv[1:3])
reference = Path(sys.argv[3])
sys.path.insert(0, str(runtime / "envs"))
sys.path.insert(0, str(runtime))
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from hetgat.policy import A2CPolicy
from hetgat.uavnet import UAVNetA2CEasy
from hetgat.utils import build_hetgraph
from action_utils import select_action
import hetgat.policy as current_policy
assert Path(current_policy.__file__).resolve().is_relative_to(runtime)


legacy_model = types.ModuleType("pre_spec_uavnet")
exec(compile(reference.read_text(), str(reference), "exec"), legacy_model.__dict__)


def policy_for(cls, vision, num_p, num_a, binary, **extra):
    arguments = (
        dict(vision=vision, P=29, A=25, state=4),
        dict(P=29, A=25, state=4),
        dict(P=16, A=16, state=16), dict(P=5, A=6, state=8),
        num_p, num_a)
    keywords = dict(num_heads=4, msg_dim=16, device=torch.device("cpu"), use_real=not binary,
        use_CNN=False, per_class_critic=True, with_two_state=True,
        obs=(2 * vision + 1) ** 2, tensor_obs=False, action_vision=-1,
        **extra)
    if cls is legacy_model.UAVNetA2CEasy:
        return SimpleNamespace(model=cls(*arguments, **keywords))
    return cls(*arguments, gamma=1, lr=1e-4, weight_decay=0, **keywords)


def graph(num_p, num_a):
    positions = [np.eye(25)[i] for i in range(num_p + num_a)]
    return build_hetgraph(positions, num_P=num_p, num_A=num_a,
                         with_state=True, with_two_state=True)


def tensors(value):
    if isinstance(value, dict):
        return [tensor for key in sorted(value) for tensor in tensors(value[key])]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensors(item)]
    return [value]


def forward_backward(model, observations, num_p, num_a):
    torch.manual_seed(822)
    hidden = model.init_hidden(1)
    trace = []
    loss = 0
    for observation in observations:
        output = model([observation, hidden], graph(num_p, num_a))
        hidden = output[-1]
        trace.extend(t.detach().clone() for t in tensors(output))
        # Include every actor output and both class critics, over recurrent steps.
        loss = loss + sum(t.square().sum() for t in tensors(output[:-1]))
    loss.backward()
    gradients = {key: None if parameter.grad is None else parameter.grad.clone()
                 for key, parameter in model.named_parameters()}
    return trace, gradients, torch.get_rng_state()


def identical(left, right):
    return len(left) == len(right) and all(
        a is b if a is None or b is None else torch.equal(a, b)
        for a, b in zip(left, right))


result = {"compatibility": [], "supplement": [], "validation": [], "attention": []}
for binary in (False, True):
    for task, num_p, num_a, vision in (("pp", 3, 0, 2), ("pcp", 2, 1, 2), ("fc", 2, 1, 1)):
        observation_rng = torch.Generator().manual_seed(255)
        observations = [torch.randn(1, num_p + num_a, 29 * (2 * vision + 1) ** 2,
                                    generator=observation_rng) for _ in range(3)]
        models, states, initial_rngs, traces, gradients, final_rngs = [], [], [], [], [], []
        for cls, kwargs in ((legacy_model.UAVNetA2CEasy, {}), (A2CPolicy, {}),
                            (A2CPolicy, {"model_spec": "public-code-v1"})):
            torch.manual_seed(931)
            policy = policy_for(cls, vision, num_p, num_a, binary, **kwargs)
            models.append(policy.model)
            states.append({k: v.clone() for k, v in policy.model.state_dict().items()})
            initial_rngs.append(torch.get_rng_state())
            trace, gradient, rng = forward_backward(policy.model, observations, num_p, num_a)
            traces.append(trace)
            gradients.append(gradient)
            final_rngs.append(rng)
        result["compatibility"].append({
            "binary": binary, "task": task,
            "parameter_names": all(list(state) == list(states[0]) for state in states),
            "parameters": all(identical(list(states[0].values()), list(state.values())) for state in states),
            "outputs": all(identical(traces[0], trace) for trace in traces),
            "gradients": all(list(g) == list(gradients[0]) and
                             identical(list(gradients[0].values()), list(g.values())) for g in gradients),
            "rng": all(torch.equal(initial_rngs[0], rng) for rng in initial_rngs) and
                   all(torch.equal(final_rngs[0], rng) for rng in final_rngs),
            "two_layers": all(not hasattr(model, "layer3") for model in models),
        })

        torch.manual_seed(932)
        policy = policy_for(A2CPolicy, vision, num_p, num_a, binary, model_spec="supplement-v1")
        model = policy.model
        weights_before = {k: p.detach().clone() for k, p in model.named_parameters()}
        calls, head_features, outputs, handles = [], {}, [], []
        def capture_heads(name):
            def hook(module, args, output):
                head_features[name] = {key: value.clone() for key, value in output.items()}
            return hook
        def capture_layer(name, hidden):
            def hook(module, args, output):
                calls.append(name)
                widths = {"P": 64, "A": 64, "state": 64} if hidden else {"P": 5, "A": 6, "state": 8}
                sizes = {"P": num_p, "A": num_a, "state": 2}
                for key, value in output.items():
                    assert tuple(value.shape) == (sizes[key], widths[key]), (name, key, value.shape)
                    heads = head_features[name][key]
                    assert tuple(heads.shape) == (sizes[key], 4, widths[key] // 4 if hidden else widths[key])
                    expected = torch.cat(tuple(heads.unbind(1)), dim=1) if hidden else heads.sum(1) / 4
                    assert torch.equal(value, expected), (name, key, "head merge")
                    value.retain_grad()
                outputs.append(output)
            return hook
        for name in ("layer1", "layer2", "layer3"):
            layer = getattr(model, name)
            handles.append(layer.gat_conv.register_forward_hook(capture_heads(name)))
            handles.append(layer.register_forward_hook(capture_layer(name, name != "layer3")))
        def forbidden_optimizer_step(*args, **kwargs):
            raise AssertionError("The inactive policy optimizer must not step")
        policy.optimizer.step = forbidden_optimizer_step
        hidden = model.init_hidden(1)
        for observation in observations:
            logits, values, hidden = policy.batch_select_action_universal([observation, hidden], 0)
            assert logits["P"].shape == (num_p, 5)
            if num_a:
                assert logits["A"].shape == (num_a, 6)
            action = select_action(SimpleNamespace(hetgat=True, nfriendly_P=num_p, nfriendly_A=num_a), logits)
            policy.append_log_probs_properly([action[0, :, 0].numpy()])
            policy.batch_rewards[0].append(np.arange(num_p + num_a) * -.1 - .1)
        losses = policy.batch_finish_per_class(1, sim_time=3, num_P=num_p, num_A=num_a)
        for handle in handles:
            handle.remove()
        layer_gradients = []
        for name in ("layer1", "layer2", "layer3"):
            layer_grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            layer_gradients.append(bool(layer_grads) and all(torch.isfinite(g).all() for g in layer_grads)
                                   and any(torch.count_nonzero(g) for g in layer_grads))
        weights_unchanged = all(torch.equal(weights_before[k], p) for k, p in model.named_parameters())
        strict_rejections = 0
        for left, right in ((model, models[0]), (models[0], model)):
            try:
                left.load_state_dict(right.state_dict(), strict=True)
            except RuntimeError:
                strict_rejections += 1
        # Rejected PyTorch loads can copy matching keys before raising; compare
        # unchanged weights before exercising the mismatched checkpoint loads.
        result["supplement"].append({
            "binary": binary, "task": task,
            "spec": policy.model_spec == model.model_spec == "supplement-v1",
            "calls": calls,
            "finite_losses": all(np.isfinite(float(v)) for v in losses.values()),
            "layer_gradients": all(bool(value) for value in layer_gradients),
            "weights_unchanged": weights_unchanged,
            "intermediate_gradients": all(any(v.grad is not None and bool(torch.count_nonzero(v.grad))
                                              for v in output.values()) for output in outputs),
            "strict_rejections": strict_rejections,
        })

# Exercise the actual DGL operator used by both model variants. Each receiver
# has two incoming physical classes: each relation normalizes independently.
for binary in (False, True):
    import hetgat.graph.fastreal as real_module
    import hetgat.graph.fastbinary as binary_module
    module = binary_module if binary else real_module
    original_softmax = module.edge_softmax
    calls = []
    def observed_softmax(relation, logits):
        actual = original_softmax(relation, logits)
        destinations = relation.edges()[1]
        expected = torch.empty_like(actual)
        for destination in destinations.unique():
            indices = (destinations == destination).nonzero().flatten()
            # Independent NumPy normalization over senders to this receiver,
            # separately for each head; it never sees another relation's edges.
            values = logits[indices].detach().numpy()
            weights = np.exp(values - values.max(axis=0, keepdims=True))
            expected[indices] = torch.from_numpy(weights / weights.sum(axis=0, keepdims=True))
        assert torch.allclose(actual, expected, rtol=1e-13, atol=1e-14)
        calls.append(relation.canonical_etypes[0][1])
        return actual
    module.edge_softmax = observed_softmax
    try:
        policy = policy_for(A2CPolicy, 2, 3, 2, binary, model_spec="supplement-v1")
        attention_graph = graph(3, 2)
        policy.model.layer1(attention_graph, {
            "P": torch.randn(3, 29), "A": torch.randn(2, 25), "state": torch.randn(2, 4)})
        combined = torch.zeros(3, 4, 1)
        for relation in ("p2p", "a2p"):
            destinations = attention_graph[relation].edges()[1]
            weights = attention_graph[relation].edata["a_" + relation]
            combined.index_add_(0, destinations, weights)
        result["attention"].append({"binary": binary,
            "relations": sorted(calls),
            "two_relation_mass": bool(torch.allclose(combined, torch.full_like(combined, 2))),
            "not_joint_softmax": not bool(torch.allclose(combined, torch.ones_like(combined)))})
    finally:
        module.edge_softmax = original_softmax

for spec, heads, hidden in (("typo", 4, 16), ("supplement-v1", 3, 16), ("supplement-v1", 4, 8)):
    try:
        UAVNetA2CEasy(dict(vision=2, P=29, A=25, state=4), dict(P=29, A=25, state=4),
                     dict(P=hidden, A=hidden, state=hidden), dict(P=5, A=6, state=8),
                     2, 1, heads, model_spec=spec)
    except ValueError:
        result["validation"].append(True)
    else:
        result["validation"].append(False)
print("PUBLICATION_SPEC_RESULT=" + json.dumps(result))
'''
    process = subprocess.run(
        [sys.executable, "-c", code, str(RUNTIME), str(ROOT), str(PUBLIC_REFERENCE)],
        cwd=RUNTIME, capture_output=True, text=True, timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    prefix = "PUBLICATION_SPEC_RESULT="
    return json.loads(next(line[len(prefix):] for line in process.stdout.splitlines() if line.startswith(prefix)))


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
def test_public_spec_preserves_prechange_parameters_outputs_gradients_and_rng(spec_probe, binary, task):
    row = next(r for r in spec_probe["compatibility"] if r["binary"] == binary and r["task"] == task)
    assert all(row[key] for key in ("parameter_names", "parameters", "outputs", "gradients", "rng", "two_layers"))


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
def test_supplement_runs_three_four_head_layers_and_backpropagates(spec_probe, binary, task):
    row = next(r for r in spec_probe["supplement"] if r["binary"] == binary and r["task"] == task)
    assert row["spec"]
    assert row["calls"] == ["layer1", "layer2", "layer3"] * 3
    assert row["finite_losses"] and row["layer_gradients"] and row["intermediate_gradients"]
    assert row["weights_unchanged"]
    assert row["strict_rejections"] == 2


def test_supplement_rejects_mislabeled_architectures(spec_probe):
    assert all(spec_probe["validation"])


@pytest.mark.parametrize("binary", [False, True])
def test_actual_dgl_attention_normalizes_per_receiver_head_and_relation(spec_probe, binary):
    row = next(r for r in spec_probe["attention"] if r["binary"] == binary)
    assert row["relations"] == ["a2a", "a2p", "a2s", "p2a", "p2p", "p2s"]
    assert row["two_relation_mass"] and row["not_joint_softmax"]
