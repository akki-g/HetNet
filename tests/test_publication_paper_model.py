"""Paper architecture and independently checked backend mathematics.

Run the isolated runtime in a subprocess: it deliberately shares top-level
module names with the untouched original implementation.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "publication_reconstruction/runtime"


@pytest.fixture(scope="module")
def paper_probe():
    code = r'''
import json
import sys
from pathlib import Path

runtime = Path(sys.argv[1])
sys.path[:0] = [str(runtime), str(runtime / "envs")]
import numpy as np
import torch
import dgl
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
from hetgat.paper import PaperNet
from hetgat.graph.paper import PaperLayer
from hetgat.graph.torch_backend import StaticTopology, aggregate, RELATIONS
from hetgat.utils import build_hetgraph


def graph(p, a, offset=0):
    positions = [np.eye(25)[(i + offset) % 25] for i in range(p + a)]
    return build_hetgraph(positions, num_P=p, num_A=a,
                         with_state=True, with_two_state=True)


def model(task, binary, backend, p=None, a=None, **extra):
    p = (3 if task == "pp" else 2) if p is None else p
    a = (0 if task == "pp" else 1) if a is None else a
    vision = 1 if task == "fc" else 2
    out = dict(P=4, A=5, state=8) if task == "fc" else dict(P=5, A=6, state=8)
    return PaperNet(dict(vision=vision, P=29, A=25, state=4),
        dict(P=29, A=25, state=4), dict(P=16, A=16, state=16), out,
        p, a, num_heads=4, msg_dim=extra.pop("msg_dim", 64), use_real=not binary,
        per_class_critic=True, with_two_state=True, obs=(2*vision+1)**2,
        device="cpu", message_backend=backend, **extra)


def tensors(value):
    if isinstance(value, dict):
        return [t for key in sorted(value) for t in tensors(value[key])]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in tensors(item)]
    return [value]


def close(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
    atol, rtol = (1e-6, 1e-5) if left.dtype == torch.float32 else (1e-10, 1e-8)
    torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def gradient_map(m):
    return {name: None if p.grad is None else p.grad.clone()
            for name, p in m.named_parameters()}


results = {"matrix": [], "contracts": {}}
for task in ("pp", "pcp", "fc"):
    for binary in (False, True):
        models, initial_states, initial_rngs = [], [], []
        for backend in ("dgl", "torch-v1"):
            torch.manual_seed(714)
            m = model(task, binary, backend)
            models.append(m)
            initial_states.append({k: v.clone() for k, v in m.state_dict().items()})
            initial_rngs.append(torch.get_rng_state())
        assert initial_states[0].keys() == initial_states[1].keys()
        assert all(torch.equal(initial_states[0][k], initial_states[1][k]) for k in initial_states[0])
        assert torch.equal(*initial_rngs)
        m = models[0]
        g = graph(m.num_P, m.num_A)
        generator = torch.Generator().manual_seed(187)
        observations = [torch.randn(1, m.num_P+m.num_A, 29*m.obs_squares,
                                     generator=generator) for _ in range(6)]
        traces, gradients, input_gradients, rngs = [], [], [], []
        for m in models:
            torch.manual_seed(1223)
            inputs = [o.clone().requires_grad_() for o in observations]
            hidden = m.init_hidden(1)
            loss = 0
            trace = []
            for step, observation in enumerate(inputs):
                output = m([observation, hidden], g, episode_step=step)
                hidden = output[-1]
                # Different output weights exercise all head dimensions.
                loss = loss + sum((i+1)*t.square().sum()
                                  for i, t in enumerate(tensors(output[:-1])))
                trace.extend(t.detach().clone() for t in tensors(output))
                if (step+1) % 5 == 0:
                    hidden = {k: tuple(t.detach() for t in pair) for k, pair in hidden.items()}
            loss.backward()
            traces.append(trace)
            gradients.append(gradient_map(m))
            input_gradients.append([o.grad.clone() for o in inputs])
            rngs.append(torch.get_rng_state())
        for left, right in zip(traces[0], traces[1]):
            close(left, right)
        assert gradients[0].keys() == gradients[1].keys()
        max_gradient_difference = 0
        for key in gradients[0]:
            left, right = gradients[0][key], gradients[1][key]
            if left is None or right is None:
                assert left is right, key
            else:
                close(left, right)
                if left.numel():
                    max_gradient_difference = max(max_gradient_difference, float((left-right).abs().max()))
        for left, right in zip(input_gradients[0], input_gradients[1]):
            close(left, right)
        assert torch.equal(*rngs)
        assert not any(g.nodes[key].data for key in g.ntypes), "DGL graph retained learned features"
        assert not any(g.edges[key].data.keys() - {"dist"} for key in g.canonical_etypes)
        # Repeated cached forwards and reset reproduce exactly within a backend.
        for m in models:
            torch.manual_seed(91)
            first = m([observations[0], m.init_hidden(1)], g, episode_step=0)
            torch.manual_seed(91)
            second = m([observations[0], m.init_hidden(1)], g, episode_step=0)
            assert all(torch.equal(l, r) for l, r in zip(tensors(first), tensors(second)))
        results["matrix"].append({"task": task, "binary": binary,
            "max_parameter_gradient_absolute_difference": max_gradient_difference,
            "recurrent_forward_backward_rng": True})

# Paper Eq.2: class relations normalize separately and add to the self transform.
# Equal attention gives P outputs 1+3+5=9, A output 5+(1+3)/2=7;
# class-routed SENs receive 2 and 5. A global sender softmax does not satisfy this.
features = {"P": torch.tensor([[1.], [3.]]), "A": torch.tensor([[5.]]),
            "state": torch.zeros(2, 1)}
for backend in ("dgl", "torch-v1"):
    layer = PaperLayer(dict(P=1,A=1,state=1), dict(P=1,A=1,state=1),
                       num_heads=1, merge="avg", message_backend=backend)
    with torch.no_grad():
        for affine in layer.heads[0].fc.values():
            affine.weight.fill_(1)
            affine.bias.zero_()
        for parameter in layer.heads[0].attention.values():
            parameter.zero_()
    output = layer(graph(2,1), features, StaticTopology(2,1))
    assert torch.equal(output["P"], torch.tensor([[9.], [9.]]))
    assert torch.equal(output["A"], torch.tensor([[7.]]))
    assert torch.equal(output["state"], torch.tensor([[2.], [5.]]))
results["contracts"]["independent_relation_equation"] = True

# Independently evaluate a nonuniform relation using a loop over heads/receivers.
messages = torch.tensor([[[1.,-2.],[2.,4.]], [[-3.,1.],[1.,-1.]], [[4.,2.],[-2.,3.]]], requires_grad=True)
receiver = torch.tensor([[[2.,1.],[-1.,2.]], [[1.,1.],[1.,1.]]], requires_grad=True)
src = torch.tensor([[.7,-.2],[.2,-.6]], dtype=torch.float32, requires_grad=True)
dst = torch.tensor([[.4,.1],[-.1,.3]], dtype=torch.float32, requires_grad=True)
mask = torch.tensor([[True,False,True], [False,False,False]])
actual = aggregate(messages, receiver, src, dst, mask)
expected = torch.zeros_like(actual)
for r in range(2):
    senders = [s for s in range(3) if mask[r,s]]
    for h in range(2):
        if senders:
            scores = torch.stack([sum(messages[s,h,d]*src[h,d]+receiver[r,h,d]*dst[h,d]
                                      for d in range(2)) for s in senders])
            scores = torch.where(scores >= 0, scores, .2*scores)
            weights = torch.exp(scores - scores.max())
            weights = weights / weights.sum()
            expected[r,h] = sum(w*messages[s,h] for w,s in zip(weights,senders))
close(actual, expected)
assert torch.equal(actual[1], torch.zeros(2,2))
actual.square().sum().backward()
assert all(bool(torch.isfinite(t.grad).all()) for t in (messages, receiver, src, dst))
results["contracts"]["nonuniform_attention_empty_receiver"] = True

# Head merging is checked against specified arithmetic, independently of either
# backend. Negative hidden features are ReLU'd before concatenation; the action
# layer averages raw head logits without ReLU.
for backend in ("dgl","torch-v1"):
    for merge, expected in (("cat", [[0.,0.,2.,4.]]), ("avg", [[.5]])):
        layer = PaperLayer(dict(P=1,A=1,state=1), dict(P=1,A=1,state=1),
                           merge=merge, message_backend=backend)
        with torch.no_grad():
            for head, bias in zip(layer.heads,(-3.,-1.,2.,4.)):
                for affine in head.fc.values():
                    affine.weight.zero_(); affine.bias.zero_()
                head.fc["P"].bias.fill_(bias)
        output = layer(graph(1,0), {"P":torch.ones(1,1),"A":torch.empty(0,1),
                                   "state":torch.zeros(2,1)}, StaticTopology(1,0))
        assert torch.equal(output["P"],torch.tensor(expected))
results["contracts"]["hidden_relu_concat_final_raw_average"] = True

# Large finite parameters exercise stable softmax and saturated Gumbel paths,
# rather than relying solely on initialization-scale random comparisons.
for binary in (False,True):
    torch.manual_seed(613)
    dgl_layer = PaperLayer(dict(P=2,A=2,state=2), dict(P=2,A=2,state=2),
                          use_real=not binary, merge="avg", message_backend="dgl")
    with torch.no_grad():
        for name, parameter in dgl_layer.named_parameters():
            parameter.mul_(1e5 if ".attention." in name else 1e2)
    dense_layer = PaperLayer(dict(P=2,A=2,state=2), dict(P=2,A=2,state=2),
                            use_real=not binary, merge="avg", message_backend="torch-v1")
    dense_layer.load_state_dict(dgl_layer.state_dict(),strict=True)
    features = {"P":torch.randn(3,2)*100, "A":torch.randn(2,2)*100,
                "state":torch.randn(2,2)*100}
    traces, gradients, rngs = [], [], []
    for layer in (dgl_layer,dense_layer):
        torch.manual_seed(937)
        output = layer(graph(3,2),features,StaticTopology(3,2))
        sum(t.square().sum() for t in output.values()).backward()
        traces.append(output)
        gradients.append(gradient_map(layer))
        rngs.append(torch.get_rng_state())
    for key in traces[0]:
        close(traces[0][key],traces[1][key])
    for key in gradients[0]:
        left,right = gradients[0][key],gradients[1][key]
        if left is None or right is None:
            assert left is right
        else:
            close(left,right)
    assert torch.equal(*rngs)
results["contracts"]["extreme_finite_parameters"] = True

torch.manual_seed(50)
m = model("pcp", True, "torch-v1")
assert m.prepro_stat["P"].weight.data_ptr() != m.prepro_stat["A"].weight.data_ptr()
assert m.f_module_stat["P"].weight_ih.data_ptr() != m.f_module_stat["A"].weight_ih.data_ptr()
for layer in (m.layer1,m.layer2,m.layer3):
    assert len(layer.heads) == 4 and layer.msg_dim == 64
    pointers = [h.encoder["P"].weight.data_ptr() for h in layer.heads]
    assert len(set(pointers)) == 4
    assert len({h.binarize.weight.data_ptr() for h in layer.heads}) == 4
    assert all(h.encoder["P"].out_features == 64 and h.decoder["A"].in_features == 64 for h in layer.heads)
    assert all(p.dtype == torch.float32 for h in layer.heads for p in h.attention.values())
assert m.layer1.merge == m.layer2.merge == "cat" and m.layer3.merge == "avg"
results["contracts"]["class_specific_recurrence_independent_64bit_heads"] = True

# Check actual stochastic call shapes and independence; expose no diagnostic in actor inputs.
original_gumbel = torch.nn.functional.gumbel_softmax
payloads = []
def observed_gumbel(logits, *args, **kwargs):
    result = original_gumbel(logits, *args, **kwargs)
    payloads.append(result.detach().clone())
    return result
torch.nn.functional.gumbel_softmax = observed_gumbel
observation = torch.randn(1,3,29*m.obs_squares)
torch.manual_seed(821)
m([observation,m.init_hidden(1)], episode_step=3)
torch.nn.functional.gumbel_softmax = original_gumbel
assert len(payloads) == 3*4*2
assert all(tuple(t.shape) == ((2 if i%2==0 else 1),64,2) for i,t in enumerate(payloads))
assert all(torch.equal(t.sum(-1), torch.ones_like(t[...,0])) for t in payloads)
assert not torch.equal(payloads[0], payloads[2])
results["contracts"]["binary_draw_shape_order"] = True

seen = []
handle = m.layer1.register_forward_pre_hook(lambda module,args: seen.append(args[1]["state"].clone()))
torch.manual_seed(27)
first = m([observation,m.init_hidden(1)], episode_step=0)
torch.manual_seed(27)
later = m([observation,m.init_hidden(1)], episode_step=19)
handle.remove()
assert torch.equal(seen[0], torch.tensor([[2.,1.,5.,0.],[2.,1.,5.,0.]]))
assert torch.equal(seen[1], torch.tensor([[2.,1.,5.,19.],[2.,1.,5.,19.]]))
assert torch.equal(first[0]["P"],later[0]["P"]) and torch.equal(first[0]["A"],later[0]["A"])
assert not torch.equal(first[1],later[1]) or not torch.equal(first[2],later[2])
results["contracts"]["timestep_critic_only"] = True

# Model-owned topology is isolated, has no tensor entries in checkpoint state,
# and Torch forward makes no graph calls even on an empty-A singleton team.
left = model("pp", True, "torch-v1", p=1, a=0)
right = model("pp", True, "torch-v1", p=1, a=0)
keys_before = list(left.state_dict())
original_builder = dgl.heterograph
dgl.heterograph = lambda *a, **k: (_ for _ in ()).throw(AssertionError("DGL called"))
for net in (left,right):
    obs = torch.randn(1,1,29*net.obs_squares)
    output = net([obs,net.init_hidden(1)])
    assert output[0]["A"].shape == (0,6)
dgl.heterograph = original_builder
assert left._topology is not right._topology
assert all(left._topology.masks[k].data_ptr() != right._topology.masks[k].data_ptr()
           for k in left._topology.masks if left._topology.masks[k].numel())
assert list(left.state_dict()) == keys_before
assert not list(left.named_buffers())
results["contracts"]["isolated_cache_empty_relations_singleton_no_graph"] = True

# Copy/strided/cast semantics are shared across backends and preserve ownership.
for dtype in (torch.float32,torch.float64):
    m = model("pcp",False,"torch-v1")
    width = 29*m.obs_squares
    raw = torch.arange(3*2*width,dtype=dtype).reshape(1,3,2*width)[:,:,::2]
    assert not raw.is_contiguous()
    p,a = m.remove_excess_action_features_from_all(raw)
    sensory = m.get_obs_features(raw)
    expected = raw.reshape(3,m.obs_squares,29)
    assert p.dtype == a.dtype == sensory.dtype == torch.float64
    assert torch.equal(p,expected[:2,:,:25].reshape(2,-1).double())
    assert torch.equal(a,expected[2:,:,:25].reshape(1,-1).double())
    assert torch.equal(sensory,expected[:2,:,25:].reshape(2,-1).double())
    assert p.is_contiguous() and a.is_contiguous() and sensory.is_contiguous()
    previous = raw.clone()
    p.zero_();a.zero_();sensory.zero_()
    assert torch.equal(previous,raw)
    counterpart = model("pcp",False,"dgl")
    counterpart.load_state_dict(m.state_dict(),strict=True)
    values, gradients = [], []
    for net in (m,counterpart):
        input_tensor = (raw/float(raw.max())).detach().requires_grad_()
        output = net([input_tensor,net.init_hidden(1)],graph(2,1))
        sum(t.square().sum() for t in tensors(output[:-1])).backward()
        values.append(tensors(output));gradients.append(input_tensor.grad)
    for left,right in zip(*values):
        close(left,right)
    close(*gradients)
results["contracts"]["strided_dtype_independent_observation_copies"] = True

# Unlimited topology is position-independent; compare its mask to actual edge IDs.
for p,a in ((3,0),(2,1),(1,0),(1,2)):
    topology = StaticTopology(p,a)
    for shift in (0,11):
        g = graph(p,a,shift)
        for source,name,target in RELATIONS:
            mask = torch.zeros(topology.counts[target],topology.counts[source],dtype=torch.bool)
            s,d = g.edges(etype=name)
            mask[d,s] = True
            assert torch.equal(mask,topology.masks[name])
results["contracts"]["topology_matches_changed_positions"] = True

for invalid in (-1, .5, True):
    try:
        m.set_episode_step(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid timestep accepted")
results["contracts"]["invalid_timestep_rejected"] = True

# The real-valued CLI retains its unused msg_dim=16 parser default. Only Binary
# uses the explicitly chosen 64-bit head width; neither backend changes this.
for backend in ("dgl","torch-v1"):
    m = model("pcp",False,backend,msg_dim=16)
    m([torch.zeros(1,3,29*m.obs_squares),m.init_hidden(1)],graph(2,1))
    try:
        model("pcp",True,backend,msg_dim=16)
    except ValueError:
        pass
    else:
        raise AssertionError("Incorrect Binary bandwidth accepted")
for options in ({"comm_range_P":2}, {"comm_range_A":0}, {"lossy_comm":True}):
    try:
        model("pcp",False,"torch-v1",**options)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported static topology accepted")
results["contracts"]["bandwidth_and_unsupported_topology_validation"] = True
print("PAPER_MODEL_PROBE="+json.dumps(results))
'''
    result = subprocess.run([sys.executable, "-c", code, str(RUNTIME)],
                            cwd=ROOT, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    prefix = "PAPER_MODEL_PROBE="
    return json.loads(next(line[len(prefix):] for line in result.stdout.splitlines()
                           if line.startswith(prefix)))


@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
@pytest.mark.parametrize("binary", [False, True])
def test_paper_recurrent_backends_match_outputs_gradients_and_rng(paper_probe, task, binary):
    result = next(row for row in paper_probe["matrix"]
                  if row["task"] == task and row["binary"] == binary)
    assert result["recurrent_forward_backward_rng"]


@pytest.mark.parametrize("contract", [
    "independent_relation_equation", "nonuniform_attention_empty_receiver",
    "hidden_relu_concat_final_raw_average", "extreme_finite_parameters",
    "class_specific_recurrence_independent_64bit_heads", "binary_draw_shape_order",
    "timestep_critic_only", "isolated_cache_empty_relations_singleton_no_graph",
    "strided_dtype_independent_observation_copies", "topology_matches_changed_positions",
    "invalid_timestep_rejected", "bandwidth_and_unsupported_topology_validation",
])
def test_paper_architecture_contracts(paper_probe, contract):
    assert paper_probe["contracts"][contract]
