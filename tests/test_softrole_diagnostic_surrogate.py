"""Independent conditional references and noninterference for the D5 probe."""

import math

import pytest
import torch

from softrole.diagnostic_surrogate import _conditional_gradients, audit_surrogate
from softrole.model import SoftRoleNet


def tiny_case(**kwargs):
    model = SoftRoleNet(base=1, n_squares=1, mode="shared", pre_dim=2,
                        hidden_dim=2, heads=2, head_dim=2, msg_dim=2,
                        comm_rounds=2, **kwargs).double()
    obs = torch.zeros(2, model.obs_dim, dtype=torch.float64)
    kappa = torch.ones(2, 2, dtype=torch.float64)
    shape = (2, model.heads, model.msg_dim) if model.independent_heads else (2, model.msg_dim)
    noise = [torch.zeros(shape, dtype=torch.float64) for _ in model.rounds]
    return model, dict(obs=obs, kappa=kappa, memory=model.initial_memory(2),
                       adjacency=~torch.eye(2, dtype=torch.bool), remaining=.8,
                       noise=noise, actions=torch.zeros(2, dtype=torch.long), advantages=1.)


def test_conditional_enumeration_matches_analytic_nonlinear_expectation():
    logits = torch.tensor([.3, -.7], dtype=torch.float64)
    result = _conditional_gradients(logits, lambda bits: (4 * bits.sum() - 1).sigmoid(),
                                    mc_samples=65536, seed=21)
    # Analytic derivative from the four scalar downstream outcomes, independent
    # of the score-function accumulation used in the implementation.
    p, q = logits.sigmoid().tolist()
    f0, f1, f2 = (1 / (1 + math.exp(-value)) for value in (-1, 3, 7))
    expected = logits.new_tensor([p * (1-p) * ((1-q) * (f1-f0) + q * (f2-f1)),
                                  q * (1-q) * ((1-p) * (f1-f0) + p * (f2-f1))])
    torch.testing.assert_close(result["exact"], expected)
    assert (result["st"] - expected).norm() > .02
    assert sum(entry["probability"] for entry in result["patterns"]) == pytest.approx(1.)


def test_actual_actor_encoder_gradient_has_known_conditional_reference():
    model = SoftRoleNet(base=1, n_squares=1, mode="shared", pre_dim=1,
                        hidden_dim=1, heads=1, head_dim=1, msg_dim=1,
                        comm_rounds=1).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.rounds[0].encoder.bias.fill_(.3)
        model.rounds[0].decoder.weight.fill_(4.)
        model.output.weight[0].fill_(1.)
    # Only sender 1 reaches receiver 0. Equal null/message scores halve the
    # decoded value, so log pi_0(action=0) = 2b - log(exp(2b) + 5).
    result = audit_surrogate(model, obs=torch.zeros(2, model.obs_dim, dtype=torch.float64),
                             kappa=torch.ones(2, 2, dtype=torch.float64),
                             memory=model.initial_memory(2),
                             adjacency=torch.tensor([[False, True], [False, False]]),
                             remaining=1., noise=[torch.zeros(2, 1, dtype=torch.float64)],
                             actions=torch.zeros(2, dtype=torch.long),
                             advantages=torch.tensor([1., 0.], dtype=torch.float64),
                             bit_indices=[1], mc_samples=65536, seed=71)
    p = 1 / (1 + math.exp(-.3))
    expected = p * (1-p) * (2 - math.log(math.exp(2) + 5) + math.log(6))
    assert result["score_encoder_gradient_norm"] == pytest.approx(expected)
    assert result["encoder_gradient_cosine"] == pytest.approx(1.)
    assert result["enumerated_patterns"][0]["objective"] == pytest.approx(-math.log(6))
    assert result["enumerated_patterns"][1]["objective"] == pytest.approx(2 - math.log(math.exp(2)+5))
    # Independent midpoint integration of the one-bit production ST derivative.
    uniform = (torch.arange(40000, dtype=torch.float64) + .5) / 40000
    noisy = .3 + uniform.log() - torch.log1p(-uniform)
    hard = (noisy > 0).double()
    action_probability = (2*hard).exp() / ((2*hard).exp() + 5)
    relaxed = noisy.sigmoid()
    st_reference = (2 * (1-action_probability) * relaxed * (1-relaxed)).mean().item()
    assert result["st_encoder_gradient_norm"] == pytest.approx(st_reference, abs=.002)


@pytest.mark.parametrize("independent_heads", [False, True])
def test_frozen_checkpoint_probe_is_replayable_and_does_not_mutate(independent_heads):
    model, arguments = tiny_case(independent_heads=independent_heads)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    memory_before = {name: value.clone() for name, value in arguments["memory"].items()}
    rng_before = torch.random.get_rng_state().clone()
    first = audit_surrogate(model, **arguments, round_index=0, n_bits=3, seed=9)
    second = audit_surrogate(model, **arguments, round_index=0, n_bits=3, seed=9)
    assert first == second
    assert len(first["enumerated_patterns"]) == 8
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    for parameter in model.parameters():
        assert not parameter.requires_grad
        assert torch.equal(parameter.grad, torch.ones_like(parameter))
    for name, value in arguments["memory"].items():
        assert torch.equal(value, memory_before[name])
    for layer in model.rounds:
        assert not layer.decoder._forward_pre_hooks
        assert not layer.encoder._forward_pre_hooks


def test_zero_objective_has_undefined_cosine_and_real_has_no_binary_audit():
    model, arguments = tiny_case()
    result = audit_surrogate(model, **{**arguments, "advantages": 0.})
    assert result["encoder_gradient_cosine"] is None
    assert result["score_encoder_gradient_norm"] == result["st_encoder_gradient_norm"] == 0.
    real_model, real_arguments = tiny_case(communication="real")
    assert audit_surrogate(real_model, **real_arguments)["status"] == "not_applicable"


def test_default_selection_excludes_senders_without_permitted_receivers():
    model, arguments = tiny_case()
    arguments["adjacency"] = torch.tensor([[False, True], [False, False]])
    result = audit_surrogate(model, **arguments, n_bits=4)
    assert set(result["selected_flat_indices"]) == {2, 3}
    arguments["adjacency"].zero_()
    assert audit_surrogate(model, **arguments)["status"] == "not_applicable"


@pytest.mark.parametrize("override", [{"n_bits": 5}, {"n_bits": True}, {"mc_samples": 1},
                                     {"round_index": 2}, {"seed": -1},
                                     {"bit_indices": [0, 0]}, {"bit_indices": [999]},
                                     {"noise": None}, {"advantages": float("nan")}])
def test_invalid_probe_configuration_fails_before_changing_model(override):
    model, arguments = tiny_case()
    with pytest.raises(ValueError):
        audit_surrogate(model, **{**arguments, **override})
    assert all(not layer.decoder._forward_pre_hooks for layer in model.rounds)


def test_failed_downstream_probe_removes_temporary_hooks():
    model, arguments = tiny_case()
    calls = 0

    def nonfinite_after_logged_forward(_module, _inputs, output):
        nonlocal calls
        calls += 1
        return output if calls == 1 else output * float("nan")

    handle = model.output.register_forward_hook(nonfinite_after_logged_forward)
    try:
        with pytest.raises(ValueError, match="finite scalar"):
            audit_surrogate(model, **arguments)
    finally:
        handle.remove()
    assert all(not layer.decoder._forward_pre_hooks for layer in model.rounds)
    assert all(not layer.encoder._forward_pre_hooks for layer in model.rounds)
