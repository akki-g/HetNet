"""Gradient references for the finite-horizon common-team objective."""
from types import SimpleNamespace

import pytest
import torch

from softrole.learning import Episode, compute_episode_loss, loss_sum, team_gae
from softrole.model import straight_through_bits


def loss_config(**overrides):
    return SimpleNamespace(**dict(gamma=1.0, gae_lambda=1.0, actor_coeff=1.0,
                                   value_coeff=0.0, **overrides))


def test_team_gae_at_finite_horizon_and_explicit_bootstrap():
    values = torch.tensor([2., 3., 4.], dtype=torch.float64, requires_grad=True)
    advantages, targets = team_gae([1., 2., 3.], values, gae_lambda=1.)
    assert torch.equal(targets, torch.tensor([6., 5., 3.], dtype=values.dtype))
    assert torch.equal(advantages, targets - values.detach())
    assert not advantages.requires_grad and not targets.requires_grad
    _, continuing = team_gae([1., 2., 3.], values, gae_lambda=1., bootstrap=7.)
    assert torch.equal(continuing, torch.tensor([13., 12., 10.], dtype=values.dtype))


def test_lambda_zero_is_one_step_td():
    advantages, _ = team_gae([1., 2.], torch.tensor([4., 8.]), gae_lambda=0.)
    assert torch.equal(advantages, torch.tensor([5., -6.]))


def test_two_agent_enumeration_recovers_cross_agent_reward_gradient():
    # Agent 1's action rewards only agent 0. Own-agent credit loses this entire
    # derivative; common team credit reproduces the enumerated objective.
    theta = torch.tensor([-0.4, 0.7], dtype=torch.float64, requires_grad=True)
    probabilities = theta.sigmoid()
    estimate = torch.zeros_like(theta)
    for a0 in (0., 1.):
        for a1 in (0., 1.):
            actions = torch.tensor([a0, a1], dtype=theta.dtype)
            dist = torch.distributions.Bernoulli(logits=theta)
            weight = dist.log_prob(actions).sum().exp().detach()
            episode = Episode(log_probs=[dist.log_prob(actions).sum()],
                              values=[theta.sum() * 0], rewards=[a1])
            loss, _ = compute_episode_loss(episode, loss_config())
            estimate -= weight * torch.autograd.grad(loss, theta)[0]
    exact = torch.autograd.grad(probabilities[1], theta)[0]
    assert torch.allclose(estimate, exact, atol=1e-12)
    assert exact[0] == 0 and exact[1] > 0


def test_episode_sums_do_not_normalize_advantages_or_weight_by_length():
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    first = Episode(log_probs=[theta], values=[theta * 0], rewards=[2.])
    second = Episode(log_probs=[theta, theta], values=[theta * 0, theta * 0], rewards=[1., 3.])
    loss, stats = loss_sum([first, second], loss_config())
    # Lambda=1: first advantage=2, second advantages=(4,3).
    assert torch.allclose(torch.autograd.grad(loss, theta)[0], theta.new_tensor(-9.))
    assert stats["episode_count"] == 2
    # Global episode averaging is exactly one division, performed by the engine.
    assert stats["policy_loss_sum"] == pytest.approx(-1.8)


def test_binary_score_function_reference_is_exact_and_st_is_biased():
    u = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    w, offset = 4., -1.
    bit_distribution = torch.distributions.Bernoulli(logits=u)
    exact_return = ((1 - u.sigmoid()) * torch.sigmoid(u.new_tensor(offset))
                    + u.sigmoid() * torch.sigmoid(u.new_tensor(w + offset)))
    exact = torch.autograd.grad(exact_return, u)[0]
    score_gradient = torch.zeros_like(u)
    for bit in (0., 1.):
        b = u.new_tensor(bit)
        bit_probability = bit_distribution.log_prob(b).exp().detach()
        action_probability = torch.sigmoid(u.new_tensor(w * bit + offset))
        score = torch.autograd.grad(bit_distribution.log_prob(b), u, retain_graph=True)[0]
        score_gradient += bit_probability * action_probability * score
    assert torch.allclose(score_gradient, exact, atol=1e-12)
    # Deterministic quadrature over exogenous logistic noise avoids a flaky MC
    # assertion. ST is a training approximation, never an unbiased reference.
    uniforms = (torch.arange(40000, dtype=u.dtype) + .5) / 40000
    logistic = uniforms.log() - torch.log1p(-uniforms)
    bits = straight_through_bits(u.expand_as(logistic), logistic)
    st_return = torch.sigmoid(w * bits + offset).mean()
    st_gradient = torch.autograd.grad(st_return, u)[0]
    assert abs(st_gradient - exact) > 0.01


def test_empty_rollout_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        loss_sum([], loss_config())
