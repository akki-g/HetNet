"""Opt-in action entropy preserves old losses and encourages feasible exploration."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Categorical

from softrole.config import Config
from softrole.env import make_env
from softrole.learning import Episode, compute_episode_loss
from softrole.model import SoftRoleNet
from softrole.rollout import run_episode
from softrole.scenarios import Scenario


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "0.1"])
def test_entropy_coefficient_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="entropy_coeff"):
        Config(entropy_coeff=value)


def test_old_config_and_zero_bonus_preserve_loss_gradient_and_model_layout():
    config = Config()
    old = config.to_dict()
    assert "entropy_coeff" not in old
    assert Config(**old).entropy_coeff == 0
    assert Config(**old).to_dict() == old
    assert Config(**dict(old, entropy_coeff=0.)).to_dict() == old
    assert replace(config, entropy_coeff=.01).to_dict()["entropy_coeff"] == .01
    assert config.model_kwargs() == replace(config, entropy_coeff=.01).model_kwargs()
    theta = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    episode = Episode(log_probs=[theta, 2 * theta], values=[theta, theta], rewards=[1., 3.])
    legacy = SimpleNamespace(**old)
    expected, old_stats = compute_episode_loss(episode, legacy)
    actual, stats = compute_episode_loss(episode, config)
    # Independent two-step lambda-return calculation, including detached critic
    # bootstrap in actor advantages, pins the pre-bonus numerical contract.
    last_advantage = 3. - theta.detach()
    first_advantage = 1. + .95 * last_advantage
    reference = (-50 * (theta * first_advantage + 2 * theta * last_advantage)
                 + torch.stack([(theta - (first_advantage + theta.detach())).square(),
                                (theta - 3.).square()]).mean())
    assert torch.equal(actual, reference)
    assert torch.equal(actual, expected)
    assert stats == old_stats
    assert torch.equal(torch.autograd.grad(actual, theta, retain_graph=True)[0],
                       torch.autograd.grad(expected, theta)[0])


@pytest.mark.parametrize("objective,scale", [("team", 1.), ("mean_agent", .5)])
def test_entropy_bonus_gradient_promotes_feasible_exploration_with_time_agent_sum(objective, scale):
    # Each of two agents has two feasible actions. Third action is physically masked.
    theta = torch.tensor([[2., 0., 7.], [2., 0., -9.]], dtype=torch.float64, requires_grad=True)
    logits = theta.masked_fill(torch.tensor([[False, False, True]]), -torch.inf)
    dist = Categorical(logits=logits)
    zero = theta.sum() * 0
    episode = Episode(log_probs=[zero, zero], values=[zero, zero], rewards=[0., 0.],
                      agent_rewards=[[0., 0.], [0., 0.]],
                      agent_entropies=[dist.entropy(), dist.entropy()])
    config = Config(entropy_coeff=.01, objective=objective)
    loss, stats = compute_episode_loss(episode, config)
    grad = torch.autograd.grad(loss, theta)[0]
    p = torch.sigmoid(torch.tensor(2., dtype=theta.dtype))
    # d(-H(sigmoid(z)))/dz = z p(1-p); two time steps, actor coefficient 50.
    expected = scale * 50 * .01 * 2 * 2 * p * (1 - p)
    assert torch.allclose(grad[:, 0], expected.expand(2))
    assert torch.allclose(grad[:, 1], -expected.expand(2))
    assert torch.equal(grad[:, 2], torch.zeros(2, dtype=theta.dtype))
    assert torch.isfinite(grad).all()
    assert loss < 0 and stats["objective_entropy_loss_sum"] == float(loss)
    before = dist.probs[:, 0]
    after = Categorical(logits=(theta - .1 * grad).masked_fill(
        torch.tensor([[False, False, True]]), -torch.inf)).probs[:, 0]
    assert (after < before).all()


def test_entropy_bonus_requires_complete_records():
    theta = torch.tensor(0., requires_grad=True)
    episode = Episode(log_probs=[theta], values=[theta], rewards=[0.])
    with pytest.raises(ValueError, match="every step"):
        compute_episode_loss(episode, Config(entropy_coeff=.01))


def test_rollout_bonus_reuses_same_trajectory_and_has_finite_gradient():
    torch.manual_seed(71)
    config = Config(max_steps=3, model="shared", hidden_dim=8, pre_dim=8,
                    heads=1, head_dim=4, msg_dim=4)
    model = SoftRoleNet(**config.model_kwargs()).double()
    scenario = Scenario("entropy", 11, 12, 13, 2, 1)
    plain = run_episode(model, make_env(config), config, scenario, training=True, trace=True)
    enabled_config = replace(config, entropy_coeff=.01)
    enabled = run_episode(model, make_env(enabled_config), enabled_config, scenario,
                          training=True, trace=True)
    assert plain.traces == enabled.traces
    assert plain.metrics == enabled.metrics
    assert not plain.agent_entropies
    assert len(enabled.agent_entropies) == len(enabled.values)
    assert all(h.shape == (3,) and torch.isfinite(h).all() for h in enabled.agent_entropies)
    baseline, _ = compute_episode_loss(enabled, config)
    bonus, _ = compute_episode_loss(enabled, enabled_config)
    assert bonus < baseline
    bonus.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
