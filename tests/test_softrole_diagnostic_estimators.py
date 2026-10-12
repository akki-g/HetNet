"""Independent finite-horizon and analytic gradient references for D3/D4."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from softrole.diagnostic_estimators import (
    GradientMoments, actor_episode_gradients, critic_episode_diagnostics,
    discounted_returns, summarize_critic_diagnostics,
)
from softrole.learning import Episode, compute_episode_loss


DTYPE = torch.float64


def config(**changes):
    values = dict(gamma=1., gae_lambda=.95, actor_coeff=50., value_coeff=0.)
    values.update(changes)
    return SimpleNamespace(**values)


class Scores(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([-.4, .7], dtype=DTYPE))
        self.value = nn.Parameter(torch.tensor(1., dtype=DTYPE))


def test_discounted_returns_match_realized_finite_horizon_sums():
    rewards = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=DTYPE)
    result = discounted_returns(rewards, .5)
    assert torch.equal(result, torch.tensor([[3.75, 5.5], [5.5, 7.], [5., 6.]], dtype=DTYPE))
    assert torch.equal(discounted_returns(rewards.sum(-1)), torch.tensor([21., 18., 11.], dtype=DTYPE))


def test_critic_mc_accuracy_and_finite_horizon_proxy_have_known_values():
    values = torch.tensor([10., 4., 1.], dtype=DTYPE, requires_grad=True)
    rewards = torch.diag(torch.tensor([1., 2., 3.], dtype=DTYPE))
    result = critic_episode_diagnostics(values, rewards, gae_lambda=.5)
    # G=(6,5,3), V-G=(4,-1,-2). Future-only proxy=(-1,-1,0).
    assert result["mc_mse"] == 7.
    assert result["lambda_target_mse"] == pytest.approx(29 / 3)
    assert result["future_error_proxy_mean"] == pytest.approx(-2 / 3)
    assert result["future_error_proxy_absolute_sum"] == 2.
    assert result["future_error_proxy_squared_sum"] == 2.
    assert result["proxy_identity_max_abs_error"] == 0.
    assert values.grad is None


@pytest.mark.parametrize("gamma", [0., .8, 1.])
@pytest.mark.parametrize("lam", [0., .5, .95, 1.])
def test_proxy_equals_independent_mixture_of_n_step_targets(gamma, lam):
    values = torch.tensor([2., -3., 1., 4.], dtype=DTYPE)
    rewards = torch.tensor([[1., -.5], [0., 1.], [-2., 0.], [1., 3.]], dtype=DTYPE)
    team = rewards.sum(-1)
    # The finite lambda return is a mixture of n-step returns and the full
    # realized return. Build it without calling team_gae or the proxy recursion.
    expected_differences = []
    for t in range(len(values)):
        horizon = len(values) - t
        mixture = 0.
        mc = sum(gamma ** k * float(team[t + k]) for k in range(horizon))
        for n in range(1, horizon):
            target = sum(gamma ** k * float(team[t + k]) for k in range(n)) + gamma ** n * float(values[t + n])
            mixture += (1 - lam) * lam ** (n - 1) * target
        mixture += lam ** (horizon - 1) * mc
        expected_differences.append(mixture - mc)
    result = critic_episode_diagnostics(values, rewards, gamma, lam)
    assert result["future_error_proxy_sum"] == pytest.approx(sum(expected_differences), abs=1e-12)
    assert result["future_error_proxy_squared_sum"] == pytest.approx(sum(x*x for x in expected_differences), abs=1e-12)
    assert result["proxy_identity_max_abs_error"] < 1e-12
    if lam == 1:
        assert result["mc_mse"] == pytest.approx(result["lambda_target_mse"])
        assert result["future_error_proxy_rms"] == 0.


def test_critic_strata_keep_composition_length_and_weighting_distinct():
    first = critic_episode_diagnostics(torch.tensor([10., 4., 1.], dtype=DTYPE),
                                       torch.diag(torch.tensor([1., 2., 3.], dtype=DTYPE)), gae_lambda=.5)
    second = critic_episode_diagnostics(torch.tensor([0.], dtype=DTYPE), [[1., 1.]], gae_lambda=.5)
    records = [dict(composition=[2, 1], steps=3, diagnostics=first),
               dict(composition=[1, 1], steps=1, diagnostics=second)]
    result = summarize_critic_diagnostics(records)
    assert result["overall"]["episodes"] == 2
    assert result["overall"]["steps"] == 4
    assert result["overall"]["mc_mse"] == 25/4
    assert result["overall"]["equal_episode_mc_mse"] == 5.5
    assert result["by_composition"]["2,1"]["mc_mse"] == 7.
    assert result["by_episode_length"]["1"]["mc_mse"] == 4.
    assert set(result["by_composition_and_episode_length"]) == {"2,1;3", "1,1;1"}
    with pytest.raises(ValueError, match="Composition"):
        summarize_critic_diagnostics([{**records[0], "composition": [1, 1]}])
    with pytest.raises(ValueError, match="protocol"):
        summarize_critic_diagnostics([records[0], {**records[1], "diagnostics": {**second, "gamma": .9}}])


def test_actor_beta_one_matches_production_loss_and_preserves_state():
    model, cfg = Scores(), config()
    actions = torch.tensor([[0., 1.], [1., 0.], [1., 1.]], dtype=DTYPE)
    scores = torch.distributions.Bernoulli(logits=model.logits).log_prob(actions)
    values = model.value.expand(3) + .2 * model.logits.sum()
    rewards = torch.tensor([[1., 0.], [0., 2.], [-1., 3.]], dtype=DTYPE)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.full_like(p, 123.)
    rng = torch.random.get_rng_state().clone()
    result = actor_episode_gradients(scores, values, rewards, model, cfg, retain_graph=True)
    episode = Episode(log_probs=list(scores.sum(-1)), values=list(values), rewards=list(rewards.sum(-1)))
    loss, _ = compute_episode_loss(episode, cfg)
    original = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    expected = torch.cat([g.reshape(-1) for g in original])
    assert torch.allclose(result["gradients"]["team_gae"], expected, atol=1e-12)
    assert torch.allclose(result["gradients"]["allocated_gae_beta_1"], expected, atol=1e-12)
    assert result["checks"]["beta_1_advantage_max_abs_error"] < 1e-12
    assert result["checks"]["beta_1_gradient_max_abs_error"] < 1e-11
    assert result["gradients"]["allocated_gae_beta_0.5"][-1] == 0.
    assert "diagnostic allocation" in result["protocol"]["allocation"]
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])
        assert torch.all(p.grad == 123.)


def test_d4_monte_carlo_reference_detects_missing_cross_agent_credit():
    # Agent 1's action changes only agent 0's reward. Enumerating every action
    # shows beta=0 drops the entire real objective gradient, despite its lower
    # sampled-gradient variance being possible.
    model, cfg = Scores(), config(actor_coeff=1.)
    expected = torch.zeros(3, dtype=DTYPE)
    own_expected = expected.clone()
    half_expected = expected.clone()
    for a0 in (0., 1.):
        for a1 in (0., 1.):
            distribution = torch.distributions.Bernoulli(logits=model.logits)
            scores = distribution.log_prob(torch.tensor([[a0, a1]], dtype=DTYPE))
            probability = scores.sum().exp().detach()
            result = actor_episode_gradients(scores, model.value.reshape(1),
                                            [[a1, 0.]], model, cfg)
            expected -= probability * result["gradients"]["mc_beta_1"]
            own_expected -= probability * result["gradients"]["mc_beta_0"]
            half_expected -= probability * result["gradients"]["mc_beta_0.5"]
    p = model.logits[1].sigmoid().detach()
    assert torch.allclose(expected, torch.tensor([0., p * (1-p), 0.], dtype=DTYPE), atol=1e-12)
    assert torch.allclose(own_expected, torch.zeros_like(expected), atol=1e-12)
    assert torch.allclose(half_expected, expected / 2, atol=1e-12)


def test_d4_snr_uses_equal_episodes_and_population_gradient_variance():
    moments = GradientMoments()
    vectors = [torch.tensor(x, dtype=DTYPE) for x in ([1., 2.], [3., 0.], [2., 1.])]
    for vector in vectors:
        moments.add({"team_gae": vector, "scaled": vector * 10})
    result = moments.summary()["variants"]
    assert result["team_gae"]["mean_gradient_squared_l2"] == 5.
    assert result["team_gae"]["mean_centered_squared_l2"] == pytest.approx(4 / 3)
    assert result["team_gae"]["snr"] == pytest.approx(15 / 4)
    assert result["scaled"]["snr"] == pytest.approx(result["team_gae"]["snr"])
    assert result["scaled"]["mean_gradient_squared_l2"] == 500.
    assert result["scaled"]["cosine_mean_to_team_gae"] == pytest.approx(1.)


def test_snr_is_explicitly_undefined_for_one_episode_or_zero_variance():
    moments = GradientMoments()
    with pytest.raises(ValueError, match="empty"):
        moments.summary()
    vector = torch.tensor([1., 2.], dtype=DTYPE)
    moments.add({"team_gae": vector})
    assert moments.summary()["variants"]["team_gae"]["snr_status"] == "insufficient_episodes"
    moments.add({"team_gae": vector})
    result = moments.summary()["variants"]["team_gae"]
    assert result["snr"] is None
    assert result["snr_status"] == "zero_empirical_variance"


def test_gradient_batch_rejects_mismatches_without_partial_update():
    moments = GradientMoments()
    moments.add({"first": torch.ones(2), "second": torch.ones(2)})
    with pytest.raises(ValueError, match="coordinates"):
        moments.add({"first": torch.zeros(2), "second": torch.ones(3)})
    assert moments.count == 1
    assert torch.equal(moments.means["first"], torch.ones(2))
    with pytest.raises(ValueError, match="variant"):
        moments.add({"other": torch.ones(2)})


@pytest.mark.parametrize("values,rewards", [([], []), ([1.], [1.]), ([float('nan')], [[0.]])])
def test_invalid_critic_samples_are_rejected(values, rewards):
    with pytest.raises(ValueError):
        critic_episode_diagnostics(values, rewards)
