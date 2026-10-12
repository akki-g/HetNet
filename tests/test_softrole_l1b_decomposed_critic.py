"""L1b: decomposed critic v_j with V = sum_j v_j and per-agent lambda-return targets."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from softrole.config import Config
from softrole.env import make_env
from softrole.evaluate import evaluate_checkpoint
from softrole.learning import Episode, agent_gae, episode_loss_terms, team_gae
from softrole.model import SoftRoleNet
from softrole.rollout import run_episode
from softrole.scenarios import make_scenarios
from softrole.train import train

DTYPE = torch.float64
HAND_REWARDS = torch.tensor([[1., 0.], [0., 2.], [3., 1.]], dtype=DTYPE)
HAND_VALUES = torch.tensor([[1., 1.], [2., 0.], [0., 1.]], dtype=DTYPE)
# gamma = 1, lambda = .5, zero terminal bootstrap, worked by hand per agent.
HAND_ADVANTAGES = torch.tensor([[1.75, .5], [-.5, 3.], [3., 0.]], dtype=DTYPE)
HAND_TARGETS = torch.tensor([[2.75, 1.5], [1.5, 3.], [3., 1.]], dtype=DTYPE)


def learner(**changes):
    values = dict(gamma=1., gae_lambda=.95, actor_coeff=50., value_coeff=1., critic_mode="decomposed")
    values.update(changes)
    return SimpleNamespace(**values)


def decomposed_config(**overrides):
    settings = dict(model="shared", optimizer="adam", communication="real", comm_rounds=3,
                    independent_heads=True, heads=2, head_dim=4, msg_dim=4, pre_dim=8, hidden_dim=8,
                    critic_mode="decomposed", max_steps=4, vision=1, epochs=1, updates_per_epoch=2,
                    batch_steps=1, nprocesses=1, save_every=1)
    settings.update(overrides)
    return Config(**settings)


def test_agent_gae_matches_hand_computed_values():
    advantages, targets = agent_gae(HAND_REWARDS, HAND_VALUES, gamma=1., gae_lambda=.5)
    torch.testing.assert_close(advantages, HAND_ADVANTAGES, rtol=0, atol=1e-15)
    torch.testing.assert_close(targets, HAND_TARGETS, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="T,N"):
        agent_gae(HAND_REWARDS[:, 0], HAND_VALUES[:, 0])


def test_summing_per_agent_gae_over_agents_gives_the_team_gae():
    generator = torch.Generator().manual_seed(3)
    rewards = torch.randn(7, 4, dtype=DTYPE, generator=generator)
    values = torch.randn(7, 4, dtype=DTYPE, generator=generator)
    advantages, targets = agent_gae(rewards, values, 1., .95)
    team_advantages, team_targets = team_gae(rewards.sum(-1), values.sum(-1), 1., .95)
    torch.testing.assert_close(advantages.sum(-1), team_advantages, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(targets.sum(-1), team_targets, rtol=1e-13, atol=1e-13)


def make_model(mode="shared", **kwargs):
    torch.manual_seed(19)
    return SoftRoleNet(base=2, n_squares=3, mode=mode, experts=3, pre_dim=8, hidden_dim=8, heads=2,
                       head_dim=3, msg_dim=5, communication="real", critic_mode="decomposed", **kwargs).double()


def model_inputs(model, n=3):
    generator = torch.Generator().manual_seed(73)
    obs = torch.randn(n, model.obs_dim, generator=generator, dtype=DTYPE)
    kappa = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=DTYPE)[torch.arange(n) % 3]
    obs[:, -2:] = kappa
    return obs, kappa, model.initial_memory(n), ~torch.eye(n, dtype=torch.bool)


@pytest.mark.parametrize("mode", ["shared", "banked"])
def test_agent_values_sum_to_the_team_value_and_are_permutation_equivariant(mode):
    model = make_model(mode)
    obs, kappa, memory, adjacency = model_inputs(model)
    memory["a_prev"] = torch.tensor([0, 4, 5])
    out = model(obs, kappa, memory, adjacency, .6)
    assert out["agent_values"].shape == (3,)
    assert torch.equal(out["value"], out["agent_values"].sum())
    permutation = torch.tensor([2, 0, 1])
    permuted = model(obs[permutation], kappa[permutation],
                     {k: v[permutation] for k, v in memory.items()},
                     adjacency[permutation][:, permutation], .6)
    torch.testing.assert_close(permuted["agent_values"], out["agent_values"][permutation])
    torch.testing.assert_close(permuted["value"], out["value"])
    torch.testing.assert_close(permuted["logits"], out["logits"][permutation])


def test_critic_parameters_cannot_change_the_actor():
    model = make_model()
    obs, kappa, memory, adjacency = model_inputs(model)
    original = model(obs, kappa, memory, adjacency, .6)
    with torch.no_grad():
        for parameter in list(model.critic_agent.parameters()) + list(model.critic.parameters()):
            parameter.add_(3.)
    changed = model(obs, kappa, model.initial_memory(3), adjacency, .1)
    assert torch.equal(original["logits"], changed["logits"])
    assert not torch.equal(original["agent_values"], changed["agent_values"])
    actor_gradient = torch.autograd.grad(changed["logits"][:, 0].sum(), tuple(model.critic.parameters()),
                                        allow_unused=True)
    assert all(g is None for g in actor_gradient)
    value_gradient = torch.autograd.grad(changed["value"], tuple(model.output.parameters()), allow_unused=True)
    assert all(g is None for g in value_gradient)


def test_rollout_records_per_agent_quantities_without_extra_random_draws():
    config = decomposed_config()
    model = SoftRoleNet(**config.model_kwargs()).double()
    scenario = make_scenarios(81, 1, ((2, 1),))[0]
    episode = run_episode(model, make_env(config), config, scenario, training=True)
    steps = episode.metrics["steps"]
    assert len(episode.agent_values) == len(episode.agent_log_probs) == len(episode.agent_rewards) == steps
    assert all(v.shape == (3,) and v.requires_grad for v in episode.agent_values)
    assert torch.equal(torch.stack(episode.values), torch.stack(episode.agent_values).sum(-1))
    assert torch.equal(torch.stack(episode.agent_log_probs).sum(-1), torch.stack(episode.log_probs))
    assert [sum(r) for r in episode.agent_rewards] == pytest.approx(episode.rewards)
    team = SoftRoleNet(**replace(config, critic_mode="team").model_kwargs()).double()
    team_episode = run_episode(team, make_env(config), replace(config, critic_mode="team"), scenario, training=True)
    assert team_episode.agent_values == [] and len(team_episode.agent_log_probs) == team_episode.metrics["steps"]


def test_beta_one_policy_gradient_equals_the_team_gae_gradient_with_summed_values():
    config = decomposed_config(max_steps=5)
    model = SoftRoleNet(**config.model_kwargs()).double()
    scenario = make_scenarios(81, 1, ((2, 1),))[0]
    episode = run_episode(model, make_env(config), config, scenario, training=True)
    terms = episode_loss_terms(episode, config)
    values, log_probs = torch.stack(episode.values), torch.stack(episode.log_probs)
    advantages, _ = team_gae(episode.rewards, values, config.gamma, config.gae_lambda)
    team_policy = -(log_probs * advantages).sum()
    torch.testing.assert_close(terms["policy"], team_policy, rtol=1e-12, atol=1e-12)
    # At beta = 1 every agent carries the team advantage as its weight.
    torch.testing.assert_close(terms["actor_weights"], advantages.unsqueeze(-1).expand(-1, 3), rtol=1e-12, atol=1e-12)
    parameters = [p for p in model.parameters() if p.requires_grad]
    left = torch.autograd.grad(terms["policy"], parameters, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(team_policy, parameters, allow_unused=True)
    for a, b in zip(left, right):
        assert (a is None and b is None) or torch.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_per_agent_targets_supervise_a_split_that_the_team_loss_cannot_see():
    a = torch.tensor(.7, dtype=DTYPE, requires_grad=True)
    agent_values = torch.stack((a, -a))
    episode = Episode(log_probs=[torch.zeros((), dtype=DTYPE)], values=[agent_values.sum()],
                      rewards=[0.], agent_log_probs=[torch.zeros(2, dtype=DTYPE)],
                      agent_rewards=[[0., 0.]], agent_values=[agent_values])
    team = episode_loss_terms(episode, learner(critic_mode="team"))
    assert team["value"] == 0
    decomposed = episode_loss_terms(episode, learner())
    # delta_j = r_j - v_j, so each target is 0 and the loss is 2 a^2 with gradient 4 a.
    torch.testing.assert_close(decomposed["value"], torch.tensor(2 * .7 ** 2, dtype=DTYPE))
    torch.testing.assert_close(torch.autograd.grad(decomposed["value"], a)[0], torch.tensor(2.8, dtype=DTYPE))


def test_value_loss_is_the_time_mean_of_agent_summed_squared_target_errors():
    values = HAND_VALUES.clone().requires_grad_(True)
    episode = Episode(log_probs=[torch.zeros((), dtype=DTYPE)] * 3, values=list(values.sum(-1)),
                      rewards=HAND_REWARDS.sum(-1).tolist(), agent_log_probs=[torch.zeros(2, dtype=DTYPE)] * 3,
                      agent_rewards=HAND_REWARDS.tolist(), agent_values=list(values))
    terms = episode_loss_terms(episode, learner(gae_lambda=.5))
    expected = (HAND_VALUES - HAND_TARGETS).square().sum(-1).mean()
    torch.testing.assert_close(terms["value"], expected, rtol=0, atol=1e-14)
    torch.testing.assert_close(terms["targets"], HAND_TARGETS, rtol=0, atol=1e-14)
    assert not terms["targets"].requires_grad
    with pytest.raises(ValueError, match="per-agent"):
        episode_loss_terms(Episode(log_probs=[torch.zeros(())], values=[torch.zeros(())], rewards=[0.]), learner())


def test_checkpoint_layout_differs_only_in_the_critic_head():
    config = decomposed_config()
    decomposed = SoftRoleNet(**config.model_kwargs()).double().state_dict()
    team = SoftRoleNet(**replace(config, critic_mode="team").model_kwargs()).double().state_dict()
    assert list(decomposed) == list(team)
    differing = {name for name in team if team[name].shape != decomposed[name].shape}
    assert differing == {"critic.0.weight"}
    assert decomposed["critic.0.weight"].shape == (64, 8 + 2 + 64 + 4) and team["critic.0.weight"].shape == (64, 68)
    assert config.model_kwargs()["critic_mode"] == "decomposed" and "critic_mode" not in replace(config, critic_mode="team").model_kwargs()


def test_training_evaluation_and_resume_guard(tmp_path):
    config = decomposed_config()
    checkpoint = train(config, tmp_path / "run")
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["learner"]["critic_mode"] == "decomposed"
    rows = [json.loads(line) for line in (tmp_path / "run" / "updates.jsonl").open()]
    assert len(rows) == 2 and all(np.isfinite(r["gradient_norm_before_clip"]) and r["gradient_norm_before_clip"] > 0 for r in rows)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["model_config"]["critic_mode"] == "decomposed"
    scenarios = make_scenarios(92, 2, ((2, 1),))
    first = evaluate_checkpoint(checkpoint, scenarios, tmp_path / "first.json", trace=True)
    second = evaluate_checkpoint(checkpoint, scenarios, tmp_path / "second.json", trace=True)
    assert first["per_episode"] == second["per_episode"]
    partial = train(replace(config, total_steps=1), tmp_path / "partial")
    with pytest.raises(ValueError, match="resume may change only"):
        train(replace(config, critic_mode="team"), tmp_path / "switched", resume=partial)
    assert not (tmp_path / "switched").exists()
