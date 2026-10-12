"""L1a: actor weights beta sum_i A_i + (1-beta) A_j on the decomposed critic.

The enumerated two-agent example shows exactly what beta < 1 drops: the
cross-agent credit term, which is the entire gradient when one agent's action
rewards only its teammate.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from softrole.config import Config
from softrole.learning import Episode, compute_episode_loss, episode_loss_terms
from softrole.train import train

DTYPE = torch.float64


def learner(beta, **changes):
    values = dict(gamma=1., gae_lambda=1., actor_coeff=1., value_coeff=0., critic_mode="decomposed",
                  credit_beta=beta, objective="team")
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("beta", [1., .5, 0.])
def test_enumerated_cross_agent_credit_scales_with_beta(beta):
    # Agent 1's action rewards only agent 0 (agent reward vector [a1, 0]).
    theta = torch.tensor([-0.4, 0.7], dtype=DTYPE, requires_grad=True)
    estimate = torch.zeros_like(theta)
    for a0 in (0., 1.):
        for a1 in (0., 1.):
            actions = torch.tensor([a0, a1], dtype=DTYPE)
            dist = torch.distributions.Bernoulli(logits=theta)
            scores = dist.log_prob(actions)
            weight = scores.sum().exp().detach()
            episode = Episode(log_probs=[scores.sum()], values=[theta.sum() * 0], rewards=[a1],
                              agent_log_probs=[scores], agent_rewards=[[a1, 0.]],
                              agent_values=[theta * 0])
            loss, _ = compute_episode_loss(episode, learner(beta))
            estimate -= weight * torch.autograd.grad(loss, theta)[0]
    exact = torch.autograd.grad(theta.sigmoid()[1], theta)[0]
    assert exact[0] == 0 and exact[1] > 0
    torch.testing.assert_close(estimate, beta * exact, rtol=0, atol=1e-12)


def test_mixed_weights_on_a_hand_example_and_own_credit_at_beta_zero():
    rewards = torch.tensor([[1., 0.], [0., 2.], [3., 1.]], dtype=DTYPE)
    values = torch.tensor([[1., 1.], [2., 0.], [0., 1.]], dtype=DTYPE)
    log_probs = torch.zeros(3, 2, dtype=DTYPE, requires_grad=True)
    episode = Episode(log_probs=list(log_probs.sum(-1)), values=list(values.sum(-1)),
                      rewards=rewards.sum(-1).tolist(), agent_log_probs=list(log_probs),
                      agent_rewards=rewards.tolist(), agent_values=list(values))
    advantages = torch.tensor([[1.75, .5], [-.5, 3.], [3., 0.]], dtype=DTYPE)  # lambda = .5 by hand
    half = episode_loss_terms(episode, learner(.5, gae_lambda=.5))
    expected = .5 * advantages.sum(-1, keepdim=True) + .5 * advantages
    torch.testing.assert_close(half["actor_weights"], expected, rtol=0, atol=1e-14)
    assert half["actor_weights"][0].tolist() == pytest.approx([2.0, 1.375])
    own = episode_loss_terms(episode, learner(0., gae_lambda=.5))
    torch.testing.assert_close(torch.autograd.grad(own["policy"], log_probs)[0], -advantages, rtol=0, atol=1e-14)


@pytest.mark.parametrize("beta", [-.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError, match="credit_beta"):
        Config(critic_mode="decomposed", credit_beta=beta)


def test_beta_below_one_requires_the_decomposed_critic(tmp_path, capsys):
    from softrole.__main__ import main
    with pytest.raises(ValueError, match="requires critic_mode=decomposed"):
        Config(credit_beta=.5)
    assert Config(critic_mode="decomposed", credit_beta=0).credit_beta == 0
    with pytest.raises(SystemExit) as error:
        main(["train", "--output", str(tmp_path / "dry"), "--dry-run", "--task", "pcp", "--credit-beta", "0.5"])
    assert error.value.code == 2
    main(["train", "--output", str(tmp_path / "dry"), "--dry-run", "--task", "pcp",
          "--credit-beta", "0.5", "--critic-mode", "decomposed"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["credit_beta"] == .5 and printed["critic_mode"] == "decomposed"
    assert "credit_beta" not in Config(critic_mode="decomposed", credit_beta=.5).model_kwargs()
    assert not (tmp_path / "dry").exists()


def tiny(**overrides):
    settings = dict(model="shared", optimizer="adam", communication="real", comm_rounds=3,
                    independent_heads=True, heads=2, head_dim=4, msg_dim=4, pre_dim=8, hidden_dim=8,
                    critic_mode="decomposed", credit_beta=.5, max_steps=3, vision=1, epochs=1,
                    updates_per_epoch=2, batch_steps=1, nprocesses=1, save_every=1)
    settings.update(overrides)
    return Config(**settings)


def test_run_record_and_resume_guard(tmp_path):
    train(tiny(), tmp_path / "half")
    record = json.loads((tmp_path / "half" / "run.json").read_text())
    assert record["learner"]["credit_beta"] == .5
    assert record["approximations"] == ["credit mixing: beta<1 drops cross-agent credit",
                                        "GAE with learned critic", "truncated BPTT"]
    train(tiny(credit_beta=1.), tmp_path / "full")
    assert json.loads((tmp_path / "full" / "run.json").read_text())["approximations"] == [
        "GAE with learned critic", "truncated BPTT"]
    partial = train(tiny(total_steps=1), tmp_path / "partial")
    with pytest.raises(ValueError, match="resume may change only"):
        train(tiny(credit_beta=1.), tmp_path / "switched", resume=partial)
    assert not (tmp_path / "switched").exists()
