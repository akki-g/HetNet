"""L1c: mean-agent objective, each complete episode loss divided by its roster size.

The scale enters the episode loss before collection; the learner still averages
once by the global episode count and clips once.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from softrole.config import Config, recipe
from softrole.env import make_env
from softrole.learning import Episode, compute_episode_loss, episode_loss_terms, loss_sum
from softrole.model import SoftRoleNet
from softrole.rollout import run_episode
from softrole.scenarios import make_scenarios
from softrole.train import collect_batch, train


def learner(objective, **changes):
    values = dict(gamma=1., gae_lambda=1., actor_coeff=1., value_coeff=0., critic_mode="team",
                  credit_beta=1., objective=objective)
    values.update(changes)
    return SimpleNamespace(**values)


def synthetic_episodes(theta):
    zero = theta * 0
    single = Episode(log_probs=[theta], values=[zero], rewards=[2.], agent_rewards=[[2.]])
    triple = Episode(log_probs=[theta, theta], values=[zero, zero], rewards=[1., 3.],
                     agent_rewards=[[1., 0., 0.], [0., 1., 2.]])
    return [single, triple]


def test_unequal_rosters_are_weighted_by_one_over_n_with_one_global_sum():
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    team_loss, team_stats = loss_sum(synthetic_episodes(theta), learner("team"))
    mean_loss, mean_stats = loss_sum(synthetic_episodes(theta), learner("mean_agent"))
    # lambda = 1 advantages: 2 for the singleton, (4, 3) for the trio.
    torch.testing.assert_close(torch.autograd.grad(team_loss, theta)[0], theta.new_tensor(-9.))
    torch.testing.assert_close(torch.autograd.grad(mean_loss, theta)[0], theta.new_tensor(-2. - 7. / 3.))
    # Recorded sums stay unscaled so ledgers remain comparable across objectives.
    assert team_stats == mean_stats and mean_stats["policy_loss_sum"] == pytest.approx(-1.8)
    assert episode_loss_terms(synthetic_episodes(theta)[1], learner("mean_agent"))["scale"] == pytest.approx(1 / 3)


def test_mean_agent_scale_needs_the_roster_size():
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    legacy = Episode(log_probs=[theta], values=[theta * 0], rewards=[2.], metrics={"num_agents": 4})
    assert episode_loss_terms(legacy, learner("mean_agent"))["scale"] == .25
    with pytest.raises(ValueError, match="roster size"):
        episode_loss_terms(Episode(log_probs=[theta], values=[theta * 0], rewards=[2.]), learner("mean_agent"))
    with pytest.raises(ValueError, match="Unknown objective"):
        episode_loss_terms(legacy, learner("per_agent"))


def composition_config(objective):
    return recipe("pcp", "composition", model="shared", optimizer="adam", objective=objective,
                  pre_dim=8, hidden_dim=8, heads=1, head_dim=4, msg_dim=4, vision=1, max_steps=3,
                  epochs=1, updates_per_epoch=1, batch_steps=6, nprocesses=1, save_every=1, seed=5)


def test_collector_gradient_sums_scale_each_episode_before_the_global_average():
    torch.manual_seed(1)
    mean_config, team_config = composition_config("mean_agent"), composition_config("team")
    model = SoftRoleNet(**mean_config.model_kwargs()).double()
    batch = collect_batch(model, mean_config, 0, 0)
    assert batch["num_episodes"] > 1 and len({e["num_agents"] for e in batch["episodes"]}) > 1
    seed = int(np.random.SeedSequence([mean_config.seed, 0, 0, 2026]).generate_state(1)[0])
    scenarios = make_scenarios(seed, mean_config.batch_steps, mean_config.training_compositions)
    parameters = list(model.parameters())
    expected = [torch.zeros_like(p) for p in parameters]
    steps = 0
    for scenario, record in zip(scenarios, batch["episodes"]):
        episode = run_episode(model, make_env(team_config), team_config, scenario, training=True)
        assert episode.metrics == record
        loss, _ = compute_episode_loss(episode, team_config)
        for total, gradient in zip(expected, torch.autograd.grad(loss / episode.metrics["num_agents"], parameters, allow_unused=True)):
            if gradient is not None:
                total.add_(gradient)
        steps += episode.metrics["steps"]
        if steps >= mean_config.batch_steps:
            break
    for total, actual in zip(expected, batch["gradients"]):
        torch.testing.assert_close(total, torch.as_tensor(actual), rtol=1e-12, atol=1e-12)


def test_cli_run_record_and_resume_guard(tmp_path, capsys):
    from softrole.__main__ import main
    main(["train", "--output", str(tmp_path / "dry"), "--dry-run", "--task", "pcp", "--study", "composition",
          "--objective", "mean_agent"])
    assert json.loads(capsys.readouterr().out)["objective"] == "mean_agent"
    with pytest.raises(ValueError, match="objective"):
        Config(objective="per_agent")
    config = composition_config("mean_agent")
    train(config, tmp_path / "run")
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["actor_objective"].endswith("complete episode loss divided by roster size")
    assert record["learner"]["objective"] == "mean_agent"
    assert "objective" not in config.model_kwargs()
    metrics = json.loads((tmp_path / "run" / "metrics.jsonl").read_text())
    assert np.isfinite(metrics["policy_loss_per_episode"]) and metrics["episodes"] >= 1
    partial = train(replace(config, total_steps=1), tmp_path / "partial")
    with pytest.raises(ValueError, match="resume may change only"):
        train(replace(config, objective="team"), tmp_path / "switched", resume=partial)
    assert not (tmp_path / "switched").exists()
