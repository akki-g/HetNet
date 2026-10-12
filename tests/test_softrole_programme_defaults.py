"""Every programme default reproduces the pre-programme model and learner exactly.

The fixture is a byte copy of ``softrole/model.py`` at commit 644f8f1, the
source executed by the running S1 and D1--D6 jobs. Both modules run on the
same machine within one process, so the comparison does not depend on the
platform that captured a numerical snapshot.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from softrole.config import Config
from softrole.learning import Episode, compute_episode_loss, episode_loss_terms, team_gae
from softrole.model import SoftRoleNet

FIXTURE = Path(__file__).parent / "fixtures" / "softrole_model_pre_programme.py"
FIXTURE_SHA256 = "8cb989455cdba6c61f43f3abd8f8c1ba6f37ef784c14aed2a2c957a172e3fa94"
DIMENSIONS = dict(base=2, n_squares=3, experts=3, pre_dim=8, hidden_dim=8, heads=2, head_dim=3, msg_dim=5)
CASES = [
    dict(),
    dict(mode="shared"),
    dict(mode="capability"),
    dict(mode="constant"),
    dict(feedback=False),
    dict(mode="shared", comm_rounds=3, independent_heads=True, communication="real"),
    dict(comm_rounds=3, independent_heads=True, communication="binary"),
]


def load_fixture():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    spec = importlib.util.spec_from_file_location("softrole_model_pre_programme", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(cls, kwargs):
    torch.manual_seed(19)
    return cls(**DIMENSIONS, **kwargs).double()


def inputs(model, kwargs, n=3):
    generator = torch.Generator().manual_seed(73)
    obs = torch.randn(n, model.obs_dim, generator=generator, dtype=torch.float64)
    kappa = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)[torch.arange(n) % 3]
    obs[:, -2:] = kappa
    adjacency = ~torch.eye(n, dtype=torch.bool)
    rounds = kwargs.get("comm_rounds", 2)
    shape = (n, DIMENSIONS["heads"], DIMENSIONS["msg_dim"]) if kwargs.get("independent_heads") else (n, DIMENSIONS["msg_dim"])
    noise = ([torch.randn(*shape, generator=generator, dtype=torch.float64) for _ in range(rounds)]
             if kwargs.get("communication", "binary") == "binary" else None)
    return obs, kappa, adjacency, noise


@pytest.mark.parametrize("kwargs", CASES, ids=[",".join(f"{k}={v}" for k, v in c.items()) or "defaults" for c in CASES])
def test_pre_programme_model_is_reproduced_bitwise(kwargs):
    old = build(load_fixture().SoftRoleNet, kwargs)
    new = build(SoftRoleNet, kwargs)
    assert list(old.state_dict()) == list(new.state_dict())
    for name, tensor in old.state_dict().items():
        assert torch.equal(tensor, new.state_dict()[name]), name
    obs, kappa, adjacency, noise = inputs(new, kwargs)
    memory_old, memory_new = old.initial_memory(3), new.initial_memory(3)
    memory_old["a_prev"] = memory_new["a_prev"] = torch.tensor([0, 4, 5])
    out_old = old(obs, kappa, memory_old, adjacency, 0.7, noise)
    out_new = new(obs, kappa, memory_new, adjacency, 0.7, noise)
    assert out_new["agent_values"] is None
    for name in ("logits", "value", "gate"):
        assert torch.equal(out_old[name], out_new[name]), name
    for name in memory_new:
        assert torch.equal(out_old["memory"][name], out_new["memory"][name]), name
    for name in ("bits", "messages", "alpha_null"):
        assert len(out_old[name]) == len(out_new[name])
        assert all(torch.equal(a, b) for a, b in zip(out_old[name], out_new[name])), name
    objective_old = out_old["logits"][:, :5].sum() + out_old["value"]
    objective_new = out_new["logits"][:, :5].sum() + out_new["value"]
    grads_old = torch.autograd.grad(objective_old, list(old.parameters()), allow_unused=True)
    grads_new = torch.autograd.grad(objective_new, list(new.parameters()), allow_unused=True)
    for (name, _), a, b in zip(old.named_parameters(), grads_old, grads_new):
        assert (a is None and b is None) or torch.equal(a, b), name


def test_default_configuration_keeps_the_pre_programme_layout_and_values():
    config = Config()
    assert config.communication == "binary" and config.null_degree_correction is False
    assert config.critic_mode == "team" and config.credit_beta == 1.0
    assert config.objective == "team" and config.actor_capabilities == "explicit"
    assert set(config.model_kwargs()) == {"base", "n_squares", "mode", "experts", "pre_dim",
                                          "hidden_dim", "heads", "head_dim", "msg_dim", "feedback"}
    legacy = config.to_dict()
    for field in ("null_degree_correction", "critic_mode", "credit_beta", "objective", "actor_capabilities"):
        legacy.pop(field)
    assert Config(**legacy) == config


def test_team_loss_builder_reproduces_the_pre_programme_arithmetic_bitwise():
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(4, dtype=torch.float64, generator=generator).requires_grad_(True)
    log_probs = torch.randn(4, dtype=torch.float64, generator=generator).requires_grad_(True)
    rewards = [-.15, -.1, -.05, 0.]
    config = SimpleNamespace(gamma=1., gae_lambda=.95, actor_coeff=50., value_coeff=1.)
    episode = Episode(log_probs=list(log_probs), values=list(values), rewards=rewards)
    advantages, targets = team_gae(rewards, values, 1., .95)
    expected = 50. * -(log_probs * advantages).sum() + 1. * (values - targets).square().mean()
    total, stats = compute_episode_loss(episode, config)
    assert torch.equal(total, expected)
    assert stats["episode_count"] == 1
    terms = episode_loss_terms(episode, config)
    assert terms["scale"] == 1.0 and torch.equal(terms["actor_weights"], advantages)
    assert torch.equal(torch.autograd.grad(total, (values, log_probs))[0],
                       torch.autograd.grad(expected, (values, log_probs))[0])
