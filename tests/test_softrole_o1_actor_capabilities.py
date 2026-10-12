"""O1: the actor sees only physical interfaces; kappa stays in masks and the critic."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from softrole.config import Config, recipe
from softrole.env import adapt_observation, make_env
from softrole.evaluate import evaluate_checkpoint
from softrole.model import SoftRoleNet
from softrole.rollout import run_episode
from softrole.scenarios import make_scenarios
from softrole.train import train

DTYPE = torch.float64


def make_model(actor_capabilities, mode="shared", seed=19):
    torch.manual_seed(seed)
    return SoftRoleNet(base=2, n_squares=3, mode=mode, experts=3, pre_dim=8, hidden_dim=8, heads=2,
                       head_dim=3, msg_dim=5, actor_capabilities=actor_capabilities).double()


def paired_inputs(model, n=3):
    generator = torch.Generator().manual_seed(73)
    obs = torch.randn(n, model.obs_dim, generator=generator, dtype=DTYPE)
    kappa = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=DTYPE)[torch.arange(n) % 3]
    other = torch.tensor([[0., 1.], [1., 0.], [0., 0.]], dtype=DTYPE)[torch.arange(n) % 3]
    obs[:, -2:] = kappa
    swapped = obs.clone()
    swapped[:, -2:] = other
    noise = [torch.randn(n, model.msg_dim, generator=generator, dtype=DTYPE) for _ in range(2)]
    adjacency = ~torch.eye(n, dtype=torch.bool)
    return (obs, kappa), (swapped, other), adjacency, noise


def test_actor_features_and_messages_ignore_kappa_while_mask_and_critic_keep_it():
    model = make_model("physical_only")
    (obs, kappa), (swapped, other), adjacency, noise = paired_inputs(model)
    left = model(obs, kappa, model.initial_memory(3), adjacency, .5, noise)
    right = model(swapped, other, model.initial_memory(3), adjacency, .5, noise)
    assert torch.equal(left["logits"][:, :5], right["logits"][:, :5])
    assert all(torch.equal(a, b) for a, b in zip(left["messages"], right["messages"]))
    assert all(torch.equal(a, b) for a, b in zip(left["alpha_null"], right["alpha_null"]))
    for name in ("H", "C", "u_bar"):
        assert torch.equal(left["memory"][name], right["memory"][name])
    assert torch.isinf(left["logits"][0, 5]) and torch.isfinite(left["logits"][1, 5]) and torch.isfinite(left["logits"][2, 5])
    assert torch.isfinite(right["logits"][0, 5]) and torch.isinf(right["logits"][1, 5]) and torch.isinf(right["logits"][2, 5])
    assert not torch.equal(left["value"], right["value"])
    explicit = make_model("explicit")
    left = explicit(obs, kappa, explicit.initial_memory(3), adjacency, .5, noise)
    right = explicit(swapped, other, explicit.initial_memory(3), adjacency, .5, noise)
    assert not torch.equal(left["logits"][:, :5], right["logits"][:, :5])


def test_banked_gate_uses_the_hidden_state_only():
    physical = make_model("physical_only", mode="banked")
    assert physical.gate_network[0].in_features == physical.hidden_dim
    (obs, kappa), (swapped, other), adjacency, noise = paired_inputs(physical)
    left = physical(obs, kappa, physical.initial_memory(3), adjacency, .5, noise)
    right = physical(swapped, other, physical.initial_memory(3), adjacency, .5, noise)
    assert torch.equal(left["gate"], right["gate"])
    explicit = make_model("explicit", mode="banked")
    assert explicit.gate_network[0].in_features == explicit.hidden_dim + 2
    left = explicit(obs, kappa, explicit.initial_memory(3), adjacency, .5, noise)
    right = explicit(swapped, other, explicit.initial_memory(3), adjacency, .5, noise)
    assert not torch.equal(left["gate"], right["gate"])


def test_capability_gated_model_contradicts_physical_only():
    with pytest.raises(ValueError, match="contradict"):
        make_model("physical_only", mode="capability")
    with pytest.raises(ValueError, match="contradict"):
        Config(model="capability", actor_capabilities="physical_only")
    with pytest.raises(ValueError, match="actor_capabilities"):
        Config(actor_capabilities="hidden")


def test_permutation_equivariance_with_physical_only_input():
    model = make_model("physical_only", mode="banked")
    (obs, kappa), _, adjacency, noise = paired_inputs(model)
    memory = model.initial_memory(3)
    memory["a_prev"] = torch.tensor([0, 4, 5])
    permutation = torch.tensor([2, 0, 1])
    original = model(obs, kappa, memory, adjacency, .6, noise)
    permuted = model(obs[permutation], kappa[permutation], {k: v[permutation] for k, v in memory.items()},
                     adjacency[permutation][:, permutation], .6, [x[permutation] for x in noise])
    for name in ("logits", "gate"):
        torch.testing.assert_close(permuted[name], original[name][permutation])
    torch.testing.assert_close(permuted["value"], original["value"])


def test_layout_and_configuration(tmp_path, capsys):
    from softrole.__main__ import main
    physical, explicit = make_model("physical_only").state_dict(), make_model("explicit").state_dict()
    assert list(physical) == list(explicit)
    assert {n for n in physical if physical[n].shape != explicit[n].shape} == {"pre.weight"}
    assert physical["pre.weight"].shape[1] == explicit["pre.weight"].shape[1] - 2
    assert Config(actor_capabilities="physical_only").model_kwargs()["actor_capabilities"] == "physical_only"
    assert "actor_capabilities" not in Config().model_kwargs()
    main(["train", "--output", str(tmp_path / "dry"), "--dry-run", "--task", "pcp",
          "--model", "shared", "--actor-capabilities", "physical_only"])
    assert json.loads(capsys.readouterr().out)["actor_capabilities"] == "physical_only"
    with pytest.raises(SystemExit) as error:
        main(["train", "--output", str(tmp_path / "dry"), "--dry-run", "--task", "pcp",
              "--model", "capability", "--actor-capabilities", "physical_only"])
    assert error.value.code == 2
    assert not (tmp_path / "dry").exists()


def test_a_blind_sensing_agent_and_an_actuator_share_the_physical_actor_input():
    base, cells = 4, 3
    raw = np.zeros((2, cells, base + 4))
    raw[:, :, :base] = np.eye(base)[[0, 1, 2]]          # identical position one-hots
    raw[0, :, base:] = [[1., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]  # a sighted P's cells
    raw[1, :, base:] = -1.                               # an A's native sensory entries
    kappa = np.array([[0., 0.], [0., 1.]])               # P after sensor loss, A
    obs = adapt_observation(raw.reshape(2, -1), kappa, base)
    assert np.array_equal(obs[0, :-2], obs[1, :-2])
    assert not np.array_equal(adapt_observation(raw.reshape(2, -1), np.array([[1., 0.], [0., 1.]]), base)[0, :-2], obs[1, :-2])


def tiny(**overrides):
    settings = dict(model="shared", optimizer="adam", communication="real", comm_rounds=3,
                    independent_heads=True, heads=2, head_dim=4, msg_dim=4, pre_dim=8, hidden_dim=8,
                    actor_capabilities="physical_only", vision=1, max_steps=3, epochs=1,
                    updates_per_epoch=2, batch_steps=1, nprocesses=1, save_every=1, failure_window=(1, 1))
    settings.update(overrides)
    return recipe("pcp", "failure", **settings)


def test_failure_training_evaluation_and_resume_guard(tmp_path):
    config = tiny()
    assert config.failure_prob == .5 and config.failure_window == (1, 1)
    checkpoint = train(config, tmp_path / "run")
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["learner"]["actor_capabilities"] == "physical_only"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["model_config"]["actor_capabilities"] == "physical_only"
    model = SoftRoleNet(**saved["model_config"]).double()
    model.load_state_dict(saved["model_state"])
    scenarios = make_scenarios(17, 3, ((2, 1),), failure_prob=1., failure_window=(1, 1))
    episode = run_episode(model, make_env(config), config, scenarios[0], training=False, event_diagnostics=True)
    assert episode.metrics["event_exposed"] or episode.metrics["steps"] <= 1
    report = evaluate_checkpoint(checkpoint, scenarios, tmp_path / "failure.json")
    sham = evaluate_checkpoint(checkpoint, scenarios, tmp_path / "sham.json", sham=True)
    assert report["scheduled_event_exposed_episodes"] == sham["scheduled_event_exposed_episodes"]
    partial = train(replace(config, total_steps=1), tmp_path / "partial")
    with pytest.raises(ValueError, match="resume may change only"):
        train(replace(config, actor_capabilities="explicit"), tmp_path / "switched", resume=partial)
    assert not (tmp_path / "switched").exists()
