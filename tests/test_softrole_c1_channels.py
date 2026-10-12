"""C1: real_noisy and binary_deterministic channels.

Forward values and surrogate derivatives are checked against injected noise
and analytic expressions; keyed randomness, payload accounting, provenance
metadata, resume and frozen evaluation are exercised for both new channels.
"""
from dataclasses import replace
import json

import pytest
import torch

from softrole.config import CHANNELS, Config
from softrole.env import make_env
from softrole.model import (SoftRoleNet, channel_payload, deterministic_bits,
                            logistic_noise_like, straight_through_bits)
from softrole.rollout import message_noise, run_episode
from softrole.scenarios import make_scenarios
from softrole.train import train


def channel_config(**overrides):
    settings = dict(model="shared", optimizer="adam", comm_rounds=3,
                    independent_heads=True, heads=4, msg_dim=64,
                    pre_dim=8, hidden_dim=8, head_dim=4, max_steps=3,
                    vision=1, epochs=1, updates_per_epoch=2, batch_steps=1,
                    nprocesses=1, save_every=1)
    settings.update(overrides)
    return Config(**settings)


def test_real_noisy_transmits_u_plus_noise_with_unit_pathwise_derivative():
    generator = torch.Generator().manual_seed(5)
    u = torch.randn(3, 2, 5, dtype=torch.float64, generator=generator).requires_grad_(True)
    noise = torch.randn(3, 2, 5, dtype=torch.float64, generator=generator)
    payload = channel_payload(u, "real_noisy", noise)
    assert torch.equal(payload.detach(), u.detach() + noise)
    weights = torch.randn(3, 2, 5, dtype=torch.float64, generator=generator)
    (payload * weights).sum().backward()
    # d(u + eps)/du = 1 exactly, coordinate by coordinate.
    assert torch.equal(u.grad, weights)
    fresh = channel_payload(u.detach(), "real_noisy", None)
    assert fresh.shape == u.shape and not torch.equal(fresh, u.detach())
    with pytest.raises(ValueError, match="same shape"):
        channel_payload(u, "real_noisy", noise[:1])


def test_binary_deterministic_thresholds_strictly_and_uses_sigmoid_prime():
    u = torch.linspace(-4, 4, 9, dtype=torch.float64).requires_grad_(True)
    bits = channel_payload(u, "binary_deterministic", None)
    assert torch.equal(bits.detach(), (u.detach() > 0).to(torch.float64))
    assert bits.detach()[4] == 0  # u = 0 sends a zero bit: the threshold is strict
    weights = torch.arange(1, 10, dtype=torch.float64)
    (bits * weights).sum().backward()
    sigma = torch.sigmoid(u.detach())
    torch.testing.assert_close(u.grad, weights * sigma * (1 - sigma), rtol=1e-14, atol=1e-16)
    with pytest.raises(ValueError, match="does not accept noise"):
        channel_payload(u, "binary_deterministic", torch.zeros_like(u))
    assert torch.equal(deterministic_bits(u).detach(), bits.detach())


def test_stochastic_binary_surrogate_is_byte_for_byte_the_previous_formula():
    generator = torch.Generator().manual_seed(11)
    u = torch.randn(4, 6, dtype=torch.float64, generator=generator).requires_grad_(True)
    noise = torch.randn(4, 6, dtype=torch.float64, generator=generator)
    payload = channel_payload(u, "binary", noise)
    relaxed = torch.sigmoid(u + noise)
    expected = ((u + noise) > 0).to(u.dtype) + (relaxed - relaxed.detach())
    assert torch.equal(payload.detach(), expected.detach())
    first = torch.autograd.grad(payload.sum(), u)[0]
    second = torch.autograd.grad(expected.sum(), u)[0]
    assert torch.equal(first, second)
    assert torch.equal(straight_through_bits(u, noise).detach(), expected.detach())


def test_fresh_logistic_noise_has_logistic_marginals():
    with torch.random.fork_rng():
        torch.manual_seed(3)
        draws = logistic_noise_like(torch.zeros(200000, dtype=torch.float64))
    # Logistic(0,1): median 0, variance pi^2/3.
    assert abs(draws.median().item()) < 0.02
    assert abs(draws.var().item() - torch.pi ** 2 / 3) < 0.1


@pytest.mark.parametrize("communication", CHANNELS)
def test_model_accepts_noise_only_for_stochastic_channels(communication):
    model = SoftRoleNet(base=2, n_squares=3, mode="shared", pre_dim=8, hidden_dim=8,
                        heads=2, head_dim=3, msg_dim=5, comm_rounds=2, independent_heads=True,
                        communication=communication).double()
    generator = torch.Generator().manual_seed(7)
    obs = torch.randn(3, model.obs_dim, generator=generator, dtype=torch.float64)
    kappa = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    obs[:, -2:] = kappa
    noise = [torch.randn(3, 2, 5, generator=generator, dtype=torch.float64) for _ in range(2)]
    adjacency = ~torch.eye(3, dtype=torch.bool)
    if communication in ("binary", "real_noisy"):
        out = model(obs, kappa, model.initial_memory(3), adjacency, 0.5, noise)
        assert len(out["messages"]) == 2
        assert (out["bits"] == out["messages"]) == (communication == "binary")
    else:
        with pytest.raises(ValueError, match="does not accept bit noise"):
            model(obs, kappa, model.initial_memory(3), adjacency, 0.5, noise)
        out = model(obs, kappa, model.initial_memory(3), adjacency, 0.5, None)
        assert bool(out["bits"]) == (communication == "binary_deterministic")
    if communication in ("binary", "binary_deterministic"):
        assert all(torch.equal(m, m.round()) and ((m == 0) | (m == 1)).all() for m in out["messages"])


@pytest.mark.parametrize("communication", ["real", "binary_deterministic"])
def test_noise_free_channels_never_draw_message_noise(communication, monkeypatch):
    config = channel_config(communication=communication)
    model = SoftRoleNet(**config.model_kwargs()).double()

    def forbid(*args, **kwargs):
        raise AssertionError(f"{communication} rollouts must not draw message noise")
    monkeypatch.setattr("softrole.rollout.message_noise", forbid)
    scenario = make_scenarios(81, 1, ((2, 1),))[0]
    first = run_episode(model, make_env(config), config, scenario, training=False, trace=True)
    second = run_episode(model, make_env(config), config, scenario, training=False, trace=True)
    assert first.metrics == second.metrics and first.traces == second.traces


def test_real_noisy_rollouts_are_keyed_by_the_message_seed():
    config = channel_config(communication="real_noisy")
    model = SoftRoleNet(**config.model_kwargs()).double()
    scenario = make_scenarios(81, 1, ((2, 1),))[0]
    captured = {}

    def run(tag, seed_scenario):
        messages = []
        handle = model.register_forward_hook(lambda m, a, out: messages.append([x.detach().clone() for x in out["messages"]]))
        try:
            episode = run_episode(model, make_env(config), config, seed_scenario, training=False, trace=True)
        finally:
            handle.remove()
        captured[tag] = (episode.metrics, episode.traces, messages)

    run("a", scenario)
    run("b", scenario)
    run("c", replace(scenario, message_seed=scenario.message_seed + 1))
    assert captured["a"][0] == captured["b"][0] and captured["a"][1] == captured["b"][1]
    assert all(torch.equal(x, y) for sa, sb in zip(captured["a"][2], captured["b"][2]) for x, y in zip(sa, sb))
    # A different message stream changes the transmitted values at the first step.
    assert not all(torch.equal(x, y) for x, y in zip(captured["a"][2][0], captured["c"][2][0]))
    draws = message_noise(scenario.message_seed, 0, 3, config.msg_dim, torch.float64,
                          comm_rounds=3, heads=config.heads)
    assert len(draws) == 3 and all(d.shape == (3, 4, 64) for d in draws)


@pytest.mark.parametrize("communication,value_bits", [("real_noisy", 64), ("binary_deterministic", 1)])
def test_payload_accounting_for_new_channels(communication, value_bits):
    config = channel_config(communication=communication)
    model = SoftRoleNet(**config.model_kwargs()).double()
    scenario = make_scenarios(81, 1, ((2, 1),))[0]
    episode = run_episode(model, make_env(config), config, scenario, training=False)
    assert episode.metrics["communication"] == communication
    assert episode.metrics["payload_values_per_agent_step"] == 3 * 4 * 64
    assert episode.metrics["payload_value_bits"] == value_bits
    assert episode.metrics["payload_bits_per_agent_step"] == 768 * value_bits
    assert episode.metrics["payload_bits_generated"] == episode.metrics["steps"] * 3 * 768 * value_bits


def test_configuration_and_cli_accept_the_four_channels(tmp_path, capsys):
    from softrole.__main__ import main
    for channel in CHANNELS:
        assert Config(communication=channel).communication == channel
    with pytest.raises(ValueError, match="communication must be one of"):
        Config(communication="ternary")
    main(["train", "--output", str(tmp_path / "dry"), "--dry-run", "--task", "pcp",
          "--communication", "real_noisy", "--comm-rounds", "3", "--independent-heads"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["communication"] == "real_noisy" and printed["comm_rounds"] == 3
    assert not (tmp_path / "dry").exists()


@pytest.mark.parametrize("communication", ["real_noisy", "binary_deterministic"])
def test_new_channels_resume_exactly_and_evaluate_frozen(tmp_path, communication):
    from softrole.evaluate import evaluate_checkpoint
    config = channel_config(communication=communication)
    full_path = train(config, tmp_path / "full")
    partial_path = train(replace(config, total_steps=1), tmp_path / "partial")
    resumed_path = train(config, tmp_path / "resumed", resume=partial_path)
    full = torch.load(full_path, weights_only=False)
    resumed = torch.load(resumed_path, weights_only=False)
    for key in ("updates", "total_steps", "total_episodes"):
        assert full[key] == resumed[key]
    for name, tensor in full["model_state"].items():
        assert torch.equal(tensor, resumed["model_state"][name])
    for identifier, state in full["optimizer_state"]["state"].items():
        for key, value in state.items():
            other = resumed["optimizer_state"]["state"][identifier][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    metadata = json.loads((tmp_path / "full" / "run.json").read_text())
    expected = {"real_noisy": [], "binary_deterministic": ["deterministic-threshold surrogate bits"]}[communication]
    assert metadata["approximations"] == expected + ["GAE with learned critic", "truncated BPTT"]
    assert metadata["learner"]["communication"] == communication
    assert full["model_config"]["communication"] == communication
    scenarios = make_scenarios(92, 2, ((2, 1),))
    first = evaluate_checkpoint(full_path, scenarios, tmp_path / "first.json", trace=True)
    second = evaluate_checkpoint(full_path, scenarios, tmp_path / "second.json", trace=True)
    assert first["per_episode"] == second["per_episode"]
    with pytest.raises(ValueError, match="resume may change only"):
        train(replace(config, communication="real"), tmp_path / "switched", resume=partial_path)
