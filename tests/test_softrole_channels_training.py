"""Exercise channel configuration through collection, checkpoints and evaluation."""
from dataclasses import replace
import json

import pytest
import torch

from softrole.config import Config
from softrole.env import make_env
from softrole.evaluate import evaluate_checkpoint
from softrole.model import SoftRoleNet
from softrole.report import summarize_reports
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


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def test_legacy_config_has_original_model_layout_and_rmsprop():
    config = Config()
    assert config.optimizer == "rmsprop"
    assert config.comm_rounds == 2 and not config.independent_heads
    assert config.communication == "binary"
    assert not {"comm_rounds", "independent_heads", "communication"} & config.model_kwargs().keys()
    old = config.to_dict()
    for field in ("optimizer", "comm_rounds", "independent_heads", "communication"):
        old.pop(field)
    assert Config(**old) == config


@pytest.mark.parametrize("override", [
    {"comm_rounds": 0}, {"comm_rounds": True}, {"comm_rounds": 1.5},
    {"independent_heads": 1}, {"communication": "float"}, {"optimizer": "sgd"},
])
def test_invalid_channel_configuration_fails_before_collection(override):
    with pytest.raises(ValueError):
        Config(**override)


def test_indexed_message_noise_replays_without_changing_global_rng():
    before = torch.get_rng_state().clone()
    arguments = dict(comm_rounds=3, heads=4)
    first = message_noise(57, 8, 500, 64, torch.float64, **arguments)
    repeat = message_noise(57, 8, 500, 64, torch.float64, **arguments)
    changed = message_noise(57, 9, 500, 64, torch.float64, **arguments)
    assert torch.equal(before, torch.get_rng_state())
    assert all(x.shape == (500, 4, 64) for x in first)
    for a, b, c in zip(first, repeat, changed):
        assert torch.equal(a, b) and not torch.equal(a, c)
    # At zero logits, independent broadcasts should have half their bits set,
    # without the perfect correlation produced by repeating one shared head.
    bits = torch.stack(first).permute(0, 2, 1, 3).flatten(0, 1).flatten(1) > 0
    rates = bits.double().mean(1)
    assert torch.all((rates > .48) & (rates < .52))
    correlations = torch.corrcoef(bits.double()) - torch.eye(12)
    assert correlations.abs().max() < .04


@pytest.mark.parametrize("communication,value_bits", [("binary", 1), ("real", 64)])
def test_channel_episode_replay_and_payload_accounting(communication, value_bits, monkeypatch):
    config = channel_config(communication=communication)
    model = SoftRoleNet(**config.model_kwargs()).double()
    scenario = make_scenarios(81, 1, ((2, 1),))[0]
    if communication == "real":
        def forbid_noise(*args, **kwargs):
            raise AssertionError("Real rollout must not generate message noise")
        monkeypatch.setattr("softrole.rollout.message_noise", forbid_noise)
    episode = run_episode(model, make_env(config), config, scenario, training=False, trace=True)
    repeated = run_episode(model, make_env(config), config, scenario, training=False, trace=True)
    assert episode.metrics == repeated.metrics and episode.traces == repeated.traces
    assert episode.metrics["payload_values_per_agent_step"] == 768
    assert episode.metrics["payload_value_bits"] == value_bits
    assert episode.metrics["payload_bits_per_agent_step"] == 768 * value_bits
    assert episode.metrics["payload_bits_generated"] == episode.metrics["steps"] * 3 * 768 * value_bits


@pytest.mark.parametrize("communication", ["binary", "real"])
def test_three_round_adam_resume_and_frozen_evaluation(tmp_path, communication):
    config = channel_config(communication=communication)
    full_path = train(config, tmp_path / "full")
    partial_path = train(replace(config, total_steps=1), tmp_path / "partial")
    resumed_path = train(config, tmp_path / "resumed", resume=partial_path)
    full = torch.load(full_path, weights_only=False)
    resumed = torch.load(resumed_path, weights_only=False)
    for key in ("model_state", "optimizer_state", "updates", "total_steps", "total_episodes"):
        assert_equal(full[key], resumed[key])
    scenarios = make_scenarios(92, 2, ((2, 1),))
    first = evaluate_checkpoint(full_path, scenarios, tmp_path / "first.json", trace=True)
    second = evaluate_checkpoint(full_path, scenarios, tmp_path / "second.json", trace=True)
    assert first["per_episode"] == second["per_episode"]
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
    metadata = json.loads((tmp_path / "full" / "run.json").read_text())
    assert ("straight-through bits" in metadata["approximations"]) == (communication == "binary")


def test_binary_and_real_evaluation_reports_remain_separate(tmp_path):
    scenarios = make_scenarios(92, 1, ((2, 1),))
    reports = []
    for channel in ("binary", "real"):
        config = channel_config(communication=channel, updates_per_epoch=1)
        checkpoint = train(config, tmp_path / channel)
        report = tmp_path / f"{channel}.json"
        evaluate_checkpoint(checkpoint, scenarios, report)
        reports.append(report)
    summary = summarize_reports(reports, bootstrap_samples=10)
    assert len(summary["groups"]) == 2
    assert {row["config"]["communication"] for row in summary["groups"]} == {"binary", "real"}
