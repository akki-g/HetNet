"""Optimizer-only checks with synthetic gradients; no environments or rollouts."""
import copy
from dataclasses import replace
import json

import pytest
import torch

from softrole.config import Config
from softrole.optimizers import make_optimizer, optimizer_spec, validate_optimizer_resume


def parameter():
    return torch.nn.Parameter(torch.tensor([1.25, -0.5], dtype=torch.float64))


def step(value, optimizer, gradient):
    value.grad = torch.tensor(gradient, dtype=value.dtype)
    optimizer.step()


def checkpoint(config, optimizer, legacy=False):
    record = dict(config=config.to_dict(), optimizer_state=copy.deepcopy(optimizer.state_dict()))
    if legacy:
        record["config"].pop("optimizer")
    else:
        record["optimizer"] = optimizer_spec(config)
    return record


def test_default_rmsprop_matches_historical_constructor_bitwise():
    config = Config()
    assert config.optimizer == "rmsprop"
    old_value, new_value = parameter(), parameter()
    old = torch.optim.RMSprop([old_value], lr=1e-4, alpha=.97, eps=1e-6)
    new = make_optimizer([new_value], config)
    assert old.param_groups[0].keys() == new.param_groups[0].keys()
    assert not new.state
    for gradient in ([2., -3.], [-1., 4.], [.125, -.25]):
        step(old_value, old, gradient)
        step(new_value, new, gradient)
        assert torch.equal(old_value, new_value)
    for name, expected in old.state[old_value].items():
        actual = new.state[new_value][name]
        assert torch.equal(expected, actual) if isinstance(expected, torch.Tensor) else expected == actual
    validate_optimizer_resume(config, checkpoint(config, old, legacy=True))


def test_adam_fresh_state_parameters_and_state_roundtrip():
    config = Config(optimizer="adam")
    value = parameter()
    optimizer = make_optimizer([value], config)
    assert isinstance(optimizer, torch.optim.Adam)
    assert not optimizer.state
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["betas"] == (.9, .999)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["foreach"] is False
    assert optimizer.param_groups[0]["fused"] is False
    step(value, optimizer, [2., -3.])
    saved = checkpoint(config, optimizer)
    validate_optimizer_resume(config, saved)
    resumed_value = torch.nn.Parameter(value.detach().clone())
    resumed = make_optimizer([resumed_value], config)
    resumed.load_state_dict(saved["optimizer_state"])
    for gradient in ([-1., 4.], [.125, -.25]):
        step(value, optimizer, gradient)
        step(resumed_value, resumed, gradient)
        assert torch.equal(value, resumed_value)
    for name, expected in optimizer.state[value].items():
        assert torch.equal(expected, resumed.state[resumed_value][name])


@pytest.mark.parametrize("legacy", [False, True])
def test_rmsprop_cannot_resume_as_adam_even_with_empty_state(legacy):
    config = Config()
    saved = checkpoint(config, make_optimizer([parameter()], config), legacy=legacy)
    with pytest.raises(ValueError, match="cannot resume rmsprop with adam"):
        validate_optimizer_resume(Config(optimizer="adam"), saved)


@pytest.mark.parametrize("tamper", ["selector", "metadata", "groups", "moments", "missing_metadata"])
def test_conflicting_adam_checkpoint_is_rejected(tamper):
    config = Config(optimizer="adam")
    value = parameter()
    optimizer = make_optimizer([value], config)
    step(value, optimizer, [2., -3.])
    saved = checkpoint(config, optimizer)
    if tamper == "selector":
        saved["config"]["optimizer"] = "rmsprop"
    elif tamper == "metadata":
        saved["optimizer"]["parameters"]["lr"] = .001
    elif tamper == "groups":
        saved["optimizer_state"]["param_groups"][0]["eps"] = 1e-6
    elif tamper == "moments":
        saved["optimizer_state"]["state"][0] = {"step": torch.tensor(1), "square_avg": value.detach()}
    else:
        saved.pop("optimizer")
    with pytest.raises(ValueError):
        validate_optimizer_resume(config, saved)


def test_unknown_optimizer_rejected_at_config_boundary():
    with pytest.raises(ValueError, match="optimizer must"):
        Config(optimizer="AdamW")


def test_training_resume_rejects_optimizer_switch_before_collection(tmp_path, monkeypatch):
    from softrole import CHECKPOINT_VERSION
    import softrole.train as engine

    config = Config()
    saved = checkpoint(config, make_optimizer([parameter()], config), legacy=True)
    saved.update(format_version=CHECKPOINT_VERSION, environment_version=config.env_version)
    path = tmp_path / "historical.pt"
    torch.save(saved, path)
    monkeypatch.setattr(engine, "collect_batch", lambda *args, **kwargs: pytest.fail("started collection"))
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="cannot resume rmsprop with adam"):
        engine.train(Config(optimizer="adam"), output, resume=path)
    assert not output.exists()


def test_adam_training_partial_resume_matches_uninterrupted_update_and_metadata(tmp_path):
    from softrole.train import train

    config = Config(optimizer="adam", model="shared", vision=1, max_steps=2,
                    pre_dim=8, hidden_dim=8, heads=1, head_dim=4, msg_dim=4,
                    seed=23, epochs=1, updates_per_epoch=2, batch_steps=1,
                    nprocesses=1, save_every=1)
    complete_path = train(config, tmp_path / "complete")
    partial_path = train(replace(config, total_steps=1), tmp_path / "partial")
    resumed_path = train(config, tmp_path / "resumed", resume=partial_path)
    complete = torch.load(complete_path, map_location="cpu", weights_only=False)
    partial = torch.load(partial_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    assert partial["updates"] == 1
    assert complete["updates"] == resumed["updates"] == 2
    assert complete["total_steps"] == resumed["total_steps"]
    assert complete["total_episodes"] == resumed["total_episodes"]
    for name, expected in complete["model_state"].items():
        assert torch.equal(expected, resumed["model_state"][name])
    for parameter_id, expected in complete["optimizer_state"]["state"].items():
        actual = resumed["optimizer_state"]["state"][parameter_id]
        assert expected.keys() == actual.keys()
        for name in expected:
            assert torch.equal(expected[name], actual[name])
    assert complete["optimizer_state"]["param_groups"] == resumed["optimizer_state"]["param_groups"]
    assert resumed["optimizer"] == optimizer_spec(config)
    metadata = json.loads((tmp_path / "resumed" / "run.json").read_text())
    assert metadata["optimizer"] == optimizer_spec(config)
    assert metadata["optimizer_initialization"] == "checkpoint"
    assert metadata["parent_source_sha256"] == partial["source_sha256"]
