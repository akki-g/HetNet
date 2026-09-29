"""Small real-domain audits of the new synchronous learner and resumability."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing as mp

import numpy as np
import pytest
import torch

from softrole.config import Config
from softrole.model import SoftRoleNet
from softrole.train import apply_gradients, train, worker_collect


def tiny_config(**overrides):
    settings = dict(task="pcp", num_p=2, num_a=1, dim=5, vision=1, max_steps=3,
                    experts=2, pre_dim=8, hidden_dim=8, heads=1, head_dim=4,
                    msg_dim=4, seed=23, epochs=1, updates_per_epoch=3,
                    batch_steps=2, nprocesses=1, save_every=1, detach_gap=2)
    settings.update(overrides)
    return Config(**settings)


@pytest.mark.parametrize("clip", [1., 1000.])
def test_unequal_collector_episode_counts_are_averaged_once_before_one_clip(monkeypatch, clip):
    model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    batches = [{"num_episodes": 1, "gradients": [np.array([[100., 0.]]), np.array([0.])]},
               {"num_episodes": 3, "gradients": [np.array([[-96., 8.]]), np.array([12.])]}]
    # The global episode mean is (1,2,3); local means or locally clipped
    # gradients would produce a different direction, not merely a scale.
    observed = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        observed.append(torch.cat([parameter.grad.flatten().clone() for parameter in parameters]))
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    norm = apply_gradients(model, optimizer, batches, clip)
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], torch.tensor([1., 2., 3.], dtype=torch.float64))
    assert norm == pytest.approx(np.sqrt(14.))
    expected = -0.5 * np.array([1., 2., 3.]) * min(1., clip / np.sqrt(14.))
    actual = torch.cat([model.weight.detach().flatten(), model.bias.detach()]).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_spawned_collectors_match_serial_frozen_snapshot_collections():
    config = tiny_config(nprocesses=2)
    torch.manual_seed(config.seed)
    model = SoftRoleNet(**config.model_kwargs()).double()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    serial = [worker_collect(state, config.model_kwargs(), config.to_dict(), rank, 7)
              for rank in range(2)]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        pending = [pool.submit(worker_collect, state, config.model_kwargs(), config.to_dict(), rank, 7)
                   for rank in range(2)]
        spawned = [future.result(timeout=60) for future in pending]
    for expected, actual in zip(serial, spawned):
        assert actual.keys() == expected.keys()
        for key in expected.keys() - {"gradients"}:
            assert actual[key] == expected[key]
        for actual_gradient, expected_gradient in zip(actual["gradients"], expected["gradients"]):
            np.testing.assert_array_equal(actual_gradient, expected_gradient)


def assert_state_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_state_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_state_equal(left, right)
    else:
        assert first == second


def test_partial_epoch_resume_matches_uninterrupted_model_and_optimizer(tmp_path):
    config = tiny_config()
    uninterrupted_path = train(config, tmp_path / "uninterrupted")
    partial_path = train(replace(config, total_steps=1), tmp_path / "partial")
    partial = torch.load(partial_path, map_location="cpu", weights_only=False)
    assert partial["updates"] == 1
    assert partial["completed_epochs"] == 0
    assert partial["updates_in_partial_epoch"] == 1
    resumed_path = train(config, tmp_path / "resumed", resume=partial_path)
    uninterrupted = torch.load(uninterrupted_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    for field in ("updates", "total_steps", "total_episodes", "completed_epochs", "updates_in_partial_epoch"):
        assert uninterrupted[field] == resumed[field]
    assert resumed["updates"] == 3
    assert_state_equal(uninterrupted["model_state"], resumed["model_state"])
    assert_state_equal(uninterrupted["optimizer_state"], resumed["optimizer_state"])
    metadata = json.loads((tmp_path / "resumed" / "run.json").read_text())
    assert metadata["parent_source_sha256"] == partial["source_sha256"]
    with pytest.raises(FileExistsError):
        train(config, tmp_path / "uninterrupted")


@pytest.mark.parametrize("override", [
    {"seed": 0.5}, {"vision": 1.5}, {"total_steps": 1.5},
    {"failure_window": (1,)}, {"failure_window": (1.5, 2)},
    {"failure_window": (-1, 2)}, {"failure_window": (3, 2)},
])
def test_malformed_programmatic_config_is_rejected_before_collection(override):
    with pytest.raises(ValueError):
        tiny_config(**override)
