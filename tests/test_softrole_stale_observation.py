"""The opt-in graded v2 closes the outside-flag leak without changing v1."""
import copy
from dataclasses import replace
import itertools
import json
import random

import numpy as np
import pytest
import torch

from softrole import CHECKPOINT_VERSION
from softrole.config import Config
from softrole.env import (ENVIRONMENT_VERSION, GRADED_ENVIRONMENT_VERSION,
                          adapt_observation, make_env)
from softrole.model import SoftRoleNet
from softrole.scenarios import make_scenarios


def _graded_config(**options):
    return Config(**dict(dict(capabilities="graded", env_version=GRADED_ENVIRONMENT_VERSION,
                              num_p=1, num_a=1, model="shared"), **options))


def _empty_window_observations(version):
    """Keep native position/outside geometry, with only focal own occupancy."""
    adapter = make_env(_graded_config(env_version=version))
    _, initial = adapter.reset(17, sensing=(3, 0))
    observations = {level: [] for level in range(4)}
    for position in itertools.product(range(5), repeat=2):
        adapter.raw.predator_loc[0] = position
        raw = adapter.raw._get_obs().copy().reshape(2, 25, adapter.base + 4)
        raw[..., adapter.base] = 0
        raw[..., adapter.base + 2:] = 0
        raw[0, 12, adapter.base] = 1
        adapter._raw_obs = raw
        for level in observations:
            physical = initial.copy()
            physical[0, 0] = level / 3
            observations[level].append(adapter.observe(physical)[0, :-2])
    return {level: np.stack(rows) for level, rows in observations.items()}


def test_v2_empty_windows_hide_positive_levels_at_all_25_positions():
    observations = _empty_window_observations(GRADED_ENVIRONMENT_VERSION)
    assert observations[1].shape == (25, 700)
    np.testing.assert_array_equal(observations[1], observations[2])
    np.testing.assert_array_equal(observations[1], observations[3])
    # Blind sensing remains distinguishable solely through own-cell occupancy.
    difference = observations[1] - observations[0]
    expected = np.zeros_like(difference)
    expected[:, 625 + 12 * 3 + 2] = 1
    np.testing.assert_array_equal(difference, expected)


def test_v1_empty_windows_retain_the_24_24_16_outside_flag_leak():
    observations = _empty_window_observations(ENVIRONMENT_VERSION)
    assert {pair: int(np.any(observations[pair[0]] != observations[pair[1]], axis=1).sum())
            for pair in ((1, 3), (2, 3), (1, 2))} == {(1, 3): 24, (2, 3): 24, (1, 2): 16}


def test_v2_preserves_position_width_and_masked_counts_without_mutation():
    config = _graded_config(num_p=3)
    adapter = make_env(config)
    _, physical = adapter.reset(19, sensing=(0, 1, 2, 3))
    raw = adapter._raw_obs.reshape(4, 25, adapter.base + 4)
    raw[..., adapter.base:] = [2, 1, 4, 3]
    before = adapter._raw_obs.copy()
    physical_before = physical.copy()
    rng_before = copy.deepcopy(adapter._rng.numpy_state)
    python_before = adapter._rng.python_state
    global_numpy, global_python = np.random.get_state(), random.getstate()
    observation = adapter.observe(physical)
    assert observation.shape == (4, adapter.observation_dim) == (4, 702)
    assert config.model_kwargs() == replace(config, env_version=ENVIRONMENT_VERSION).model_kwargs()
    np.testing.assert_array_equal(observation[:, :625], raw[..., :adapter.base].reshape(4, -1))
    sensory = observation[:, 625:-2].reshape(4, 25, 3)
    np.testing.assert_array_equal(sensory[..., 0], 1)
    visible = np.zeros((4, 5, 5), dtype=bool)
    visible[1, 2, 2] = True
    visible[2, 1:4, 1:4] = True
    visible[3] = True
    np.testing.assert_array_equal(sensory[..., 1], visible.reshape(4, 25) * 4)
    np.testing.assert_array_equal(sensory[..., 2], visible.reshape(4, 25) * 5)
    np.testing.assert_array_equal(observation[:, -2:], physical)
    np.testing.assert_array_equal(adapter._raw_obs, before)
    np.testing.assert_array_equal(physical, physical_before)
    np.testing.assert_array_equal(adapter.kappa, physical_before)
    for old, new in ((rng_before, adapter._rng.numpy_state), (global_numpy, np.random.get_state())):
        assert old[0] == new[0]
        np.testing.assert_array_equal(old[1], new[1])
        assert old[2:] == new[2:]
    assert adapter._rng.python_state == python_before
    assert random.getstate() == global_python
    observation[:] = -10
    np.testing.assert_array_equal(adapter._raw_obs, before)


@pytest.mark.parametrize("sensing", [0, 1])
def test_mask_outside_option_does_not_change_binary_transform(sensing):
    raw = np.arange(2 * 9 * 8).reshape(2, 9, 8)
    physical = np.array([[sensing, 0], [0, 1]])
    np.testing.assert_array_equal(adapt_observation(raw, physical, 4),
                                  adapt_observation(raw, physical, 4, mask_outside=False))


@pytest.mark.parametrize("options", [
    {"capabilities": "binary"},
    {"task": "pp", "num_a": 0},
    {"task": "fc"},
    {"env_version": "paper-v1"},
])
def test_config_and_direct_adapter_reject_unsupported_v2_or_graded_combinations(options):
    values = dict(task="pcp", num_p=1, num_a=1, capabilities="graded",
                  env_version=GRADED_ENVIRONMENT_VERSION)
    values.update(options)
    for constructor in (Config, lambda **settings: make_env(settings)):
        with pytest.raises(ValueError, match="graded-observation-v2|PCP corrected-observation-v1"):
            constructor(**values)


def test_v2_cannot_use_archived_paper_simulator(tmp_path):
    with pytest.raises(ValueError, match="archived paper simulator requires env_version=paper-v1"):
        make_env(_graded_config(), environment_source=tmp_path)


def test_v2_cli_is_opt_in_and_dry_run_creates_no_output(tmp_path, capsys):
    from softrole.__main__ import main
    output = tmp_path / "unused"
    main(["train", "--output", str(output), "--capabilities", "graded",
          "--env-version", GRADED_ENVIRONMENT_VERSION, "--dry-run"])
    settings = json.loads(capsys.readouterr().out)
    assert settings["env_version"] == GRADED_ENVIRONMENT_VERSION
    assert Config().env_version == ENVIRONMENT_VERSION
    assert make_env({}).environment_version == ENVIRONMENT_VERSION
    assert not output.exists()


def test_v2_checkpoint_refuses_a_forced_v1_evaluation_adapter(tmp_path, monkeypatch):
    import softrole.env
    from softrole.evaluate import evaluate_checkpoint

    config = _graded_config(max_steps=2, pre_dim=8, hidden_dim=4, heads=1,
                            head_dim=4, msg_dim=3)
    with torch.random.fork_rng():
        torch.manual_seed(27)
        model = SoftRoleNet(**config.model_kwargs()).double()
    checkpoint = tmp_path / "v2.pt"
    torch.save({"format_version": CHECKPOINT_VERSION, "config": config.to_dict(),
                "model_config": config.model_kwargs(), "model_state": model.state_dict(),
                "environment_version": GRADED_ENVIRONMENT_VERSION}, checkpoint)
    calls = []

    def forced_v1_adapter(saved_config, environment_source=None):
        calls.append(saved_config.env_version)
        return make_env(replace(saved_config, env_version=ENVIRONMENT_VERSION), environment_source)

    monkeypatch.setattr(softrole.env, "make_env", forced_v1_adapter)
    scenarios = make_scenarios(31, 1, [(1, 1)], sensing_p=(1,))
    output = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="Checkpoint environment version differs"):
        evaluate_checkpoint(checkpoint, scenarios, output)
    assert calls == [GRADED_ENVIRONMENT_VERSION]
    assert not output.exists()
