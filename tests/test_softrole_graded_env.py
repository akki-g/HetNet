"""Physical radius geometry and exact native compatibility for graded sensing."""
from argparse import Namespace
import copy
import random

import numpy as np
import pytest
import torch

from softrole.env import adapt_observation, make_env
from softrole.model import SoftRoleNet


def config(**options):
    values = dict(task="pcp", dim=5, vision=2, max_steps=8, num_p=2, num_a=1,
                  capabilities="graded", descriptor="capability")
    values.update(options)
    return Namespace(**values)


def numpy_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def sensor_channels(adapter, observation):
    count = (2 * adapter.vision + 1) ** 2
    return observation[:, count * adapter.base:-2].reshape(len(observation), count, 3)


@pytest.mark.parametrize("composition", [(2, 1), (1, 2), (2, 2), (3, 1)])
def test_native_graded_episodes_equal_binary_observations_dynamics_and_rng(composition):
    n_p, n_a = composition
    binary = make_env(config(capabilities="binary", num_p=n_p, num_a=n_a))
    graded = make_env(config(num_p=n_p, num_a=n_a))
    global_numpy, global_python = np.random.get_state(), random.getstate()
    action_rng = np.random.default_rng(151)
    for seed in range(10):
        left, left_kappa = binary.reset(seed)
        right, right_kappa = graded.reset(seed)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left_kappa, right_kappa)
        assert (binary.raw_observation[n_p:, ..., binary.base:] == -1).all()
        assert (graded.raw_observation[n_p:, ..., graded.base:] >= 0).all()
        for _ in range(binary.max_steps):
            actions = action_rng.integers(0, 6, len(left_kappa))
            actions[:n_p] %= 5
            left = binary.step(actions, left_kappa)
            right = graded.step(actions, right_kappa)
            for index in (0, 1):
                np.testing.assert_array_equal(left[index], right[index])
            assert left[2] == right[2]
            assert left[3].keys() == right[3].keys()
            for key in left[3]:
                np.testing.assert_array_equal(left[3][key], right[3][key])
            np.testing.assert_array_equal(binary.positions, graded.positions)
            numpy_state_equal(binary._rng.numpy_state, graded._rng.numpy_state)
            assert binary._rng.python_state == graded._rng.python_state
            if left[2]:
                break
    numpy_state_equal(global_numpy, np.random.get_state())
    assert global_python == random.getstate()


def test_native_observations_give_exact_fixed_model_logits():
    binary = make_env(config(capabilities="binary"))
    graded = make_env(config())
    left, kappa = binary.reset(31)
    right, _ = graded.reset(31)
    with torch.random.fork_rng():
        torch.manual_seed(33)
        model = SoftRoleNet(mode="shared", communication="real").double()
    adjacency = ~torch.eye(len(kappa), dtype=torch.bool)
    with torch.no_grad():
        first = model(torch.from_numpy(left), torch.from_numpy(kappa),
                      model.initial_memory(len(kappa)), adjacency, 1.0)
        second = model(torch.from_numpy(right), torch.from_numpy(kappa),
                       model.initial_memory(len(kappa)), adjacency, 1.0)
    torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)


def test_radius_masks_centered_squares_and_preserves_positions_and_counts():
    adapter = make_env(config(num_p=3, num_a=1))
    _, kappa = adapter.reset(3, sensing=(0, 1, 2, 3))
    # Every cell is occupied: zeros after adaptation must be sensor masking.
    raw = adapter._raw_obs.reshape(4, 25, adapter.base + 4)
    raw[..., adapter.base:] = [2, 1, 1, 3]
    before = adapter._raw_obs.copy()
    observation = adapter.observe(kappa)
    channels = sensor_channels(adapter, observation).reshape(4, 5, 5, 3)
    expected = np.zeros((4, 5, 5), dtype=bool)
    expected[1, 2, 2] = True
    expected[2, 1:4, 1:4] = True
    expected[3] = True
    np.testing.assert_array_equal(channels[..., 0] > 0, expected)
    assert np.count_nonzero(channels[..., 0], axis=(1, 2)).tolist() == [0, 1, 9, 25]
    np.testing.assert_array_equal(channels[..., 1], expected)
    np.testing.assert_array_equal(channels[..., 2], expected * 5)
    np.testing.assert_array_equal(observation[:, :625], raw[..., :adapter.base].reshape(4, -1))
    np.testing.assert_array_equal(adapter._raw_obs, before)


@pytest.mark.parametrize("level,prey,visible", [(1, (2, 2), True), (1, (2, 3), False),
                                               (2, (3, 3), True), (2, (4, 2), False),
                                               (3, (4, 4), True)])
def test_sensing_capture_agent_sees_only_targets_in_its_radius(level, prey, visible):
    adapter = make_env(config())
    _, kappa = adapter.reset(3, sensing=(3, 3, level))
    adapter.raw.predator_loc[:] = [[0, 0], [0, 1]]
    adapter.raw.predator_capture_loc[:] = [2, 2]
    adapter.raw.prey_loc[:] = prey
    adapter._raw_obs = adapter.raw._get_obs().copy()
    observation = adapter.observe(kappa)
    assert bool(np.any(sensor_channels(adapter, observation)[2, :, 1])) is visible
    assert adapter.pcp_sensor_status(2)["target_visible"] is visible


@pytest.mark.parametrize("capabilities", ["binary", "graded"])
def test_type_descriptor_survives_failure_and_keeps_physical_capture_mask(capabilities):
    adapter = make_env(config(capabilities=capabilities, descriptor="type"))
    sensing = {} if capabilities == "binary" else {"sensing": (1, 3, 2)}
    _, kappa = adapter.reset(1, **sensing)
    before = adapter.descriptor(kappa)
    failed = kappa.copy()
    failed[0, 0] = 0
    observation = adapter.observe(failed)
    np.testing.assert_array_equal(observation[:, -2:], before)
    np.testing.assert_array_equal(adapter.descriptor(failed), before)
    np.testing.assert_array_equal(sensor_channels(adapter, observation)[0], 0)
    np.testing.assert_array_equal(before[:, 1] > 0, failed[:, 1] > 0)
    before[:] = -1
    np.testing.assert_array_equal(adapter.descriptor(failed), [[1, 0], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match="actuation action"):
        adapter.step(np.array([5, 4, 4]), failed)


def test_capability_descriptor_is_physical_and_pure_observations_do_not_commit_failure(monkeypatch):
    adapter = make_env(config())
    _, kappa = adapter.reset(4, sensing=(2, 3, 1))
    assert adapter.descriptor(kappa) is kappa
    before = adapter._raw_obs.copy()
    rng_before = copy.deepcopy(adapter._rng.numpy_state)
    failed = kappa.copy()
    failed[2, 0] = 0
    monkeypatch.setattr(adapter.raw, "_get_obs", lambda: pytest.fail("observe called simulator"))
    observation = adapter.observe(failed)
    np.testing.assert_array_equal(observation[:, -2:], failed)
    np.testing.assert_array_equal(sensor_channels(adapter, observation)[2], 0)
    np.testing.assert_array_equal(adapter.kappa, kappa)
    np.testing.assert_array_equal(adapter._raw_obs, before)
    numpy_state_equal(rng_before, adapter._rng.numpy_state)


def test_sensor_status_is_pure_and_rejects_committed_blind_victim(monkeypatch):
    adapter = make_env(config())
    _, kappa = adapter.reset(5, sensing=(3, 3, 2))
    status = adapter.pcp_sensor_status(2)
    monkeypatch.setattr(adapter.raw, "_get_obs", lambda: pytest.fail("status called simulator"))
    assert adapter.pcp_sensor_status(2) == status
    monkeypatch.undo()
    kappa[2, 0] = 0
    adapter.step(np.full(3, 4), kappa)
    with pytest.raises(ValueError, match="sensing agent"):
        adapter.pcp_sensor_status(2)


@pytest.mark.parametrize("sensing", [(3, 3), (3, 3, 4), (3, 3, -1), (3., 3., 1.),
                                    (True, True, False), (3, 3, np.nan), [[3, 3, 1]]])
def test_reset_rejects_invalid_level_vectors(sensing):
    with pytest.raises(ValueError, match="one integer"):
        make_env(config()).reset(3, sensing=sensing)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1 / 3, 4 / 3, 0.5])
def test_observe_rejects_nonfinite_or_nondiscrete_levels(bad):
    adapter = make_env(config())
    _, kappa = adapter.reset(2)
    kappa[0, 0] = bad
    with pytest.raises(ValueError, match="finite|integers"):
        adapter.observe(kappa)


def test_initial_sensing_ceiling_capture_class_shape_and_composition_are_checked():
    adapter = make_env(config())
    _, kappa = adapter.reset(2, sensing=(1, 3, 2))
    grown = kappa.copy()
    grown[0, 0] = 2 / 3
    with pytest.raises(ValueError, match="cannot grow"):
        adapter.observe(grown)
    changed_class = kappa.copy()
    changed_class[2, 1] = 0
    with pytest.raises(ValueError, match="actuation"):
        adapter.observe(changed_class)
    with pytest.raises(ValueError, match="Nx2"):
        adapter.observe(kappa[:2])
    _, resized = adapter.reset(4, num_p=1, num_a=2, sensing=(2, 1, 3))
    np.testing.assert_array_equal(adapter.native_levels, [3, 0, 0])
    np.testing.assert_array_equal(adapter.type_descriptor, [[1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(resized, [[2 / 3, 0], [1 / 3, 1], [1, 1]])
    _, native = adapter.reset(4)
    np.testing.assert_array_equal(native, adapter.native_kappa)


@pytest.mark.parametrize("options", [{"task": "fc"}, {"task": "pp", "num_a": 0},
                                      {"env_version": "paper-v1"}])
def test_graded_mode_rejects_unsupported_environments(options):
    with pytest.raises(ValueError, match="PCP corrected-observation-v1"):
        make_env(config(**options))


def test_binary_reset_and_option_validation():
    adapter = make_env(config(capabilities="binary"))
    with pytest.raises(ValueError, match="empty sensing"):
        adapter.reset(0, sensing=(3, 3, 0))
    empty, _ = adapter.reset(0, sensing=())
    default, _ = adapter.reset(0)
    np.testing.assert_array_equal(empty, default)
    for options, message in [({"capabilities": "other"}, "capabilities"),
                             ({"descriptor": "other"}, "descriptor"),
                             ({"descriptor": "type", "actor_capabilities": "physical_only"}, "physical_only")]:
        with pytest.raises(ValueError, match=message):
            make_env(config(**options))


def test_adapt_observation_checks_explicit_mask_and_tail_shape():
    raw = np.zeros((1, 2, 8))
    kappa = np.array([[0.5, 1.]])
    with pytest.raises(ValueError, match="per agent and cell"):
        adapt_observation(raw, kappa, 4, visible=[[1]])
    with pytest.raises(ValueError, match="per agent and cell"):
        adapt_observation(raw, kappa, 4, visible=[[1, 0.5]])
    with pytest.raises(ValueError, match="Nx2 descriptor"):
        adapt_observation(raw, kappa, 4, visible=[[1, 0]], tail=[[1]])
