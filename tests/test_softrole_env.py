"""Physical observations and scenario reproducibility for the new learner."""
from argparse import Namespace
import copy
import random

import numpy as np
import pytest

from ic3net_envs.fire_commander_env import FireCommanderEnv
from ic3net_envs.predator_capture_env import PredatorCaptureEnv
from softrole.env import ENVIRONMENT_VERSION, adapt_observation, make_env
from softrole.scenarios import IsolatedRNG, make_scenarios


def config(task="pcp", **options):
    defaults = dict(task=task, dim=5, vision=1 if task == "fc" else 2,
                    max_steps=4, num_p=3 if task == "pp" else 2,
                    num_a=0 if task == "pp" else 1, nfires=1, reward_type=3)
    defaults.update(options)
    return Namespace(**defaults)


def assert_numpy_state_equal(first, second):
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2:] == second[2:]


def test_observation_removes_type_labels_but_retains_counts_and_proprioception():
    raw = np.zeros((2, 2, 8))  # B=4, two visible cells.
    raw[:, 0, 2] = 1
    raw[:, 1, 3] = 1
    raw[:, 0, 4:] = [2, 0, 1, 3]  # Five co-located agents and a prey.
    raw[:, 1, 4:] = [0, 1, 0, 0]
    kappa = np.array([[1., 0.], [0., 1.]])
    before = raw.copy()
    adapted = adapt_observation(raw, kappa, base=4)
    swapped = raw.copy()
    swapped[..., [4, 7]] = raw[..., [7, 4]]
    np.testing.assert_array_equal(adapted, adapt_observation(swapped, kappa, 4))
    np.testing.assert_array_equal(adapted[0, 8:14], [0, 1, 5, 1, 0, 0])
    np.testing.assert_array_equal(adapted[1, 8:14], 0)
    np.testing.assert_array_equal(adapted[:, :8], raw[..., :4].reshape(2, -1))
    np.testing.assert_array_equal(adapted[:, -2:], kappa)
    np.testing.assert_array_equal(raw, before)


def test_blindness_copy_correction_is_opt_in_and_preserves_visible_prey():
    assert PredatorCaptureEnv().independent_observations is False
    assert FireCommanderEnv().independent_observations is False
    adapter = make_env(config())
    adapter.reset(0)
    raw = adapter.raw
    raw.predator_loc = np.array([[2, 2], [0, 0]])
    raw.predator_capture_loc = np.array([[2, 3]])
    raw.prey_loc = np.array([[3, 2]])
    corrected = raw._get_obs()
    assert corrected[0, 3, 2, raw.PREY_CLASS] == 1
    assert np.all(corrected[2, ..., raw.BASE:] == -1)
    raw.independent_observations = False
    legacy = raw._get_obs()
    assert legacy[0, 3, 2, raw.PREY_CLASS] == -1
    assert not np.array_equal(legacy[:2], corrected[:2])


def test_fc_copy_correction_preserves_visible_fire():
    adapter = make_env(config("fc"))
    adapter.reset(0)
    raw = adapter.raw
    raw.predator_loc = np.array([[2, 2], [0, 0]])
    raw.predator_capture_loc = np.array([[2, 3]])
    raw.fire_loc = np.array([[3, 2]])
    raw.discovered_fire = [np.array([3, 2])]
    corrected = raw._get_obs()
    assert corrected[0, 2, 1, raw.FIRE_CLASS] == 1
    assert np.all(corrected[2, ..., raw.BASE:] == 0)
    raw.independent_observations = False
    legacy = raw._get_obs()
    assert legacy[0, 2, 1, raw.FIRE_CLASS] == 0


@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
def test_domain_step_shape_and_truncation(task):
    adapter = make_env(config(task))
    obs, kappa = adapter.reset(seed=19)
    n = len(kappa)
    assert obs.shape == (n, adapter.observation_dim)
    assert obs.dtype == np.float64
    assert adapter.positions.shape == (n, 2)
    assert adapter.raw.independent_observations is True
    for _ in range(adapter.max_steps):
        obs, reward, done, info = adapter.step(np.full(n, 4), kappa)
        assert obs.shape == (n, adapter.observation_dim)
        assert reward.shape == (n,)
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        assert isinstance(info["success"], bool)
        if done:
            break
    assert done
    assert info["terminated"] or info["truncated"]
    assert info["environment_version"] == ENVIRONMENT_VERSION
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(np.full(n, 4), kappa)


@pytest.mark.parametrize("num_p,num_a", [(3, 2), (1, 2)])
def test_composition_reset_updates_physical_bookkeeping(num_p, num_a):
    adapter = make_env(config())
    adapter.reset(0)
    obs, kappa = adapter.reset(1, num_p, num_a)
    assert len(obs) == len(kappa) == len(adapter.positions) == num_p + num_a
    assert adapter.raw.args.nfriendly_P == adapter.raw.predator_capture_index == num_p
    assert adapter.raw.args.nfriendly_A == num_a
    assert adapter.raw.captured_prey_index == num_p + num_a
    assert len(adapter.raw.reached_prey) == num_p + num_a
    assert len(adapter.raw.captured_prey) == num_a


@pytest.mark.parametrize("task", ["pp", "pcp"])
def test_actual_success_requires_reaching_and_capture(task):
    adapter = make_env(config(task))
    _, kappa = adapter.reset(3)
    prey = adapter.raw.prey_loc[0]
    adapter.raw.predator_loc[:] = prey
    adapter.raw.predator_capture_loc[:] = prey
    actions = np.where(kappa[:, 1] == 1, 5, 4)
    _, _, done, info = adapter.step(actions, kappa)
    assert done and info["success"] and info["terminated"]
    assert not info["truncated"]


def test_fc_success_is_reported_when_final_fire_is_extinguished():
    adapter = make_env(config("fc"))
    _, kappa = adapter.reset(3)
    adapter.raw.fire_spread_off = True
    adapter.raw.predator_capture_loc[:] = adapter.raw.fire_loc[0]
    _, _, done, info = adapter.step(np.where(kappa[:, 1] == 1, 5, 4), kappa)
    assert done and info["success"] and info["terminated"]
    assert not info["truncated"]


def test_sensor_failure_masks_only_victim_and_observe_has_no_side_effects():
    adapter = make_env(config())
    obs, kappa = adapter.reset(9)
    cache = adapter._raw_obs.copy()
    failed = kappa.copy()
    failed[0, 0] = 0
    result = adapter.observe(failed)
    np.testing.assert_array_equal(result[1:], obs[1:])
    position_size = (2 * adapter.vision + 1) ** 2 * adapter.base
    np.testing.assert_array_equal(result[0, :position_size], obs[0, :position_size])
    np.testing.assert_array_equal(result[0, position_size:-2], 0)
    np.testing.assert_array_equal(adapter._raw_obs, cache)
    np.testing.assert_array_equal(adapter.kappa, kappa)
    adapter.step(np.full(len(kappa), 4), failed)
    np.testing.assert_array_equal(adapter.kappa, failed)


def test_fc_observe_never_repeats_discovery_or_calls_simulator(monkeypatch):
    adapter = make_env(config("fc"))
    obs, kappa = adapter.reset(12)
    discovered = copy.deepcopy(adapter.raw.discovered_fire)
    monkeypatch.setattr(adapter.raw, "_get_obs", lambda: pytest.fail("observe invoked simulator"))
    for _ in range(3):
        np.testing.assert_array_equal(adapter.observe(kappa), obs)
    np.testing.assert_array_equal(adapter.raw.discovered_fire, discovered)


@pytest.mark.parametrize("task", ["pp", "fc"])
def test_unsupported_sensor_events_are_rejected(task):
    adapter = make_env(config(task))
    _, kappa = adapter.reset(1)
    kappa[0, 0] = 0
    with pytest.raises(ValueError, match="only for PCP"):
        adapter.observe(kappa)


def test_invalid_physical_capabilities_are_rejected():
    adapter = make_env(config())
    _, kappa = adapter.reset(1)
    kappa[0, 1] = 1
    with pytest.raises(ValueError, match="actuation"):
        adapter.observe(kappa)
    with pytest.raises(ValueError, match="zero actuation"):
        make_env(config("pp", num_a=1))


def test_isolated_rng_retains_local_state_and_restores_both_global_streams():
    np_before, python_before = np.random.get_state(), random.getstate()
    first, second = IsolatedRNG(13), IsolatedRNG(13)
    with first.use():
        a = (np.random.random(), random.random())
    with first.use():
        b = (np.random.random(), random.random())
    with second.use():
        assert a == (np.random.random(), random.random())
        assert b == (np.random.random(), random.random())
    assert_numpy_state_equal(np_before, np.random.get_state())
    assert python_before == random.getstate()
    with pytest.raises(RuntimeError), first.use():
        np.random.random()
        random.random()
        raise RuntimeError("exception must still restore streams")
    assert_numpy_state_equal(np_before, np.random.get_state())
    assert python_before == random.getstate()


@pytest.mark.parametrize("task", ["pcp", "fc"])
def test_environment_replay_ignores_other_random_consumers(task):
    first, second = make_env(config(task)), make_env(config(task))
    np_before, py_before = np.random.get_state(), random.getstate()
    x, kappa = first.reset(83)
    y, other_kappa = second.reset(83)
    np.testing.assert_array_equal(x, y)
    unrelated = IsolatedRNG(999)
    for _ in range(3):
        with unrelated.use():
            np.random.random(101)
            random.random()
        left = first.step(np.full(len(kappa), 4), kappa)
        right = second.step(np.full(len(kappa), 4), other_kappa)
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2] == right[2]
        if left[2]:
            break
    assert_numpy_state_equal(np_before, np.random.get_state())
    assert py_before == random.getstate()


def test_scenarios_have_stable_prefixes_and_independent_streams():
    np_before, py_before = np.random.get_state(), random.getstate()
    short = make_scenarios(44, 5, [(2, 1), (3, 2)], 1.0, (10, 12))
    long = make_scenarios(44, 20, [(2, 1), (3, 2)], 1.0, (10, 12))
    assert short == long[:5]
    no_events = make_scenarios(44, 5, [(2, 1), (3, 2)], 0.0)
    for event, none in zip(short, no_events):
        assert (event.env_seed, event.action_seed, event.message_seed) == (
            none.env_seed, none.action_seed, none.message_seed)
        assert len({event.env_seed, event.action_seed, event.message_seed}) == 3
        assert 10 <= event.event_step <= 12 and 0 <= event.victim < event.num_p
        assert none.event_step == none.victim == -1
    assert_numpy_state_equal(np_before, np.random.get_state())
    assert py_before == random.getstate()
    with pytest.raises(ValueError, match="at least two"):
        make_scenarios(1, 2, [(1, 1)], 1.0)
