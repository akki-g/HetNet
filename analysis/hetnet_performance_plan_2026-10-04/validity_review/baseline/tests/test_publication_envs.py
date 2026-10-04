"""Physical contracts for the isolated historical reconstruction.

Every check imports the frozen runtime in a fresh interpreter. Its original Gym
IDs and module names therefore cannot replace the existing experiments' modules.
"""
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


RUNTIME = Path(__file__).resolve().parents[1] / "publication_reconstruction" / "runtime"
COMMON = """
import argparse
import inspect
import numpy as np
from ic3net_envs.predator_prey_env import PredatorPreyEnv
from ic3net_envs.predator_capture_env import PredatorCaptureEnv
from ic3net_envs.fire_commander_env import FireCommanderEnv

VERSIONS = ('historical-2022', 'corrected-v1')

def make(task, version='historical-2022', seed=501):
    cls = dict(pp=PredatorPreyEnv, pcp=PredatorCaptureEnv, fc=FireCommanderEnv)[task]
    assert 'publication_reconstruction/runtime/' in inspect.getfile(cls)
    env = cls()
    parser = argparse.ArgumentParser()
    env.init_args(parser)
    args = parser.parse_args([])
    args.publication_env_version = version
    args.nagents = 3
    args.max_steps = 300 if task == 'fc' else 80
    args.eval = False
    args.vision = 1 if task == 'fc' else 2
    args.tensor_obs = False
    env.multi_agent_init(args)
    np.random.seed(seed)
    env.reset()
    return env

def place_fire(env, cell):
    env.fire_loc = np.array([cell], dtype=int)
    env.nfire = 1
    env.fire_out = False
    env.episode_over = False
    env.discovered_fire = []
    env._fire_init()
    # Zero spread makes a controlled static-fire test independent of wind RNG.
    env.geo_phys_info['spread_rate'][:] = 0
    env.geo_phys_info['wind_speed'][:] = 0
    env.geo_phys_info['wind_direction'][:] = 0
"""


def isolated(body):
    setup = f"import sys\nsys.path[:0] = [{str(RUNTIME)!r}, {str(RUNTIME / 'envs')!r}]\n"
    result = subprocess.run([sys.executable, "-I", "-c", setup + COMMON + textwrap.dedent(body)],
                            cwd=RUNTIME, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
def test_publication_reset_and_step_are_finite(task):
    isolated(f"""
        for version in VERSIONS:
            env = make({task!r}, version)
            obs, reward, done, info = env.step([4, 4, 4])
            assert obs.shape == (3, 3 if {task!r} == 'fc' else 5,
                                 3 if {task!r} == 'fc' else 5, 29)
            assert reward.shape == (3,)
            assert np.isfinite(obs).all() and np.isfinite(reward).all()
            assert env.publication_env_version == version
    """)


@pytest.mark.parametrize("task", ["pp", "pcp"])
def test_publication_capture_requires_the_declared_physical_conditions(task):
    isolated(f"""
        for version in VERSIONS:
            env = make({task!r}, version)
            env.prey_loc[:] = [2, 2]
            if {task!r} == 'pp':
                env.predator_loc[:] = [[2, 1], [1, 2], [2, 3]]
                _, reward, done, _ = env.step([1, 2, 3])
                assert done and env.stat['success'] == 1
                np.testing.assert_array_equal(reward, [0., 0., 0.])
            else:
                env.predator_loc[:] = [[2, 1], [1, 2]]
                env.predator_capture_loc[:] = [2, 2]
                _, reward, done, _ = env.step([1, 2, 4])
                assert not done and env.stat['success'] == 0
                np.testing.assert_array_equal(reward, [0., 0., -.05])
                _, reward, done, _ = env.step([4, 4, 5])
                assert done and env.stat['success'] == 1
                np.testing.assert_array_equal(reward, [0., 0., 0.])
    """)


def test_publication_sensor_views_are_corrected_only_in_corrected_mode():
    isolated("""
        for task in ('pcp', 'fc'):
            for version in VERSIONS:
                env = make(task, version)
                env.predator_loc[:] = [[2, 1], [4, 4]]
                env.predator_capture_loc[:] = [2, 2]
                if task == 'pcp':
                    env.prey_loc[:] = [2, 2]
                    channel, row, col, erased = env.PREY_CLASS, 2, 3, -1
                else:
                    place_fire(env, [2, 2])
                    channel, row, col, erased = env.FIRE_CLASS, 1, 2, 0
                obs = env._get_obs()
                expected = 1 if version == 'corrected-v1' else erased
                assert obs[0, row, col, channel] == expected
                assert np.all(obs[2, ..., env.BASE:] == erased)
    """)


def test_publication_fc_all_initial_cells_preserve_explicit_version_semantics():
    isolated("""
        historical_phantoms = []
        for version in VERSIONS:
            for row in range(5):
                for col in range(5):
                    env = make('fc', version)
                    place_fire(env, [row, col])
                    env.step([4, 4, 4])
                    fires = env.fire_loc.tolist()
                    if version == 'corrected-v1':
                        assert fires == [[row, col]]
                        np.testing.assert_array_equal(env.ign_points_all[:, :2], [[row, col]])
                    elif (row == 0 or col == 0) and [row, col] != [0, 0]:
                        assert [0, 0] in fires
                        historical_phantoms.append([row, col])
        assert len(historical_phantoms) == 8
        env = make('fc', 'corrected-v1')
        place_fire(env, [2, 2])
        env.ign_points_all = np.array([[0., 2., 1.], [2., 2., 1.]])
        env.previous_terrain_map = env.ign_points_all.copy()
        env.fire_loc = np.array([[0, 2], [2, 2]])
        env._fire_propagation()
        np.testing.assert_array_equal(env.ign_points_all[:, :2], [[0, 2], [2, 2]])
    """)


def test_publication_fc_reward_and_termination_keep_legacy_individual_rewards():
    isolated("""
        for version in VERSIONS:
            env = make('fc', version)
            place_fire(env, [2, 2])
            env.predator_capture_loc[:] = [2, 2]
            _, reward, done, _ = env.step([4, 4, 5])
            assert done and env.stat['success'] and env.nfire == 0
            np.testing.assert_array_equal(reward, [0., 0., 10.])
        env = make('fc', 'historical-2022')
        place_fire(env, [0, 2])
        env.predator_capture_loc[:] = [0, 0]
        env.step([4, 4, 4])
        _, reward, done, _ = env.step([4, 4, 5])
        np.testing.assert_array_equal(reward, [-.1, -.1, 10.])
        assert not done and env.fire_loc.tolist() == [[0, 2]]
        env = make('fc', 'corrected-v1')
        place_fire(env, [0, 2])
        env.predator_capture_loc[:] = [0, 0]
        env.step([4, 4, 4])
        _, reward, done, _ = env.step([4, 4, 5])
        np.testing.assert_array_equal(reward, [-.1, -.1, -.2])
        assert not done and env.fire_loc.tolist() == [[0, 2]]
    """)


def test_publication_fc_discovery_membership_is_versioned():
    isolated("""
        for version in VERSIONS:
            env = make('fc', version)
            place_fire(env, [2, 2])
            env.predator_loc[:] = [[2, 2], [4, 4]]
            env.predator_capture_loc[:] = [0, 0]
            env.fire_loc = np.array([[2, 2], [2, 3]])
            env._get_obs()
            # Historical NumPy membership mistakes a shared component for a
            # match; corrected-v1 compares complete coordinate rows.
            assert len(env.discovered_fire) == (1 if version == 'historical-2022' else 2)
    """)


def test_publication_fc_corrected_discovery_and_source_need_both_coordinates():
    isolated("""
        for version in VERSIONS:
            env = make('fc', version)
            place_fire(env, [1, 2])
            env.predator_loc[:] = [[1, 2], [4, 4]]
            env.predator_capture_loc[:] = [0, 0]
            env.fire_loc = np.array([[1, 2], [1, 3]])
            env.discovered_fire = [np.array([1, 2])]
            env.just_discovered_source[:] = 0
            env.just_discovered_nonsource[:] = 0
            env._get_obs()
            if version == 'corrected-v1':
                assert len(env.discovered_fire) == 2
                np.testing.assert_array_equal(env.just_discovered_source, [0, 0])
                np.testing.assert_array_equal(env.just_discovered_nonsource, [1, 0])
            else:
                assert len(env.discovered_fire) == 1

            # Isolate source classification from already-discovered filtering:
            # (1,3) shares one component with source (1,2), but is not its row.
            env.fire_loc = np.array([[1, 3]])
            env.discovered_fire = []
            env.just_discovered_source[:] = 0
            env.just_discovered_nonsource[:] = 0
            env._get_obs()
            if version == 'corrected-v1':
                np.testing.assert_array_equal(env.just_discovered_source, [0, 0])
                np.testing.assert_array_equal(env.just_discovered_nonsource, [1, 0])
            else:
                np.testing.assert_array_equal(env.just_discovered_source, [1, 0])
                np.testing.assert_array_equal(env.just_discovered_nonsource, [0, 0])
    """)


def test_publication_fc_corrected_outgoing_front_retains_previous_position():
    isolated("""
        for version in VERSIONS:
            env = make('fc', version)
            place_fire(env, [2, 2])
            env.geo_phys_info['spread_rate'][:] = 1000
            env.geo_phys_info['wind_speed'][:] = 10
            env.geo_phys_info['wind_direction'][:] = np.pi / 4
            env._fire_propagation()
            if version == 'corrected-v1':
                np.testing.assert_array_equal(env.ign_points_all[:, :2], [[2., 2.]])
            else:
                assert (env.ign_points_all[:, :2] > 4).any()
    """)


def test_publication_fc_bounded_random_rollouts_are_finite():
    isolated("""
        for version in VERSIONS:
            for seed in range(5):
                env = make('fc', version, seed)
                actions = np.random.RandomState(100 + seed)
                for step in range(300):
                    obs, reward, done, info = env.step(actions.randint(0, 6, size=3))
                    assert np.isfinite(obs).all() and np.isfinite(reward).all()
                    assert ((env.fire_loc >= 0) & (env.fire_loc <= 4)).all()
                    if version == 'corrected-v1':
                        assert ((env.ign_points_all[:, :2] >= 0) &
                                (env.ign_points_all[:, :2] <= 4)).all()
                    if done:
                        break
    """)


def test_publication_environment_rejects_unknown_versions():
    isolated("""
        for task in ('pp', 'pcp', 'fc'):
            try:
                make(task, 'unknown')
            except ValueError as exc:
                assert 'publication_env_version' in str(exc)
            else:
                raise AssertionError('unknown version accepted')
    """)
