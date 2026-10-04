"""Paper FC events and the shared, source-bound SoftRole simulator contract."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from softrole.config import Config, recipe
from softrole.env import ENVIRONMENT_VERSION, make_env
from softrole.model import SoftRoleNet
from softrole.publication_env import (SIMULATOR_FILES, checkpoint_binding,
    load_environment_class, simulator_identity, source_binding)
from softrole.scenarios import IsolatedRNG, make_scenarios


ROOT = Path(__file__).resolve().parents[1]


def native(task="fc", version="paper-v1", seed=41):
    env = load_environment_class(task)()
    parser = argparse.ArgumentParser()
    env.init_args(parser)
    args = parser.parse_args([])
    args.publication_env_version = version
    args.max_steps = 300 if task == "fc" else 80
    args.vision = 1 if task == "fc" else 2
    args.nfriendly_P, args.nfriendly_A = ((3, 0) if task == "pp" else (2, 1))
    args.eval = False
    env.multi_agent_init(args)
    rng = IsolatedRNG(seed)
    with rng.use():
        env.reset()
    return env, rng


def place(env, fire=(2, 2), discovered=True):
    env.predator_loc[:] = [[0, 0], [4, 4]]
    env.predator_capture_loc[:] = fire
    env.fire_loc = np.array([fire])
    env.nfire = 1
    env.fire_out = env.episode_over = False
    env.discovered_fire = [np.array(fire)] if discovered else []
    env._fire_init()
    env.fire_spread_off = True


def test_paper_fc_terminal_extinction_rewards_every_agent_and_costs_final_step():
    env, _ = native()
    place(env)
    _, reward, done, info = env.step([0, 2, 4])
    np.testing.assert_array_equal(reward, [9.9, 9.9, 9.9])
    assert done and env.stat["success"] and not len(env.fire_loc)
    assert info["paper_events"] == {"new_ignitions": 0, "extinctions": 1, "false_drops": [0]}
    before = dict(env.stat, enemy_count=list(env.stat["enemy_count"]))
    np.testing.assert_array_equal(env.reward_terminal(), [0, 0, 0])
    assert env.stat == before


def test_discovery_from_new_observation_cannot_authorize_same_transition_drop():
    env, _ = native()
    place(env, discovered=False)
    env.predator_loc[0] = [2, 0]  # One move brings the fire into its FOV.
    _, reward, done, info = env.step([1, 2, 4])
    np.testing.assert_array_equal(reward, [-.1, -.1, -.2])
    assert not done and env.fire_loc.tolist() == [[2, 2]]
    assert env._contains_fire_position([2, 2], env.discovered_fire)
    assert info["paper_events"]["false_drops"] == [1]
    _, reward, done, _ = env.step([0, 2, 4])
    assert done
    np.testing.assert_array_equal(reward, [9.9, 9.9, 9.9])


def test_absent_fire_drop_is_only_an_extra_penalty_for_acting_agent():
    env, _ = native()
    place(env)
    env.predator_capture_loc[:] = [1, 1]
    _, reward, done, info = env.step([0, 2, 4])
    np.testing.assert_array_equal(reward, [-.1, -.1, -.2])
    assert not done and info["paper_events"]["extinctions"] == 0


def test_extinguishing_nonfront_fire_has_same_shared_reward():
    env, _ = native()
    place(env)
    env.fire_loc = np.array([[2, 2], [3, 3]])
    env.nfire = 2
    env.ign_points_all[:, :2] = [3, 3]
    _, reward, done, info = env.step([0, 2, 4])
    np.testing.assert_array_equal(reward, [9.9, 9.9, 9.9])
    assert not done and env.fire_loc.tolist() == [[3, 3]]
    assert info["paper_events"]["extinctions"] == 1


def test_new_fire_penalty_counts_unique_ignitions_not_total_or_front_rows(monkeypatch):
    env, _ = native()
    place(env)
    env.fire_spread_off = False
    fronts = np.array([[2., 2., 1.], [3., 2., 1.], [3., 2., 2.], [4., 2., 1.]])
    monkeypatch.setattr(env.fire_mdl, "fire_propagation", lambda *a, **k: (fronts.copy(), fronts.copy()))
    _, reward, _, info = env.step([0, 2, 0])
    np.testing.assert_allclose(reward, [-.3, -.3, -.3], rtol=0, atol=1e-15)
    assert info["paper_events"]["new_ignitions"] == 2
    _, reward, _, info = env.step([0, 2, 0])
    np.testing.assert_array_equal(reward, [-.1, -.1, -.1])
    assert info["paper_events"]["new_ignitions"] == 0


def test_suppression_precedes_spread_and_new_sensing(monkeypatch):
    env, _ = native()
    place(env)
    env.fire_spread_off = False

    def spread():
        assert len(env.fire_loc) == 0
        env.fire_loc = np.array([[4, 3]])
        env.nfire = 1

    monkeypatch.setattr(env, "_fire_propagation", spread)
    _, reward, done, info = env.step([0, 2, 4])
    np.testing.assert_allclose(reward, [9.8, 9.8, 9.8], rtol=0, atol=1e-15)
    assert not done and not env.fire_out
    assert env._contains_fire_position([4, 3], env.discovered_fire)
    assert info["paper_events"] == {"new_ignitions": 1, "extinctions": 1, "false_drops": [0]}


@pytest.mark.parametrize("actions", [[4, 0, 0], [0, 0, 5], [-1, 0, 0], [0, .5, 0], [0, 0], [0, 0, float("nan")]])
def test_paper_native_actions_reject_invalid_input_before_mutation(actions):
    env, _ = native()
    before = env.predator_loc.copy()
    with pytest.raises(ValueError, match="native action"):
        env.step(actions)
    np.testing.assert_array_equal(env.predator_loc, before)
    assert env.naction == 5


@pytest.mark.parametrize("task", ["pp", "pcp"])
def test_paper_pp_pcp_match_corrected_observations_rewards_and_rng(task):
    left, lrng = native(task, "paper-v1")
    right, rrng = native(task, "corrected-v1")
    np.testing.assert_array_equal(left.obs, right.obs)
    for actions in ([0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 4, 4]):
        with lrng.use():
            lresult = left.step(actions)
        with rrng.use():
            rresult = right.step(actions)
        np.testing.assert_array_equal(lresult[0], rresult[0])
        np.testing.assert_array_equal(lresult[1], rresult[1])
        assert lresult[2] == rresult[2]


@pytest.mark.parametrize("cell", [(0, 2), (2, 0), (2, 2), (4, 4)])
def test_paper_keeps_corrected_farsite_boundary_contract(cell):
    left, lrng = native(version="paper-v1")
    right, rrng = native(version="corrected-v1")
    for env, rng in ((left, lrng), (right, rrng)):
        with rng.use():
            place(env, cell)
        env.geo_phys_info["spread_rate"][:] = 1000
        env.geo_phys_info["wind_speed"][:] = 10
        env.geo_phys_info["wind_direction"][:] = np.pi / 4
        with rng.use():
            env._fire_propagation()
    np.testing.assert_array_equal(left.fire_loc, right.fire_loc)
    np.testing.assert_array_equal(left.ign_points_all, right.ign_points_all)
    np.testing.assert_array_equal(left.ign_points_all[:, :2], [cell])


def test_softrole_masks_stay_without_changing_six_head_parameters_or_rng():
    config = recipe("fc", env_version="paper-v1")
    torch.manual_seed(9)
    paper = SoftRoleNet(**config.model_kwargs()).double()
    paper_rng = torch.get_rng_state()
    torch.manual_seed(9)
    old = SoftRoleNet(**replace(config, env_version=ENVIRONMENT_VERSION).model_kwargs()).double()
    assert torch.equal(paper_rng, torch.get_rng_state())
    assert paper.state_dict().keys() == old.state_dict().keys()
    for name, value in paper.state_dict().items():
        assert torch.equal(value, old.state_dict()[name])
    adapter = make_env(config)
    obs, kappa = adapter.reset(9)
    output = paper(torch.from_numpy(obs), torch.from_numpy(kappa), paper.initial_memory(3),
                   torch.ones(3, 3, dtype=torch.bool), 1., noise=[torch.zeros(3, 16)] * 2)
    probabilities = output["logits"].softmax(-1)
    assert torch.equal(probabilities[:, 4], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(probabilities[:2, 5], torch.zeros(2, dtype=torch.float64))
    assert probabilities[2, 5] > 0 and paper.output.out_features == 6
    with pytest.raises(ValueError, match="stay"):
        adapter.step(np.array([0, 0, 4]), kappa)


def test_softrole_and_isolated_publication_fc_have_identical_physical_replay():
    actions = [[0, 2, 5], [1, 3, 0], [2, 0, 1], [3, 1, 5], [0, 2, 2], [1, 3, 3]]
    runtime = ROOT / "publication_reconstruction/runtime"
    code = f"""
import sys, json, argparse, numpy as np
sys.path[:0] = [{str(runtime)!r}, {str(runtime / 'envs')!r}]
from ic3net_envs.fire_commander_env import FireCommanderEnv
env = FireCommanderEnv()
parser = argparse.ArgumentParser()
env.init_args(parser)
args = parser.parse_args([])
args.publication_env_version='paper-v1'
args.max_steps=300
args.vision=1
env.multi_agent_init(args)
np.random.seed(81)
initial=env.reset().tolist()
records=[]
for actions in {actions!r}:
    actions=[4 if a==5 else a for a in actions]
    obs, reward, done, info=env.step(actions)
    records.append([obs.tolist(), reward.tolist(), bool(done), env.fire_loc.tolist(), env.ign_points_all.tolist(), info['paper_events']])
    if done: break
print(json.dumps([initial, records]))
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    initial, expected = json.loads(result.stdout)
    adapter = make_env(recipe("fc", env_version="paper-v1"))
    _, kappa = adapter.reset(81)
    np.testing.assert_array_equal(adapter.raw_observation, initial)
    for action, record in zip(actions, expected):
        _, reward, done, info = adapter.step(np.array(action), kappa)
        np.testing.assert_array_equal(adapter.raw_observation, record[0])
        np.testing.assert_array_equal(reward, record[1])
        assert done == record[2]
        np.testing.assert_array_equal(adapter.raw.fire_loc, record[3])
        np.testing.assert_array_equal(adapter.raw.ign_points_all, record[4])
        assert info["paper_events"] == record[5]


def copy_simulator(tmp_path):
    for name in SIMULATOR_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return source_binding(tmp_path)


def test_loader_uses_verified_archive_and_ignores_other_wildfire_imports(tmp_path, monkeypatch):
    binding = copy_simulator(tmp_path)
    monkeypatch.setitem(sys.modules, "WildFire_Simulate_Original", SimpleNamespace(WildFire=None))
    cls = load_environment_class("fc", binding)
    assert cls._fire_init.__globals__["WildFire"] is not None
    assert cls._fire_init.__code__.co_filename.startswith(str(tmp_path))
    with (tmp_path / SIMULATOR_FILES[0]).open("a") as stream:
        stream.write("\nraise RuntimeError('must not execute unverified source')\n")
    with pytest.raises(ValueError, match="changed after binding"):
        load_environment_class("fc", binding)  # Even an already-cached class is reverified.


def tiny_paper_config(**options):
    settings = dict(env_version="paper-v1", model="shared", max_steps=2,
                    pre_dim=8, hidden_dim=8, heads=1, head_dim=4, msg_dim=4,
                    epochs=1, updates_per_epoch=1, batch_steps=1, nprocesses=1,
                    save_every=1)
    settings.update(options)
    return recipe("fc", **settings)


@pytest.mark.parametrize("collectors", [1, 2])
def test_paper_softrole_checkpoint_uses_archive_for_evaluation_and_resume(tmp_path, collectors):
    from softrole.train import train
    from softrole.evaluate import evaluate_checkpoint
    config = tiny_paper_config(nprocesses=collectors)
    checkpoint = train(config, tmp_path / "trained")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["environment_version"] == "paper-v1"
    assert saved["simulator_source"] == simulator_identity()
    binding = checkpoint_binding(checkpoint, saved)
    assert binding["root"] == str(tmp_path / "trained/source")
    report = evaluate_checkpoint(checkpoint, make_scenarios(11, 1, [(2, 1)]), tmp_path / "evaluation.json")
    assert report["environment_version"] == "paper-v1"
    assert report["simulator_source"] == saved["simulator_source"]
    resumed = train(replace(config, epochs=2), tmp_path / "resumed", resume=checkpoint)
    resumed_state = torch.load(resumed, map_location="cpu", weights_only=False)
    assert resumed_state["updates"] == 2
    uninterrupted = train(replace(config, epochs=2), tmp_path / "uninterrupted")
    full_state = torch.load(uninterrupted, map_location="cpu", weights_only=False)
    for name, value in full_state["model_state"].items():
        assert torch.equal(value, resumed_state["model_state"][name])
    assert full_state["optimizer_state"]["param_groups"] == resumed_state["optimizer_state"]["param_groups"]
    for parameter, state in full_state["optimizer_state"]["state"].items():
        for name, value in state.items():
            actual = resumed_state["optimizer_state"]["state"][parameter][name]
            assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
    with (Path(binding["root"]) / SIMULATOR_FILES[1]).open("a") as stream:
        stream.write("\n# altered archive\n")
    with pytest.raises(ValueError, match="source bytes"):
        evaluate_checkpoint(checkpoint, make_scenarios(11, 1, [(2, 1)]), tmp_path / "rejected.json")
    assert not (tmp_path / "rejected.json").exists()
    with pytest.raises(ValueError, match="source bytes"):
        train(replace(config, epochs=2), tmp_path / "rejected_resume", resume=checkpoint)
    assert not (tmp_path / "rejected_resume").exists()


def test_old_missing_environment_field_keeps_legacy_model_layout():
    original = recipe("fc").to_dict()
    original.pop("env_version")
    restored = Config(**original)
    assert restored.env_version == ENVIRONMENT_VERSION
    assert "allow_stay" not in restored.model_kwargs()
    assert make_env(restored).raw.__class__.__module__ == "ic3net_envs.fire_commander_env"


def test_old_checkpoint_without_env_config_field_still_evaluates(tmp_path):
    from softrole.evaluate import evaluate_checkpoint
    from softrole import CHECKPOINT_VERSION
    config = recipe("fc", max_steps=2, model="shared")
    saved_config = config.to_dict()
    saved_config.pop("env_version")
    model = SoftRoleNet(**config.model_kwargs()).double()
    checkpoint = tmp_path / "old.pt"
    torch.save({"format_version": CHECKPOINT_VERSION, "config": saved_config,
                "model_config": config.model_kwargs(), "model_state": model.state_dict(),
                "environment_version": ENVIRONMENT_VERSION}, checkpoint)
    report = evaluate_checkpoint(checkpoint, make_scenarios(21, 1, [(2, 1)]), tmp_path / "old.json")
    assert report["environment_version"] == ENVIRONMENT_VERSION
    assert report["simulator_source"] is None
