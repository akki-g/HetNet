"""Bounded behavior probes against February 2022 source snapshots.

Run from repository root with .venv/bin/python. Only np.int=int compatibility
alias and injected initial coordinates are used; no training is performed.
FC's historical reset fails, so downstream FC function probes manually establish
the initial state and explicitly report that intervention. Snapshot files remain
unmodified. Full FC steps are not claimed: historical NumPy containment behavior
in _get_obs is also incompatible with the installed NumPy version.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "59d598"
np.int = int  # NumPy 1.26 compatibility for historical alias, no numerical change.
sys.path.insert(0, str(SNAPSHOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fc_path = SNAPSHOT / "envs/ic3net_envs/fire_commander_env.py"
pcp_path = SNAPSHOT / "envs/ic3net_envs/predator_capture_env.py"
fc_module = load("historical_fc", fc_path)
FC = fc_module.FireCommanderEnv
PCP = load("historical_pcp", pcp_path).PredatorCaptureEnv


def make(cls, points, vision, inject_hotspot=False):
    env = cls()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", default=300)
    parser.add_argument("--eval", default=False)
    env.init_args(parser)
    args = parser.parse_args([])
    args.vision = vision
    env.multi_agent_init(args)
    coordinate_method = "_get_coordinates" if cls is FC else "_get_cordinates"
    setattr(env, coordinate_method, lambda: np.array(points, dtype=int))
    if inject_hotspot:
        env.predator_loc = np.array(points[:2], dtype=int)
        env.predator_capture_loc = np.array(points[2:3], dtype=int)
        env.fire_loc = np.array(points[3:], dtype=int)
        env.fire_mdl = fc_module.WildFire(terrain_sizes=[5, 5], hotspot_areas=[], num_ign_points=1, duration=300)
        env.ign_points_all = np.array([points[-1] + [1.]], dtype=float)
        env.previous_terrain_map = env.ign_points_all.copy()
        env.geo_phys_info = env.fire_mdl.geo_phys_info_init(avg_wind_speed=0, avg_wind_direction=0)
        env.pruned_list = []
        env.discovered_fire = []
        env.fire_out = False
        env.false_water_drop = np.zeros(1)
        env.fire_extinguished = np.zeros(1)
        env.extinguishing = np.zeros(1)
        env.just_discovered_source = np.zeros(2)
        env.just_discovered_nonsource = np.zeros(2)
        env.stat = {"enemy_count": []}
        env._set_grid()
    else:
        env.reset()
    return env


def main():
    np.random.seed(501)
    findings = {}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            make(FC, [[4, 4], [3, 4], [4, 3], [0, 2]], 1)
        except ValueError as error:
            findings["unmodified_fc_reset"] = {"error": str(error), "type": type(error).__name__,
                "reason": "Wrapper supplies [x,x,y,y] hotspot bounds; hotspot_init uses randint(low=x, high=x), whose upper bound is exclusive"}
        else:
            raise AssertionError("Historical reset unexpectedly succeeded")

        pcp = make(PCP, [[2, 1], [4, 4], [2, 2], [2, 2]], 2)
        blind = pcp._get_obs()[0, ..., pcp.PREY_CLASS].copy()
        pcp.action_blind = False
        pcp.A_agents_have_vision = False
        sighted = pcp._get_obs()[0, ..., pcp.PREY_CLASS].copy()
        assert sighted.sum() == 1 and blind.max() == 0 and (blind == -1).any()
        findings["pcp_view_aliasing"] = {
            "same_physical_state_target_count_without_blind_mask": int(sighted.sum()),
            "visible_target_after_blind_mask": bool((blind > 0).any()),
            "negative_mask_entries_in_sensing_agent_view": int((blind == -1).sum()),
        }

        boundary_examples = []
        for r in range(5):
            for c in range(5):
                env = make(FC, [[4, 4], [3, 4], [4, 3], [r, c]], 1, inject_hotspot=True)
                env._fire_propagation()
                if r == 0 or c == 0:
                    assert env.ign_points_all.tolist() == [[0., 0., 0.]]
                    boundary_examples.append({"initial": [r, c], "fires": env.fire_loc.tolist()})
        phantom_count = sum(e["initial"] != [0, 0] and [0, 0] in e["fires"]
                            for e in boundary_examples)
        assert phantom_count == 8
        findings["phantom_fire"] = {"new_origin_fires_of_25_initial_cells": phantom_count,
                                    "boundary_examples": boundary_examples}

        env = make(FC, [[4, 4], [3, 4], [0, 0], [0, 2]], 1, inject_hotspot=True)
        env._fire_propagation()
        env._take_action(2, 5)
        env._fire_propagation()
        reward = env._get_reward()
        done = env.episode_over
        assert reward.tolist() == [-.1, -.1, 10.] and not done
        assert env.fire_loc.tolist() == [[0, 2]]
        findings["phantom_capture"] = {"reward": reward.tolist(), "team_sum": float(reward.sum()),
                                       "done": bool(done), "remaining_fire": env.fire_loc.tolist()}

        env = make(FC, [[4, 4], [3, 4], [0, 2], [0, 2]], 1, inject_hotspot=True)
        assert len(env.discovered_fire) == 0
        env._take_action(2, 5)
        env._fire_propagation()
        reward = env._get_reward()
        done = env.episode_over
        assert done and env.nfire == 0
        findings["undiscovered_final_fire_capture"] = {
            "discovered_before": False, "done": bool(done), "remaining_fire_count": int(env.nfire),
            "reward": reward.tolist(),
        }

    findings["source_sha256"] = {
        str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (fc_path, pcp_path, SNAPSHOT / "WildFire_Simulate_Original.py")
    }
    findings["source_commit"] = "59d598319d6285acfa42a1d9d8c859305a9bbf1e"
    findings["scope"] = "February 14 2022 source files unmodified; NumPy np.int compatibility alias; injected initial coordinates; FC reset tested as-is, downstream FC function probes manually establish initial state with [x,y,intensity=1] front and zero wind. FC _get_obs/full steps bypassed due additional old NumPy containment incompatibility; no training."
    out = HERE / "historical_probe.json"
    with out.open("x") as stream:
        json.dump(findings, stream, indent=2)
        stream.write("\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
