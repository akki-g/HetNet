"""Read-only FC audit: bounded diagnostic panels; never edits simulator code.

Run from the repository root with .venv/bin/python and an explicit fresh output:
  .venv/bin/python analysis/fc_audit_2026-09-30/simulator_probe.py --output PATH
Policies here are diagnostics, not learned-policy performance estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from softrole.config import recipe
from softrole.env import make_env


def fixed_env(fire, seed=100):
    env = make_env(recipe("fc"))
    env.reset(seed)
    positions = [list(p) for p in ((4, 4), (4, 3), (3, 4), (3, 3))
                 if list(p) != list(fire)][:3]
    with env._rng.use():
        env._raw_obs = env.raw.reset(positions + [list(fire)])
    return env


def snapshot(raw):
    return {"fire_cells": raw.fire_loc.tolist(),
            "front": raw.ign_points_all.tolist(), "nfire": int(raw.nfire)}


def deterministic_checks():
    boundary = []
    for x in range(5):
        for y in range(5):
            env = fixed_env((x, y))
            before = snapshot(env.raw)
            env.step(np.array([4, 4, 4]), env.kappa)
            if x == 0 or y == 0:
                assert np.array_equal(env.raw.ign_points_all, [[0., 0., 0.]])
                boundary.append({"before": before, "after": snapshot(env.raw)})
    assert len(boundary) == 9
    added = sum([0, 0] not in entry["before"]["fire_cells"] and
                [0, 0] in entry["after"]["fire_cells"] for entry in boundary)
    assert added == 8

    env = fixed_env((0, 2))
    env.raw.predator_capture_loc[:] = (0, 0)
    env.step(np.array([4, 4, 4]), env.kappa)
    before = snapshot(env.raw)
    _, reward, done, info = env.step(np.array([4, 4, 5]), env.kappa)
    after = snapshot(env.raw)
    assert np.array_equal(reward, [-.1, -.1, 10.])
    assert not done and after["fire_cells"] == [[0, 2]] and not after["front"]
    ghost_reward = {"before": before, "after": after,
                    "reward": reward.tolist(), "done": bool(done)}

    movement = []
    for idx in (0, 2):
        for x in range(5):
            for y in range(5):
                for action, delta in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                    env.raw.predator_loc[0] = (x, y)
                    env.raw.predator_capture_loc[0] = (x, y)
                    env.raw._take_action(idx, action)
                    result = (env.raw.predator_loc[0] if idx == 0
                              else env.raw.predator_capture_loc[0])
                    expected = np.clip(np.array((x, y)) + delta, 0, 4)
                    if not np.array_equal(result, expected):
                        movement.append({"agent": idx, "position": [x, y], "action": action})
    assert not movement

    env = fixed_env((2, 2))
    raw = env.raw
    raw.predator_loc[:] = ((2, 1), (4, 4))
    raw.predator_capture_loc[:] = (2, 2)
    raw.independent_observations = False
    legacy = raw._get_obs()
    raw.independent_observations = True
    corrected = raw._get_obs()
    observed = {"legacy_P0_fire_count": int(legacy[0, ..., raw.FIRE_CLASS].sum()),
                "independent_P0_fire_count": int(corrected[0, ..., raw.FIRE_CLASS].sum())}
    assert observed == {"legacy_P0_fire_count": 0, "independent_P0_fire_count": 1}

    # Zero motion fixes the front at (1.6,2.0). Physics samples rounded (2,2),
    # whereas observation and extinguishing use truncated/floored (1,2).
    env = fixed_env((1, 2))
    raw = env.raw
    raw.ign_points_all[:] = [[1.6, 2., 1.]]
    raw.geo_phys_info["spread_rate"][:] = 0
    raw.geo_phys_info["spread_rate"][2, 2] = 1.25
    raw.geo_phys_info["wind_speed"][:] = 0
    raw.geo_phys_info["wind_direction"][:] = 0
    with env._rng.use():
        propagated, physics = raw.fire_mdl.fire_propagation(
            5, raw.ign_points_all, raw.geo_phys_info, raw.previous_terrain_map, [])
    assert physics[0, 0] == 1.25
    round_floor = {"front": [1.6, 2.], "display_capture_cell": [1, 2],
                   "fuel_lookup_cell": [2, 2], "sampled_fuel": float(physics[0, 0])}

    # An already extinguished tile is not excluded as a destination. A moving
    # front can enter it, re-add it, and then become stationary there.
    env = fixed_env((2, 2))
    raw = env.raw
    raw.pruned_list = [[2, 1]]
    raw.geo_phys_info["spread_rate"][:] = 2
    raw.geo_phys_info["wind_speed"][:] = 1
    raw.geo_phys_info["wind_direction"][:] = np.pi
    before = snapshot(raw)
    env.step(np.array([4, 4, 4]), env.kappa)
    readded = snapshot(raw)
    first_front = raw.ign_points_all.copy()
    env.step(np.array([4, 4, 4]), env.kappa)
    stalled = snapshot(raw)
    assert [2, 1] in raw.fire_loc.tolist()
    np.testing.assert_array_equal(raw.ign_points_all[:, :2], first_front[:, :2])

    # Membership is coordinate-wise, not row-wise, in source discovery only.
    env = fixed_env((2, 2))
    raw = env.raw
    raw.fire_loc = np.array([[2, 2], [1, 2]])
    raw.predator_loc[:] = ((0, 2), (4, 4))
    raw.discovered_fire = []
    raw.just_discovered_source[:] = 0
    raw.just_discovered_nonsource[:] = 0
    raw._get_obs()
    assert raw.just_discovered_source[0] == 1
    discovery = {"observed_fire": [1, 2], "true_source": [2, 2],
                 "marked_source": bool(raw.just_discovered_source[0]),
                 "source_discovery_reward": raw.DISCOVER_SOURCE_REWARD,
                 "nonsource_discovery_reward": raw.DISCOVER_NONSOURCE_REWARD}

    # Capture the only source immediately: propagation cannot resurrect it and
    # native termination agrees with adapter success.
    env = fixed_env((2, 2))
    env.raw.predator_capture_loc[:] = (2, 2)
    _, reward, done, info = env.step(np.array([4, 4, 5]), env.kappa)
    assert done and info["success"] and env.raw.nfire == 0
    return {"boundary_cases": boundary, "immediate_origin_additions_of_25_cells": added,
            "ghost_source_reward": ghost_reward, "movement_cases_checked": 200,
            "movement_mismatches": movement, "legacy_aliasing": observed,
            "round_floor": round_floor,
            "pruned_reignition": {"pruned_cell": [2, 1], "before": before,
                                   "after_reentry": readded, "after_stall": stalled},
            "discovery_membership": discovery,
            "immediate_source_extinction": {"reward": reward.tolist(), "done": bool(done),
                                              "success": info["success"]}}


def passive_panel(count):
    rows = []
    wind = []
    for seed in range(count):
        env = make_env(recipe("fc"))
        env.reset(seed)
        raw = env.raw
        wind.extend(raw.geo_phys_info["wind_speed"].ravel().tolist())
        initial = raw.fire_loc[0].tolist()
        origin_at = None
        for step in range(1, 301):
            with env._rng.use():
                raw._fire_propagation()
            if np.array_equal(raw.ign_points_all, [[0., 0., 0.]]):
                origin_at = step
                break
        rows.append({"seed": seed, "initial_fire": initial,
                     "zero_front_at_step": origin_at, "final_fire_count": int(raw.nfire)})
    hit = [r["zero_front_at_step"] for r in rows if r["zero_front_at_step"] is not None]
    wind = np.asarray(wind)
    return {"count": count, "origin_placeholder_by_300": len(hit),
            "origin_placeholder_by_10": sum(t <= 10 for t in hit),
            "origin_time_quantiles": np.quantile(hit, [0, .25, .5, .75, 1]).tolist(),
            "final_fire_count_quantiles": np.quantile(
                [r["final_fire_count"] for r in rows], [0, .25, .5, .75, 1]).tolist(),
            "wind_samples": len(wind), "wind_negative": int((wind < 0).sum()),
            "wind_above_configured_maximum_1": int((wind > 1).sum()),
            "wind_min": float(wind.min()), "wind_max": float(wind.max()), "episodes": rows}


def greedy_action(raw):
    loc = raw.predator_capture_loc[0]
    # Privileged full-state: pursue the current source first, then nearest
    # residual fire. This is not available information for a native A agent.
    targets = raw.ign_points_all[:, :2].astype(int)
    if not len(targets):
        targets = raw.fire_loc
    target = targets[np.abs(targets - loc).sum(axis=1).argmin()]
    if np.array_equal(loc, target):
        return 5
    if loc[0] != target[0]:
        return 2 if loc[0] < target[0] else 0
    return 1 if loc[1] < target[1] else 3


def policy_panel(count):
    results = {}
    for name in ("random", "privileged_source_first"):
        rows = []
        for seed in range(count):
            env = make_env(recipe("fc"))
            _, kappa = env.reset(seed)
            action_rng = np.random.default_rng(np.random.SeedSequence([99001, seed]))
            reward_sum, source_hits, residual_hits = 0., 0, 0
            for step in range(1, 301):
                if name == "random":
                    actions = np.array([action_rng.integers(5), action_rng.integers(5),
                                        action_rng.integers(6)])
                else:
                    actions = np.array([4, 4, greedy_action(env.raw)])
                _, reward, done, info = env.step(actions, kappa)
                reward_sum += float(reward.sum())
                source_hits += int(env.raw.fire_extinguished[0] == 2)
                residual_hits += int(env.raw.fire_extinguished[0] == 1)
                if done:
                    break
            rows.append({"seed": seed, "success": info["success"], "steps": step,
                         "team_return": reward_sum, "source_extinctions": source_hits,
                         "residual_extinctions": residual_hits,
                         "remaining_fires": int(env.raw.nfire)})
        results[name] = {"episodes": rows, "count": count,
                         "success_count": sum(r["success"] for r in rows),
                         "mean_steps": float(np.mean([r["steps"] for r in rows])),
                         "mean_team_return": float(np.mean([r["team_return"] for r in rows])),
                         "max_steps": max(r["steps"] for r in rows)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passive-count", type=int, default=1000)
    parser.add_argument("--policy-count", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = [Path(__file__).resolve(), ROOT / "envs/ic3net_envs/fire_commander_env.py",
              ROOT / "WildFire_Simulate_Original.py", ROOT / "softrole/env.py",
              ROOT / "softrole/scenarios.py", ROOT / "softrole/config.py"]
    report = {"scope": "Current default 5x5 FC, 2P1A, vision 1, one initial fire; diagnostic only",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source}, "numpy_version": np.__version__,
              "deterministic_checks": deterministic_checks(),
              "passive_panel": passive_panel(args.passive_count),
              "policy_panel": policy_panel(args.policy_count)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output),
                      "passive": {k: v for k, v in report["passive_panel"].items() if k != "episodes"},
                      "policies": {name: {k: v for k, v in value.items() if k != "episodes"}
                                   for name, value in report["policy_panel"].items()}}, indent=2))


if __name__ == "__main__":
    main()
