"""Independent bounded probes of current FC physics/observation behavior.

Run from repository root: .venv/bin/python analysis/results_audit_2026-09-30/fc/confirm_fc.py
Writes only a fresh adjacent findings.json; never changes runtime code.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from softrole.config import recipe
from softrole.env import make_env


def fixed(fire):
    env = make_env(recipe("fc"))
    env.reset(501)
    candidates = [[4, 4], [3, 4], [4, 3], [3, 3]]
    positions = [p for p in candidates if p != fire][:3] + [fire]
    with env._rng.use():
        env._raw_obs = np.array(env.raw.reset(positions), copy=True)
    return env


def main():
    # Each cell is checked using the public adapter step. No simulator replacement.
    boundaries = []
    for row in range(5):
        for col in range(5):
            env = fixed([row, col])
            _, _, _, _ = env.step(np.array([4, 4, 4]), env.kappa)
            if row == 0 or col == 0:
                assert env.raw.ign_points_all.tolist() == [[0., 0., 0.]]
                boundaries.append({"initial": [row, col], "fires_after_stay": env.raw.fire_loc.tolist(),
                                   "front_after_stay": env.raw.ign_points_all.tolist()})
    assert len(boundaries) == 9
    phantom_additions = sum(r["initial"] != [0, 0] and [0, 0] in r["fires_after_stay"] for r in boundaries)
    assert phantom_additions == 8

    env = fixed([0, 2])
    env.raw.predator_capture_loc[:] = [0, 0]
    env.step(np.array([4, 4, 4]), env.kappa)
    _, rewards, done, info = env.step(np.array([4, 4, 5]), env.kappa)
    assert rewards.tolist() == [-.1, -.1, 10.] and not done
    assert env.raw.fire_loc.tolist() == [[0, 2]]
    assert len(env.raw.ign_points_all) == 0
    phantom_reward = {"physical_fire": [0, 2], "water_drop": [0, 0], "agent_rewards": rewards.tolist(),
                      "team_reward": float(rewards.sum()), "done": done,
                      "residual_fires": env.raw.fire_loc.tolist()}

    env = fixed([2, 2])
    env.raw.predator_loc[:] = [[2, 1], [4, 4]]
    env.raw.predator_capture_loc[:] = [2, 2]
    env.raw.independent_observations = False
    legacy = env.raw._get_obs()
    env.raw.independent_observations = True
    corrected = env.raw._get_obs()
    legacy_count = int(legacy[0, ..., env.raw.FIRE_CLASS].sum())
    corrected_count = int(corrected[0, ..., env.raw.FIRE_CLASS].sum())
    assert (legacy_count, corrected_count) == (0, 1)

    # Capture final source immediately, checking actual returned termination.
    env = fixed([2, 2])
    env.raw.predator_capture_loc[:] = [2, 2]
    _, rewards, done, info = env.step(np.array([4, 4, 5]), env.kappa)
    assert done and info["success"] and info["terminated"] and env.raw.nfire == 0

    source_names = ["WildFire_Simulate_Original.py", "envs/ic3net_envs/fire_commander_env.py",
                    "softrole/env.py", "softrole/config.py"]
    current = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_names}
    archive_matches = []
    for model in ["shared", "banked"]:
        for seed in range(3):
            run = ROOT / "stokes_runs/runs/softrole_primary" / f"fc_{model}" / f"seed{seed}"
            matches = {p: hashlib.sha256((run / "source" / p).read_bytes()).hexdigest() == current[p]
                       for p in source_names[:2]}
            assert all(matches.values())
            archive_matches.append({"model": model, "seed": seed, "physics_matches_current": matches})
    findings = {"scope": "Current default FC, 5x5, 2P1A, vision1, reward3; deterministic diagnostic only",
                "boundary_examples": boundaries, "new_phantom_origin_fires_of_25_initial_cells": phantom_additions,
                "phantom_source_capture": phantom_reward,
                "observation_aliasing": {"legacy_sensor_fire_count": legacy_count,
                                         "softrole_sensor_fire_count": corrected_count},
                "final_fire_capture": {"done": done, "success": info["success"],
                                       "rewards": rewards.tolist()},
                "physics_source_matches_six_archived_runs": archive_matches,
                "source_sha256": current,
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = Path(__file__).with_name("findings.json")
    with output.open("x") as handle:
        json.dump(findings, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "phantom_additions": phantom_additions,
                      "phantom_reward": phantom_reward, "legacy_count": legacy_count,
                      "corrected_count": corrected_count, "archived_fc_runs_matching_physics": len(archive_matches)}, indent=2))


if __name__ == "__main__":
    main()
