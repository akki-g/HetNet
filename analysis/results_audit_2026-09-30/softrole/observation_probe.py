"""Quantify preserved legacy observation aliasing on a fixed reset panel."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from softrole.config import recipe
from softrole.env import make_env

results = {}
for task in ("pp", "pcp", "fc"):
    config = recipe(task)
    corrected, legacy = make_env(config), make_env(config)
    legacy.raw.independent_observations = False
    row = dict(reset_seeds=1000, sensing_views=0, differing_sensing_views=0,
               corrected_target_visible=0, legacy_target_visible=0,
               visible_target_erased=0, lost_target_frames=0)
    for seed in range(1000):
        corrected.reset(seed)
        legacy.reset(seed)
        fresh, old = corrected.raw_observation[:config.num_p], legacy.raw_observation[:config.num_p]
        different = (fresh != old).reshape(config.num_p, -1).any(1)
        fresh_target = (fresh[..., corrected.base + 2] > 0).reshape(config.num_p, -1).any(1)
        old_target = (old[..., legacy.base + 2] > 0).reshape(config.num_p, -1).any(1)
        row["sensing_views"] += config.num_p
        row["differing_sensing_views"] += int(different.sum())
        row["corrected_target_visible"] += int(fresh_target.sum())
        row["legacy_target_visible"] += int(old_target.sum())
        row["visible_target_erased"] += int((fresh_target & ~old_target).sum())
        row["lost_target_frames"] += int(np.any(fresh_target & ~old_target))
    # Changing the copy path must preserve dynamics/rewards under matched actions.
    transitions = 0
    for seed in range(30):
        _, cap = corrected.reset(seed)
        legacy.reset(seed)
        rng = np.random.default_rng(seed + 93001)
        for step in range(config.max_steps):
            actions = np.array([rng.integers(6 if act else 5) for act in cap[:, 1]])
            _, cr, cd, ci = corrected.step(actions, cap)
            _, lr, ld, li = legacy.step(actions, cap)
            np.testing.assert_array_equal(corrected.positions, legacy.positions)
            np.testing.assert_array_equal(cr, lr)
            assert cd == ld
            assert ci.keys() == li.keys()
            for key in ci:
                np.testing.assert_array_equal(ci[key], li[key])
            transitions += 1
            if cd:
                break
    row["matched_action_episodes"] = 30
    row["identical_physics_reward_transitions"] = transitions
    results[task] = row
with (Path(__file__).parent / "observation_results.json").open("x") as stream:
    json.dump(results, stream, indent=2)
    stream.write("\n")
print(json.dumps(results, indent=2))
