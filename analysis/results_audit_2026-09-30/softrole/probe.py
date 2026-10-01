"""Read-only diagnostic audit, fixed nominal panel; no training or selection by outcome."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from softrole.config import Config, recipe
from softrole.env import make_env
from softrole.model import SoftRoleNet
from softrole.rollout import message_noise, run_episode
from softrole.scenarios import make_scenarios

DEST = Path(__file__).parent
torch.set_num_threads(1)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    result = {"panel_seed": 93001, "episodes_per_condition": 12,
              "selection": "Latest locally archived seed-0 checkpoint by epoch, selected before outcomes",
              "limits": "Small nominal diagnostic panel; not held-out, not a ranking or final performance estimate"}
    source = {}
    paths = ["softrole/model.py", "softrole/learning.py", "softrole/scenarios.py",
             "softrole/rollout.py", "softrole/env.py", "softrole/train.py",
             "envs/ic3net_envs/predator_capture_env.py", "envs/ic3net_envs/fire_commander_env.py"]
    for run in sorted((ROOT / "stokes_runs/runs/softrole_primary").glob("*/seed*")):
        manifest = json.loads((run / "source_manifest.json").read_text())["files"]
        source[str(run.relative_to(ROOT))] = {
            p: {"matches_current": sha(run / "source" / p) == sha(ROOT / p),
                "matches_manifest": sha(run / "source" / p) == manifest[p]}
            for p in paths}
    result["archived_source"] = source

    # Actor cannot read another agent's input without an allowed message path;
    # fresh current communication randomness must not enter the pre-action critic.
    model = SoftRoleNet(**recipe().model_kwargs()).double()
    adapter = make_env(recipe())
    obs, cap = adapter.reset(100)
    obs, cap = torch.as_tensor(obs), torch.as_tensor(cap)
    memory = model.initial_memory(3)
    empty, full = torch.zeros(3, 3, dtype=torch.bool), ~torch.eye(3, dtype=torch.bool)
    noise = message_noise(1, 0, 3, 16, torch.float64)
    original = model(obs, cap, memory, empty, 1, noise)
    changed_obs, changed_cap = obs.clone(), cap.clone()
    changed_obs[1:] = 200
    changed_cap[1:] = 0
    changed = model(changed_obs, changed_cap, memory, empty, .3, noise)
    assert torch.equal(original["logits"][0], changed["logits"][0])
    assert torch.equal(original["gate"][0], changed["gate"][0])
    first = model(obs, cap, memory, full, 1, noise)
    second = model(obs, cap, memory, full, 1, message_noise(2, 0, 3, 16, torch.float64))
    assert torch.equal(first["value"], second["value"])
    assert torch.equal(first["gate"], second["gate"])
    assert not torch.equal(first["logits"], second["logits"])
    result["information_probes"] = {"empty_graph_local_actor_unchanged": True,
        "current_message_noise_changes_actor_but_not_critic_or_gate": True}

    conditions = []
    for task in ("pp", "pcp", "fc"):
        for mode in ("shared", "banked"):
            run = ROOT / f"stokes_runs/runs/softrole_primary/{task}_{mode}/seed0"
            path = sorted((run / "checkpoints").glob("epoch*.pt"))[-1]
            checkpoint_hash = sha(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            config = Config(**saved["config"])
            scenarios = make_scenarios(93001, 12, config.training_compositions)
            for trained in (False, True):
                torch.manual_seed(config.seed)
                model = SoftRoleNet(**saved["model_config"]).double()
                if trained:
                    model.load_state_dict(saved["model_state"], strict=True)
                model.eval()
                adapter = make_env(config)
                records = []
                for scenario in scenarios:
                    episode = run_episode(model, adapter, config, scenario, training=False, trace=True)
                    # Replay only logged actions in a new simulator; independently
                    # aggregate rewards and inspect physical final-state success.
                    replay = make_env(config)
                    _, cap = replay.reset(scenario.env_seed)
                    returns = np.zeros(len(cap))
                    for index, entry in enumerate(episode.traces):
                        _, reward, done, info = replay.step(np.array(entry["actions"]), cap)
                        returns += reward
                        assert np.isclose(reward.sum(), entry["team_reward"])
                        assert not done or index == len(episode.traces) - 1
                    if task in ("pp", "pcp"):
                        physical_success = bool(np.all(replay.positions == replay.raw.prey_loc[0])
                                                and np.all(replay.raw.captured_prey == 1))
                    else:
                        physical_success = len(replay.raw.fire_loc) == 0
                    assert done
                    assert physical_success == episode.metrics["success"]
                    assert len(episode.traces) == episode.metrics["steps"]
                    np.testing.assert_allclose(returns, episode.metrics["agent_returns"], rtol=0, atol=1e-12)
                    assert np.isclose(returns.sum(), episode.metrics["team_return"])
                    records.append({key: episode.metrics[key] for key in
                                    ("scenario_id", "env_seed", "action_seed", "message_seed", "success", "steps", "team_return")})
                condition = {"task": task, "model": mode, "trained": trained,
                    "checkpoint": str(path.relative_to(ROOT)) if trained else None,
                    "checkpoint_sha256": checkpoint_hash if trained else None,
                    "checkpoint_epoch": saved["epoch"] if trained else 0,
                    "checkpoint_steps": saved["total_steps"] if trained else 0,
                    "successes": sum(row["success"] for row in records),
                    "mean_steps": float(np.mean([row["steps"] for row in records])),
                    "mean_team_return": float(np.mean([row["team_return"] for row in records])),
                    "independent_replay_passed": True, "episodes": records}
                conditions.append(condition)
                print(json.dumps({k: v for k, v in condition.items() if k != "episodes"}), flush=True)
            assert sha(path) == checkpoint_hash
    result["conditions"] = conditions
    result["probe_sha256"] = sha(Path(__file__))
    result["current_source_sha256"] = {p: sha(ROOT / p) for p in paths}
    with (DEST / "probe_results.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")

if __name__ == "__main__":
    main()
