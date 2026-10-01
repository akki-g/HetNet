"""Bounded FC actor/critic gradient diagnostic; no optimization or mutation.

Run from the repository root with .venv/bin/python. Replays the first successful
and first failed trajectory in the existing shared-seed-0 24-scenario panel.
The outcomes are selected for contrasting diagnostics, not statistical inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from softrole.config import Config
from softrole.env import make_env
from softrole.learning import compute_episode_loss, team_gae
from softrole.model import SoftRoleNet
from softrole.rollout import run_episode
from softrole.scenarios import make_scenarios


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(tensors):
    return float(sum(tensor.square().sum() for tensor in tensors).sqrt())


def main():
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=directory / "learner_findings.json")
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError(output)
    panel_path = directory / "policy_panel/shared_seed0.json"
    panel = json.loads(panel_path.read_text())
    checkpoint_path = ROOT / panel["summary"]["checkpoint"]
    source_paths = [
        "softrole/config.py", "softrole/model.py", "softrole/learning.py",
        "softrole/scenarios.py", "softrole/env.py", "softrole/rollout.py",
        "envs/ic3net_envs/fire_commander_env.py", "WildFire_Simulate_Original.py",
    ]
    tracked_paths = [ROOT / name for name in source_paths]
    tracked_paths += [Path(__file__).resolve(), checkpoint_path, panel_path]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in tracked_paths}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert before[str(checkpoint_path.relative_to(ROOT))] == panel["summary"]["checkpoint_sha256"]
    config = Config(**checkpoint["config"])
    model = SoftRoleNet(**checkpoint["model_config"]).double()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.train()
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    scenarios = make_scenarios(260930, 24, [(2, 1)])
    indices = [next(index for index, entry in enumerate(panel["episodes"])
                    if entry["metrics"]["success"] == success)
               for success in (True, False)]
    parameters = list(model.named_parameters())
    tensors = [tensor for _, tensor in parameters]
    shared_indices = [index for index, (name, _) in enumerate(parameters)
                      if name.startswith(("pre.", "lstm."))]
    results = []
    for index in indices:
        episode = run_episode(model, make_env(config), config, scenarios[index], training=True)
        expected = panel["episodes"][index]["metrics"]
        assert episode.metrics == expected, (index, "trajectory differs from frozen panel")
        values = torch.stack(episode.values)
        scores = torch.stack(episode.log_probs)
        advantages, targets = team_gae(episode.rewards, values, config.gamma, config.gae_lambda)
        policy_loss = -(scores * advantages).sum()
        value_loss = (values - targets).square().mean()
        total_loss, _ = compute_episode_loss(episode, config)
        assert torch.equal(total_loss, config.actor_coeff * policy_loss + config.value_coeff * value_loss)
        actor_raw = torch.autograd.grad(policy_loss, tensors, retain_graph=True, allow_unused=True)
        value_raw = torch.autograd.grad(value_loss, tensors, allow_unused=True)
        actor_grads = [torch.zeros_like(tensor) if gradient is None else gradient.detach() * config.actor_coeff
                       for gradient, tensor in zip(actor_raw, tensors)]
        value_grads = [torch.zeros_like(tensor) if gradient is None else gradient.detach() * config.value_coeff
                       for gradient, tensor in zip(value_raw, tensors)]
        actor_norm, value_norm = norm(actor_grads), norm(value_grads)
        actor_shared = [actor_grads[i] for i in shared_indices]
        value_shared = [value_grads[i] for i in shared_indices]
        actor_shared_norm, value_shared_norm = norm(actor_shared), norm(value_shared)
        dot = sum((a * b).sum() for a, b in zip(actor_shared, value_shared))
        shared_cosine = float(dot) / (actor_shared_norm * value_shared_norm)
        assert all(torch.isfinite(gradient).all() for gradient in actor_grads + value_grads)
        assert all(parameter.grad is None for parameter in tensors)
        combined_norm = norm([a + v for a, v in zip(actor_grads, value_grads)])
        results.append({
            "scenario_index": index, "scenario": vars(scenarios[index]),
            "episode_metrics": episode.metrics,
            "policy_loss_raw": float(policy_loss.detach()),
            "value_loss_raw_mean_squared": float(value_loss.detach()),
            "actor_coefficient": config.actor_coeff, "value_coefficient": config.value_coeff,
            "actor_gradient_norm_raw": actor_norm / config.actor_coeff,
            "actor_gradient_norm_weighted": actor_norm,
            "value_gradient_norm_weighted": value_norm,
            "weighted_actor_to_value_norm_ratio": actor_norm / value_norm,
            "total_episode_gradient_norm": combined_norm,
            "shared_encoder_lstm_actor_gradient_norm_weighted": actor_shared_norm,
            "shared_encoder_lstm_value_gradient_norm_weighted": value_shared_norm,
            "shared_encoder_lstm_weighted_actor_to_value_norm_ratio": actor_shared_norm / value_shared_norm,
            "shared_encoder_lstm_actor_value_cosine": shared_cosine,
            "value_prediction_first": float(values[0].detach()),
            "value_prediction_last": float(values[-1].detach()),
            "final_reward": episode.rewards[-1],
            "final_advantage": float(advantages[-1]),
            "value_prediction_mean": float(values.detach().mean()),
            "lambda_target_mean": float(targets.mean()),
            "lambda_target_rmse": float(value_loss.detach().sqrt()),
            "advantage_mean": float(advantages.mean()),
            "advantage_min": float(advantages.min()), "advantage_max": float(advantages.max()),
            "reward_min": min(episode.rewards), "reward_max": max(episode.rewards),
            "trajectory_matches_existing_panel_exactly": True,
        })
    assert all(torch.equal(original[name], tensor) for name, tensor in model.state_dict().items())
    after = {str(path.relative_to(ROOT)): sha256(path) for path in tracked_paths}
    assert before == after
    record = {
        "description": "Two selected episode gradient diagnostics; no optimization or clipping performed",
        "selection": "First successful and first failed shared-seed-0 epoch-200 outcome among make_scenarios(260930,24,[(2,1)])",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_updates": checkpoint["updates"],
        "checkpoint_total_steps": checkpoint["total_steps"],
        "training_source_sha256": checkpoint["source_sha256"],
        "input_and_source_sha256": before,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "dtype": "float64"},
        "gradient_definitions": {
            "all_parameters": "Euclidean norms across all parameters; unused gradients are zero",
            "shared_encoder_lstm": "Only pre.* and lstm.* parameters; these feed both policy and value",
            "weighted": "Actor raw loss gradient times 50; value MSE gradient times 1",
        },
        "caveats": [
            "Selected trajectories are descriptive and do not estimate expected batch gradients",
            "Episode norms do not predict optimizer steps: batch averaging, clipping and RMSprop follow",
            "Actor/value gradient norms do not prove that critic learning is inadequate or causes seed variation",
            "No current checkpoint for the later stdout-only training regression was available",
        ],
        "weights_sources_inputs_unchanged": True,
        "results": results,
    }
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "results": results}, allow_nan=False))


if __name__ == "__main__":
    main()
