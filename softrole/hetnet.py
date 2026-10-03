"""Frozen released HetNet with target-team bookkeeping and corrected observations.

This is a contextual baseline: its observations retain P/A occupancy labels and
its weights retain physical-class indexing. It has no capability-health input,
so this evaluator supports composition changes, not sensor-failure scenarios.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hetgat.uavnet import UAVNetA2CEasy
from hetgat.utils import build_hetgraph
from hetnet_ext.signatures import model_signature
from softrole.env import _option, make_env
from softrole.rollout import sample_actions


def _new_model(config, num_p, num_a, use_binary):
    """Mirror main.py's released two-layer, four-head, per-class-critic recipe."""
    base = int(_option(config, "dim", 5)) ** 2
    vision = int(_option(config, "vision", 1 if _option(config, "task") == "fc" else 2))
    dtype = torch.get_default_dtype()
    # The release creates an unregistered binarization constant using the
    # default dtype. Construct in float64 without disturbing caller RNG/dtype.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)
        try:
            model = UAVNetA2CEasy(
                {"vision": vision, "P": base + 4, "A": base, "state": 4},
                {"P": base + 4, "A": base, "state": 4},
                {"P": 16, "A": 16, "state": 16},
                {"P": 5, "A": 6, "state": 8},
                num_p, num_a, num_heads=4, msg_dim=int(_option(config, "msg_dim", 16)),
                use_CNN=False, use_real=not use_binary, use_tanh=False,
                per_class_critic=True, per_agent_critic=False,
                device=torch.device("cpu"), with_two_state=True,
                obs=(2 * vision + 1) ** 2, action_vision=-1).double()
        finally:
            torch.set_default_dtype(dtype)
    return model


def _singleton_batch(_module, inputs):
    """Undo only the release's accidental squeeze of a one-agent batch."""
    x, hidden = inputs
    if x.ndim == 1 and hidden[0].ndim == 2:
        return x.unsqueeze(0), hidden
    return None


def load_hetnet_model(weights, config, num_p, num_a, use_binary=True):
    model = _new_model(config, num_p, num_a, use_binary)
    model.load_state_dict(weights, strict=True)
    for name, tensor in model.state_dict().items():
        original = weights[name].detach().cpu()
        if tensor.dtype != original.dtype or not torch.equal(tensor, original):
            raise ValueError("HetNet evaluation requires unchanged CPU float64 release weights")
    model.f_module_stat.register_forward_pre_hook(_singleton_batch)
    model.f_module_obs.register_forward_pre_hook(_singleton_batch)
    model.eval()
    return model


def initial_memory(model):
    return {key: tuple(value.double() for value in state)
            for key, state in model.init_hidden(1).items()}


def _graph(adapter, config):
    # Position channel decoding from the corner of an observation window is not
    # the agent location. Construct its native one-hot from physical positions.
    positions = adapter.positions
    flat = positions[:, 0] * adapter.dim + positions[:, 1]
    one_hot = np.eye(adapter.base)[flat.astype(int)]
    comm_range = float(_option(config, "comm_range", -1))
    return build_hetgraph(one_hot, num_P=adapter.num_p, num_A=adapter.num_a,
                          with_state=True, with_two_state=True,
                          with_self_loop=False, comm_range_P=comm_range,
                          comm_range_A=comm_range)


def hetnet_logits(model, adapter, config, memory, message_seed, step):
    obs = torch.as_tensor(adapter.raw_observation, dtype=torch.float64)
    obs = obs.reshape(1, adapter.num_p + adapter.num_a, -1)
    key = np.random.SeedSequence([int(message_seed), int(step)])
    seed = int(key.generate_state(1, dtype=np.uint64)[0]) % (2**63 - 1)
    # Original Binary uses module-level Torch random draws for Gumbel messages.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        # The release also allocates intermediate zeros without an explicit
        # dtype, matching main.py's process-wide float64 default.
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            result, _, _, memory = model([obs, memory], _graph(adapter, config))
        finally:
            torch.set_default_dtype(dtype)
    p_logits = torch.cat((result["P"], result["P"].new_full((adapter.num_p, 1), -torch.inf)), 1)
    logits = torch.cat((p_logits, result["A"]), 0) if adapter.num_a else p_logits
    return logits, memory


def evaluate_hetnet(checkpoint, config, scenarios, output, use_binary=True, trace=False):
    """Strictly load original checkpoint['policy_net']; never adapt its weights."""
    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("HetNet evaluation requires at least one scenario")
    if any(s.event_step != -1 for s in scenarios):
        raise ValueError("legacy HetNet has no known-health input; sensor-failure evaluation is unsupported")
    ids = [str(s.scenario_id) for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation scenario IDs must be unique")
    if any(not isinstance(getattr(s, name), int) or getattr(s, name) < 0
           for s in scenarios for name in ("env_seed", "action_seed", "message_seed")):
        raise ValueError("Scenario random seeds must be nonnegative integers")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "policy_net" not in saved:
        raise ValueError("expected an original HetNet checkpoint with policy_net weights")
    weights = saved["policy_net"]
    adapter = make_env(config)
    models, signatures, episodes = {}, {}, []
    with torch.no_grad():
        for scenario in scenarios:
            team = (scenario.num_p, scenario.num_a)
            _, kappa = adapter.reset(scenario.env_seed, *team)
            if team not in models:
                models[team] = load_hetnet_model(weights, config, *team, use_binary=use_binary)
                signatures[team] = model_signature(models[team])["sha256"]
            model = models[team]
            memory = initial_memory(model)
            returns = np.zeros(sum(team), dtype=np.float64)
            episode_trace = []
            for step in range(adapter.max_steps):
                logits, memory = hetnet_logits(model, adapter, config, memory,
                                               scenario.message_seed, step)
                actions, _ = sample_actions(logits, scenario.action_seed, step)
                _, rewards, done, info = adapter.step(actions.numpy(), kappa)
                returns += rewards
                if trace:
                    episode_trace.append({"step": step, "actions": actions.tolist(),
                                          "team_reward": float(rewards.sum())})
                if done:
                    break
            record = {"scenario": asdict(scenario), "scenario_id": str(scenario.scenario_id),
                      "composition": list(team), "num_p": team[0], "num_a": team[1],
                      "steps": step + 1, "success": info["success"],
                      "terminated": info["terminated"], "truncated": info["truncated"],
                      "team_return": float(returns.sum()), "per_agent_returns": returns.tolist(),
                      "mean_agent_return": float(returns.mean()),
                      "environment_version": adapter.environment_version,
                      "event_step": -1, "victim": -1, "event_exposed": False,
                      "intervention": "none", "intervention_step": -1}
            if trace:
                record["trace"] = episode_trace
            episodes.append(record)
    for team, model in models.items():
        if model_signature(model)["sha256"] != signatures[team]:
            raise RuntimeError("Frozen HetNet evaluation changed parameters or buffers")
    composition_metrics = {}
    for team in models:
        records = [e for e in episodes if (e["num_p"], e["num_a"]) == team]
        composition_metrics[f"{team[0]}P{team[1]}A"] = {
            "episodes": len(records), "success_rate": float(np.mean([e["success"] for e in records])),
            "mean_steps": float(np.mean([e["steps"] for e in records])),
            "mean_team_return": float(np.mean([e["team_return"] for e in records])),
            "mean_agent_return": float(np.mean([e["mean_agent_return"] for e in records]))}
    method = "HetNet-Binary" if use_binary else "HetNet-Real"
    report_config = {"model": method, "task": adapter.task, "dim": adapter.dim,
                     "vision": adapter.vision, "max_steps": adapter.max_steps,
                     **{name: _option(config, name, default) for name, default in (
                         ("msg_dim", 16), ("comm_range", -1), ("reward_type", 3), ("nfires", 1))}}
    # Old external checkpoints may omit their training seed. Never substitute
    # the evaluation scenario seed; such reports cannot support seed summaries.
    if "seed" in saved:
        report_config["seed"] = int(saved["seed"])
    shared_signature = next(iter(signatures.values()))
    if len(set(signatures.values())) != 1:
        raise RuntimeError("Team composition unexpectedly changed model identity")
    report = {
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "method": method,
        "comparison": "contextual baseline with richer P/A-typed observations and class-specific weights",
        "supported_comparison": "frozen team composition; no sensor-failure comparison",
        "environment_version": adapter.environment_version,
        "config": report_config, "training_seed": saved.get("seed"),
        "source_sha256": saved.get("source_sha256"),
        "checkpoint_progress": {key: saved.get(key) for key in (
            "epoch", "updates", "total_steps", "total_episodes")},
        "model_signature": shared_signature,
        "scenarios": [asdict(scenario) for scenario in scenarios], "intervention": "none",
        "model_signatures": {f"{p}P{a}A": signature for (p, a), signature in signatures.items()},
        "parameters_unchanged": True, "episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "mean_team_return": float(np.mean([e["team_return"] for e in episodes])),
        "mean_agent_return": float(np.mean([e["mean_agent_return"] for e in episodes])),
        "per_composition": composition_metrics, "per_episode": episodes}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
