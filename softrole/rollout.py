"""Shared training/evaluation rollout with indexed, independent random streams."""
import numpy as np
import torch
from torch.distributions import Categorical

from softrole.learning import Episode


INTERVENTIONS = ("none", "freeze_affected", "freeze_all", "comm_off")


def _generator(seed, step, substream=0):
    # Index by timestep rather than RNG consumption: interventions cannot shift
    # later action/message randomness or another episode's initial condition.
    key = np.random.SeedSequence([int(seed), int(step), int(substream)])
    keyed_seed = int(key.generate_state(1, dtype=np.uint64)[0]) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(keyed_seed)


def message_noise(seed, step, n_agents, n_bits, dtype):
    draws = []
    for round_index in range(2):
        uniform = torch.rand((n_agents, n_bits), dtype=dtype,
                             generator=_generator(seed, step, round_index))
        uniform = uniform.clamp(torch.finfo(dtype).tiny, 1 - torch.finfo(dtype).eps)
        draws.append(uniform.log() - torch.log1p(-uniform))
    return draws


def sample_actions(logits, seed, step):
    """Inverse-CDF sampling allows exact common-random-number interventions."""
    distribution = Categorical(logits=logits)
    uniform = torch.rand(logits.shape[0], dtype=logits.dtype,
                         generator=_generator(seed, step))
    cdf = distribution.probs.detach().cumsum(-1).contiguous()
    # Renormalize the whole CDF: setting only its final entry to 1 would create
    # a tiny interval for a masked final action when cumulative rounding is low.
    cdf = cdf / cdf[:, -1:]
    action = torch.searchsorted(cdf, uniform[:, None], right=True).squeeze(-1)
    action = action.clamp_max(logits.shape[-1] - 1)
    return action, distribution.log_prob(action).sum()


def communication_adjacency(positions, comm_range):
    positions = np.asarray(positions)
    difference = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(difference, axis=-1)
    allowed = np.ones(distance.shape, dtype=bool) if comm_range < 0 else distance <= comm_range
    np.fill_diagonal(allowed, False)
    return torch.as_tensor(allowed, dtype=torch.bool)


def validate_scenario(config, scenario, intervention="none", intervention_step=None, sham=False):
    """Reject impossible or silently unexposed configured events before rollout."""
    if intervention not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention {intervention!r}; choose {INTERVENTIONS}")
    if not -1 <= scenario.event_step < config.max_steps:
        raise ValueError("event_step must be -1 or inside the task horizon")
    if sham and scenario.event_step < 0:
        raise ValueError("A sham control requires a scheduled sensor failure on every scenario")
    if scenario.event_step >= 0 and getattr(config, "task", "pcp") != "pcp":
        raise ValueError("Sensor failures are supported only for PCP")
    for name in ("env_seed", "action_seed", "message_seed"):
        if not isinstance(getattr(scenario, name), int) or getattr(scenario, name) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    trigger = scenario.event_step if intervention_step is None else intervention_step
    if intervention_step is not None and not 0 <= intervention_step < config.max_steps:
        raise ValueError("intervention_step must be inside the task horizon")
    if intervention != "none" and trigger < 0:
        raise ValueError("An intervention on a no-event episode needs intervention_step")
    if intervention.startswith("freeze") and trigger < 1:
        raise ValueError("Gate freezing needs at least one pre-intervention step")
    n_agents = scenario.num_p + scenario.num_a
    if scenario.event_step >= 0 and not 0 <= scenario.victim < scenario.num_p:
        raise ValueError("Sensor failure victim must identify a sensing agent")
    if intervention == "freeze_affected" and not 0 <= scenario.victim < n_agents:
        raise ValueError("freeze_affected needs a valid victim, also in a no-event control")
    return trigger


def run_episode(model, adapter, config, scenario, training=True, intervention="none",
                intervention_step=None, trace=False, sham=False):
    """Run one scenario; sham suppresses failure but retains its intervention clock.

    ``event_exposed`` and recovery metrics always describe actual sensor loss.
    Scheduled exposure/completion metrics provide the corresponding no-failure
    follow-up in a sham control, including right censoring at the task horizon.
    """
    trigger = validate_scenario(config, scenario, intervention, intervention_step, sham)
    n_agents = scenario.num_p + scenario.num_a
    dtype = next(model.parameters()).dtype
    obs, kappa = adapter.reset(seed=scenario.env_seed, num_p=scenario.num_p, num_a=scenario.num_a)
    kappa = np.asarray(kappa).copy()
    memory = model.initial_memory(n_agents)
    episode = Episode()
    returns = np.zeros(n_agents, dtype=np.float64)
    prior_gate, pinned_gate = None, None
    gate_sum, entropy_sum, null_sum = None, 0.0, 0.0
    event_exposed = scheduled_event_exposed = intervention_exposed = False
    info = {}
    with torch.set_grad_enabled(training):
        for t in range(config.max_steps):
            if t == scenario.event_step:
                scheduled_event_exposed = True
                if not sham:
                    kappa[scenario.victim, 0] = 0.0
                    obs = adapter.observe(kappa)
                    event_exposed = True
            if intervention.startswith("freeze") and t == trigger:
                pinned_gate = prior_gate.clone()
            gate_override = None
            if pinned_gate is not None:
                mask = torch.ones(n_agents, dtype=torch.bool)
                if intervention == "freeze_affected":
                    mask[:] = False
                    mask[scenario.victim] = True
                gate_override = (mask, pinned_gate)
            is_intervened = intervention != "none" and t >= trigger
            intervention_exposed |= is_intervened
            adjacency = communication_adjacency(adapter.positions, config.comm_range)
            out = model(
                torch.as_tensor(obs, dtype=dtype), torch.as_tensor(kappa, dtype=dtype),
                memory, adjacency, remaining=(config.max_steps - t) / config.max_steps,
                noise=message_noise(scenario.message_seed, t, n_agents, config.msg_dim, dtype),
                gate_override=gate_override,
                comm_off=intervention == "comm_off" and is_intervened,
            )
            actions, log_prob = sample_actions(out["logits"], scenario.action_seed, t)
            obs, rewards, done, info = adapter.step(actions.detach().cpu().numpy(), kappa)
            returns += np.asarray(rewards)
            episode.log_probs.append(log_prob)
            episode.values.append(out["value"].reshape(()))
            episode.rewards.append(float(np.asarray(rewards).sum()))
            memory = out["memory"]
            memory["a_prev"] = actions.detach()
            if training and (t + 1) % config.detach_gap == 0:
                memory = model.detach_memory(memory)
            prior_gate = out["gate"].detach()
            current_load = prior_gate.sum(0).cpu().numpy()
            gate_sum = current_load.copy() if gate_sum is None else gate_sum + current_load
            entropy_sum += float(-(prior_gate * prior_gate.clamp_min(1e-15).log()).sum())
            null = out["alpha_null"]
            null = torch.stack(null) if isinstance(null, (list, tuple)) else null
            null_sum += float(null.detach().mean())
            if trace:
                episode.traces.append({"step": t, "gate": prior_gate.cpu().tolist(),
                                       "alpha_null": null.detach().cpu().tolist(),
                                       "actions": actions.tolist(),
                                       "team_reward": episode.rewards[-1]})
            if done:
                break
    steps = len(episode.rewards)
    success = bool(info.get("success", False))
    cap = kappa[:, 1].astype(bool)
    episode.metrics = {
        "scenario_id": str(scenario.scenario_id), "num_p": int(scenario.num_p),
        "env_seed": int(scenario.env_seed), "action_seed": int(scenario.action_seed),
        "message_seed": int(scenario.message_seed),
        "num_a": int(scenario.num_a), "num_agents": n_agents,
        "composition": [int(scenario.num_p), int(scenario.num_a)],
        "steps": steps, "success": success,
        "team_return": float(returns.sum()), "agent_returns": returns.tolist(),
        "return_nocap": float(returns[~cap].mean()) if (~cap).any() else None,
        "return_cap": float(returns[cap].mean()) if cap.any() else None,
        "event_step": int(scenario.event_step), "victim": int(scenario.victim),
        "sham": bool(sham), "scheduled_event_exposed": scheduled_event_exposed,
        "event_exposed": event_exposed,
        "pre_event_success": bool(success and scenario.event_step >= 0 and not scheduled_event_exposed),
        "scheduled_completion_steps": steps - scenario.event_step if scheduled_event_exposed and success else None,
        "scheduled_censored": bool(scheduled_event_exposed and not success),
        "post_schedule_steps": steps - scenario.event_step if scheduled_event_exposed else 0,
        "recovery_steps": steps - scenario.event_step if event_exposed and success else None,
        "recovery_censored": bool(event_exposed and not success),
        "post_event_steps": steps - scenario.event_step if event_exposed else 0,
        "intervention": intervention, "intervention_step": int(trigger),
        "intervention_exposed": bool(intervention_exposed),
        "gate_entropy": entropy_sum / (steps * n_agents),
        "expert_load": (gate_sum / (steps * n_agents)).tolist(),
        "alpha_null": null_sum / steps,
        "payload_bits_per_agent_step": 2 * config.msg_dim,
        "payload_bits_generated": steps * n_agents * 2 * config.msg_dim,
        "environment_version": info.get("environment_version", "unspecified"),
    }
    return episode
