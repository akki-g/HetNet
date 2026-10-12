"""Common team GAE and unnormalized episode losses.

Collectors sum these episode gradients. The learner averages by the global
episode count and clips once, after aggregation. No agent/group normalization is
performed. Finite task horizons bootstrap at zero; a continuing task can supply
an explicit bootstrap to ``team_gae``.

Opt-in learner variants share one loss-term builder with the diagnostics:
``critic_mode="decomposed"`` trains per-agent values against per-agent
lambda-return targets and mixes advantages with ``credit_beta`` (beta=1
reproduces the team advantage exactly because GAE is linear);
``objective="mean_agent"`` multiplies the complete episode loss by 1/N.
Every default reproduces the pre-existing arithmetic.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class Episode:
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    traces: list = field(default_factory=list)
    # Per-agent records retained without additional sampling: [N] per step.
    agent_log_probs: list = field(default_factory=list)
    agent_rewards: list = field(default_factory=list)
    agent_values: list = field(default_factory=list)


def team_gae(rewards, values, gamma=1.0, gae_lambda=0.95, bootstrap=0.0):
    """Return detached scalar team advantages and lambda-return targets (T,)."""
    values = torch.as_tensor(values).detach()
    rewards = torch.as_tensor(rewards, dtype=values.dtype, device=values.device)
    if values.ndim != 1 or rewards.shape != values.shape or values.numel() == 0:
        raise ValueError("Team rewards and values must be nonempty vectors of equal length")
    next_value = torch.as_tensor(bootstrap, dtype=values.dtype, device=values.device).detach()
    running = torch.zeros_like(next_value)
    advantages = torch.empty_like(values)
    for t in reversed(range(values.numel())):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * gae_lambda * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


def agent_gae(rewards, values, gamma=1.0, gae_lambda=0.95):
    """Per-agent GAE on [T,N] rewards and values; zero terminal bootstrap for every agent.

    Because the recursion is linear in (rewards, values), summing the returned
    advantages over agents equals ``team_gae`` of the summed rewards and values.
    """
    values = torch.as_tensor(values).detach()
    rewards = torch.as_tensor(rewards, dtype=values.dtype, device=values.device)
    if values.ndim != 2 or rewards.shape != values.shape or values.numel() == 0:
        raise ValueError("Per-agent rewards and values must be nonempty [T,N] tensors of equal shape")
    next_value = torch.zeros_like(values[0])
    running = torch.zeros_like(values[0])
    advantages = torch.empty_like(values)
    for t in reversed(range(values.shape[0])):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * gae_lambda * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


def episode_loss_terms(episode, config):
    """Production actor and value loss terms before coefficients and scaling.

    Team mode: ``policy = -sum_t (sum_j log pi_jt) A_t`` and
    ``value = mean_t (V_t - R_t)^2`` exactly as before. Decomposed mode:
    ``policy = -sum_t sum_j log pi_jt [beta sum_i A_it + (1-beta) A_jt]`` and
    ``value = (1/tau) sum_t sum_j (v_jt - R_jt)^2`` with per-agent targets.
    ``scale`` is 1/N for the mean-agent objective and 1 otherwise. Diagnostics
    reuse this builder so their decomposition matches production arithmetic;
    ``actor_weights`` returns the multipliers applied to the log probabilities.
    """
    if not episode.values or len(episode.log_probs) != len(episode.values):
        raise ValueError("Episode needs one summed action log probability and value per step")
    critic_mode = getattr(config, "critic_mode", "team")
    beta = float(getattr(config, "credit_beta", 1.0))
    objective = getattr(config, "objective", "team")
    if critic_mode == "team":
        values = torch.stack(episode.values)
        log_probs = torch.stack(episode.log_probs)
        advantages, targets = team_gae(
            episode.rewards, values, config.gamma, config.gae_lambda,
        )
        policy_loss = -(log_probs * advantages).sum()
        value_loss = (values - targets).square().mean()
        actor_weights = advantages
    elif critic_mode == "decomposed":
        steps = len(episode.values)
        if (len(episode.agent_values) != steps or len(episode.agent_log_probs) != steps
                or len(episode.agent_rewards) != steps):
            raise ValueError("The decomposed critic needs per-agent values, log probabilities and rewards for every step")
        agent_values = torch.stack(episode.agent_values)
        agent_log_probs = torch.stack(episode.agent_log_probs)
        if agent_values.ndim != 2 or agent_log_probs.shape != agent_values.shape:
            raise ValueError("Per-agent values and log probabilities must share the [T,N] shape")
        agent_rewards = torch.as_tensor(episode.agent_rewards, dtype=agent_values.dtype,
                                        device=agent_values.device)
        advantages, targets = agent_gae(agent_rewards, agent_values, config.gamma, config.gae_lambda)
        mixed = beta * advantages.sum(-1, keepdim=True) + (1 - beta) * advantages
        policy_loss = -(agent_log_probs * mixed).sum()
        value_loss = (agent_values - targets).square().sum(-1).mean()
        actor_weights = mixed
    else:
        raise ValueError(f"Unknown critic_mode: {critic_mode!r}")
    if objective == "mean_agent":
        if episode.agent_rewards:
            n_agents = len(episode.agent_rewards[0])
        else:
            n_agents = episode.metrics.get("num_agents")
        if type(n_agents) is not int or n_agents < 1:
            raise ValueError("The mean-agent objective needs the episode roster size")
        scale = 1.0 / n_agents
    elif objective == "team":
        scale = 1.0
    else:
        raise ValueError(f"Unknown objective: {objective!r}")
    # ``actor_weights`` are the detached per-step ([T]) or per-agent-step ([T,N])
    # multipliers of the log probabilities inside ``policy``; diagnostics reuse them.
    return {"policy": policy_loss, "value": value_loss, "scale": scale,
            "advantages": advantages, "targets": targets, "actor_weights": actor_weights}


def compute_episode_loss(episode, config):
    """Sum time/action scores against the configured advantages; mean the value error."""
    terms = episode_loss_terms(episode, config)
    total = config.actor_coeff * terms["policy"] + config.value_coeff * terms["value"]
    if terms["scale"] != 1.0:
        total = terms["scale"] * total
    return total, {"policy_loss_sum": float(terms["policy"].detach()),
                   "value_loss_sum": float(terms["value"].detach()), "episode_count": 1}


def loss_sum(episodes, config):
    """Return the differentiable SUM over episodes, with detached sum metrics."""
    if not episodes:
        raise ValueError("Cannot update from an empty episode collection")
    losses = []
    stats = {"policy_loss_sum": 0.0, "value_loss_sum": 0.0, "episode_count": 0}
    for episode in episodes:
        loss, fresh = compute_episode_loss(episode, config)
        losses.append(loss)
        for key in stats:
            stats[key] += fresh[key]
    return torch.stack(losses).sum(), stats
