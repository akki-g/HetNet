"""Common team GAE and unnormalized episode losses.

Collectors sum these episode gradients. The learner averages by the global
episode count and clips once, after aggregation. No agent/group normalization is
performed. Finite task horizons bootstrap at zero; a continuing task can supply
an explicit bootstrap to ``team_gae``.
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


def compute_episode_loss(episode, config):
    """Sum time/action scores against one team advantage; mean the value error."""
    if not episode.values or len(episode.log_probs) != len(episode.values):
        raise ValueError("Episode needs one summed action log probability and value per step")
    values = torch.stack(episode.values)
    log_probs = torch.stack(episode.log_probs)
    advantages, targets = team_gae(
        episode.rewards, values, config.gamma, config.gae_lambda,
    )
    policy_loss = -(log_probs * advantages).sum()
    value_loss = (values - targets).square().mean()
    total = config.actor_coeff * policy_loss + config.value_coeff * value_loss
    return total, {"policy_loss_sum": float(policy_loss.detach()),
                   "value_loss_sum": float(value_loss.detach()), "episode_count": 1}


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
