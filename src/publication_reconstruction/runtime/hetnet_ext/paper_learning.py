"""Equation-led per-class learner, separate from the archived public learner.

HetNet Eq. 6 uses individual rewards and total team-size normalization.
Section 5.2 and Algorithm 1 use class-average rewards-to-go for the critic;
GAE is used only for the actor. Episode averaging, gamma/lambda, truncation,
and global gradient clipping are declared reconstruction choices.
"""
from __future__ import annotations

import torch


LEARNER_SPEC = "paper-equations-v1"


def returns_and_advantages(rewards, values, *, gamma=1.0, gae_lambda=0.95):
    """Return detached Monte-Carlo targets and GAE for one complete episode.

    ``rewards`` is [time, agents in class]; ``values`` is [time]. Both natural
    termination and the finite task horizon have zero terminal bootstrap.
    """
    if rewards.ndim != 2 or values.ndim != 1 or rewards.shape[0] != values.shape[0]:
        raise ValueError("Expected rewards [time, agents] and class values [time]")
    if rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("Returns require a nonempty class and episode")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and GAE lambda must lie in [0, 1]")
    with torch.no_grad():
        rewards = rewards.detach()
        values = values.detach()
        returns = torch.empty_like(rewards)
        advantages = torch.empty_like(rewards)
        future_return = torch.zeros_like(rewards[0])
        gae = torch.zeros_like(rewards[0])
        next_value = values.new_zeros(())
        for step in reversed(range(rewards.shape[0])):
            future_return = rewards[step] + gamma * future_return
            delta = rewards[step] + gamma * next_value - values[step]
            gae = delta + gamma * gae_lambda * gae
            returns[step] = future_return
            advantages[step] = gae
            next_value = values[step]
    return returns, advantages


def episode_losses(rewards, class_log_probs, class_values, *, gamma=1.0, gae_lambda=0.95):
    """Unnormalized episode losses, with total-N weighting from Algorithm 1.

    Classes appear in physical agent order; an absent class is omitted. The
    caller sums episodes before backward and averages gradients globally once.
    No padded timesteps, advantage standardization, or team-reward substitution
    enter either loss.
    """
    if len(class_log_probs) != len(class_values) or not class_values:
        raise ValueError("Each present class needs log probabilities and a critic")
    first = class_values[0]
    rewards = torch.as_tensor(rewards, dtype=first.dtype, device=first.device)
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("Expected a nonempty [time, total agents] reward array")
    total_agents = rewards.shape[1]
    policy_loss = first.new_zeros(())
    value_loss = first.new_zeros(())
    offset = 0
    for log_probs, values in zip(class_log_probs, class_values):
        if log_probs.ndim != 2 or log_probs.shape[0] != rewards.shape[0]:
            raise ValueError("Expected log probabilities [time, agents in class]")
        count = log_probs.shape[1]
        if count == 0 or offset + count > total_agents:
            raise ValueError("Class counts must partition the reward columns")
        targets, advantages = returns_and_advantages(
            rewards[:, offset:offset + count], values,
            gamma=gamma, gae_lambda=gae_lambda)
        policy_loss = policy_loss - (log_probs * advantages).sum() / total_agents
        value_loss = value_loss + count / total_agents * (
            values - targets.mean(dim=1)).square().sum()
        offset += count
    if offset != total_agents:
        raise ValueError("Class counts must partition the reward columns")
    return policy_loss, value_loss


def normalize_and_clip_gradients(parameters, episode_count, max_norm=0.75):
    """Average the global episode-gradient sum, then clip exactly once."""
    if isinstance(episode_count, bool) or int(episode_count) != episode_count or episode_count <= 0:
        raise ValueError("A positive global episode count is required")
    parameters = list(parameters)
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(episode_count)
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
