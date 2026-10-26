"""Read-only D3/D4 estimators for a fixed batch at a fixed parameter snapshot.

Monte Carlo returns are noisy realized targets, not the true conditional value.
The existing checkpoint has one team critic: V/N below is a declared,
sum-preserving diagnostic allocation, not a trained Chapter 6 decomposed critic.
No diagnostic changes a model, parameter .grad, optimizer, or training objective.
"""
from __future__ import annotations

import math

import torch

from softrole.learning import agent_gae, team_gae


def _inputs(values, rewards, gamma, gae_lambda):
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (gamma, gae_lambda)):
        raise ValueError("gamma and gae_lambda must be finite and in [0, 1]")
    values = torch.as_tensor(values).detach()
    if not values.is_floating_point():
        values = values.to(torch.float64)
    rewards = torch.as_tensor(rewards, dtype=values.dtype, device=values.device).detach()
    if (values.ndim != 1 or not values.numel() or rewards.ndim != 2
            or rewards.shape[0] != values.numel() or not rewards.shape[1]):
        raise ValueError("Need nonempty values [T] and physical-agent rewards [T,N]")
    if not torch.isfinite(values).all() or not torch.isfinite(rewards).all():
        raise ValueError("Values and rewards must be finite")
    return values, rewards


def discounted_returns(rewards, gamma=1.0):
    """Realized finite-horizon returns; both natural ends and horizon ends use zero."""
    rewards = torch.as_tensor(rewards).detach()
    if (rewards.ndim not in (1, 2) or not rewards.shape[0]
            or (rewards.ndim == 2 and not rewards.shape[1])):
        raise ValueError("rewards must be a nonempty [T] or [T,N] tensor")
    if not rewards.is_floating_point():
        rewards = rewards.to(torch.float64)
    if not math.isfinite(gamma) or not 0 <= gamma <= 1 or not torch.isfinite(rewards).all():
        raise ValueError("Need finite rewards and gamma in [0, 1]")
    running = torch.zeros_like(rewards[0])
    result = torch.empty_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * running
        result[t] = running
    return result


def critic_episode_diagnostics(values, rewards, gamma=1.0, gae_lambda=0.95):
    """D3 sufficient statistics for one complete episode, without gradient paths.

    With e_hat[t] = V[t] - G[t], report the finite-horizon empirical proxy
    (1-lambda) sum_{n=1}^{T-t-1} gamma**n lambda**(n-1) e_hat[t+n].
    Terminal value/return errors are zero. This equals the realized GAE minus
    MC advantage; it does not estimate conditional critic bias without further
    assumptions or independent return samples from the same histories.
    """
    values, rewards = _inputs(values, rewards, gamma, gae_lambda)
    team_rewards = rewards.sum(-1)
    returns = discounted_returns(team_rewards, gamma)
    errors = values - returns
    advantages, targets = team_gae(team_rewards, values, gamma, gae_lambda)
    proxy = torch.zeros_like(values)
    for t in range(values.numel() - 2, -1, -1):
        proxy[t] = gamma * ((1 - gae_lambda) * errors[t + 1] + gae_lambda * proxy[t + 1])
    advantage_difference = advantages - (returns - values)
    target_errors = values - targets
    return {
        "steps": values.numel(), "agents": rewards.shape[1],
        "mc_squared_error_sum": float(errors.square().sum()),
        "mc_error_sum": float(errors.sum()),
        "mc_mse": float(errors.square().mean()),
        "lambda_target_squared_error_sum": float(target_errors.square().sum()),
        "lambda_target_mse": float(target_errors.square().mean()),
        "future_error_proxy_sum": float(proxy.sum()),
        "future_error_proxy_absolute_sum": float(proxy.abs().sum()),
        "future_error_proxy_squared_sum": float(proxy.square().sum()),
        "future_error_proxy_mean": float(proxy.mean()),
        "future_error_proxy_rms": float(proxy.square().mean().sqrt()),
        "proxy_identity_max_abs_error": float((proxy - advantage_difference).abs().max()),
        "gamma": gamma, "gae_lambda": gae_lambda,
    }


def summarize_critic_diagnostics(records):
    """Aggregate records with composition, steps and a `diagnostics` result.

    MC and lambda-target errors use the same sampled steps. Equal-episode
    means are also shown, matching the learner's episode-mean value objective.
    Exact length strata avoid selecting bin edges after observing outcomes.
    """
    if not records:
        raise ValueError("Need at least one critic diagnostic episode")
    protocols = {(r["diagnostics"]["gamma"], r["diagnostics"]["gae_lambda"]) for r in records}
    if len(protocols) != 1:
        raise ValueError("Cannot pool different return/GAE protocols")
    groups = {"overall": [], "by_composition": {}, "by_episode_length": {},
              "by_composition_and_episode_length": {}}
    for row in records:
        d = row["diagnostics"]
        composition = row["composition"]
        if (len(composition) != 2 or any(type(x) is not int or x < 0 for x in composition)
                or sum(composition) != d["agents"] or row["steps"] != d["steps"]):
            raise ValueError("Composition or episode length disagrees with critic diagnostics")
        key, length = ",".join(map(str, composition)), str(row["steps"])
        groups["overall"].append(d)
        groups["by_composition"].setdefault(key, []).append(d)
        groups["by_episode_length"].setdefault(length, []).append(d)
        groups["by_composition_and_episode_length"].setdefault(f"{key};{length}", []).append(d)

    def aggregate(rows):
        count = sum(row["steps"] for row in rows)
        result = {"episodes": len(rows), "steps": count}
        for source, target in (
            ("mc_squared_error_sum", "mc_mse"), ("mc_error_sum", "mc_error_mean"),
            ("lambda_target_squared_error_sum", "lambda_target_mse"),
            ("future_error_proxy_sum", "future_error_proxy_mean"),
            ("future_error_proxy_absolute_sum", "future_error_proxy_mean_absolute"),
            ("future_error_proxy_squared_sum", "future_error_proxy_mean_squared"),
        ):
            result[target] = sum(row[source] for row in rows) / count
        result["future_error_proxy_rms"] = math.sqrt(result["future_error_proxy_mean_squared"])
        result["equal_episode_mc_mse"] = sum(row["mc_mse"] for row in rows) / len(rows)
        result["equal_episode_lambda_target_mse"] = sum(row["lambda_target_mse"] for row in rows) / len(rows)
        result["proxy_identity_max_abs_error"] = max(row["proxy_identity_max_abs_error"] for row in rows)
        return result

    return {
        "protocol": {
            "gamma": next(iter(protocols))[0], "gae_lambda": next(iter(protocols))[1],
            "terminal_bootstrap": 0,
            "error": "V_t minus realized finite-horizon team Monte Carlo return G_t",
            "proxy": "(1-lambda) sum_{n=1}^{T-t-1} gamma^n lambda^(n-1) (V_{t+n}-G_{t+n})",
            "limitation": "Monte Carlo residuals include return noise; the proxy is not true conditional value bias.",
        },
        "overall": aggregate(groups["overall"]),
        **{name: {key: aggregate(rows) for key, rows in mapping.items()}
           for name, mapping in groups.items() if name != "overall"},
    }


def beta_key(beta):
    """Round-trip a binary64 credit coefficient without merging nearby variants."""
    value = float(beta)
    return "0" if value == 0 else format(value, ".17g")


def actor_episode_gradients(log_probs, values, rewards, model, config, retain_graph=False,
                            agent_values=None, scale=1.0):
    """Return unclipped actor loss gradient sums for paired D4 comparisons.

    `log_probs[T,N]` must come from the same logged actions with the production
    message surrogate and recurrent detach boundaries. Advantages are detached.
    The coefficient ca is retained; there is no agent/time normalization.
    Critic-only coordinates are zero. torch.autograd.grad preserves .grad.

    ``agent_values`` [T,N], the predictions of a checkpoint's learned decomposed
    critic, add ``learned_gae_beta_*`` variants that stay labelled separately
    from the ``allocated_gae_*`` V/N allocation. ``scale`` is the production
    per-episode loss factor (1/N under the mean-agent objective) and multiplies
    every variant, so the production variant equals the production gradient.
    """
    values, rewards = _inputs(values, rewards, config.gamma, config.gae_lambda)
    if (log_probs.shape != rewards.shape or not log_probs.requires_grad
            or not torch.isfinite(log_probs).all()):
        raise ValueError("Need differentiable finite per-agent log probabilities [T,N]")
    if not math.isfinite(config.actor_coeff) or config.actor_coeff <= 0:
        raise ValueError("actor_coeff must be positive and finite")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite episode loss factor")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model has no differentiable parameters")
    coefficient = config.actor_coeff * scale
    team_advantage, _ = team_gae(rewards.sum(-1), values, config.gamma, config.gae_lambda)
    own_advantages = torch.stack([
        team_gae(rewards[:, j], values / rewards.shape[1], config.gamma, config.gae_lambda)[0]
        for j in range(rewards.shape[1])
    ], dim=-1)
    returns = discounted_returns(rewards, config.gamma)
    losses = {"team_gae": -coefficient * (log_probs.sum(-1) * team_advantage).sum()}
    for prefix, own in (("allocated_gae", own_advantages), ("mc", returns)):
        for beta in (1.0, 0.5, 0.0):
            mixed = beta * own.sum(-1, keepdim=True) + (1 - beta) * own
            losses[f"{prefix}_beta_{beta_key(beta)}"] = -coefficient * (log_probs * mixed).sum()
    critic_mode = getattr(config, "critic_mode", "team")
    configured_beta = float(getattr(config, "credit_beta", 1.0))
    learned = None
    if agent_values is not None:
        agent_values = torch.as_tensor(agent_values).detach()
        if not agent_values.is_floating_point():
            agent_values = agent_values.to(values.dtype)
        if agent_values.shape != rewards.shape or not torch.isfinite(agent_values).all():
            raise ValueError("Learned per-agent values must be finite [T,N]")
        learned, _ = agent_gae(rewards, agent_values, config.gamma, config.gae_lambda)
        for beta in sorted({1.0, 0.5, 0.0, configured_beta}, reverse=True):
            mixed = beta * learned.sum(-1, keepdim=True) + (1 - beta) * learned
            losses[f"learned_gae_beta_{beta_key(beta)}"] = -coefficient * (log_probs * mixed).sum()
    elif critic_mode == "decomposed":
        raise ValueError("A decomposed critic needs its learned per-agent values for D4")
    production = "team_gae" if critic_mode == "team" else f"learned_gae_beta_{beta_key(configured_beta)}"
    gradients = {}
    for index, (name, loss) in enumerate(losses.items()):
        parts = torch.autograd.grad(loss, parameters, allow_unused=True,
                                    retain_graph=retain_graph or index < len(losses) - 1)
        vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                            for p, g in zip(parameters, parts)])
        if not torch.isfinite(vector).all():
            raise ValueError("D4 encountered a nonfinite actor gradient")
        gradients[name] = vector
    difference = gradients["allocated_gae_beta_1"] - gradients["team_gae"]
    denominator = gradients["team_gae"].norm()
    checks = {
        "beta_1_advantage_max_abs_error": float((own_advantages.sum(-1) - team_advantage).abs().max()),
        "beta_1_gradient_max_abs_error": float(difference.abs().max()),
        "beta_1_gradient_relative_l2_error": float(difference.norm() / denominator) if denominator > 0 else None,
        "actor_coeff": config.actor_coeff,
        "episode_scale": scale,
    }
    if learned is not None:
        learned_difference = gradients["learned_gae_beta_1"] - gradients["team_gae"]
        checks.update({
            "learned_values_sum_max_abs_error": float((agent_values.sum(-1) - values).abs().max()),
            "learned_beta_1_advantage_max_abs_error": float((learned.sum(-1) - team_advantage).abs().max()),
            "learned_beta_1_gradient_max_abs_error": float(learned_difference.abs().max()),
        })
    return {
        "gradients": gradients,
        "checks": checks,
        "protocol": {
            "allocation": "v_jt = V_t/N; diagnostic allocation only, not a trained decomposed critic",
            "learned": ("v_jt from the checkpoint's learned decomposed critic; beta variants mix its per-agent GAE"
                        if learned is not None else None),
            "mc": "per-agent realized return, zero baseline; beta=1 is the team-return action-score reference",
            "objective": "sum of physical-agent rewards; unnormalized per-episode actor loss gradient sums",
            "production_variant": production,
            "episode_scale": "1/N under the mean-agent objective, otherwise 1; applied to every variant",
            "limitations": [
                "beta<1 drops cross-agent credit and can introduce bias; larger empirical SNR does not imply a better estimator",
                "V/N cannot test learned per-agent critic accuracy or the Chapter 6 value objective",
                "MC variants retain the logged actor graph's straight-through message derivative and recurrent truncation",
            ],
        },
    }


def agent_critic_episode_diagnostics(agent_values, rewards, gamma=1.0, gae_lambda=0.95):
    """D3 on a learned decomposition: each agent's v_jt against its own realized return.

    Agent j is treated as a one-agent episode, so every statistic is a sum over
    agent-steps rather than steps. The team-level record on sum_j v_jt has a
    different denominator and is reported separately.
    """
    agent_values = torch.as_tensor(agent_values).detach()
    if not agent_values.is_floating_point():
        agent_values = agent_values.to(torch.float64)
    rewards = torch.as_tensor(rewards, dtype=agent_values.dtype, device=agent_values.device).detach()
    if agent_values.ndim != 2 or rewards.shape != agent_values.shape or not agent_values.numel():
        raise ValueError("Need learned per-agent values and rewards of equal nonempty [T,N] shape")
    if not torch.isfinite(agent_values).all() or not torch.isfinite(rewards).all():
        raise ValueError("Per-agent values and rewards must be finite")
    steps, agents = agent_values.shape
    per_agent = [critic_episode_diagnostics(agent_values[:, j], rewards[:, j:j + 1], gamma, gae_lambda)
                 for j in range(agents)]
    agent_steps = steps * agents
    sums = {key: sum(row[key] for row in per_agent) for key in (
        "mc_squared_error_sum", "mc_error_sum", "lambda_target_squared_error_sum",
        "future_error_proxy_sum", "future_error_proxy_absolute_sum", "future_error_proxy_squared_sum")}
    return {
        "steps": steps, "agents": agents, "agent_steps": agent_steps, **sums,
        "mc_mse": sums["mc_squared_error_sum"] / agent_steps,
        "lambda_target_mse": sums["lambda_target_squared_error_sum"] / agent_steps,
        "future_error_proxy_mean": sums["future_error_proxy_sum"] / agent_steps,
        "future_error_proxy_rms": math.sqrt(sums["future_error_proxy_squared_sum"] / agent_steps),
        "proxy_identity_max_abs_error": max(row["proxy_identity_max_abs_error"] for row in per_agent),
        "per_agent_mc_mse": [row["mc_mse"] for row in per_agent],
        "gamma": gamma, "gae_lambda": gae_lambda,
    }


def summarize_agent_critic_diagnostics(records):
    """Aggregate learned per-agent critic records over agent-steps, by composition."""
    if not records:
        raise ValueError("Need at least one learned critic diagnostic episode")
    protocols = {(r["diagnostics"]["gamma"], r["diagnostics"]["gae_lambda"]) for r in records}
    if len(protocols) != 1:
        raise ValueError("Cannot pool different return/GAE protocols")
    groups = {"overall": [], "by_composition": {}}
    for row in records:
        d = row["diagnostics"]
        composition = row["composition"]
        if (len(composition) != 2 or any(type(x) is not int or x < 0 for x in composition)
                or sum(composition) != d["agents"] or row["steps"] != d["steps"]):
            raise ValueError("Composition or episode length disagrees with learned critic diagnostics")
        groups["overall"].append(d)
        groups["by_composition"].setdefault(",".join(map(str, composition)), []).append(d)

    def aggregate(rows):
        count = sum(row["agent_steps"] for row in rows)
        result = {"episodes": len(rows), "agent_steps": count}
        for source, target in (
            ("mc_squared_error_sum", "mc_mse"), ("mc_error_sum", "mc_error_mean"),
            ("lambda_target_squared_error_sum", "lambda_target_mse"),
            ("future_error_proxy_sum", "future_error_proxy_mean"),
            ("future_error_proxy_absolute_sum", "future_error_proxy_mean_absolute"),
            ("future_error_proxy_squared_sum", "future_error_proxy_mean_squared"),
        ):
            result[target] = sum(row[source] for row in rows) / count
        result["future_error_proxy_rms"] = math.sqrt(result["future_error_proxy_mean_squared"])
        result["equal_episode_mc_mse"] = sum(row["mc_mse"] for row in rows) / len(rows)
        result["equal_episode_lambda_target_mse"] = sum(row["lambda_target_mse"] for row in rows) / len(rows)
        result["proxy_identity_max_abs_error"] = max(row["proxy_identity_max_abs_error"] for row in rows)
        return result

    gamma, gae_lambda = next(iter(protocols))
    return {
        "protocol": {
            "gamma": gamma, "gae_lambda": gae_lambda, "terminal_bootstrap": 0,
            "unit": "agent-steps of the checkpoint's learned per-agent critic",
            "error": "v_jt minus agent j's realized finite-horizon Monte Carlo return",
            "limitation": "Monte Carlo residuals include return noise; sums over agents are compared in the team record.",
        },
        "overall": aggregate(groups["overall"]),
        "by_composition": {key: aggregate(rows) for key, rows in groups["by_composition"].items()},
    }


class GradientMoments:
    """Streaming paired episode-gradient means and scalar Welford variance.

    The reported SNR is ||mean(g)||^2 / mean(||g-mean(g)||^2), using a
    population denominator across sampled episodes (not variance of the mean).
    It is descriptive for this batch; it is not an independent-training-seed CI.
    """

    def __init__(self):
        self.count = 0
        self.means = {}
        self.centered_sums = {}

    def add(self, gradients):
        if not gradients or (self.count and gradients.keys() != self.means.keys()):
            raise ValueError("Every episode needs the same nonempty gradient variant set")
        for name, vector in gradients.items():
            if vector.ndim != 1 or not vector.numel() or not torch.isfinite(vector).all():
                raise ValueError("Gradient samples must be finite nonempty vectors")
            if self.count and (vector.shape != self.means[name].shape
                               or vector.dtype != self.means[name].dtype
                               or vector.device != self.means[name].device):
                raise ValueError("Gradient coordinates, dtype and device must match")
        self.count += 1
        for name, vector in gradients.items():
            vector = vector.detach()
            if self.count == 1:
                self.means[name] = vector.clone()
                self.centered_sums[name] = 0.0
            else:
                delta = vector - self.means[name]
                self.means[name] += delta / self.count
                self.centered_sums[name] += float(torch.dot(delta, vector - self.means[name]))

    def summary(self):
        if not self.count:
            raise ValueError("Cannot summarize an empty gradient batch")
        rows = {}
        reference = self.means.get("team_gae")
        for name, mean in self.means.items():
            signal = float(mean.square().sum())
            variance = max(0.0, self.centered_sums[name] / self.count)
            defined = self.count > 1 and variance > 0
            rows[name] = {
                "episodes": self.count, "mean_gradient_l2": math.sqrt(signal),
                "mean_gradient_squared_l2": signal,
                "mean_centered_squared_l2": variance,
                "rms_episode_gradient_l2": math.sqrt(signal + variance),
                "snr": signal / variance if defined else None,
                "snr_status": "finite" if defined else (
                    "insufficient_episodes" if self.count < 2 else "zero_empirical_variance"),
            }
            if reference is not None:
                norm_product = reference.norm() * mean.norm()
                rows[name]["cosine_mean_to_team_gae"] = (
                    float(torch.dot(reference, mean) / norm_product) if norm_product > 0 else None)
                rows[name]["mean_difference_from_team_gae_l2"] = float((mean - reference).norm())
        return {"formula": "||mean(g_e)||^2 / mean(||g_e-mean(g_e)||^2)",
                "weighting": "equal complete episodes; no length or team-size normalization",
                "variants": rows}
