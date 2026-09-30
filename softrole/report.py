"""Seed-cluster summaries of frozen evaluation reports, never episode-level SEs."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


METRICS = ("success_rate", "team_return", "completion_steps_horizon_capped")


def summarize_reports(paths, output=None, bootstrap_samples=10000, seed=0):
    """Average per-seed metrics and bootstrap independent training seeds.

    Stratify by configuration (except seed), checkpoint epoch/update, source,
    environment version, team composition, scheduled failure, and intervention.
    Within a stratum, each training seed has equal weight regardless of its
    number of evaluation episodes. Unsuccessful completion times are capped at
    the task horizon; this is not a successful-episode completion-time estimate.
    One seed cannot support a between-training-seed confidence interval.

    Inputs must use the same intended evaluation distribution within a stratum.
    Scenario IDs and counts remain in the output so panels can be audited.
    """
    if not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rng = np.random.default_rng(seed)
    groups, records = {}, defaultdict(lambda: defaultdict(list))
    seen, signatures = set(), {}
    for path in paths:
        with Path(path).open() as stream:
            report = json.load(stream)
        config = dict(report.get("config", {}))
        if not {"seed", "model", "task", "max_steps"}.issubset(config):
            raise ValueError("Evaluation reports must contain config with seed/model/task/max_steps")
        training_seed = int(config.pop("seed"))
        horizon = int(config["max_steps"])
        if horizon <= 0:
            raise ValueError("Evaluation horizon must be positive")
        progress = report.get("checkpoint_progress", {})
        environment = report.get("environment_version")
        if not environment or environment == "unspecified":
            raise ValueError("Evaluation reports must identify their environment version")
        if not report.get("per_episode"):
            raise ValueError("Evaluation reports must contain episode records")
        for episode in report["per_episode"]:
            if episode.get("environment_version", environment) != environment:
                raise ValueError("Report and episode environment versions disagree")
            intervention = episode.get("intervention", report.get("intervention", "none"))
            scheduled_failure = episode.get("event_step", -1) >= 0
            identity = {"config": config, "environment_version": environment,
                        "source_sha256": report.get("source_sha256"),
                        "checkpoint_epoch": progress.get("epoch"),
                        "checkpoint_updates": progress.get("updates"),
                        "composition": episode["composition"],
                        "scheduled_sensor_failure": scheduled_failure,
                        "sham": bool(report.get("sham", episode.get("sham", False))),
                        "intervention": intervention}
            # No-event controls with different intervention times are different
            # experiments. Failure interventions normally track the event time.
            if intervention != "none":
                identity["intervention_offset_from_event"] = (
                    int(episode["intervention_step"]) - int(episode["event_step"])
                    if scheduled_failure else int(episode["intervention_step"]))
            key = json.dumps(identity, sort_keys=True)
            groups[key] = identity
            unique = (key, training_seed, str(episode["scenario_id"]))
            if unique in seen:
                raise ValueError("Duplicate scenario for the same training seed and evaluation stratum")
            seen.add(unique)
            checkpoint_key = (key, training_seed)
            signature = report.get("model_signature")
            if checkpoint_key in signatures and signatures[checkpoint_key] != signature:
                raise ValueError("Conflicting checkpoints for the same training seed and stratum")
            signatures[checkpoint_key] = signature
            steps, reward = int(episode["steps"]), float(episode["team_return"])
            if not 1 <= steps <= horizon or not np.isfinite(reward):
                raise ValueError("Episode steps/return are outside the finite evaluation contract")
            records[key][training_seed].append(episode)
    if not groups:
        raise ValueError("Supply at least one evaluation report")

    summaries = []
    for key in sorted(groups):
        identity, per_seed = groups[key], []
        horizon = identity["config"]["max_steps"]
        for training_seed, episodes in sorted(records[key].items()):
            count = len(episodes)
            successes = sum(bool(episode["success"]) for episode in episodes)
            per_seed.append({
                "training_seed": training_seed, "episodes": count,
                "successes": successes, "censored_completions": count - successes,
                "event_exposed_episodes": sum(bool(episode.get("event_exposed")) for episode in episodes),
                "scenario_ids": sorted(str(episode["scenario_id"]) for episode in episodes),
                "success_rate": successes / count,
                "team_return": float(np.mean([episode["team_return"] for episode in episodes])),
                "completion_steps_horizon_capped": float(np.mean([
                    episode["steps"] if episode["success"] else horizon for episode in episodes])),
            })
        values = np.array([[row[metric] for metric in METRICS] for row in per_seed], dtype=float)
        intervals = None
        if len(per_seed) >= 2:
            draws = rng.integers(0, len(per_seed), size=(bootstrap_samples, len(per_seed)))
            intervals = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], axis=0)
        metrics = {metric: {"mean": float(values[:, index].mean()),
                            "ci95": intervals[:, index].tolist() if intervals is not None else None}
                   for index, metric in enumerate(METRICS)}
        summaries.append(dict(identity, training_seeds=len(per_seed),
                              episodes=sum(row["episodes"] for row in per_seed),
                              metrics=metrics, per_seed=per_seed))
    summary = {
        "method": "Equal-weight training-seed means; percentile bootstrap resamples whole training seeds within strata",
        "confidence_level": 0.95, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "completion_time_definition": "Success step; unsuccessful episodes are capped at the task horizon",
        "limitations": ["One seed has no between-training-seed confidence interval",
                        "Percentile intervals with few training seeds are unstable",
                        "Intervals describe training-seed variability conditional on the supplied evaluation panels",
                        "Inputs within a stratum must represent the same intended evaluation distribution"],
        "groups": summaries,
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return summary
