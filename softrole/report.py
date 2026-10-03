"""Seed-cluster summaries of frozen evaluation reports, never episode-level SEs."""

from collections import defaultdict
import json
import hashlib
from pathlib import Path

import numpy as np


METRICS = ("success_rate", "team_return", "completion_steps_horizon_capped", "mean_agent_return")


def protocol_identity(selection, distribution):
    """Content identity for a declared selector and evaluation distribution."""
    encoded = json.dumps({"selection": selection, "distribution": distribution},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_evaluation_protocol(protocol, progress, scenarios, scenario_sha256):
    """Validate metadata without silently certifying a 'first saved' selection.

    The preparer, which sees all checkpoints, verifies that it selected the first
    saved checkpoint at the threshold. An evaluator can only verify the budget.
    """
    selection, distribution = protocol["selection"], protocol["distribution"]
    if protocol.get("protocol_id") != protocol_identity(selection, distribution):
        raise ValueError("Evaluation protocol identity does not match its contents")
    target = selection.get("target_steps")
    if (selection.get("rule") != "first_saved_at_or_above"
            or type(target) is not int or target <= 0):
        raise ValueError("Protocol requires a positive first-saved step budget")
    actual = progress.get("total_steps")
    if type(actual) is not int or actual < target:
        raise ValueError("Checkpoint does not reach the declared step budget")
    if not scenario_sha256 or protocol.get("scenario_sha256") != scenario_sha256:
        raise ValueError("Scenario file hash differs from the evaluation protocol")
    compositions = distribution.get("compositions", [])
    count = distribution.get("episodes_per_composition")
    if (distribution.get("task") not in ("pp", "pcp", "fc")
            or type(count) is not int or count <= 0
            or not compositions or len({tuple(c) for c in compositions}) != len(compositions)
            or any(len(c) != 2 or any(type(n) is not int for n in c)
                   or c[0] < 1 or c[1] < 0 for c in compositions)):
        raise ValueError("Invalid declared task/composition/episode distribution")
    probability = distribution.get("failure_probability")
    window = distribution.get("failure_window")
    if (not isinstance(probability, (float, int)) or not 0 <= probability <= 1
            or not isinstance(window, (tuple, list)) or len(window) != 2
            or any(type(t) is not int for t in window) or not 0 <= window[0] <= window[1]
            or type(distribution.get("scenario_seed")) is not int
            or distribution["scenario_seed"] < 0
            or distribution.get("victim_rule") != "uniform_sensing_agent"):
        raise ValueError("Invalid declared scenario randomization")
    if probability and distribution["task"] != "pcp":
        raise ValueError("Sensor-failure protocols are PCP only")
    counts, identifiers = defaultdict(int), set()
    for row in scenarios:
        sid = str(row["scenario_id"])
        if sid in identifiers:
            raise ValueError("Protocol scenarios contain duplicate scenario IDs")
        identifiers.add(sid)
        composition = (row["num_p"], row["num_a"])
        counts[composition] += 1
        event, victim = row["event_step"], row["victim"]
        if (type(event) is not int or type(victim) is not int
                or any(type(row[name]) is not int or row[name] < 0
                       for name in ("env_seed", "action_seed", "message_seed"))):
            raise ValueError("Invalid scenario timing or random streams")
        if ((probability == 0 and event != -1) or (probability == 1 and event < 0)
                or (event >= 0 and (not window[0] <= event <= window[1]
                                    or not 0 <= victim < composition[0]))
                or (event < 0 and (event != -1 or victim != -1))):
            raise ValueError("Scenario event differs from the declared distribution")
    if dict(counts) != {tuple(c): count for c in compositions}:
        raise ValueError("Scenario counts/compositions differ from the declared distribution")


def _metrics(episodes, horizon):
    return np.array([[float(bool(e["success"])), e["team_return"],
                      e["steps"] if e["success"] else horizon,
                      e["team_return"] / sum(e["composition"])] for e in episodes], dtype=float)


def _uncertainty(rows, rng, samples):
    values = np.array([[r[m] for m in METRICS] for r in rows], dtype=float)
    intervals = None
    if len(rows) >= 2:
        draws = rng.integers(0, len(rows), size=(samples, len(rows)))
        intervals = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], axis=0)
    return {m: {"mean": float(values[:, i].mean()),
                "ci95": intervals[:, i].tolist() if intervals is not None else None}
            for i, m in enumerate(METRICS)}


def _protocol_for_report(report, sidecar, sidecar_path):
    protocol = report.get("evaluation_protocol")
    if sidecar is not None:
        if protocol is not None and protocol != sidecar:
            raise ValueError("Report protocol conflicts with the supplied sidecar")
        protocol = sidecar
    if protocol is None:
        return None
    scenarios = report.get("scenarios")
    if not scenarios:
        raise ValueError("Protocol grouping requires complete scenario records")
    scenario_hash = report.get("scenarios_sha256")
    if not scenario_hash and sidecar_path is not None:
        panel = sidecar_path.with_name("scenarios.json")
        data = panel.read_bytes()
        if json.loads(data) != scenarios:
            raise ValueError("Legacy report scenarios do not match the sidecar's scenarios.json")
        scenario_hash = hashlib.sha256(data).hexdigest()
    validate_evaluation_protocol(protocol, report.get("checkpoint_progress", {}), scenarios, scenario_hash)
    if protocol["distribution"]["task"] != report["config"]["task"]:
        raise ValueError("Report task differs from the declared evaluation protocol")
    return protocol


def _paired_summaries(groups, records, metadata, panels, rng, samples):
    """Full-panel within-checkpoint failure minus sham, then whole-seed inference."""
    candidates = defaultdict(dict)
    for key, identity in groups.items():
        if not identity["scheduled_sensor_failure"]:
            continue
        base = {k: v for k, v in identity.items() if k != "sham"}
        candidates[json.dumps(base, sort_keys=True)][identity["sham"]] = key
    paired = []
    for base, conditions in sorted(candidates.items()):
        if set(conditions) != {False, True}:
            continue
        actual, sham = conditions[False], conditions[True]
        # Older report schemas may not contain the full random-stream panel.
        # Their absolute strata remain supported, but cannot certify pairing.
        if not all(panels.get((k, seed)) is not None for k in (actual, sham) for seed in records[k]):
            continue
        if set(records[actual]) != set(records[sham]):
            raise ValueError("Failure/sham pairs require identical training-seed sets")
        identity, per_seed = json.loads(base), []
        for training_seed in sorted(records[actual]):
            left = sorted(records[actual][training_seed], key=lambda e: str(e["scenario_id"]))
            right = sorted(records[sham][training_seed], key=lambda e: str(e["scenario_id"]))
            if panels[(actual, training_seed)] != panels[(sham, training_seed)]:
                raise ValueError("Failure/sham pairs require identical complete scenarios")
            if metadata[(actual, training_seed)] != metadata[(sham, training_seed)]:
                raise ValueError("Failure/sham pairs require the same exact checkpoint")
            if [str(e["scenario_id"]) for e in left] != [str(e["scenario_id"]) for e in right]:
                raise ValueError("Failure/sham pairs require the same full assigned panel")
            for failure, control in zip(left, right):
                for diagnostic in ("scheduled_event_exposed", "pre_event_victim_reached",
                                   "pre_event_victim_target_visible", "pre_event_victim_target_seen",
                                   "initial_state"):
                    if failure.get(diagnostic) != control.get(diagnostic):
                        raise ValueError("Failure/sham pre-event diagnostics disagree")
                if "trace" in failure and "trace" in control:
                    event = failure["event_step"]
                    if failure["trace"][:event] != control["trace"][:event]:
                        raise ValueError("Failure/sham pre-event traces disagree")
            delta = (_metrics(left, identity["config"]["max_steps"])
                     - _metrics(right, identity["config"]["max_steps"])).mean(axis=0)
            per_seed.append({"training_seed": training_seed, "episodes": len(left),
                             "scenario_ids": [str(e["scenario_id"]) for e in left],
                             "checkpoint": metadata[(actual, training_seed)],
                             **dict(zip(METRICS, delta.tolist()))})
        paired.append(dict(identity, contrast="failure_minus_sham", training_seeds=len(per_seed),
                           episodes=sum(r["episodes"] for r in per_seed), per_seed=per_seed,
                           metrics=_uncertainty(per_seed, rng, samples)))
    return paired


def summarize_reports(paths, output=None, bootstrap_samples=10000, seed=0, protocol=None):
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
    sidecar_path = Path(protocol) if isinstance(protocol, (str, Path)) else None
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path else protocol
    rng = np.random.default_rng(seed)
    groups, records = {}, defaultdict(lambda: defaultdict(list))
    seen, signatures, metadata, panels = set(), {}, {}, {}
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
        declared = _protocol_for_report(report, sidecar, sidecar_path)
        scenario_map = ({str(row["scenario_id"]): row for row in report["scenarios"]}
                        if report.get("scenarios") else None)
        if scenario_map is not None:
            if (len(scenario_map) != len(report["scenarios"])
                    or set(scenario_map) != {str(e["scenario_id"]) for e in report["per_episode"]}):
                raise ValueError("Report outcomes must cover its complete scenario panel")
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
                        "evaluation_version": report.get("evaluation_version"),
                        "evaluator_source_sha256": report.get("evaluator", {}).get("source", {}).get("sha256"),
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
            if declared is not None:
                identity.pop("checkpoint_epoch")
                identity.pop("checkpoint_updates")
                identity["evaluation_protocol"] = {k: declared[k] for k in ("protocol_id", "selection", "distribution")}
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
            checkpoint_metadata = {"model_signature": signature, "checkpoint_sha256": report.get("checkpoint_sha256"),
                                   "progress": progress}
            if checkpoint_key in metadata and metadata[checkpoint_key] != checkpoint_metadata:
                raise ValueError("Conflicting checkpoints for the same training seed and stratum")
            metadata[checkpoint_key] = checkpoint_metadata
            if scenario_map is None:
                panels[checkpoint_key] = None
            else:
                row = scenario_map[str(episode["scenario_id"])]
                for name, value in row.items():
                    if name in episode and episode[name] != value:
                        raise ValueError("Episode does not match its recorded scenario")
                panels.setdefault(checkpoint_key, {})[str(episode["scenario_id"])] = row
            steps, reward = int(episode["steps"]), float(episode["team_return"])
            if not 1 <= steps <= horizon or not np.isfinite(reward):
                raise ValueError("Episode steps/return are outside the finite evaluation contract")
            if sum(episode["composition"]) <= 0:
                raise ValueError("Episode composition must contain at least one agent")
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
                "checkpoint": metadata[(key, training_seed)],
                "successes": successes, "censored_completions": count - successes,
                "event_exposed_episodes": sum(bool(episode.get("event_exposed")) for episode in episodes),
                "scenario_ids": sorted(str(episode["scenario_id"]) for episode in episodes),
                "success_rate": successes / count,
                "team_return": float(np.mean([episode["team_return"] for episode in episodes])),
                # Derive from the physical roster for compatibility with archived reports.
                "mean_agent_return": float(np.mean([
                    episode["team_return"] / sum(episode["composition"]) for episode in episodes])),
                "completion_steps_horizon_capped": float(np.mean([
                    episode["steps"] if episode["success"] else horizon for episode in episodes])),
            })
        metrics = _uncertainty(per_seed, rng, bootstrap_samples)
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
        "paired_groups": _paired_summaries(groups, records, metadata, panels, rng, bootstrap_samples),
        "paired_method": "Within-checkpoint full-panel failure-minus-sham; equal training-seed weights and whole-seed bootstrap",
        "pairing_limitations": ["Pairing requires complete scenario records; legacy reports without them retain absolute summaries only",
                                "Failure/sham evaluation estimates sensor-loss effects; it does not by itself identify a gate mechanism"],
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return summary
