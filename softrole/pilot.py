"""Small native-composition unexpected-failure pilot, using the existing evaluator."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from softrole import CHECKPOINT_VERSION, ENVIRONMENT_VERSION, EVALUATION_VERSION
from softrole.config import Config
from softrole.evaluate import evaluate_checkpoint
from softrole.scenarios import make_scenarios
from softrole.train import source_snapshot, write_json


DIAGNOSTICS = ("pre_event_victim_reached", "pre_event_victim_target_visible",
               "pre_event_victim_target_seen")


def paired_summary(failure, sham):
    """Descriptive within-policy contrasts over the entire assigned panel.

    This is intentionally not a confidence interval or a cross-model estimator.
    Pair full scenario records, then verify the common prefix and diagnostics.
    """
    for field in ("scenarios", "checkpoint_sha256", "model_signature", "config",
                  "evaluation_version", "evaluator"):
        if failure[field] != sham[field]:
            raise ValueError(f"Failure/sham pairing differs in {field}")
    if failure["sham"] or not sham["sham"] or any(
            report["intervention"] != "none" for report in (failure, sham)):
        raise ValueError("Pilot pairing requires ordinary failure and matched sham")
    left = {row["scenario_id"]: row for row in failure["per_episode"]}
    right = {row["scenario_id"]: row for row in sham["per_episode"]}
    ids = [str(row["scenario_id"]) for row in failure["scenarios"]]
    if (not ids or len(set(ids)) != len(ids) or set(left) != set(ids)
            or set(right) != set(ids) or len(failure["per_episode"]) != len(ids)
            or len(sham["per_episode"]) != len(ids)):
        raise ValueError("Pairing requires exactly one outcome per assigned scenario")
    for scenario in failure["scenarios"]:
        a, b = left[str(scenario["scenario_id"])], right[str(scenario["scenario_id"])]
        for row in (a, b):
            if any(row[key] != scenario[key] for key in
                   ("env_seed", "action_seed", "message_seed", "num_p", "num_a", "event_step", "victim")):
                raise ValueError("Episode outcome differs from its assigned scenario")
        event = scenario["event_step"]
        if event < 0 or any(a[key] != b[key] for key in
                           (*DIAGNOSTICS, "scheduled_event_exposed")):
            raise ValueError("Paired pre-event diagnostics or exposure disagree")
        if a["trace"][:event] != b["trace"][:event]:
            raise ValueError("Failure/sham pre-event trajectory prefixes disagree")
    horizon = failure["config"]["max_steps"]

    def outcomes(rows):
        rows = list(rows.values())
        exposed = [row for row in rows if row["scheduled_event_exposed"]]
        return {
            "episodes": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "team_return": float(np.mean([row["team_return"] for row in rows])),
            "completion_steps_horizon_capped": float(np.mean([
                row["steps"] if row["success"] else horizon for row in rows])),
            "scheduled_event_exposed": len(exposed),
            "actual_event_exposed": sum(row["event_exposed"] for row in rows),
            "pre_event_successes": sum(row["pre_event_success"] for row in rows),
            "censored_completions": sum(not row["success"] for row in rows),
            "post_schedule_censored": sum(row["scheduled_censored"] for row in rows),
            "diagnostic_denominator": len(exposed),
            "diagnostic_true_counts": {key: sum(row[key] is True for row in exposed)
                                       for key in DIAGNOSTICS},
        }

    failed, control = outcomes(left), outcomes(right)
    return {
        "model": failure["config"]["model"], "training_seed": failure["training_seed"],
        "checkpoint": failure["checkpoint"], "checkpoint_sha256": failure["checkpoint_sha256"],
        "model_signature": failure["model_signature"],
        "training_source_sha256": failure["source_sha256"],
        "checkpoint_progress": failure["checkpoint_progress"],
        "failure": failed, "sham": control,
        "failure_minus_sham": {key: failed[key] - control[key] for key in
                               ("success_rate", "team_return", "completion_steps_horizon_capped")},
        "failure_only_successes": sum(left[i]["success"] and not right[i]["success"] for i in ids),
        "sham_only_successes": sum(right[i]["success"] and not left[i]["success"] for i in ids),
        "paired_prefixes_and_diagnostics_verified": True,
    }


def run_pilot(checkpoints, output, checkpoint_rule, episodes=20, seed=1700):
    """Evaluate nominal shared/banked pairs; never select checkpoints by outcomes."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")
    if type(episodes) is not int or episodes <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative integers")
    if not checkpoint_rule.strip():
        raise ValueError("Declare the checkpoint selection rule before evaluation")
    policies, common_config, training_source = {}, None, None
    for checkpoint in checkpoints:
        checkpoint = Path(checkpoint).resolve()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = Config(**saved["config"])
        if (saved.get("format_version") != CHECKPOINT_VERSION
                or saved.get("environment_version") != ENVIRONMENT_VERSION
                or saved.get("model_config") != config.model_kwargs()
                or not saved.get("source_sha256")):
            raise ValueError("Pilot requires a versioned checkpoint with training source identity")
        if (config.task != "pcp" or config.training_compositions != ((2, 1),)
                or (config.dim, config.vision, config.max_steps) != (5, 2, 80)
                or config.failure_prob != 0 or config.model not in ("shared", "banked")):
            raise ValueError("Pilot requires nominal shared/banked PCP 2P1A, map5, vision2, horizon80")
        comparison = config.to_dict()
        for field in ("model", "seed", "epochs", "total_steps", "save_every"):
            comparison.pop(field)
        if common_config is not None and comparison != common_config:
            raise ValueError("Pilot training configurations differ beyond model/seed/budget")
        common_config = comparison
        if training_source is not None and training_source != saved["source_sha256"]:
            raise ValueError("Pilot checkpoints must share their training source identity")
        training_source = saved["source_sha256"]
        progress = {name: saved.get(name) for name in
                    ("epoch", "updates", "total_steps", "total_episodes")}
        if any(type(value) is not int or value < 0 for value in progress.values()):
            raise ValueError("Pilot checkpoints must record nonnegative integer training progress")
        key = (config.seed, config.model)
        if key in policies:
            raise ValueError("Supply one checkpoint per model and training seed")
        policies[key] = {
            "checkpoint": str(checkpoint), "model": config.model, "training_seed": config.seed,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_source_sha256": saved["source_sha256"],
            "checkpoint_progress": progress,
        }
    if not policies or any((training_seed, model) not in policies
                           for training_seed, _ in policies for model in ("shared", "banked")):
        raise ValueError("Supply matched shared and banked checkpoints for each training seed")
    # Same composition namespace as `evaluate --compositions 2,1 --seed SEED`.
    panel_seed = int(np.random.SeedSequence([seed, 2, 1]).generate_state(1)[0])
    scenarios = make_scenarios(panel_seed, episodes, [(2, 1)], 1., (10, 30))
    output.mkdir(parents=True)
    source = source_snapshot(output)
    write_json(output / "scenarios.json", [asdict(scenario) for scenario in scenarios])
    manifest = {
        "evaluation_version": EVALUATION_VERSION,
        "purpose": "Native-composition unexpected-failure diagnostic pilot; nominal training only",
        "checkpoint_rule": checkpoint_rule, "policies": list(policies.values()),
        "composition": [2, 1], "episodes_per_policy_condition": episodes, "scenario_seed": seed,
        "failure_probability": 1., "failure_window": [10, 30], "intervention": "none",
        "scenarios_sha256": hashlib.sha256((output / "scenarios.json").read_bytes()).hexdigest(),
        "evaluator_source_sha256": source["sha256"],
    }
    # Written before any outcomes: an interrupted pilot retains its declared panel.
    write_json(output / "pilot.json", manifest)
    summaries = []
    for (training_seed, model), policy in sorted(policies.items()):
        reports = []
        for sham in (False, True):
            condition = "sham" if sham else "failure"
            report = evaluate_checkpoint(policy["checkpoint"], scenarios,
                     output / f"{model}_seed{training_seed}_{condition}.json", trace=True, sham=sham)
            if (report["evaluator"]["source"]["sha256"] != source["sha256"]
                    or report["checkpoint_sha256"] != policy["checkpoint_sha256"]):
                raise RuntimeError("Pilot source or checkpoint changed after its manifest was written")
            reports.append(report)
        summaries.append(paired_summary(*reports))
    summary = {
        "evaluation_version": EVALUATION_VERSION, "evaluator_source_sha256": source["sha256"],
        "method": "Per-policy paired full-panel means; descriptive only, no inferential intervals",
        "limitations": ["A tiny nominal-transfer pilot does not establish robustness or architecture superiority",
                        "Exposure-conditioned diagnostics can select different subsets across policies",
                        "Direct target observation does not measure knowledge obtained through messages",
                        "Checkpoint sample counts can differ; the manifest retains actual progress"],
        "policies": summaries,
    }
    write_json(output / "summary.json", summary)
    return summary
