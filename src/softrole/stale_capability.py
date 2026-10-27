"""Locked stale-capability stability gate; no main-study or evaluation launcher."""
import argparse
from bisect import bisect_right
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from softrole.programme import CAPCON_PROFILE, normalized, programme_config
from softrole.train import source_snapshot, train, write_json


VERSION = 1
STUDY = "stale_capability_stability_gate_v1"
GATE_COEFFICIENTS = (0.003, 0.01)
GATE_SEEDS = (0, 1, 2)
GATE_BUDGET = 2_000_000
GATE_WARMUP_STEPS = 1_000_000
GATE_WINDOW_STEPS = 500_000
GATE_MIN_SUCCESS = 0.05
GATE_MIN_ENTROPY = 0.01
GATE_PROTOCOL = {
    "purpose": "Exploratory training-stability gate; no held-out evaluation or superiority claim",
    "primary_arm": "l1b_decomposed_beta1",
    "primary_profile": CAPCON_PROFILE,
    "changed_variable": "entropy_coeff",
    "coefficients": list(GATE_COEFFICIENTS),
    "seeds": list(GATE_SEEDS),
    "target_joint_environment_steps_per_run": GATE_BUDGET,
    "stopping": "first complete collector batch reaching the target; record actual counts",
    "checkpoint_rule": "first_saved_at_or_above the training budget; never best epoch",
    "collapse_rule": {
        "after_steps": GATE_WARMUP_STEPS,
        "window_steps": GATE_WINDOW_STEPS,
        "success_below": GATE_MIN_SUCCESS,
        "entropy_below_nats_per_agent_step": GATE_MIN_ENTROPY,
        "comparison": "strictly below either threshold",
        "episode_order": "recorded ledger order with cumulative episode-completion steps",
        "success_windows": "every full sliding 500000-step interval contained in [1000000, final recorded step]",
        "window_membership": "episode completion step in (start, end]",
        "success_rate": "successful complete episodes divided by complete episodes in the window",
        "entropy_updates": "every update ending strictly after 1000000 steps",
        "entropy_mean": "action_entropy_sum / sum(episode.steps * episode.num_agents) within the update",
    },
    "selection": "user confirms the smallest coefficient with no collapsed seed; no automatic selection",
    "both_coefficients_collapse": "stop and report; no further coefficients without user authorization",
    "settings_authority": "user-confirmed gate settings and sliding-window convention on 2026-10-26",
    "evaluation": "none; training diagnostics only",
    "episode_log": "chunked-gzip",
    "checkpoint_commits": True,
    "device": "cpu",
    "dtype": "float64",
    "threads_per_collector": 1,
}
PROTOCOL = GATE_PROTOCOL


def gate_config(coefficient, seed):
    if type(coefficient) not in (int, float) or coefficient not in GATE_COEFFICIENTS:
        raise ValueError("gate coefficient must be one of the confirmed values 0.003 and 0.01")
    if type(seed) is not int or seed not in GATE_SEEDS:
        raise ValueError("gate seed must be 0, 1 or 2")
    primary = programme_config("l1b_decomposed_beta1", seed, 1e-3, profile=CAPCON_PROFILE)
    config = replace(primary, capabilities="graded", descriptor="capability",
                     compositions=((1, 1), (2, 1)), sensing_p=(1, 2, 3), sensing_a=(0,),
                     held_out_sensing_p=(), held_out_sensing_a=(), failure_prob=0.0,
                     env_version="graded-observation-v2", actor_capabilities="explicit",
                     entropy_coeff=float(coefficient), total_steps=GATE_BUDGET)
    changed = {key for key in primary.__dataclass_fields__
               if getattr(primary, key) != getattr(config, key)}
    allowed = {"capabilities", "descriptor", "compositions", "sensing_p", "sensing_a",
               "held_out_sensing_p", "held_out_sensing_a", "failure_prob", "env_version",
               "actor_capabilities", "entropy_coeff", "total_steps"}
    if changed - allowed:
        raise ValueError("Gate settings differ from the primary arm beyond declared fields")
    return config


def expected_entries():
    entries = []
    for coefficient in GATE_COEFFICIENTS:
        arm = "a1_entropy_" + str(coefficient).replace(".", "p")
        for seed in GATE_SEEDS:
            entries.append(dict(index=len(entries), arm=arm, coefficient=coefficient, seed=seed,
                                relative_output=f"{arm}/seed{seed}",
                                config=normalized(gate_config(coefficient, seed))))
    return entries


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
            "version", "study", "stage", "protocol", "entries", "source", "manifest_sha256"}:
        raise ValueError("Unsupported stale-capability gate manifest")
    if (type(manifest["version"]) is not int or manifest["version"] != VERSION
            or manifest["study"] != STUDY or manifest["stage"] != "gate"
            or digest(manifest["protocol"]) != digest(GATE_PROTOCOL)
            or digest(manifest["entries"]) != digest(expected_entries())):
        raise ValueError("Manifest differs from the locked stability gate")
    source = manifest["source"]
    if not isinstance(source, dict) or not isinstance(source.get("files"), dict) or not source["files"]:
        raise ValueError("Manifest requires a source file inventory")
    for name, value in source["files"].items():
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("Malformed source file inventory")
    if source.get("sha256") != digest(source["files"]):
        raise ValueError("Manifest source inventory hash differs")
    if manifest["manifest_sha256"] != digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Manifest hash differs")
    return manifest


def prepare(output, stage="gate", dry_run=False):
    if stage != "gate":
        raise ValueError("Only the stability-gate stage is implemented")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Gate manifest already exists: {output}")
    manifest = dict(version=VERSION, study=STUDY, stage=stage, protocol=GATE_PROTOCOL,
                    entries=expected_entries(), source=source_snapshot())
    manifest["manifest_sha256"] = digest(manifest)
    validate_manifest(manifest)
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def launch(manifest_path, index, run_root, dry_run=False):
    if type(index) is not int or index not in range(len(GATE_COEFFICIENTS) * len(GATE_SEEDS)):
        raise ValueError("gate index must be an integer from 0 through 5")
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    source = source_snapshot()
    if source["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Source changed since gate preparation; use the frozen checkout")
    entry = manifest["entries"][index]
    root = Path(run_root).resolve()
    output = root / entry["relative_output"]
    if not output.resolve().is_relative_to(root):
        raise ValueError("Gate output must stay inside the run root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Gate output already exists; fresh runs only: {output}")
    record = dict(study=STUDY, stage="gate", index=index, arm=entry["arm"],
                  coefficient=entry["coefficient"], seed=entry["seed"], config=entry["config"],
                  output=str(output), dry_run=bool(dry_run), source_sha256=source["sha256"],
                  manifest_sha256=manifest["manifest_sha256"])
    if dry_run:
        return record
    checkpoint = train(gate_config(entry["coefficient"], entry["seed"]), output,
                       episode_log="chunked-gzip", checkpoint_commits=True)
    if source_snapshot()["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed during the gate; inspect the run before using its results")
    record["checkpoint"] = str(checkpoint)
    write_json(output / "stale_capability.json", record)
    return record


def _gate_windows(completions, successes, final_step, start, width, threshold):
    """Inspect all distinct (start,end] memberships, including expiration events.

    Episode order is the recorded collector/episode order, not wall-clock order.
    Membership changes only at a completion or at completion + width. Checking
    both families also covers intervals between consecutive completions.
    """
    first_end = start + width
    if final_step < first_end:
        return {"windows_checked": 0, "violating_windows": 0, "worst_window": None}
    ends = {first_end, final_step}
    ends.update(step for step in completions if first_end <= step <= final_step)
    ends.update(step + width for step in completions if first_end <= step + width <= final_step)
    prefix = [0]
    for success in successes:
        prefix.append(prefix[-1] + int(success))
    worst = None
    violations = 0
    for end in sorted(ends):
        left, right = bisect_right(completions, end - width), bisect_right(completions, end)
        count = right - left
        if not count:
            raise ValueError("A full gate window contains no complete episode")
        wins = prefix[right] - prefix[left]
        window = dict(start_exclusive=end - width, end_inclusive=end,
                      episodes=count, successes=wins, success_rate=wins / count)
        violations += window["success_rate"] < threshold
        if worst is None or wins * worst["episodes"] < worst["successes"] * count:
            worst = window
    return dict(windows_checked=len(ends), violating_windows=violations, worst_window=worst)


def _gate_statistics(updates, episodes):
    """Compute the confirmed gate rule on independently audited episode rows."""
    completions, successes = [], []
    total_steps = 0
    agent_steps = {}
    for episode in episodes:
        total_steps += episode["steps"]
        completions.append(total_steps)
        successes.append(episode["success"])
        update = episode["update"]
        agent_steps[update] = agent_steps.get(update, 0) + episode["steps"] * episode["num_agents"]
    if not updates or updates[-1]["total_steps"] != total_steps:
        raise ValueError("Gate episode/update totals disagree")
    minimum = None
    violations = []
    checked = 0
    for update in updates:
        entropy = update.get("action_entropy_sum")
        if (isinstance(entropy, bool) or not isinstance(entropy, (int, float))
                or not math.isfinite(entropy) or entropy < 0):
            raise ValueError("Gate update requires finite nonnegative action_entropy_sum")
        denominator = agent_steps.get(update["update"], 0)
        if denominator <= 0:
            raise ValueError("Gate update has no agent-steps")
        if update["total_steps"] <= GATE_WARMUP_STEPS:
            continue
        checked += 1
        row = dict(update=update["update"], total_steps=update["total_steps"],
                   action_entropy_sum=entropy, agent_steps=denominator,
                   mean_action_entropy=entropy / denominator)
        if minimum is None or row["mean_action_entropy"] < minimum["mean_action_entropy"]:
            minimum = row
        if row["mean_action_entropy"] < GATE_MIN_ENTROPY:
            violations.append(row)
    windows = _gate_windows(completions, successes, total_steps, GATE_WARMUP_STEPS,
                            GATE_WINDOW_STEPS, GATE_MIN_SUCCESS)
    complete = total_steps >= GATE_BUDGET
    collapsed = bool(windows["violating_windows"] or violations)
    return dict(total_steps=total_steps, total_episodes=len(completions),
                status=("collapsed" if collapsed else "pass") if complete else "incomplete",
                collapsed=collapsed if complete else None, success_windows=windows,
                entropy_updates_checked=checked, minimum_entropy=minimum,
                entropy_violations=violations)


def _gate_seed_verdict(manifest, entry, run_root):
    from softrole.ledger import (audit_run, committed_checkpoints, iter_episodes,
                                 ledger_files, sha)

    root = Path(run_root).resolve()
    run = root / entry["relative_output"]
    if not run.resolve().is_relative_to(root):
        raise ValueError("Gate evidence escapes the run root")
    identity = dict(index=entry["index"], seed=entry["seed"], arm=entry["arm"], run=str(run))
    if not (run / "finished.json").is_file() or not (run / "stale_capability.json").is_file():
        return dict(identity, status="incomplete", collapsed=None,
                    reason="Missing training completion or successful post-training source-check receipt")
    if (run / "attempt.json").exists() or (run / "attempts").exists():
        raise ValueError("The gate supports fresh runs only, not continuations")
    read = lambda name: json.loads((run / name).read_text())
    if read("config.json") != entry["config"]:
        raise ValueError("Gate run configuration differs from its manifest")
    source = read("source_manifest.json")
    if (source.get("sha256") != manifest["source"]["sha256"]
            or source.get("files") != manifest["source"]["files"]):
        raise ValueError("Gate run source differs from its prepared manifest")
    for name, checksum in source["files"].items():
        if sha(run / "source" / name) != checksum:
            raise ValueError(f"Gate archived source changed: {name}")
    receipt = read("stale_capability.json")
    expected = dict(study=STUDY, stage="gate", index=entry["index"], arm=entry["arm"],
                    coefficient=entry["coefficient"], seed=entry["seed"], config=entry["config"],
                    dry_run=False, source_sha256=source["sha256"],
                    manifest_sha256=manifest["manifest_sha256"])
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Gate launch receipt differs from its manifest")
    runtime = read("run.json")
    if (runtime.get("source_sha256") != source["sha256"]
            or runtime.get("environment_version") != entry["config"]["env_version"]
            or runtime.get("dtype") != "float64" or runtime.get("resume") is not None
            or runtime.get("episode_log") != "chunked-gzip"
            or runtime.get("checkpoint_commits") is not True):
        raise ValueError("Gate run runtime/source contract differs")
    evidence = [run / name for name in ("config.json", "source_manifest.json", "run.json",
                                        "finished.json", "stale_capability.json", "updates.jsonl")]
    evidence += ledger_files(run)
    evidence_ids = {str(path.relative_to(run)): sha(path) for path in evidence}
    audit = audit_run(run)
    finished = read("finished.json")
    if any(finished.get(key) != audit[key] for key in ("total_steps", "total_episodes")) or (
            finished.get("updates") != audit["covered_updates"][1]):
        raise ValueError("Gate completion counts differ from audited ledgers")
    if audit["total_steps"] < GATE_BUDGET:
        return dict(identity, status="incomplete", collapsed=None,
                    total_steps=audit["total_steps"], reason="Training budget was not reached")
    if len(audit["updates"]) > 1 and audit["updates"][-2]["total_steps"] >= GATE_BUDGET:
        raise ValueError("Gate training continued beyond the first batch reaching its budget")
    checkpoints = committed_checkpoints(run)
    eligible = [row for row in checkpoints if row["total_steps"] >= GATE_BUDGET]
    if not eligible:
        raise ValueError("Gate has no committed checkpoint at its training budget")
    checkpoint = eligible[0]
    if (any(checkpoint[key] != finished[key] for key in ("updates", "total_steps", "total_episodes"))
            or checkpoint["source_sha256"] != source["sha256"]
            or Path(receipt.get("checkpoint", "")).name != Path(checkpoint["checkpoint"]).name
            or Path(finished.get("checkpoint", "")).name != Path(checkpoint["checkpoint"]).name):
        raise ValueError("Gate checkpoint differs from its completion/source records")
    import torch
    from softrole.config import Config
    checkpoint_path = Path(checkpoint["checkpoint"])
    checkpoint_files = [checkpoint_path, Path(str(checkpoint_path) + ".signature.json"),
                        Path(str(checkpoint_path) + ".complete.json")]
    evidence_ids.update({str(path.relative_to(run)): sha(path) for path in checkpoint_files})
    if evidence_ids[str(checkpoint_path.relative_to(run))] != checkpoint["sha256"]:
        raise ValueError("Gate checkpoint changed while summarizing")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_config = gate_config(entry["coefficient"], entry["seed"])
    if (normalized(Config(**saved["config"])) != entry["config"]
            or saved.get("model_config") != expected_config.model_kwargs()
            or saved.get("environment_version") != expected_config.env_version
            or any(value.is_floating_point() and value.dtype != torch.float64
                   for value in saved["model_state"].values())):
        raise ValueError("Gate checkpoint configuration/environment/dtype differs from its manifest")
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
    statistics = _gate_statistics(updates, iter_episodes(run))
    if any(sha(run / relative) != checksum for relative, checksum in evidence_ids.items()):
        raise ValueError("Gate evidence changed while summarizing")
    return dict(identity, **statistics, checkpoint=checkpoint,
                audit_sha256=audit["audit_sha256"], evidence_sha256=evidence_ids)


def summarize_gate(manifest_path, run_root, output):
    """Write per-coefficient evidence; selecting a coefficient remains user work."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Gate summary already exists: {output}")
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    summarizer_source = source_snapshot()["sha256"]
    coefficients = []
    for coefficient in GATE_COEFFICIENTS:
        seeds = []
        for entry in manifest["entries"]:
            if entry["coefficient"] != coefficient:
                continue
            try:
                seeds.append(_gate_seed_verdict(manifest, entry, run_root))
            except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
                seeds.append(dict(index=entry["index"], seed=entry["seed"], arm=entry["arm"],
                                  status="error", collapsed=None, reason=str(error)))
        statuses = {seed["status"] for seed in seeds}
        status = next((value for value in ("error", "incomplete", "collapsed") if value in statuses), "pass")
        coefficients.append(dict(coefficient=coefficient, status=status, seeds=seeds))
    eligible = [row["coefficient"] for row in coefficients if row["status"] == "pass"]
    complete = all(row["status"] in ("pass", "collapsed") for row in coefficients)
    decision = ("awaiting_user_coefficient_confirmation" if complete and eligible else
                "both_coefficients_collapsed_stop" if complete else "awaiting_complete_valid_evidence")
    result = dict(version=VERSION, study=STUDY, stage="gate", protocol=GATE_PROTOCOL,
                  manifest_sha256=manifest["manifest_sha256"], source_sha256=manifest["source"]["sha256"],
                  summarizer_source_sha256=summarizer_source,
                  coefficients=coefficients, eligible_coefficients=eligible,
                  selected_coefficient=None, decision=decision)
    if source_snapshot()["sha256"] != summarizer_source:
        raise RuntimeError("Summarizer source changed during the gate summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = cli.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", allow_abbrev=False)
    preparation.add_argument("--stage", choices=("gate",), required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--dry-run", action="store_true")
    execution = commands.add_parser("launch", allow_abbrev=False)
    execution.add_argument("--manifest", type=Path, required=True)
    execution.add_argument("--index", type=int, choices=range(6), required=True)
    execution.add_argument("--run-root", type=Path, required=True)
    execution.add_argument("--dry-run", action="store_true")
    summary = commands.add_parser("gate-summary", allow_abbrev=False)
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.output, stage=args.stage, dry_run=args.dry_run)
        elif args.command == "launch":
            result = launch(args.manifest, args.index, args.run_root, args.dry_run)
        else:
            result = summarize_gate(args.manifest, args.run_root, args.output)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
