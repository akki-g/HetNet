"""Locked CapCom proposal programme: arm table, S1 rate selection, manifests, launch.

Twenty-one arms x three seeds x 20M joint steps = 63 runs. Every arm changes
only its declared factors relative to its control; everything else is the
common protocol below. The learning rate is selected from the S1 pilot by a
rule fixed before any programme outcome exists, and the selection record is
embedded in the locked manifest together with the source identity that every
run must execute. ``launch`` refuses a checkout that differs from that source.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# The preparation CLI may run on a login node; cap numerical threads early.
if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

from softrole.config import COMPOSITION_TEST, COMPOSITION_TRAIN, FAILURE_TRAIN, Config

PROGRAMME = "capcom_proposals_2026-10-12"
MANIFEST_VERSION = 1
SEEDS = (0, 1, 2)
TOTAL_STEPS = 20_000_000
S1_RATES = (3e-4, 1e-3)
S1_WINDOW = (8_000_000, 12_000_000)
S1_RULE = ("Equal-seed mean horizon-capped completion over episodes of complete updates whose "
           "step endpoint lies in (8M, 12M]; the lower mean wins; ties select the lower rate; "
           "every rate needs all three seeds with the window fully covered")
DEVELOPMENT_WINDOWS = ((4_000_000, 8_000_000), (8_000_000, 12_000_000), (18_000_000, 20_000_000))
RUNTIME_KEYS = ("epochs", "updates_per_epoch", "batch_steps", "nprocesses", "total_steps", "save_every")
COMMON = dict(task="pcp", env_version="corrected-observation-v1", dim=5, vision=2, max_steps=80,
              nfires=1, reward_type=3, experts=4, pre_dim=128, hidden_dim=64, heads=4, head_dim=16,
              comm_rounds=3, independent_heads=True, feedback=True, comm_range=-1.0,
              epochs=2000, updates_per_epoch=10, batch_steps=500, nprocesses=4, total_steps=TOTAL_STEPS,
              optimizer="adam", gamma=1.0, gae_lambda=0.95, actor_coeff=50.0, value_coeff=1.0,
              detach_gap=5, max_grad_norm=0.75, save_every=10)
PRESETS = {
    "fixed": dict(num_p=2, num_a=1, compositions=(), held_out=(), failure_prob=0.0, failure_window=(10, 30)),
    "composition": dict(num_p=2, num_a=1, compositions=COMPOSITION_TRAIN, held_out=COMPOSITION_TEST,
                        failure_prob=0.0, failure_window=(10, 30)),
    # Zero-indexed event step 1, probability .5, uniform victim among sensing agents.
    "failure": dict(num_p=2, num_a=1, compositions=FAILURE_TRAIN, held_out=COMPOSITION_TEST,
                    failure_prob=0.5, failure_window=(1, 1)),
}
FACTOR_DEFAULTS = dict(model="shared", communication="real", msg_dim=64, null_degree_correction=False,
                       critic_mode="team", credit_beta=1.0, objective="team", actor_capabilities="explicit")


def _arm(name, stage, preset, control, **factors):
    unknown = set(factors) - set(FACTOR_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown arm factors: {sorted(unknown)}")
    return {"name": name, "stage": stage, "preset": preset, "control": control,
            "factors": dict(FACTOR_DEFAULTS, **factors), "changed": sorted(factors)}


ARMS = (
    # C1: channel factorial on the nominal fixed 2P1A team, shared actor.
    _arm("c1_real64", "C1", "fixed", None),
    _arm("c1_real_noisy64", "C1", "fixed", "c1_real64", communication="real_noisy"),
    _arm("c1_binary64", "C1", "fixed", "c1_real64", communication="binary"),
    _arm("c1_binary_deterministic64", "C1", "fixed", "c1_binary64", communication="binary_deterministic"),
    _arm("c1_real16", "C1", "fixed", "c1_real64", msg_dim=16),
    _arm("c1_binary16", "C1", "fixed", "c1_binary64", communication="binary", msg_dim=16),
    # A1: composition-trained shared Real64, original versus degree-corrected null.
    _arm("a1_composition_original_null", "A1", "composition", None),
    _arm("a1_composition_corrected_null", "A1", "composition", "a1_composition_original_null",
         null_degree_correction=True),
    # L1b then L1a: decomposed critic at beta 1, then credit mixing; C1 Real64 is the control.
    _arm("l1b_decomposed_beta1", "L1", "fixed", "c1_real64", critic_mode="decomposed", credit_beta=1.0),
    _arm("l1a_decomposed_beta05", "L1", "fixed", "l1b_decomposed_beta1", critic_mode="decomposed", credit_beta=0.5),
    _arm("l1a_decomposed_beta0", "L1", "fixed", "l1b_decomposed_beta1", critic_mode="decomposed", credit_beta=0.0),
    # O1: physical-only actor input; C1 Real64 is the control.
    _arm("o1_physical_only", "O1", "fixed", "c1_real64", actor_capabilities="physical_only"),
    # Banks under the composition and failure presets with the team objective; the
    # shared composition arm is reused from A1. The shared failure arm's control
    # differs in the training composition set as well as in failure exposure.
    _arm("banks_composition_banked", "BANKS", "composition", "a1_composition_original_null", model="banked"),
    _arm("banks_composition_capability", "BANKS", "composition", "a1_composition_original_null", model="capability"),
    _arm("banks_composition_constant", "BANKS", "composition", "a1_composition_original_null", model="constant"),
    _arm("banks_failure_shared", "BANKS", "failure", "a1_composition_original_null"),
    _arm("banks_failure_banked", "BANKS", "failure", "banks_failure_shared", model="banked"),
    _arm("banks_failure_capability", "BANKS", "failure", "banks_failure_shared", model="capability"),
    _arm("banks_failure_constant", "BANKS", "failure", "banks_failure_shared", model="constant"),
    # L1c: shared mean-agent objective per preset.
    _arm("l1c_composition_mean_agent", "L1C", "composition", "a1_composition_original_null", objective="mean_agent"),
    _arm("l1c_failure_mean_agent", "L1C", "failure", "banks_failure_shared", objective="mean_agent"),
)
ARM_INDEX = {arm["name"]: arm for arm in ARMS}
STAGES = ("C1", "A1", "L1", "O1", "BANKS", "L1C")


def programme_config(arm_name, seed, lr):
    arm = ARM_INDEX[arm_name]
    return Config(**COMMON, **PRESETS[arm["preset"]], **arm["factors"], lr=lr, seed=seed)


def normalized(config):
    """JSON form of a configuration (tuples become lists), as manifests store it."""
    return json.loads(json.dumps(config.to_dict(), sort_keys=True))


def arm_table():
    rows = []
    for arm in ARMS:
        control = ARM_INDEX[arm["control"]] if arm["control"] else None
        changed = sorted(key for key in FACTOR_DEFAULTS
                         if control is None or arm["factors"][key] != control["factors"][key])
        rows.append({**arm, "changed_from_control": changed,
                     "preset_differs_from_control": bool(control) and control["preset"] != arm["preset"]})
    return {"programme": PROGRAMME, "arms": rows, "seeds": list(SEEDS), "runs": len(ARMS) * len(SEEDS),
            "total_steps_per_run": TOTAL_STEPS, "stages": {stage: sum(a["stage"] == stage for a in ARMS) for stage in STAGES}}


def entries(lr):
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            config = programme_config(arm["name"], seed, lr)
            rows.append({"index": len(rows), "arm": arm["name"], "stage": arm["stage"], "seed": seed,
                         "control": arm["control"], "relative_output": f"{arm['stage']}/{arm['name']}/seed{seed}",
                         "config": normalized(config)})
    return rows


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def window_statistics(run, windows, horizon):
    """Episode-weighted statistics of complete updates whose step endpoint lies in (lo, hi].

    Reads the run's update and episode ledgers; a stdout-logged run has no
    episode file and is rejected. ``complete`` says whether the run's last
    recorded update reaches the window end.
    """
    run = Path(run)
    update_steps = {}
    with (run / "updates.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            update_steps[row["update"]] = row["total_steps"]
    if not update_steps:
        raise ValueError(f"{run} has no recorded updates")
    last = max(update_steps.values())
    windows = [tuple(int(x) for x in window) for window in windows]
    if any(len(w) != 2 or w[0] < 0 or w[1] <= w[0] for w in windows):
        raise ValueError("Windows must be (start_exclusive, end_inclusive) with start < end")
    stats = {w: dict(episodes=0, steps=0, successes=0, team_return=0., mean_agent_return=0.,
                     capped_completion=0., event_exposed=0) for w in windows}
    with (run / "episodes.jsonl").open() as stream:
        for line in stream:
            episode = json.loads(line)
            if episode["update"] not in update_steps:
                raise ValueError(f"{run}: episode ledger references an unrecorded update")
            endpoint = update_steps[episode["update"]]
            for (lo, hi), row in stats.items():
                if lo < endpoint <= hi:
                    row["episodes"] += 1
                    row["steps"] += episode["steps"]
                    row["successes"] += bool(episode["success"])
                    row["team_return"] += episode["team_return"]
                    row["mean_agent_return"] += (episode["mean_agent_return"] if episode.get("mean_agent_return") is not None
                                                 else episode["team_return"] / episode["num_agents"])
                    row["capped_completion"] += episode["steps"] if episode["success"] else horizon
                    row["event_exposed"] += bool(episode.get("event_exposed"))
    result = {}
    for (lo, hi), row in stats.items():
        n = row["episodes"]
        result[f"{lo}-{hi}"] = {
            "start_exclusive": lo, "end_inclusive": hi, "episodes": n, "steps": row["steps"],
            "complete": last >= hi, "event_exposed_episodes": row["event_exposed"],
            "success_rate": row["successes"] / n if n else None,
            "capped_completion": row["capped_completion"] / n if n else None,
            "team_return": row["team_return"] / n if n else None,
            "mean_agent_return": row["mean_agent_return"] / n if n else None,
        }
    return result, last


def _strip_runtime(config_dict):
    return {k: v for k, v in config_dict.items() if k not in RUNTIME_KEYS}


def select_learning_rate(runs, output, window=S1_WINDOW, bounded=False):
    """Apply the fixed S1 rule to the six pilot runs and record the selection.

    ``bounded`` admits engineering runs with reduced budgets and a smaller
    window; such a selection is marked and can only feed a bounded manifest.
    """
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    window = tuple(int(x) for x in window)
    if not bounded and window != S1_WINDOW:
        raise ValueError("The research selection uses the fixed (8M, 12M] window")
    records = []
    for run in runs:
        run = Path(run)
        config = Config(**read_json(run / "config.json"))
        if config.lr not in S1_RATES or config.seed not in SEEDS:
            raise ValueError(f"{run}: not an S1 rate/seed")
        expected = programme_config("c1_real64", config.seed, config.lr).to_dict()
        actual = config.to_dict()
        if (_strip_runtime(actual) != _strip_runtime(expected)
                or (not bounded and actual != expected)):
            raise ValueError(f"{run}: configuration differs from the S1 Real64 protocol")
        stats, last = window_statistics(run, [window], config.max_steps)
        row = stats[f"{window[0]}-{window[1]}"]
        if not row["complete"] or not row["episodes"]:
            raise ValueError(f"{run}: window not fully covered; partial comparisons are refused")
        inputs = {str(run / name): sha(run / name) for name in ("config.json", "updates.jsonl", "episodes.jsonl")}
        records.append({"run": str(run.resolve()), "lr": config.lr, "seed": config.seed,
                        "last_recorded_steps": last, "window": row, "inputs": inputs})
    by_rate = {}
    for row in records:
        by_rate.setdefault(row["lr"], {})
        if row["seed"] in by_rate[row["lr"]]:
            raise ValueError(f"Duplicate S1 run for rate {row['lr']} seed {row['seed']}")
        by_rate[row["lr"]][row["seed"]] = row
    if set(by_rate) != set(S1_RATES) or any(set(seeds) != set(SEEDS) for seeds in by_rate.values()):
        raise ValueError("Every S1 rate needs all three seeds; partial comparisons are refused")
    means = {lr: sum(by_rate[lr][s]["window"]["capped_completion"] for s in SEEDS) / len(SEEDS) for lr in S1_RATES}
    best = min(means.values())
    selected = min(lr for lr in S1_RATES if math.isclose(means[lr], best, rel_tol=0., abs_tol=1e-12))
    record = {"rule": S1_RULE, "window": list(window), "metric": "capped_completion",
              "equal_seed_means": {f"{lr:g}": means[lr] for lr in S1_RATES},
              "selected_lr": selected, "bounded_check": bool(bounded), "runs": records}
    write_json(destination, record)
    return record


def prepare(lr_selection, output, diagnostics_every=50, bounded=False):
    """Write the locked 63-entry manifest from a recorded S1 selection and this checkout."""
    from softrole.train import source_snapshot
    selection_path = Path(lr_selection)
    selection = read_json(selection_path)
    if selection.get("rule") != S1_RULE or selection.get("selected_lr") not in S1_RATES:
        raise ValueError("Learning-rate selection record does not follow the fixed S1 rule")
    if selection.get("bounded_check") and not bounded:
        raise ValueError("A bounded (engineering) selection cannot lock the research manifest")
    if diagnostics_every is not None and (type(diagnostics_every) is not int or diagnostics_every < 1):
        raise ValueError("diagnostics_every must be a positive integer or None")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    source = source_snapshot()
    lr = float(selection["selected_lr"])
    manifest = {"version": MANIFEST_VERSION, "programme": PROGRAMME, "bounded_check": bool(bounded),
                "lr": lr, "lr_selection_sha256": sha(selection_path), "lr_selection": selection,
                "source_sha256": source["sha256"], "git_commit": source["git_commit"],
                "diagnostics_every": diagnostics_every, "runtime_override_keys": list(RUNTIME_KEYS),
                "common": COMMON, "presets": json.loads(json.dumps(PRESETS)),
                "arms": arm_table()["arms"], "entries": entries(lr)}
    write_json(destination, manifest)
    return manifest


def runtime_overrides(options):
    overrides = {}
    for key, value in (options or {}).items():
        if key not in RUNTIME_KEYS:
            raise ValueError(f"{key} is not a runtime option; scientific settings come from the manifest")
        if value is None:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        overrides[key] = value
    return overrides


def launch(manifest_path, index, run_root, dry_run=False, overrides=None, episode_log="file",
           diagnostics_every=None):
    """Train manifest entry ``index`` under ``run_root``; record provenance beside the run."""
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("programme") != PROGRAMME:
        raise ValueError("Unsupported programme manifest")
    if type(index) is not int or not 0 <= index < len(manifest["entries"]):
        raise ValueError("Entry index outside the manifest")
    entry = manifest["entries"][index]
    if entry["index"] != index:
        raise ValueError("Manifest entry order is corrupted")
    expected = normalized(programme_config(entry["arm"], entry["seed"], manifest["lr"]))
    if entry["config"] != expected or entry["seed"] not in SEEDS:
        raise ValueError("Manifest entry configuration differs from the programme arm table")
    overrides = runtime_overrides(overrides)
    config = Config(**dict(entry["config"], **overrides))
    if diagnostics_every is None:
        diagnostics_every = manifest.get("diagnostics_every")
    if diagnostics_every is not None and (type(diagnostics_every) is not int or diagnostics_every < 1):
        raise ValueError("diagnostics_every must be a positive integer")
    output = Path(run_root) / entry["relative_output"]
    record_path = output.parent / (output.name + ".programme.json")
    if dry_run:
        return config
    if output.exists() or record_path.exists():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    from softrole.train import source_snapshot, train
    source = source_snapshot()
    if source["sha256"] != manifest["source_sha256"]:
        raise ValueError("Checkout differs from the locked manifest source; prepare the manifest on the "
                         "final fixed checkout or restore that checkout before launching")
    write_json(record_path, {"programme": PROGRAMME, "manifest": str(manifest_path.resolve()),
                             "manifest_sha256": sha(manifest_path), "index": index, "arm": entry["arm"],
                             "stage": entry["stage"], "seed": entry["seed"], "control": entry["control"],
                             "lr": manifest["lr"], "source_sha256": source["sha256"],
                             "runtime_overrides": overrides, "episode_log": episode_log,
                             "diagnostics_every": diagnostics_every,
                             "bounded_check": bool(manifest.get("bounded_check")) or bool(overrides),
                             "output": str(output.resolve())})
    train(config, output, episode_log=episode_log, diagnostics_every=diagnostics_every)
    return config


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("arms", help="print the locked arm table")
    select = commands.add_parser("select-lr", help="apply the fixed S1 learning-rate rule")
    select.add_argument("--runs", nargs="+", required=True, help="the six S1 run directories")
    select.add_argument("--output", required=True)
    select.add_argument("--window", nargs=2, type=int, default=list(S1_WINDOW),
                        help="engineering checks only; the research window is fixed")
    select.add_argument("--bounded", action="store_true", help="engineering runs with reduced budgets")
    prep = commands.add_parser("prepare", help="write the locked 63-entry manifest")
    prep.add_argument("--lr-selection", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--diagnostics-every", type=positive, default=50)
    prep.add_argument("--bounded", action="store_true", help="lock a manifest from a bounded selection")
    run = commands.add_parser("launch", help="train one manifest entry")
    run.add_argument("--manifest", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--episode-log", choices=("file", "stdout"), default="file")
    run.add_argument("--diagnostics-every", type=positive)
    for key in RUNTIME_KEYS:
        run.add_argument("--" + key.replace("_", "-"), dest=key, type=positive)
    args = parser.parse_args(argv)
    try:
        if args.command == "arms":
            print(json.dumps(arm_table(), indent=2))
        elif args.command == "select-lr":
            record = select_learning_rate(args.runs, args.output, tuple(args.window), args.bounded)
            print(json.dumps({"selected_lr": record["selected_lr"], "equal_seed_means": record["equal_seed_means"],
                              "bounded_check": record["bounded_check"]}, indent=2))
        elif args.command == "prepare":
            manifest = prepare(args.lr_selection, args.output, args.diagnostics_every, args.bounded)
            print(json.dumps({"entries": len(manifest["entries"]), "lr": manifest["lr"],
                              "source_sha256": manifest["source_sha256"], "manifest": args.output}))
        else:
            overrides = {key: getattr(args, key) for key in RUNTIME_KEYS}
            config = launch(args.manifest, args.index, args.run_root, args.dry_run, overrides,
                            args.episode_log, args.diagnostics_every)
            if args.dry_run:
                print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
