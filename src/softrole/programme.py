"""Versioned CapCom/CapCon profiles: S1 certification, manifests and launch.

The historical default remains 21 arms x three seeds x 20M joint steps. The
explicit CapCon profile selects four existing learners x three seeds x 40M,
using certified Adam 0.001. Its fixed S1 evidence retains the historical 20M
recipe. Every manifest locks the complete configuration table, selection
record and source identity. ``launch`` refuses a different source checkout.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from fractions import Fraction
import shutil
import fcntl
import os
from pathlib import Path
import sys

# The preparation CLI may run on a login node; cap numerical threads early.
if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

from softrole.config import COMPOSITION_TEST, COMPOSITION_TRAIN, FAILURE_TRAIN, Config

PROGRAMME = "capcom_proposals_2026-10-12"
CAPCON_PROFILE = "capcon_l1_40m_v1"
PROGRAMMES = (PROGRAMME, CAPCON_PROFILE)
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
CAPCON_ARMS = ("c1_real64", "l1b_decomposed_beta1", "l1a_decomposed_beta05", "l1a_decomposed_beta0")
CAPCON_CHECKPOINT_POLICY = {"rule": "first_saved_at_or_above", "target_steps": 40_000_000}
CAPCON_EVALUATION_POLICY = {
    "phase": "nominal-selection", "compositions": [[2, 1]], "conditions": ["nominal"],
    "panel_seed": 2704, "episodes_per_composition": 500,
    "selection_metric": "completion_steps_horizon_capped", "weighting": "equal_training_seed_means",
    "tie_order": list(CAPCON_ARMS),
}


def profile_definition(profile=PROGRAMME):
    """Resolve a versioned protocol without changing the historical S1 recipe."""
    if profile == PROGRAMME:
        return {"arms": ARMS, "common": dict(COMMON), "presets": PRESETS, "stages": STAGES,
                "total_steps": TOTAL_STEPS, "checkpoint_policy": None, "evaluation_policy": None}
    if profile == CAPCON_PROFILE:
        return {"arms": tuple(ARM_INDEX[name] for name in CAPCON_ARMS),
                "common": dict(COMMON, total_steps=40_000_000), "presets": {"fixed": PRESETS["fixed"]},
                "stages": ("C1", "L1"), "total_steps": 40_000_000,
                "checkpoint_policy": deepcopy(CAPCON_CHECKPOINT_POLICY),
                "evaluation_policy": deepcopy(CAPCON_EVALUATION_POLICY)}
    raise ValueError(f"Unsupported programme profile: {profile}")


def programme_config(arm_name, seed, lr, profile=PROGRAMME):
    definition = profile_definition(profile)
    if arm_name not in {arm["name"] for arm in definition["arms"]}:
        raise ValueError(f"Arm {arm_name} is not in programme profile {profile}")
    if profile == CAPCON_PROFILE and lr != 1e-3:
        raise ValueError("The CapCon profile requires certified Adam 0.001")
    arm = ARM_INDEX[arm_name]
    return Config(**definition["common"], **PRESETS[arm["preset"]], **arm["factors"], lr=lr, seed=seed)


def normalized(config):
    """JSON form of a configuration (tuples become lists), as manifests store it."""
    return json.loads(json.dumps(config.to_dict(), sort_keys=True))


def arm_table(profile=PROGRAMME):
    definition = profile_definition(profile)
    rows = []
    for arm in definition["arms"]:
        control = ARM_INDEX[arm["control"]] if arm["control"] else None
        changed = sorted(key for key in FACTOR_DEFAULTS
                         if control is None or arm["factors"][key] != control["factors"][key])
        rows.append({**arm, "changed_from_control": changed,
                     "preset_differs_from_control": bool(control) and control["preset"] != arm["preset"]})
    return {"programme": profile, "arms": rows, "seeds": list(SEEDS), "runs": len(rows) * len(SEEDS),
            "total_steps_per_run": definition["total_steps"],
            "stages": {stage: sum(a["stage"] == stage for a in definition["arms"]) for stage in definition["stages"]}}


def entries(lr, profile=PROGRAMME):
    rows = []
    for arm in profile_definition(profile)["arms"]:
        for seed in SEEDS:
            config = programme_config(arm["name"], seed, lr, profile=profile)
            rows.append({"index": len(rows), "arm": arm["name"], "stage": arm["stage"], "seed": seed,
                         "control": arm["control"], "relative_output": f"{arm['stage']}/{arm['name']}/seed{seed}",
                         "config": normalized(config)})
    return rows


def validate_manifest(manifest):
    """Validate the entire locked mapping before any entry or evaluation executes."""
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("programme") not in PROGRAMMES:
        raise ValueError("Unsupported programme manifest")
    profile = manifest["programme"]
    definition = profile_definition(profile)
    if manifest.get("lr") not in S1_RATES:
        raise ValueError("Manifest learning rate differs from the certified S1 candidates")
    expected = {
        "common": definition["common"], "presets": json.loads(json.dumps(definition["presets"])),
        "arms": arm_table(profile)["arms"], "entries": entries(manifest["lr"], profile=profile),
        "runtime_override_keys": list(RUNTIME_KEYS),
    }
    if profile == CAPCON_PROFILE:
        expected.update(checkpoint_policy=definition["checkpoint_policy"], evaluation_policy=definition["evaluation_policy"])
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest {key} differs from the locked programme profile")
    selection = manifest.get("lr_selection", {})
    if selection.get("selected_lr") != manifest["lr"] or selection.get("rule") != S1_RULE:
        raise ValueError("Manifest learning-rate selection differs from the certified S1 rule")
    if type(manifest.get("bounded_check")) is not bool:
        raise ValueError("Manifest bounded_check must be explicit")
    if selection.get("bounded_check") and not manifest["bounded_check"]:
        raise ValueError("A bounded selection cannot enter a research manifest")
    if profile == CAPCON_PROFILE:
        if type(selection.get("bounded_check")) is not bool:
            raise ValueError("CapCon selection bounded_check must be an explicit boolean")
        # A matching arm table and source hash do not certify the learning rate.
        # Recheck the six pinned admitted receipts, without re-auditing raw runs.
        validate_selection(selection)
    return manifest


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


def window_statistics(run, windows, horizon, receipt=None):
    """Independently audit accepted records before computing window statistics."""
    from softrole.ledger import audit_run, stats_from_audit
    if receipt is None:
        record = Path(run).parent / (Path(run).name + ".programme.json")
        if record.exists():
            _, receipt = accepted_checkpoints(run, return_audit=True)
            if receipt is None:
                raise ValueError("No committed programme checkpoint")
        else:
            receipt = audit_run(run)
    if receipt["config"]["max_steps"] != horizon:
        raise ValueError("Horizon differs from the audited run configuration")
    return stats_from_audit(receipt, windows)


def accepted_checkpoints(run, return_audit=False):
    """All committed checkpoints on the authoritative attempt lineage, in progress order."""
    from softrole.ledger import logical_segments, committed_checkpoints, audit_run
    records = []
    for segment in logical_segments(run):
        for row in committed_checkpoints(segment["run"], verify_ledgers=False):
            if segment["through_update"] is None or row["updates"] <= segment["through_update"]:
                records.append(row)
    records.sort(key=lambda row: row["updates"])
    if not records:
        return ([], None) if return_audit else []
    if len({row["updates"] for row in records}) != len(records):
        raise ValueError("Duplicate saved updates in accepted lineage")
    receipt = audit_run(run, records[-1]["updates"])
    by_update = {row["update"]: row for row in receipt["updates"]}
    config = Config(**read_json(Path(run) / "config.json"))
    source = read_json(Path(run) / "source_manifest.json")["sha256"]
    import torch
    for row in records:
        verified = by_update[row["updates"]]
        _verify_committed_ledger(row, receipt)
        if any(row[key] != verified[key] for key in ("total_steps", "total_episodes")):
            raise ValueError("Checkpoint progress differs from independently audited episodes")
        payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
        if Config(**payload["config"]).to_dict() != config.to_dict() or row["source_sha256"] != source:
            raise ValueError("Checkpoint configuration/source differs from logical run")
    return (records, receipt) if return_audit else records


def _verify_committed_ledger(checkpoint, receipt):
    """Compare independently streamed hashes with the immutable publication marker."""
    marker = Path(checkpoint["checkpoint"] + ".complete.json")
    if not marker.exists():
        return  # legacy S1 checkpoints predate publication markers
    raw_files = {item["path"]: item for item in receipt["inputs"]}
    prefixes = {item["path"]: item["updates"] for item in receipt["prefixes"]}
    for item in read_json(marker)["files"]:
        if item["path"].startswith("checkpoints/"):
            continue
        file = str((marker.parent.parent / item["path"]).resolve())
        admitted = prefixes[file].get(str(checkpoint["updates"])) if file in prefixes else raw_files.get(file)
        if admitted is None or any(admitted[k] != item[k] for k in ("bytes", "sha256")):
            raise ValueError("Raw ledger differs from committed checkpoint evidence")


def audit_s1_run(run, output, target_steps=S1_WINDOW[1], bounded=False):
    """Freeze the earliest complete checkpoint prefix into fresh compressed evidence."""
    from softrole.ledger import (committed_checkpoints, audit_run, iter_episodes, ChunkedEpisodes,
                                 canonical, atomic_json)
    run, output = Path(run).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    pinned_metadata = {name: sha(run / name) for name in ("config.json", "run.json", "source_manifest.json")}
    config = Config(**read_json(run / "config.json"))
    if config.lr not in S1_RATES or config.seed not in SEEDS:
        raise ValueError(f"{run}: not an S1 rate/seed")
    expected = programme_config("c1_real64", config.seed, config.lr).to_dict()
    actual = config.to_dict()
    if _strip_runtime(actual) != _strip_runtime(expected) or (not bounded and actual != expected):
        raise ValueError(f"{run}: configuration differs from the S1 Real64 protocol")
    candidates = sorted(committed_checkpoints(run, verify_ledgers=False, stop_at_steps=target_steps), key=lambda x: x["updates"])
    candidates = [row for row in candidates if row["total_steps"] >= target_steps]
    if not candidates:
        raise ValueError(f"{run}: window not fully covered by a complete checkpoint; partial comparisons are refused")
    chosen = candidates[0]
    receipt = audit_run(run, chosen["updates"])
    _verify_committed_ledger(chosen, receipt)
    if any(chosen[key] != receipt[key] for key in ("total_steps", "total_episodes")):
        raise ValueError("S1 checkpoint and audited prefix counts disagree")
    import torch
    payload = torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)
    source = read_json(run / "source_manifest.json")
    source_id = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    if source_id != source["sha256"] or source_id != chosen["source_sha256"]:
        raise ValueError("S1 checkpoint/source manifest identity mismatch")
    for name, expected_hash in source["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(run / "source" / name) != expected_hash:
            raise ValueError("S1 archived source hash mismatch")
    if Config(**payload["config"]).to_dict() != actual or read_json(run / "run.json")["source_sha256"] != source_id:
        raise ValueError("S1 checkpoint/configuration/source mismatch")
    output.mkdir(parents=True)
    for name in ("config.json", "run.json", "source_manifest.json"):
        shutil.copyfile(run / name, output / name)
    shutil.copytree(run / "source", output / "source")
    (output / "checkpoints").mkdir()
    checkpoint = Path(chosen["checkpoint"])
    shutil.copyfile(checkpoint, output / "checkpoints" / checkpoint.name)
    shutil.copyfile(str(checkpoint) + ".signature.json", str(output / "checkpoints" / checkpoint.name) + ".signature.json")
    # Re-emit the verified prefix byte-for-byte for plain JSONL; gzip preserves
    # all decoded records and gives the immutable archive its own chunk index.
    update_count = chosen["updates"]
    with (run / "updates.jsonl").open("rb") as src, (output / "updates.jsonl").open("xb") as dst:
        for _ in range(update_count):
            dst.write(src.readline())
    writer = ChunkedEpisodes(output, config.max_steps)
    iterator = iter(iter_episodes(run, chosen["updates"]))
    try:
        for row in receipt["updates"]:
            writer.append([next(iterator) for _ in range(row["episodes"])], row["update"])
        writer.seal()
    finally:
        writer.close()
    for name, digest in pinned_metadata.items():
        if sha(run / name) != digest or sha(output / name) != digest:
            raise ValueError("S1 identity metadata changed while importing")
    for name, digest in source["files"].items():
        if sha(output / "source" / name) != digest:
            raise ValueError("S1 source archive changed while importing")
    archived = audit_run(output, chosen["updates"])
    if archived["updates"] != receipt["updates"]:
        raise ValueError("S1 prefix changed while importing")
    from softrole.ledger import sha as prefix_sha
    for item in receipt["inputs"]:
        if prefix_sha(item["path"], item["bytes"]) != item["sha256"]:
            raise ValueError("S1 committed prefix changed while importing")
    if sha(checkpoint) != chosen["sha256"] or sha(output / "checkpoints" / checkpoint.name) != chosen["sha256"]:
        raise ValueError("S1 checkpoint changed while importing")
    archived.update(anchor={**chosen, "original_run": str(run), "source_sha256": source_id},
                    original_prefix_inputs=receipt["inputs"], bounded_check=bool(bounded))
    archived["audit_sha256"] = canonical({k: v for k, v in archived.items() if k != "audit_sha256"})
    atomic_json(output / "audit.json", archived)
    return archived


def _strip_runtime(config_dict):
    return {k: v for k, v in config_dict.items() if k not in RUNTIME_KEYS}


def _select_exact_rate(means):
    """Compare rational scientific estimates without a floating point tolerance."""
    if set(means) != set(S1_RATES) or any(not isinstance(value, Fraction) for value in means.values()):
        raise ValueError("Two exact rational S1 means are required")
    return min(S1_RATES, key=lambda lr: (means[lr], lr))


def select_learning_rate(runs, output, window=S1_WINDOW, bounded=False, audits=None):
    """Select with exact rational equal-seed means from six independently audited prefixes."""
    from softrole.ledger import validate_receipt, stats_from_audit
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    window = tuple(int(x) for x in window)
    if not bounded and window != S1_WINDOW:
        raise ValueError("The research selection uses the fixed (8M, 12M] window")
    if bool(runs) == bool(audits):
        raise ValueError("Supply either six runs or six audit receipts")
    if runs:
        archive_root = destination.parent / (destination.stem + "_audits")
        audits = []
        for index, run in enumerate(runs):
            archive = archive_root / f"run{index:02d}"
            audit_s1_run(run, archive, window[1], bounded)
            audits.append(archive / "audit.json")
    records, by_rate, source_ids = [], {}, set()
    for audit in audits:
        receipt = validate_receipt(audit)
        if "anchor" not in receipt:
            raise ValueError("S1 evidence requires a checkpoint-anchored receipt")
        source_ids.add(receipt["anchor"]["source_sha256"])
        config = Config(**receipt["config"])
        expected = programme_config("c1_real64", config.seed, config.lr).to_dict() if config.lr in S1_RATES else None
        if (config.seed not in SEEDS or expected is None or _strip_runtime(config.to_dict()) != _strip_runtime(expected)
                or (not bounded and (config.to_dict() != expected or receipt.get("bounded_check")))):
            raise ValueError("Audit configuration differs from the S1 Real64 protocol")
        stats, last = stats_from_audit(receipt, [window])
        row = stats[f"{window[0]}-{window[1]}"]
        if not row["complete"] or not row["episodes"]:
            raise ValueError("Window not fully covered; partial comparisons are refused")
        record = {"run": receipt["run"], "lr": config.lr, "seed": config.seed, "last_recorded_steps": last,
                  "window": row, "audit": str(Path(audit).resolve()), "audit_sha256": sha(audit),
                  "receipt_sha256": receipt["audit_sha256"], "inputs": {i["path"]: i["sha256"] for i in receipt["inputs"]}}
        seeds = by_rate.setdefault(config.lr, {})
        if config.seed in seeds:
            raise ValueError("Duplicate S1 run for rate/seed")
        seeds[config.seed] = record
        records.append(record)
    if set(by_rate) != set(S1_RATES) or any(set(seeds) != set(SEEDS) for seeds in by_rate.values()):
        raise ValueError("Every S1 rate needs all three seeds; partial comparisons are refused")
    if len(source_ids) != 1:
        raise ValueError("S1 rate comparison requires one training-source identity")
    means = {lr: sum((Fraction(by_rate[lr][s]["window"]["capped_completion_sum"],
                                   by_rate[lr][s]["window"]["episodes"]) for s in SEEDS), Fraction()) / len(SEEDS)
             for lr in S1_RATES}
    selected = _select_exact_rate(means)
    record = {"rule": S1_RULE, "window": list(window), "metric": "capped_completion",
              "equal_seed_means": {f"{lr:g}": float(means[lr]) for lr in S1_RATES},
              "equal_seed_means_exact": {f"{lr:g}": [means[lr].numerator, means[lr].denominator] for lr in S1_RATES},
              "selected_lr": selected, "bounded_check": bool(bounded), "runs": records}
    write_json(destination, record)
    return record


def validate_selection(selection):
    """Validate and pin receipt identities without pretending to re-audit raw bytes."""
    from softrole.ledger import validate_receipt, stats_from_audit
    if not selection.get("bounded_check") and tuple(selection.get("window", ())) != S1_WINDOW:
        raise ValueError("Research selection must use the fixed S1 window")
    grouped, source_ids = {}, set()
    for row in selection.get("runs", []):
        if sha(row["audit"]) != row["audit_sha256"]:
            raise ValueError("Pinned S1 audit receipt changed")
        receipt = validate_receipt(row["audit"])
        if receipt["audit_sha256"] != row["receipt_sha256"]:
            raise ValueError("Pinned S1 receipt identity mismatch")
        anchor = receipt.get("anchor", {})
        if any(anchor.get(k) != receipt[k] for k in ("total_steps", "total_episodes")) or anchor.get("updates") != receipt["covered_updates"][1]:
            raise ValueError("S1 receipt lacks matching checkpoint anchor")
        if anchor.get("total_steps", 0) < selection["window"][1]:
            raise ValueError("S1 anchor does not cover the selection window")
        source_ids.add(anchor["source_sha256"])
        config = Config(**receipt["config"])
        expected = programme_config("c1_real64", config.seed, config.lr).to_dict()
        if (_strip_runtime(config.to_dict()) != _strip_runtime(expected)
                or (not selection.get("bounded_check") and (config.to_dict() != expected or receipt.get("bounded_check")))):
            raise ValueError("Pinned receipt differs from S1 protocol")
        stats, _ = stats_from_audit(receipt, [tuple(selection["window"])])
        verified = stats[f'{selection["window"][0]}-{selection["window"][1]}']
        if not verified["complete"] or not verified["episodes"] or verified != row["window"]:
            raise ValueError("Selection window differs from verified receipt")
        key = (config.lr, config.seed)
        if key in grouped or key != (row["lr"], row["seed"]):
            raise ValueError("Duplicate or mismatched S1 receipt")
        grouped[key] = Fraction(verified["capped_completion_sum"], verified["episodes"])
    if set(grouped) != {(lr, seed) for lr in S1_RATES for seed in SEEDS}:
        raise ValueError("Six verified S1 prefixes are required")
    if len(source_ids) != 1:
        raise ValueError("S1 rate comparison requires one training-source identity")
    means = {lr: sum(grouped[lr, seed] for seed in SEEDS) / len(SEEDS) for lr in S1_RATES}
    if selection["selected_lr"] != _select_exact_rate(means):
        raise ValueError("Selected rate differs from exact audited comparison")
    return selection


def prepare(lr_selection, output, diagnostics_every=50, bounded=False, profile=PROGRAMME):
    """Write a locked profile manifest from a recorded S1 selection and this checkout."""
    from softrole.train import source_snapshot
    definition = profile_definition(profile)
    selection_path = Path(lr_selection)
    selection = read_json(selection_path)
    if selection.get("rule") != S1_RULE or selection.get("selected_lr") not in S1_RATES:
        raise ValueError("Learning-rate selection record does not follow the fixed S1 rule")
    if selection.get("bounded_check") and not bounded:
        raise ValueError("A bounded (engineering) selection cannot lock the research manifest")
    validate_selection(selection)
    if diagnostics_every is not None and (type(diagnostics_every) is not int or diagnostics_every < 1):
        raise ValueError("diagnostics_every must be a positive integer or None")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    source = source_snapshot()
    lr = float(selection["selected_lr"])
    manifest = {"version": MANIFEST_VERSION, "programme": profile, "bounded_check": bool(bounded),
                "lr": lr, "lr_selection_sha256": sha(selection_path), "lr_selection": selection,
                "source_sha256": source["sha256"], "git_commit": source["git_commit"],
                "diagnostics_every": diagnostics_every, "runtime_override_keys": list(RUNTIME_KEYS),
                "common": definition["common"], "presets": json.loads(json.dumps(definition["presets"])),
                "arms": arm_table(profile)["arms"], "entries": entries(lr, profile=profile)}
    if profile == CAPCON_PROFILE:
        manifest.update(checkpoint_policy=definition["checkpoint_policy"], evaluation_policy=definition["evaluation_policy"])
    validate_manifest(manifest)
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


def launch(manifest_path, index, run_root, dry_run=False, overrides=None, episode_log="chunked-gzip",
           diagnostics_every=None, resume=False):
    """Train manifest entry ``index`` under ``run_root``; record provenance beside the run."""
    if episode_log not in ("file", "chunked-gzip"):
        raise ValueError("Programme episode_log must be file or chunked-gzip; stdout cannot provide evidence")
    manifest_path = Path(manifest_path)
    manifest = validate_manifest(read_json(manifest_path))
    if type(index) is not int or not 0 <= index < len(manifest["entries"]):
        raise ValueError("Entry index outside the manifest")
    entry = manifest["entries"][index]
    if entry["index"] != index:
        raise ValueError("Manifest entry order is corrupted")
    expected = normalized(programme_config(entry["arm"], entry["seed"], manifest["lr"], profile=manifest["programme"]))
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
    if not dry_run and not resume and (output.exists() or record_path.exists()):
        raise FileExistsError(f"refusing to overwrite run: {output}")
    from softrole.train import source_snapshot, train
    source = source_snapshot()
    if source["sha256"] != manifest["source_sha256"]:
        raise ValueError("Checkout differs from the locked manifest source; prepare the manifest on the "
                         "final fixed checkout or restore that checkout before launching")
    if dry_run:
        if resume:
            if not record_path.exists():
                raise ValueError("No programme run exists to resume")
            original = read_json(record_path)
            if original["manifest_sha256"] != sha(manifest_path) or original["index"] != index:
                raise ValueError("Resume manifest identity mismatch")
            if overrides and overrides != original["runtime_overrides"]:
                raise ValueError("Programme recovery keeps the original runtime configuration")
            config = Config(**dict(entry["config"], **original["runtime_overrides"]))
        return config
    if resume:
        if not record_path.exists():
            raise ValueError("No programme run exists to resume")
        original = read_json(record_path)
        if (original["manifest_sha256"] != sha(manifest_path) or original["index"] != index
                or original["source_sha256"] != source["sha256"]):
            raise ValueError("Resume manifest/source identity mismatch")
        if overrides and overrides != original["runtime_overrides"]:
            raise ValueError("Programme recovery keeps the original runtime configuration")
        config = Config(**dict(entry["config"], **original["runtime_overrides"]))
        episode_log = original["episode_log"]
        diagnostics_every = original["diagnostics_every"]
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / (output.name + ".programme.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Programme entry already has an active attempt") from error
        if resume:
            # A timeout can precede attempt-zero initialization. Restore only
            # absent identity files from the locked manifest, never overwrite tails.
            output.mkdir(exist_ok=True)
            initialization = {"config.json": config.to_dict(), "source_manifest.json": source,
                              "run.json": {"checkpoint_commits": True, "source_sha256": source["sha256"],
                                           "environment_version": config.env_version}}
            for name, value in initialization.items():
                path = output / name
                if path.exists():
                    try:
                        read_json(path)
                    except (ValueError, UnicodeDecodeError):
                        if list((output / "checkpoints").glob("*.complete.json")):
                            raise ValueError("Committed run has damaged initialization metadata")
                        suffix = 0
                        abandoned = output / f"{name}.abandoned{suffix}"
                        while abandoned.exists():
                            suffix += 1
                            abandoned = output / f"{name}.abandoned{suffix}"
                        path.rename(abandoned)
                if not path.exists():
                    write_json(path, value)
        if not resume:
            write_json(record_path, {"programme": manifest["programme"], "manifest": str(manifest_path.resolve()),
                                     "manifest_sha256": sha(manifest_path), "index": index, "arm": entry["arm"],
                                     "stage": entry["stage"], "seed": entry["seed"], "control": entry["control"],
                                     "lr": manifest["lr"], "source_sha256": source["sha256"],
                                     "runtime_overrides": overrides, "episode_log": episode_log,
                                     "diagnostics_every": diagnostics_every,
                                     "bounded_check": bool(manifest.get("bounded_check")) or bool(overrides),
                                     "output": str(output.resolve())})
            train(config, output, episode_log=episode_log, diagnostics_every=diagnostics_every,
                  checkpoint_commits=True)
        else:
            checkpoints = accepted_checkpoints(output)
            parent = checkpoints[-1] if checkpoints else None
            if parent is not None and (parent["total_steps"] >= config.total_steps
                                       or parent["updates"] >= config.epochs * config.updates_per_epoch):
                return config
            attempts = output / "attempts"
            attempts.mkdir(exist_ok=True)
            numbers = [int(p.name) for p in attempts.iterdir() if p.name.isdigit()]
            number = max(numbers, default=0) + 1
            destination = attempts / f"{number:04d}"
            metadata = {"version": 1, "attempt": number, "logical_run": str(output.resolve()),
                        "manifest_sha256": sha(manifest_path), "source_sha256": source["sha256"],
                        "config_sha256": hashlib.sha256(json.dumps(normalized(config), sort_keys=True).encode()).hexdigest(),
                        "parent_checkpoint": parent["checkpoint"] if parent else None,
                        "parent_checkpoint_sha256": parent["sha256"] if parent else None,
                        "accepted_updates": parent["updates"] if parent else 0,
                        "accepted_total_steps": parent["total_steps"] if parent else 0,
                        "accepted_total_episodes": parent["total_episodes"] if parent else 0}
            train(config, destination, resume=parent["checkpoint"] if parent else None,
                  episode_log=episode_log, diagnostics_every=diagnostics_every,
                  checkpoint_commits=True, attempt=metadata)
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
    arms = commands.add_parser("arms", help="print a locked profile's arm table")
    arms.add_argument("--profile", choices=PROGRAMMES, default=PROGRAMME)
    select = commands.add_parser("select-lr", help="apply the fixed S1 learning-rate rule")
    inputs = select.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--runs", nargs="+", help="six S1 run directories; import committed prefixes")
    inputs.add_argument("--audits", nargs="+", help="six pinned checkpoint-anchored audit receipts")
    select.add_argument("--output", required=True)
    select.add_argument("--window", nargs=2, type=int, default=list(S1_WINDOW),
                        help="engineering checks only; the research window is fixed")
    select.add_argument("--bounded", action="store_true", help="engineering runs with reduced budgets")
    prep = commands.add_parser("prepare", help="write the locked profile manifest")
    prep.add_argument("--profile", choices=PROGRAMMES, default=PROGRAMME)
    prep.add_argument("--lr-selection", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--diagnostics-every", type=positive, default=50)
    prep.add_argument("--bounded", action="store_true", help="lock a manifest from a bounded selection")
    audit = commands.add_parser("audit", help="import and audit an immutable S1 checkpoint prefix")
    audit.add_argument("--run", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--target-steps", type=positive, default=S1_WINDOW[1])
    audit.add_argument("--bounded", action="store_true")
    reaudit = commands.add_parser("reaudit", help="explicitly reread raw bytes pinned by an audit receipt")
    reaudit.add_argument("--audit", required=True)
    run = commands.add_parser("launch", help="train one manifest entry")
    recover = commands.add_parser("resume", help="continue one logical run in a fresh immutable attempt", parents=[], add_help=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--episode-log", choices=("file", "chunked-gzip"), default="chunked-gzip")
    run.add_argument("--diagnostics-every", type=positive)
    for key in RUNTIME_KEYS:
        run.add_argument("--" + key.replace("_", "-"), dest=key, type=positive)
    for action in run._actions:
        if action.dest != "help":
            recover._add_action(action)
    args = parser.parse_args(argv)
    try:
        if args.command == "arms":
            print(json.dumps(arm_table(args.profile), indent=2))
        elif args.command == "select-lr":
            record = select_learning_rate(args.runs, args.output, tuple(args.window), args.bounded, args.audits)
            print(json.dumps({"selected_lr": record["selected_lr"], "equal_seed_means": record["equal_seed_means"],
                              "bounded_check": record["bounded_check"]}, indent=2))
        elif args.command == "audit":
            if not args.bounded and args.target_steps != S1_WINDOW[1]:
                raise ValueError("Research S1 audits use the fixed 12M anchor")
            receipt = audit_s1_run(args.run, args.output, args.target_steps, args.bounded)
            print(json.dumps({"audit": str(Path(args.output) / "audit.json"), "audit_sha256": receipt["audit_sha256"]}))
        elif args.command == "reaudit":
            from softrole.ledger import validate_receipt
            receipt = validate_receipt(args.audit, reaudit=True)
            print(json.dumps({"verified": receipt["audit_sha256"]}))
        elif args.command == "prepare":
            manifest = prepare(args.lr_selection, args.output, args.diagnostics_every, args.bounded, profile=args.profile)
            print(json.dumps({"entries": len(manifest["entries"]), "lr": manifest["lr"],
                              "source_sha256": manifest["source_sha256"], "manifest": args.output}))
        else:
            overrides = {key: getattr(args, key) for key in RUNTIME_KEYS}
            config = launch(args.manifest, args.index, args.run_root, args.dry_run, overrides,
                            args.episode_log, args.diagnostics_every, resume=args.command == "resume")
            if args.dry_run:
                print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
