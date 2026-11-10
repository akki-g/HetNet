"""Locked training launcher for the capability-mechanism study; no evaluation or job submission."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from softrole.config import COMPOSITION_TEST, COMPOSITION_TRAIN, EXTRAPOLATION_TEST
from softrole.programme import CAPCON_PROFILE, normalized, programme_config
from softrole.train import source_snapshot, train, write_json


VERSION = 2
STUDY = "capability_mechanism_study_v2"
CONDITIONS = (("s5", 5), ("s6", 6))
# (arm name, Config.model, Config.descriptor)
ARMS = (("capcon", "shared", "capability"), ("capcon_class_input", "shared", "type"),
        ("class_typed", "class_typed", "type"), ("ca_mlp", "ca_mlp", "capability"),
        ("ca_cc_gnn", "ca_cc_gnn", "capability"), ("rnn_exp", "rnn_exp", "capability"),
        ("cash", "cash", "capability"))
SEEDS = (0, 1, 2, 3, 4)
TARGET_STEPS = 10_000_000
COUNT = len(CONDITIONS) * len(ARMS) * len(SEEDS)
PROTOCOL = {
    "purpose": "Confirmatory training for the capability-mechanism comparison; "
               "evaluation is a separate declared stage",
    "primary_arm": "l1b_decomposed_beta1",
    "primary_profile": CAPCON_PROFILE,
    "conditions": [{"name": name, "dim": dim} for name, dim in CONDITIONS],
    "arms": [{"name": name, "model": model, "descriptor": descriptor}
             for name, model, descriptor in ARMS],
    "seeds": list(SEEDS),
    "target_joint_environment_steps_per_run": TARGET_STEPS,
    "stopping": "first complete collector batch reaching the target; record actual counts",
    "checkpoint_rule": "first committed checkpoint at or above the target; never best epoch",
    "episode_log": "chunked-gzip",
    "checkpoint_commits": True,
    "device": "cpu",
    "dtype": "float64",
    "threads_per_collector": 1,
}


def require_index(index):
    if type(index) is not int or index not in range(COUNT):
        raise ValueError(f"study index must be an integer from 0 through {COUNT - 1}")


def locate(index):
    """Return (condition, dim, arm, model, descriptor, seed) for a study index."""
    require_index(index)
    condition, arm, seed = index // (len(ARMS) * len(SEEDS)), index // len(SEEDS) % len(ARMS), index % len(SEEDS)
    return CONDITIONS[condition] + ARMS[arm] + (SEEDS[seed],)


def study_config(index):
    _, dim, _, model, descriptor, seed = locate(index)
    primary = programme_config("l1b_decomposed_beta1", seed, 1e-3, profile=CAPCON_PROFILE)
    config = replace(primary, model=model, descriptor=descriptor, dim=dim, vision=2, capabilities="graded",
                     env_version="graded-observation-v2", sensing_p=(1, 3), held_out_sensing_p=(2,),
                     compositions=COMPOSITION_TRAIN, held_out=COMPOSITION_TEST + EXTRAPOLATION_TEST,
                     entropy_coeff=0.01, null_degree_correction=True, total_steps=TARGET_STEPS)
    changed = {key for key in primary.__dataclass_fields__
               if getattr(primary, key) != getattr(config, key)}
    allowed = {"model", "descriptor", "dim", "vision", "capabilities", "env_version", "sensing_p",
               "held_out_sensing_p", "compositions", "held_out", "entropy_coeff",
               "null_degree_correction", "total_steps"}
    if changed - allowed:
        raise ValueError("Study settings differ from the primary arm beyond declared fields")
    return config


def expected_entries():
    rows = []
    for index in range(COUNT):
        condition, _, arm, _, _, seed = locate(index)
        rows.append(dict(index=index, condition=condition, arm=arm, seed=seed,
                         relative_output=f"{condition}/{arm}/seed{seed}",
                         config=normalized(study_config(index))))
    return rows


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
            "version", "study", "protocol", "entries", "source", "manifest_sha256"}:
        raise ValueError("Unsupported mechanism study manifest")
    if (type(manifest["version"]) is not int or manifest["version"] != VERSION
            or manifest["study"] != STUDY or digest(manifest["protocol"]) != digest(PROTOCOL)
            or digest(manifest["entries"]) != digest(expected_entries())):
        raise ValueError("Manifest differs from the locked mechanism study")
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


def prepare(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Study manifest already exists: {output}")
    manifest = dict(version=VERSION, study=STUDY, protocol=PROTOCOL,
                    entries=expected_entries(), source=source_snapshot())
    manifest["manifest_sha256"] = digest(manifest)
    validate_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest)
    return manifest


def launch(manifest_path, index, run_root, dry_run=False):
    require_index(index)
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    source = source_snapshot()
    if source["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Source changed since study preparation; use the frozen checkout")
    entry = manifest["entries"][index]
    root = Path(run_root).resolve()
    output = root / entry["relative_output"]
    if not output.resolve().is_relative_to(root):
        raise ValueError("Study output must stay inside the run root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Study output already exists; fresh runs only: {output}")
    record = dict(study=STUDY, index=index, condition=entry["condition"], arm=entry["arm"],
                  seed=entry["seed"], config=entry["config"], output=str(output),
                  dry_run=bool(dry_run), source_sha256=source["sha256"],
                  manifest_sha256=manifest["manifest_sha256"])
    if dry_run:
        return record
    checkpoint = train(study_config(index), output, episode_log="chunked-gzip", checkpoint_commits=True)
    if source_snapshot()["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed during the study run; inspect the run before using its results")
    record["checkpoint"] = str(checkpoint)
    write_json(output / "study.json", record)
    return record


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = cli.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", allow_abbrev=False)
    preparation.add_argument("--output", type=Path, required=True)
    execution = commands.add_parser("launch", allow_abbrev=False)
    execution.add_argument("--manifest", type=Path, required=True)
    execution.add_argument("--index", type=int, choices=range(COUNT), metavar="INDEX", required=True)
    execution.add_argument("--run-root", type=Path, required=True)
    execution.add_argument("--dry-run", action="store_true")
    args = cli.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.output)
        else:
            result = launch(args.manifest, args.index, args.run_root, args.dry_run)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
