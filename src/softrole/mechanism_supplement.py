"""Locked training launcher for the capcon_hyper supplement to the capability-mechanism study; no evaluation or job submission."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from softrole import mechanism_study
from softrole.programme import normalized
from softrole.train import source_snapshot, train, write_json


VERSION = 1
STUDY = "capability_mechanism_supplement_v1"
CONDITIONS = mechanism_study.CONDITIONS
# (arm name, Config.model, Config.descriptor)
ARMS = (("capcon_hyper", "capcon_hyper", "capability"),)
SEEDS = mechanism_study.SEEDS
TARGET_STEPS = mechanism_study.TARGET_STEPS
COUNT = len(CONDITIONS) * len(ARMS) * len(SEEDS)
# The supplement's configurations are the parent study's arm of this name with only `model` replaced.
PARENT_ARM = "capcon"
PROTOCOL = {
    **deepcopy(mechanism_study.PROTOCOL),
    "purpose": "Supplementary arm of the capability-mechanism comparison (hypernetwork head on the shared "
               "body); evaluation is a separate declared stage",
    "arms": [{"name": name, "model": model, "descriptor": descriptor} for name, model, descriptor in ARMS],
    "parent_study": mechanism_study.STUDY,
    "declared": "2026-11-10, after the parent study's training array was launched and before any frozen "
                "evaluation was run",
}
digest = mechanism_study.digest


def require_index(index):
    if type(index) is not int or index not in range(COUNT):
        raise ValueError(f"study index must be an integer from 0 through {COUNT - 1}")


def locate(index):
    """Return (condition, dim, arm, model, descriptor, seed) for a supplement index."""
    require_index(index)
    condition, arm, seed = index // (len(ARMS) * len(SEEDS)), index // len(SEEDS) % len(ARMS), index % len(SEEDS)
    return CONDITIONS[condition] + ARMS[arm] + (SEEDS[seed],)


def parent_index(index):
    """Index in the parent study of its `capcon` entry for this index's condition and seed."""
    condition, _, _, _, _, seed = locate(index)
    for number in range(mechanism_study.COUNT):
        parent = mechanism_study.locate(number)
        if parent[0] == condition and parent[2] == PARENT_ARM and parent[5] == seed:
            return number
    raise ValueError(f"The parent study has no {PARENT_ARM} entry for {condition} seed {seed}")


def study_config(index):
    _, _, _, model, descriptor, _ = locate(index)
    parent = mechanism_study.study_config(parent_index(index))
    config = replace(parent, model=model)
    changed = {key for key in parent.__dataclass_fields__ if getattr(parent, key) != getattr(config, key)}
    if changed != {"model"} or config.descriptor != descriptor:
        raise ValueError("Supplement settings differ from the parent capcon arm beyond the model")
    return config


def expected_entries():
    rows = []
    for index in range(COUNT):
        condition, _, arm, _, _, seed = locate(index)
        rows.append(dict(index=index, condition=condition, arm=arm, seed=seed,
                         relative_output=f"{condition}/{arm}/seed{seed}",
                         config=normalized(study_config(index))))
    return rows


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
            "version", "study", "protocol", "entries", "source", "manifest_sha256"}:
        raise ValueError("Unsupported mechanism supplement manifest")
    if (type(manifest["version"]) is not int or manifest["version"] != VERSION
            or manifest["study"] != STUDY or digest(manifest["protocol"]) != digest(PROTOCOL)
            or digest(manifest["entries"]) != digest(expected_entries())):
        raise ValueError("Manifest differs from the locked mechanism supplement")
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
        raise FileExistsError(f"Supplement manifest already exists: {output}")
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
        raise ValueError("Source changed since supplement preparation; use the frozen checkout")
    entry = manifest["entries"][index]
    root = Path(run_root).resolve()
    output = root / entry["relative_output"]
    if not output.resolve().is_relative_to(root):
        raise ValueError("Supplement output must stay inside the run root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Supplement output already exists; fresh runs only: {output}")
    record = dict(study=STUDY, index=index, condition=entry["condition"], arm=entry["arm"],
                  seed=entry["seed"], config=entry["config"], output=str(output),
                  dry_run=bool(dry_run), source_sha256=source["sha256"],
                  manifest_sha256=manifest["manifest_sha256"])
    if dry_run:
        return record
    checkpoint = train(study_config(index), output, episode_log="chunked-gzip", checkpoint_commits=True)
    if source_snapshot()["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed during the supplement run; inspect the run before using its results")
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
