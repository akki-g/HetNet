"""Locked, exploratory G-cap/G-type pilot; no evaluation or job submission."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from softrole.programme import CAPCON_PROFILE, normalized, programme_config
from softrole.train import source_snapshot, train, write_json


VERSION = 1
PILOT = "graded_sensing_pilot_2m_v1"
ARMS = (("g_cap", "capability"), ("g_type", "type"))
PROTOCOL = {
    "purpose": "Exploratory execution, throughput and early learning; no superiority or failure verdict",
    "primary_arm": "l1b_decomposed_beta1",
    "primary_profile": CAPCON_PROFILE,
    "training_seed": 0,
    "target_joint_environment_steps_per_arm": 2_000_000,
    "stopping": "first complete collector batch reaching the target; record actual counts",
    "evaluation": "none; no held-out outcomes or automatic selection",
    "episode_log": "chunked-gzip",
    "checkpoint_commits": True,
    "device": "cpu",
    "dtype": "float64",
    "threads_per_collector": 1,
}


def require_index(index):
    if type(index) is not int or index not in (0, 1):
        raise ValueError("pilot index must be 0 (G-cap) or 1 (G-type)")


def pilot_config(index):
    require_index(index)
    primary = programme_config("l1b_decomposed_beta1", 0, 1e-3, profile=CAPCON_PROFILE)
    config = replace(primary, capabilities="graded", descriptor=ARMS[index][1],
                     sensing_p=(1, 3), sensing_a=(0,), held_out_sensing_p=(2,),
                     held_out_sensing_a=(1, 2, 3), total_steps=2_000_000)
    changed = {key for key in primary.__dataclass_fields__
               if getattr(primary, key) != getattr(config, key)}
    allowed = {"capabilities", "descriptor", "sensing_p", "sensing_a",
               "held_out_sensing_p", "held_out_sensing_a", "total_steps"}
    if changed - allowed:
        raise ValueError("Pilot settings differ from the primary arm beyond declared fields")
    return config


def expected_entries():
    return [dict(index=index, arm=arm, seed=0, relative_output=f"{arm}/seed0",
                 config=normalized(pilot_config(index))) for index, (arm, _) in enumerate(ARMS)]


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
            "version", "pilot", "protocol", "entries", "source", "manifest_sha256"}:
        raise ValueError("Unsupported graded pilot manifest")
    if (type(manifest["version"]) is not int or manifest["version"] != VERSION
            or manifest["pilot"] != PILOT or digest(manifest["protocol"]) != digest(PROTOCOL)
            or digest(manifest["entries"]) != digest(expected_entries())):
        raise ValueError("Manifest differs from the locked graded pilot")
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
        raise FileExistsError(f"Pilot manifest already exists: {output}")
    manifest = dict(version=VERSION, pilot=PILOT, protocol=PROTOCOL,
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
        raise ValueError("Source changed since pilot preparation; use the frozen checkout")
    entry = manifest["entries"][index]
    output = Path(run_root).resolve() / entry["relative_output"]
    if output.exists():
        raise FileExistsError(f"Pilot output already exists; fresh runs only: {output}")
    record = dict(pilot=PILOT, index=index, arm=entry["arm"], seed=entry["seed"],
                  config=entry["config"], output=str(output), dry_run=bool(dry_run),
                  source_sha256=source["sha256"], manifest_sha256=manifest["manifest_sha256"])
    if dry_run:
        return record
    checkpoint = train(pilot_config(index), output, episode_log="chunked-gzip", checkpoint_commits=True)
    if source_snapshot()["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed during the pilot; inspect the run before using its results")
    record["checkpoint"] = str(checkpoint)
    write_json(output / "pilot.json", record)
    return record


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = cli.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", allow_abbrev=False)
    preparation.add_argument("--output", type=Path, required=True)
    execution = commands.add_parser("launch", allow_abbrev=False)
    execution.add_argument("--manifest", type=Path, required=True)
    execution.add_argument("--index", type=int, choices=(0, 1), required=True)
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
