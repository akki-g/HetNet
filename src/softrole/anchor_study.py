"""Locked training launcher for the 6x6 external anchor: the original HetNet and the shared-body arms on plain PCP; no evaluation or job submission."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

from softrole import mechanism_study
from softrole.pcp_additional_seeds import run_entry
from softrole.programme import normalized
from softrole.train import source_snapshot, train, write_json


VERSION = 1
STUDY = "pcp_6x6_anchor_v1"
MAP_DIM = 6
TEAM = (2, 1)
SENSING_P = (3,)  # full sensing: vision 2, so the physical sensing level is vision + 1
# (arm name, kind, Config.model or HetNet model_spec, Config.descriptor)
ARMS = (("hetnet_original", "hetnet", "paper-v1", None),
        ("capcon", "softrole", "shared", "capability"),
        ("capcon_class_input", "softrole", "shared", "type"),
        ("class_typed", "softrole", "class_typed", "type"))
SEEDS = (0, 1, 2)
TARGET_STEPS = 10_000_000
COUNT = len(ARMS) * len(SEEDS)
PARENT_CONDITION = "s6"
# The shared-body configurations are the parent study's s6 entries of the same arm and seed with only these replaced.
CHANGED_FIELDS = {"compositions", "sensing_p", "held_out_sensing_p"}
# The manifest is written before any run root exists. Its HetNet entries therefore name this stand-in run root and
# interpreter; launch substitutes the real run root, and the real interpreter appears in the launch record.
PLACEHOLDER_ROOT = "/anchor_run_root"
PLACEHOLDER_PYTHON = "<python>"
HETNET_WORKLOAD = (3, "paper-v1", "torch-v1")  # publication_reconstruction.study.train_arguments(3, root, spec, backend)
PROTOCOL = {
    "purpose": "External anchor: the original HetNet and the shared-body arms on plain 6x6 "
               "Predator-Capture-Prey with one fixed team; compared on training logs",
    "study": STUDY,
    "arms": [{"name": name, "kind": kind, "model": model, "descriptor": descriptor}
             for name, kind, model, descriptor in ARMS],
    "seeds": list(SEEDS),
    "target_joint_environment_steps_per_run": TARGET_STEPS,
    "map": {"dim": MAP_DIM},
    "team": {"perception": TEAM[0], "capture": TEAM[1]},
    "sensing": {"level": "full", "sensing_p": list(SENSING_P), "vision": 2},
    "task": "plain Predator-Capture-Prey (pcp), fixed team, no held-out panel, no failure injection",
    "parent_study": mechanism_study.STUDY,
    "declared": "2026-11-10, after the parent study's training array was launched and before any frozen "
                "evaluation was run",
    "comparison": "training logs only; no frozen evaluation",
    "hetnet_workload": {"reconstruction_spec": "paper-v1", "message_backend": "torch-v1", "variant": "real",
                        "collectors": 4, "horizon": 80, "batch_steps": 500, "updates_per_epoch": 10,
                        "runtime": "publication_reconstruction (archived source copy, one run per seed)",
                        "changed_from_5x5_workload": ["--seed", "--output", "--dim", "--max-env-steps",
                                                      "--milestones"]},
    "softrole_training": {key: deepcopy(mechanism_study.PROTOCOL[key]) for key in (
        "primary_profile", "stopping", "checkpoint_rule", "episode_log", "checkpoint_commits", "device", "dtype",
        "threads_per_collector")},
    "differences": [
        "The hetnet_original rows use the original HetNet system's own learner, observation and critics (the "
        "paper-v1 reconstruction with the torch-v1 message backend), not the study's learner, observation and "
        "critics; the rows are therefore not a controlled comparison with the shared-body rows.",
        "The hetnet_original rows are the PCP Real paper-v1 workload of the first-wave HetNet study with only "
        "--seed, --output, --dim 6, --max-env-steps 10000000 and --milestones 10000000 changed.",
        "The shared-body rows are the parent study's s6 configuration of the same arm and seed with only "
        "compositions=((2, 1),), sensing_p=(3,) and held_out_sensing_p=() replaced.",
    ],
}
digest = mechanism_study.digest


def require_index(index):
    if type(index) is not int or index not in range(COUNT):
        raise ValueError(f"study index must be an integer from 0 through {COUNT - 1}")


def locate(index):
    """Return (arm, kind, model, descriptor, seed) for an anchor index."""
    require_index(index)
    return ARMS[index // len(SEEDS)] + (SEEDS[index % len(SEEDS)],)


def parent_index(index):
    """Index in the parent study of its s6 entry for this index's arm and seed."""
    arm, kind, _, _, seed = locate(index)
    if kind != "softrole":
        raise ValueError(f"The {arm} arm has no parent-study configuration")
    for number in range(mechanism_study.COUNT):
        parent = mechanism_study.locate(number)
        if parent[0] == PARENT_CONDITION and parent[2] == arm and parent[5] == seed:
            return number
    raise ValueError(f"The parent study has no {arm} entry for {PARENT_CONDITION} seed {seed}")


def study_config(index):
    """The Config of a shared-body index: the parent s6 configuration with the three support fields replaced."""
    _, _, model, descriptor, _ = locate(index)
    parent = mechanism_study.study_config(parent_index(index))
    config = replace(parent, compositions=(TEAM,), sensing_p=SENSING_P, held_out_sensing_p=())
    changed = {key for key in parent.__dataclass_fields__ if getattr(parent, key) != getattr(config, key)}
    if (changed != CHANGED_FIELDS or (config.model, config.descriptor) != (model, descriptor)
            or (config.dim, config.total_steps) != (MAP_DIM, TARGET_STEPS)):
        raise ValueError("Anchor settings differ from the parent s6 configuration beyond the team and sensing fields")
    config.validate()
    return config


def hetnet_arguments(seed, output):
    """Arguments of publication_reconstruction for the 6x6 workload: the 5x5 PCP Real paper-v1 list, five items changed."""
    from publication_reconstruction.study import train_arguments
    index, spec, backend = HETNET_WORKLOAD
    arguments = train_arguments(index, PLACEHOLDER_ROOT, reconstruction_spec=spec, message_backend=backend)
    arguments[arguments.index("--seed") + 1] = str(seed)
    arguments[arguments.index("--output") + 1] = str(output)
    arguments[arguments.index("--max-env-steps") + 1] = str(TARGET_STEPS)
    start = arguments.index("--milestones") + 1
    end = next((i for i in range(start, len(arguments)) if arguments[i].startswith("--")), len(arguments))
    arguments[start:end] = [str(TARGET_STEPS)]
    arguments[arguments.index("--seed") + 2:arguments.index("--seed") + 2] = ["--dim", str(MAP_DIM)]
    return arguments


def hetnet_protocol(arguments):
    """The protocol publication_reconstruction resolves for an argument list; nothing is created."""
    from publication_reconstruction.__main__ import parser, resolve
    return resolve(parser().parse_args(arguments))


def locked_protocol(resolved):
    """A resolved protocol with the interpreter replaced, so that the manifest does not depend on the environment."""
    resolved = deepcopy(resolved)
    resolved["command"][0] = PLACEHOLDER_PYTHON
    return resolved


def expected_entries():
    rows = []
    for index in range(COUNT):
        arm, kind, _, _, seed = locate(index)
        row = dict(index=index, arm=arm, kind=kind, seed=seed, relative_output=f"{arm}/seed{seed}")
        if kind == "hetnet":
            arguments = hetnet_arguments(seed, f"{PLACEHOLDER_ROOT}/{row['relative_output']}/training")
            row.update(arguments=arguments, resolved=locked_protocol(hetnet_protocol(arguments)))
        else:
            row.update(config=normalized(study_config(index)))
        rows.append(row)
    return rows


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
            "version", "study", "protocol", "entries", "source", "manifest_sha256"}:
        raise ValueError("Unsupported anchor study manifest")
    if (type(manifest["version"]) is not int or manifest["version"] != VERSION
            or manifest["study"] != STUDY or digest(manifest["protocol"]) != digest(PROTOCOL)
            or digest(manifest["entries"]) != digest(expected_entries())):
        raise ValueError("Manifest differs from the locked anchor study")
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
        raise FileExistsError(f"Anchor manifest already exists: {output}")
    manifest = dict(version=VERSION, study=STUDY, protocol=PROTOCOL,
                    entries=expected_entries(), source=source_snapshot())
    manifest["manifest_sha256"] = digest(manifest)
    validate_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest)
    return manifest


def launch(manifest_path, index, run_root, dry_run=False):
    """Run (or, with dry_run, describe) one entry; the record of a HetNet entry carries the launcher's exit code."""
    require_index(index)
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    source = source_snapshot()
    if source["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Source changed since anchor preparation; use the frozen checkout")
    entry = manifest["entries"][index]
    root = Path(run_root).resolve()
    output = root / entry["relative_output"]
    if not output.resolve().is_relative_to(root):
        raise ValueError("Anchor output must stay inside the run root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Anchor output already exists; fresh runs only: {output}")
    record = dict(study=STUDY, index=index, arm=entry["arm"], kind=entry["kind"], seed=entry["seed"],
                  output=str(output), dry_run=bool(dry_run), source_sha256=source["sha256"],
                  manifest_sha256=manifest["manifest_sha256"])
    if entry["kind"] == "hetnet":
        # The run uses the locked arguments with only the stand-in run root replaced.
        arguments = list(entry["arguments"])
        position = arguments.index("--output") + 1
        if arguments[position] != f"{PLACEHOLDER_ROOT}/{entry['relative_output']}/training":
            raise ValueError("Locked HetNet output differs from the anchor layout")
        arguments[position] = str(output / "training")
        record.update(arguments=arguments, resolved=hetnet_protocol(arguments))
        if dry_run:
            return record
        selected = dict(index=index, arm=entry["arm"], seed=entry["seed"], output=str(output / "training"),
                        fresh=True, resume=None, starting_counts=dict(env_steps=0, episodes=0, updates=0),
                        kind="hetnet", arguments=arguments, resolved=record["resolved"], study=STUDY,
                        source_sha256=source["sha256"], anchor_manifest_sha256=manifest["manifest_sha256"])
        record["exit_code"] = run_entry(selected, manifest_path, dict(executable=sys.executable, python=sys.version))
        return record
    record["config"] = entry["config"]
    if dry_run:
        return record
    checkpoint = train(study_config(index), output, episode_log="chunked-gzip", checkpoint_commits=True)
    if source_snapshot()["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed during the anchor run; inspect the run before using its results")
    record["checkpoint"] = str(checkpoint)
    write_json(output / "study.json", record)
    return record


def main(argv=None):
    """Return the process exit status: the HetNet launcher's exit code for a HetNet entry, otherwise 0."""
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
    return result.get("exit_code", 0)


if __name__ == "__main__":
    raise SystemExit(main())
