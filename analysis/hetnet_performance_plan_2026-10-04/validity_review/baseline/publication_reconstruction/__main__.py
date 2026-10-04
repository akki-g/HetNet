"""Launch a frozen source copy with fresh evidence and complete-update budgets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys

from . import SCHEMA_VERSION, SCAFFOLD_REFERENCE, UPSTREAM_REFERENCE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def seed_value(value):
    value = int(value)
    if not 0 <= value < 2**32 - 16:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**32-16)")
    return value


def seconds(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("wall seconds must be finite and nonnegative")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="separate reconstruction run; never overwrites")
    train.add_argument("--task", choices=["pp", "pcp", "fc"], required=True)
    train.add_argument("--variant", choices=["real", "binary"], default="real")
    train.add_argument("--model-spec", choices=["public-code-v1", "supplement-v1"], default="public-code-v1")
    train.add_argument("--seed", type=seed_value, required=True)
    train.add_argument("--env-version", choices=["historical-2022", "corrected-v1"], default="historical-2022")
    train.add_argument("--recipe", choices=["june-2022", "october-2022"], default="june-2022")
    train.add_argument("--output", type=Path)
    train.add_argument("--epochs", type=positive)
    train.add_argument("--collectors", type=positive)
    train.add_argument("--updates-per-epoch", type=positive, default=10)
    train.add_argument("--batch-steps", type=positive, default=500)
    train.add_argument("--horizon", type=positive)
    train.add_argument("--save-every", type=positive, default=50)
    train.add_argument("--max-env-steps", type=positive,
                       help="stop after a complete update reaches this count; overshoot is recorded")
    train.add_argument("--milestones", nargs="*", type=positive, default=[])
    train.add_argument("--episode-log", choices=["file", "stdout"], default="file")
    train.add_argument("--wall-seconds", type=seconds, default=0,
                       help="request a recoverable stop after this elapsed time; zero disables")
    train.add_argument("--dry-run", action="store_true")
    resume = commands.add_parser("resume", help="continue a checkpoint into a fresh immutable segment")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--checkpoint", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--episode-log", choices=["file", "stdout"])
    resume.add_argument("--wall-seconds", type=seconds, default=0)
    resume.add_argument("--dry-run", action="store_true")
    from .evaluation import add_arguments
    add_arguments(commands.add_parser("evaluate", help="frozen archived-runtime evaluation"))
    from .study import add_commands
    add_commands(commands)
    return p


def resolve(args):
    fc = args.task == "fc"
    epochs = args.epochs or (1400 if fc else 2000)
    horizon = args.horizon or (300 if fc else 80)
    collectors = args.collectors or (4 if fc or args.recipe == "october-2022" else 1)
    if args.seed + collectors >= 2**32:
        raise ValueError("seed plus collector count exceeds NumPy seed range")
    run = (args.output or ROOT / "runs/publication_reconstruction" / args.model_spec / args.env_version /
           args.recipe / f"{args.task}_{args.variant}" / f"seed{args.seed}").resolve()
    supplement = args.model_spec == "supplement-v1"
    if args.milestones != sorted(set(args.milestones)):
        raise ValueError("Milestones must be unique and increasing")
    if args.milestones and args.max_env_steps is not None and args.milestones[-1] > args.max_env_steps:
        raise ValueError("Milestones cannot exceed the scientific step target")
    # The June PP wrapper needs --nagents3, absent from the released example.
    environment = "fire_commander" if fc else (
        "predator_prey" if args.task == "pp" and args.recipe == "june-2022" else "predator_capture")
    command = [sys.executable, "-u", str(run / "source/runtime/main.py"),
        "--env_name", environment, "--nfriendly_P", "3" if args.task == "pp" else "2",
        "--nfriendly_A", "0" if args.task == "pp" else "1", "--nagents", "3",
        "--hetgat", "--hetgat_a2c", "--num_epochs", str(epochs),
        "--nprocesses", str(collectors), "--epoch_size", str(args.updates_per_epoch),
        "--batch_size", str(args.batch_steps), "--max_steps", str(horizon),
        "--detach_gap", "5", "--lrate", "0.001" if supplement else "0.0001", "--hid_size", "128", "--dim", "5",
        "--vision", "1" if fc else "2", "--seed", str(args.seed),
        "--save_every", str(args.save_every), "--experiment_name", f"{args.task}_{args.variant}",
        "--save_dir", str(run / "checkpoints"), "--metrics_file", str(run / "metrics.jsonl"),
        "--publication_env_version", args.env_version, "--source_manifest", str(run / "source_manifest.json"),
        "--max_env_steps", str(args.max_env_steps or 0), "--model_spec", args.model_spec,
        "--wall_seconds", str(args.wall_seconds), "--episode_log", args.episode_log]
    if args.milestones:
        command += ["--milestones", *map(str, args.milestones)]
    if fc:
        command += ["--nfires", "1", "--reward_type", "3"]
    if args.variant == "binary":
        command += ["--use_binary", "--msg_dim", "16"]
    floor = epochs * args.updates_per_epoch * collectors * args.batch_steps
    if args.max_env_steps is not None and floor < args.max_env_steps:
        raise ValueError("Epoch cap cannot guarantee the requested step target at the batch floor")
    batch_max = collectors * (args.batch_steps + horizon - 1)
    return {"schema_version": SCHEMA_VERSION, "task": args.task, "variant": args.variant,
            "env_version": args.env_version, "recipe": args.recipe, "seed": args.seed,
            "output": str(run), "command": command, "epochs": epochs, "collectors": collectors,
            "updates_per_epoch": args.updates_per_epoch, "batch_step_floor_per_collector": args.batch_steps,
            "episode_horizon": horizon, "epoch_budget_min_steps_without_step_cap": floor,
            "max_env_steps": args.max_env_steps, "max_overshoot_steps": batch_max - 1,
            "upstream_reference": UPSTREAM_REFERENCE, "runtime_scaffold_reference": SCAFFOLD_REFERENCE,
            "model_spec": args.model_spec,
            "learner_spec": "public-code-v1", "rng_scheme": "seedsequence-v1" if supplement else "legacy-offset-v1",
            "milestones": args.milestones, "episode_log": args.episode_log,
            "wall_seconds": args.wall_seconds,
            "claim": "supplement-aligned architecture/optimizer; public-code learner; not a verified publication training commit" if supplement else "runnable reconstruction, not a verified publication training commit",
            "architecture": f"{'three' if supplement else 'two'} HetGAT layers; four heads; per-class critics",
            "optimizer": ({"name": "Adam", "lr": .001, "betas": [.9, .999], "epsilon": 1e-8,
                           "weight_decay": 0, "amsgrad": False, "foreach": False, "fused": False}
                          if supplement else {"name": "RMSprop", "lr": 1e-4, "alpha": .97, "epsilon": 1e-6}),
            "budget_semantics": "first of epoch cap or complete-update step threshold; no partial episodes"}


def resolve_resume(args):
    from .artifacts import digest, load_checkpoint, verify_run
    original, _ = verify_run(args.run_dir)
    saved = load_checkpoint(args.run_dir, args.checkpoint, require_recovery=True)
    if original.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Run protocol does not support continuation")
    run = args.output.resolve()
    result = dict(original)
    command = list(original["command"])
    command[:3] = [sys.executable, "-u", str(run / "source/runtime/main.py")]
    updates = {"--save_dir": str(run / "checkpoints"), "--metrics_file": str(run / "metrics.jsonl"),
               "--source_manifest": str(run / "source_manifest.json"),
               "--wall_seconds": str(args.wall_seconds),
               "--episode_log": args.episode_log or original["episode_log"],
               "--resume_checkpoint": str(args.checkpoint.resolve())}
    for flag, value in updates.items():
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command.extend([flag, value])
    result.update(output=str(run), command=command, wall_seconds=args.wall_seconds,
                  episode_log=updates["--episode_log"], continuation={
                      "run_dir": str(args.run_dir.resolve()), "checkpoint": str(args.checkpoint.resolve()),
                      "checkpoint_sha256": digest(args.checkpoint),
                      "counts": saved["reconstruction"]["counts"],
                      "completed_epochs": saved["recovery"]["completed_epochs"]})
    return result


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inspect_source_origins():
    """Read the complete inventory without changing or accepting local provenance."""
    origins_bytes = (HERE / "ORIGINS.json").read_bytes()
    origins = json.loads(origins_bytes)["files"]
    paths = [p for p in sorted((HERE / "runtime").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    payloads = {p.relative_to(HERE / "runtime").as_posix(): p.read_bytes() for p in paths}
    expected = {name: row["current_sha256"] for name, row in origins.items()}
    actual = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    missing, unexpected = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    changed = {name: {"expected_sha256": expected[name], "actual_sha256": actual[name]}
               for name in sorted(expected.keys() & actual.keys()) if actual[name] != expected[name]}
    audit = {"schema_version": 1, "valid": not (missing or unexpected or changed),
             "runtime_root": str(HERE / "runtime"), "runtime_files": len(actual),
             "expected_runtime_files": len(expected), "origins_sha256": hashlib.sha256(origins_bytes).hexdigest(),
             "missing": missing, "unexpected": unexpected, "changed": changed,
             "expected_inventory": expected, "actual_inventory": actual}
    return payloads, origins_bytes, audit


def validate_source_origins():
    payloads, origins_bytes, audit = inspect_source_origins()
    if not audit["valid"]:
        kind = "Runtime file inventory differs" if audit["missing"] or audit["unexpected"] else "Runtime source differs"
        differences = {key: audit[key] for key in ("missing", "unexpected", "changed")}
        raise RuntimeError(f"{kind} from ORIGINS.json: {json.dumps(differences, sort_keys=True)}. "
                           "Restore the approved source and matching manifest; do not regenerate "
                           "provenance merely to accept an incomplete or unexpected transfer.")
    return payloads, origins_bytes


def archive_source(run):
    runtime, origins_bytes = validate_source_origins()
    payloads = {f"runtime/{name}": payload for name, payload in runtime.items()}
    payloads["ORIGINS.json"] = origins_bytes
    for name in ["__init__.py", "__main__.py", "README.md", "artifacts.py", "study.py", "STUDY.json"]:
        payloads[name] = (HERE / name).read_bytes()
    files = {}
    for name, payload in payloads.items():
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    for name in ["uv.lock", "requirements.txt"]:
        shutil.copy2(ROOT / name, run / name)
    manifest = {"schema_version": SCHEMA_VERSION, "upstream_reference": UPSTREAM_REFERENCE,
                "runtime_scaffold_reference": SCAFFOLD_REFERENCE, "files": files}
    write_json(run / "source_manifest.json", manifest)


def copy_resume_source(original, run):
    from .artifacts import verify_run, verify_source_archive
    _, manifest = verify_run(original)
    for name in manifest["files"]:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / "source" / name).read_bytes())
    for name in ("source_manifest.json", "uv.lock", "requirements.txt"):
        shutil.copy2(original / name, run / name)
    verify_run(original)
    copied_manifest = json.loads((run / "source_manifest.json").read_text())
    if copied_manifest != manifest:
        raise ValueError("Continuation source manifest changed during copying")
    verify_source_archive(run, copied_manifest)


def stop_process_group(child):
    """Bound shutdown of the isolated training process and its collectors."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    finally:
        # A child can exit before its descendants; remove any remaining group.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stream_child(command, cwd, environment, log):
    child = subprocess.Popen(command, cwd=cwd, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, start_new_session=True)
    try:
        for line in child.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = child.wait()
        return 128 - code if code < 0 else code
    finally:
        stop_process_group(child)
        child.stdout.close()


def terminate_launcher(signum, frame):
    raise SystemExit(128 + signum)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "evaluate":
        from .evaluation import evaluate
        report = evaluate(args)
        print(json.dumps({key: report[key] for key in ("checkpoint", "checkpoint_sha256", "episodes",
              "success_rate", "mean_team_return", "mean_agent_return", "checkpoint_progress")}, sort_keys=True))
        return 0
    if args.command not in ("train", "resume"):
        from .study import dispatch
        result = dispatch(args)
        if isinstance(result, int):
            return result
        print(json.dumps(result, sort_keys=True))
        return 0
    if not args.dry_run and args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f"Run output already exists: {args.output}")
    protocol = resolve_resume(args) if args.command == "resume" else resolve(args)
    if args.dry_run:
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0
    if not (HERE / "ORIGINS.json").is_file():
        raise RuntimeError("Missing reconstruction source provenance")
    if args.command == "train":
        validate_source_origins()
    run = Path(protocol["output"])
    # Atomic exclusive creation also protects failed and interrupted runs.
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(exist_ok=False)
    code = 1
    previous_term = signal.signal(signal.SIGTERM, terminate_launcher)
    try:
        if args.command == "resume":
            copy_resume_source(args.run_dir.resolve(), run)
        else:
            archive_source(run)
        write_json(run / "protocol.json", protocol)
        write_json(run / "environment.json", {"python": sys.version, "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                         if d.metadata["Name"]}, "git_head": subprocess.check_output(
                             ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        environment = {**os.environ, "DGLBACKEND": "pytorch", "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        # Execute the archived bytes, making ongoing repository edits irrelevant.
        with (run / "stdout.log").open("x") as log:
            code = stream_child(protocol["command"], run / "source/runtime", environment, log)
        return code
    except KeyboardInterrupt:
        code = 130
        return code
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        raise
    finally:
        (run / "exit_code.txt").write_text(f"{code}\n")
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())
