"""Launch a frozen source copy with fresh evidence and complete-update budgets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="separate reconstruction run; never overwrites")
    train.add_argument("--task", choices=["pp", "pcp", "fc"], required=True)
    train.add_argument("--variant", choices=["real", "binary"], default="real")
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
    train.add_argument("--dry-run", action="store_true")
    return p


def resolve(args):
    fc = args.task == "fc"
    epochs = args.epochs or (1400 if fc else 2000)
    horizon = args.horizon or (300 if fc else 80)
    collectors = args.collectors or (4 if fc or args.recipe == "october-2022" else 1)
    if args.seed + collectors >= 2**32:
        raise ValueError("seed plus collector count exceeds NumPy seed range")
    run = (args.output or ROOT / "runs/publication_reconstruction" / args.env_version /
           args.recipe / f"{args.task}_{args.variant}" / f"seed{args.seed}").resolve()
    # The June PP wrapper needs --nagents3, absent from the released example.
    environment = "fire_commander" if fc else (
        "predator_prey" if args.task == "pp" and args.recipe == "june-2022" else "predator_capture")
    command = [sys.executable, "-u", str(run / "source/runtime/main.py"),
        "--env_name", environment, "--nfriendly_P", "3" if args.task == "pp" else "2",
        "--nfriendly_A", "0" if args.task == "pp" else "1", "--nagents", "3",
        "--hetgat", "--hetgat_a2c", "--num_epochs", str(epochs),
        "--nprocesses", str(collectors), "--epoch_size", str(args.updates_per_epoch),
        "--batch_size", str(args.batch_steps), "--max_steps", str(horizon),
        "--detach_gap", "5", "--lrate", "0.0001", "--hid_size", "128", "--dim", "5",
        "--vision", "1" if fc else "2", "--seed", str(args.seed),
        "--save_every", str(args.save_every), "--experiment_name", f"{args.task}_{args.variant}",
        "--save_dir", str(run / "checkpoints"), "--metrics_file", str(run / "metrics.jsonl"),
        "--publication_env_version", args.env_version, "--source_manifest", str(run / "source_manifest.json"),
        "--max_env_steps", str(args.max_env_steps or 0)]
    if fc:
        command += ["--nfires", "1", "--reward_type", "3"]
    if args.variant == "binary":
        command += ["--use_binary", "--msg_dim", "16"]
    floor = epochs * args.updates_per_epoch * collectors * args.batch_steps
    batch_max = collectors * (args.batch_steps + horizon - 1)
    return {"schema_version": SCHEMA_VERSION, "task": args.task, "variant": args.variant,
            "env_version": args.env_version, "recipe": args.recipe, "seed": args.seed,
            "output": str(run), "command": command, "epochs": epochs, "collectors": collectors,
            "updates_per_epoch": args.updates_per_epoch, "batch_step_floor_per_collector": args.batch_steps,
            "episode_horizon": horizon, "epoch_budget_min_steps_without_step_cap": floor,
            "max_env_steps": args.max_env_steps, "max_overshoot_steps": batch_max - 1,
            "upstream_reference": UPSTREAM_REFERENCE, "runtime_scaffold_reference": SCAFFOLD_REFERENCE,
            "claim": "runnable reconstruction, not a verified publication training commit",
            "architecture": "two HetGAT layers; four heads; per-class critics",
            "optimizer": {"name": "RMSprop", "lr": 1e-4, "alpha": .97, "epsilon": 1e-6},
            "budget_semantics": "first of epoch cap or complete-update step threshold; no partial episodes"}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_source_origins():
    origins_bytes = (HERE / "ORIGINS.json").read_bytes()
    origins = json.loads(origins_bytes)["files"]
    paths = [p for p in sorted((HERE / "runtime").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    payloads = {p.relative_to(HERE / "runtime").as_posix(): p.read_bytes() for p in paths}
    if payloads.keys() != origins.keys():
        raise RuntimeError("Runtime file inventory differs from ORIGINS.json; regenerate provenance")
    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != origins[name]["current_sha256"]:
            raise RuntimeError(f"Runtime source differs from ORIGINS.json: {name}")
    return payloads, origins_bytes


def archive_source(run):
    runtime, origins_bytes = validate_source_origins()
    payloads = {f"runtime/{name}": payload for name, payload in runtime.items()}
    payloads["ORIGINS.json"] = origins_bytes
    for name in ["__init__.py", "__main__.py", "README.md"]:
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


def main():
    args = parser().parse_args()
    protocol = resolve(args)
    if args.dry_run:
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0
    if not (HERE / "ORIGINS.json").is_file():
        raise RuntimeError("Missing reconstruction source provenance")
    validate_source_origins()
    run = Path(protocol["output"])
    # Atomic exclusive creation also protects failed and interrupted runs.
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(exist_ok=False)
    code = 1
    previous_term = signal.signal(signal.SIGTERM, terminate_launcher)
    try:
        archive_source(run)
        write_json(run / "protocol.json", protocol)
        write_json(run / "environment.json", {"python": sys.version, "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                         if d.metadata["Name"]}, "git_head": subprocess.check_output(
                             ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        environment = {**os.environ, "DGLBACKEND": "pytorch", "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
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
