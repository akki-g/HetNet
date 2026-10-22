"""Fresh PCP seeds 3/4/5; preserve the existing trainers and scientific recipes.

Indices 0..2: HetNet paper-v1 Real; 3..5: CapCon beta0; 6..8: beta0.5.
Indices 9..11: CapCon team critic; 12..14: decomposed beta1.
Indices 15..41: banked K=2,3,4 x decomposed beta=1,.5,0 x seeds=3,4,5.
This is a separate extension manifest, not a change to programme.SEEDS or a
manifest accepted by the original seeds-0..2 programme/evaluation interface.
"""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
STUDY = "pcp-additional-seeds-v3"
SEEDS = (3, 4, 5)
ARMS = ("hetnet_pcp_real", "l1a_decomposed_beta0", "l1a_decomposed_beta05",
        "c1_real64", "l1b_decomposed_beta1")
# Preserve every v2 index; append the declared actor/K factorial only.
BANKED_ARMS = {
    f"banked_k{k}_decomposed_beta{beta}": (base, k)
    for k in (2, 3, 4)
    for beta, base in (("1", "l1b_decomposed_beta1"),
                       ("05", "l1a_decomposed_beta05"),
                       ("0", "l1a_decomposed_beta0"))
}
ARMS += tuple(BANKED_ARMS)
ENTRY_COUNT = len(ARMS) * len(SEEDS)


def index_value(value):
    if not isinstance(value, str) or value not in tuple(map(str, range(ENTRY_COUNT))):
        raise argparse.ArgumentTypeError(f"index must be a canonical decimal integer, 0..{ENTRY_COUNT - 1}")
    return int(value)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def entry(index, run_root):
    if type(index) is not int or not 0 <= index < ENTRY_COUNT:
        raise ValueError(f"index must be 0..{ENTRY_COUNT - 1}")
    seed = SEEDS[index % 3]
    arm = ARMS[index // 3]
    task = Path(run_root).resolve() / arm / f"seed{seed}"
    result = dict(index=index, arm=arm, seed=seed, output=str(task / "training"),
                  fresh=True, resume=None, starting_counts=dict(env_steps=0, episodes=0, updates=0))
    if index < 3:
        from publication_reconstruction.__main__ import parser, resolve
        from publication_reconstruction.study import train_arguments
        args = train_arguments(3, run_root, reconstruction_spec="paper-v1", message_backend="torch-v1")
        args[args.index("--seed") + 1] = str(seed)
        args[args.index("--output") + 1] = result["output"]
        result.update(kind="hetnet", arguments=args, resolved=resolve(parser().parse_args(args)))
    else:
        from softrole.programme import CAPCON_PROFILE, normalized, programme_config
        base, experts = BANKED_ARMS.get(arm, (arm, None))
        config = programme_config(base, seed, .001, profile=CAPCON_PROFILE)
        if experts is not None:
            config = replace(config, model="banked", experts=experts)
        config.validate()
        config = normalized(config)
        result.update(kind="capcon", resolved=config, episode_log="chunked-gzip", diagnostics_every=50)
    return result


def environment_identity():
    """Reject a wrong activation/dependency set before creating run artifacts."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the existing pinned Python 3.12 environment")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    from packaging.requirements import Requirement
    installed = {}
    for requirement in project["project"]["dependencies"]:
        required = Requirement(requirement)
        actual = importlib.metadata.version(required.name)
        if actual not in required.specifier:
            raise RuntimeError(f"Wrong environment: {required.name} {actual}, expected {required.specifier}")
        installed[required.name] = actual
    # Package metadata alone does not check compiled-library compatibility.
    import dgl  # noqa: F401
    import numpy  # noqa: F401
    import torch
    torch.set_num_threads(1)
    return dict(executable=sys.executable, python=sys.version, packages=installed, torch_threads=1)


def baseline_manifest(path):
    from softrole.programme import CAPCON_PROFILE, validate_manifest
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing admitted CapCon manifest: {path}. Use the existing Stokes 944671 preparation; no training was started.")
    payload = path.read_bytes()
    baseline = validate_manifest(json.loads(payload))
    if (baseline["programme"] != CAPCON_PROFILE or baseline["bounded_check"] is not False
            or baseline["lr"] != .001 or baseline.get("diagnostics_every") != 50):
        raise ValueError("Require the non-bounded capcon_l1_40m_v1 manifest, Adam .001, diagnostics every 50 updates")
    return dict(path=str(path), sha256=digest(payload), original_source_sha256=baseline["source_sha256"],
                lr_selection=baseline["lr_selection"], lr_selection_sha256=baseline["lr_selection_sha256"])


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=ROOT)
    patch = git("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    names = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    untracked = {}
    for name in filter(None, names):
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Cannot archive untracked source path: {name}")
        untracked[name] = digest(path.read_bytes())
    identity = dict(commit=git("rev-parse", "HEAD").decode().strip(),
                    status=git("status", "--short", "--untracked-files=all").decode(),
                    diff_sha256=digest(patch), untracked_files=untracked)
    return identity, patch


def prepare(run_root, baseline, environment):
    """Lock one source/configuration identity across all members of an array."""
    from softrole.train import source_snapshot
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = source_snapshot()
        git, patch = git_identity()
        plan = dict(version=1, study=STUDY, seeds=list(SEEDS),
                    entries=[entry(i, run_root) for i in range(ENTRY_COUNT)],
                    source=source, git=git, baseline_capcon=baseline,
                    environment=environment,
                    scope=("Additional independent training seeds and declared banked actor/K factorial; "
                           "banked entries change only model and experts from corresponding shared "
                           "decomposed recipes. Baseline S1 receipts support the reused baseline "
                           "optimizer choice, not independent certification of banked actors. "
                           "No automatic submission or resume"),
                    losses="CapCon actor/value 50/1; HetNet active paper actor/critic 1/1; no equalization")
        target = run_root / "manifest.json"
        if target.exists():
            previous = json.loads(target.read_text())
            if previous != plan:
                raise ValueError("Array source, Git diff, environment or resolved configuration changed; preserve this tree and use a fresh submission")
            return previous
        archive = run_root / "provenance"
        archive.mkdir(exist_ok=False)  # A failed preparation is never overwritten.
        archived = source_snapshot(archive)
        if archived != source:
            raise ValueError("Source changed during provenance capture")
        (archive / "working_tree.patch").write_bytes(patch)
        for name, checksum in git["untracked_files"].items():
            payload = (ROOT / name).read_bytes()
            if digest(payload) != checksum:
                raise ValueError("Untracked source changed during provenance capture: " + name)
            path = archive / "untracked" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if git_identity()[0] != git:
            raise ValueError("Git state changed during provenance capture")
        temporary = run_root / ".manifest.pending.json"
        write_json(temporary, plan)
        temporary.rename(target)
        return plan


def run_entry(selected, manifest_path, environment):
    """Use existing launch/train functions, with an exclusive fresh parent."""
    output = Path(selected["output"])
    output.parent.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(manifest_path).read_text())
    from softrole.train import source_snapshot
    if source_snapshot() != manifest["source"]:
        raise ValueError("Checkout changed after array preparation")
    write_json(output.parent / "launch.json", dict(
        **selected, manifest=str(Path(manifest_path).resolve()),
        manifest_sha256=digest(Path(manifest_path).read_bytes()), environment=environment,
        slurm={key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_ARRAY_TASK_COUNT")}))
    print(json.dumps({"record_type":"additional_seed_launch", **selected}, sort_keys=True), flush=True)
    code = 1
    try:
        if selected["kind"] == "hetnet":
            from publication_reconstruction.__main__ import main as launch
            code = launch(selected["arguments"])
        else:
            from softrole.config import Config
            from softrole.train import train
            train(Config(**selected["resolved"]), output, episode_log=selected["episode_log"],
                  diagnostics_every=selected["diagnostics_every"], checkpoint_commits=True)
            code = 0
        return code
    finally:
        write_json(output.parent / "launcher_exit.json", {"exit_code":code})


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--index", type=index_value, required=True)
    cli.add_argument("--run-root", type=Path, required=True)
    cli.add_argument("--capcon-manifest", type=Path,
                     default=ROOT / "runs/capcon_l1_40m_20261017/manifest.json")
    cli.add_argument("--dry-run", action="store_true", help="print configuration only; no files, imports of trainers, or execution")
    args = cli.parse_args(argv)
    selected = entry(args.index, args.run_root)
    if args.dry_run:
        print(json.dumps(selected, indent=2, sort_keys=True))
        return 0
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_TASK_ID") != str(args.index)
            or os.environ.get("SLURM_CPUS_PER_TASK") != "4"):
        raise RuntimeError("Launch only through the four-CPU Slurm array; --dry-run is safe locally")
    environment = environment_identity()
    baseline = baseline_manifest(args.capcon_manifest)
    plan = prepare(args.run_root, baseline, environment)
    if selected != plan["entries"][args.index]:
        raise ValueError("Resolved entry changed during preparation")
    return run_entry(selected, args.run_root.resolve() / "manifest.json", environment)


if __name__ == "__main__":
    raise SystemExit(main())
