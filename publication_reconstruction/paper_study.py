"""Prepare fresh, explicitly matched paper-v1 study commands; never submit jobs."""
from pathlib import Path
import hashlib
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_FILES = (
    "slurm/publication_backend_benchmark.sbatch",
    "slurm/publication_paper_preflight.sbatch",
    "slurm/publication_paper_train.sbatch",
    "slurm/softrole_paper_fc.sbatch",
    "slurm/publication_resume.sbatch",
    "slurm/publication_evaluate.sbatch",
)
DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")


def _manifest(files):
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "files": dict(sorted(files.items()))}


def prepared_identity():
    """Read source/dependency/launcher identities without accepting new origins.

    Preparation is not a launch gate: a pending provenance amendment is recorded
    as invalid, never silently accepted. Training still requires the established
    ORIGINS validation. Reprepare after final source review before cluster use.
    """
    from .__main__ import inspect_source_origins
    from softrole.train import source_snapshot
    from softrole.publication_env import simulator_identity

    runtime, origins, audit = inspect_source_origins()
    softrole = source_snapshot()
    files = dict(softrole["files"])
    payloads = {"publication_reconstruction/runtime/" + name: payload
                for name, payload in runtime.items()}
    payloads["publication_reconstruction/ORIGINS.json"] = origins
    for path in sorted((ROOT / "publication_reconstruction").glob("*.py")):
        payloads[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("README.md", "FIDELITY.md", "STUDY.json"):
        path = ROOT / "publication_reconstruction" / name
        payloads[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name, payload in payloads.items():
        checksum = hashlib.sha256(payload).hexdigest()
        if name in files and files[name] != checksum:
            raise RuntimeError("Source changed while preparing study identity: " + name)
        files[name] = checksum
    dependencies = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in DEPENDENCY_FILES}
    launchers = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in LAUNCHER_FILES}
    for name, checksum in {**dependencies, **launchers}.items():
        if name in files and files[name] != checksum:
            raise RuntimeError("Source changed while preparing study identity: " + name)
        files[name] = checksum
    simulator = simulator_identity()
    if any(files.get(name) != checksum for name, checksum in simulator["files"].items()):
        raise RuntimeError("Simulator changed while preparing its shared source identity")
    return {"source": _manifest(files), "dependencies": _manifest(dependencies),
            "launchers": _manifest(launchers), "simulator_source": simulator,
            "git_commit": softrole["git_commit"],
            "runtime_provenance": {key: audit[key] for key in
                ("valid", "origins_sha256", "missing", "unexpected", "changed")},
            "meaning": "Reviewed preparation bytes; verify against executed run archives. "
                       "Reprepare after source/provenance changes; this does not certify a launch."}


def add_commands(commands):
    cli = commands.add_parser("paper-study-plan", help="prepare12 HetNet +6 matched SoftRole FC protocols")
    cli.add_argument("--run-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--message-backend", choices=["dgl", "torch-v1"], default="dgl")
    cli.add_argument("--dry-run", action="store_true")


def study_plan(run_root, backend):
    from .__main__ import parser, resolve
    from .study import train_arguments, panel
    from softrole.config import recipe
    run_root = Path(run_root).resolve()
    hetnet = [dict(array_index=i, **resolve(parser().parse_args(train_arguments(
        i, run_root / "hetnet", reconstruction_spec="paper-v1", message_backend=backend))))
        for i in range(12)]
    softrole = []
    for model in ("shared", "banked"):
        for seed in range(3):
            output = run_root / "softrole_fc" / model / f"seed{seed}"
            command = [sys.executable, "-u", "-m", "softrole", "train", "--task", "fc",
                       "--model", model, "--seed", str(seed), "--env-version", "paper-v1",
                       "--nprocesses", "4", "--batch-steps", "500", "--epochs", "1400",
                       "--updates-per-epoch", "10", "--total-steps", "28000000",
                       "--save-every", "10", "--output", str(output)]
            config = recipe("fc", model=model, seed=seed, env_version="paper-v1", total_steps=28_000_000,
                            epochs=1400, nprocesses=4, batch_steps=500, save_every=10)
            softrole.append({"array_index": len(softrole), "model": model, "seed": seed,
                             "output": str(output), "command": command, "config": config.to_dict()})
    scenarios = panel("fc", [(2, 1)], 500, 2703)
    payload = (json.dumps(scenarios, indent=2, sort_keys=True) + "\n").encode()
    return {"schema_version": 1, "study": "paper-v1-matched-fc", "submitted": False,
            "prepared_identity": prepared_identity(),
            "hetnet": hetnet, "softrole_fc": softrole, "fc_scenarios": scenarios,
            "fc_scenarios_sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint_selection": "first saved checkpoint at or above28M environment steps for each FC model/seed",
            "budget_basis": "28MFC/40MPP-PCP are retained study budgets, not recovered publication budgets",
            "comparison_boundary": "Old FC training/evaluation uses different physical actions/rewards and must remain separate",
            "binary_interpretation": "64 bits per independent head;256 total per sender/round",
            "launch_gate": "fidelity/backend/recovery validation; same-spec paired Stokes benchmark; official100-update preflight"}


def prepare(args):
    from .__main__ import write_json
    result = study_plan(args.run_root, args.message_backend)
    if args.dry_run:
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "study.json", result)
    write_json(args.output / "fc_scenarios.json", result["fc_scenarios"])
    print(json.dumps({"output": str(args.output.resolve()), "hetnet_runs": 12,
                      "matched_softrole_fc_runs": 6, "submitted": False}))
    return 0
