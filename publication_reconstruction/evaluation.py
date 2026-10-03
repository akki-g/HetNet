"""Frozen evaluation dispatch: verify archived sources, then isolate their imports."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def add_arguments(parser):
    for name in ("run-dir", "checkpoint", "scenarios", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        help="declared evaluation distribution and checkpoint-selection JSON")
    parser.add_argument("--sham", action="store_true")
    parser.add_argument("--trace", action="store_true")
    return parser


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_identity(run):
    """Validate all archived bytes, including the complete executable inventory."""
    manifest_path = run / "source_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Invalid archived source manifest")
    source = (run / "source").resolve()
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source manifest paths must remain inside the archive")
        path = source / relative
        if not path.resolve().is_relative_to(source) or digest(path.read_bytes()) != expected:
            raise ValueError(f"Archived source hash mismatch: {name}")
    actual = {p.relative_to(source).as_posix() for p in (source / "runtime").rglob("*.py")}
    declared = {name for name in files if name.startswith("runtime/") and name.endswith(".py")}
    if actual != declared:
        raise ValueError("Archived runtime Python inventory differs from its manifest")
    for required in ("runtime/hetgat/uavnet.py", "runtime/hetgat/utils.py",
                     "runtime/envs/ic3net_envs/predator_capture_env.py"):
        if required not in files:
            raise ValueError(f"Archived source manifest lacks {required}")
    return {"sha256": digest(manifest_bytes), "manifest": manifest,
            "runtime": str(source / "runtime")}


def evaluator_snapshot():
    names = ("publication_reconstruction/__init__.py", "publication_reconstruction/__main__.py",
             "publication_reconstruction/evaluation.py", "publication_reconstruction/evaluation_worker.py",
             "publication_reconstruction/artifacts.py",
             "publication_reconstruction/runtime/hetnet_ext/__init__.py",
             "publication_reconstruction/runtime/hetnet_ext/recovery.py", "softrole/__init__.py",
             "softrole/scenarios.py", "softrole/rollout.py", "softrole/learning.py",
             "softrole/report.py")
    payloads = {name: (ROOT / name).read_bytes() for name in names}
    files = {name: digest(payload) for name, payload in payloads.items()}
    identity = {"files": files, "sha256": digest(json.dumps(files, sort_keys=True).encode())}
    if (ROOT / "manifest.json").is_file():
        verify_evaluator_archive(ROOT, identity)
    return identity, payloads


def evaluator_identity():
    return evaluator_snapshot()[0]


def archive_path(output):
    return Path(str(output) + ".sources")


def write_evaluator_archive(destination, identity, payloads):
    """Write exactly the captured bytes into an exclusively created archive."""
    if set(payloads) != set(identity["files"]) or any(
            digest(payload) != identity["files"][name] for name, payload in payloads.items()):
        raise ValueError("Captured evaluator bytes do not match their identity")
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    manifest = {"schema_version": 1, "source": identity,
                "replay": "python -I -B <this-directory>/publication_reconstruction/evaluation.py --run-dir <training-run> --checkpoint <checkpoint> --scenarios <panel.json> --output <fresh-report.json> [--protocol <protocol.json>] [--sham] [--trace]"}
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return verify_evaluator_archive(destination, identity)


def verify_evaluator_archive(destination, identity):
    """Check archive inventory and bytes without referring to the live checkout."""
    destination = Path(destination)
    manifest_bytes = (destination / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or manifest.get("source") != identity:
        raise ValueError("Evaluator archive manifest differs from the declared identity")
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    if actual != set(identity["files"]) | {"manifest.json"}:
        raise ValueError("Evaluator archive inventory differs from its manifest")
    for name, expected in identity["files"].items():
        path = destination / name
        if not path.resolve().is_relative_to(destination.resolve()) or digest(path.read_bytes()) != expected:
            raise ValueError(f"Evaluator archive hash mismatch: {name}")
    return digest(manifest_bytes)


def evaluate(args):
    run, checkpoint = Path(args.run_dir).resolve(), Path(args.checkpoint).resolve()
    output, scenarios_path = Path(args.output), Path(args.scenarios).resolve()
    archive = archive_path(output)
    if (ROOT / "manifest.json").is_file() and output.resolve().is_relative_to(ROOT):
        raise ValueError("Replay output must remain outside the immutable evaluator archive")
    for destination in (output, archive):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Evaluation output already exists: {destination}")
    run_protocol_bytes = (run / "protocol.json").read_bytes()
    training = source_identity(run)
    evaluator, evaluator_bytes = evaluator_snapshot()
    checkpoint_hash = digest(checkpoint.read_bytes())
    scenarios_bytes = scenarios_path.read_bytes()
    scenarios = json.loads(scenarios_bytes)
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Evaluation requires a nonempty scenario list")
    protocol_path = getattr(args, "protocol", None)
    protocol_bytes = Path(protocol_path).read_bytes() if protocol_path else None
    request = {"root": str(ROOT), "run_dir": str(run), "runtime": training["runtime"],
               "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
               "training_source_sha256": training["sha256"], "evaluator_source": evaluator,
               "scenarios": scenarios, "scenarios_sha256": digest(scenarios_bytes),
               "evaluation_protocol": json.loads(protocol_bytes) if protocol_bytes else None,
               "sham": bool(args.sham), "trace": bool(args.trace)}
    environment = dict(os.environ, DGLBACKEND="pytorch", PYTHONDONTWRITEBYTECODE="1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    # Execute the captured evaluator/helper bytes, separate from the archived
    # actor/environment runtime. A fresh cache prefix excludes stale .pyc files.
    with tempfile.TemporaryDirectory(prefix="hetnet-evaluation-") as temporary:
        temporary = Path(temporary)
        staged = temporary / "sources"
        manifest_hash = write_evaluator_archive(staged, evaluator, evaluator_bytes)
        request["root"] = str(staged)
        worker = staged / "publication_reconstruction/evaluation_worker.py"
        result = subprocess.run([sys.executable, "-I", "-B", "-X", f"pycache_prefix={temporary / 'cache'}",
                                 str(worker)], input=json.dumps(request), text=True,
                                capture_output=True, env=environment, cwd=run)
        if result.returncode:
            raise ValueError("Reconstruction evaluation failed:\n" + result.stderr.strip())
        report = json.loads(result.stdout)
        if (verify_evaluator_archive(staged, evaluator) != manifest_hash
                or source_identity(run) != training or evaluator_identity() != evaluator
                or (run / "protocol.json").read_bytes() != run_protocol_bytes
                or digest(checkpoint.read_bytes()) != checkpoint_hash
                or scenarios_path.read_bytes() != scenarios_bytes
                or (protocol_path and Path(protocol_path).read_bytes() != protocol_bytes)):
            raise RuntimeError("Evaluation input or source changed during frozen evaluation")
        report["evaluator"]["source_archive"] = {
            "path": str(archive.resolve()), "manifest_sha256": manifest_hash,
            "source_sha256": evaluator["sha256"]}
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        # Reserve both final paths exclusively. Roll back only our own newly
        # created archive if a concurrent writer wins the report path.
        archive.mkdir(exist_ok=False)
        report_created = False
        try:
            for source in staged.rglob("*"):
                if source.is_file():
                    target = archive / source.relative_to(staged)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(source.read_bytes())
            if verify_evaluator_archive(archive, evaluator) != manifest_hash:
                raise RuntimeError("Published evaluator archive differs from the evaluated snapshot")
            with output.open("x") as stream:
                report_created = True
                stream.write(encoded)
        except BaseException:
            if report_created:
                output.unlink()
            shutil.rmtree(archive)
            raise
    return report


if __name__ == "__main__":
    # A source archive can replay without importing the current checkout's CLI.
    import argparse
    arguments = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    result = evaluate(arguments)
    print(json.dumps({"output": str(Path(arguments.output).resolve()),
                      "episodes": result["episodes"], "parameters_unchanged": result["parameters_unchanged"]}))
