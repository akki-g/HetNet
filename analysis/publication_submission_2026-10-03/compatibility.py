"""Compare actual public-code training with the immutable pre-change runtime.

Run from this repository with its pinned Python environment.  The historical git
object named by --reference is an explicit prerequisite; this is a standalone
audit, not a pytest test that silently requires a full-history checkout.
All copied source, checkpoints and process logs live in a TemporaryDirectory.
Only the requested small JSON evidence file is retained. No research job runs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["DGLBACKEND"] = "pytorch"

import numpy as np
import torch


PROBE = '''import atexit, hashlib, json, os, pickle, sys
def record_final_rng():
    if "trainer" not in sys.modules or "torch" not in sys.modules:
        return
    import multiprocessing, random
    from pathlib import Path
    import numpy as np
    import torch
    state = np.random.get_state()
    name = multiprocessing.current_process().name
    result = {
        "python_sha256": hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
        "numpy_sha256": hashlib.sha256(state[1].tobytes() + repr((state[0], *state[2:])).encode()).hexdigest(),
        "torch_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "next_draws": {"python": [random.random() for _ in range(5)],
                       "numpy": np.random.normal(size=5).tolist(),
                       "torch": torch.rand(5).tolist()},
    }
    Path(os.environ["COMPAT_RNG_DIR"], name + ".json").write_text(json.dumps(result, sort_keys=True))
atexit.register(record_final_rng)
'''


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def source_hashes(runtime):
    return {path.relative_to(runtime).as_posix(): sha(path.read_bytes())
            for path in sorted(runtime.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def state_digest(value):
    def normalized(item):
        if torch.is_tensor(item):
            raw = item.detach().cpu().contiguous()
            return {"shape": list(raw.shape), "dtype": str(raw.dtype),
                    "sha256": sha(raw.reshape(-1).view(torch.uint8).numpy().tobytes())}
        if isinstance(item, np.ndarray):
            return {"shape": list(item.shape), "dtype": str(item.dtype), "sha256": sha(item.tobytes())}
        if isinstance(item, dict):
            return {str(k): normalized(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalized(v) for v in item]
        return item
    return sha(json.dumps(normalized(value), sort_keys=True, allow_nan=False).encode())


def execute(runtime, destination, probe, binary, collectors):
    destination.mkdir()
    rng_dir = destination / "rng"
    rng_dir.mkdir()
    manifest = destination / "source_manifest.json"
    manifest.write_text(json.dumps({"files": source_hashes(runtime)}, sort_keys=True))
    command = [sys.executable, "-u", str(runtime / "main.py"),
        "--publication_env_version", "corrected-v1", "--source_manifest", str(manifest),
        "--env_name", "predator_capture", "--nfriendly_P", "2", "--nfriendly_A", "1", "--nagents", "3",
        "--hetgat", "--hetgat_a2c", "--seed", "11", "--nprocesses", str(collectors),
        "--num_epochs", "2", "--epoch_size", "2", "--batch_size", "4", "--max_steps", "3",
        "--detach_gap", "5", "--dim", "5", "--vision", "2", "--hid_size", "128", "--lrate", "0.0001",
        "--save_every", "1", "--save_dir", str(destination / "checkpoints"),
        "--metrics_file", str(destination / "metrics.jsonl"), "--experiment_name", "compatibility"]
    if binary:
        command += ["--use_binary", "--msg_dim", "16"]
    env = {**os.environ, "PYTHONPATH": str(probe), "COMPAT_RNG_DIR": str(rng_dir)}
    result = subprocess.run(command, cwd=runtime, env=env, text=True, capture_output=True, timeout=100)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:] + result.stderr[-6000:])
    paths = list((destination / "checkpoints").rglob("*.pt"))
    checkpoints = [(p, torch.load(p, map_location="cpu", weights_only=False)) for p in paths]
    path, saved = max(checkpoints, key=lambda pair: pair[1]["reconstruction"]["counts"]["updates"])
    metrics = [json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()]
    rng = {p.stem: json.loads(p.read_text()) for p in sorted(rng_dir.glob("*.json"))}
    if len(rng) != collectors:
        raise ValueError(f"Expected {collectors} exited collector RNG records, got {list(rng)}")
    return saved, metrics, rng, {"checkpoint_sha256": sha(path.read_bytes()),
        "stdout_sha256": sha(result.stdout.encode()), "stderr_sha256": sha(result.stderr.encode())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="9a436e6d9f8d36c864c17744d4d55819f5e5e504")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    archive = subprocess.run(["git", "archive", args.reference + ":publication_reconstruction/runtime"],
                             cwd=ROOT, check=True, capture_output=True).stdout
    evidence = {"schema_version": 1, "purpose": "bounded engineering compatibility audit",
        "reference_commit": args.reference, "reference_archive_sha256": sha(archive),
        "audit_script_sha256": sha(Path(__file__).read_bytes()), "exit_rng_probe_sha256": sha(PROBE.encode()),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "runs": []}
    with tempfile.TemporaryDirectory(prefix="publication-compatibility-") as temporary:
        temporary = Path(temporary)
        original = temporary / "original"
        original.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(original, filter="data")
        current = temporary / "current"
        shutil.copytree(ROOT / "publication_reconstruction/runtime", current,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        evidence["reference_source_sha256"] = source_hashes(original)
        evidence["current_source_sha256"] = source_hashes(current)
        probe = temporary / "probe"
        probe.mkdir()
        (probe / "sitecustomize.py").write_text(PROBE)
        for binary in (False, True):
            for collectors in (1, 4):
                name = ("binary" if binary else "real") + f"_{collectors}collectors"
                old, old_metrics, old_rng, old_files = execute(original, temporary / ("old_" + name), probe, binary, collectors)
                new, new_metrics, new_rng, new_files = execute(current, temporary / ("new_" + name), probe, binary, collectors)
                common_metrics = [{k: v for k, v in row.items() if k != "wall_time_seconds"} for row in old_metrics]
                checks = {"model_tensors_equal": equal(old["policy_net"], new["policy_net"]),
                          "rmsprop_equal": equal(old["trainer"], new["trainer"]),
                          "historical_log_equal": equal(old["log"], new["log"]),
                          "counts_equal": old["reconstruction"]["counts"] == new["reconstruction"]["counts"],
                          "numeric_epoch_metrics_equal": common_metrics == [{k: row[k] for k in keys} for row, keys in zip(new_metrics, common_metrics)],
                          "all_process_rng_states_and_next_draws_equal": old_rng == new_rng}
                if not all(checks.values()):
                    raise AssertionError({"workload": name, "checks": checks})
                evidence["runs"].append({"workload": name, "checks": checks,
                    "counts": new["reconstruction"]["counts"], "metrics": common_metrics,
                    "model_value_sha256": state_digest(new["policy_net"]),
                    "optimizer_value_sha256": state_digest(new["trainer"]),
                    "rng": new_rng, "reference_files": old_files, "current_files": new_files})
                print(name + ": exact learner, optimizer, metrics and all RNG streams", flush=True)
        if source_hashes(ROOT / "publication_reconstruction/runtime") != evidence["current_source_sha256"]:
            raise RuntimeError("Runtime changed during the compatibility audit")
    evidence["all_checks_passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
