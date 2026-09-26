"""Run one unchanged 20-epoch CPU calibration on a native Apple-silicon Mac.

Dry-run is standard-library-only. Execution is deliberately separate from the
Stokes launcher: no invented Slurm allocation, full-study mode, or resume path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

from .grid import (DEFAULT_GRID, ROOT, build_command, calibration_entries,
                   code_inventory, code_sha256, file_sha256, run_directory)
from .train_job import checked_number, read_json, utc_now, validate_gate, write_json


THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "HETNET_TORCH_THREADS": "1", "DGLBACKEND": "pytorch",
              "PYTHONUNBUFFERED": "1"}
EXPECTED_RECIPE = {"env_name": "predator_capture", "num_epochs": 2000, "epoch_size": 10,
                   "nprocesses": 4, "hid_size": 128, "detach_gap": 5, "lrate": 0.0001,
                   "dim": 5, "batch_size": 500, "max_steps": 80, "save_every": 50,
                   "hetgat": True, "hetgat_a2c": True}
SAMPLE_INTERVAL_SECONDS = 5.0
RESOURCE_LIMITATIONS = (
    "Sampled sum of process-group RSS in KiB; shared pages can be counted more than once. "
    "A sampled maximum is not an exact whole-job peak. Descendants that leave the process group "
    "are outside this sampler. Swap and vm_stat are system-wide snapshots, not job-attributed "
    "memory or a calibrated macOS memory-pressure score."
)


def calibration_plan(index: int, output_root: Path) -> dict:
    settings = read_json(DEFAULT_GRID)
    if settings.get("recipe") != EXPECTED_RECIPE or settings.get("calibration") != {
        "num_epochs": 20, "compositions": ["2P1A", "4P6A"], "seed": 0
    }:
        raise ValueError("Local pilots require the unchanged committed endpoint calibration recipe")
    entries = calibration_entries()
    if type(index) is not int or index not in (0, 1):
        raise ValueError("Only calibration indices 0 and 1 are authorized")
    entry = entries[index]
    directory = run_directory(entry, "calibration", output_root)
    return {"entry": entry.as_dict(), "mode": "calibration", "num_epochs": 20,
            "recipe": settings["recipe"], "directory": str(directory),
            "command": build_command(entry, directory, mode="calibration", python=sys.executable)}


def validate_native_host() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Local pilots require native macOS arm64; no Rosetta, Linux, or GPU fallback")


def validate_runtime() -> dict:
    validate_native_host()
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The locked Python 3.12 interpreter is required")
    if any(os.environ.get(key) for key in ("PYTHONHOME", "PYTHONPATH")):
        raise ValueError("Clear inherited PYTHONHOME/PYTHONPATH before starting the locked interpreter")
    # Set thread controls before any numerical-library import in this process.
    os.environ.update(THREAD_ENV)
    import torch
    import dgl
    import torchdata
    import gym
    import numpy

    versions = {"torch": torch.__version__, "dgl": dgl.__version__,
                "torchdata": torchdata.__version__, "gym": gym.__version__, "numpy": numpy.__version__}
    expected = {"torch": "2.2.1", "dgl": "2.1.0", "torchdata": "0.7.1", "gym": "0.26.2", "numpy": "1.26.4"}
    if any(versions[name].split("+")[0] != version for name, version in expected.items()):
        raise ValueError(f"Runtime versions differ from the lock: {versions}")
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise ValueError("Torch must use exactly one intra-op thread")
    module_files = {name: str(Path(module.__file__).absolute()) for name, module in
                    (("torch", torch), ("dgl", dgl), ("torchdata", torchdata), ("gym", gym), ("numpy", numpy))}
    prefix = Path(sys.prefix).absolute()
    if prefix == Path(sys.base_prefix).absolute() or any(not Path(path).resolve().is_relative_to(prefix.resolve())
                                                        for path in module_files.values()):
        raise ValueError("Pinned numerical modules must all come from this virtual environment")
    return {"versions": versions, "python": sys.version, "executable": str(Path(sys.executable).absolute()),
            "prefix": str(prefix), "base_prefix": sys.base_prefix,
            "interpreter_sha256": file_sha256(sys.executable), "module_files": module_files,
            "module_sha256": {name: file_sha256(path) for name, path in module_files.items()},
            "system": platform.system(), "machine": platform.machine(), "hostname": platform.node(),
            "torch_num_threads": torch.get_num_threads(), "cpu_tensor_device": "cpu"}


def capture(command: list[str], *, required: bool = True, timeout: float = 5.0) -> dict:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, check=False)
        record = {"command": command, "returncode": result.returncode,
                  "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        record = {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    if required and record["returncode"] != 0:
        raise ValueError(f"Required provenance command failed: {record}")
    return record


def hardware_inventory(directory: Path) -> dict:
    raw = {}
    for key in ("hw.memsize", "hw.physicalcpu", "hw.logicalcpu", "machdep.cpu.brand_string", "hw.model", "hw.optional.arm64"):
        raw[key] = capture(["/usr/sbin/sysctl", "-n", key])
    for key in ("hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu", "sysctl.proc_translated"):
        raw[key] = capture(["/usr/sbin/sysctl", "-n", key], required=False)
    if raw["sysctl.proc_translated"]["stdout"] == "1":
        raise ValueError("Translated execution is not the native arm64 experiment")
    memory = int(raw["hw.memsize"]["stdout"])
    physical = int(raw["hw.physicalcpu"]["stdout"])
    logical = int(raw["hw.logicalcpu"]["stdout"])
    if memory <= 0 or physical < 4 or logical < 4:
        raise ValueError("Host must report positive RAM and at least four CPU cores")
    ancestor = directory
    while not ancestor.exists():
        ancestor = ancestor.parent
    disk = shutil.disk_usage(ancestor)
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_model": raw["machdep.cpu.brand_string"]["stdout"],
            "memory_bytes": memory, "physical_cores": physical, "logical_cores": logical,
            "sysctl": raw, "sw_vers": capture(["/usr/bin/sw_vers"]),
            "disk": {"path": str(ancestor.resolve()), "total_bytes": disk.total,
                     "used_bytes": disk.used, "free_bytes": disk.free}}


def validate_target_gate(gate: dict, hardware: dict, runtime: dict) -> None:
    from .gate_a import verify_clean_environment

    records = [r for r in gate.get("check_records", []) if r.get("name") == "clean_lock_environment"]
    if len(records) != 1:
        raise ValueError("Local Gate A must preserve exactly one clean target-environment proof")
    record = records[0]
    evidence_path = Path(record.get("evidence", ""))
    if not evidence_path.is_file() or file_sha256(evidence_path) != record.get("evidence_sha256"):
        raise ValueError("Gate A clean-environment evidence hash is missing or changed")
    verified = verify_clean_environment(evidence_path)
    proof = verified["runtime"]
    if record.get("runtime") != proof:
        raise ValueError("Gate A runtime differs from its verified clean-environment evidence")
    for key in ("executable", "prefix", "base_prefix", "interpreter_sha256", "python", "system", "machine", "hostname"):
        if proof.get(key) != runtime.get(key):
            raise ValueError(f"Gate A belongs to a different target runtime/host: {key}")
    for key in ("versions", "module_files", "module_sha256"):
        if any(proof.get(key, {}).get(name) != value for name, value in runtime[key].items()):
            raise ValueError(f"Gate A target environment differs: {key}")
    identity = {"model_identifier": hardware["sysctl"]["hw.model"]["stdout"],
                "memory_bytes": hardware["memory_bytes"], "physical_cpus": hardware["physical_cores"],
                "logical_cpus": hardware["logical_cores"], "cpu_brand": hardware["cpu_model"],
                "arm64_capable": int(hardware["sysctl"]["hw.optional.arm64"]["stdout"])}
    if proof.get("hardware_identity") != identity:
        raise ValueError("Gate A hardware identity differs from this Mac; rerun on the actual target")


def user_lock_path() -> Path:
    # /tmp resolves to the same inode namespace across this user's checkouts.
    return Path("/tmp") / f"hetnet-local-calibration-{os.getuid()}.lock"


@contextmanager
def single_job_lock():
    path = user_lock_path()
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
            raise ValueError("Local job lock must be a regular file owned by this user")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another local calibration holds the per-user lock: {path}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "checkout": str(ROOT), "started_at_utc": utc_now()}).encode())
        yield str(path)
    finally:
        # Never unlink: replacing the inode would allow two simultaneous locks.
        os.close(fd)


def sample_resources(process_group: int | None, elapsed: float, *, deadline: float | None = None) -> dict:
    sample = {"elapsed_seconds": elapsed, "process_group": process_group,
              "members": [], "group_rss_kib_sum": None, "process_count": None, "sampling_errors": []}
    def bounded_capture(command):
        remaining = 1.0 if deadline is None else min(1.0, deadline - time.monotonic())
        if remaining <= 0:
            return {"command": command, "returncode": None, "stdout": "", "stderr": "sampling deadline reached"}
        return capture(command, required=False, timeout=remaining)

    if process_group is not None:
        result = bounded_capture(["/bin/ps", "-axo", "pid=,pgid=,rss=,stat="])
        if result["returncode"] == 0:
            try:
                members = []
                for line in result["stdout"].splitlines():
                    pid, pgid, rss, state = line.split(None, 3)
                    if int(pgid) == process_group:
                        members.append({"pid": int(pid), "rss_kib": int(rss), "state": state})
                sample.update(members=members, group_rss_kib_sum=sum(p["rss_kib"] for p in members),
                              process_count=len(members))
            except (ValueError, TypeError) as exc:
                sample["sampling_errors"].append(f"ps parse: {exc}")
        else:
            sample["sampling_errors"].append(result)
    for name, command in (("system_swap_usage_raw", ["/usr/sbin/sysctl", "vm.swapusage"]),
                          ("vm_stat_raw", ["/usr/bin/vm_stat"])):
        result = bounded_capture(command)
        sample[name] = result["stdout"] if result["returncode"] == 0 else None
        if result["returncode"] != 0:
            sample["sampling_errors"].append(result)
    return sample


def group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can report EPERM for a just-terminated, unreaped group. Inspect
        # membership rather than mistaking that race for a live descendant.
        result = subprocess.run(["/bin/ps", "-axo", "pgid=,stat="], capture_output=True,
                                text=True, check=True, timeout=1)
        return any(int(parts[0]) == pgid and not parts[1].startswith("Z")
                   for line in result.stdout.splitlines() if len(parts := line.split()) == 2)


def terminate_group(child: subprocess.Popen, *, grace_seconds: float = 2.0) -> None:
    """Terminate the entire session group even if its original parent exited."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    child.poll()
    while group_exists(child.pid) and time.monotonic() < deadline:
        child.poll()  # Reap our direct child; grandchildren are reaped by their parent/init.
        time.sleep(0.02)
    if group_exists(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=5)


def run_child(command: list[str], directory: Path, max_wall_seconds: float,
              launcher_started: float, env: dict) -> dict:
    """Drain output without blocking deadlines; always clean up the process group."""
    checked_number(max_wall_seconds, "maximum child wall time")
    started = time.monotonic()
    result = {"returncode": None, "stop_reason": None, "signals": [],
              "setup_wall_time_seconds": started - launcher_started}
    samples = []
    child = None
    old_handlers = {}

    def stop(signum, _frame):
        result["signals"].append(signum)

    try:
        old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        with (directory / "stdout.log").open("xb") as log, (directory / "resource_samples.jsonl").open("x") as resources:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
            result.update(pid=child.pid, process_group=child.pid, started_at_utc=utc_now())
            assert child.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                next_sample = 0.0
                while True:
                    elapsed = time.monotonic() - started
                    if result["signals"]:
                        result["stop_reason"] = "signal"
                        break
                    if elapsed >= max_wall_seconds:
                        result["stop_reason"] = "timeout"
                        break
                    if elapsed >= next_sample:
                        sample = sample_resources(child.pid, elapsed, deadline=started + max_wall_seconds)
                        samples.append(sample)
                        resources.write(json.dumps(sample, sort_keys=True) + "\n")
                        resources.flush()
                        next_sample = elapsed + SAMPLE_INTERVAL_SECONDS
                    remaining = max_wall_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        result["stop_reason"] = "timeout"
                        break
                    for key, _ in selector.select(timeout=min(0.1, remaining)):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if data:
                            log.write(data)
                            log.flush()
                            sys.stdout.write(data.decode("utf-8", errors="replace"))
                            sys.stdout.flush()
                        else:
                            selector.unregister(key.fileobj)
                    if child.poll() is not None and not selector.get_map():
                        break
            sample = sample_resources(child.pid, time.monotonic() - started, deadline=started + max_wall_seconds)
            samples.append(sample)
            resources.write(json.dumps(sample, sort_keys=True) + "\n")
    except Exception as exc:
        result.update(stop_reason="launcher_error", launcher_error=f"{type(exc).__name__}: {exc}")
    finally:
        if child is not None:
            try:
                terminate_group(child)
            except Exception as exc:
                result.update(stop_reason="cleanup_error", cleanup_error=f"{type(exc).__name__}: {exc}")
            result["returncode"] = child.returncode
            if child.stdout is not None:
                child.stdout.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        result["child_wall_time_seconds"] = time.monotonic() - started
        rss = [s["group_rss_kib_sum"] for s in samples if s.get("group_rss_kib_sum") is not None]
        result["resource_summary"] = {"sample_count": len(samples),
                                      "peak_group_rss_kib_sum": max(rss) if rss else None,
                                      "samples_with_errors": sum(bool(s["sampling_errors"]) for s in samples),
                                      "sampling_interval_seconds": SAMPLE_INTERVAL_SECONDS,
                                      "limitations": RESOURCE_LIMITATIONS}
    return result


def jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def signature_metadata(signature: dict) -> dict:
    if signature.get("schema_version") != 1 or set(signature) != {"schema_version", "parameters", "buffers", "sha256"}:
        raise ValueError("Invalid model-signature schema")
    body = {k: signature[k] for k in ("schema_version", "parameters", "buffers")}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if signature["sha256"] != digest or not signature["parameters"]:
        raise ValueError("Model-signature digest/parameters are invalid")
    metadata = {}
    for kind in ("parameters", "buffers"):
        for entry in signature[kind]:
            name = entry["name"]
            if name in metadata or not isinstance(name, str) or not name:
                raise ValueError("Invalid duplicate/empty tensor name")
            if (not isinstance(entry["shape"], list) or any(type(v) is not int or v < 0 for v in entry["shape"])
                    or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
                raise ValueError("Invalid tensor shape/hash")
            # Original fastreal attention tensors are float32 despite main's
            # float64 default. Preserve that mixed dtype contract without casts.
            if kind == "parameters" and entry["dtype"] not in {"torch.float32", "torch.float64"}:
                raise ValueError("Unexpected original HetNet parameter dtype")
            metadata[name] = {"kind": kind, "shape": entry["shape"], "dtype": entry["dtype"]}
    return metadata


def gate_model_contract(gate_path: Path, gate: dict, composition: str) -> dict:
    path = gate_path.parent / "cross_composition_signatures.json"
    if str(path.resolve()) not in gate.get("artifacts", []):
        raise ValueError("Gate A does not identify its original-model cross-composition artifact")
    data = read_json(path)
    record = data.get("compositions", {}).get(composition, {})
    if data.get("schema_version") != 1 or record.get("strict_load") is not True:
        raise ValueError("Gate A lacks the endpoint's strict original-model loading contract")
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "metadata": signature_metadata(record["signature"])}


def verify_completion(directory: Path, plan: dict, original_metadata: dict) -> dict:
    import torch
    from .signatures import model_signature

    rows = jsonl(directory / "metrics.jsonl")
    signatures = jsonl(directory / "epoch_signatures.jsonl")
    if [r.get("epoch") for r in rows] != list(range(1, 21)) or [r.get("epoch") for r in signatures] != list(range(1, 21)):
        raise ValueError("Calibration requires exactly contiguous epochs/signatures 1..20")
    total_steps = total_episodes = 0
    agent_count = plan["entry"]["nfriendly_P"] + plan["entry"]["nfriendly_A"]
    for row in rows:
        if type(row.get("steps")) is not int or not 20000 <= row["steps"] <= 23160:
            raise ValueError("Recorded steps violate the production four-collector epoch bounds")
        if type(row.get("episodes")) is not int or row["episodes"] < 1:
            raise ValueError("Recorded episode count must be positive")
        total_steps += row["steps"]
        total_episodes += row["episodes"]
        if row.get("total_steps") != total_steps or row.get("total_episodes") != total_episodes:
            raise ValueError("Cumulative fresh-step/episode accounting is inconsistent")
        checked_number(row.get("wall_time_seconds"), "epoch wall time")
        if not 0 <= row.get("success_rate", -1) <= 1 or not 1 <= row.get("steps_taken", -1) <= 80:
            raise ValueError("Recorded success/episode length is outside the PCP bounds")
        reward = row.get("reward_per_agent")
        if not isinstance(reward, list) or len(reward) != agent_count:
            raise ValueError("Reward vector does not match endpoint roster")
        numeric = reward + [row.get(k) for k in ("success_rate", "steps_taken", "policy_loss", "value_loss")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in numeric):
            raise ValueError("Non-finite or nonnumeric training diagnostics")
    initial = read_json(directory / "initial_signature.json")
    expected_metadata = signature_metadata(initial)
    if expected_metadata != original_metadata:
        raise ValueError("Initial model names/shapes/dtypes differ from the verified original Gate A model")
    for row in signatures:
        if signature_metadata(row["signature"]) != expected_metadata:
            raise ValueError("Parameter/buffer names, shapes or dtypes changed during calibration")
    checkpoints = list((directory / "checkpoints").rglob("model_ep20.pt"))
    if len(checkpoints) != 1:
        raise ValueError("Expected exactly one final epoch-20 checkpoint")
    checkpoint = checkpoints[0]
    sidecar = read_json(Path(str(checkpoint) + ".signature.json"))
    if sidecar != signatures[-1]["signature"]:
        raise ValueError("Final checkpoint sidecar differs from final epoch signature")
    # Only load the checkpoint just produced in this exclusive, newly created run.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("policy_net")
    if saved.get("seed") != 0 or not isinstance(state, dict) or set(state) != set(expected_metadata):
        raise ValueError("Checkpoint seed or state_dict tensor inventory differs")
    for name, tensor in state.items():
        meta = expected_metadata[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or list(tensor.shape) != meta["shape"] or str(tensor.dtype) != meta["dtype"]
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"Checkpoint tensor shape/dtype/device/value mismatch: {name}")

    class SavedTensors:
        def named_parameters(self):
            return [(name, value) for name, value in state.items() if expected_metadata[name]["kind"] == "parameters"]

        def named_buffers(self):
            return [(name, value) for name, value in state.items() if expected_metadata[name]["kind"] == "buffers"]

    if model_signature(SavedTensors()) != sidecar:
        raise ValueError("Checkpoint tensor values do not match the recorded signature")
    saves = jsonl(directory / "checkpoint_records.jsonl")
    if len(saves) != 1 or saves[0].get("epoch") != 20:
        raise ValueError("Calibration must have one final checkpoint-save record")
    save = saves[0]
    if (Path(save.get("path", "")).resolve() != checkpoint.resolve()
            or save.get("bytes") != checkpoint.stat().st_size or save.get("parameter_sha256") != sidecar["sha256"]):
        raise ValueError("Checkpoint save accounting does not match the actual final artifact")
    checked_number(save.get("wall_time_seconds"), "checkpoint-save duration", positive=False)
    resolved = read_json(directory / "resolved_args.json")
    expected = {**EXPECTED_RECIPE, "num_epochs": 20, "seed": 0,
                "nfriendly_P": plan["entry"]["nfriendly_P"], "nfriendly_A": plan["entry"]["nfriendly_A"],
                "use_binary": False, "use_cuda": False, "commnet": False, "hetcomm": False,
                "ic3net": False, "eval": False, "random": False, "load": "",
                "comm_range_P": -1, "comm_range_A": -1, "lossy_comm": False,
                "gamma": 1.0, "vision": 2, "A_vision": -1, "tensor_obs": False,
                "nenemies": 1, "moving_prey": False, "no_stay": False, "mode": "mixed",
                "enemy_comm": False, "second_reward_scheme": False,
                "resolved_model": {"class": "hetgat.uavnet.UAVNetA2CEasy", "per_class_critic": True, "with_two_state": True}}
    if any(resolved.get(k) != v for k, v in expected.items()):
        raise ValueError("Resolved arguments differ from the authorized CPU endpoint recipe")
    return {"final_checkpoint": str(checkpoint.resolve()), "final_checkpoint_sha256": file_sha256(checkpoint),
            "final_parameter_signature": sidecar, "initial_parameter_signature": initial,
            "resolved_args": resolved, "resolved_args_sha256": file_sha256(directory / "resolved_args.json"),
            "total_steps": total_steps, "total_episodes": total_episodes,
            "epoch_wall_time_seconds_sum": sum(row["wall_time_seconds"] for row in rows),
            "checkpoint_save_wall_time_seconds": save["wall_time_seconds"]}


def execute(plan: dict, gate_path: Path, max_wall_seconds: float) -> int:
    started = time.monotonic()
    checked_number(max_wall_seconds, "maximum child wall time")
    validate_native_host()
    expected_code, lock_hash = code_sha256(), file_sha256(ROOT / "uv.lock")
    gate = read_json(gate_path)
    validate_gate(gate, expected_code, lock_hash)
    model_contract = gate_model_contract(gate_path, gate, plan["entry"]["composition"])
    directory = Path(plan["directory"])
    with single_job_lock() as lock_path:
        if directory.exists():
            raise FileExistsError(f"Refusing existing local run directory: {directory}")
        hardware = hardware_inventory(directory)
        provenance = {"schema_version": 1, "status": "starting", "mode": "calibration",
                      "backend": "local_mac_cpu", "started_at_utc": utc_now(), "entry": plan["entry"],
                      "command": plan["command"], "expected_epochs": 20,
                      "git_commit": capture(["git", "rev-parse", "HEAD"])["stdout"],
                      "git_status": capture(["git", "status", "--porcelain"])["stdout"],
                      "code_sha256": expected_code, "code_inventory": code_inventory(), "uv_lock_sha256": lock_hash,
                      "gate_a_file": str(gate_path.resolve()), "gate_a_sha256": file_sha256(gate_path),
                      "original_model_contract": model_contract,
                      "hardware": hardware, "thread_settings": THREAD_ENV,
                      "dgl_default_dir": str(directory / "dgl"),
                      "max_child_wall_seconds": max_wall_seconds, "lock_path": lock_path,
                      "stdout_file": str(directory / "stdout.log"), "resource_limitations": RESOURCE_LIMITATIONS,
                      "resource_before": sample_resources(None, 0)}
        provenance["git_dirty"] = bool(provenance["git_status"])
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "config.json", plan)
        write_json(directory / "provenance.json", provenance)
        result = 1
        try:
            os.environ["DGLDEFAULTDIR"] = str(directory / "dgl")
            provenance["runtime"] = validate_runtime()
            validate_target_gate(gate, hardware, provenance["runtime"])
            if code_sha256() != expected_code or file_sha256(ROOT / "uv.lock") != lock_hash:
                raise ValueError("Source/lock changed during local preflight")
            child_env = dict(os.environ, **THREAD_ENV)
            for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
                child_env.pop(key, None)
            child = run_child(plan["command"], directory, max_wall_seconds, started, child_env)
            provenance.update(child)
            if child["stop_reason"] or child["returncode"] != 0:
                raise RuntimeError(f"Child failed: reason={child['stop_reason']}, returncode={child['returncode']}")
            completed = verify_completion(directory, plan, model_contract["metadata"])
            if completed["epoch_wall_time_seconds_sum"] + completed["checkpoint_save_wall_time_seconds"] > child["child_wall_time_seconds"]:
                raise ValueError("Child elapsed time is shorter than recorded epoch/save accounting")
            if code_sha256() != expected_code or file_sha256(ROOT / "uv.lock") != lock_hash:
                raise ValueError("Scientific source or dependency lock changed during the local pilot")
            if (file_sha256(gate_path) != provenance["gate_a_sha256"]
                    or file_sha256(model_contract["path"]) != model_contract["sha256"]):
                raise ValueError("Gate A evidence/model contract changed during the local pilot")
            provenance.update(completed, status="complete")
            result = 0
        except Exception as exc:
            provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"Local pilot failed; evidence preserved at {directory}: {exc}", file=sys.stderr)
        finally:
            provenance.update(resource_after=sample_resources(None, time.monotonic() - started),
                              ended_at_utc=utc_now(), launcher_wall_time_seconds=time.monotonic() - started)
            write_json(directory / "provenance.json", provenance)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gate-a", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True,
                        help="Positive child-process deadline; cleanup/validation is recorded separately")
    parser.add_argument("--dry-run", action="store_true", help="Print one command without imports, files, or execution")
    args = parser.parse_args(argv)
    try:
        checked_number(args.max_wall_seconds, "maximum child wall time")
        plan = calibration_plan(args.index, args.output_root)
        if args.dry_run:
            print(json.dumps({**plan, "gate_a": str(args.gate_a), "max_child_wall_seconds": args.max_wall_seconds,
                              "backend": "local_mac_cpu", "thread_settings": THREAD_ENV}, sort_keys=True))
            return 0
        return execute(plan, args.gate_a, args.max_wall_seconds)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Local preflight refused: {exc}\nNo training launch is authorized without valid evidence.\n")


if __name__ == "__main__":
    raise SystemExit(main())
