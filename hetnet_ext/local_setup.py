"""Create fresh, checkout-local Mac dependency evidence for scientific Gate A.

python3 -m hetnet_ext.local_setup --venv .venv-mac-pilot \
    --output runs/mac_setup/<unique-id> [--uv /path/to/uv]

Uses only the standard library until the new environment's import check. It
does not install uv, change system Python or HOME, run training, or submit jobs.
Missing Python 3.12 may be provisioned by uv under this checkout's .tools/python.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.12.5"
PINNED = {"torch": "2.2.1", "dgl": "2.1.0", "gym": "0.26.2",
          "torchdata": "0.7.1", "numpy": "1.26.4", "pytest": "8.3.5"}
REQUIRED_LOGS = ("uv_sync.log", "imports.stdout.log", "imports.stderr.log")
RUNTIME_PROGRAM = """import hashlib, importlib, json, platform, subprocess, sys
from pathlib import Path
names = ('torch', 'dgl', 'gym', 'torchdata', 'numpy', 'pytest')
modules = {name: importlib.import_module(name) for name in names}
torch = modules['torch']
torch.set_num_threads(1)
hardware_keys = {'model_identifier': 'hw.model', 'memory_bytes': 'hw.memsize',
                 'physical_cpus': 'hw.physicalcpu', 'logical_cpus': 'hw.logicalcpu',
                 'cpu_brand': 'machdep.cpu.brand_string', 'arm64_capable': 'hw.optional.arm64'}
hardware = {name: subprocess.check_output(['/usr/sbin/sysctl', '-n', key], text=True).strip()
            for name, key in hardware_keys.items()}
for name in ('memory_bytes', 'physical_cpus', 'logical_cpus', 'arm64_capable'):
    hardware[name] = int(hardware[name])
value = {'python': sys.version, 'executable': sys.executable,
         'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
         'interpreter_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
         'platform': platform.platform(), 'machine': platform.machine(),
         'hostname': platform.node(), 'hardware_identity': hardware,
         'system': platform.system(), 'cpu_tensor_device': str(torch.ones(1, device='cpu').device),
         'torch_num_threads': torch.get_num_threads(),
         'versions': {name: module.__version__ for name, module in modules.items()},
         'module_files': {name: str(Path(module.__file__).resolve()) for name, module in modules.items()},
         'module_sha256': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                           for name, module in modules.items()}}
print(json.dumps(value, sort_keys=True))
"""


class SetupInterrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"setup interrupted by signal {signum}; artifacts preserved")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def new_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.exists() or path.is_symlink():
        raise ValueError(f"Refusing existing path: {path}")
    return path.resolve()


def capture(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            check=False, timeout=30)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def hardware_inventory() -> dict:
    """Preserve only requested hardware fields; no device serials or UUIDs."""
    fields = {"model_identifier": "hw.model", "memory_bytes": "hw.memsize",
              "physical_cpus": "hw.physicalcpu", "logical_cpus": "hw.logicalcpu",
              "cpu_brand": "machdep.cpu.brand_string", "arm64_capable": "hw.optional.arm64"}
    raw = {name: capture(["/usr/sbin/sysctl", "-n", key]) for name, key in fields.items()}
    values = {}
    for name, record in raw.items():
        if record["exit_code"] or not record["stdout"].strip():
            raise ValueError(f"Unable to read required Mac hardware field {name}: {record}")
        values[name] = record["stdout"].strip()
    for name in ("memory_bytes", "physical_cpus", "logical_cpus", "arm64_capable"):
        values[name] = int(values[name])
        if values[name] <= 0:
            raise ValueError(f"Invalid Mac hardware field {name}")
    if values["arm64_capable"] != 1:
        raise ValueError("The hardware does not report native arm64 capability")
    return {"platform": platform.platform(), "system": platform.system(), "hostname": platform.node(),
            "machine": platform.machine(), "values": values, "raw_commands": raw}


def run_logged(command: list[str], stdout: Path, stderr: Path | None, env: dict,
               timeout: float) -> int:
    """Terminate the whole dependency/import process group on timeout/interruption."""
    with stdout.open("a") as out:
        err = stderr.open("a") if stderr is not None else None
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out,
                                       stderr=err if err is not None else subprocess.STDOUT,
                                       start_new_session=True)
            try:
                return process.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                # The leader can exit before a grandchild that ignores TERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
                raise
        finally:
            if err is not None:
                err.close()


def validate_runtime(runtime: dict, venv: Path) -> None:
    if not runtime.get("python", "").startswith("3.12."):
        raise ValueError("Fresh environment did not use Python 3.12")
    if runtime.get("machine") != "arm64" or runtime.get("system") != "Darwin":
        raise ValueError("Fresh environment must run natively on macOS arm64")
    if runtime.get("executable") != str(venv / "bin" / "python"):
        raise ValueError("Imports ran with an unexpected Python executable")
    if runtime.get("prefix") != str(venv) or runtime.get("base_prefix") == str(venv):
        raise ValueError("Imports did not report the newly created virtual environment prefix")
    if runtime.get("cpu_tensor_device") != "cpu" or runtime.get("torch_num_threads") != 1:
        raise ValueError("CPU import probe did not preserve the one-thread CPU contract")
    for name, version in PINNED.items():
        if runtime.get("versions", {}).get(name, "").split("+")[0] != version:
            raise ValueError(f"Fresh environment has wrong/missing {name}; expected {version}")
        origin = Path(runtime.get("module_files", {}).get(name, "")).resolve()
        if venv not in origin.parents:
            raise ValueError(f"{name} was imported from outside the fresh environment: {origin}")
        if runtime.get("module_sha256", {}).get(name) != file_sha256(origin):
            raise ValueError(f"Imported {name} file hash does not match preserved runtime evidence")
    if runtime.get("interpreter_sha256") != file_sha256(venv / "bin" / "python"):
        raise ValueError("Imported interpreter hash does not match the fresh environment executable")


def setup(venv: Path, output: Path, requested_uv: str) -> int:
    # Validate both paths before creating anything. Existing evidence is immutable.
    venv, output = new_path(venv), new_path(output)
    if venv.parent != ROOT or venv == ROOT / ".venv":
        raise ValueError("Use a new checkout-local environment such as .venv-mac-pilot, not .venv")
    if output == venv or venv in output.parents:
        raise ValueError("Setup output cannot be inside the environment being created")
    output.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_LOGS:
        (output / name).touch(exist_ok=False)
    started = time.monotonic()
    evidence = {"schema_version": 1, "status": "ERROR", "passed": False,
                "created_at": datetime.now(timezone.utc).isoformat(), "cwd": str(ROOT),
                "venv": str(venv), "output": str(output), "requested_uv": requested_uv,
                "venv_absent_before_sync": True, "sync_exit_code": None,
                "imports_exit_code": None, "runtime": {}, "stage": "preflight"}
    overrides = {"UV_PROJECT_ENVIRONMENT": str(venv), "UV_CACHE_DIR": str(output / "uv_cache"),
                 "UV_PYTHON_INSTALL_DIR": str(ROOT / ".tools" / "python"),
                 "UV_PYTHON_DOWNLOADS": "automatic", "DGLBACKEND": "pytorch",
                 "DGLDEFAULTDIR": str(output / "dgl"),
                 "XDG_CACHE_HOME": str(output / "cache"),
                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "PYTHONDONTWRITEBYTECODE": "1"}
    evidence["environment_overrides"] = overrides
    cleared = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    environment = {key: value for key, value in os.environ.items() if key not in cleared}
    environment.update(overrides)
    evidence["cleared_environment_variables"] = list(cleared)
    def interrupted(signum, _frame):
        raise SetupInterrupted(signum)
    old_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        evidence["lock_sha256"] = file_sha256(ROOT / "uv.lock")
        evidence["pyproject_sha256"] = file_sha256(ROOT / "pyproject.toml")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("Local Mac setup requires native macOS arm64; Rosetta/x86 and other hosts are refused")
        hardware = hardware_inventory()
        write_json(output / "hardware.json", hardware)
        evidence["hardware"] = hardware
        executable = shutil.which(requested_uv)
        if not executable:
            raise ValueError(f"Pinned uv {UV_VERSION} is required but was not found: {requested_uv}; no uv installation attempted")
        version = capture([executable, "--version"])
        write_json(output / "uv_version.json", version)
        evidence["uv_version"] = version["stdout"].strip()
        if version["exit_code"] or evidence["uv_version"].split()[:2] != ["uv", UV_VERSION]:
            raise ValueError(f"Require uv {UV_VERSION}; received {evidence['uv_version']!r}")
        command = [executable, "sync", "--locked", "--python", "3.12"]
        evidence["command"] = command
        if venv.exists() or venv.is_symlink():
            raise ValueError("Environment appeared during preflight; refusing to sync into an existing environment")
        evidence["stage"] = "sync"
        print(f"Creating locked environment at {venv}; dependency logs: {output}", flush=True)
        evidence["sync_exit_code"] = run_logged(command, output / "uv_sync.log", None, environment, 1800)
        if evidence["sync_exit_code"]:
            raise RuntimeError(f"uv sync failed with exit code {evidence['sync_exit_code']}")
        if "Creating virtual environment at:" not in (output / "uv_sync.log").read_text():
            raise RuntimeError("uv did not report creating a fresh environment; refusing clean-install proof")
        evidence["stage"] = "imports"
        evidence["imports_command"] = [str(venv / "bin" / "python"), "-c", RUNTIME_PROGRAM]
        evidence["imports_exit_code"] = run_logged(evidence["imports_command"], output / "imports.stdout.log",
                                                    output / "imports.stderr.log", environment, 180)
        if evidence["imports_exit_code"]:
            raise RuntimeError(f"Pinned import check failed with exit code {evidence['imports_exit_code']}")
        evidence["runtime"] = json.loads((output / "imports.stdout.log").read_text())
        validate_runtime(evidence["runtime"], venv)
        if (evidence["runtime"].get("hardware_identity") != hardware["values"]
                or evidence["runtime"].get("hostname") != hardware["hostname"]):
            raise ValueError("Import runtime does not match the Mac hardware/hostname recorded during setup")
        for key, filename in (("lock_sha256", "uv.lock"), ("pyproject_sha256", "pyproject.toml")):
            if evidence[key] != file_sha256(ROOT / filename):
                raise RuntimeError(f"{filename} changed during setup; evidence is not valid for one locked environment")
        evidence.update(status="PASS", passed=True, stage="complete")
        result = 0
    except KeyboardInterrupt:
        evidence["error"] = "KeyboardInterrupt: setup interrupted; artifacts preserved"
        result = 130
    except SetupInterrupted as exc:
        evidence["error"] = f"SetupInterrupted: {exc}"
        result = 128 + exc.signum
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        result = 2
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        evidence["wall_time_seconds"] = time.monotonic() - started
        evidence["logs"] = {path.name: file_sha256(path) for path in
                            [output / name for name in REQUIRED_LOGS] +
                            [output / name for name in ("hardware.json", "uv_version.json") if (output / name).is_file()]}
        write_json(output / "evidence.json", evidence)
    print(f"Local setup {evidence['status']}: {output / 'evidence.json'}", flush=True)
    if not evidence["passed"]:
        print(evidence.get("error", "Setup did not pass"), file=sys.stderr)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venv", type=Path, required=True, help="New checkout-local path, e.g. .venv-mac-pilot")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory, e.g. runs/mac_setup/<id>")
    parser.add_argument("--uv", default="uv", help=f"Existing uv {UV_VERSION} executable (default: PATH)")
    args = parser.parse_args(argv)
    try:
        return setup(args.venv, args.output, args.uv)
    except (OSError, ValueError) as exc:
        print(f"Local setup refused: {exc}. Existing paths were not changed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
