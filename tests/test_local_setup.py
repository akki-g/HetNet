"""Mocked Mac setup contracts; never install dependencies or launch training."""
import json
import os
from pathlib import Path
import signal
import subprocess
from unittest.mock import Mock

import pytest

from hetnet_ext import gate_a, local_setup


HARDWARE = {"model_identifier": "Mac16,1", "memory_bytes": 34359738368,
            "physical_cpus": 10, "logical_cpus": 10, "cpu_brand": "Apple M4",
            "arm64_capable": 1}


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(local_setup, "ROOT", tmp_path)
    monkeypatch.setattr(gate_a, "ROOT", tmp_path)
    (tmp_path / "uv.lock").write_text("locked test fixture\n")
    (tmp_path / "pyproject.toml").write_text("project test fixture\n")
    monkeypatch.setattr(local_setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_setup.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(local_setup.platform, "platform", lambda: "mock-macOS-arm64")
    monkeypatch.setattr(local_setup.platform, "node", lambda: "target-mac")
    monkeypatch.setattr(local_setup.shutil, "which", lambda _command: "/mock/uv")
    mapping = dict(zip(("hw.model", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu",
                        "machdep.cpu.brand_string", "hw.optional.arm64"), HARDWARE.values()))

    def capture(command):
        stdout = "uv 0.12.5 (mock native arm64)\n" if command[-1] == "--version" else str(mapping[command[-1]]) + "\n"
        return {"command": command, "exit_code": 0, "stdout": stdout, "stderr": ""}

    monkeypatch.setattr(local_setup, "capture", capture)
    return tmp_path


def runtime_fixture(venv):
    files, hashes = {}, {}
    for name in local_setup.PINNED:
        path = venv / "lib" / "python3.12" / "site-packages" / name / "__init__.py"
        path.parent.mkdir(parents=True)
        path.write_text(f"# mocked {name}; not an installed package\n")
        files[name] = str(path)
        hashes[name] = local_setup.file_sha256(path)
    return {"python": "3.12.13 (mock)", "executable": str(venv / "bin" / "python"),
            "prefix": str(venv), "base_prefix": str(venv.parent / ".tools" / "python"),
            "interpreter_sha256": local_setup.file_sha256(venv / "bin" / "python"),
            "system": "Darwin", "machine": "arm64", "platform": "mock-macOS-arm64",
            "hostname": "target-mac", "hardware_identity": HARDWARE.copy(),
            "cpu_tensor_device": "cpu", "torch_num_threads": 1,
            "versions": local_setup.PINNED.copy(), "module_files": files, "module_sha256": hashes}


def install_mock_runner(monkeypatch, root, *, sync_code=0, import_code=0, mutate=None, marker=True):
    calls = []

    def runner(command, stdout, stderr, env, timeout):
        calls.append({"command": command, "environment": env.copy(), "timeout": timeout})
        venv = Path(env["UV_PROJECT_ENVIRONMENT"])
        if command[1] == "sync":
            venv.mkdir()
            (venv / "bin").mkdir()
            (venv / "bin" / "python").write_bytes(b"mock interpreter, never executed")
            text = f"Creating virtual environment at: {venv}\n" if marker else "Reused environment\n"
            stdout.write_text(text + ("mock sync failed\n" if sync_code else "mock sync complete\n"))
            return sync_code
        if import_code:
            stderr.write_text("mock native-library import failure\n")
            return import_code
        runtime = runtime_fixture(venv)
        if mutate:
            mutate(runtime)
        stdout.write_text(json.dumps(runtime) + "\n")
        stderr.write_text("mock upstream warning\n")
        return 0

    monkeypatch.setattr(local_setup, "run_logged", runner)
    return calls


def invoke(root):
    return local_setup.main(["--venv", ".venv-mac-pilot", "--output", "runs/mac_setup/example"])


def proof(root):
    return json.loads((root / "runs/mac_setup/example/evidence.json").read_text())


def test_success_matches_gate_schema_and_binds_cpu_hardware_paths_and_hashes(checkout, monkeypatch):
    task_original_home = os.environ.get("HOME")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/imports")
    monkeypatch.setenv("PYTHONHOME", "/old/python")
    monkeypatch.setenv("VIRTUAL_ENV", "/old/venv")
    calls = install_mock_runner(monkeypatch, checkout)
    assert invoke(checkout) == 0
    data = proof(checkout)
    assert data["passed"] and data["status"] == "PASS"
    assert data["command"] == ["/mock/uv", "sync", "--locked", "--python", "3.12"]
    assert data["hardware"]["values"] == HARDWARE
    assert data["hardware"]["raw_commands"]["memory_bytes"]["stdout"] == "34359738368\n"
    assert data["runtime"]["hardware_identity"] == HARDWARE
    assert data["runtime"]["hostname"] == "target-mac"
    assert len(calls) == 2
    for call in calls:
        env = call["environment"]
        assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & env.keys()
        assert env.get("HOME") == task_original_home
        assert env["UV_PROJECT_ENVIRONMENT"] == str(checkout / ".venv-mac-pilot")
        assert env["UV_PYTHON_INSTALL_DIR"] == str(checkout / ".tools/python")
        assert env["UV_PYTHON_DOWNLOADS"] == "automatic"
        assert env["UV_CACHE_DIR"] == str(checkout / "runs/mac_setup/example/uv_cache")
        assert env["DGLBACKEND"] == "pytorch"
        assert env["DGLDEFAULTDIR"] == str(checkout / "runs/mac_setup/example/dgl")
    checked = gate_a.verify_clean_environment(checkout / "runs/mac_setup/example/evidence.json")
    assert checked["status"] == "PASS"
    assert checked["runtime"] == data["runtime"]
    for name, digest in data["logs"].items():
        assert digest == local_setup.file_sha256(checkout / "runs/mac_setup/example" / name)


@pytest.mark.parametrize("existing", ["venv", "output", "broken_symlink"])
def test_existing_paths_are_refused_without_changing_them(checkout, monkeypatch, existing):
    runner = Mock()
    monkeypatch.setattr(local_setup, "run_logged", runner)
    venv, output = checkout / ".venv-mac-pilot", checkout / "runs/mac_setup/example"
    if existing == "broken_symlink":
        venv.symlink_to(checkout / "absent-target")
    else:
        target = venv if existing == "venv" else output
        target.mkdir(parents=True)
        (target / "keep.txt").write_text("preserve me")
    assert invoke(checkout) == 2
    runner.assert_not_called()
    if existing == "broken_symlink":
        assert venv.is_symlink() and not output.exists()
    else:
        assert (target / "keep.txt").read_text() == "preserve me"
        assert not (output / "evidence.json").exists()


@pytest.mark.parametrize("system,machine", [("Linux", "aarch64"), ("Darwin", "x86_64")])
def test_non_native_hosts_fail_before_sync_and_preserve_evidence(checkout, monkeypatch, system, machine):
    monkeypatch.setattr(local_setup.platform, "system", lambda: system)
    monkeypatch.setattr(local_setup.platform, "machine", lambda: machine)
    runner = Mock()
    monkeypatch.setattr(local_setup, "run_logged", runner)
    assert invoke(checkout) == 2
    data = proof(checkout)
    assert data["status"] == "ERROR" and not data["passed"]
    assert data["stage"] == "preflight" and data["sync_exit_code"] is None
    assert "native macOS arm64" in data["error"]
    assert set(local_setup.REQUIRED_LOGS) <= data["logs"].keys()
    runner.assert_not_called()


def test_wrong_uv_version_is_preserved_and_never_installed(checkout, monkeypatch):
    original = local_setup.capture
    def capture(command):
        result = original(command)
        if command[-1] == "--version":
            result["stdout"] = "uv 0.12.4\n"
        return result
    monkeypatch.setattr(local_setup, "capture", capture)
    runner = Mock()
    monkeypatch.setattr(local_setup, "run_logged", runner)
    assert invoke(checkout) == 2
    assert proof(checkout)["uv_version"] == "uv 0.12.4"
    assert "Require uv 0.12.5" in proof(checkout)["error"]
    assert not (checkout / ".venv-mac-pilot").exists()
    runner.assert_not_called()


@pytest.mark.parametrize("sync_code,import_code,stage", [(1, 0, "sync"), (0, 1, "imports")])
def test_failed_commands_preserve_logs_and_never_claim_pass(checkout, monkeypatch, sync_code, import_code, stage):
    calls = install_mock_runner(monkeypatch, checkout, sync_code=sync_code, import_code=import_code)
    assert invoke(checkout) == 2
    data = proof(checkout)
    assert not data["passed"] and data["status"] == "ERROR" and data["stage"] == stage
    assert data["sync_exit_code"] == sync_code
    assert data["imports_exit_code"] == (None if sync_code else import_code)
    assert len(calls) == (1 if sync_code else 2)
    assert (checkout / ".venv-mac-pilot").exists(), "Failed artifacts must not be deleted"
    with pytest.raises(ValueError, match="does not report"):
        gate_a.verify_clean_environment(checkout / "runs/mac_setup/example/evidence.json")


@pytest.mark.parametrize("mutation", ["wrong_version", "outside_origin", "wrong_host", "changed_file"])
def test_runtime_mismatches_are_not_clean_evidence(checkout, monkeypatch, mutation):
    def mutate(runtime):
        if mutation == "wrong_version":
            runtime["versions"]["torch"] = "2.3.0"
        elif mutation == "outside_origin":
            runtime["module_files"]["torch"] = str(checkout / "elsewhere/torch.py")
        elif mutation == "wrong_host":
            runtime["hostname"] = "different-mac"
        else:
            Path(runtime["module_files"]["torch"]).write_text("changed after hash")
    install_mock_runner(monkeypatch, checkout, mutate=mutate)
    assert invoke(checkout) == 2
    assert proof(checkout)["status"] == "ERROR"


def test_reused_environment_marker_does_not_pass(checkout, monkeypatch):
    calls = install_mock_runner(monkeypatch, checkout, marker=False)
    assert invoke(checkout) == 2
    assert len(calls) == 1
    assert "fresh environment" in proof(checkout)["error"]


def test_term_is_preserved_and_original_handler_restored(checkout, monkeypatch):
    before = signal.getsignal(signal.SIGTERM)
    def interrupted(*_args):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    monkeypatch.setattr(local_setup, "run_logged", interrupted)
    assert invoke(checkout) == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == before
    assert "signal 15" in proof(checkout)["error"]
    assert not proof(checkout)["passed"]


def test_timeout_kills_group_even_when_leader_exits_first(tmp_path, monkeypatch):
    process = Mock(pid=4321)
    process.wait.side_effect = [subprocess.TimeoutExpired(["mock"], 1), -15, -15]
    monkeypatch.setattr(local_setup.subprocess, "Popen", Mock(return_value=process))
    killed = []
    monkeypatch.setattr(local_setup.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        local_setup.run_logged(["mock"], tmp_path / "out.log", tmp_path / "err.log", {}, 1)
    assert killed == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
