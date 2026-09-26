"""Scientific assertions: an intact upstream failure must fail Gate A."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def gradient_probe(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    configured = os.environ.get("HETNET_GRADIENT_PROBE_OUTPUT")
    output = Path(configured) if configured else tmp_path_factory.mktemp("gradient_probe") / "probe.json"
    command = [sys.executable, str(root / "tests/helpers/shared_gradient_probe.py"),
               "--output", str(output), "--updates", "3", "--timeout-seconds", "60"]
    with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        try:
            stdout, _ = process.communicate(timeout=85)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, _ = process.communicate(timeout=5)
            pytest.fail(f"Gradient probe timed out; worker process group terminated.\n{stdout}")
    assert output.is_file(), f"No gradient evidence emitted (exit {process.returncode}):\n{stdout}"
    data = json.loads(output.read_text())
    assert data["status"] != "ERROR", f"Probe execution error, not a passed gate:\n{stdout}"
    assert process.returncode in (0, 1), f"Unexpected probe exit: {process.returncode}\n{stdout}"
    return data, stdout, output


def test_workers_observe_current_shared_parameters(gradient_probe):
    data, stdout, output = gradient_probe
    assert data["shared_weights_passed"], f"Shared weights failed; {output}\n{stdout}"


def test_each_update_uses_fresh_aggregated_gradients(gradient_probe):
    data, stdout, output = gradient_probe
    assert data["fresh_gradient_aggregation_passed"], (
        f"Gate A FAIL: optimizer is not using the fresh normalized multi-process gradient; "
        f"{output}\n{stdout}")


def test_cached_gradient_storage_is_current(gradient_probe):
    data, stdout, output = gradient_probe
    assert data["gradient_storage_passed"], (
        f"Gate A FAIL: cached gradient tensors no longer back parameter.grad; {output}\n{stdout}")
