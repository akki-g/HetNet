"""Actual HetNet scientific gradient gate; known failure is not an xfail."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def actual_probe(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    configured = os.environ.get("HETNET_ACTUAL_GRADIENT_PROBE_OUTPUT")
    output = Path(configured) if configured else tmp_path_factory.mktemp("actual_gradient") / "probe.json"
    command = [sys.executable, str(root / "tests/helpers/hetnet_gradient_probe.py"),
               "--output", str(output), "--updates", "3", "--batch-size", "4", "--horizon", "4"]
    with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        try:
            stdout, _ = process.communicate(timeout=145)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, _ = process.communicate(timeout=5)
            pytest.fail(f"Actual HetNet probe timed out; process group terminated.\n{stdout}")
    assert output.is_file(), f"No actual HetNet evidence; exit={process.returncode}\n{stdout}"
    result = json.loads(output.read_text())
    assert result["status"] != "ERROR", f"Probe execution error; not a passed gate:\n{stdout}"
    assert process.returncode in (0, 1), stdout
    return result, stdout, output


def test_actual_hetnet_shared_weights_after_every_step(actual_probe):
    result, stdout, output = actual_probe
    assert result["shared_weights_passed"], f"Shared weights failed: {output}\n{stdout}"
    assert all(row["parameter_update_observed"] for row in result["updates"]), (
        f"Shared parameter equality is insufficient if no update occurred: {output}\n{stdout}")


def test_actual_hetnet_current_named_gradients_are_fresh_sum(actual_probe):
    result, stdout, output = actual_probe
    assert result["fresh_gradient_aggregation_passed"], (
        f"Gate A FAIL in actual HetNet: optimizer gradient differs from fresh process sum; {output}\n{stdout}")


def test_actual_hetnet_cached_storage_remains_current(actual_probe):
    result, stdout, output = actual_probe
    assert result["gradient_storage_passed"], f"Gate A FAIL: stale actual HetNet gradients; {output}\n{stdout}"
