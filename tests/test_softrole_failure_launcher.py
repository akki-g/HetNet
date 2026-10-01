"""The dedicated array cannot drift into nominal runs or unsupported domains."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "softrole_failure.sh"


def launch(tmp_path, *args, capture_command=False):
    interpreter = sys.executable
    if capture_command:
        stub = tmp_path / "capture-python"
        stub.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        stub.chmod(0o755)
        interpreter = str(stub)
    env = dict(os.environ, HETNET_PYTHON=interpreter,
               SOFTROLE_RUN_ROOT=str(tmp_path / "nominal"),
               SOFTROLE_FAILURE_RUN_ROOT=str(tmp_path / "failure"))
    return subprocess.run(["bash", str(LAUNCHER), *args], cwd=tmp_path,
                          env=env, text=True, capture_output=True)


@pytest.mark.parametrize("index,model,seed", [
    (0, "shared", 0), (1, "shared", 1), (2, "shared", 2),
    (3, "banked", 0), (4, "banked", 1), (5, "banked", 2),
])
def test_failure_index_selects_model_seed_and_isolated_output(tmp_path, index, model, seed):
    result = launch(tmp_path, str(index), capture_command=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "-u", "-m", "softrole", "train", "--task", "pcp", "--model", model,
        "--seed", str(seed), "--study", "failure", "--output",
        str(tmp_path / "failure" / f"pcp_{model}" / f"seed{seed}"),
    ]


def test_failure_launcher_real_dry_run_preserves_reference_protocol(tmp_path):
    result = launch(tmp_path, "3", "--dry-run", "--total-steps", "40", "--nprocesses", "1")
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["task"] == "pcp" and config["model"] == "banked" and config["seed"] == 0
    assert config["compositions"] == [[2, 1], [2, 2], [3, 1]]
    assert config["held_out"] == [[3, 2]]
    assert config["failure_prob"] == 0.5 and config["failure_window"] == [10, 30]
    assert (config["dim"], config["vision"], config["max_steps"]) == (5, 2, 80)
    assert config["total_steps"] == 40 and config["nprocesses"] == 1
    assert not (tmp_path / "nominal").exists() and not (tmp_path / "failure").exists()


@pytest.mark.parametrize("args", [
    ("6",), ("-1",), ("00",),
    ("0", "--task", "fc"), ("0", "--task=pp"), ("0", "--ta", "fc"),
    ("0", "--study", "fixed"), ("0", "--output", "other"),
    ("0", "--model", "banked"), ("0", "--seed", "7"),
    ("0", "--resume", "checkpoint.pt"), ("0", "--failure-prob", "0"),
    ("0", "--failure-window", "1", "3"), ("0", "--epochs"),
])
def test_failure_launcher_rejects_invalid_index_and_protocol_overrides(tmp_path, args):
    result = launch(tmp_path, *args, capture_command=True)
    assert result.returncode == 2
    assert not result.stdout  # The underlying training process was never called.
