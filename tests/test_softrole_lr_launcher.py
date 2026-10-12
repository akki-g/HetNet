"""The Real-only S1 array changes rate and seed, with isolated fresh outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/softrole_lr_pcp.sh"
SLURM = ROOT / "slurm/softrole_lr_pcp.sbatch"


def launch(tmp_path, *args):
    return subprocess.run(["bash", str(LAUNCHER), *args], cwd=tmp_path, text=True,
                          capture_output=True, env=dict(os.environ, HETNET_PYTHON=sys.executable,
                          SOFTROLE_LR_PCP_RUN_ROOT=str(tmp_path / "runs")))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_only_rate_changes_for_matched_seed(tmp_path, seed):
    configs = []
    for index, rate in ((seed, .0003), (seed + 3, .001)):
        result = launch(tmp_path, str(index), "--dry-run")
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config.pop("lr") == rate
        configs.append(config)
    assert configs[0] == configs[1]
    config = configs[0]
    assert config["seed"] == seed and config["communication"] == "real"
    assert config["model"] == "shared" and config["optimizer"] == "adam"
    assert config["total_steps"] == 20_000_000 and config["nprocesses"] == 4
    assert config["batch_steps"] == 500 and config["save_every"] == 10
    assert config["comm_rounds"] == 3 and config["independent_heads"] and config["msg_dim"] == 64
    assert config["failure_prob"] == 0 and not config["held_out"]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("args", [(), ("6",), ("00",), ("-1",),
    ("0", "--communication", "binary"), ("0", "--lr", ".1"), ("0", "--resume", "x"),
    ("0", "--model", "banked"), ("0", "--seed", "9"), ("0", "--diagnostics-every", "0"),
    ("0", "--total-steps"), ("0", "--nprocesses", "0")])
def test_invalid_mapping_or_protocol_override_rejected(tmp_path, args):
    result = launch(tmp_path, *args)
    assert result.returncode == 2 and not result.stdout
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", range(6))
def test_slurm_mapping_without_scheduler(tmp_path, index):
    for name, content in {
        "module": "#!/bin/sh\nexit 0\n",
        "srun": '#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n',
    }.items():
        path = tmp_path / name
        path.write_text(content)
        path.chmod(0o755)
    result = subprocess.run(["bash", str(SLURM), "--dry-run"], text=True, capture_output=True,
        env=dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", HETNET_PYTHON=sys.executable,
                 SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_JOB_ID="12345", SLURM_ARRAY_TASK_ID=str(index),
                 SOFTROLE_LR_PCP_RUN_ROOT=str(tmp_path / "runs")))
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["seed"] == index % 3 and config["lr"] == (.0003 if index < 3 else .001)
    assert not (tmp_path / "runs").exists()


def test_existing_run_refused(tmp_path):
    run = tmp_path / "runs/pcp_shared/real/adam_lr3e-4/seed0"
    run.mkdir(parents=True)
    sentinel = run / "checkpoint.pt"
    sentinel.write_bytes(b"original")
    result = launch(tmp_path, "0", "--dry-run")
    assert result.returncode == 2 and "already exists" in result.stderr
    assert sentinel.read_bytes() == b"original"


def test_shell_syntax():
    result = subprocess.run(["bash", "-n", str(LAUNCHER), str(SLURM)], capture_output=True)
    assert result.returncode == 0, result.stderr
