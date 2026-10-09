"""Resolve the fixed historical PCP recipe without starting a training process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "softrole_adam_pcp.sh"


def launch(tmp_path, *args):
    env = dict(os.environ, HETNET_PYTHON=sys.executable,
               SOFTROLE_ADAM_PCP_RUN_ROOT=str(tmp_path / "adam"))
    return subprocess.run(["bash", str(LAUNCHER), *args], cwd=tmp_path,
                          env=env, text=True, capture_output=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adam_array_seed_dry_run_keeps_historical_recipe(tmp_path, seed):
    result = launch(tmp_path, str(seed), "--dry-run")
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["optimizer"] == "adam" and config["lr"] == .0001
    assert config["task"] == "pcp" and config["model"] == "shared" and config["seed"] == seed
    assert config["env_version"] == "corrected-observation-v1"
    assert (config["num_p"], config["num_a"], config["dim"], config["vision"], config["max_steps"]) == (2, 1, 5, 2, 80)
    assert (config["epochs"], config["updates_per_epoch"], config["nprocesses"], config["batch_steps"]) == (2000, 10, 4, 500)
    assert config["total_steps"] is None and config["save_every"] == 50
    assert (config["actor_coeff"], config["value_coeff"], config["gamma"], config["gae_lambda"]) == (50., 1., 1., .95)
    assert (config["heads"], config["head_dim"], config["msg_dim"], config["hidden_dim"]) == (4, 16, 16, 64)
    assert config["feedback"] and config["detach_gap"] == 5 and config["max_grad_norm"] == .75
    assert config["compositions"] == [] and config["held_out"] == [] and config["failure_prob"] == 0
    assert not (tmp_path / "adam").exists()


@pytest.mark.parametrize("args", [(), ("3",), ("00",), ("-1",), ("0", "--resume"),
                                  ("0", "--optimizer=adam"), ("0", "--lr=.001"),
                                  ("0", "--dry-run", "--dry-run")])
def test_launcher_rejects_protocol_override_before_python(tmp_path, args):
    result = launch(tmp_path, *args)
    assert result.returncode == 2
    assert not result.stdout and not (tmp_path / "adam").exists()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_slurm_wrapper_resolves_all_indices_without_scheduler(tmp_path, seed):
    # Harmless substitutes: module does nothing; srun executes only a dry run.
    module = tmp_path / "module"
    module.write_text("#!/bin/sh\nexit 0\n")
    module.chmod(0o755)
    srun = tmp_path / "srun"
    srun.write_text('#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n')
    srun.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", HETNET_PYTHON=sys.executable,
               SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_TASK_ID=str(seed), SLURM_ARRAY_JOB_ID="812345",
               SOFTROLE_ADAM_PCP_RUN_ROOT=str(tmp_path / "adam"))
    result = subprocess.run(["bash", str(ROOT / "slurm" / "softrole_adam_pcp.sbatch"), "--dry-run"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["seed"] == seed and config["optimizer"] == "adam"
    assert not (tmp_path / "adam").exists()
