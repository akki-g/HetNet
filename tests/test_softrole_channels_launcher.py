"""The six-run channel comparison changes only Binary versus Real and seed."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "softrole_channels_pcp.sh"
SLURM = ROOT / "slurm" / "softrole_channels_pcp.sbatch"


def launch(tmp_path, *args, capture_command=False, default_root=False):
    interpreter = sys.executable
    if capture_command:
        stub = tmp_path / "capture-python"
        stub.write_text(
            f"#!{sys.executable}\nimport json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'threads': {"
            "key: os.environ[key] for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', "
            "'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}))\n"
        )
        stub.chmod(0o755)
        interpreter = str(stub)
    env = dict(os.environ, HETNET_PYTHON=interpreter,
               SOFTROLE_RUN_ROOT=str(tmp_path / "unrelated"),
               SOFTROLE_CHANNELS_PCP_RUN_ROOT=str(tmp_path / "channels"),
               SLURM_ARRAY_JOB_ID="812345")
    if default_root:
        env.pop("SOFTROLE_CHANNELS_PCP_RUN_ROOT")
    return subprocess.run(["bash", str(LAUNCHER), *args], cwd=tmp_path,
                          env=env, text=True, capture_output=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_paired_channel_dry_runs_have_identical_other_settings(tmp_path, seed):
    configs = []
    for index, channel in ((seed, "binary"), (seed + 3, "real")):
        result = launch(tmp_path, str(index), "--dry-run")
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config.pop("communication") == channel
        configs.append(config)
    assert configs[0] == configs[1]
    config = configs[0]
    assert config["task"] == "pcp" and config["model"] == "shared" and config["seed"] == seed
    assert config["optimizer"] == "adam" and config["lr"] == .0001
    assert config["env_version"] == "corrected-observation-v1"
    assert (config["num_p"], config["num_a"], config["dim"], config["vision"], config["max_steps"]) == (2, 1, 5, 2, 80)
    assert (config["epochs"], config["updates_per_epoch"], config["nprocesses"], config["batch_steps"]) == (2000, 10, 4, 500)
    assert config["total_steps"] == 40_000_000 and config["save_every"] == 50
    assert (config["actor_coeff"], config["value_coeff"], config["gamma"], config["gae_lambda"]) == (50., 1., 1., .95)
    assert (config["heads"], config["head_dim"], config["msg_dim"], config["hidden_dim"]) == (4, 16, 64, 64)
    assert config["comm_rounds"] == 3 and config["independent_heads"]
    assert config["feedback"] and config["detach_gap"] == 5 and config["max_grad_norm"] == .75
    assert config["compositions"] == [] and config["held_out"] == [] and config["failure_prob"] == 0
    assert not (tmp_path / "channels").exists() and not (tmp_path / "unrelated").exists()


@pytest.mark.parametrize("index,channel,seed", [
    (0, "binary", 0), (1, "binary", 1), (2, "binary", 2),
    (3, "real", 0), (4, "real", 1), (5, "real", 2),
])
def test_mapping_output_isolation_and_thread_limits(tmp_path, index, channel, seed):
    result = launch(tmp_path, str(index), capture_command=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    args = record["args"]
    assert args[:4] == ["-u", "-m", "softrole", "train"]
    assert args[args.index("--communication") + 1] == channel
    assert args[args.index("--seed") + 1] == str(seed)
    assert args[args.index("--output") + 1] == str(tmp_path / "channels" / "pcp_shared" / channel / f"seed{seed}")
    assert set(record["threads"].values()) == {"1"}
    assert not (tmp_path / "channels").exists()


def test_default_root_is_job_specific(tmp_path):
    result = launch(tmp_path, "4", capture_command=True, default_root=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)["args"]
    assert args[args.index("--output") + 1] == "runs/softrole_channels_pcp_812345/pcp_shared/real/seed1"


def test_existing_run_is_rejected_before_starting_python(tmp_path):
    run = tmp_path / "channels" / "pcp_shared" / "binary" / "seed0"
    run.mkdir(parents=True)
    sentinel = run / "checkpoint.pt"
    sentinel.write_bytes(b"existing checkpoint")
    result = launch(tmp_path, "0", capture_command=True)
    assert result.returncode == 2 and "already exists" in result.stderr
    assert not result.stdout and sentinel.read_bytes() == b"existing checkpoint"


@pytest.mark.parametrize("mode", ["file", "stdout"])
def test_bounded_execution_overrides_preserve_protocol(tmp_path, mode):
    result = launch(tmp_path, "3", "--dry-run", "--epochs", "1", "--updates-per-epoch", "1",
                    "--batch-steps", "1", "--nprocesses", "1", "--total-steps", "10",
                    "--save-every", "1", "--episode-log", mode)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert (config["epochs"], config["updates_per_epoch"], config["batch_steps"], config["nprocesses"],
            config["total_steps"], config["save_every"]) == (1, 1, 1, 1, 10, 1)
    assert config["communication"] == "real" and config["comm_rounds"] == 3
    assert config["max_steps"] == 80 and config["model"] == "shared"
    assert "episode_log" not in config and not (tmp_path / "channels").exists()


@pytest.mark.parametrize("args", [
    (), ("6",), ("-1",), ("00",),
    ("0", "--task", "fc"), ("0", "--task=pp"), ("0", "--ta", "fc"),
    ("0", "--study", "failure"), ("0", "--output", "other"),
    ("0", "--model", "banked"), ("0", "--seed", "7"),
    ("0", "--resume", "checkpoint.pt"), ("0", "--failure-prob", "1"),
    ("0", "--comm-rounds", "2"), ("0", "--communication", "real"),
    ("0", "--heads", "3"), ("0", "--msg-dim", "16"), ("0", "--head-dim", "64"),
    ("0", "--optimizer", "rmsprop"), ("0", "--lr", ".001"),
    ("0", "--epochs"), ("0", "--epochs", "0"), ("0", "--nprocesses", "-1"),
    ("0", "--total-steps", "1.5"), ("0", "--episode-log", "discard"), ("0", "--episode-log"),
])
def test_invalid_index_and_scientific_overrides_never_start_python(tmp_path, args):
    result = launch(tmp_path, *args, capture_command=True)
    assert result.returncode == 2 and not result.stdout
    assert not (tmp_path / "channels").exists()


@pytest.mark.parametrize("index", range(6))
def test_slurm_wrapper_resolves_all_indices_without_scheduler(tmp_path, index):
    module = tmp_path / "module"
    module.write_text("#!/bin/sh\nexit 0\n")
    module.chmod(0o755)
    srun = tmp_path / "srun"
    srun.write_text('#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n')
    srun.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", HETNET_PYTHON=sys.executable,
               SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_TASK_ID=str(index), SLURM_ARRAY_JOB_ID="812345",
               SOFTROLE_CHANNELS_PCP_RUN_ROOT=str(tmp_path / "channels"))
    result = subprocess.run(["bash", str(SLURM), "--dry-run"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["seed"] == index % 3 and config["communication"] == ("binary" if index < 3 else "real")
    assert not (tmp_path / "channels").exists()


def test_launcher_shell_syntax():
    result = subprocess.run(["bash", "-n", str(LAUNCHER), str(SLURM)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
