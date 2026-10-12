"""D1-D6 launchers preserve the explicit manifest/entry mapping and fresh outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/softrole_diagnostics.sh"
SLURM = ROOT / "slurm/softrole_diagnostics.sbatch"


@pytest.fixture
def setup(tmp_path):
    # A stand-in interpreter verifies argv, not any diagnostics implementation.
    interpreter = tmp_path / "capture interpreter"
    interpreter.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "env": {
    key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "DGLBACKEND")}}))
""")
    interpreter.chmod(0o755)
    manifest = tmp_path / "locked manifest.json"
    manifest.write_text('{"entries": []}\n')
    output = tmp_path / "diagnostics output"
    env = dict(os.environ, HETNET_PYTHON=str(interpreter), OMP_NUM_THREADS="8")
    return manifest, output, env


def launch(tmp_path, env, args):
    return subprocess.run(["bash", str(LAUNCHER), *args], cwd=tmp_path,
                          text=True, capture_output=True, env=env)


def test_exact_arguments_thread_limits_and_noncreating_dry_run(tmp_path, setup):
    manifest, output, env = setup
    args = ["--manifest", str(manifest), "--index", "12", "--output", str(output), "--dry-run"]
    result = launch(tmp_path, env, args)
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    assert captured["argv"] == ["-u", "-m", "softrole.diagnostics", *args]
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"]["PYTHONPATH"].split(":")[:2] == [str(ROOT), str(ROOT / "envs")]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert captured["env"][key] == "1"
    assert captured["env"]["DGLBACKEND"] == "pytorch"
    assert not output.exists()


def test_relative_paths_preserve_caller_meaning(tmp_path, setup):
    _, _, env = setup
    args = ["--manifest", "locked manifest.json", "--index", "0", "--output", "relative output"]
    result = launch(tmp_path, env, args)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"] == ["-u", "-m", "softrole.diagnostics", *args]
    assert not (tmp_path / "relative output").exists()


@pytest.mark.parametrize("index", ["-1", "00", "1.0", "one"])
def test_bad_indices_rejected_before_runner(tmp_path, setup, index):
    manifest, output, env = setup
    result = launch(tmp_path, env, ["--manifest", str(manifest), "--index", index, "--output", str(output)])
    assert result.returncode == 2 and not result.stdout
    assert not output.exists()


@pytest.mark.parametrize("extra", [["--seed", "1"], ["--checkpoint", "other.pt"],
    ["--compositions", "3,2"], ["--resume", "old"], ["--index", "1"],
    ["--manifest", "other.json"], ["--output", "other"], ["--dry-run", "--dry-run"]])
def test_protocol_overrides_and_duplicates_rejected(tmp_path, setup, extra):
    manifest, output, env = setup
    args = ["--manifest", str(manifest), "--index", "0", "--output", str(output), *extra]
    result = launch(tmp_path, env, args)
    assert result.returncode == 2 and not result.stdout
    assert not output.exists()


@pytest.mark.parametrize("args", [[], ["--manifest"], ["--index"], ["--output"],
                                    ["--manifest", "missing.json", "--index", "0", "--output", "new"]])
def test_missing_inputs_rejected(tmp_path, setup, args):
    _, _, env = setup
    result = launch(tmp_path, env, args)
    assert result.returncode == 2 and not result.stdout


@pytest.mark.parametrize("kind", ["directory", "file", "broken_symlink"])
def test_existing_output_refused_without_mutation(tmp_path, setup, kind):
    manifest, output, env = setup
    if kind == "directory":
        output.mkdir()
        (output / "sentinel").write_text("original")
    elif kind == "file":
        output.write_text("original")
    else:
        output.symlink_to(tmp_path / "absent")
    result = launch(tmp_path, env, ["--manifest", str(manifest), "--index", "0", "--output", str(output), "--dry-run"])
    assert result.returncode == 2 and "already exists" in result.stderr and not result.stdout
    if kind == "directory":
        assert (output / "sentinel").read_text() == "original"
    elif kind == "file":
        assert output.read_text() == "original"
    else:
        assert output.is_symlink()


def slurm_env(tmp_path, setup):
    manifest, output, env = setup
    for name, content in {
        "module": "#!/bin/sh\nexit 0\n",
        "srun": '#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n',
    }.items():
        path = tmp_path / name
        path.write_text(content)
        path.chmod(0o755)
    return dict(env, PATH=f"{tmp_path}:{env['PATH']}", SLURM_SUBMIT_DIR=str(ROOT),
                SLURM_ARRAY_JOB_ID="12345", CAPCOM_DIAGNOSTICS_MANIFEST=str(manifest),
                CAPCOM_DIAGNOSTICS_OUTPUT_ROOT=str(output))


@pytest.mark.parametrize("index", [0, 1, 12])
def test_slurm_mapping_without_scheduler(tmp_path, setup, index):
    manifest, output, _ = setup
    env = dict(slurm_env(tmp_path, setup), SLURM_ARRAY_TASK_ID=str(index))
    result = subprocess.run(["bash", str(SLURM), "--dry-run"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"] == ["-u", "-m", "softrole.diagnostics", "--manifest",
        str(manifest), "--index", str(index), "--output", str(output / f"task_{index}"), "--dry-run"]
    assert not output.exists()


def test_slurm_default_is_job_specific(tmp_path, setup):
    env = dict(slurm_env(tmp_path, setup), SLURM_ARRAY_TASK_ID="0")
    env.pop("CAPCOM_DIAGNOSTICS_OUTPUT_ROOT")
    result = subprocess.run(["bash", str(SLURM), "--dry-run"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"][-2] == "runs/capcom_diagnostics_12345/task_0"
    assert not (ROOT / "runs/capcom_diagnostics_12345").exists()


@pytest.mark.parametrize("extra", [["--index", "1"], ["--output", "elsewhere"], ["--dry-run", "--dry-run"]])
def test_slurm_rejects_overrides(tmp_path, setup, extra):
    env = dict(slurm_env(tmp_path, setup), SLURM_ARRAY_TASK_ID="0")
    result = subprocess.run(["bash", str(SLURM), *extra], env=env, text=True, capture_output=True)
    assert result.returncode == 2 and not result.stdout


def test_shell_syntax():
    result = subprocess.run(["bash", "-n", str(LAUNCHER), str(SLURM)], capture_output=True)
    assert result.returncode == 0, result.stderr
