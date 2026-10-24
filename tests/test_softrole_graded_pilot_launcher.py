"""The exploratory pilot shell route cannot override its locked experiment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/softrole_graded_pilot.sh"
SLURM = ROOT / "slurm/softrole_graded_pilot.sbatch"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def capture_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"names = {(*THREADS, 'PYTHONPATH', 'PYTHONUNBUFFERED', 'DGLBACKEND')!r}\n"
        "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'environment': {name: os.environ.get(name) for name in names}}))\n"
    )
    path.chmod(0o755)
    return path


def environment(tmp_path, *, manifest=True, python=True):
    env = dict(os.environ)
    for name in ("SOFTROLE_GRADED_PILOT_MANIFEST", "SOFTROLE_GRADED_PILOT_RUN_ROOT", "HETNET_PYTHON"):
        env.pop(name, None)
    env.update({name: "32" for name in THREADS})
    env.update(PYTHONPATH="inherited-path", PYTHONUNBUFFERED="0", DGLBACKEND="other")
    if manifest:
        file = tmp_path / "locked manifest.json"
        file.write_text("{}\n")
        env["SOFTROLE_GRADED_PILOT_MANIFEST"] = str(file)
    if python:
        env["HETNET_PYTHON"] = str(capture_python(tmp_path / "capture python"))
    return env


def run_shell(script, args, tmp_path, env):
    return subprocess.run(["bash", str(script), *args], cwd=tmp_path, env=env,
                          text=True, capture_output=True)


def assert_execution(result, root, manifest, index, run_root, options):
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    assert captured["args"] == ["-u", "-m", "softrole.graded_pilot", "launch",
                                 "--manifest", str(manifest), "--index", str(index),
                                 "--run-root", str(run_root), *options]
    assert captured["cwd"] == str(root)
    assert all(captured["environment"][name] == "1" for name in THREADS)
    assert captured["environment"]["PYTHONPATH"] == f"{root}:{root}/envs:inherited-path"
    assert captured["environment"]["PYTHONUNBUFFERED"] == "1"
    assert captured["environment"]["DGLBACKEND"] == "pytorch"


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("options", [[], ["--dry-run"]])
def test_direct_shell_routes_only_manifest_index_and_optional_preview(tmp_path, index, options):
    env = environment(tmp_path)
    env["SOFTROLE_GRADED_PILOT_RUN_ROOT"] = str(tmp_path / "fresh pilot runs")
    # The manifest path is resolved relative to the caller before cd.
    env["SOFTROLE_GRADED_PILOT_MANIFEST"] = "locked manifest.json"
    result = run_shell(LAUNCHER, [str(index), *options], tmp_path, env)
    assert_execution(result, ROOT, tmp_path / "locked manifest.json", index,
                     tmp_path / "fresh pilot runs", options)
    assert not (tmp_path / "fresh pilot runs").exists()


def test_direct_shell_uses_default_output_root(tmp_path):
    env = environment(tmp_path)
    result = run_shell(LAUNCHER, ["0", "--dry-run"], tmp_path, env)
    assert_execution(result, ROOT, tmp_path / "locked manifest.json", 0,
                     "runs/graded_sensing_pilot_20261024/training", ["--dry-run"])


@pytest.mark.parametrize("args", [
    [], ["2"], ["-1"], ["00"], ["01"], ["x"],
    ["0", "--dry"], ["0", "--dry-run=true"], ["0", "--resume", "checkpoint.pt"],
    ["0", "--total-steps", "1"], ["0", "--epochs", "1"], ["0", "--nprocesses", "1"],
    ["0", "--max-steps", "4"], ["0", "--seed", "1"], ["0", "--descriptor", "type"],
    ["0", "--manifest", "other.json"], ["0", "--run-root", "other"],
    ["0", "--sensing-a", "2"], ["0", "--held-out-sensing-p", "2"],
    ["0", "--episode-log", "stdout"],
])
def test_direct_shell_rejects_invalid_indices_and_all_configuration_overrides(tmp_path, args):
    result = run_shell(LAUNCHER, args, tmp_path, environment(tmp_path))
    assert result.returncode == 2 and not result.stdout


@pytest.mark.parametrize("missing", ["unset", "absent-file"])
def test_direct_shell_requires_an_existing_manifest(tmp_path, missing):
    env = environment(tmp_path, manifest=False)
    if missing == "absent-file":
        env["SOFTROLE_GRADED_PILOT_MANIFEST"] = str(tmp_path / "missing.json")
    result = run_shell(LAUNCHER, ["0", "--dry-run"], tmp_path, env)
    assert result.returncode == 2 and not result.stdout
    assert ("SOFTROLE_GRADED_PILOT_MANIFEST" if missing == "unset" else "not found") in result.stderr


def fake_checkout(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(LAUNCHER, root / "scripts" / LAUNCHER.name)
    capture_python(root / ".venv/bin/python")
    return root


def slurm_environment(tmp_path, index, *, defaults=False):
    root = fake_checkout(tmp_path)
    env = environment(tmp_path, manifest=not defaults, python=not defaults)
    if defaults:
        manifest = root / "runs/graded_sensing_pilot_20261024/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n")
    tools = tmp_path / "mock-tools"
    tools.mkdir()
    for name, code in {
        "module": '#!/bin/bash\nprintf "%s\\n" "$*" > "$MODULE_CAPTURE"\n',
        "srun": '#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n',
    }.items():
        path = tools / name
        path.write_text(code)
        path.chmod(0o755)
    env.update(PATH=f"{tools}:{env['PATH']}", MODULE_CAPTURE=str(tmp_path / "module.txt"),
               SLURM_SUBMIT_DIR=str(root), SLURM_ARRAY_TASK_ID=str(index), SLURM_ARRAY_JOB_ID="4242")
    return root, env


@pytest.mark.parametrize("index", [0, 1])
def test_mocked_slurm_preserves_mapping_binding_and_thread_limits(tmp_path, index):
    root, env = slurm_environment(tmp_path, index)
    env["SOFTROLE_GRADED_PILOT_RUN_ROOT"] = str(tmp_path / "cluster runs")
    result = run_shell(SLURM, ["--dry-run"], tmp_path, env)
    assert_execution(result, root, tmp_path / "locked manifest.json", index,
                     tmp_path / "cluster runs", ["--dry-run"])
    assert (tmp_path / "module.txt").read_text() == "load anaconda/anaconda-2024.10\n"


def test_mocked_slurm_defaults_manifest_python_and_output_paths(tmp_path):
    root, env = slurm_environment(tmp_path, 1, defaults=True)
    result = run_shell(SLURM, [], tmp_path, env)
    assert_execution(result, root, root / "runs/graded_sensing_pilot_20261024/manifest.json", 1,
                     "runs/graded_sensing_pilot_20261024/training", [])


@pytest.mark.parametrize("index", ["2", "-1", "01", "x"])
def test_mocked_slurm_rejects_indices_outside_its_two_arms(tmp_path, index):
    _, env = slurm_environment(tmp_path, index)
    result = run_shell(SLURM, ["--dry-run"], tmp_path, env)
    assert result.returncode == 2 and not result.stdout
    assert not (tmp_path / "module.txt").exists()


def test_mocked_slurm_refuses_runtime_override_before_invoking_python(tmp_path):
    _, env = slurm_environment(tmp_path, 0)
    result = run_shell(SLURM, ["--total-steps", "40"], tmp_path, env)
    assert result.returncode == 2 and not result.stdout


def test_shell_syntax_and_declared_stokes_resources():
    result = subprocess.run(["bash", "-n", str(LAUNCHER), str(SLURM)], capture_output=True)
    assert result.returncode == 0, result.stderr
    script = SLURM.read_text()
    for setting in ("--account=cenyioha", "--partition=normal", "--constraint=sapphirerapids",
                    "--cpus-per-task=4", "--mem=16G", "--time=06:00:00", "--array=0-1%2"):
        assert f"#SBATCH {setting}\n" in script
