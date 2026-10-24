"""Slurm-only pilot preparation and launch, without a scheduler or training."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/softrole_graded_pilot.sbatch"
PREPARE = ROOT / "slurm/softrole_graded_pilot_prepare.sbatch"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
DEFAULT_MANIFEST = "runs/graded_sensing_pilot_20261024/manifest.json"
DEFAULT_RUN_ROOT = "runs/graded_sensing_pilot_20261024/training"


def capture_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"names = {(*THREADS, 'PYTHONPATH', 'PYTHONUNBUFFERED', 'DGLBACKEND')!r}\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['PYTHON_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': args, 'cwd': os.getcwd(), 'python': sys.argv[0], "
        "'environment': {name: os.environ.get(name) for name in names}}) + '\\n')\n"
        "assert args[:3] == ['-u', '-m', 'softrole.graded_pilot']\n"
        "operation = args[3]\n"
        "key = operation if operation == 'prepare' else 'launch:' + args[args.index('--index') + 1]\n"
        "if os.environ.get('MOCK_FAIL') == key: sys.exit(17)\n"
        "if operation == 'prepare':\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with output.open('x') as stream: stream.write('{\"mock_manifest\": true}\\n')\n"
        "else:\n"
        "    assert pathlib.Path(args[args.index('--manifest') + 1]).is_file()\n"
    )
    path.chmod(0o755)
    return path


def setup_environment(tmp_path, *, index="0", existing=False, defaults=False):
    root = tmp_path / "checkout"
    root.mkdir()
    tools = tmp_path / "mock-tools"
    tools.mkdir()
    for name, code in {
        "module": '#!/bin/bash\nprintf "%s\\n" "$*" > "$MODULE_CAPTURE"\n',
        "srun": '#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n',
    }.items():
        file = tools / name
        file.write_text(code)
        file.chmod(0o755)
    env = dict(os.environ)
    for name in ("SOFTROLE_GRADED_PILOT_MANIFEST", "SOFTROLE_GRADED_PILOT_RUN_ROOT", "HETNET_PYTHON", "MOCK_FAIL"):
        env.pop(name, None)
    env.update({name: "32" for name in THREADS})
    env.update(PATH=f"{tools}:{env['PATH']}", PYTHONPATH="inherited-path", PYTHONUNBUFFERED="0", DGLBACKEND="other",
               MODULE_CAPTURE=str(tmp_path / "module.txt"), PYTHON_CAPTURE=str(tmp_path / "python.jsonl"),
               SLURM_SUBMIT_DIR=str(root), SLURM_ARRAY_TASK_ID=index, SLURM_ARRAY_JOB_ID="4242")
    if defaults:
        capture_python(root / ".venv/bin/python")
        manifest, run_root = root / DEFAULT_MANIFEST, root / DEFAULT_RUN_ROOT
    else:
        env["HETNET_PYTHON"] = str(capture_python(tmp_path / "capture python"))
        manifest, run_root = tmp_path / "locked manifest.json", tmp_path / "training runs"
        env.update(SOFTROLE_GRADED_PILOT_MANIFEST=str(manifest), SOFTROLE_GRADED_PILOT_RUN_ROOT=str(run_root))
    if existing:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"existing": "preserve these bytes"}\n')
    return root, manifest, run_root, env


def run_slurm(script, arguments, tmp_path, env):
    return subprocess.run(["bash", str(script), *arguments], cwd=tmp_path, env=env,
                          text=True, capture_output=True)


def calls(env):
    path = Path(env["PYTHON_CAPTURE"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_environment(call, root):
    assert call["cwd"] == str(root)
    assert all(call["environment"][name] == "1" for name in THREADS)
    assert call["environment"]["PYTHONPATH"] == f"{root}:{root}/envs:inherited-path"
    assert call["environment"]["PYTHONUNBUFFERED"] == "1"
    assert call["environment"]["DGLBACKEND"] == "pytorch"


def launch_args(manifest, index, run_root, *, dry_run=False):
    result = ["-u", "-m", "softrole.graded_pilot", "launch", "--manifest", str(manifest),
              "--index", str(index), "--run-root", str(run_root)]
    return result + (["--dry-run"] if dry_run else [])


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("dry_run", [False, True])
def test_training_slurm_invokes_python_directly_with_locked_index(tmp_path, index, dry_run):
    root, manifest, run_root, env = setup_environment(tmp_path, index=str(index), existing=True)
    result = run_slurm(SLURM, ["--dry-run"] if dry_run else [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)
    assert len(observed) == 1
    assert observed[0]["args"] == launch_args(manifest, index, run_root, dry_run=dry_run)
    assert_environment(observed[0], root)
    assert Path(env["MODULE_CAPTURE"]).read_text() == "load anaconda/anaconda-2024.10\n"
    assert not run_root.exists()


def test_training_slurm_default_paths_and_interpreter(tmp_path):
    root, manifest, run_root, env = setup_environment(tmp_path, existing=True, defaults=True)
    result = run_slurm(SLURM, ["--dry-run"], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)[0]
    assert observed["args"] == launch_args(DEFAULT_MANIFEST, 0, DEFAULT_RUN_ROOT, dry_run=True)
    assert observed["python"] == str(root / ".venv/bin/python")
    assert_environment(observed, root)


@pytest.mark.parametrize("index", ["2", "-1", "00", "01", "x"])
def test_training_slurm_rejects_invalid_array_indices_before_execution(tmp_path, index):
    _, _, _, env = setup_environment(tmp_path, index=index, existing=True)
    result = run_slurm(SLURM, [], tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


@pytest.mark.parametrize("args", [
    ["--dry"], ["--dry-run=true"], ["--dry-run", "--dry-run"], ["--resume", "checkpoint.pt"],
    ["--total-steps", "1"], ["--epochs", "1"], ["--nprocesses", "1"], ["--max-steps", "4"],
    ["--seed", "1"], ["--descriptor", "type"], ["--manifest", "other.json"],
    ["--run-root", "other"], ["--sensing-a", "2"], ["--episode-log", "stdout"],
])
def test_training_slurm_rejects_all_config_and_runtime_overrides_before_execution(tmp_path, args):
    _, _, _, env = setup_environment(tmp_path, existing=True)
    result = run_slurm(SLURM, args, tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


def test_training_slurm_requires_prepared_manifest(tmp_path):
    _, _, _, env = setup_environment(tmp_path)
    result = run_slurm(SLURM, ["--dry-run"], tmp_path, env)
    assert result.returncode == 2 and "not found" in result.stderr and not calls(env)


@pytest.mark.parametrize("defaults", [False, True])
def test_preparation_creates_manifest_then_previews_both_arms_without_training(tmp_path, defaults):
    root, manifest, run_root, env = setup_environment(tmp_path, defaults=defaults)
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    manifest_arg = DEFAULT_MANIFEST if defaults else manifest
    root_arg = DEFAULT_RUN_ROOT if defaults else run_root
    expected = [["-u", "-m", "softrole.graded_pilot", "prepare", "--output", str(manifest_arg)],
                launch_args(manifest_arg, 0, root_arg, dry_run=True),
                launch_args(manifest_arg, 1, root_arg, dry_run=True)]
    observed = calls(env)
    assert [call["args"] for call in observed] == expected
    for call in observed:
        assert_environment(call, root)
    assert manifest.is_file() and not run_root.exists()


def test_existing_manifest_is_validated_by_both_previews_and_never_rewritten(tmp_path):
    _, manifest, run_root, env = setup_environment(tmp_path, existing=True)
    before = manifest.read_bytes()
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert [call["args"] for call in calls(env)] == [launch_args(manifest, i, run_root, dry_run=True) for i in (0, 1)]
    assert manifest.read_bytes() == before and not run_root.exists()


@pytest.mark.parametrize("failure, expected_count", [("prepare", 1), ("launch:0", 2), ("launch:1", 3)])
def test_preparation_stops_on_first_failed_step(tmp_path, failure, expected_count):
    _, _, run_root, env = setup_environment(tmp_path)
    env["MOCK_FAIL"] = failure
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 17 and len(calls(env)) == expected_count
    assert not run_root.exists()


def test_existing_invalid_manifest_fails_validation_without_regeneration(tmp_path):
    _, manifest, run_root, env = setup_environment(tmp_path, existing=True)
    before = manifest.read_bytes()
    env["MOCK_FAIL"] = "launch:0"
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 17
    assert [call["args"] for call in calls(env)] == [launch_args(manifest, 0, run_root, dry_run=True)]
    assert manifest.read_bytes() == before and not run_root.exists()


@pytest.mark.parametrize("args", [["--dry-run"], ["--output", "other.json"], ["--total-steps", "1"]])
def test_preparation_rejects_arguments_before_execution(tmp_path, args):
    _, _, _, env = setup_environment(tmp_path)
    result = run_slurm(PREPARE, args, tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


def test_slurm_shell_syntax_and_declared_resources():
    result = subprocess.run(["bash", "-n", str(SLURM), str(PREPARE)], capture_output=True)
    assert result.returncode == 0, result.stderr
    for path, extra in ((SLURM, ("--cpus-per-task=4", "--mem=16G", "--time=06:00:00", "--array=0-1%2")),
                        (PREPARE, ("--cpus-per-task=1", "--mem=2G", "--time=00:10:00"))):
        text = path.read_text()
        for setting in ("--account=cenyioha", "--partition=normal", "--constraint=sapphirerapids", *extra):
            assert f"#SBATCH {setting}\n" in text
        assert "scripts/softrole_graded_pilot.sh" not in text
