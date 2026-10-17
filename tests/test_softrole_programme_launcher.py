"""Programme launchers: explicit manifest/index mapping, runtime-only options, fresh outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from softrole.programme import entries, prepare
from test_softrole_programme import synthetic_selection

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/softrole_programme.sh"
SLURM = ROOT / "slurm/softrole_programme.sbatch"


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("programme")
    _, selection = synthetic_selection(root)
    prepare(selection, root / "locked manifest.json", bounded=True)
    return root / "locked manifest.json"


def launch(tmp_path, manifest, *args, env_extra=None):
    env = dict(os.environ, HETNET_PYTHON=sys.executable, SOFTROLE_PROGRAMME_RUN_ROOT=str(tmp_path / "runs"),
               SOFTROLE_PROGRAMME_MANIFEST=str(manifest))
    env.update(env_extra or {})
    return subprocess.run(["bash", str(LAUNCHER), *args], cwd=tmp_path, text=True, capture_output=True, env=env)


@pytest.mark.parametrize("index", [0, 30, 62])
def test_dry_run_resolves_the_manifest_entry(tmp_path, manifest, index):
    result = launch(tmp_path, manifest, str(index), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == entries(3e-4)[index]["config"]
    assert not (tmp_path / "runs").exists()


def test_runtime_options_change_only_runtime_fields(tmp_path, manifest):
    result = launch(tmp_path, manifest, "7", "--dry-run", "--epochs", "1", "--total-steps", "9", "--diagnostics-every", "1",
                    "--episode-log", "chunked-gzip")
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    expected = entries(3e-4)[7]["config"]
    assert {k for k in expected if expected[k] != config[k]} == {"epochs", "total_steps"}
    assert config["epochs"] == 1 and config["total_steps"] == 9


@pytest.mark.parametrize("args", [(), ("63",), ("00",), ("-1",), ("a",), ("0", "--lr", ".1"), ("0", "--communication", "binary"),
                                  ("0", "--seed", "4"), ("0", "--epochs", "0"), ("0", "--total-steps"), ("0", "--episode-log", "x"),
                                  ("0", "--resume", "x"), ("0", "--episode-log", "stdout")])
def test_invalid_index_or_scientific_override_rejected(tmp_path, manifest, args):
    result = launch(tmp_path, manifest, *args)
    assert result.returncode == 2 and not result.stdout
    assert not (tmp_path / "runs").exists()


def test_missing_manifest_rejected(tmp_path, manifest):
    env = dict(os.environ, HETNET_PYTHON=sys.executable, SOFTROLE_PROGRAMME_RUN_ROOT=str(tmp_path / "runs"))
    env.pop("SOFTROLE_PROGRAMME_MANIFEST", None)
    result = subprocess.run(["bash", str(LAUNCHER), "0", "--dry-run"], cwd=tmp_path, text=True, capture_output=True, env=env)
    assert result.returncode == 2 and "SOFTROLE_PROGRAMME_MANIFEST" in result.stderr
    result = launch(tmp_path, tmp_path / "absent.json", "0", "--dry-run")
    assert result.returncode == 2 and "not found" in result.stderr


def test_existing_run_or_record_refused(tmp_path, manifest):
    run = tmp_path / "runs" / entries(3e-4)[0]["relative_output"]
    run.mkdir(parents=True)
    sentinel = run / "checkpoint.pt"
    sentinel.write_bytes(b"original")
    result = launch(tmp_path, manifest, "0", "--epochs", "1", "--updates-per-epoch", "1", "--batch-steps", "1",
                    "--nprocesses", "1", "--total-steps", "1", "--save-every", "1")
    assert result.returncode == 2 and "refusing to overwrite" in result.stderr
    assert sentinel.read_bytes() == b"original"
    assert not (run.parent / (run.name + ".programme.json")).exists()


def slurm_env(tmp_path, manifest, index):
    for name, content in {
        "module": "#!/bin/sh\nexit 0\n",
        "srun": '#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n',
    }.items():
        path = tmp_path / name
        path.write_text(content)
        path.chmod(0o755)
    return dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", HETNET_PYTHON=sys.executable,
                SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_JOB_ID="12345", SLURM_ARRAY_TASK_ID=str(index),
                SOFTROLE_PROGRAMME_MANIFEST=str(manifest), SOFTROLE_PROGRAMME_RUN_ROOT=str(tmp_path / "runs"))


@pytest.mark.parametrize("index", [0, 31, 62])
def test_slurm_mapping_without_scheduler(tmp_path, manifest, index):
    result = subprocess.run(["bash", str(SLURM), "--dry-run"], env=slurm_env(tmp_path, manifest, index), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == entries(3e-4)[index]["config"]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("index", ["63", "-1", "x"])
def test_slurm_rejects_indices_outside_the_array(tmp_path, manifest, index):
    result = subprocess.run(["bash", str(SLURM), "--dry-run"], env=slurm_env(tmp_path, manifest, index), text=True, capture_output=True)
    assert result.returncode == 2 and not result.stdout


def test_shell_syntax():
    result = subprocess.run(["bash", "-n", str(LAUNCHER), str(SLURM)], capture_output=True)
    assert result.returncode == 0, result.stderr


def test_resume_without_other_options_reaches_python_with_empty_bash_array(tmp_path, manifest):
    # macOS Bash3 with `set -u` treats an empty array expansion as unbound.
    result = launch(tmp_path, manifest, "0", "--resume")
    assert "unbound variable" not in result.stderr
    assert result.returncode == 2 and "No programme run exists" in result.stderr


def test_dry_run_refuses_a_stale_source_lock_without_creating_outputs(tmp_path, manifest):
    stale = json.loads(manifest.read_text())
    stale["source_sha256"] = "0" * 64
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(stale))
    result = launch(tmp_path, path, "0", "--dry-run")
    assert result.returncode == 2
    assert "Checkout differs from the locked manifest source" in result.stderr
    assert not (tmp_path / "runs").exists()
