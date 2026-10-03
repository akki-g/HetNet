"""Preparation validates checkpoints and writes commands; no policies or Slurm run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hetnet_ext.signatures import model_signature
from scripts.prepare_pcp_frozen import ROOT, prepare
from softrole import CHECKPOINT_VERSION, ENVIRONMENT_VERSION
from softrole.config import Config
from softrole.model import SoftRoleNet


THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "runs with spaces"
    # Synthetic ledgers/checkpoints test selection and metadata only; no training.
    metrics = [{"epoch": epoch, "steps": 20000, "episodes": 250,
                "total_steps": 20000 * epoch, "total_episodes": 250 * epoch,
                "updates": 10 * epoch} for epoch in range(1, 1551)]
    for mode in ("shared", "banked"):
        for seed in range(3):
            run = root / f"pcp_{mode}" / f"seed{seed}"
            (run / "checkpoints").mkdir(parents=True)
            (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
            config = Config(model=mode, seed=seed, pre_dim=4, hidden_dim=4,
                            heads=1, head_dim=2, msg_dim=2, experts=2)
            model = SoftRoleNet(**config.model_kwargs()).double()
            signature = model_signature(model)
            for epoch in (1450, 1500, 1550):
                path = run / "checkpoints" / f"epoch{epoch:04d}.pt"
                payload = {"format_version": CHECKPOINT_VERSION, "environment_version": ENVIRONMENT_VERSION,
                    "config": config.to_dict(), "model_config": config.model_kwargs(),
                    "model_state": model.state_dict(), "signature": signature,
                    "source_sha256": "a" * 64,
                    "epoch": epoch, "updates": epoch * 10,
                    "total_steps": epoch * 20000, "total_episodes": epoch * 250}
                torch.save(payload, path)
                Path(str(path) + ".signature.json").write_text(json.dumps(signature))
    return root


def test_prepare_selects_first_threshold_checkpoint_and_writes_explicit_panels(run_root, tmp_path):
    output = tmp_path / "prepared plan"
    manifest = prepare(run_root, output)
    assert manifest["submitted"] is False and manifest["min_steps"] == 30000000
    assert len(manifest["policies"]) == 6 and len(manifest["jobs"]) == 18
    assert {(row["model"], row["training_seed"]) for row in manifest["policies"]} == {
        (mode, seed) for mode in ("shared", "banked") for seed in range(3)}
    for policy in manifest["policies"]:
        assert policy["checkpoint_progress"]["epoch"] == 1500
        assert policy["checkpoint_progress"]["updates"] == 15000
        assert policy["checkpoint_progress"]["total_steps"] == 30000000
        assert policy["checkpoint_sha256"] == hashlib.sha256(Path(policy["checkpoint"]).read_bytes()).hexdigest()
    nominal = json.loads((output / "scenarios_nominal.json").read_text())
    failure = json.loads((output / "scenarios_failure.json").read_text())
    assert len(nominal) == len({row["scenario_id"] for row in nominal}) == 2500
    for team in [(2, 1), (1, 2), (2, 2), (3, 1), (3, 2)]:
        assert sum((row["num_p"], row["num_a"]) == team for row in nominal) == 500
    assert all(row["event_step"] == -1 for row in nominal)
    assert len(failure) == 100 and all(10 <= row["event_step"] <= 30 for row in failure)
    assert all((row["num_p"], row["num_a"]) == (2, 1) and row["victim"] in (0, 1) for row in failure)
    for mode in ("shared", "banked"):
        for seed in range(3):
            paired = [row for row in manifest["jobs"] if row["model"] == mode
                      and row["training_seed"] == seed and row["condition"] in ("failure", "sham")]
            assert paired[0]["scenarios"] == paired[1]["scenarios"]
    assert all(Path(job[key]).is_absolute() for job in manifest["jobs"] for key in ("checkpoint", "output", "scenarios"))
    assert not (output / "results").exists() and not (output / "job_ids.tsv").exists()
    source = json.loads((output / "source_manifest.json").read_text())
    assert source["sha256"] == manifest["evaluation_source_sha256"]
    assert manifest["preparation_source_sha256"] == hashlib.sha256((output / "preparation_source.py").read_bytes()).hexdigest()
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for name in ("submit.sh", "submit.sbatch"):
        assert name in manifest["artifact_sha256"]
        subprocess.run(["bash", "-n", str(output / name)], check=True)


def test_missing_threshold_checkpoint_fails_before_output_creation(run_root, tmp_path):
    for epoch in (1500, 1550):
        (run_root / "pcp_banked/seed2/checkpoints" / f"epoch{epoch:04d}.pt").unlink()
    output = tmp_path / "missing" / "plan"
    with pytest.raises(ValueError, match="No saved checkpoint reaches"):
        prepare(run_root, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("mutation,match", [
    ("count", "total_steps mismatch"), ("seed", "fixed nominal PCP"),
    ("failure", "fixed nominal PCP"), ("config", "scientific configuration"),
    ("nonfinite", "Nonfinite model tensor"), ("source", "training source"),
])
def test_ineligible_panel_fails_before_output_creation(run_root, tmp_path, mutation, match):
    path = run_root / "pcp_banked/seed2/checkpoints/epoch1500.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if mutation == "count":
        saved["total_steps"] += 1
    elif mutation == "seed":
        saved["config"]["seed"] = 7
    elif mutation == "failure":
        saved["config"]["failure_prob"] = .5
    elif mutation == "config":
        saved["config"]["lr"] *= 2
    elif mutation == "nonfinite":
        next(iter(saved["model_state"].values())).view(-1)[0] = float("nan")
    else:
        saved["source_sha256"] = "b" * 64
    torch.save(saved, path)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        prepare(run_root, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    evidence = output / "keep.txt"
    evidence.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "nonexistent inputs", output)
    assert evidence.read_text() == "prior evidence"


def test_submit_script_quotes_paths_records_array_tasks_and_refuses_reuse(run_root, tmp_path):
    output = tmp_path / "plan with spaces ' $(not-a-command)"
    manifest = prepare(run_root, output)
    tools = tmp_path / "mock tools"
    tools.mkdir()
    capture = tmp_path / "captured.jsonl"
    stub = tools / "sbatch"
    stub.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        "path = pathlib.Path(os.environ['PREPARE_SUBMIT_CAPTURE'])\n"
        "count = len(path.read_text().splitlines()) if path.exists() else 0\n"
        "with path.open('a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(7000 + count)\n")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
           "PREPARE_SUBMIT_CAPTURE": str(capture)}
    first = subprocess.run(["bash", str(output / "submit.sh")], cwd=tmp_path,
                           env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    commands = [json.loads(line) for line in capture.read_text().splitlines()]
    assert commands == [["--parsable", str(output / "submit.sbatch")]]
    rows = (output / "job_ids.tsv").read_text().splitlines()
    assert rows[0] == "model\tseed\tcondition\tjob_id"
    assert [row.split("\t") for row in rows[1:]] == [
        [job["model"], str(job["training_seed"]), job["condition"], f"7000_{index}"]
        for index, job in enumerate(manifest["jobs"])]
    second = subprocess.run(["bash", str(output / "submit.sh")], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert second.returncode != 0
    assert len(capture.read_text().splitlines()) == 1
    assert (output / "job_ids.tsv").read_text().splitlines() == rows
    assert not (output / "results").exists()


def test_generated_slurm_array_routes_all_tasks_and_rejects_invalid_indices(run_root, tmp_path):
    output = tmp_path / "array with spaces ' $(not-a-command)"
    manifest = prepare(run_root, output)
    batch_script = output / "submit.sbatch"
    directives = dict(line[len("#SBATCH --"):].split("=", 1)
                      for line in batch_script.read_text().splitlines()
                      if line.startswith("#SBATCH --"))
    for name, expected in {"account": "cenyioha", "partition": "normal", "nodes": "1",
                           "ntasks": "1", "cpus-per-task": "1", "mem": "4G",
                           "time": "04:00:00", "array": "0-17%3"}.items():
        assert directives[name] == expected
    assert directives["job-name"]
    for name in ("output", "error"):
        assert directives[name].startswith("logs_sr/")
        assert "%A_%a" in directives[name]
    tools = tmp_path / "mock Slurm tools"
    tools.mkdir()
    srun_body = ''.join(f'[[ "${{{name}}}" == 1 ]] || exit 91\n' for name in THREAD_VARIABLES)
    for name, body in {"module": "exit 0\n", "srun": srun_body + 'exec "$@"\n'}.items():
        stub = tools / name
        stub.write_text("#!/usr/bin/env bash\n" + body)
        stub.chmod(0o755)
    interpreter = tools / "capture python"
    interpreter.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'argv':sys.argv[1:], 'cwd':os.getcwd(), 'environment':"
        "{key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',"
        "'NUMEXPR_NUM_THREADS','DGLBACKEND','PYTHONUNBUFFERED']}}))\n")
    interpreter.chmod(0o755)
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
           "SLURM_SUBMIT_DIR": str(ROOT), "HETNET_PYTHON": str(interpreter),
           **dict.fromkeys(THREAD_VARIABLES, "64")}
    env.pop("SLURM_ARRAY_TASK_ID", None)
    for index, job in enumerate(manifest["jobs"]):
        result = subprocess.run(["bash", str(batch_script)], cwd=tmp_path,
            env={**env, "SLURM_ARRAY_TASK_ID": str(index)}, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        record = json.loads(result.stdout)
        expected = ["-u", "-m", "softrole", "evaluate", "--scenarios", job["scenarios"]]
        if job["condition"] != "nominal":
            expected.append("--trace")
        if job["condition"] == "sham":
            expected.append("--sham")
        expected += ["--checkpoint", job["checkpoint"], "--output", job["output"]]
        assert record["argv"] == expected
        assert record["cwd"] == str(ROOT)
        assert record["environment"] == {**dict.fromkeys(THREAD_VARIABLES, "1"),
                                          "DGLBACKEND": "pytorch", "PYTHONUNBUFFERED": "1"}
    for index in (None, "", "-1", "18", "wrong", "1.5"):
        invalid_env = env if index is None else {**env, "SLURM_ARRAY_TASK_ID": index}
        result = subprocess.run(["bash", str(batch_script)], cwd=tmp_path,
                                env=invalid_env, text=True, capture_output=True)
        assert result.returncode != 0
        assert not result.stdout  # The Python/evaluator mock was never reached.
    assert not (output / "results").exists() and not (output / "job_ids.tsv").exists()


@pytest.mark.parametrize("inherited_threads", [None, "64"])
def test_prepare_limits_threads_before_numerical_imports(tmp_path, inherited_threads):
    # A fresh process observes the environment at the import boundary, before
    # OpenBLAS could attempt to allocate the inherited number of threads.
    probe = """
import builtins, json, os, runpy, sys
original_import = builtins.__import__
def inspect_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy'):
        print(json.dumps({key: os.environ.get(key) for key in sys.argv[2:]}))
        raise SystemExit(0)
    return original_import(name, *args, **kwargs)
builtins.__import__ = inspect_import
runpy.run_path(sys.argv[1], run_name='__main__')
raise SystemExit('Numerical import boundary was never reached')
"""
    env = dict(os.environ)
    for name in THREAD_VARIABLES:
        if inherited_threads is None:
            env.pop(name, None)
        else:
            env[name] = inherited_threads
    result = subprocess.run([sys.executable, "-c", probe,
                             str(ROOT / "scripts/prepare_pcp_frozen.py"), *THREAD_VARIABLES],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict.fromkeys(THREAD_VARIABLES, "1")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("launcher", ["scripts/softrole_evaluate.sh", "slurm/softrole_evaluate.sbatch",
                                      "slurm/softrole_prepare_frozen.sbatch"])
def test_standalone_launchers_override_inherited_threads_and_forward_arguments(tmp_path, launcher):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()  # Mock Python never reads or evaluates this file.
    interpreter = tmp_path / "capture-python"
    interpreter.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                           "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'environment': "
                           f"{{key: os.environ.get(key) for key in {THREAD_VARIABLES!r}}}}}))\n")
    interpreter.chmod(0o755)
    srun_body = ''.join(f'[[ "${{{name}}}" == 1 ]] || exit 91\n' for name in THREAD_VARIABLES)
    for name, body in {"module": "exit 0\n", "srun": srun_body + 'exec "$@"\n'}.items():
        stub = tmp_path / name
        stub.write_text("#!/usr/bin/env bash\n" + body)
        stub.chmod(0o755)
    env = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "64"),
           "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "SLURM_SUBMIT_DIR": str(ROOT), "HETNET_PYTHON": str(interpreter)}
    if launcher == "slurm/softrole_prepare_frozen.sbatch":
        arguments = ["--run-root", str(tmp_path / "runs with spaces ' $(not-a-command)"),
                     "--output", str(tmp_path / "prepared panel"), "--min-steps", "123"]
        expected_argv = ["-u", "scripts/prepare_pcp_frozen.py", *arguments]
    else:
        arguments = [str(checkpoint), str(tmp_path / "result.json")]
        expected_argv = ["-u", "-m", "softrole", "evaluate", "--checkpoint", arguments[0],
                         "--output", arguments[1]]
    result = subprocess.run(["bash", str(ROOT / launcher), *arguments],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"argv": expected_argv, "cwd": str(ROOT),
                                         "environment": dict.fromkeys(THREAD_VARIABLES, "1")}
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "prepared panel").exists()


def test_prepare_cli_help_does_not_create_output(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_pcp_frozen.py"), "--help"],
                            cwd=tmp_path, env={**os.environ, **dict.fromkeys(THREAD_VARIABLES, "64")},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "never submit jobs" in result.stdout and "--min-steps" in result.stdout
    assert not list(tmp_path.iterdir())
