"""Execute batch-file shell routes with fake Slurm and Python, never real jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from publication_reconstruction import study
from publication_reconstruction.__main__ import parser, resolve


ROOT = Path(__file__).resolve().parents[1]
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SCRIPTS = ("train", "preflight", "resume", "prepare_evaluation", "evaluate")
WORKLOADS = (("pp", "real", 40_000_000), ("pcp", "real", 40_000_000),
             ("fc", "real", 28_000_000), ("pcp", "binary", 40_000_000))


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


@pytest.fixture
def shell(tmp_path):
    tools = tmp_path / "mock tools"
    submit = tmp_path / "repository with spaces ' $(not-executed);"
    submit.mkdir()
    capture = tmp_path / "calls.jsonl"
    preamble = ("import json, os, pathlib, sys\n"
        "def record(kind):\n"
        f"    row = {{'kind': kind, 'argv': sys.argv[1:], 'cwd': os.getcwd(), 'environment': "
        f"{{key: os.environ.get(key) for key in {(*THREADS, 'DGLBACKEND', 'PYTHONUNBUFFERED')!r}}}}}\n"
        "    with pathlib.Path(os.environ['MOCK_CAPTURE']).open('a') as stream: "
        "stream.write(json.dumps(row) + '\\n')\n"
        "    return row\n")
    executable(tools / "module", preamble + "record('module')\n")
    executable(tools / "srun", preamble + "row = record('srun')\n"
        "if os.environ.get('MOCK_CAPTURE_ONLY') == '1':\n"
        "    print(json.dumps(row))\n"
        "    raise SystemExit(int(os.environ.get('MOCK_PYTHON_EXIT', '0')))\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n")
    executable(tools / "sbatch", preamble + "record('FORBIDDEN_SUBMISSION')\nraise SystemExit(99)\n")
    interpreter = tools / "capture python ' $(not-executed)"
    python_body = preamble + "print(json.dumps(record('python')))\nraise SystemExit(int(os.environ.get('MOCK_PYTHON_EXIT', '0')))\n"
    executable(interpreter, python_body)
    executable(submit / ".venv/bin/python", python_body)
    environment = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "SLURM_SUBMIT_DIR": str(submit), "SLURM_ARRAY_JOB_ID": "4321",
        "HETNET_PYTHON": str(interpreter), "MOCK_CAPTURE": str(capture),
        **dict.fromkeys(THREADS, "64")}
    environment.pop("SLURM_ARRAY_TASK_ID", None)
    def invoke(name, arguments=(), extra=None, remove=()):
        env = {**environment, **(extra or {})}
        for key in remove:
            env.pop(key, None)
        script = ROOT / "slurm" / f"publication_{name}.sbatch" if isinstance(name, str) else name
        result = subprocess.run(["bash", str(script), *arguments], cwd=tmp_path,
                                env=env, text=True, capture_output=True, timeout=10)
        records = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        assert not any(row["kind"] == "FORBIDDEN_SUBMISSION" for row in records)
        return result, records
    return submit, interpreter, invoke


def assert_runtime(record, submit):
    assert record["cwd"] == str(submit)
    assert record["environment"] == {**dict.fromkeys(THREADS, "1"),
                                       "DGLBACKEND": "pytorch", "PYTHONUNBUFFERED": "1"}


@pytest.mark.parametrize("preflight,index", [(False, i) for i in range(12)] + [(True, i) for i in range(4)])
def test_all_locked_array_shell_routes_reach_expected_protocol(shell, tmp_path, preflight, index):
    submit, interpreter, invoke = shell
    run_root = tmp_path / "runs ' $(not-executed); with spaces"
    name = "preflight" if preflight else "train"
    variable = "PUBLICATION_PREFLIGHT_ROOT" if preflight else "PUBLICATION_RUN_ROOT"
    result, records = invoke(name, extra={variable: str(run_root), "SLURM_ARRAY_TASK_ID": str(index)})
    assert result.returncode == 0, result.stderr
    assert [r["kind"] for r in records] == ["module", "srun", "python"]
    assert records[0]["argv"] == ["load", "anaconda/anaconda-2024.10"]
    expected = ["-u", "-m", "publication_reconstruction", "run-index"]
    if preflight:
        expected += ["--preflight"]
    expected += ["--index", str(index), "--run-root", str(run_root)]
    assert records[-1]["argv"] == expected
    assert records[-2]["argv"] == [str(interpreter), *expected]
    assert_runtime(records[-1], submit)
    assert_runtime(records[-2], submit)
    routed = parser().parse_args(records[-1]["argv"][3:])
    protocol = resolve(parser().parse_args(study.train_arguments(routed.index, routed.run_root, routed.preflight)))
    task, variant, target = WORKLOADS[index if preflight else index // 3]
    assert (protocol["task"], protocol["variant"], protocol["seed"]) == (task, variant, 991 if preflight else index % 3)
    assert protocol["model_spec"] == "supplement-v1" and protocol["env_version"] == "corrected-v1"
    assert protocol["collectors"] == 4 and protocol["episode_horizon"] == (300 if task == "fc" else 80)
    assert protocol["max_env_steps"] == (None if preflight else target)
    if preflight:
        assert protocol["epochs"] * protocol["updates_per_epoch"] == 100
    assert not run_root.exists()


@pytest.mark.parametrize("name", ["resume", "prepare_evaluation", "evaluate"])
def test_single_job_wrappers_preserve_quoted_arguments_and_thread_limits(shell, tmp_path, name):
    submit, interpreter, invoke = shell
    unusual = tmp_path / "path with spaces ' $value; `not-executed` $(not-executed)"
    arguments = {
        "resume": ["--run-dir", str(unusual / "parent"), "--checkpoint", str(unusual / "model.pt"),
                   "--output", str(unusual / "continued")],
        "prepare_evaluation": ["--run-root", str(unusual / "runs"), "--pcp-panel-dir", str(unusual / "pcp panel"),
                               "--run-map", str(unusual / "map.json"), "--output", str(unusual / "prepared")],
        "evaluate": ["--run-dir", str(unusual / "parent"), "--checkpoint", str(unusual / "model.pt"),
                     "--scenarios", str(unusual / "scenarios.json"), "--protocol", str(unusual / "protocol.json"),
                     "--trace", "--sham", "--output", str(unusual / "result.json")],
    }[name]
    result, records = invoke(name, arguments)
    assert result.returncode == 0, result.stderr
    prefix = ["-u", "-m", "publication_reconstruction", name.replace("_", "-")]
    if name == "resume":
        prefix += ["--wall-seconds", "165600", "--episode-log", "stdout"]
    assert records[-1]["argv"] == prefix + arguments
    assert records[-2]["argv"] == [str(interpreter), *prefix, *arguments]
    assert_runtime(records[-1], submit)
    assert_runtime(records[-2], submit)
    assert not unusual.exists()


@pytest.mark.parametrize("name", SCRIPTS)
def test_batch_files_propagate_job_failure_without_nested_submission(shell, name):
    _, _, invoke = shell
    result, records = invoke(name, extra={"SLURM_ARRAY_TASK_ID": "0", "MOCK_PYTHON_EXIT": "37"})
    assert result.returncode == 37
    assert records[-1]["kind"] == "python"


@pytest.mark.parametrize("name", SCRIPTS)
def test_default_interpreter_uses_submit_checkout(shell, name):
    submit, _, invoke = shell
    result, records = invoke(name, extra={"SLURM_ARRAY_TASK_ID": "0"}, remove=("HETNET_PYTHON",))
    assert result.returncode == 0, result.stderr
    assert records[-2]["argv"][0] == str(submit / ".venv/bin/python")
    assert_runtime(records[-1], submit)


@pytest.mark.parametrize("name", ["train", "preflight"])
def test_array_requires_scheduler_index_before_python(shell, name):
    _, _, invoke = shell
    result, records = invoke(name)
    assert result.returncode != 0
    assert not any(row["kind"] in ("srun", "python") for row in records)


@pytest.mark.parametrize("name", SCRIPTS)
def test_batch_files_have_required_directives_and_valid_shell(name):
    path = ROOT / "slurm" / f"publication_{name}.sbatch"
    subprocess.run(["bash", "-n", str(path)], check=True)
    content = path.read_text()
    directives = dict(line[len("#SBATCH --"):].split("=", 1)
                      for line in content.splitlines() if line.startswith("#SBATCH --"))
    assert directives["account"] == "cenyioha" and directives["partition"] == "normal"
    assert directives["nodes"] == directives["ntasks"] == "1"
    training = name in ("train", "preflight", "resume")
    assert directives["cpus-per-task"] == ("4" if training else "1")
    assert directives["mem"] == ("16G" if training else "4G")
    assert directives["time"] == {"train": "48:00:00", "resume": "48:00:00", "preflight": "02:00:00",
                                   "prepare_evaluation": "00:30:00", "evaluate": "04:00:00"}[name]
    assert directives.get("array") == {"train": "0-11%3", "preflight": "0-3%1"}.get(name)
    for key in ("output", "error"):
        assert directives[key].startswith("logs_1/") and "%A_%a" in directives[key]
    assert "sbatch " not in content and "exec srun " in content


@pytest.mark.parametrize("condition", ["nominal", "failure", "sham"])
def test_generated_frozen_array_uses_exact_paths_and_condition_without_policy_execution(shell, tmp_path, condition):
    submit, _, invoke = shell
    unusual = str(tmp_path / "source ' $(not-executed); with spaces")
    job = {"run_dir": unusual, "path": unusual + "/model.pt", "scenarios": unusual + "/scenarios.json",
           "protocol": unusual + "/protocol.json", "output": unusual + "/result.json", "condition": condition}
    path = tmp_path / "prepared.sbatch"
    path.write_text(study.slurm_array([job]))
    subprocess.run(["bash", "-n", str(path)], check=True)
    result, records = invoke(path, extra={"SLURM_ARRAY_TASK_ID": "0", "MOCK_CAPTURE_ONLY": "1"})
    assert result.returncode == 0, result.stderr
    # Capture-only srun deliberately never executes the archived interpreter.
    assert [row["kind"] for row in records] == ["module", "srun"]
    expected = [str(ROOT / ".venv/bin/python"), "-m", "publication_reconstruction", "evaluate",
                "--run-dir", job["run_dir"], "--checkpoint", job["path"], "--scenarios", job["scenarios"],
                "--protocol", job["protocol"], "--output", job["output"]]
    if condition != "nominal":
        expected += ["--trace"]
    if condition == "sham":
        expected += ["--sham"]
    assert records[-1]["argv"] == expected
    assert_runtime(records[-1], submit)
    assert not Path(unusual).exists()
