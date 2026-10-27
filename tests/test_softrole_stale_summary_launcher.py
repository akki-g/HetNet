"""Stability-gate summary routing and exit codes without training or a scheduler."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from softrole import stale_capability as gate
from test_softrole_graded_pilot_launcher import THREADS, assert_environment, calls, run_slurm
from test_softrole_stale_launcher import setup_environment


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "slurm/softrole_stale_gate_summary.sbatch"
ROUTES = {
    "SOFTROLE_STALE_MANIFEST": "manifest.json",
    "SOFTROLE_STALE_RUN_ROOT": "training",
    "SOFTROLE_STALE_SUMMARY": "gate_summary.json",
}


def summary_environment(tmp_path, **kwargs):
    root, manifest, run_root, env = setup_environment(tmp_path, **kwargs)
    env.pop("SOFTROLE_STALE_SUMMARY", None)
    python = Path(env.get("HETNET_PYTHON", root / ".venv/bin/python"))
    python.write_text(python.read_text().replace(
        "operation if operation == 'prepare' else",
        "operation if operation in ('prepare', 'gate-summary') else"))
    summary = Path(env["HETNET_RUN_DIR"]) / "gate_summary.json"
    return root, manifest, run_root, summary, env


def summary_args(manifest, run_root, summary):
    return ["-u", "-m", "softrole.stale_capability", "gate-summary",
            "--manifest", str(manifest), "--run-root", str(run_root), "--output", str(summary)]


def assert_not_executed(env):
    assert not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


def test_summary_defaults_route_one_call_under_execution_contract(tmp_path):
    root, manifest, run_root, summary, env = summary_environment(tmp_path, existing=True, defaults=True)
    assert not any(variable in env for variable in ROUTES)
    result = run_slurm(SUMMARY, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)
    assert len(observed) == 1
    assert observed[0]["args"] == summary_args(manifest, run_root, summary)
    assert observed[0]["python"] == str(root / ".venv/bin/python")
    assert_environment(observed[0], root)
    assert Path(env["MODULE_CAPTURE"]).read_text() == "load anaconda/anaconda-2024.10\n"
    assert not run_root.exists()


def test_summary_preserves_contained_overrides_with_spaces(tmp_path):
    root, manifest, run_root, _, env = summary_environment(tmp_path, existing=True)
    summary = Path(env["HETNET_RUN_DIR"]) / "summary reports" / "gate summary.json"
    env["SOFTROLE_STALE_SUMMARY"] = str(summary)
    assert all(" " in env[variable] for variable in ROUTES)
    result = run_slurm(SUMMARY, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)
    assert len(observed) == 1
    assert observed[0]["args"] == summary_args(manifest, run_root, summary)
    assert observed[0]["python"] == env["HETNET_PYTHON"]
    assert_environment(observed[0], root)


@pytest.mark.parametrize("arguments", [
    ["--dry-run"], ["--output", "x"], ["--run-root", "x"], ["--entropy-coeff", "1"],
])
def test_summary_refuses_all_arguments_before_execution(tmp_path, arguments):
    _, _, _, summary, env = summary_environment(tmp_path)
    result = run_slurm(SUMMARY, arguments, tmp_path, env)
    assert result.returncode == 2 and "Usage:" in result.stderr
    assert_not_executed(env)
    assert not summary.exists()


@pytest.mark.parametrize("variable", ROUTES)
@pytest.mark.parametrize("symlink", [False, True])
def test_summary_refuses_direct_and_symlink_escapes_without_writes(tmp_path, variable, symlink):
    _, manifest, _, summary, env = summary_environment(tmp_path, existing=True)
    outside = tmp_path / "outside run"
    outside.mkdir()
    outside_manifest = outside / "manifest.json"
    outside_manifest.write_bytes(b'{"outside": "preserve these bytes"}\n')
    outside_before = {path.name: path.read_bytes() for path in outside.iterdir()}
    route = outside
    if symlink:
        route = Path(env["HETNET_RUN_DIR"]) / "escaped output"
        route.symlink_to(outside, target_is_directory=True)
    env[variable] = str(route / ROUTES[variable])
    manifest_before = manifest.read_bytes()
    result = run_slurm(SUMMARY, [], tmp_path, env)
    assert result.returncode != 0 and "must stay inside" in result.stderr
    assert_not_executed(env)
    assert manifest.read_bytes() == manifest_before
    assert {path.name: path.read_bytes() for path in outside.iterdir()} == outside_before
    assert not summary.exists()


@pytest.mark.parametrize("manifest_kind", ["missing", "directory"])
def test_summary_requires_a_regular_manifest_file(tmp_path, manifest_kind):
    _, manifest, _, summary, env = summary_environment(tmp_path)
    if manifest_kind == "directory":
        manifest.mkdir()
    result = run_slurm(SUMMARY, [], tmp_path, env)
    assert result.returncode == 2 and "Manifest file not found" in result.stderr
    assert_not_executed(env)
    assert not summary.exists()


@pytest.mark.parametrize("summary_kind", ["file", "symlink", "dangling_symlink"])
def test_summary_preserves_existing_files_and_symlinks(tmp_path, summary_kind):
    _, _, _, summary, env = summary_environment(tmp_path, existing=True)
    contents = b'{"summary": "preserve these bytes"}\n'
    target = summary.parent / "earlier summary.json"
    if summary_kind == "file":
        summary.write_bytes(contents)
    else:
        if summary_kind == "symlink":
            target.write_bytes(contents)
        summary.symlink_to(target)
    result = run_slurm(SUMMARY, [], tmp_path, env)
    assert result.returncode == 2 and "Summary exists" in result.stderr
    assert_not_executed(env)
    if summary_kind == "file":
        assert not summary.is_symlink() and summary.read_bytes() == contents
    else:
        assert summary.is_symlink() and summary.readlink() == target
        if summary_kind == "symlink":
            assert target.read_bytes() == contents and summary.read_bytes() == contents
        else:
            assert not summary.exists() and not target.exists()


def test_summary_propagates_the_command_failure_exit_code(tmp_path):
    root, manifest, run_root, summary, env = summary_environment(tmp_path, existing=True)
    env["MOCK_FAIL"] = "gate-summary"
    result = run_slurm(SUMMARY, [], tmp_path, env)
    assert result.returncode == 17
    observed = calls(env)
    assert len(observed) == 1 and observed[0]["args"] == summary_args(manifest, run_root, summary)
    assert_environment(observed[0], root)
    assert not summary.exists()


def test_submission_helper_routes_summary_without_writes_or_scheduler(tmp_path, capsys, monkeypatch):
    spec = importlib.util.spec_from_file_location("stale_summary_submit_test", ROOT / "scripts/submit_stokes.py")
    submit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submit)
    checkout = tmp_path / "checkout"
    script = checkout / "slurm" / SUMMARY.name
    script.parent.mkdir(parents=True)
    script.write_bytes(SUMMARY.read_bytes())
    monkeypatch.setattr(submit, "ROOT", checkout)
    for variable in ROUTES:
        monkeypatch.delenv(variable, raising=False)

    def forbid_command(*args, **kwargs):
        pytest.fail("Submission dry-run must not invoke a scheduler or any other command")

    monkeypatch.setattr(submit.subprocess, "run", forbid_command)
    before = sorted(path.relative_to(checkout) for path in checkout.rglob("*"))
    result = submit.main(["--title", "stale gate summary check", "--purpose", "Engineering dry-run",
                          "--dry-run", "--", str(script)])
    assert result == 0
    preview = json.loads(capsys.readouterr().out)
    run = Path(preview["run_dir"])
    assert not run.exists() and not (checkout / "runs").exists()
    assert sorted(path.relative_to(checkout) for path in checkout.rglob("*")) == before
    for variable, relative in ROUTES.items():
        assert preview["output_environment"][variable] == str(run / relative)
    assert preview["output_environment"]["HETNET_RUN_DIR"] == str(run)
    assert preview["command"][-1] == str(script)
    assert script.name in submit.RUN_ROOT_OUTPUT_JOBS
    with pytest.raises(ValueError, match="inside"):
        submit.route_outputs(script, [], run, {"SOFTROLE_STALE_SUMMARY": str(tmp_path / "escape.json")})
    with pytest.raises(ValueError, match="inside"):
        submit.route_outputs(script, ["--run-root", str(tmp_path / "escape")], run, {})


def test_source_inventory_includes_summary_job():
    assert "slurm/softrole_stale_gate_summary.sbatch" in gate.source_snapshot()["files"]


def test_summary_shell_syntax_and_unmeasured_single_task_resources():
    result = subprocess.run(["bash", "-n", str(SUMMARY)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    script = SUMMARY.read_text()
    assert "unmeasured starting values" in script
    for setting in ("--job-name=stale-capability-gate-summary", "--account=cenyioha",
                    "--partition=normal", "--constraint=sapphirerapids", "--nodes=1",
                    "--ntasks=1", "--cpus-per-task=1", "--mem=4G", "--time=00:30:00"):
        assert f"#SBATCH {setting}\n" in script
    assert not any(line.startswith("#SBATCH") and "--array" in line for line in script.splitlines())


def test_real_cli_writes_incomplete_summary_and_refuses_overwrite_without_training(tmp_path):
    manifest = tmp_path / "manifest.json"
    run_root = tmp_path / "empty training"
    run_root.mkdir()
    summary = tmp_path / "gate_summary.json"
    env = dict(os.environ)
    env.update({name: "1" for name in THREADS})
    env.update(PYTHONPATH=f"{ROOT}/src:{ROOT}/src/envs", DGLBACKEND="pytorch")
    command = [sys.executable, "-u", "-m", "softrole.stale_capability"]
    prepared = subprocess.run(command + ["prepare", "--stage", "gate", "--output", str(manifest)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert prepared.returncode == 0, prepared.stderr
    manifest_before = manifest.read_bytes()
    arguments = summary_args(manifest, run_root, summary)[4:]
    first = subprocess.run(command + ["gate-summary", *arguments], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stderr
    report = json.loads(summary.read_text())
    assert json.loads(first.stdout) == report
    assert report["decision"] == "awaiting_complete_valid_evidence"
    assert [(row["coefficient"], row["status"]) for row in report["coefficients"]] == [
        (0.003, "incomplete"), (0.01, "incomplete")]
    for coefficient in report["coefficients"]:
        assert [(row["seed"], row["status"]) for row in coefficient["seeds"]] == [
            (0, "incomplete"), (1, "incomplete"), (2, "incomplete")]
    assert report["selected_coefficient"] is None and report["eligible_coefficients"] == []
    before = summary.read_bytes()
    second = subprocess.run(command + ["gate-summary", *arguments], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert second.returncode == 2 and "Gate summary already exists" in second.stderr
    assert summary.read_bytes() == before and manifest.read_bytes() == manifest_before
    assert list(run_root.iterdir()) == []
