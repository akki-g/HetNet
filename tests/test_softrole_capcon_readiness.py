"""The readiness driver fails explicitly and confines interruption to disposable work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capcon_readiness", ROOT / "scripts/softrole_capcon_readiness.py")
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


def test_shell_environment_isolation_retains_compute_runtime(tmp_path):
    original = {"BASH_FUNC_module%%": "function", "BASH_FUNC_srun%%": "function", "BASH_ENV": "bad",
                "ENV": "bad", "PATH": "/module/bin", "LD_LIBRARY_PATH": "/module/lib", "SLURM_JOB_ID": "7"}
    actual = readiness.test_environment(tmp_path, tmp_path / "tools", original)
    assert not any(k.startswith("BASH_FUNC_") or k in ("BASH_ENV", "ENV") for k in actual)
    assert {k: actual[k] for k in ("PATH", "LD_LIBRARY_PATH", "SLURM_JOB_ID")} == {
        "PATH": "/module/bin", "LD_LIBRARY_PATH": "/module/lib", "SLURM_JOB_ID": "7"}
    assert actual["HETNET_GRADIENT_PROBE_OUTER_TIMEOUT"] == "180"
    assert actual["PYTHONPATH"].split(os.pathsep) == [str(tmp_path), str(tmp_path / "envs"), str(tmp_path / "tools")]
    assert original["BASH_ENV"] == "bad"


@pytest.mark.parametrize("updates,epochs,per_epoch", [(60, 6, 10), (2, 2, 1), (17, 17, 1)])
def test_bounded_runtime_keeps_four_collectors_and_complete_update_budget(updates, epochs, per_epoch):
    args = readiness.bounded_options(updates)
    options = dict(zip(args[::2], args[1::2]))
    assert int(options["--epochs"]) * int(options["--updates-per-epoch"]) == updates
    assert options["--epochs"] == str(epochs) and options["--updates-per-epoch"] == str(per_epoch)
    assert options["--batch-steps"] == "500" and options["--nprocesses"] == "4"
    assert options["--diagnostics-every"] == "50" and options["--save-every"] == "1"
    assert int(options["--total-steps"]) > updates * 4 * (500 + 80)


@pytest.mark.parametrize("updates", [True, 1, 0, -1, 2.5])
def test_too_short_or_invalid_check_cannot_claim_interruption(updates):
    with pytest.raises(ValueError):
        readiness.bounded_options(updates)


def test_runner_records_original_failure_and_preserves_logs(tmp_path):
    runner = readiness.Runner(ROOT, tmp_path / "out")
    with pytest.raises(RuntimeError, match="specific fixture error"):
        runner.run("bad", [sys.executable, "-c", "import sys;sys.stderr.write('specific fixture error');sys.exit(3)"])
    record = json.loads((runner.output / "readiness.json").read_text())["commands"][0]
    assert record["returncode"] == 3 and record["status"] == "failed"
    with pytest.raises(FileExistsError):
        readiness.Runner(ROOT, runner.output)


def test_runner_timeout_stops_a_real_disposable_process(tmp_path):
    runner = readiness.Runner(ROOT, tmp_path / "out", timeout=.2)
    end = tmp_path / "should_not_appear"
    code = f"import time;from pathlib import Path;time.sleep(1);Path({str(end)!r}).write_text('alive')"
    with pytest.raises(TimeoutError):
        runner.run("timeout", [sys.executable, "-c", code])
    assert runner.record["commands"][0]["timed_out"] is True
    time.sleep(1)
    assert not end.exists()


def test_committed_marker_kills_parent_and_spawned_child_only(tmp_path):
    runner = readiness.Runner(ROOT, tmp_path / "out")
    markers = tmp_path / "checkpoints"
    markers.mkdir()
    escaped = tmp_path / "child_survived"
    child = f"import time;from pathlib import Path;time.sleep(1);Path({str(escaped)!r}).write_text('alive')"
    code = ("import subprocess,sys,time;from pathlib import Path;"
            f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
            f"Path({str(markers / 'epoch0001.pt.complete.json')!r}).write_text('{{}}');time.sleep(20)")
    row = runner.run("interrupted", [sys.executable, "-c", code], interrupt_marker=markers)
    assert row["interrupted_after_commit"] and row["returncode"] < 0
    time.sleep(1)
    assert not escaped.exists()


def test_finished_run_cannot_count_as_an_interruption(tmp_path):
    runner = readiness.Runner(ROOT, tmp_path / "out")
    with pytest.raises(RuntimeError, match="before a checkpoint"):
        runner.run("finished", [sys.executable, "-c", "pass"], interrupt_marker=tmp_path / "none")


def test_xml_requires_tests_and_exit_report_consistency(tmp_path):
    path = tmp_path / "suite.xml"
    with pytest.raises(RuntimeError, match="No usable pytest XML"):
        readiness.test_results(path, 1, ROOT)
    path.write_text("<testsuite/>")
    with pytest.raises(RuntimeError, match="exit/report mismatch"):
        readiness.test_results(path, 0, ROOT)
    path.write_text('<testsuite><testcase classname="tests.test_fixture" name="test_a"/></testsuite>')
    assert readiness.test_results(path, 0, ROOT)["tests"] == 1
    with pytest.raises(RuntimeError, match="exit/report mismatch"):
        readiness.test_results(path, 1, ROOT)


def test_xml_preserves_specific_failure_identity(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_fixture.py").write_text("")
    xml = tmp_path / "suite.xml"
    xml.write_text('<testsuite><testcase classname="tests.test_fixture" name="test_a[3]">'
                   '<failure message="actual contract failure"/></testcase></testsuite>')
    assert readiness.test_results(xml, 1, tmp_path)["failures"] == {
        "tests/test_fixture.py::test_a[3]": {"kind": "failure", "message": "actual contract failure"}}


@pytest.mark.parametrize("corruption", [None, "parent", "model"])
def test_resume_helper_requires_immutable_parent_and_exact_uninterrupted_state(tmp_path, monkeypatch, corruption):
    import softrole.programme as programme
    output = tmp_path / "out"
    output.mkdir()
    logical = output / "interrupted" / "control/seed0"
    parent = logical / "checkpoints/epoch0001.pt"
    completed = logical / "attempts/0001/checkpoints/epoch0002.pt"
    reference = tmp_path / "reference.pt"
    payload = {"model_state": {"w": torch.tensor([1.])}, "optimizer_state": {"state": torch.tensor([2.])},
               "signature": "same", "updates": 2, "total_steps": 4000, "total_episodes": 50}
    torch.save(payload, reference)
    before = {"checkpoint": str(parent), "updates": 1}
    final = {"checkpoint": str(completed), "updates": 2}
    calls = []
    mutate = {"parent": corruption == "parent", "model": corruption == "model"}
    def run(name, command, **kwargs):
        calls.append((name, command, kwargs))
        if name == "interrupt":
            parent.parent.mkdir(parents=True)
            parent.write_bytes(b"immutable parent")
        else:
            completed.parent.mkdir(parents=True)
            saved = dict(payload)
            if mutate["model"]:
                saved["model_state"] = {"w": torch.tensor([9.])}
            torch.save(saved, completed)
            (completed.parent.parent / "attempt.json").write_text(json.dumps({"parent_checkpoint_sha256": readiness.sha(parent)}))
            if mutate["parent"]:
                parent.write_bytes(b"tampered")
    runner = SimpleNamespace(output=output, env={}, run=run)
    monkeypatch.setattr(programme, "accepted_checkpoints", lambda path: [before])
    monkeypatch.setattr(readiness, "inspect_run", lambda path, updates: (
        [final if Path(path) == logical else {"checkpoint": str(reference)}], {"updates": [{"update": 1}, {"update": 2}]}))
    manifest = {"entries": [{"relative_output": "control/seed0"}]}
    if corruption:
        message = "changed its parent" if corruption == "parent" else "differs from uninterrupted"
        with pytest.raises(RuntimeError, match=message):
            readiness.interruption_check(runner, manifest, ["--epochs", "2"], 2, tmp_path / "reference")
    else:
        result = readiness.interruption_check(runner, manifest, ["--epochs", "2"], 2, tmp_path / "reference")
        assert result["parent_and_abandoned_files_unchanged"] and result["uninterrupted_state_and_statistics_equal"]
        assert calls[1][1] == ["bash", "scripts/softrole_programme.sh", "0", "--resume"]


@pytest.mark.parametrize("difference", ["dtype", "value", "shape", "nan"])
def test_state_comparison_rejects_material_tensor_differences(difference):
    a = torch.tensor([1.], dtype=torch.float64)
    b = {"dtype": a.float(), "value": a + 1, "shape": a.reshape(1, 1), "nan": a * float("nan")}[difference]
    assert not readiness.equal_state({"state": [a]}, {"state": [b]})


def test_cli_requires_explicit_evidence_and_reduced_flags():
    parser = readiness.parser()
    args = parser.parse_args(["--baseline", "/baseline", "--output", "/new", "--selection", "/selection"])
    assert args.updates == 60 and args.timeout_seconds == 3600 and not args.skip_tests
    args = parser.parse_args(["--baseline", "/baseline", "--output", "/new", "--selection", "/selection",
                              "--updates", "2", "--skip-tests"])
    assert args.updates == 2 and args.skip_tests
    with pytest.raises(SystemExit):
        parser.parse_args(["--baseline", "/baseline", "--output", "/new"])


def test_slurm_wrapper_preserves_arguments_and_uses_existing_interpreter(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    capture = tmp_path / "captured.json"
    submit = tmp_path / "checkout with spaces"
    submit.mkdir()
    def executable(path, code):
        path.write_text(f"#!{sys.executable}\n" + code)
        path.chmod(0o755)
    executable(tools / "module", "raise SystemExit(0)\n")
    executable(tools / "srun", "import os,sys\nos.execv(sys.argv[2],sys.argv[2:])\n")
    interpreter = tools / "python"
    executable(interpreter, "import json,os,sys\nfrom pathlib import Path\n"
               f"Path({str(capture)!r}).write_text(json.dumps({{'args':sys.argv[1:],'tmp':os.environ['TMPDIR'],'threads':os.environ['OMP_NUM_THREADS']}}))\n")
    env = readiness.test_environment(ROOT, environment={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "SLURM_SUBMIT_DIR": str(submit), "SLURM_JOB_ID": "123", "HETNET_PYTHON": str(interpreter)})
    args = ["--baseline", "/path with spaces/base", "--selection", "/selection", "--output", "/new"]
    result = subprocess.run(["bash", str(ROOT / "slurm/softrole_capcon_readiness.sbatch"), *args],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    actual = json.loads(capture.read_text())
    assert actual["args"] == ["-u", "scripts/softrole_capcon_readiness.py", "--checkout", str(submit), *args]
    assert actual["threads"] == "1" and Path(actual["tmp"]).is_dir()
    temporary = Path(actual["tmp"])
    assert str(temporary).startswith("/tmp/capcon-123.")
    assert temporary.stat().st_mode & 0o077 == 0
    temporary.rmdir()


@pytest.mark.parametrize("fixture_scope", ["matched", "other_fixture", "other_source"])
@pytest.mark.parametrize("fixture_error", [False, True])
def test_external_baseline_timeout_adapter_is_narrow_and_restored(tmp_path, monkeypatch, fixture_scope, fixture_error):
    baseline = tmp_path / "baseline"
    target = baseline / "tests/test_gradient_contract.py"
    target.parent.mkdir(parents=True)
    target.write_text("def gradient_probe():\n    pass\n")
    original_bytes = target.read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    plugin_path = readiness.baseline_timeout_plugin(output, baseline)
    plugin_spec = importlib.util.spec_from_file_location("baseline_timeout_test", plugin_path)
    plugin = importlib.util.module_from_spec(plugin_spec)
    plugin_spec.loader.exec_module(plugin)
    source = target if fixture_scope != "other_source" else baseline / "other.py"
    namespace = {}
    exec(compile(original_bytes, str(source), "exec"), namespace)
    fixture = SimpleNamespace(argname="gradient_probe" if fixture_scope != "other_fixture" else "another_probe",
                              func=namespace["gradient_probe"])
    observed = []
    def communicate(process, input=None, timeout=None):
        observed.append((list(process.args), input, timeout))
        return "result", None
    monkeypatch.setattr(subprocess.Popen, "communicate", communicate)
    hook = plugin.pytest_fixture_setup(fixture, None)
    next(hook)
    probe_command = [sys.executable, str(baseline / "tests/helpers/shared_gradient_probe.py"),
                     "--updates", "3", "--timeout-seconds", "60"]
    process = SimpleNamespace(args=probe_command)
    assert subprocess.Popen.communicate(process, input="payload", timeout=85) == ("result", None)
    subprocess.Popen.communicate(process, timeout=5)
    subprocess.Popen.communicate(SimpleNamespace(args=[sys.executable, "other.py"]), timeout=85)
    if fixture_error:
        with pytest.raises(RuntimeError, match="fixture failed"):
            hook.throw(RuntimeError("fixture failed"))
    else:
        with pytest.raises(StopIteration):
            next(hook)
    assert subprocess.Popen.communicate is communicate
    assert observed[0] == (probe_command, "payload", 180 if fixture_scope == "matched" else 85)
    assert observed[1][2] == 5 and observed[2][2] == 85
    assert process.args == probe_command and process.args[-2:] == ["--timeout-seconds", "60"]
    assert target.read_bytes() == original_bytes


def test_warm_throughput_excludes_startup_epoch():
    result = readiness.epoch_timing([
        {"steps": 100, "wall_time_seconds": 100.},
        {"steps": 200, "wall_time_seconds": 2.},
        {"steps": 300, "wall_time_seconds": 3.},
    ])
    assert result["first_epoch_loop_seconds"] == 100.
    assert result["post_first_epoch_steps"] == 500
    assert result["post_first_epoch_loop_seconds"] == 5.
    assert result["post_first_epoch_steps_per_loop_second"] == 100.
    assert result["epoch_loop_seconds"] == 105.
    one = readiness.epoch_timing([{"steps": 100, "wall_time_seconds": 100.}])
    assert one["post_first_epoch_steps_per_loop_second"] is None
