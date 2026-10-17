"""Focused, bounded CapCon readiness checks; never submits or prepares research runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
INDICES = (0, 3, 6, 9)
BASELINE_COMMIT = "644f8f17944d01c9d10e3a4fb670cefef32ceb3f"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def test_environment(checkout, test_tools=None, environment=None):
    """Isolate mocked shell commands after the real module/allocation setup."""
    env = {k: v for k, v in (os.environ if environment is None else environment).items()
           if not k.startswith("BASH_FUNC_") and k not in ("BASH_ENV", "ENV")}
    paths = [str(checkout), str(Path(checkout) / "envs")]
    if test_tools is not None:
        paths.append(str(test_tools))
    env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONNOUSERSITE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", HETNET_GRADIENT_PROBE_OUTER_TIMEOUT="180")
    return env



def baseline_timeout_plugin(output, baseline):
    """Adapt only the immutable baseline's gradient-fixture startup allowance."""
    folder = Path(output) / "baseline_test_plugin"
    folder.mkdir()
    path = folder / "capcon_baseline_timeout.py"
    code = r'''"""External startup-time allowance; baseline code and probe arguments are unchanged."""
from pathlib import Path
import subprocess
import pytest

BASELINE_TEST = Path(__BASELINE_TEST__)
PROBE = Path(__PROBE__)

@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    if (fixturedef.argname != "gradient_probe"
            or Path(fixturedef.func.__code__.co_filename).resolve() != BASELINE_TEST):
        yield
        return
    original = subprocess.Popen.communicate
    def communicate(process, input=None, timeout=None):
        command = process.args
        if (timeout == 85 and isinstance(command, (list, tuple))
                and any(Path(str(arg)).resolve() == PROBE for arg in command[1:])):
            timeout = 180
        return original(process, input=input, timeout=timeout)
    subprocess.Popen.communicate = communicate
    try:
        yield
    finally:
        subprocess.Popen.communicate = original
'''
    code = code.replace("__BASELINE_TEST__", repr(str(Path(baseline).resolve() / "tests/test_gradient_contract.py")))
    code = code.replace("__PROBE__", repr(str(Path(baseline).resolve() / "tests/helpers/shared_gradient_probe.py")))
    with path.open("x") as stream:
        stream.write(code)
    return path


def epoch_timing(metrics):
    """Separate the first epoch, which includes collector startup, from later loops."""
    seconds = sum(row["wall_time_seconds"] for row in metrics)
    warm_seconds = sum(row["wall_time_seconds"] for row in metrics[1:])
    warm_steps = sum(row["steps"] for row in metrics[1:])
    return {"epoch_loop_seconds": seconds,
            "first_epoch_steps": metrics[0]["steps"],
            "first_epoch_loop_seconds": metrics[0]["wall_time_seconds"],
            "post_first_epoch_steps": warm_steps,
            "post_first_epoch_loop_seconds": warm_seconds,
            "post_first_epoch_steps_per_loop_second": warm_steps / warm_seconds if warm_seconds else None}


def stop_group(process):
    """Terminate only the disposable process group created by this driver."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    # The session leader can exit before spawned collectors; stop those too.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 10
    while True:
        rows = subprocess.check_output(["ps", "-axo", "pgid=,pid=,stat="], text=True, timeout=5)
        live = [int(pid) for group, pid, state in (line.split() for line in rows.splitlines())
                if int(group) == process.pid and not state.startswith("Z")]
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Disposable process group still has live members: {live}")
        time.sleep(.05)


def group_rss_kib(pgid):
    """Sample summed process RSS, which can double-count shared pages."""
    try:
        rows = subprocess.check_output(["ps", "-axo", "pgid=,rss="], text=True, timeout=5)
        return sum(int(rss) for group, rss in (line.split() for line in rows.splitlines())
                   if int(group) == pgid)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def test_results(path, returncode, checkout):
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError) as error:
        raise RuntimeError(f"No usable pytest XML at {path}: {error}; inspect command stdout/stderr") from error
    cases = list(tree.iter("testcase"))
    failures = {}
    for case in cases:
        detail = case.find("failure")
        if detail is None:
            detail = case.find("error")
        if detail is not None:
            name = case.attrib["classname"].replace(".", "/") + ".py"
            if not (Path(checkout) / name).exists():
                name = "tests/" + name
            message = detail.attrib.get("message", "").replace(str(checkout), "<checkout>")
            message = re.sub(r"pytest-\d+", "pytest-<n>", message)
            failures[name + "::" + case.attrib["name"]] = {"kind": detail.tag, "message": message}
    if not cases or returncode != int(bool(failures)):
        raise RuntimeError(f"pytest exit/report mismatch: exit={returncode}, tests={len(cases)}, failures={len(failures)}")
    return {"tests": len(cases), "skipped": sum(c.find("skipped") is not None for c in cases),
            "failures": failures}


def bounded_options(updates):
    if type(updates) is not int or updates < 2:
        raise ValueError("At least two updates are required for an actual interrupted continuation")
    per_epoch = 10 if updates % 10 == 0 else 1
    # Each collector stops after >=500 steps; its final complete episode adds <80.
    return ["--epochs", str(updates // per_epoch), "--updates-per-epoch", str(per_epoch),
            "--batch-steps", "500", "--nprocesses", "4", "--total-steps", str(updates * 4 * 580 + 1),
            "--save-every", "1", "--diagnostics-every", "50"]


def equal_state(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


class Runner:
    def __init__(self, checkout, output, timeout=3600):
        self.checkout, self.output, self.timeout = Path(checkout), Path(output), timeout
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "logs").mkdir()
        self.record = {"version": 1, "status": "running", "scope": "bounded engineering only; no research submission",
                       "host": platform.node(), "platform": platform.platform(), "python": sys.version,
                       "commands": [], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "temporary_directory": os.environ.get("TMPDIR"),
                       "cleanup_scope": "Training interruption waits for its disposable process group. Separate sessions created by pytest helpers are not claimed to belong to that group."}
        self.env = dict(os.environ, HETNET_PYTHON=sys.executable,
                        PYTHONPATH=os.pathsep.join((str(checkout), str(Path(checkout) / "envs"))),
                        PYTHONUNBUFFERED="1", DGLBACKEND="pytorch", **dict.fromkeys(THREADS, "1"))
        self.save()

    def save(self):
        write(self.output / "readiness.json", self.record)

    def run(self, name, command, *, cwd=None, env=None, permitted=(0,), timeout=None,
            interrupt_marker=None, first_update=None, resources=False):
        cwd = Path(cwd or self.checkout)
        command = list(map(str, command))
        measured = command
        resource_path = self.output / f"{name}.time.txt"
        if resources and platform.system() == "Linux" and Path("/usr/bin/time").is_file():
            measured = ["/usr/bin/time", "-v", "-o", str(resource_path), *command]
        row = {"name": name, "command": command, "cwd": str(cwd), "status": "running"}
        self.record["commands"].append(row)
        self.save()
        print(f"Starting {name}", flush=True)
        started, peak, next_sample = time.monotonic(), None, 0.0
        process = None
        try:
            with (self.output / "logs" / f"{name}.out").open("x") as out, (self.output / "logs" / f"{name}.err").open("x") as err:
                process = subprocess.Popen(measured, cwd=cwd, env=env or self.env, stdout=out, stderr=err,
                                           start_new_session=True)
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if first_update is not None and "first_update_seconds" not in row and first_update.exists() and first_update.stat().st_size:
                        row["first_update_seconds"] = elapsed
                    if resources and elapsed >= next_sample:
                        sample = group_rss_kib(process.pid)
                        if sample is not None:
                            peak = max(peak or 0, sample)
                        next_sample = elapsed + 1
                    if interrupt_marker is not None and list(interrupt_marker.glob("*.complete.json")):
                        row["interrupted_after_commit"] = True
                        stop_group(process)
                        break
                    if elapsed >= (timeout or self.timeout):
                        row["timed_out"] = True
                        stop_group(process)
                        raise TimeoutError(f"{name} exceeded {timeout or self.timeout}s")
                    time.sleep(.1)
                row["returncode"] = process.wait()
            if interrupt_marker is not None and not row.get("interrupted_after_commit"):
                raise RuntimeError("Disposable run exited before a checkpoint could be interrupted")
            if interrupt_marker is None and row["returncode"] not in permitted:
                tail = "\n".join((self.output / "logs" / f"{name}.{suffix}").read_text(errors="replace")[-6000:]
                                 for suffix in ("err", "out"))
                raise RuntimeError(f"{name} exited {row['returncode']}; logs/{name}.out and .err\n{tail}")
            row["status"] = "completed"
        except BaseException as error:
            row.update(status="failed", error=repr(error))
            raise
        finally:
            if process is not None:
                stop_group(process)
                row["returncode"] = process.returncode
            row["seconds"] = time.monotonic() - started
            if resources:
                row["sampled_process_group_peak_rss_kib"] = peak
                row["memory_note"] = "Sampled sum may double-count shared pages and miss peaks; retain Slurm accounting."
            self.save()
        print(json.dumps(row, sort_keys=True), flush=True)
        return row


def inspect_run(logical, updates):
    from softrole.programme import accepted_checkpoints
    records, receipt = accepted_checkpoints(logical, return_audit=True)
    if not records or records[-1]["updates"] != updates:
        raise RuntimeError(f"Expected {updates} committed updates in {logical}")
    latest = records[-1]
    if latest["total_steps"] != receipt["total_steps"] or latest["total_episodes"] != receipt["total_episodes"]:
        raise RuntimeError("Committed checkpoint and independently audited ledger differ")
    # accepted_checkpoints calls ledger.committed_checkpoints: strict model load,
    # model/signature and all optimizer tensor finite checks run for every commit.
    return records, receipt


def interruption_check(runner, manifest, options, updates, uninterrupted):
    from softrole.programme import accepted_checkpoints
    import torch
    root = runner.output / "interrupted"
    logical = root / manifest["entries"][0]["relative_output"]
    env = dict(runner.env, SOFTROLE_PROGRAMME_MANIFEST=str(runner.output / "bounded_manifest.json"),
               SOFTROLE_PROGRAMME_RUN_ROOT=str(root))
    command = ["bash", "scripts/softrole_programme.sh", "0"]
    runner.run("interrupt", command + options, env=env, interrupt_marker=logical / "checkpoints", resources=True)
    before = accepted_checkpoints(logical)[-1]
    if before["updates"] >= updates:
        raise RuntimeError("Run completed before interruption; recovery was not exercised")
    parent = Path(before["checkpoint"])
    parent_hash = sha(parent)
    retained = {str(path.relative_to(logical)): sha(path) for path in logical.rglob("*") if path.is_file()}
    runner.run("resume", command + ["--resume"], env=env, resources=True)
    after, receipt = inspect_run(logical, updates)
    if sha(parent) != parent_hash or any(sha(logical / name) != digest for name, digest in retained.items()):
        raise RuntimeError("Continuation changed its parent checkpoint or abandoned files")
    attempts = list((logical / "attempts").glob("*/attempt.json"))
    if len(attempts) != 1:
        raise RuntimeError("Expected one fresh numbered continuation attempt")
    attempt = json.loads(attempts[0].read_text())
    if attempt["parent_checkpoint_sha256"] != parent_hash:
        raise RuntimeError("Continuation attempt references a different parent")
    reference, reference_receipt = inspect_run(uninterrupted, updates)
    left = torch.load(after[-1]["checkpoint"], map_location="cpu", weights_only=False)
    right = torch.load(reference[-1]["checkpoint"], map_location="cpu", weights_only=False)
    for key in ("model_state", "optimizer_state", "signature", "updates", "total_steps", "total_episodes"):
        if not equal_state(left[key], right[key]):
            raise RuntimeError(f"Interrupted continuation differs from uninterrupted execution: {key}")
    if receipt["updates"] != reference_receipt["updates"]:
        raise RuntimeError("Interrupted and uninterrupted episode statistics differ")
    return {"parent": before, "completed": after[-1], "attempt": str(attempts[0]),
            "parent_and_abandoned_files_unchanged": True, "uninterrupted_state_and_statistics_equal": True}


def regression_check(runner, baseline, test_tools):
    probe = "import json,pytest; from importlib.metadata import version; print(json.dumps({n:version(n) for n in ('pytest','iniconfig','pluggy','packaging')}))"
    candidate_env = test_environment(runner.checkout, test_tools, runner.env)
    runner.run("test_tools", [sys.executable, "-c", probe], env=candidate_env)
    installed = json.loads((runner.output / "logs" / "test_tools.out").read_text())
    expected = {"pytest": "8.3.5", "iniconfig": "2.3.0", "pluggy": "1.6.0", "packaging": "26.3"}
    if installed != expected:
        raise RuntimeError(f"Test tools differ from the pinned versions: {installed}")
    command = [sys.executable, "-m", "pytest", "-q", "-o", "faulthandler_timeout=180"]
    xml = runner.output / "candidate.xml"
    row = runner.run("candidate_tests", command + ["--junitxml=" + str(xml)], env=candidate_env, permitted=(0, 1))
    result = test_results(xml, row["returncode"], runner.checkout)
    if result["failures"]:
        # Only corresponding legacy failures need repeat evidence; never waive new tests.
        xml = runner.output / "baseline.xml"
        plugin = baseline_timeout_plugin(runner.output, baseline)
        baseline_env = test_environment(baseline, test_tools, runner.env)
        baseline_env["PYTHONPATH"] += os.pathsep + str(plugin.parent)
        row = runner.run("baseline_affected_tests", command + ["-p", plugin.stem, "--junitxml=" + str(xml), *result["failures"]],
                         cwd=baseline, env=baseline_env, permitted=(0, 1))
        result["baseline_test_adapter"] = {"path": str(plugin), "sha256": sha(plugin),
            "scope": "Only the unchanged baseline gradient_probe fixture and shared_gradient_probe.py communicate(timeout=85) use 180s; internal probe timeout remains 60s; restored after fixture setup."}
        baseline_result = test_results(xml, row["returncode"], baseline)
        if result["failures"] != baseline_result["failures"]:
            raise RuntimeError("Candidate regression failures differ from the unchanged same-environment baseline")
        result["baseline_matched_failures"] = baseline_result
    return result


def execute(args):
    checkout = args.checkout.resolve()
    if checkout != ROOT:
        raise ValueError("--checkout must be the checkout containing this readiness driver")
    baseline = args.baseline.resolve()
    if baseline == checkout:
        raise ValueError("Baseline and candidate must be separate checkouts")
    if args.test_tools is not None and not args.test_tools.is_dir():
        raise ValueError("--test-tools must name an existing directory")
    sys.path[:0] = [str(checkout), str(checkout / "envs")]
    runner = Runner(checkout, args.output.resolve(), args.timeout_seconds)
    try:
        from importlib.metadata import version
        import torch
        import softrole
        if not Path(softrole.__file__).resolve().is_relative_to(checkout):
            raise RuntimeError("Imported SoftRole belongs to a different checkout")
        from softrole.train import source_snapshot
        from softrole.programme import CAPCON_PROFILE, prepare, validate_selection
        torch.set_num_threads(1)
        source = source_snapshot()
        runner.record.update(source_sha256=source["sha256"], git_commit=source["git_commit"],
                             driver_sha256=sha(__file__), updates_per_case=args.updates,
                             full_readiness_requested=not args.skip_tests and args.updates >= 60,
                             packages={name: version(name) for name in ("torch", "numpy", "dgl", "torchdata")})
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("The pinned runtime requires Python 3.12")
        for name, expected in {"torch": "2.2.1", "numpy": "1.26.4", "dgl": "2.1.0", "torchdata": "0.7.1"}.items():
            if runner.record["packages"][name].split("+")[0] != expected:
                raise RuntimeError(f"Unexpected runtime {name}: {runner.record['packages'][name]}")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=baseline, text=True).strip()
        if commit != BASELINE_COMMIT or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=baseline).strip():
            raise RuntimeError("Baseline must be clean unchanged 644f8f1")
        runner.record["baseline_commit"] = commit
        selection_path = args.selection.resolve() if args.selection else runner.output / "lr_selection.json"
        if args.evidence:
            if not args.inventory_sha256:
                raise ValueError("--evidence requires an independently pinned --inventory-sha256")
            runner.run("verify_evidence", [sys.executable, "analysis/capcom_programme_2026-10-12/integration/stage_evidence.py",
                       "verify", "--bundle", args.evidence.resolve(), "--inventory-sha256", args.inventory_sha256,
                       "--output", selection_path])
        selection = json.loads(selection_path.read_text())
        validate_selection(selection)
        if selection.get("bounded_check") or selection["selected_lr"] != .001:
            raise ValueError("Readiness requires the certified research Adam 0.001 selection")
        runner.record["selection"] = {"path": str(selection_path), "sha256": sha(selection_path),
                                      "selected_lr": .001, "bounded_check": False}
        if args.skip_tests:
            runner.record["regressions"] = {"skipped_explicitly": True}
        else:
            runner.record["regressions"] = regression_check(runner, baseline, args.test_tools)
        manifest_path = runner.output / "bounded_manifest.json"
        manifest = prepare(selection_path, manifest_path, diagnostics_every=50, bounded=True, profile=CAPCON_PROFILE)
        if len(manifest["entries"]) != 12 or not manifest["bounded_check"]:
            raise RuntimeError("Expected the bounded twelve-entry CapCon profile")
        env = dict(runner.env, SOFTROLE_PROGRAMME_MANIFEST=str(manifest_path),
                   SOFTROLE_PROGRAMME_RUN_ROOT=str(runner.output / "throughput"))
        for index in range(12):
            runner.run(f"dry_run_{index:02}", ["bash", "scripts/softrole_programme.sh", str(index), "--dry-run"], env=env)
            observed = json.loads((runner.output / "logs" / f"dry_run_{index:02}.out").read_text())
            if observed != manifest["entries"][index]["config"]:
                raise RuntimeError("Dry run differs from the locked profile")
        options = bounded_options(args.updates)
        runner.record["throughput"] = {}
        for index in INDICES:
            logical = runner.output / "throughput" / manifest["entries"][index]["relative_output"]
            row = runner.run(f"throughput_{index}", ["bash", "scripts/softrole_programme.sh", str(index), *options],
                             env=env, first_update=logical / "updates.jsonl", resources=True)
            checkpoints, _ = inspect_run(logical, args.updates)
            metrics = [json.loads(line) for line in (logical / "metrics.jsonl").read_text().splitlines()]
            timing = epoch_timing(metrics)
            epoch_seconds = timing["epoch_loop_seconds"]
            runner.record["throughput"][str(index)] = {"checkpoint": checkpoints[-1],
                "committed_checkpoints_validated": len(checkpoints),
                "finite_model_and_optimizer_tensors": True,
                "checkpoint_validation": "programme.accepted_checkpoints -> ledger.committed_checkpoints; all committed model/optimizer tensors, signatures and independently audited counts",
                "command_seconds": row["seconds"], "first_update_seconds": row.get("first_update_seconds"),
                **timing,
                "setup_checkpoint_and_teardown_seconds": max(0., row["seconds"] - epoch_seconds),
                "total_steps_per_command_second": checkpoints[-1]["total_steps"] / row["seconds"],
                "timing_note": "First update includes process startup, source/model setup and first collection/update. First epoch includes collector startup; post-first-epoch throughput excludes that warmup. All epoch loops include collection/learning, diagnostics and episode logging; residual time includes setup, checkpoint writes and teardown. Pure process startup is not isolated.",
                "sampled_process_group_peak_rss_kib": row["sampled_process_group_peak_rss_kib"]}
            runner.save()
        uninterrupted = runner.output / "throughput" / manifest["entries"][0]["relative_output"]
        runner.record["interrupted_continuation"] = interruption_check(runner, manifest, options, args.updates, uninterrupted)
        if source_snapshot()["sha256"] != source["sha256"]:
            raise RuntimeError("Candidate source changed during readiness")
        runner.record["status"] = "passed" if not args.skip_tests and args.updates >= 60 else "passed_reduced_engineering_check"
        runner.record["resource_review"] = "Review measured time/RSS and retain sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocCPUS before resource approval."
    except BaseException as error:
        runner.record.update(status="failed", error=repr(error))
        raise
    finally:
        runner.save()
        files = {str(p.relative_to(runner.output)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(runner.output.rglob("*")) if p.is_file() and p != runner.output / "artifact_manifest.json"}
        write(runner.output / "artifact_manifest.json", {"version": 1, "files": files})
    print(f"Readiness results: {runner.output}/readiness.json; no research manifest or job created.")
    return runner.record


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkout", type=Path, default=ROOT)
    result.add_argument("--baseline", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    evidence = result.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--selection", type=Path)
    evidence.add_argument("--evidence", type=Path)
    result.add_argument("--inventory-sha256")
    result.add_argument("--test-tools", type=Path)
    result.add_argument("--updates", type=int, default=60, help="Updates per four-collector case; <60 is a reduced engineering check")
    result.add_argument("--timeout-seconds", type=int, default=3600, help="Per-command deadline, including full pytest")
    result.add_argument("--skip-tests", action="store_true", help="Explicit reduced engineering check; cannot establish full readiness")
    return result


def main():
    options = parser().parse_args()
    bounded_options(options.updates)
    if options.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be positive")
    def terminated(signum, frame):
        raise InterruptedError(f"Readiness interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, terminated)
    execute(options)


if __name__ == "__main__":
    main()
