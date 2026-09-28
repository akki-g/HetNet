"""Run the bounded scientific Gate A checks without submitting any cluster job.

Plan: uv run --locked --python 3.12 python -m hetnet_ext.gate_a --plan
Run:  uv run --locked --python 3.12 python -m hetnet_ext.gate_a \
          --output runs/gate_a/<unique-name> --run-smokes \
          --clean-environment-evidence <clean-sync-evidence.json> \
          --allowed-upstream-commit <reviewed-commit> [repeat for each deviation]

The gradient contract is checked before expensive smokes. A failed or absent
check always leaves passed=false; partial engineering evidence is retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
COMPOSITIONS = ((2, 1), (3, 3), (4, 6), (3, 1), (2, 2))
REQUIRED_CHECKS = ("environment", "clean_lock_environment", "baseline_diff", "smoke_all_compositions", "shared_weights", "gradient_storage",
                   "fresh_gradient_aggregation", "determinism", "cross_composition_load", "grid_and_scripts", "unit_contracts")


def smoke_plan():
    runs = [{"id": f"{p}P{a}A_np{n}", "nfriendly_P": p, "nfriendly_A": a,
             "nprocesses": n, "seed": 0, "relative_dir": f"smokes/{p}P{a}A_np{n}"}
            for p, a in COMPOSITIONS for n in (1, 4)]
    runs.extend([
        {"id": "2P1A_np1_repeat", "nfriendly_P": 2, "nfriendly_A": 1, "nprocesses": 1,
         "seed": 0, "relative_dir": "smokes/2P1A_np1_repeat"},
        {"id": "2P1A_np1_other_seed", "nfriendly_P": 2, "nfriendly_A": 1, "nprocesses": 1,
         "seed": 1, "relative_dir": "smokes/2P1A_np1_other_seed"},
    ])
    lower = sum(3 * 10 * 40 * r["nprocesses"] for r in runs)
    upper = sum(3 * 10 * 59 * r["nprocesses"] for r in runs)
    return {"schema_version": 1, "purpose": "engineering evidence only; not research outcomes",
            "runs": runs, "epochs_per_run": 3, "epoch_size": 10, "batch_size": 40, "max_steps": 20,
            "maximum_concurrent_processes": 4, "joint_environment_steps_bounds": [lower, upper],
            "optimizer_update_calls": len(runs) * 30,
            "runtime_note": "Wall time is unmeasured. Runs are sequential; no cluster job is submitted."}


def smoke_command(run: dict, directory: Path):
    return [sys.executable, str(ROOT / "main.py"), "--env_name", "predator_capture",
            "--nfriendly_P", str(run["nfriendly_P"]), "--nfriendly_A", str(run["nfriendly_A"]),
            "--nprocesses", str(run["nprocesses"]), "--num_epochs", "3", "--hid_size", "128",
            "--detach_gap", "5", "--lrate", "0.0001", "--dim", "5", "--batch_size", "40",
            "--max_steps", "20", "--hetgat", "--hetgat_a2c", "--seed", str(run["seed"]),
            "--save_every", "1", "--save_dir", str(directory / "checkpoints"),
            "--experiment_name", run["id"], "--metrics_file", str(directory / "metrics.jsonl")]


def run_command(command: list[str], log_path: Path, *, timeout: float, env: dict):
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        log.write(json.dumps({"command": command, "cwd": str(ROOT)}) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=10)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
    return {"command": command, "log": str(log_path), "exit_code": code,
            "timed_out": timed_out, "wall_time_seconds": time.monotonic() - started,
            "status": "PASS" if code == 0 and not timed_out else "FAIL"}


def pytest_case_results(path: Path):
    if not path.exists():
        return {}
    tree = ET.parse(path)
    return {case.attrib["name"]: not any(case.find(kind) is not None
                                        for kind in ("failure", "error", "skipped"))
            for case in tree.iter("testcase")}



def verify_clean_environment(path: Path):
    """Require preserved successful clean-sync evidence for the current lock."""
    data = json.loads(path.read_text())
    if data.get("passed") is not True or data.get("status") != "PASS":
        raise ValueError("Clean environment evidence does not report a successful sync/import check")
    for key, filename in (("lock_sha256", "uv.lock"), ("pyproject_sha256", "pyproject.toml")):
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if data.get(key) != actual:
            raise ValueError(f"Clean environment evidence is stale: {filename} hash differs")
    expected = {"torch": "2.2.1", "dgl": "2.1.0", "gym": "0.26.2", "torchdata": "0.7.1", "numpy": "1.26.4"}
    versions = data.get("runtime", {}).get("versions", {})
    for package, version in expected.items():
        if versions.get(package, "").split("+")[0] != version:
            raise ValueError(f"Clean environment has wrong/missing {package} version")
    if not data.get("runtime", {}).get("python", "").startswith("3.12."):
        raise ValueError("Clean environment proof is not Python 3.12")
    if data.get("sync_exit_code") != 0 or data.get("imports_exit_code") != 0:
        raise ValueError("Clean sync/import process did not complete successfully")
    command = data.get("command", [])
    if command[1:] != ["sync", "--locked", "--python", "3.12"]:
        raise ValueError("Clean environment evidence lacks the required locked Python 3.12 sync command")
    if Path(data.get("cwd", "")).absolute() != ROOT:
        raise ValueError("Clean environment evidence was produced outside this checkout")
    venv = Path(data.get("venv", "")).absolute()
    if venv == ROOT / ".venv" or venv.parent != ROOT:
        raise ValueError("Clean environment proof must identify a separate checkout-local virtual environment")
    if Path(data.get("runtime", {}).get("executable", "")).absolute() != venv / "bin" / "python":
        raise ValueError("Pinned imports were not executed in the clean environment")
    required_logs = {"uv_sync.log", "imports.stdout.log", "imports.stderr.log"}
    if not required_logs <= data.get("logs", {}).keys():
        raise ValueError("Clean environment proof omits preserved logs")
    for name in required_logs:
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != data["logs"][name]:
            raise ValueError(f"Clean environment log hash mismatch: {name}")
    if "Creating virtual environment at:" not in (path.parent / "uv_sync.log").read_text():
        raise ValueError("Locked sync log does not show creation of a fresh virtual environment")
    if json.loads((path.parent / "imports.stdout.log").read_text()) != data["runtime"]:
        raise ValueError("Reported runtime differs from preserved import stdout")
    return {"status": "PASS", "evidence": str(path.resolve()),
            "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "runtime": data["runtime"], "lock_sha256": data["lock_sha256"]}


def verify_upstream_inventory(allowed_commits: list[str], output: Path):
    """Fail closed if any original tracked file has an unapproved committed delta."""
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT)

    baseline = git("rev-parse", "upstream-bff9f7f^{commit}").decode().strip()
    expected_baseline = "bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec"
    if baseline != expected_baseline:
        raise ValueError("Baseline tag no longer identifies the audited upstream revision")
    git("merge-base", "--is-ancestor", baseline, "HEAD")
    originals = [p for p in git("ls-tree", "-r", "--name-only", baseline).decode().splitlines() if p]
    allowed = {git("rev-parse", commit + "^{commit}").decode().strip() for commit in allowed_commits}
    touching = git("rev-list", "--reverse", f"{baseline}..HEAD", "--", *originals).decode().splitlines()
    dirty = git("diff", "--name-only", "HEAD", "--", *originals).decode().splitlines()
    patch = git("diff", "--binary", baseline, "HEAD", "--", *originals)
    (output / "upstream_diff.patch").write_bytes(patch)
    records = [{"commit": commit,
                "subject": git("show", "-s", "--format=%s", commit).decode().strip(),
                "approved_by_cli_manifest": commit in allowed,
                "changed_original_files": git("diff-tree", "--no-commit-id", "--name-only", "-r", commit,
                                               "--", *originals).decode().splitlines()}
               for commit in touching]
    result = {"schema_version": 1, "baseline": baseline, "allowed_commits": sorted(allowed),
              "modifying_commits": records, "uncommitted_original_files": dirty,
              "patch_sha256": hashlib.sha256(patch).hexdigest(),
              "passed": not dirty and all(row["approved_by_cli_manifest"] for row in records)}
    (output / "upstream_diff_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise ValueError("Baseline diff contains unlisted commits or uncommitted original-file changes; inspect upstream_diff_inventory.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--plan", action="store_true", help="Print smoke/resource plan; do not execute checks")
    parser.add_argument("--clean-environment-evidence", type=Path, help="Successful clean uv sync/import evidence JSON for current lock")
    parser.add_argument("--allowed-upstream-commit", action="append", default=[],
                        help="Explicit reviewed deviation commit; repeat for every commit modifying original files")
    parser.add_argument("--run-smokes", action="store_true", help="Run smokes after prerequisite contracts pass")
    parser.add_argument("--timeout-per-smoke", type=float, default=900,
                        help="Maximum seconds for one bounded smoke run (default 900)")
    args = parser.parse_args(argv)
    plan = smoke_plan()
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required unless --plan is used")
    if args.timeout_per_smoke <= 0:
        parser.error("--timeout-per-smoke must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "passed": False, "status": "INCOMPLETE",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "checks": {name: False for name in REQUIRED_CHECKS},
              "check_statuses": {name: "NOT_RUN" for name in REQUIRED_CHECKS},
              "check_records": [], "artifacts": [],
              "scope": "Gate A engineering evidence only; no cluster submission or Phase B"}
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONHASHSEED="0", GATE_A_OUTPUT=str(output))
    try:
        from hetnet_ext.grid import code_sha256
        report["code_sha256"] = code_sha256(ROOT)
        report["uv_lock_sha256"] = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        (output / "smoke_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
        report["artifacts"].append(str(output / "smoke_manifest.json"))
        print(f"Gate A: {len(plan['runs'])} bounded smokes planned; joint-step bounds {plan['joint_environment_steps_bounds']}", flush=True)
        if args.clean_environment_evidence is None:
            report["check_statuses"]["clean_lock_environment"] = "FAIL"
            raise ValueError("--clean-environment-evidence is required for Gate A")
        report["check_statuses"]["clean_lock_environment"] = "FAIL"
        clean = verify_clean_environment(args.clean_environment_evidence.resolve())
        report["checks"]["clean_lock_environment"] = True
        report["check_statuses"]["clean_lock_environment"] = "PASS"
        report["check_records"].append({"name": "clean_lock_environment", **clean})
        report["artifacts"].append(clean["evidence"])
        report["check_statuses"]["baseline_diff"] = "FAIL"
        baseline = verify_upstream_inventory(args.allowed_upstream_commit, output)
        report["checks"]["baseline_diff"] = baseline["passed"]
        report["check_statuses"]["baseline_diff"] = "PASS"
        report["artifacts"].extend([str(output / "upstream_diff_inventory.json"), str(output / "upstream_diff.patch")])
        print("Gate A: checking runtime, grid and script contracts first", flush=True)
        preflight_xml = output / "preflight.xml"
        record = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_gate_a_contracts.py",
                              "-k", "runtime_versions or grid_is_requested_bijection or singleton_p_is_rejected or slurm_scripts_parse",
                              f"--junitxml={preflight_xml}"], output / "preflight.log", timeout=180, env=environment)
        record["name"] = "preflight_contracts"
        report["check_records"].append(record)
        cases = pytest_case_results(preflight_xml)
        report["checks"]["environment"] = cases.get("test_runtime_versions", False)
        report["checks"]["grid_and_scripts"] = all(cases.get(name, False) for name in (
            "test_grid_is_requested_bijection", "test_singleton_p_is_rejected", "test_slurm_scripts_parse"))
        for name in ("environment", "grid_and_scripts"):
            report["check_statuses"][name] = "PASS" if report["checks"][name] else "FAIL"
        report["artifacts"].extend([str(preflight_xml), str(output / "preflight.log")])
        if record["status"] != "PASS":
            report["status"] = "FAIL"
            report["reason"] = "Runtime/configuration/script contract failed; smokes not launched"
            return 1

        print("Gate A: checking recorder and infrastructure unit contracts", flush=True)
        unit_xml = output / "unit_contracts.xml"
        record = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_recording.py", "tests/test_grid_slurm.py",
                              "tests/test_local_setup.py", "tests/test_local_job.py",
                              "tests/test_progress.py", "tests/test_submit_run1.py",
                              f"--junitxml={unit_xml}"], output / "unit_contracts.log", timeout=180, env=environment)
        record["name"] = "unit_contracts"
        report["check_records"].append(record)
        cases = pytest_case_results(unit_xml)
        report["checks"]["unit_contracts"] = record["status"] == "PASS" and bool(cases) and all(cases.values())
        report["check_statuses"]["unit_contracts"] = "PASS" if report["checks"]["unit_contracts"] else "FAIL"
        report["artifacts"].extend([str(unit_xml), str(output / "unit_contracts.log")])
        if not report["checks"]["unit_contracts"]:
            report["status"] = "FAIL"
            report["reason"] = "Recorder/infrastructure unit contract failed; smokes not launched"
            return 1

        print("Gate A: checking three-update shared-weight and fresh-gradient contracts", flush=True)
        gradient_jobs = [
            ("gradient_probe", "tests/helpers/shared_gradient_probe.py", [], 90),
            ("actual_gradient_2P1A", "tests/helpers/hetnet_gradient_probe.py",
             ["--batch-size", "4", "--horizon", "4", "--num-p", "2", "--num-a", "1", "--timeout-seconds", "120"], 150),
            ("actual_gradient_4P6A", "tests/helpers/hetnet_gradient_probe.py",
             ["--batch-size", "4", "--horizon", "4", "--num-p", "4", "--num-a", "6", "--timeout-seconds", "120"], 150),
        ]
        probes = []
        for name, helper, extra, timeout in gradient_jobs:
            print(f"Gate A: running {name}", flush=True)
            probe_path = output / f"{name}.json"
            log_path = output / f"{name}.log"
            record = run_command([sys.executable, helper, "--output", str(probe_path), "--updates", "3", *extra],
                                 log_path, timeout=timeout, env=environment)
            record["name"] = name
            report["check_records"].append(record)
            probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
            probes.append((record, probe))
            report["artifacts"].extend([str(probe_path), str(log_path)])
            report["artifacts"].extend(str(path) for path in sorted(output.glob(f"{name}*.npz")))
        for name in ("shared_weights", "gradient_storage", "fresh_gradient_aggregation"):
            report["checks"][name] = all(probe.get(f"{name}_passed") is True for _, probe in probes)
        for name in ("shared_weights", "gradient_storage", "fresh_gradient_aggregation"):
            report["check_statuses"][name] = "PASS" if report["checks"][name] else "FAIL"
        if not (all(record["status"] == "PASS" and probe.get("status") == "PASS" and probe.get("gate_passed") is True
                    for record, probe in probes)
                and all(report["checks"][name] for name in ("shared_weights", "gradient_storage", "fresh_gradient_aggregation"))):
            report["status"] = "FAIL"
            report["reason"] = "Shared-weight/gradient contract failed; expensive smokes were not launched"
            return 1
        if not args.run_smokes:
            report["reason"] = "Prerequisites checked only; --run-smokes was not supplied"
            return 1

        smoke_failed = False
        for row in plan["runs"]:
            directory = output / row["relative_dir"]
            directory.mkdir(parents=True, exist_ok=False)
            print(f"Gate A: running {row['id']} (3 epochs, 10 updates/epoch, batch 40, horizon 20)", flush=True)
            record = run_command(smoke_command(row, directory), directory / "stdout.log",
                                 timeout=args.timeout_per_smoke, env=environment)
            record["name"] = row["id"]
            report["check_records"].append(record)
            report["artifacts"].append(str(directory))
            if record["status"] != "PASS":
                smoke_failed = True
                print(f"Gate A: {row['id']} failed; preserving log and stopping remaining smokes", flush=True)
                break
        if smoke_failed:
            report["check_statuses"]["smoke_all_compositions"] = "FAIL"
            report["status"] = "FAIL"
            report["reason"] = "A smoke process failed; remaining experiments were not launched"
            return 1

        artifact_xml = output / "artifact_contracts.xml"
        record = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_gate_a_contracts.py",
                              "-k", "smoke_artifacts or determinism or cross_composition",
                              f"--junitxml={artifact_xml}"], output / "artifact_contracts.log", timeout=180, env=environment)
        record["name"] = "artifact_contracts"
        report["check_records"].append(record)
        cases = pytest_case_results(artifact_xml)
        report["checks"]["smoke_all_compositions"] = cases.get("test_smoke_artifacts_are_finite_and_complete", False)
        report["checks"]["determinism"] = cases.get("test_determinism_matches_non_timing_metrics_and_signatures", False)
        report["checks"]["cross_composition_load"] = cases.get("test_cross_composition_load_preserves_every_tensor", False)
        report["artifacts"].extend([str(artifact_xml), str(output / "artifact_contracts.log"),
                                     str(output / "determinism_evidence.json"),
                                     str(output / "cross_composition_signatures.json")])
        for name in ("smoke_all_compositions", "determinism", "cross_composition_load"):
            report["check_statuses"][name] = "PASS" if report["checks"][name] else "FAIL"
        report["passed"] = all(report["checks"].values()) and record["status"] == "PASS"
        report["code_sha256_after"] = code_sha256(ROOT)
        report["source_unchanged"] = report["code_sha256_after"] == report["code_sha256"]
        if not report["source_unchanged"]:
            report["passed"] = False
            report["reason"] = "Scientific code changed during Gate A; preserve this attempt and rerun from a fixed revision"
        report["status"] = "PASS" if report["passed"] else "FAIL"
        return 0 if report["passed"] else 1
    except Exception as exc:
        report["status"] = "ERROR"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        return 2
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["not_passed"] = [name for name, passed in report["checks"].items() if not passed]
        (output / "gate_a_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Gate A {report['status']}: {output / 'gate_a_report.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
