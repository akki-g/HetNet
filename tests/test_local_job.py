"""Bounded local-launcher contracts: fake processes/artifacts, never HetNet training."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from hetnet_ext import local_job as job
from hetnet_ext.train_job import GATE_CHECKS


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + "\n")


def sample(*args, **kwargs):
    return {"elapsed_seconds": args[1], "group_rss_kib_sum": 64, "process_count": 1,
            "members": [], "sampling_errors": [], "system_swap_usage_raw": "fixture", "vm_stat_raw": "fixture"}


def fixture_artifacts(directory, plan):
    import torch
    from hetnet_ext.signatures import model_signature

    # Deliberately preserve the original architecture's mixed float32/float64 idea.
    model = torch.nn.Module()
    model.register_parameter("attention", torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32)))
    model.register_parameter("dense", torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64)))
    model.register_buffer("counter", torch.tensor(0, dtype=torch.int64))
    signature = model_signature(model)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoints" / "fixture" / "run1" / "model_ep20.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"seed": 0, "policy_net": model.state_dict()}, checkpoint)
    write_json(Path(str(checkpoint) + ".signature.json"), signature)
    write_json(directory / "initial_signature.json", signature)
    metrics = [{"epoch": i, "steps": 20000, "episodes": 250, "total_steps": i * 20000,
                "total_episodes": i * 250, "wall_time_seconds": 0.000001, "success_rate": 0,
                "steps_taken": 80, "reward_per_agent": [0, 0, 0], "policy_loss": -1, "value_loss": 1}
               for i in range(1, 21)]
    (directory / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    (directory / "epoch_signatures.jsonl").write_text("".join(
        json.dumps({"epoch": i, "signature": signature}) + "\n" for i in range(1, 21)))
    write_json(directory / "checkpoint_records.jsonl", {"epoch": 20, "path": str(checkpoint.resolve()),
               "bytes": checkpoint.stat().st_size, "wall_time_seconds": 0, "parameter_sha256": signature["sha256"]})
    resolved = {**job.EXPECTED_RECIPE, "num_epochs": 20, "seed": 0, "nfriendly_P": 2, "nfriendly_A": 1,
                "use_binary": False, "use_cuda": False, "commnet": False, "hetcomm": False, "ic3net": False,
                "eval": False, "random": False, "load": "", "comm_range_P": -1, "comm_range_A": -1,
                "lossy_comm": False, "gamma": 1.0, "vision": 2, "A_vision": -1, "tensor_obs": False,
                "nenemies": 1, "moving_prey": False, "no_stay": False, "mode": "mixed", "enemy_comm": False,
                "second_reward_scheme": False, "resolved_model": {"class": "hetgat.uavnet.UAVNetA2CEasy",
                "per_class_critic": True, "with_two_state": True}}
    write_json(directory / "resolved_args.json", resolved)
    return model, signature, checkpoint


class PlanningTests(unittest.TestCase):
    def test_dry_run_without_site_packages_and_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "absent"
            command = [sys.executable, "-S", "-m", "hetnet_ext.local_job", "--index", "1", "--gate-a", "absent.json",
                       "--output-root", str(root), "--max-wall-seconds", "7200", "--dry-run"]
            result = subprocess.run(command, cwd=job.ROOT, capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["entry"]["composition"], "4P6A")
            for flag, value in {"--num_epochs": "20", "--nprocesses": "4", "--epoch_size": "10",
                                "--batch_size": "500", "--max_steps": "80", "--seed": "0"}.items():
                self.assertEqual(plan["command"][plan["command"].index(flag) + 1], value)
            self.assertFalse(root.exists())
            self.assertNotIn("PYTHONHASHSEED", plan["thread_settings"])
            self.assertNotIn("--use_cuda", plan["command"])
            for value in ("0", "-1", "nan", "inf"):
                invalid = list(command)
                invalid[invalid.index("--max-wall-seconds") + 1] = value
                self.assertNotEqual(subprocess.run(invalid, cwd=job.ROOT, capture_output=True).returncode, 0)

    def test_recipe_or_native_host_drift_is_rejected(self):
        config = job.read_json(job.DEFAULT_GRID)
        config["recipe"]["nprocesses"] = 1
        with patch.object(job, "read_json", return_value=config):
            with self.assertRaisesRegex(ValueError, "unchanged"):
                job.calibration_plan(0, Path("unused"))
        for system, machine in (("Linux", "aarch64"), ("Darwin", "x86_64")):
            with patch.object(job.platform, "system", return_value=system), patch.object(job.platform, "machine", return_value=machine):
                with self.assertRaisesRegex(ValueError, "native"):
                    job.validate_native_host()

    def test_per_user_lock_rejects_an_independent_process_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lock"
            program = "from pathlib import Path; import sys; from hetnet_ext import local_job as j; j.user_lock_path=lambda:Path(sys.argv[1]);\nwith j.single_job_lock(): print('acquired')"
            with patch.object(job, "user_lock_path", return_value=lock):
                with job.single_job_lock():
                    blocked = subprocess.run([sys.executable, "-c", program, str(lock)], cwd=job.ROOT,
                                             capture_output=True, text=True)
                    self.assertNotEqual(blocked.returncode, 0)
                    self.assertIn("Another local calibration", blocked.stderr)
                released = subprocess.run([sys.executable, "-c", program, str(lock)], cwd=job.ROOT,
                                          capture_output=True, text=True, check=True)
                self.assertIn("acquired", released.stdout)
                self.assertTrue(lock.is_file())  # The shared lock inode is never unlinked.


class ChildTests(unittest.TestCase):
    def test_success_logs_bytes_and_timeout_kills_descendant_after_leader_exit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(job, "sample_resources", side_effect=sample), redirect_stdout(io.StringIO()):
            root = Path(tmp)
            good = root / "success"
            good.mkdir()
            result = job.run_child([sys.executable, "-c", "print('fake child only')"], good, 5, time.monotonic(), dict(os.environ))
            self.assertEqual(result["returncode"], 0)
            self.assertIsNone(result["stop_reason"])
            self.assertIn("fake child only", (good / "stdout.log").read_text())
            bad = root / "timeout"
            bad.mkdir()
            pid_path = bad / "grandchild.pid"
            grandchild = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(%r).write_text(str(os.getpid())); time.sleep(60)" % str(pid_path)
            leader = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(.05)" % grandchild
            result = job.run_child([sys.executable, "-c", leader], bad, 0.5, time.monotonic(), dict(os.environ))
            self.assertEqual(result["stop_reason"], "timeout")
            self.assertEqual(result["returncode"], 0)  # A zero-exit leader cannot hide its live child.
            self.assertLess(result["child_wall_time_seconds"], 5)
            pid = int(pid_path.read_text())
            state = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
            self.assertTrue(state.returncode != 0 or state.stdout.strip().startswith("Z"), state.stdout)

    def test_sigterm_marks_failure_and_cleans_group(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(job, "sample_resources", side_effect=sample), redirect_stdout(io.StringIO()):
            program = "import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(60)"
            old = signal.getsignal(signal.SIGTERM)
            result = job.run_child([sys.executable, "-c", program], Path(tmp), 5, time.monotonic(), dict(os.environ))
            self.assertEqual(result["stop_reason"], "signal")
            self.assertEqual(result["signals"], [signal.SIGTERM])
            self.assertEqual(signal.getsignal(signal.SIGTERM), old)
            self.assertFalse(job.group_exists(result["process_group"]))


class ArtifactTests(unittest.TestCase):
    def test_mixed_dtype_checkpoint_passes_and_actual_tensor_mutation_fails(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = job.calibration_plan(0, root)
            directory = Path(plan["directory"])
            model, signature, checkpoint = fixture_artifacts(directory, plan)
            metadata = job.signature_metadata(signature)
            verified = job.verify_completion(directory, plan, metadata)
            self.assertEqual(verified["total_steps"], 400000)
            self.assertEqual({entry["dtype"] for entry in signature["parameters"]}, {"torch.float32", "torch.float64"})
            with torch.no_grad():
                model.dense.add_(1)
            torch.save({"seed": 0, "policy_net": model.state_dict()}, checkpoint)
            with self.assertRaisesRegex(ValueError, "tensor values"):
                job.verify_completion(directory, plan, metadata)

    def test_global_dtype_cast_cannot_be_hidden_by_consistent_new_signatures(self):
        from hetnet_ext.signatures import model_signature
        with tempfile.TemporaryDirectory() as tmp:
            plan = job.calibration_plan(0, Path(tmp))
            directory = Path(plan["directory"])
            model, original, _ = fixture_artifacts(directory, plan)
            wrong = model_signature(model.float())
            write_json(directory / "initial_signature.json", wrong)
            with self.assertRaisesRegex(ValueError, "verified original Gate A model"):
                job.verify_completion(directory, plan, job.signature_metadata(original))

    def test_missing_epochs_bad_fresh_accounting_and_endpoint_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = job.calibration_plan(0, Path(tmp))
            directory = Path(plan["directory"])
            _, signature, _ = fixture_artifacts(directory, plan)
            metadata = job.signature_metadata(signature)
            path = directory / "metrics.jsonl"
            original = path.read_text()
            path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                job.verify_completion(directory, plan, metadata)
            rows = [json.loads(line) for line in original.splitlines()]
            rows[-1]["total_steps"] += 1
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "accounting"):
                job.verify_completion(directory, plan, metadata)
            path.write_text(original)
            args = job.read_json(directory / "resolved_args.json")
            args["nfriendly_A"] = 6
            write_json(directory / "resolved_args.json", args)
            with self.assertRaisesRegex(ValueError, "authorized CPU endpoint"):
                job.verify_completion(directory, plan, metadata)

    def test_gate_contract_requires_artifact_and_strict_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = job.calibration_plan(0, root)
            _, signature, _ = fixture_artifacts(Path(plan["directory"]), plan)
            path = root / "cross_composition_signatures.json"
            data = {"schema_version": 1, "compositions": {"2P1A": {"strict_load": True, "signature": signature}}}
            write_json(path, data)
            gate = {"artifacts": [str(path.resolve())]}
            contract = job.gate_model_contract(root / "gate.json", gate, "2P1A")
            self.assertEqual(contract["metadata"], job.signature_metadata(signature))
            data["compositions"]["2P1A"]["strict_load"] = False
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "strict"):
                job.gate_model_contract(root / "gate.json", gate, "2P1A")


class ProvenanceTests(unittest.TestCase):
    def test_stale_gate_never_launches_or_creates_an_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = job.calibration_plan(0, root / "absent")
            gate_path = root / "gate.json"
            write_json(gate_path, {"schema_version": 1, "passed": True, "code_sha256": "old-code",
                       "uv_lock_sha256": job.file_sha256(job.ROOT / "uv.lock"), "checks": {k: True for k in GATE_CHECKS}})
            with patch.object(job, "validate_native_host"), patch.object(job, "run_child") as launch:
                with self.assertRaisesRegex(ValueError, "stale"):
                    job.execute(plan, gate_path, 5)
                launch.assert_not_called()
            self.assertFalse(Path(plan["directory"]).exists())

    def test_target_gate_rejects_other_host_and_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "evidence.json"
            write_json(evidence, {})
            runtime = {"executable": "/checkout/.venv/bin/python", "prefix": "/checkout/.venv", "base_prefix": "/python",
                       "interpreter_sha256": "python", "python": "3.12.fixture", "system": "Darwin", "machine": "arm64",
                       "hostname": "target", "versions": {"torch": "2.2.1"}, "module_files": {"torch": "/venv/torch.py"},
                       "module_sha256": {"torch": "module"}}
            hardware = {"memory_bytes": 24, "physical_cores": 12, "logical_cores": 12, "cpu_model": "actual-chip",
                        "sysctl": {"hw.model": {"stdout": "actual-model"}, "hw.optional.arm64": {"stdout": "1"}}}
            proof = {**runtime, "hardware_identity": {"model_identifier": "actual-model", "memory_bytes": 24,
                     "physical_cpus": 12, "logical_cpus": 12, "cpu_brand": "actual-chip", "arm64_capable": 1}}
            gate = {"check_records": [{"name": "clean_lock_environment", "evidence": str(evidence),
                    "evidence_sha256": job.file_sha256(evidence), "runtime": proof}]}
            with patch("hetnet_ext.gate_a.verify_clean_environment", return_value={"runtime": proof}):
                job.validate_target_gate(gate, hardware, runtime)
                with self.assertRaisesRegex(ValueError, "hardware identity"):
                    job.validate_target_gate(gate, {**hardware, "memory_bytes": 8}, runtime)
                with self.assertRaisesRegex(ValueError, "executable"):
                    job.validate_target_gate(gate, hardware, {**runtime, "executable": "/other/.venv/bin/python"})

    def test_source_drift_failure_is_preserved_and_existing_attempt_refused(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            gate_path = root / "gate.json"
            write_json(gate_path, {"schema_version": 1, "passed": True, "code_sha256": "code",
                       "uv_lock_sha256": job.file_sha256(job.ROOT / "uv.lock"), "checks": {k: True for k in GATE_CHECKS}})
            contract_path = root / "contract.json"
            write_json(contract_path, {})
            contract = {"path": str(contract_path), "sha256": job.file_sha256(contract_path), "metadata": {}}
            plan = job.calibration_plan(0, root / "outputs")
            stack.enter_context(patch.object(job, "user_lock_path", return_value=root / "lock"))
            stack.enter_context(patch.object(job, "validate_native_host"))
            stack.enter_context(patch.object(job, "hardware_inventory", return_value={"cpu_model": "fixture"}))
            stack.enter_context(patch.object(job, "validate_runtime", return_value={"versions": "fixture"}))
            stack.enter_context(patch.object(job, "validate_target_gate"))
            stack.enter_context(patch.object(job, "gate_model_contract", return_value=contract))
            stack.enter_context(patch.object(job, "capture", return_value={"stdout": "fixture"}))
            stack.enter_context(patch.object(job, "sample_resources", side_effect=sample))
            stack.enter_context(patch.object(job, "code_inventory", return_value=[]))
            code_hash = stack.enter_context(patch.object(job, "code_sha256", side_effect=["code", "code", "drift"]))
            child = stack.enter_context(patch.object(job, "run_child", return_value={"returncode": 0, "stop_reason": None,
                     "child_wall_time_seconds": 1, "setup_wall_time_seconds": 0.1, "resource_summary": {"sample_count": 1}}))
            stack.enter_context(patch.object(job, "verify_completion", return_value={"epoch_wall_time_seconds_sum": 0.1,
                     "checkpoint_save_wall_time_seconds": 0.01}))
            result = job.execute(plan, gate_path, 5)
            self.assertEqual(result, 1)
            provenance = job.read_json(Path(plan["directory"]) / "provenance.json")
            self.assertEqual(provenance["status"], "failed")
            self.assertIn("source or dependency lock changed", provenance["error"])
            self.assertEqual(provenance["mode"], "calibration")
            self.assertEqual(provenance["expected_epochs"], 20)
            self.assertIn("launcher_wall_time_seconds", provenance)
            code_hash.side_effect = None
            code_hash.return_value = "code"
            with self.assertRaisesRegex(FileExistsError, "existing"):
                job.execute(plan, gate_path, 5)
            self.assertEqual(child.call_count, 1)


if __name__ == "__main__":
    unittest.main()
