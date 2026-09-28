"""Submission contracts with temporary evidence and mocked scheduler calls only."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hetnet_ext import submit_run1 as submission
from hetnet_ext.grid import ROOT as REAL_ROOT, load_grid
from hetnet_ext.train_job import GATE_CHECKS


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (self.root / "uv.lock").write_text("test lock only")
        self.stack.enter_context(patch.object(submission, "ROOT", self.root))
        self.stack.enter_context(patch.object(submission, "code_sha256", return_value="source"))
        self.preflight_path = self.root / "preflight.json"
        self.gate_path = self.root / "gate.json"
        self.budget_path = self.root / "budget with spaces $(literal).json"
        self.approval_path = self.root / "approval.json"
        self.output_root = self.root / "new outputs"
        self.preflight = {"schema_version": 1, "verified": True, "cluster": "stokes", "partition": "normal",
                          "max_wall_time_seconds": 86400, "remaining_core_hours": 60000,
                          "max_concurrent_jobs": 8, "max_submit_jobs": 100,
                          "account_required": True, "account": "verified_pi", "verified_at_utc": "fixture",
                          "evidence_paths": ["fixture raw evidence"], "python_major_minor": "3.12",
                          "compute_network_verified": True}
        self.gate = {"schema_version": 1, "passed": True, "code_sha256": "source",
                     "uv_lock_sha256": submission.file_sha256(self.root / "uv.lock"),
                     "checks": {name: True for name in GATE_CHECKS}}
        self.budget = {"schema_version": 1, "code_sha256": "source", "uv_lock_sha256": self.gate["uv_lock_sha256"],
                       "projected_total_core_hours": 500, "remaining_core_hours": 60000,
                       "within_partition_cap": True, "array_uniform_time_seconds": 7200,
                       "array_uniform_memory_gb": 6,
                       "rows": [{"composition": name, "seeds": seeds, "cpus_per_task": 4,
                                 "requested_wall_seconds": 7200, "memory_gb_with_50pct_margin": 6}
                                for name, seeds in (("2P1A", 5), ("3P3A", 5), ("4P6A", 5), ("3P1A", 3), ("2P2A", 3))]}
        self.approval = {"schema_version": 1, "approved": True, "approved_by": "fixture only",
                         "approved_at_utc": "fixture", "budget_file": self.budget_path.name,
                         "threshold_override_approved": False}
        self.save_evidence()

    def save_evidence(self):
        self.preflight_path.write_text(json.dumps(self.preflight))
        self.gate_path.write_text(json.dumps(self.gate))
        self.budget["preflight_sha256"] = submission.file_sha256(self.preflight_path)
        self.budget_path.write_text(json.dumps(self.budget))
        self.approval["budget_sha256"] = submission.file_sha256(self.budget_path)
        self.approval_path.write_text(json.dumps(self.approval))

    def prepare(self, concurrency=3):
        return submission.prepare(self.preflight_path, self.gate_path, self.approval_path,
                                  concurrency, self.output_root)

    def test_full_mapping_resources_account_and_safe_rendering_without_writes(self):
        plan = self.prepare()
        self.assertEqual(plan["entries"], [entry.as_dict() for entry in load_grid()])
        for argument in ("--array=0-20%3", "--cpus-per-task=4", "--ntasks=1", "--nodes=1", "--partition=normal",
                         "--mem=6G", "--time=0-02:00:00", "--account=verified_pi", "--parsable"):
            self.assertIn(argument, plan["command"])
        with patch.dict(os.environ, {"SBATCH_GRES": "gpu:1", "SBATCH_ACCOUNT": "wrong", "HETNET_OUTPUT_ROOT": "/old/pilot"}):
            rendered = shlex.split(submission.shell_command(plan))
            self.assertEqual(rendered[-len(plan["command"]):], plan["command"])
            self.assertIn("-u", rendered)
            env = submission.clean_environment(plan)
            self.assertFalse(any(key.startswith("SBATCH_") for key in env))
            self.assertEqual(env["HETNET_OUTPUT_ROOT"], str(self.output_root.resolve()))
        self.assertFalse(self.output_root.exists())
        self.assertFalse((self.root / "logs").exists())

    def test_cli_defaults_to_dry_run_and_never_calls_scheduler(self):
        arguments = ["--preflight", str(self.preflight_path), "--gate-a", str(self.gate_path),
                     "--budget-approval", str(self.approval_path), "--concurrency", "3", "--output-root", str(self.output_root)]
        with patch.object(submission, "submit") as submit, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(submission.main(arguments), 0)
            submit.assert_not_called()
        self.assertIn("Dry-run only", output.getvalue())
        self.assertFalse(self.output_root.exists())

    def test_optional_account_and_unknown_or_excessive_resources(self):
        self.preflight.update(account_required=False, account=None)
        self.save_evidence()
        self.assertFalse(any(arg.startswith("--account") for arg in self.prepare()["command"]))
        for value in (0, -1, 9, 22):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepare(value)
        self.preflight["max_submit_jobs"] = 20
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, "21 tasks"):
            self.prepare()
        self.preflight["max_submit_jobs"] = 100
        for field, value in (("array_uniform_time_seconds", None), ("array_uniform_time_seconds", 7201),
                             ("array_uniform_memory_gb", 0), ("array_uniform_memory_gb", 7)):
            old = self.budget[field]
            self.budget[field] = value
            self.save_evidence()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.prepare()
            self.budget[field] = old

    def test_partial_budget_missing_approval_and_stale_gate_are_rejected(self):
        original = deepcopy(self.budget)
        self.budget["rows"] = self.budget["rows"][:-1]
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, "five training compositions"):
            self.prepare()
        self.budget = original
        self.approval["approved"] = False
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.prepare()
        self.approval["approved"] = True
        self.gate["code_sha256"] = "old"
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, "stale"):
            self.prepare()

    def test_changed_preflight_or_confirmation_hash_does_not_pass(self):
        self.preflight["remaining_core_hours"] = 1000
        self.preflight_path.write_text(json.dumps(self.preflight))
        with self.assertRaisesRegex(ValueError, "current Stokes preflight"):
            self.prepare()
        self.preflight["remaining_core_hours"] = 60000
        self.save_evidence()
        self.budget_path.write_text(json.dumps({**self.budget, "projected_total_core_hours": 1}))
        with self.assertRaisesRegex(ValueError, "hash"):
            self.prepare()

    def test_existing_run_or_submission_record_is_preserved(self):
        old = self.output_root / "2P1A" / "seed0"
        old.mkdir(parents=True)
        (old / "keep.txt").write_text("failed attempt")
        with self.assertRaisesRegex(FileExistsError, "existing run"):
            self.prepare()
        self.assertEqual((old / "keep.txt").read_text(), "failed attempt")

    def test_explicit_submission_creates_logs_before_sbatch_and_records_job_id(self):
        plan = self.prepare()
        calls = []

        def fake_scheduler(command, **kwargs):
            calls.append(command)
            if command[0] == "scontrol":
                return subprocess.CompletedProcess(command, 0, "ClusterName = stokes\n", "")
            self.assertTrue((self.root / "logs").is_dir())
            self.assertEqual(submission.read_json(Path(plan["submission_record"]))["status"], "submitting")
            self.assertEqual(kwargs["env"]["HETNET_MEM_GB"], "6")
            self.assertEqual(kwargs["env"]["HETNET_TIME_SECONDS"], "7200")
            self.assertEqual(kwargs["cwd"], self.root)
            return subprocess.CompletedProcess(command, 0, "12345;stokes\n", "")

        with patch.object(submission.subprocess, "run", side_effect=fake_scheduler):
            self.assertEqual(submission.submit(plan), "12345")
        record = submission.read_json(Path(plan["submission_record"]))
        self.assertEqual(record["status"], "submitted")
        self.assertEqual(record["array_job_id"], "12345")
        self.assertEqual(record["command"], plan["command"])
        self.assertEqual([call[0] for call in calls], ["scontrol", "sbatch"])
        with self.assertRaisesRegex(FileExistsError, "Submission record"):
            self.prepare()

    def test_wrong_cluster_or_changed_evidence_never_calls_sbatch(self):
        plan = self.prepare()
        with patch.object(submission.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ClusterName = newton\n", "")) as query:
            with self.assertRaisesRegex(ValueError, "not verified as Stokes"):
                submission.submit(plan)
            self.assertEqual(query.call_count, 1)
        self.assertFalse(self.output_root.exists())
        self.gate_path.write_text("changed")
        with patch.object(submission.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ClusterName = stokes\n", "")) as query:
            with self.assertRaisesRegex(ValueError, "changed"):
                submission.submit(plan)
            self.assertEqual(query.call_count, 1)

    def test_ambiguous_submission_preserves_evidence_without_automatic_retry(self):
        plan = self.prepare()
        responses = [subprocess.CompletedProcess([], 0, "ClusterName = stokes\n", ""),
                     subprocess.TimeoutExpired(plan["command"], 60)]
        with patch.object(submission.subprocess, "run", side_effect=responses) as process:
            with self.assertRaises(subprocess.TimeoutExpired):
                submission.submit(plan)
            self.assertEqual(process.call_count, 2)
        record = submission.read_json(Path(plan["submission_record"]))
        self.assertEqual(record["status"], "submission_unknown")
        with self.assertRaises(FileExistsError):
            self.prepare()


class EntrypointTests(unittest.TestCase):
    def test_shell_syntax_and_help_without_ml_packages(self):
        subprocess.run(["bash", "-n", str(REAL_ROOT / "slurm" / "submit_run1.sh")], check=True)
        result = subprocess.run([sys.executable, "-S", "-m", "hetnet_ext.submit_run1", "--help"],
                                cwd=REAL_ROOT, check=True, capture_output=True, text=True)
        self.assertIn("--submit", result.stdout)
        self.assertIn("--concurrency", result.stdout)


if __name__ == "__main__":
    unittest.main()
