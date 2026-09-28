"""Phase A infrastructure contracts; standard-library tests, no training/imports."""

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hetnet_ext import budget, grid, train_job


def preflight():
    return {"schema_version": 1, "verified": True, "cluster": "stokes", "partition": "normal",
            "max_wall_time_seconds": 86400, "remaining_core_hours": 60000,
            "max_concurrent_jobs": 8, "max_submit_jobs": 100, "account_required": False,
            "account": None, "verified_at_utc": "2026-09-26T00:00:00Z", "evidence_paths": ["example.txt"],
            "python_major_minor": "3.12", "compute_network_verified": True}


class GridTests(unittest.TestCase):
    def test_exact_bijection_and_priorities(self):
        entries = grid.load_grid()
        expected = {(2, 1, s) for s in range(5)} | {(3, 3, s) for s in range(5)}
        expected |= {(4, 6, s) for s in range(5)} | {(3, 1, s) for s in range(3)} | {(2, 2, s) for s in range(3)}
        self.assertEqual(len(entries), 21)
        self.assertEqual({(e.nfriendly_P, e.nfriendly_A, e.seed) for e in entries}, expected)
        self.assertEqual([e.index for e in entries], list(range(21)))
        self.assertEqual(Counter(e.priority for e in entries), {"P0": 5, "P1": 10, "P2": 6})
        self.assertEqual((entries[0].composition, entries[20].composition, entries[20].seed), ("2P1A", "2P2A", 2))

    def test_singleton_and_duplicate_seed_rejected(self):
        original = json.loads(grid.DEFAULT_GRID.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.json"
            singleton = deepcopy(original)
            singleton["compositions"][0].update(name="1P1A", nfriendly_P=1)
            path.write_text(json.dumps(singleton))
            with self.assertRaisesRegex(ValueError, "singleton"):
                grid.load_grid(path)
            duplicate = deepcopy(original)
            duplicate["compositions"][0]["seeds"] = [0, 0]
            path.write_text(json.dumps(duplicate))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                grid.load_grid(path)

    def test_calibration_matches_recipe_and_endpoints(self):
        entries = grid.calibration_entries()
        self.assertEqual([(e.index, e.composition, e.seed) for e in entries], [(0, "2P1A", 0), (1, "4P6A", 0)])
        for entry in entries:
            command = grid.build_command(entry, "/example/run", mode="calibration")
            for flag, value in {"--num_epochs": "20", "--epoch_size": "10", "--batch_size": "500",
                                "--nprocesses": "4", "--max_steps": "80", "--dim": "5",
                                "--lrate": "0.0001", "--save_every": "50"}.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--hetgat_a2c", command)
            self.assertIn("--metrics_file", command)
            self.assertNotIn("--eval", command)
            self.assertNotIn("--use_binary", command)

    def test_dry_run_prints_all_commands_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "not-created"
            output = subprocess.check_output([sys.executable, "-m", "hetnet_ext.grid", "--dry-run",
                                               "--output-root", str(target)], cwd=grid.ROOT, text=True)
            rows = [json.loads(line) for line in output.splitlines()]
            self.assertEqual(len(rows), 21)
            self.assertFalse(target.exists())
            self.assertTrue(all(row["command"][0] == sys.executable for row in rows))

    def test_scientific_hash_ignores_docs_but_includes_code_tests_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("original")
            (root / "README.md").write_text("one")
            (root / "uv.lock").write_text("locked")
            (root / "tests").mkdir()
            (root / "tests" / "contract.py").write_text("check")
            with patch.object(grid.subprocess, "check_output", return_value=b"main.py\0README.md\0uv.lock\0"):
                initial = grid.code_sha256(root)
                (root / "README.md").write_text("two")
                self.assertEqual(initial, grid.code_sha256(root))
                (root / "tests" / "contract.py").write_text("weakened")
                self.assertNotEqual(initial, grid.code_sha256(root))


class GateTests(unittest.TestCase):
    def test_partition_unlimited_is_explicit_and_job_requests_remain_finite(self):
        data = {**preflight(), "max_wall_time_seconds": "unlimited"}
        train_job.validate_preflight(data)
        self.assertTrue(train_job.within_partition_time_limit(172800, "unlimited"))
        self.assertFalse(train_job.within_partition_time_limit(172800, 86400))
        for value in (None, True, 0, -1, "UNLIMITED", "infinite", "86400", "", float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                train_job.validate_preflight({**data, "max_wall_time_seconds": value})
        for value in ("unlimited", "UNLIMITED", float("inf"), float("nan"), 0):
            with self.subTest(request=value), self.assertRaises(ValueError):
                train_job.within_partition_time_limit(value, "unlimited")

    def test_unknown_cluster_limits_or_network_do_not_pass(self):
        train_job.validate_preflight(preflight())
        for key in ("max_wall_time_seconds", "remaining_core_hours", "max_concurrent_jobs",
                    "max_submit_jobs", "account_required", "compute_network_verified"):
            data = preflight()
            data[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                train_job.validate_preflight(data)
        data = preflight()
        data["cluster"] = "newton"
        with self.assertRaises(ValueError):
            train_job.validate_preflight(data)

    def test_gate_requires_every_scientific_check_and_exact_code_lock(self):
        gate = {"schema_version": 1, "passed": True, "code_sha256": "code", "uv_lock_sha256": "lock",
                "checks": {name: True for name in train_job.GATE_CHECKS}}
        train_job.validate_gate(gate, "code", "lock")
        for name in train_job.GATE_CHECKS:
            missing = deepcopy(gate)
            missing["checks"].pop(name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                train_job.validate_gate(missing, "code", "lock")
        with self.assertRaisesRegex(ValueError, "stale"):
            train_job.validate_gate(gate, "other", "lock")

    def test_slurm_time_and_actual_allocation_are_checked(self):
        self.assertEqual(train_job.parse_slurm_time("1-02:03:04"), 93784)
        for bad in ("UNLIMITED", "12:99:00", "0:10", "-1:00:00"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                train_job.parse_slurm_time(bad)
        env = {"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "stokes", "SLURM_CPUS_PER_TASK": "4",
               "SLURM_MEM_PER_NODE": "4096", "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
               "SLURM_JOB_CPUS_PER_NODE": "4"}
        with patch.dict(os.environ, env, clear=True), patch.object(train_job.subprocess, "check_output",
                return_value="NumCPUs=4 NumNodes=1 NumTasks=1 Partition=normal TimeLimit=01:00:00 ArrayTaskThrottle=2 Account=pi StdOut=/a StdErr=/b") as query:
            allocation = train_job.verify_allocation(preflight(), 4, 3600, 2)
            self.assertEqual(allocation["time_seconds"], 3600)
            unlimited = {**preflight(), "max_wall_time_seconds": "unlimited"}
            self.assertEqual(train_job.verify_allocation(unlimited, 4, 3600, 2)["time_seconds"], 3600)
            finite_job = query.return_value
            query.return_value = finite_job.replace("TimeLimit=01:00:00", "TimeLimit=UNLIMITED")
            with self.assertRaises(ValueError):
                train_job.verify_allocation(unlimited, 4, 3600, 2)
            query.return_value = finite_job
            for mem, seconds, concurrency in ((8, 3600, 2), (4, 7200, 2), (4, 3600, 3)):
                with self.assertRaises(ValueError):
                    train_job.verify_allocation(preflight(), mem, seconds, concurrency)
            query.return_value = "NumCPUs=8 NumNodes=1 NumTasks=2 Partition=normal TimeLimit=01:00:00 ArrayTaskThrottle=2 Account=pi"
            with self.assertRaisesRegex(ValueError, "four total CPUs"):
                train_job.verify_allocation(preflight(), 4, 3600, 2)

    def test_approval_is_explicit_hashed_and_threshold_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight_path = root / "preflight.json"
            preflight_path.write_text(json.dumps(preflight()))
            report = {"code_sha256": "code", "uv_lock_sha256": "lock", "projected_total_core_hours": 5000,
                      "remaining_core_hours": 60000, "within_partition_cap": True,
                      "preflight_sha256": grid.file_sha256(preflight_path)}
            path = root / "budget.json"
            path.write_text(json.dumps(report))
            approval = {"approved": False, "approved_by": "Akki", "approved_at_utc": "now",
                        "budget_file": "budget.json", "budget_sha256": grid.file_sha256(path)}
            approval_path = root / "approval.json"
            approval_path.write_text(json.dumps(approval))
            with self.assertRaisesRegex(ValueError, "explicit"):
                train_job.validate_approval(approval_path, "code", "lock", preflight_path)
            approval["approved"] = True
            approval_path.write_text(json.dumps(approval))
            with self.assertRaisesRegex(ValueError, "override"):
                train_job.validate_approval(approval_path, "code", "lock", preflight_path)
            approval["threshold_override_approved"] = True
            approval_path.write_text(json.dumps(approval))
            train_job.validate_approval(approval_path, "code", "lock", preflight_path)
            preflight_path.write_text(json.dumps({**preflight(), "remaining_core_hours": 10000}))
            with self.assertRaisesRegex(ValueError, "current Stokes preflight hash"):
                train_job.validate_approval(approval_path, "code", "lock", preflight_path)
            preflight_path.write_text(json.dumps(preflight()))
            path.write_text(json.dumps({**report, "projected_total_core_hours": 1}))
            with self.assertRaisesRegex(ValueError, "hash"):
                train_job.validate_approval(approval_path, "code", "lock", preflight_path)


class BudgetTests(unittest.TestCase):
    def test_unlimited_partition_preserves_finite_measured_budget_requests(self):
        data = {"p90_seconds_per_epoch": 100, "setup_seconds": 1, "checkpoint_save_seconds": 0,
                "whole_job_peak_rss_gb": 2, "allocation_wall_seconds": 2001,
                "residual_per_epoch_seconds": 0, "first_epoch_excess_seconds": 0}
        projected = budget.project({"2P1A": data, "4P6A": data},
                                   {**preflight(), "max_wall_time_seconds": "unlimited"},
                                   evaluation_reserve=0, other_reserve=0)
        self.assertTrue(projected["within_partition_cap"])
        self.assertEqual(projected["max_wall_time_seconds"], "unlimited")
        self.assertEqual(projected["array_uniform_time_seconds"], 260040)
        self.assertEqual(projected["rows"][0]["estimated_wall_seconds"], 200001)

    def test_cost_counts_all_seeds_cores_saves_and_reserves(self):
        def endpoint(slope):
            return {"p90_seconds_per_epoch": slope, "setup_seconds": 10, "checkpoint_save_seconds": 2,
                    "whole_job_peak_rss_gb": 4, "allocation_wall_seconds": 500,
                    "residual_per_epoch_seconds": 0, "first_epoch_excess_seconds": 0}
        projected = budget.project({"2P1A": endpoint(10), "4P6A": endpoint(20)}, preflight(),
                                   evaluation_reserve=100, other_reserve=20)
        rows = {r["composition"]: r for r in projected["rows"]}
        self.assertEqual(sum(r["seeds"] for r in rows.values()), 21)
        self.assertEqual(rows["2P1A"]["estimated_wall_seconds"], 20090)
        self.assertEqual(rows["3P3A"]["estimated_wall_seconds"], 40090)
        self.assertEqual(rows["3P3A"]["slope_source"], "modeled_max_endpoint_p90")
        expected = (5 * 4 * 26160 + 16 * 4 * 52140) / 3600 + 4000 / 3600 + 120
        self.assertAlmostEqual(projected["projected_total_core_hours"], expected)
        self.assertEqual(projected["array_uniform_memory_gb"], 6)
        self.assertTrue(projected["within_partition_cap"])

    def test_cap_violation_is_reported_without_changing_epochs(self):
        data = {"p90_seconds_per_epoch": 100, "setup_seconds": 1, "checkpoint_save_seconds": 0,
                "whole_job_peak_rss_gb": 2, "allocation_wall_seconds": 2001,
                "residual_per_epoch_seconds": 0, "first_epoch_excess_seconds": 0}
        projected = budget.project({"2P1A": data, "4P6A": data}, preflight(), evaluation_reserve=0, other_reserve=0)
        self.assertFalse(projected["within_partition_cap"])
        self.assertTrue(projected["threshold_exceeded"])
        self.assertEqual(projected["rows"][0]["estimated_wall_seconds"], 200001)

    def test_calibration_requires_complete_metrics_and_artifact_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model_ep20.pt"
            checkpoint.write_bytes(b"test fixture only; not a trained checkpoint")
            entry = grid.calibration_entries()[0].as_dict()
            recipe = json.loads(grid.DEFAULT_GRID.read_text())["recipe"]
            resolved = {**recipe, "num_epochs": 20, "nfriendly_P": 2, "nfriendly_A": 1, "seed": 0,
                        "use_binary": False, "use_cuda": False, "commnet": False, "hetcomm": False,
                        "ic3net": False, "eval": False, "random": False, "load": "",
                        "comm_range_P": -1, "comm_range_A": -1, "lossy_comm": False,
                        "gamma": 1.0, "vision": 2, "A_vision": -1, "tensor_obs": False,
                        "nenemies": 1, "moving_prey": False, "no_stay": False, "mode": "mixed",
                        "enemy_comm": False, "second_reward_scheme": False,
                        "resolved_model": {"class": "hetgat.uavnet.UAVNetA2CEasy", "per_class_critic": True,
                                           "with_two_state": True}}
            resolved_path = root / "resolved_args.json"
            resolved_path.write_text(json.dumps(resolved))
            provenance = {"status": "complete", "mode": "calibration", "code_sha256": "code",
                          "uv_lock_sha256": "lock", "final_checkpoint": str(checkpoint),
                          "final_checkpoint_sha256": grid.file_sha256(checkpoint),
                          "final_parameter_signature": {"sha256": "parameters"}, "entry": entry,
                          "resolved_args": resolved, "resolved_args_sha256": grid.file_sha256(resolved_path)}
            (root / "provenance.json").write_text(json.dumps(provenance))
            (root / "config.json").write_text(json.dumps({"num_epochs": 20, "recipe": recipe, "entry": entry}))
            metrics = [{"epoch": i, "wall_time_seconds": 20 if i > 1 else 40, "steps": 20000} for i in range(1, 21)]
            (root / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))
            save = {"epoch": 20, "path": str(checkpoint), "wall_time_seconds": 1,
                    "bytes": checkpoint.stat().st_size, "parameter_sha256": "parameters"}
            (root / "checkpoint_records.jsonl").write_text(json.dumps(save) + "\n")
            measured = {"whole_job_peak_rss_gb": 4, "setup_seconds": 10,
                        "allocation_wall_seconds": 500, "peak_rss_source": "fixture", "accounting_evidence": "fixture"}
            summary = budget.summarize_calibration(root, measured, "code", "lock", "2P1A")
            self.assertEqual(summary["mean_seconds_per_epoch"], 20)
            self.assertEqual(summary["actual_steps"], 400000)
            self.assertEqual(summary["seconds_per_update"], 2.1)
            self.assertEqual(summary["checkpoint_save_seconds"], 1)
            self.assertEqual(summary["first_epoch_excess_seconds"], 20)
            self.assertEqual(summary["residual_per_epoch_seconds"], 69 / 20)
            projection = budget.project({"2P1A": summary, "4P6A": summary}, preflight(), evaluation_reserve=0, other_reserve=0)
            self.assertEqual(projection["rows"][0]["estimated_wall_seconds"], 10 + 20 + 2000 * (20 + 69 / 20) + 40)
            with self.assertRaisesRegex(ValueError, "endpoint identity"):
                budget.summarize_calibration(root, measured, "code", "lock", "4P6A")
            with self.assertRaisesRegex(ValueError, "shorter than"):
                budget.summarize_calibration(root, {**measured, "allocation_wall_seconds": 400}, "code", "lock", "2P1A")
            resolved["use_binary"] = True
            resolved_path.write_text(json.dumps(resolved))
            provenance.update(resolved_args=resolved, resolved_args_sha256=grid.file_sha256(resolved_path))
            (root / "provenance.json").write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "original real/A2C"):
                budget.summarize_calibration(root, measured, "code", "lock", "2P1A")
            resolved["use_binary"] = False
            resolved_path.write_text(json.dumps(resolved))
            provenance.update(resolved_args=resolved, resolved_args_sha256=grid.file_sha256(resolved_path))
            (root / "provenance.json").write_text(json.dumps(provenance))
            (root / "checkpoint_records.jsonl").write_text(json.dumps({**save, "bytes": 0}) + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                budget.summarize_calibration(root, measured, "code", "lock", "2P1A")
            (root / "checkpoint_records.jsonl").write_text(json.dumps(save) + "\n")
            (root / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics[:-1]))
            with self.assertRaisesRegex(ValueError, "complete epochs"):
                budget.summarize_calibration(root, measured, "code", "lock", "2P1A")


class SlurmTests(unittest.TestCase):
    def test_scripts_have_cpu_only_explicit_resource_contract(self):
        for name in ("run1_train.sbatch", "calibrate.sbatch", "bootstrap.sh"):
            script = grid.ROOT / "slurm" / name
            subprocess.run(["bash", "-n", str(script)], check=True)
            text = script.read_text()
            self.assertNotIn("#SBATCH --gres", text)
            self.assertNotIn("#SBATCH --time=", text)
            self.assertNotIn("#SBATCH --mem=", text)
        bootstrap = (grid.ROOT / "slurm" / "bootstrap.sh").read_text()
        self.assertIn(".venv-stokes", bootstrap)
        self.assertIn("sync --locked", bootstrap)
        self.assertIn("flock --exclusive", bootstrap)
        self.assertNotIn(".venv-newton", bootstrap)
        self.assertFalse((grid.ROOT / "slurm" / "run2_frozen_eval.sbatch").exists())


if __name__ == "__main__":
    unittest.main()
