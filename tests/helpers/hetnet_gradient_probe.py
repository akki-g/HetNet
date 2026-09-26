"""Read-only gradient instrumentation around real HetNet A2C/PCP updates.

No loss, zero_grad call, clipping, aggregation or optimizer is replaced.
This small three-update fixture is a structural diagnostic, not learning data.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.multiprocessing as mp

# main.py executes the equivalent DoubleTensor/thread context on spawned import.
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

import data
from action_utils import parse_action_args
from hetgat.policy import A2CPolicy
from ic3net_envs.predator_capture_env import PredatorCaptureEnv
from multi_processing import MultiProcessTrainer
from trainer import Trainer
from shared_gradient_probe import model_signature, timeout_handler

SOURCES = ["multi_processing.py", "trainer.py", "hetgat/policy.py", "hetgat/uavnet.py",
           "hetgat/graph/fastreal.py", "hetgat/utils.py", "data.py", "env_wrappers.py",
           "envs/ic3net_envs/predator_capture_env.py", "uv.lock",
           "tests/helpers/hetnet_gradient_probe.py", "tests/helpers/shared_gradient_probe.py"]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def make_args(seed, batch_size, horizon, num_p, num_a):
    parser = argparse.ArgumentParser()
    # Read only top-level parser.add_argument expressions, retaining current
    # upstream defaults without importing the side-effectful main module.
    tree = ast.parse((ROOT / "main.py").read_text())
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "parser"
                and node.value.func.attr == "add_argument"):
            expression = ast.Expression(node.value)
            eval(compile(expression, "main.py parser declarations", "eval"), {"parser": parser})
    PredatorCaptureEnv().init_args(parser)
    args = parser.parse_args([
        "--env_name", "predator_capture", "--nfriendly_P", str(num_p), "--nfriendly_A", str(num_a),
        "--nprocesses", "4", "--num_epochs", "1", "--epoch_size", "1",
        "--hid_size", "128", "--detach_gap", "5", "--lrate", "0.0001",
        "--dim", "5", "--vision", "2", "--batch_size", str(batch_size),
        "--max_steps", str(horizon), "--hetgat", "--hetgat_a2c", "--seed", str(seed)])
    args.nfriendly = args.nfriendly_P + args.nfriendly_A
    env = data.init(args.env_name, args, False)
    args.num_inputs = env.observation_dim
    args.num_actions = [env.num_actions]
    args.dim_actions = env.dim_actions
    parse_action_args(args)
    return args


def make_policy(args):
    pos = args.dim ** 2
    return A2CPolicy(
        {"vision": args.vision, "P": pos + 4, "A": pos, "state": 4},
        {"P": pos + 4, "A": pos, "state": 4},
        {"P": 16, "A": 16, "state": 16}, {"P": 5, "A": 6, "state": 8},
        args.nfriendly_P, args.nfriendly_A, num_heads=4, msg_dim=args.msg_dim,
        device=torch.device("cpu"), gamma=args.gamma, lr=args.lrate,
        weight_decay=0, milestones=[200, 400], lr_gamma=0.1,
        use_real=True, use_CNN=False, use_tanh=False, per_class_critic=True,
        per_agent_critic=False, with_two_state=True, obs=(2 * args.vision + 1) ** 2,
        comm_range_P=args.comm_range_P, comm_range_A=args.comm_range_A,
        lossy_comm=args.lossy_comm, min_comm_loss=args.min_comm_loss,
        max_comm_loss=args.max_comm_loss, tensor_obs=args.tensor_obs,
        action_vision=args.A_vision)


class ObservedTrainer(Trainer):
    def __init__(self, args, policy, rank, telemetry):
        super().__init__(args, policy.model, data.init(args.env_name, args), policy)
        self.probe_rank = rank
        self.probe_telemetry = telemetry
        self.probe_update = -1

    def run_batch(self, epoch):
        self.probe_update = epoch
        return super().run_batch(epoch)

    def compute_grad(self, batch):
        stat = super().compute_grad(batch)
        gradients = []
        for name, parameter in self.policy_net.named_parameters():
            if parameter.grad is not None:
                gradients.append({"name": name, "shape": list(parameter.shape),
                                  "pointer": parameter.grad.data_ptr(),
                                  "value": parameter.grad.detach().cpu().numpy().copy()})
        stat["gradient_probe_records"] = [{"rank": self.probe_rank,
            "update": self.probe_update, "gradients": gradients,
            "policy_loss": stat.get("action_loss"), "value_loss": stat.get("value_loss")}]
        return stat

    def get_memory_peak(self):
        self.probe_telemetry.put({"rank": self.probe_rank, "update": self.probe_update,
            "parameter_signature": model_signature(self.policy_net),
            "gradient_pointers_after_transport": {
                name: parameter.grad.data_ptr()
                for name, parameter in self.policy_net.named_parameters()
                if parameter.grad is not None}})
        return super().get_memory_peak()


class Factory:
    def __init__(self, args, policy, telemetry):
        self.args, self.policy, self.telemetry = args, policy, telemetry
        self.rank = 0

    def __call__(self):
        result = ObservedTrainer(copy.deepcopy(self.args), self.policy, self.rank, self.telemetry)
        self.rank += 1
        return result


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = make_args(args.seed, args.batch_size, args.horizon, args.num_p, args.num_a)
    policy = make_policy(config)
    for parameter in policy.model.parameters():
        parameter.data.share_memory_()
    before_sources = hashes()
    result = {"probe": "actual_hetnet_multiprocess_gradient_contract",
        "scope": "real A2CPolicy + Trainer + PCP; bounded structural fixture, not a learning result",
        "status": "ERROR", "gate_passed": False, "shared_weights_passed": False,
        "fresh_gradient_aggregation_passed": False, "gradient_storage_passed": False,
        "errors": [], "updates": [], "resolved_args": vars(config),
        "provenance": {"torch": torch.__version__, "python": sys.version,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_hashes_before": before_sources, "start_method": mp.get_start_method(),
            "instrumentation": "Trainer subclass only snapshots after super().compute_grad and after optimizer step",
            "probe_mutates_learner": False}}
    telemetry = mp.get_context("spawn").Queue()
    trainer = None
    previous_children = {child.pid for child in mp.active_children()}
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout_seconds)
    try:
        trainer = MultiProcessTrainer(config, Factory(config, policy, telemetry))
        first_worker_pointers = None
        for update in range(args.updates):
            signature_before = model_signature(policy.model)
            stat, _, _ = trainer.train_batch(update)
            signature_after = model_signature(policy.model)
            records = sorted(stat["gradient_probe_records"], key=lambda item: item["rank"])
            if [r["rank"] for r in records] != [0, 1, 2, 3]:
                raise AssertionError("Missing a process's fresh gradient snapshot")
            names = [[g["name"] for g in r["gradients"]] for r in records]
            if any(n != names[0] for n in names):
                raise AssertionError("Processes have different named non-null gradient sets/order")
            actual = {n: p for n, p in policy.model.named_parameters() if p.grad is not None}
            if list(actual) != names[0] or len(trainer.grads) != len(actual):
                raise AssertionError("Cached/current gradient parameter set or order changed")
            workers = sorted([telemetry.get(timeout=10) for _ in range(3)], key=lambda r: r["rank"])
            if [r["rank"] for r in workers] != [1, 2, 3] or any(r["update"] != update for r in workers):
                raise AssertionError("Post-step worker signatures missing or mistimed")
            worker_pointers = [row["gradient_pointers_after_transport"] for row in workers]
            if any(list(pointers) != names[0] for pointers in worker_pointers):
                raise AssertionError("Worker post-transport gradient parameter set/order changed")
            # The first tensor IPC can move storage into shared memory. Use
            # pointers observed AFTER that transfer as the stable baseline.
            if first_worker_pointers is None:
                first_worker_pointers = worker_pointers
            snapshots = {}
            details = []
            for index, name in enumerate(names[0]):
                fresh = [r["gradients"][index]["value"] for r in records]
                expected = fresh[0].copy()
                for worker_gradient in fresh[1:]:
                    expected += worker_gradient
                expected /= stat["num_steps"]
                current = actual[name].grad.detach().cpu().numpy().copy()
                cached = trainer.grads[index].detach().cpu().numpy().copy()
                cached_workers = [worker[index].detach().cpu().numpy().copy()
                                  for worker in trainer.worker_grads]
                worker_cached_matches = [bool(np.array_equal(cached_worker, fresh[rank + 1]))
                                         for rank, cached_worker in enumerate(cached_workers)]
                worker_storage_matches = [pointers[name] == first_worker_pointers[rank][name]
                                          for rank, pointers in enumerate(worker_pointers)]
                main_storage_matches = actual[name].grad.data_ptr() == trainer.grads[index].data_ptr()
                error = float(np.max(np.abs(current - expected)))
                for rank, gradient in enumerate(fresh):
                    snapshots[f"rank{rank}_param{index}"] = gradient
                snapshots[f"expected_param{index}"] = expected
                snapshots[f"actual_param{index}"] = current
                snapshots[f"cached_param{index}"] = cached
                details.append({"name": name, "shape": list(current.shape), "snapshot_index": index,
                    "max_absolute_error": error,
                    "fresh_gradient_aggregation_passed": bool(np.allclose(current, expected, rtol=1e-10, atol=1e-12)),
                    "main_gradient_storage_passed": main_storage_matches,
                    "worker_cached_gradients_match_fresh": worker_cached_matches,
                    "worker_current_storage_matches_first_transport": worker_storage_matches,
                    "gradient_storage_passed": main_storage_matches and all(worker_storage_matches)
                                               and all(worker_cached_matches),
                    "worker_cached_gradient_sha256": [array_digest(value) for value in cached_workers],
                    "expected_sha256": array_digest(expected), "actual_sha256": array_digest(current),
                    "current_pointer": actual[name].grad.data_ptr(), "cached_pointer": trainer.grads[index].data_ptr()})
            snapshot_path = args.output.with_name(args.output.stem + f"_update{update+1}.npz")
            with snapshot_path.open("xb") as handle:
                np.savez_compressed(handle, **snapshots)
            result["updates"].append({"update": update + 1, "total_steps": stat["num_steps"],
                "total_episodes": stat["num_episodes"], "parameters_with_gradients": len(details),
                "fresh_gradient_aggregation_passed": all(d["fresh_gradient_aggregation_passed"] for d in details),
                "gradient_storage_passed": all(d["gradient_storage_passed"] for d in details),
                "shared_weights_passed": all(r["parameter_signature"] == signature_after for r in workers),
                "parameter_update_observed": signature_before != signature_after,
                "max_absolute_gradient_error": max(d["max_absolute_error"] for d in details),
                "snapshot_file": str(snapshot_path.relative_to(ROOT) if snapshot_path.is_relative_to(ROOT) else snapshot_path),
                "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                "parameter_details": details, "main_parameter_signature_after_step": signature_after,
                "worker_parameter_signatures_after_step": workers,
                "fresh_process_losses": [{"rank": r["rank"], "policy_loss": r["policy_loss"],
                                          "value_loss": r["value_loss"]} for r in records]})
        after_sources = hashes()
        result["provenance"]["source_hashes_after"] = after_sources
        if before_sources != after_sources:
            raise AssertionError("Scientific source files changed while the probe was running")
        for key in ("shared_weights_passed", "fresh_gradient_aggregation_passed", "gradient_storage_passed"):
            result[key] = all(r[key] for r in result["updates"])
        result["gate_passed"] = all(result[key] for key in
            ("shared_weights_passed", "fresh_gradient_aggregation_passed", "gradient_storage_passed"))
        result["gate_passed"] &= all(r["parameter_update_observed"] for r in result["updates"])
        result["status"] = "PASS" if result["gate_passed"] else "FAIL"
        if not result["gate_passed"]:
            result["errors"].append("Actual HetNet learner violates current-gradient aggregation/storage contract; probe made no learner changes")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        if trainer is not None:
            try:
                trainer.quit()
            except (BrokenPipeError, EOFError, OSError):
                pass
        children = [c for c in mp.active_children() if c.pid not in previous_children]
        for child in children:
            child.join(timeout=2)
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=2)
        telemetry.close()
        telemetry.join_thread()
        signal.signal(signal.SIGALRM, previous_handler)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-p", type=int, default=2)
    parser.add_argument("--num-a", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or any(args.output.parent.glob(args.output.stem + "_update*.npz")):
        parser.error("Refusing to overwrite existing JSON or snapshot evidence")
    if args.updates < 3 or args.batch_size < 2 or args.horizon < 2:
        parser.error("Require at least 3 updates, 2 batch steps and horizon 2")
    if args.num_p < 2 or args.num_a < 1 or args.num_p + args.num_a + 1 > 25:
        parser.error("Require at least two P, one A, and at most 25 occupied grid cells")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mp.set_start_method("spawn")
    result = run(args)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Actual HetNet gradient contract: {result['status']} ({args.output})")
    for row in result["updates"]:
        print(f"  update {row['update']}: shared_weights={row['shared_weights_passed']} "
              f"current_storage={row['gradient_storage_passed']} "
              f"fresh_aggregation={row['fresh_gradient_aggregation_passed']} "
              f"max_gradient_error={row['max_absolute_gradient_error']:.9g}")
    for error in result["errors"]:
        print(f"  {error}")
    return {"PASS": 0, "FAIL": 1, "ERROR": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
