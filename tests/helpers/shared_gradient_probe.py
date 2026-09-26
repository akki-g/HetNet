"""Exercise upstream multiprocessing without repairing its gradient lifecycle.

The deterministic trainer isolates transport/aggregation from RL sampling. It
retains both upstream clearing sites: outer RMSprop and policy-local Adam.
Exit 1 means a scientific gate failed, not an expected/accepted test result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import traceback
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

import torch
import torch.multiprocessing as mp

from multi_processing import MultiProcessTrainer


def policy_clear_keywords():
    """Read the real selected loss's clearing contract; do not invent a fix.

    The initial historical probe used the empty upstream keyword set. Reading
    it now lets the surrogate follow an explicitly approved compatibility fix
    while the actual-HetNet probe verifies the complete loss independently.
    """
    tree = ast.parse((REPOSITORY / "hetgat/policy.py").read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "A2CPolicy")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                  and node.name == "batch_finish_per_class")
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "zero_grad"
             and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "optimizer"
             and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "self"]
    if len(calls) != 1 or calls[0].args or any(k.arg is None for k in calls[0].keywords):
        raise RuntimeError("Selected upstream loss has an unsupported gradient-clear contract")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in calls[0].keywords}


def model_signature(model):
    digest = hashlib.sha256()
    rows = []
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        value_hash = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        row = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
               "sha256": value_hash}
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True).encode())
    return {"sha256": digest.hexdigest(), "tensors": rows}


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.2, 0.8], dtype=torch.float64))


class DeterministicTrainer:
    """Implements the interface consumed by the unchanged upstream aggregator."""

    def __init__(self, model, rank, telemetry):
        self.model = model
        self.params = list(model.parameters())
        self.rank = rank
        self.telemetry = telemetry
        self.optimizer = torch.optim.RMSprop(self.params, lr=1e-4, alpha=0.97, eps=1e-6)
        self.policy_optimizer = torch.optim.Adam(self.params, lr=1e-4)
        self.update = -1
        self.first_gradient_pointer = None
        self.policy_clear_kwargs = policy_clear_keywords()

    def run_batch(self, epoch):
        self.update = epoch
        return epoch, {"num_steps": 3 * (self.rank + 1) + epoch, "num_episodes": 1}

    def compute_grad(self, batch):
        # Mirrors policy.py:846; upstream worker/main already cleared RMSprop.
        self.policy_optimizer.zero_grad(**self.policy_clear_kwargs)
        coefficients = torch.tensor(
            [(self.rank + 1) * (batch + 1), self.rank + 2 + 0.5 * batch,
             (-1) ** self.rank * (batch + 3)], dtype=self.model.weight.dtype)
        loss = (self.model.weight * coefficients).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 0.75)
        pointer = self.model.weight.grad.data_ptr()
        if self.first_gradient_pointer is None:
            self.first_gradient_pointer = pointer
        return {"probe_records": [{
            "rank": self.rank, "update": batch,
            "fresh_clipped_gradient": self.model.weight.grad.detach().tolist(),
            "current_gradient_pointer": pointer,
            "storage_matches_first_backward": pointer == self.first_gradient_pointer,
        }]}

    def reset_memory_peak(self):
        pass

    def get_memory_peak(self):
        # Upstream requests this only after its main optimizer.step(). Observe
        # the worker's shared parameters then, without adding a new command.
        self.telemetry.put({"rank": self.rank, "update": self.update,
                            "parameter_signature": model_signature(self.model)})
        return 0, 0


class TrainerFactory:
    def __init__(self, model, telemetry):
        self.model = model
        self.telemetry = telemetry
        self.next_rank = 0

    def __call__(self):
        rank = self.next_rank
        self.next_rank += 1
        return DeterministicTrainer(self.model, rank, self.telemetry)


def timeout_handler(signum, frame):
    raise TimeoutError("Multiprocessing gradient probe exceeded its wall-time limit")


def run_probe(updates, timeout_seconds):
    if updates < 2:
        raise ValueError("At least two updates are required to test cached-gradient reuse")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = TinyPolicy()
    for parameter in model.parameters():
        parameter.data.share_memory_()
    telemetry = mp.get_context("spawn").Queue()
    args = SimpleNamespace(nprocesses=4, random=False, seed=17)
    trainer = None
    existing_children = {child.pid for child in mp.active_children()}
    output = {
        "probe": "upstream_multiprocess_gradient_contract",
        "scope": "unchanged MultiProcessTrainer; deterministic surrogate loss, not HetNet training",
        "status": "ERROR", "gate_passed": False, "errors": [], "updates": [],
        "shared_weights_passed": False, "fresh_gradient_aggregation_passed": False,
        "gradient_storage_passed": False,
        "provenance": {
            "torch": torch.__version__, "python": sys.version,
            "start_method": mp.get_start_method(), "nprocesses": 4,
            "zero_grad_signature": str(inspect.signature(torch.optim.Optimizer.zero_grad)),
            "upstream_aggregator_sha256": hashlib.sha256(
                (REPOSITORY / "multi_processing.py").read_bytes()).hexdigest(),
            "clearing_sites": ["upstream outer RMSprop.zero_grad()",
                               "surrogate policy-local Adam.zero_grad() matching policy.py:846"],
            "policy_clear_keywords_from_selected_source": policy_clear_keywords(),
            "clipping": "per-process norm 0.75 before aggregation, matching policy.py:865",
        },
    }
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        trainer = MultiProcessTrainer(args, TrainerFactory(model, telemetry))
        for update in range(updates):
            before = model_signature(model)
            stat, _, _ = trainer.train_batch(update)
            after = model_signature(model)
            observations = sorted(stat["probe_records"], key=lambda row: row["rank"])
            if [row["rank"] for row in observations] != [0, 1, 2, 3]:
                raise AssertionError("Did not receive one fresh gradient from every process")
            expected = [sum(row["fresh_clipped_gradient"][k] for row in observations)
                        / stat["num_steps"] for k in range(3)]
            current = model.weight.grad.detach().tolist()
            cached = trainer.grads[0].detach().tolist()
            worker_records = sorted(
                [telemetry.get(timeout=min(10, timeout_seconds)) for _ in range(3)],
                key=lambda row: row["rank"])
            if [row["rank"] for row in worker_records] != [1, 2, 3]:
                raise AssertionError("Did not receive post-step signatures from all workers")
            if any(row["update"] != update for row in worker_records):
                raise AssertionError("Worker signature was from a different optimizer step")
            shared_ok = all(row["parameter_signature"] == after for row in worker_records)
            storage_ok = trainer.grads[0].data_ptr() == model.weight.grad.data_ptr()
            aggregation_ok = all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
                                 for x, y in zip(current, expected))
            output["updates"].append({
                "update": update + 1, "actual_total_steps": stat["num_steps"],
                "fresh_process_gradients": observations,
                "expected_current_gradient": expected,
                "actual_optimizer_gradient": current,
                "cached_main_gradient": cached,
                "max_absolute_gradient_error": max(abs(x-y) for x, y in zip(current, expected)),
                "main_current_gradient_pointer": model.weight.grad.data_ptr(),
                "main_cached_gradient_pointer": trainer.grads[0].data_ptr(),
                "gradient_storage_passed": storage_ok,
                "fresh_gradient_aggregation_passed": aggregation_ok,
                "shared_weights_passed": shared_ok,
                "parameter_update_observed": before != after,
                "main_parameter_signature_after_step": after,
                "worker_parameter_signatures_after_step": worker_records,
            })
        output["shared_weights_passed"] = all(row["shared_weights_passed"] and
            row["parameter_update_observed"] for row in output["updates"])
        output["fresh_gradient_aggregation_passed"] = all(
            row["fresh_gradient_aggregation_passed"] for row in output["updates"])
        output["gradient_storage_passed"] = all(
            row["gradient_storage_passed"] for row in output["updates"])
        output["gate_passed"] = all(output[key] for key in
            ["shared_weights_passed", "fresh_gradient_aggregation_passed", "gradient_storage_passed"])
        output["status"] = "PASS" if output["gate_passed"] else "FAIL"
        if not output["gate_passed"]:
            output["errors"].append(
                "Scientific Gate A failure: shared parameters can remain correct while cached "
                "gradient storage becomes stale after zero_grad(set_to_none=True). "
                "No upstream clearing or aggregation behavior was patched by this probe.")
    except Exception as exc:
        output["errors"].append(f"{type(exc).__name__}: {exc}")
        output["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        if trainer is not None:
            try:
                trainer.quit()
            except (BrokenPipeError, EOFError, OSError):
                pass
        children = [child for child in mp.active_children() if child.pid not in existing_children]
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
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing evidence: {args.output}")
    if args.timeout_seconds < 1 or args.timeout_seconds > 120:
        parser.error("timeout-seconds must lie between 1 and 120")
    mp.set_start_method("spawn")
    result = run_probe(args.updates, args.timeout_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Gradient contract: {result['status']} (evidence: {args.output})")
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
