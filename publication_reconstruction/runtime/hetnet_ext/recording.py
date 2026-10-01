"""External accounting; never changes upstream collection, loss or optimizer.

Count each fresh train_batch statistic once. Upstream main.py instead adds the
growing epoch statistic on each iteration, overcounting cumulative samples.
Losses use upstream's reporting denominator: summed returned losses / joint
environment steps. These are diagnostics, not a redefinition of its objective.
"""
from __future__ import annotations

from argparse import Namespace
import json
import math
from pathlib import Path

import numpy as np

from hetnet_ext.signatures import model_signature


def write_json_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_checkpoint_signature(checkpoint: Path, model):
    signature = model_signature(model)
    write_json_new(Path(str(checkpoint) + ".signature.json"), signature)
    return signature


def record_checkpoint(metrics_file, checkpoint: Path, epoch, wall_time_seconds, signature):
    record = {"epoch": int(epoch), "path": str(checkpoint.resolve()),
              "wall_time_seconds": float(wall_time_seconds), "bytes": checkpoint.stat().st_size,
              "parameter_sha256": signature["sha256"]}
    with (Path(metrics_file).parent / "checkpoint_records.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


class TrainingRecorder:
    def __init__(self, args: Namespace, model):
        self.path = Path(args.metrics_file)
        self.signature_path = self.path.parent / "epoch_signatures.jsonl"
        args_path = self.path.parent / "resolved_args.json"
        initial_path = self.path.parent / "initial_signature.json"
        for path in (self.path, self.signature_path, args_path, initial_path,
                     self.path.parent / "checkpoint_records.jsonl"):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite training evidence: {path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.signature_path.touch(exist_ok=False)
        resolved = dict(vars(args))
        resolved["resolved_model"] = {
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "per_class_critic": getattr(model, "per_class_critic", None),
            "with_two_state": getattr(model, "with_two_state", None),
        }
        write_json_new(args_path, resolved)
        write_json_new(initial_path, model_signature(model))
        self.total_steps = 0
        self.total_episodes = 0
        self.last_epoch = 0
        self._reset_epoch()

    def _reset_epoch(self):
        self.steps = 0
        self.episodes = 0
        self.success = 0.0
        self.steps_taken = 0.0
        self.reward = None
        self.policy_loss = 0.0
        self.value_loss = 0.0

    def add_batch(self, fresh):
        steps, episodes = int(fresh["num_steps"]), int(fresh["num_episodes"])
        if steps <= 0 or episodes <= 0:
            raise ValueError("A training batch must contain steps and completed episodes")
        self.steps += steps
        self.episodes += episodes
        self.success += float(fresh["success"])
        self.steps_taken += float(fresh["steps_taken"])
        reward = np.array(fresh["reward"], dtype=np.float64, copy=True)
        if reward.ndim != 1:
            raise ValueError("Expected the per-agent reward vector")
        self.reward = reward if self.reward is None else self.reward + reward
        self.policy_loss += float(fresh["action_loss"])
        self.value_loss += float(fresh["value_loss"])

    def finish_epoch(self, epoch: int, wall_time_seconds: float, model):
        if epoch != self.last_epoch + 1 or self.episodes <= 0 or self.steps <= 0:
            raise ValueError("Epoch records must be contiguous and contain fresh batches")
        self.total_steps += self.steps
        self.total_episodes += self.episodes
        metrics = {
            "epoch": epoch, "wall_time_seconds": float(wall_time_seconds),
            "steps": self.steps, "episodes": self.episodes,
            "total_steps": self.total_steps, "total_episodes": self.total_episodes,
            "success_rate": self.success / self.episodes,
            "steps_taken": self.steps_taken / self.episodes,
            "reward_per_agent": (self.reward / self.episodes).tolist(),
            "policy_loss": self.policy_loss / self.steps,
            "value_loss": self.value_loss / self.steps,
        }
        numeric = [v for k, v in metrics.items() if k != "reward_per_agent"] + metrics["reward_per_agent"]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"Non-finite training metrics at epoch {epoch}: {metrics}")
        with self.path.open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
        with self.signature_path.open("a") as stream:
            stream.write(json.dumps({"epoch": epoch, "signature": model_signature(model)},
                                    sort_keys=True, allow_nan=False) + "\n")
        self.last_epoch = epoch
        self._reset_epoch()
        return metrics
