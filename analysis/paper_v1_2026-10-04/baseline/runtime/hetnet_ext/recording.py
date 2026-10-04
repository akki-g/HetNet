"""External accounting; never changes upstream collection, loss or optimizer.

Count each fresh train_batch statistic once. Upstream main.py instead adds the
growing epoch statistic on each iteration, overcounting cumulative samples.
Losses use upstream's reporting denominator: summed returned losses / joint
environment steps. These are diagnostics, not a redefinition of its objective.
"""
from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
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


def record_checkpoint(metrics_file, checkpoint: Path, epoch, wall_time_seconds, signature, *, counts=None):
    record = {"epoch": int(epoch), "path": str(checkpoint.resolve()),
              "wall_time_seconds": float(wall_time_seconds), "bytes": checkpoint.stat().st_size,
              "parameter_sha256": signature["sha256"]}
    record["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if counts is not None:
        record.update(counts=dict(counts), update=int(counts["updates"]))
    with (Path(metrics_file).parent / "checkpoint_records.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


class TrainingRecorder:
    def __init__(self, args: Namespace, model):
        self.path = Path(args.metrics_file)
        self.signature_path = self.path.parent / "epoch_signatures.jsonl"
        args_path = self.path.parent / "resolved_args.json"
        initial_path = self.path.parent / "initial_signature.json"
        for path in (self.path, self.signature_path, args_path, initial_path,
                     self.path.parent / "checkpoint_records.jsonl",
                     self.path.parent / "updates.jsonl", self.path.parent / "episodes.jsonl"):
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
        self.episode_log = getattr(args, "episode_log", "file")
        self._reset_epoch()

    def state_dict(self):
        names = ("total_steps", "total_episodes", "last_epoch", "steps", "episodes",
                 "success", "steps_taken", "reward", "policy_loss", "value_loss")
        return {name: copy.deepcopy(getattr(self, name)) for name in names}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError("Incomplete training recorder recovery state")
        for name, value in state.items():
            setattr(self, name, copy.deepcopy(value))

    def record_update(self, fresh, episodes, counts, epoch, update_in_epoch, wall_seconds):
        if len(episodes) != int(fresh["num_episodes"]):
            raise ValueError("Episode records do not match the completed update")
        records = []
        for index, episode in enumerate(episodes):
            records.append({**episode, "update": counts["updates"], "epoch": epoch,
                            "episode": counts["episodes"] - len(episodes) + index + 1})
        # Complete-update buffering bounds memory and performs one file open or
        # one flushed stdout write, independent of the number of episodes.
        if self.episode_log == "stdout":
            print(json.dumps({"record_type": "publication_episode_batch", "schema_version": 1,
                              "update": counts["updates"], "episodes": records},
                             sort_keys=True, allow_nan=False), flush=True)
        elif self.episode_log == "file":
            with (self.path.parent / "episodes.jsonl").open("a") as stream:
                stream.write("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                     for row in records))
        else:
            raise ValueError("Unsupported episode logging destination")
        reward = np.asarray(fresh["reward"], dtype=np.float64) / len(records)
        row = {"update": counts["updates"], "epoch": epoch, "update_in_epoch": update_in_epoch,
               "steps": int(fresh["num_steps"]), "episodes": len(records),
               "total_steps": counts["env_steps"], "total_episodes": counts["episodes"],
               "success_rate": float(fresh["success"]) / len(records),
               "reward_per_agent": reward.tolist(), "team_return": float(reward.sum()),
               "mean_agent_return": float(np.mean([r["mean_agent_return"] for r in records])),
               "policy_loss": float(fresh["action_loss"]) / fresh["num_steps"],
               "value_loss": float(fresh["value_loss"]) / fresh["num_steps"],
               "wall_time_seconds": float(wall_seconds)}
        with (self.path.parent / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

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

    def finish_epoch(self, epoch: int, wall_time_seconds: float, model, *, updates=None,
                     updates_in_epoch=None):
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
            "team_return": float(self.reward.sum() / self.episodes),
            "mean_agent_return": float(self.reward.mean() / self.episodes),
            "policy_loss": self.policy_loss / self.steps,
            "value_loss": self.value_loss / self.steps,
        }
        if updates is not None:
            metrics["updates"] = int(updates)
        if updates_in_epoch is not None:
            metrics["updates_in_epoch"] = int(updates_in_epoch)
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
