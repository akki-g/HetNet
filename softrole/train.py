"""Synchronous A2C: sum episode gradients, divide once, clip once, step once.

Workers receive frozen state snapshots and return ordinary gradient arrays. This
avoids shared gradient-pointer lifetimes and preserves the global episode average.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from hetnet_ext.signatures import model_signature
from softrole import CHECKPOINT_VERSION
from softrole.config import Config
from softrole.env import make_env
from softrole.learning import loss_sum
from softrole.model import SoftRoleNet
from softrole.optimizers import make_optimizer, optimizer_spec, validate_optimizer_resume
from softrole.rollout import run_episode
from softrole.scenarios import make_scenarios


def write_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def append_json(path, payload):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def log_episode_batch(path, episodes, update, mode="file"):
    """Preserve every record, with one file open or one stdout flush per update."""
    if mode == "file":
        with Path(path).open("a") as stream:
            for episode in episodes:
                stream.write(json.dumps(episode, sort_keys=True, allow_nan=False) + "\n")
    elif mode == "stdout":
        print(json.dumps({"record_type": "softrole_episode_batch", "schema_version": 1,
                          "update": update, "episodes": episodes},
                         sort_keys=True, allow_nan=False), flush=True)
    else:
        raise ValueError("episode_log must be file or stdout")


def source_snapshot(output=None):
    """Identify actual source bytes; optionally archive them in a fresh output.

    The read-only form lets evaluation record the same provenance as training.
    """
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files = {root / name for name in tracked if name and Path(name).suffix in
             (".py", ".sh", ".toml", ".lock", ".txt", ".sbatch")}
    files.update((root / "softrole").glob("*.py"))
    files.update((root / "tests").glob("test_softrole*.py"))
    files.update((root / "scripts").glob("softrole*"))
    files.update((root / "slurm").glob("softrole*"))
    files.update([root / "AGENTS.md", root / "softrole" / "RESEARCH.md"])
    from softrole.publication_env import SIMULATOR_FILES
    files.update(root / name for name in SIMULATOR_FILES)
    manifest = {}
    for path in sorted(files):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        data = path.read_bytes()
        if output is not None:
            target = Path(output) / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        manifest[str(relative)] = hashlib.sha256(data).hexdigest()
    source_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    record = {"sha256": source_id, "files": manifest,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root).decode()}
    if output is not None:
        write_json(Path(output) / "source_manifest.json", record)
    return record


def collect_batch(model, config, collector, update, environment_source=None):
    """Return *sums*, never locally normalized or clipped gradients."""
    model.train()
    model.zero_grad(set_to_none=True)
    adapter = make_env(config, environment_source)
    # Seed coordinates describe work, not process scheduling. Restarts and different
    # executor scheduling therefore do not alter a collector's scenario sequence.
    seed = int(np.random.SeedSequence([config.seed, update, collector, 2026]).generate_state(1)[0])
    scenarios = make_scenarios(seed, config.batch_steps, config.training_compositions,
                               config.failure_prob, config.failure_window)
    episodes, steps = [], 0
    policy_sum = value_sum = 0.0
    for scenario in scenarios:
        episode = run_episode(model, adapter, config, scenario, training=True)
        loss, losses = loss_sum([episode], config)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite episode loss")
        loss.backward()
        policy_sum += losses["policy_loss_sum"]
        value_sum += losses["value_loss_sum"]
        episodes.append(episode.metrics)
        steps += episode.metrics["steps"]
        if steps >= config.batch_steps:
            break
    gradients = [np.zeros(tuple(p.shape), dtype=np.float64) if p.grad is None
                 else p.grad.detach().cpu().numpy().copy() for p in model.parameters()]
    if not all(np.isfinite(g).all() for g in gradients):
        raise FloatingPointError("non-finite collector gradient")
    return {"gradients": gradients, "episodes": episodes, "num_steps": steps,
            "num_episodes": len(episodes), "policy_loss_sum": policy_sum,
            "value_loss_sum": value_sum, "collector": collector}


def worker_collect(state, model_config, config_dict, collector, update, environment_source=None):
    torch.set_num_threads(1)
    # Construction randomness is irrelevant after strict loading; communication and
    # actions use explicit scenario streams rather than Torch's global stream.
    model = SoftRoleNet(**model_config).double()
    model.load_state_dict(state, strict=True)
    return collect_batch(model, Config(**config_dict), collector, update, environment_source)


def apply_gradients(model, optimizer, batches, max_grad_norm):
    """(sum_w sum_e grad L_e) / (sum_w E_w), followed by one global clip."""
    count = sum(batch["num_episodes"] for batch in batches)
    if count <= 0:
        raise ValueError("cannot update without complete episodes")
    parameters = list(model.parameters())
    if any(len(batch["gradients"]) != len(parameters) for batch in batches):
        raise ValueError("collector gradient structure differs from model")
    optimizer.zero_grad(set_to_none=True)
    for index, parameter in enumerate(parameters):
        gradient = torch.zeros_like(parameter)
        for batch in batches:
            contribution = torch.as_tensor(batch["gradients"][index], dtype=parameter.dtype,
                                           device=parameter.device)
            if contribution.shape != parameter.shape:
                raise ValueError("collector gradient shape differs from parameter")
            gradient.add_(contribution)
        parameter.grad = gradient.div_(count)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def summarize(episodes):
    count = len(episodes)
    if not count:
        raise ValueError("an update must contain complete episodes")
    row = {"episodes": count, "steps": sum(ep["steps"] for ep in episodes),
           "success_rate": sum(ep["success"] for ep in episodes) / count,
           "steps_taken": sum(ep["steps"] for ep in episodes) / count,
           "team_return": sum(ep["team_return"] for ep in episodes) / count}
    # Average each episode's per-agent return before combining variable rosters.
    # Archived episode records predate the explicit diagnostic but retain N.
    row["mean_agent_return"] = sum(
        ep["mean_agent_return"] if ep.get("mean_agent_return") is not None
        else ep["team_return"] / ep["num_agents"] for ep in episodes) / count
    for key in ("return_nocap", "return_cap", "gate_entropy", "alpha_null", "num_agents"):
        values = [ep[key] for ep in episodes if ep.get(key) is not None]
        row[key] = float(np.mean(values)) if values else None
    row["event_exposed_episodes"] = sum(ep["event_exposed"] for ep in episodes)
    return row


def train(config, output, resume=None, episode_log="file"):
    """Run bounded training in a fresh directory; no changes to legacy entrypoints."""
    config.validate()
    if episode_log not in ("file", "stdout"):
        raise ValueError("episode_log must be file or stdout")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    model = SoftRoleNet(**config.model_kwargs()).double()
    optimizer = make_optimizer(model.parameters(), config)
    completed_updates = total_steps = total_episodes = completed_epochs = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint format")
        if checkpoint.get("environment_version") != config.env_version:
            raise ValueError("cannot resume a different environment version")
        validate_optimizer_resume(config, checkpoint)
        previous = Config(**checkpoint["config"]).to_dict()
        current = config.to_dict()
        for key in ("epochs", "total_steps", "save_every"):
            previous.pop(key)
            current.pop(key)
        if previous != current:
            raise ValueError("resume may change only epochs, total_steps, and save_every")
        if config.env_version == "paper-v1":
            from softrole.publication_env import checkpoint_binding, simulator_identity
            checkpoint_binding(resume, checkpoint)
            if simulator_identity() != checkpoint["simulator_source"]:
                raise ValueError("Resume requires unchanged paper simulator source")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed_updates = checkpoint["updates"]
        total_steps, total_episodes = checkpoint["total_steps"], checkpoint["total_episodes"]
        completed_epochs = completed_updates // config.updates_per_epoch
    if completed_epochs >= config.epochs or (config.total_steps is not None and total_steps >= config.total_steps):
        raise ValueError("the requested budget is already exhausted by the resume checkpoint")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    provenance = source_snapshot(output)
    environment_source = simulator_source = None
    if config.env_version == "paper-v1":
        from softrole.publication_env import source_binding, SIMULATOR_FILES
        environment_source = source_binding(output / "source")
        simulator_source = {key: environment_source[key] for key in ("sha256", "files")}
        if any(provenance["files"].get(name) != simulator_source["files"][name]
               for name in SIMULATOR_FILES):
            raise ValueError("Archived paper simulator differs from the source manifest")
    write_json(output / "config.json", config.to_dict())
    write_json(output / "initial_signature.json", model_signature(model))
    write_json(output / "run.json", {"environment_version": config.env_version,
               "simulator_source": simulator_source,
               "source_sha256": provenance["sha256"], "resume": str(resume) if resume else None,
               "parent_source_sha256": checkpoint["source_sha256"] if resume else None,
               "torch": torch.__version__, "numpy": np.__version__, "dtype": "float64",
               "episode_log": episode_log,
               "optimizer": optimizer_spec(config),
               "optimizer_initialization": "checkpoint" if resume else "fresh",
               "parameters": sum(p.numel() for p in model.parameters()),
               "actor_objective": "undiscounted sum of physical-agent rewards",
               "approximations": (["straight-through bits"] if config.communication == "binary" else []) +
                                  ["GAE with learned critic", "truncated BPTT"]})

    def save(epoch):
        if environment_source is not None:
            source_binding(environment_source["root"], simulator_source)
        path = output / "checkpoints" / f"epoch{epoch:04d}.pt"
        payload = {"format_version": CHECKPOINT_VERSION, "config": config.to_dict(),
                   "model_config": config.model_kwargs(), "model_state": model.state_dict(),
                   "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                   "optimizer": optimizer_spec(config),
                   "updates": completed_updates, "total_steps": total_steps,
                   "total_episodes": total_episodes, "environment_version": config.env_version,
                   "completed_epochs": completed_updates // config.updates_per_epoch,
                   "updates_in_partial_epoch": completed_updates % config.updates_per_epoch,
                   "source_sha256": provenance["sha256"], "signature": model_signature(model)}
        if simulator_source is not None:
            payload["simulator_source"] = simulator_source
        with path.open("xb") as stream:
            torch.save(payload, stream)
        write_json(Path(str(path) + ".signature.json"), payload["signature"])
        return path

    executor = (ProcessPoolExecutor(max_workers=config.nprocesses, mp_context=mp.get_context("spawn"))
                if config.nprocesses > 1 else None)
    checkpoint_path = None
    try:
        for epoch in range(completed_epochs + 1, config.epochs + 1):
            started = time.monotonic()
            epoch_episodes = []
            policy_sum = value_sum = 0.0
            # A step-budget stop may checkpoint partway through an epoch. Resume
            # the remaining updates instead of silently skipping that work.
            remaining_updates = config.updates_per_epoch - completed_updates % config.updates_per_epoch
            for _ in range(remaining_updates):
                if executor is None:
                    batches = [collect_batch(model, config, 0, completed_updates, environment_source)]
                else:
                    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    futures = [executor.submit(worker_collect, state, config.model_kwargs(),
                               config.to_dict(), rank, completed_updates, environment_source)
                               for rank in range(config.nprocesses)]
                    batches = [future.result() for future in futures]
                norm = apply_gradients(model, optimizer, batches, config.max_grad_norm)
                completed_updates += 1
                episode_records = []
                for batch in batches:
                    total_steps += batch["num_steps"]
                    total_episodes += batch["num_episodes"]
                    policy_sum += batch["policy_loss_sum"]
                    value_sum += batch["value_loss_sum"]
                    epoch_episodes.extend(batch["episodes"])
                    episode_records.extend(dict(episode, update=completed_updates,
                                                collector=batch["collector"])
                                           for episode in batch["episodes"])
                log_episode_batch(output / "episodes.jsonl", episode_records, completed_updates,
                                  episode_log)
                append_json(output / "updates.jsonl", {"update": completed_updates,
                    "episodes": sum(b["num_episodes"] for b in batches),
                    "steps": sum(b["num_steps"] for b in batches), "total_steps": total_steps,
                    "gradient_norm_before_clip": norm})
                if config.total_steps is not None and total_steps >= config.total_steps:
                    break
            row = summarize(epoch_episodes)
            row.update(epoch=epoch, total_steps=total_steps, total_episodes=total_episodes,
                       updates=completed_updates, wall_time_seconds=time.monotonic() - started,
                       partial_epoch=bool(completed_updates % config.updates_per_epoch),
                       policy_loss_per_episode=policy_sum / row["episodes"],
                       value_loss_per_episode=value_sum / row["episodes"])
            append_json(output / "metrics.jsonl", row)
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            exhausted = config.total_steps is not None and total_steps >= config.total_steps
            if epoch % config.save_every == 0 or epoch == config.epochs or exhausted:
                checkpoint_path = save(epoch)
            if exhausted:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    write_json(output / "finished.json", {"checkpoint": str(checkpoint_path.resolve()),
               "total_steps": total_steps, "total_episodes": total_episodes,
               "updates": completed_updates})
    return checkpoint_path
