"""Complete-update recovery for the isolated CPU reconstruction runtime.

RNG snapshots are taken only while every collector is idle.  The supplement's
stream derivation is opt-in; the public-code initialization is unchanged.
"""
from __future__ import annotations

import os
import math
from pathlib import Path
import random
import resource
import sys
import tempfile

import numpy as np
import torch


RECOVERY_VERSION = 1
RUNTIME_ARGUMENTS = {
    "metrics_file", "save_dir", "source_manifest", "resume_checkpoint",
    "wall_seconds", "episode_log", "save_every", "experiment_name", "load",
}


def rng_scheme(model_spec):
    return "seedsequence-v1" if model_spec == "supplement-v1" else "legacy-offset-v1"


def seed_stream(seed, collector_id=None):
    """Separate initialization from collectors, and libraries from each other.

    Keep the legacy global NumPy API used by the simulators.  SeedSequence only
    derives its initial multiword seed; no generator replacement is involved.
    """
    purpose = 0 if collector_id is None else 1
    index = 0 if collector_id is None else int(collector_id)
    for library in range(3):
        words = np.random.SeedSequence([library, purpose, index, int(seed)]).generate_state(4)
        integer = sum(int(word) << (32 * i) for i, word in enumerate(words))
        if library == 0:
            random.seed(integer)
        elif library == 1:
            np.random.seed(words)
        else:
            torch.manual_seed(integer & ((1 << 64) - 1))


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone()}


def restore_rng(state):
    if set(state) != {"python", "numpy", "torch"}:
        raise ValueError("Incomplete collector RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def process_resources(who=resource.RUSAGE_SELF):
    usage = resource.getrusage(who)
    result = {"max_rss": usage.ru_maxrss,
              "max_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
              "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime}
    if who == resource.RUSAGE_SELF:
        result.update(torch_threads=torch.get_num_threads(),
                      cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    return result


def scientific_arguments(args):
    return {key: value for key, value in vars(args).items() if key not in RUNTIME_ARGUMENTS}


def atomic_checkpoint(path, value):
    """Publish a complete checkpoint without replacing earlier evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # link is an atomic, exclusive publication on this same filesystem.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_recovery(checkpoint, args, source_sha256):
    reconstruction = checkpoint.get("reconstruction", {})
    recovery = checkpoint.get("recovery", {})
    if reconstruction.get("schema_version") != 2 or recovery.get("version") != RECOVERY_VERSION:
        raise ValueError("Checkpoint lacks validated complete-update recovery state")
    if recovery.get("stop_reason") in {"budget_completed", "epoch_cap_completed"}:
        raise ValueError("The checkpoint has already completed its locked training limits")
    if reconstruction.get("source_manifest_sha256") != source_sha256:
        raise ValueError("Recovery source identity differs from the checkpoint")
    if recovery.get("scientific_args") != scientific_arguments(args):
        raise ValueError("Recovery scientific configuration differs from the checkpoint")
    if reconstruction.get("model_spec") != args.model_spec:
        raise ValueError("Recovery model specification differs from the checkpoint")
    elapsed = recovery.get("active_time_seconds")
    if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid checkpoint active elapsed time")
    counts = recovery.get("counts", {})
    if counts != reconstruction.get("counts") or set(counts) != {"env_steps", "episodes", "updates", "epoch"}:
        raise ValueError("Checkpoint progress records disagree")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Invalid checkpoint progress counters")
    completed = recovery.get("completed_epochs")
    partial = recovery.get("updates_in_epoch")
    if (type(completed) is not int or completed < 0 or type(partial) is not int
            or not 0 <= partial < args.epoch_size
            or counts["updates"] != completed * args.epoch_size + partial):
        raise ValueError("Checkpoint update and epoch counters disagree")
    if counts["epoch"] != completed + bool(partial):
        raise ValueError("Checkpoint absolute epoch counter disagrees")
    recorder = recovery.get("recorder_state")
    if not isinstance(recorder, dict) or recorder.get("last_epoch") != completed:
        raise ValueError("Checkpoint recorder epoch disagrees")
    if (recorder.get("total_steps", -1) + recorder.get("steps", -1) != counts["env_steps"]
            or recorder.get("total_episodes", -1) + recorder.get("episodes", -1) != counts["episodes"]):
        raise ValueError("Checkpoint recorder counts disagree")
    epoch_stat = recovery.get("epoch_stat", {})
    if (epoch_stat.get("num_steps", 0) != recorder["steps"]
            or epoch_stat.get("num_episodes", 0) != recorder["episodes"]):
        raise ValueError("Checkpoint partial epoch statistics disagree")
    if len(recovery.get("rng_states", [])) != args.nprocesses:
        raise ValueError("Checkpoint collector RNG count differs")
    return recovery
