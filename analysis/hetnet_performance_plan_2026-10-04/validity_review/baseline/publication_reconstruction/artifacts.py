"""Read-only validation of an archived reconstruction run and its checkpoints."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source_archive(run, manifest):
    source = (Path(run) / "source").resolve()
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files or "runtime/main.py" not in files:
        raise ValueError("Missing archived runtime inventory")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Archived source path leaves the archive: {name}")
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Archived source mismatch: {name}")
    actual = {p.relative_to(source).as_posix() for p in (source / "runtime").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    declared = {name for name in files if name.startswith("runtime/")}
    if actual != declared:
        raise ValueError("Archived runtime inventory differs from its manifest")


def verify_run(run):
    run = Path(run).resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    manifest = json.loads((run / "source_manifest.json").read_text())
    if not isinstance(protocol, dict) or not isinstance(manifest, dict):
        raise ValueError("Invalid reconstruction protocol or source manifest")
    verify_source_archive(run, manifest)
    return protocol, manifest


def _scientific_command(command):
    from .runtime.hetnet_ext.recovery import RUNTIME_ARGUMENTS
    if not isinstance(command, list) or len(command) < 3 or not all(isinstance(x, str) for x in command):
        raise ValueError("Invalid archived training command")
    result = {}
    index = 3
    while index < len(command):
        flag = command[index]
        if not flag.startswith("--"):
            raise ValueError("Malformed archived training command")
        key = flag[2:]
        index += 1
        values = []
        while index < len(command) and not command[index].startswith("--"):
            values.append(command[index])
            index += 1
        if key not in RUNTIME_ARGUMENTS:
            if key in result:
                raise ValueError("Duplicate scientific command flag")
            result[key] = values
    return result


def _validate_scientific_protocol(protocol, args):
    """Cross-check the declared protocol, executable command and saved arguments."""
    from .__main__ import resolve
    try:
        choices = {"task": ("pp", "pcp", "fc"), "variant": ("real", "binary"),
                   "model_spec": ("public-code-v1", "supplement-v1"),
                   "env_version": ("historical-2022", "corrected-v1"),
                   "recipe": ("june-2022", "october-2022")}
        if any(protocol[key] not in values for key, values in choices.items()):
            raise ValueError("Unknown scientific protocol version or domain")
        expected = resolve(SimpleNamespace(
            task=protocol["task"], variant=protocol["variant"], model_spec=protocol["model_spec"],
            seed=protocol["seed"], env_version=protocol["env_version"], recipe=protocol["recipe"],
            output=Path(protocol["output"]), epochs=protocol["epochs"], collectors=protocol["collectors"],
            updates_per_epoch=protocol["updates_per_epoch"], batch_steps=protocol["batch_step_floor_per_collector"],
            horizon=protocol["episode_horizon"], save_every=args["save_every"],
            max_env_steps=protocol["max_env_steps"], milestones=protocol["milestones"],
            episode_log=protocol["episode_log"], wall_seconds=protocol["wall_seconds"]))
        for key in ("model_spec", "optimizer", "rng_scheme", "learner_spec"):
            if protocol[key] != expected[key]:
                raise ValueError(f"Scientific protocol mismatch: {key}")
        command = _scientific_command(protocol["command"])
        if command != _scientific_command(expected["command"]):
            raise ValueError("Scientific command differs from its declared protocol")
        for key, values in command.items():
            value = args[key]
            if not values:
                matches = value is True
            elif isinstance(value, (list, tuple)):
                matches = list(map(str, value)) == values
            elif isinstance(value, float):
                matches = len(values) == 1 and float(values[0]) == value
            else:
                matches = values == [str(value)]
            if not matches:
                raise ValueError(f"Scientific command/checkpoint mismatch: {key}")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete scientific protocol or checkpoint arguments") from error


def _validate_optimizer(saved, protocol, updates):
    import math
    import torch
    optimizer = saved.get("trainer")
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict)
            or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]):
        raise ValueError("Checkpoint lacks active optimizer state")
    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite(optimizer):
        raise ValueError("Checkpoint optimizer state is nonfinite")
    expected = protocol["optimizer"]
    parameters = []
    for group in optimizer["param_groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list) or not group["params"]:
            raise ValueError("Invalid optimizer parameter group")
        parameters.extend(group["params"])
        for key in ("lr", "alpha", "epsilon", "betas", "weight_decay", "amsgrad", "foreach", "fused"):
            if key in expected:
                actual = group.get("eps" if key == "epsilon" else key)
                declared = tuple(expected[key]) if key == "betas" else expected[key]
                if actual != declared:
                    raise ValueError(f"Checkpoint optimizer differs from protocol: {key}")
    if (any(type(p) is not int for p in parameters) or len(parameters) != len(set(parameters))
            or not set(optimizer["state"]).issubset(parameters)):
        raise ValueError("Checkpoint optimizer parameter identity is inconsistent")
    moment_keys = {"exp_avg", "exp_avg_sq"} if expected["name"] == "Adam" else {"square_avg"}
    if updates and (not optimizer["state"] or any(not isinstance(state, dict) or not moment_keys.issubset(state)
                                                 for state in optimizer["state"].values())):
        raise ValueError("Checkpoint optimizer moments are missing")


def _validate_recovery(saved, protocol):
    import random
    import numpy as np
    import torch
    from .runtime.hetnet_ext.recovery import scientific_arguments
    meta = saved["reconstruction"]
    counts = meta["counts"]
    recovery = saved.get("recovery", {})
    if (meta["schema_version"] != 2 or not isinstance(recovery, dict) or recovery.get("version") != 1
            or recovery.get("counts") != counts):
        raise ValueError("Checkpoint lacks supported continuation state")
    if (meta.get("model_spec") != protocol["model_spec"] or meta.get("rng_scheme") != protocol["rng_scheme"]
            or recovery.get("scientific_args") != scientific_arguments(SimpleNamespace(**meta["resolved_args"]))):
        raise ValueError("Checkpoint recovery scientific configuration differs")
    completed, partial = recovery.get("completed_epochs"), recovery.get("updates_in_epoch")
    if (set(counts) != {"env_steps", "episodes", "updates", "epoch"}
            or type(completed) is not int or completed < 0 or type(partial) is not int
            or not 0 <= partial < protocol["updates_per_epoch"]
            or counts["updates"] != completed * protocol["updates_per_epoch"] + partial
            or counts["epoch"] != completed + bool(partial)):
        raise ValueError("Checkpoint update and epoch counters disagree")
    if (counts["env_steps"] >= (protocol.get("max_env_steps") or float("inf"))
            or recovery.get("stop_reason") == "budget_completed"):
        raise ValueError("The checkpoint has already completed its scientific step budget")
    if completed >= protocol["epochs"] or recovery.get("stop_reason") == "epoch_cap_completed":
        raise ValueError("The checkpoint has already completed its epoch budget")
    states = recovery.get("rng_states", [])
    if not isinstance(states, list) or len(states) != protocol["collectors"]:
        raise ValueError("Checkpoint collector RNG state is incomplete")
    try:
        for state in states:
            if set(state) != {"python", "numpy", "torch"}:
                raise ValueError("Incomplete RNG state")
            random.Random().setstate(state["python"])
            np.random.RandomState().set_state(state["numpy"])
            torch.Generator().set_state(state["torch"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint collector RNG state is invalid") from error
    recorder = recovery.get("recorder_state", {})
    if (not isinstance(recorder, dict) or recorder.get("last_epoch") != completed
            or counts["env_steps"] != recorder.get("total_steps", -1) + recorder.get("steps", -1)
            or counts["episodes"] != recorder.get("total_episodes", -1) + recorder.get("episodes", -1)):
        raise ValueError("Checkpoint recorder progress disagrees")
    _validate_optimizer(saved, protocol, counts["updates"])


def load_checkpoint(run, checkpoint, require_recovery=False):
    """Load our trusted research artifact only after validating its source record."""
    import torch
    protocol, _ = verify_run(run)
    payload = Path(checkpoint).read_bytes()
    try:
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("Checkpoint cannot be decoded") from error
    if not isinstance(saved, dict):
        raise ValueError("Invalid checkpoint payload")
    meta = saved.get("reconstruction", {})
    if not isinstance(meta, dict) or meta.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported reconstruction checkpoint schema")
    if meta.get("source_manifest_sha256") != digest(Path(run) / "source_manifest.json"):
        raise ValueError("Checkpoint source manifest mismatch")
    args = meta.get("resolved_args", {})
    if not isinstance(args, dict):
        raise ValueError("Invalid checkpoint arguments")
    pairs = {"seed": "seed", "nprocesses": "collectors", "epoch_size": "updates_per_epoch",
             "batch_size": "batch_step_floor_per_collector", "max_steps": "episode_horizon",
             "num_epochs": "epochs"}
    for arg, field in pairs.items():
        if args.get(arg) != protocol.get(field):
            raise ValueError(f"Checkpoint/protocol mismatch: {field}")
    if (saved.get("seed") != protocol.get("seed") or
            meta.get("env_version") != protocol.get("env_version") or
            args.get("publication_env_version") != protocol.get("env_version") or
            bool(args.get("use_binary")) != (protocol.get("variant") == "binary") or
            args.get("model_spec", "public-code-v1") != protocol.get("model_spec", "public-code-v1") or
            args.get("max_env_steps", 0) != (protocol.get("max_env_steps") or 0)):
        raise ValueError("Checkpoint scientific configuration differs from run protocol")
    counts = meta.get("counts", {})
    if not isinstance(counts, dict) or any(type(counts.get(k)) is not int or counts[k] < 0
           for k in ("env_steps", "episodes", "updates", "epoch")):
        raise ValueError("Invalid checkpoint progress")
    model = saved.get("policy_net")
    if not isinstance(model, dict) or not model or not all(torch.is_tensor(t) and torch.isfinite(t).all() for t in model.values()):
        raise ValueError("Missing or nonfinite checkpoint model")
    if protocol.get("schema_version") == 2:
        _validate_scientific_protocol(protocol, args)
    if require_recovery:
        _validate_recovery(saved, protocol)
    return saved


def lineage_metrics(run):
    """Yield only the retained parent prefix plus this immutable segment's epochs."""
    run = Path(run).resolve()
    visited = set()

    def collect(path, limit=None):
        if path in visited:
            raise ValueError("Continuation lineage cycle")
        visited.add(path)
        protocol = json.loads((path / "protocol.json").read_text())
        rows = []
        parent = protocol.get("continuation")
        if parent:
            retained = parent["completed_epochs"]
            if type(retained) is not int or retained < 0:
                raise ValueError("Invalid retained parent epoch count")
            rows = collect(Path(parent["run_dir"]).resolve(),
                           retained if limit is None else min(limit, retained))
        metric_path = path / "metrics.jsonl"
        if metric_path.exists() and (limit is None or len(rows) < limit):
            with metric_path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
                    if limit is not None and len(rows) == limit:
                        break
        epochs = [row["epoch"] for row in rows]
        if epochs != list(range(1, len(rows) + 1)):
            raise ValueError("Continuation epoch ledger is not contiguous")
        if limit is not None and len(rows) != limit:
            raise ValueError("Continuation retained prefix is incomplete")
        return rows

    return collect(run)
