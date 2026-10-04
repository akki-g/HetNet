"""Frozen checkpoint evaluation on explicit, reusable scenario records."""
import hashlib
import io
import json
from importlib.metadata import distributions
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from softrole.rollout import run_episode, validate_scenario


def model_signature(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluator_identity():
    """Keep evaluator source/runtime separate from the checkpoint's training source."""
    from softrole.train import source_snapshot
    return {
        "source": source_snapshot(),
        "runtime": {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
            "device": "cpu", "dtype": "float64",
            "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                    for dist in distributions() if dist.metadata["Name"])),
        },
    }


def evaluate_checkpoint(checkpoint, scenarios, output, intervention="none",
                        intervention_step=None, trace=False, sham=False,
                        protocol=None, scenarios_sha256=None):
    from softrole.config import Config
    from softrole.env import make_env
    from softrole.model import SoftRoleNet
    from softrole import CHECKPOINT_VERSION, EVALUATION_VERSION
    from softrole.train import source_snapshot

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    torch.set_num_threads(1)
    evaluator = evaluator_identity()
    checkpoint_bytes = Path(checkpoint).read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    if saved.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    config = Config(**saved["config"])
    if saved["model_config"] != config.model_kwargs():
        raise ValueError("Checkpoint model layout disagrees with its saved configuration")
    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("Evaluation requires at least one scenario")
    if protocol is not None:
        from softrole.report import validate_evaluation_protocol
        validate_evaluation_protocol(protocol, {"total_steps": saved.get("total_steps")},
                                     [asdict(s) for s in scenarios], scenarios_sha256)
        if protocol["distribution"]["task"] != config.task:
            raise ValueError("Evaluation protocol task differs from checkpoint")
    ids = [str(scenario.scenario_id) for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation scenario IDs must be unique")
    for scenario in scenarios:
        config.validate_composition((scenario.num_p, scenario.num_a))
        validate_scenario(config, scenario, intervention, intervention_step, sham)
    environment_source = None
    if config.env_version == "paper-v1":
        from softrole.publication_env import checkpoint_binding
        environment_source = checkpoint_binding(checkpoint, saved)
    adapter = make_env(config, environment_source)
    if saved.get("environment_version") != adapter.environment_version:
        raise ValueError("Checkpoint environment version differs from the evaluation adapter")
    model = SoftRoleNet(**saved["model_config"]).double()
    model.load_state_dict(saved["model_state"])
    model.eval()
    before = model_signature(model)
    episodes = []
    with torch.no_grad():
        for scenario in scenarios:
            episode = run_episode(model, adapter, config, scenario, training=False,
                                  intervention=intervention, intervention_step=intervention_step,
                                  trace=trace, sham=sham, event_diagnostics=config.task == "pcp")
            record = dict(episode.metrics)
            if trace:
                record["trace"] = episode.traces
            episodes.append(record)
    after = model_signature(model)
    if before != after:
        raise RuntimeError("Frozen evaluation changed model parameters or buffers")
    if evaluator["source"]["sha256"] != source_snapshot()["sha256"]:
        raise RuntimeError("Evaluator source changed during evaluation; keep the checkout fixed")
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != checkpoint_sha256:
        raise RuntimeError("Checkpoint file changed during frozen evaluation")
    if environment_source is not None:
        # Verify the archive again before publishing the evaluation record.
        checkpoint_binding(checkpoint, saved)
    exposed = [episode for episode in episodes if episode["event_exposed"]]
    scheduled = [episode for episode in episodes if episode["scheduled_event_exposed"]]
    report = {
        "checkpoint": str(Path(checkpoint)), "model_signature": before,
        "checkpoint_sha256": checkpoint_sha256,
        "format_version": CHECKPOINT_VERSION, "training_seed": config.seed,
        "evaluation_version": EVALUATION_VERSION, "evaluator": evaluator,
        "config": config.to_dict(), "model_config": saved["model_config"],
        "environment_version": adapter.environment_version,
        "simulator_source": saved.get("simulator_source"),
        "source_sha256": saved.get("source_sha256"),
        "checkpoint_progress": {key: saved.get(key) for key in
                                ("epoch", "updates", "total_steps", "total_episodes")},
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "intervention": intervention, "sham": bool(sham), "episodes": len(episodes),
        "success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "mean_team_return": sum(episode["team_return"] for episode in episodes) / len(episodes),
        "mean_agent_return": sum(episode["team_return"] / episode["num_agents"]
                                 for episode in episodes) / len(episodes),
        "event_exposed_episodes": len(exposed),
        "post_event_success_rate": sum(episode["success"] for episode in exposed) / len(exposed) if exposed else None,
        "scheduled_event_exposed_episodes": len(scheduled),
        "post_schedule_success_rate": sum(episode["success"] for episode in scheduled) / len(scheduled) if scheduled else None,
        "per_episode": episodes,
    }
    if scenarios_sha256 is not None:
        report["scenarios_sha256"] = scenarios_sha256
    if protocol is not None:
        report["evaluation_protocol"] = protocol
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
