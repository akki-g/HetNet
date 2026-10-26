"""Isolated archived-runtime evaluator. Only evaluation.py launches this worker."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys


def signature(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def sensory_mask(observation, victim, base):
    """A private typed observation copy; physical classes/positions are untouched."""
    masked = observation.copy()
    masked[victim].reshape(-1, base + 4)[:, base:] = 0
    return masked


def singleton_batch(_module, inputs):
    x, hidden = inputs
    if x.ndim == 1 and hidden[0].ndim == 2:
        return x.unsqueeze(0), hidden


def run(request):
    runtime, root = Path(request["runtime"]).resolve(), Path(request["root"]).resolve()
    sys.path[:0] = [str(runtime), str(runtime / "envs"), str(root)]
    import numpy as np
    import torch
    from hetgat.uavnet import UAVNetA2CEasy
    from hetgat.utils import build_hetgraph
    from ic3net_envs.predator_prey_env import PredatorPreyEnv
    from ic3net_envs.predator_capture_env import PredatorCaptureEnv
    from ic3net_envs.fire_commander_env import FireCommanderEnv
    from softrole.scenarios import IsolatedRNG, Scenario
    from softrole.rollout import sample_actions
    from softrole.report import validate_evaluation_protocol
    from publication_reconstruction.artifacts import load_checkpoint

    imported = {obj.__name__: str(Path(inspect.getfile(obj)).resolve()) for obj in
                (UAVNetA2CEasy, build_hetgraph, PredatorPreyEnv, PredatorCaptureEnv, FireCommanderEnv)}
    if any(not Path(path).is_relative_to(runtime) for path in imported.values()):
        raise ValueError("Evaluation imported a model/environment outside the archived runtime")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    checkpoint_bytes = Path(request["checkpoint"]).read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != request["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed before worker loading")
    saved = load_checkpoint(request["run_dir"], request["checkpoint"])
    meta = saved.get("reconstruction", {})
    if meta.get("schema_version") not in (1, 2):
        raise ValueError("Expected a supported reconstruction checkpoint schema")
    if meta.get("source_manifest_sha256") != request["training_source_sha256"]:
        raise ValueError("Checkpoint does not identify the supplied run's source manifest")
    values = meta.get("resolved_args", {})
    version = meta.get("env_version")
    if version not in ("historical-2022", "corrected-v1", "paper-v1") or values.get("publication_env_version") != version:
        raise ValueError("Checkpoint environment version is missing or inconsistent")
    if not isinstance(saved.get("seed"), int) or saved["seed"] != values.get("seed"):
        raise ValueError("Checkpoint training seed is missing or inconsistent")
    for flag in ("tensor_obs", "moving_prey", "enemy_comm", "lossy_comm", "no_stay", "shared_reward"):
        if values.get(flag, False):
            raise ValueError(f"Frozen reconstruction evaluation does not support {flag}")
    if values.get("mode", "mixed") != "mixed" or values.get("A_vision", -1) != -1:
        raise ValueError("Evaluation supports the released mixed-reward/blind-A recipes")
    if any(values.get(name, -1) != -1 for name in ("comm_range_P", "comm_range_A")):
        raise ValueError("Evaluation currently supports all-to-all communication only")
    env_name = values.get("env_name")
    if env_name not in ("predator_prey", "predator_capture", "fire_commander"):
        raise ValueError("Unsupported reconstruction environment")
    p, a = values["nfriendly_P"], values["nfriendly_A"]
    task = "fc" if env_name == "fire_commander" else ("pp" if a == 0 else "pcp")
    if env_name == "predator_prey" and a != 0:
        raise ValueError("PredatorPrey checkpoints must have zero A agents")
    config = {"model": "HetNet-Binary" if values.get("use_binary") else "HetNet-Real",
              "task": task, "seed": saved["seed"], "num_p": p, "num_a": a,
              "dim": values["dim"], "vision": values["vision"], "max_steps": values["max_steps"],
              "msg_dim": values.get("msg_dim", 16), "reward_type": values.get("reward_type", 3),
              "nfires": values.get("nfires", 1), "comm_range": -1,
              "environment_class": env_name, "publication_env_version": version,
              "second_reward_scheme": bool(values.get("second_reward_scheme", False))}
    counts = meta.get("counts", {})
    progress = {key: counts.get(original) for key, original in
                (("epoch", "epoch"), ("updates", "updates"), ("total_steps", "env_steps"),
                 ("total_episodes", "episodes"))}
    if any(type(value) is not int or value < 0 for value in progress.values()):
        raise ValueError("Checkpoint needs nonnegative integer training counts")
    model_spec = meta.get("model_spec", values.get("model_spec", "public-code-v1"))
    if model_spec not in ("public-code-v1", "supplement-v1", "paper-v1"):
        raise ValueError("Unsupported checkpoint model specification")
    if values.get("model_spec", model_spec) != model_spec:
        raise ValueError("Inconsistent checkpoint model specification")
    config["model_spec"] = model_spec
    config["message_backend"] = values.get("message_backend", "dgl")
    if model_spec == "paper-v1":
        from hetgat.paper import PaperNet
        imported["PaperNet"] = str(Path(inspect.getfile(PaperNet)).resolve())
        if not Path(imported["PaperNet"]).is_relative_to(runtime):
            raise ValueError("Paper model was not imported from its source archive")
        config["binary_bandwidth"] = {"bits_per_head": 64, "heads": 4, "bits_per_sender_per_round": 256}
    else:
        PaperNet = None
    protocol = json.loads((Path(request["run_dir"]) / "protocol.json").read_text())
    if protocol.get("task") != task:
        raise ValueError("Checkpoint task differs from the run protocol")
    if meta["schema_version"] == 1:
        config["learner_spec"] = "public-code-v1"
        config["rng_scheme"] = "legacy-offset-v1"
        config["version_metadata_basis"] = "schema-v1 public-code compatibility defaults"
    else:
        config["learner_spec"] = protocol.get("learner_spec", "public-code-v1")
        config["rng_scheme"] = meta.get("rng_scheme", protocol.get("rng_scheme"))
        config["version_metadata_basis"] = "checkpoint and run protocol"
        if config["learner_spec"] not in ("public-code-v1", "paper-equations-v1") or config["rng_scheme"] not in ("seedsequence-v1", "legacy-offset-v1"):
            raise ValueError("Unsupported checkpoint learner/RNG specification")
        if (protocol.get("rng_scheme", config["rng_scheme"]) != config["rng_scheme"]
                or meta.get("learner_spec", config["learner_spec"]) != config["learner_spec"]):
            raise ValueError("Checkpoint learner/RNG specification differs from the run protocol")
    if any(type(config[name]) is not int or config[name] <= 0 for name in ("dim", "max_steps", "msg_dim", "num_p")):
        raise ValueError("Checkpoint dimensions and horizon must be positive integers")
    if type(config["vision"]) is not int or config["vision"] < 0 or type(a) is not int or a < 0:
        raise ValueError("Checkpoint vision and A count must be nonnegative integers")
    scenarios = [Scenario(**row) for row in request["scenarios"]]
    if len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("Scenario IDs must be unique")
    for scenario in scenarios:
        ints = (scenario.num_p, scenario.num_a, scenario.event_step, scenario.victim,
                scenario.env_seed, scenario.action_seed, scenario.message_seed)
        if any(type(x) is not int for x in ints):
            raise ValueError("Scenario counts, timing and seeds must be integers")
        if min(scenario.env_seed, scenario.action_seed, scenario.message_seed) < 0:
            raise ValueError("Scenario seeds must be nonnegative")
        if scenario.num_p < 1 or scenario.num_a < 0 or (task != "pp" and scenario.num_a < 1):
            raise ValueError("Invalid physical composition")
        targets = config["nfires"] if task == "fc" else values.get("nenemies", 1)
        if scenario.num_p + scenario.num_a + targets > config["dim"] ** 2:
            raise ValueError("Scenario composition does not fit the grid")
        if task != "pcp" and (scenario.num_p, scenario.num_a) != (p, a):
            raise ValueError("Composition transfer is supported only for PCP")
        if not -1 <= scenario.event_step < config["max_steps"]:
            raise ValueError("Event step is outside the task horizon")
        if scenario.event_step >= 0 and (task != "pcp" or not 0 <= scenario.victim < scenario.num_p):
            raise ValueError("Sensor failures require a PCP sensing victim")
        if scenario.event_step == -1 and scenario.victim != -1:
            raise ValueError("No-event scenarios must not assign a victim")
        if request["sham"] and scenario.event_step < 0:
            raise ValueError("Sham evaluation requires a scheduled event")
    if request["evaluation_protocol"] is not None:
        if request["evaluation_protocol"]["distribution"]["task"] != task:
            raise ValueError("Evaluation protocol task differs from checkpoint")
        validate_evaluation_protocol(request["evaluation_protocol"], progress,
                                     request["scenarios"], request["scenarios_sha256"])
    environment_version = f"publication-{version}:{env_name}"
    models, environments, identities, records = {}, {}, {}, []

    def model_for(team):
        base = config["dim"] ** 2
        model_type = PaperNet if model_spec == "paper-v1" else UAVNetA2CEasy
        kwargs = dict(num_P=team[0], num_A=team[1], num_heads=4, msg_dim=config["msg_dim"],
                      use_CNN=False, use_real=not values.get("use_binary", False), use_tanh=False,
                      per_class_critic=True, per_agent_critic=False, device=torch.device("cpu"),
                      with_two_state=True, obs=(2 * config["vision"] + 1) ** 2, action_vision=-1)
        if "model_spec" in inspect.signature(model_type).parameters:
            kwargs["model_spec"] = model_spec
        elif model_spec != "public-code-v1":
            raise ValueError("Archived model does not accept its recorded model_spec")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            if model_spec == "paper-v1":
                kwargs["message_backend"] = config["message_backend"]
            model = model_type(dict(vision=config["vision"], P=base + 4, A=base, state=4),
                                 dict(P=base + 4, A=base, state=4), dict(P=16, A=16, state=16),
                                 dict(P=4, A=5, state=8) if model_spec == "paper-v1" and task == "fc"
                                 else dict(P=5, A=6, state=8), **kwargs)
        model.load_state_dict(saved["policy_net"], strict=True)
        for name, tensor in model.state_dict().items():
            original = saved["policy_net"][name]
            if tensor.dtype != original.dtype or not torch.equal(tensor, original) or not torch.isfinite(tensor).all():
                raise ValueError("Checkpoint tensors must remain finite and exactly unchanged")
        if model_spec != "paper-v1":
            model.f_module_stat.register_forward_pre_hook(singleton_batch)
            model.f_module_obs.register_forward_pre_hook(singleton_batch)
        return model.eval()

    with torch.no_grad():
        for scenario in scenarios:
            team = (scenario.num_p, scenario.num_a)
            if team not in models:
                models[team] = model_for(team)
                identities[team] = signature(models[team])
                cls = {"predator_prey": PredatorPreyEnv, "predator_capture": PredatorCaptureEnv,
                       "fire_commander": FireCommanderEnv}[env_name]
                env = cls()
                parser = argparse.ArgumentParser()
                env.init_args(parser)
                args = parser.parse_args([])
                vars(args).update(values)
                args.nfriendly_P, args.nfriendly_A = team
                args.nagents = args.nfriendly = sum(team)
                args.eval = False
                env.multi_agent_init(args)
                environments[team] = env
            model, env = models[team], environments[team]
            rng = IsolatedRNG(scenario.env_seed)
            with rng.use():
                raw = np.array(env.reset(), copy=True)
            initial = {"P": env.predator_loc.tolist(),
                       "A": getattr(env, "predator_capture_loc", np.empty((0, 2), dtype=int)).tolist(),
                       "targets": (env.fire_loc if task == "fc" else env.prey_loc).tolist()}
            memory = model.init_hidden(1)
            returns, trace = np.zeros(sum(team), dtype=float), []
            seen = exposed = scheduled = False
            diagnostics = dict(pre_event_victim_reached=None, pre_event_victim_target_visible=None,
                               pre_event_victim_target_seen=None)
            for step in range(config["max_steps"]):
                if task == "pcp" and 0 <= step <= scenario.event_step:
                    visible = bool(np.any(raw[scenario.victim].reshape(-1, config["dim"] ** 2 + 4)[:, config["dim"] ** 2 + 2] > 0))
                    if step == scenario.event_step:
                        diagnostics.update(pre_event_victim_reached=bool(env.reached_prey[scenario.victim]),
                                           pre_event_victim_target_visible=visible,
                                           pre_event_victim_target_seen=seen)
                    else:
                        seen |= visible
                if step == scenario.event_step:
                    scheduled = True
                    exposed = not request["sham"]
                obs = sensory_mask(raw, scenario.victim, config["dim"] ** 2) if exposed else raw.copy()
                positions = np.vstack((env.predator_loc, getattr(env, "predator_capture_loc", np.empty((0, 2), dtype=int))))
                one_hot = np.eye(config["dim"] ** 2)[(positions[:, 0] * config["dim"] + positions[:, 1]).astype(int)]
                graph = None
                if config["message_backend"] == "dgl":
                    graph = build_hetgraph(one_hot, num_P=team[0], num_A=team[1], with_state=True,
                                          with_two_state=True, with_self_loop=False, comm_range_P=-1, comm_range_A=-1)
                if model_spec == "paper-v1":
                    model.set_episode_step(step)
                key = np.random.SeedSequence([scenario.message_seed, step])
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(key.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 1))
                    result, _, _, memory = model([torch.as_tensor(obs, dtype=torch.float64).reshape(1, sum(team), -1), memory], graph)
                logits = torch.cat((result["P"], result["P"].new_full((team[0], 1), -torch.inf)), dim=1)
                if team[1]:
                    logits = torch.cat((logits, result["A"]), dim=0)
                actions, _ = sample_actions(logits, scenario.action_seed, step)
                with rng.use():
                    next_obs, reward, done, _ = env.step(actions.numpy())
                raw = np.array(next_obs, copy=True)
                returns += reward
                if request["trace"]:
                    trace.append({"step": step, "actions": actions.tolist(), "team_reward": float(np.sum(reward))})
                if done:
                    break
            steps, success = step + 1, bool(env.stat.get("success", False))
            record = {**asdict(scenario), "composition": list(team), "num_agents": sum(team),
                      "steps": steps, "success": success, "terminated": bool(done),
                      "truncated": bool(not done and steps == config["max_steps"]),
                      "team_return": float(returns.sum()), "agent_returns": returns.tolist(),
                      "mean_agent_return": float(returns.mean()), "initial_state": initial,
                      "return_nocap": float(returns[:team[0]].mean()),
                      "return_cap": float(returns[team[0]:].mean()) if team[1] else None,
                      "sham": request["sham"], "event_exposed": exposed,
                      "scheduled_event_exposed": scheduled,
                      "pre_event_success": bool(success and scenario.event_step >= 0 and not scheduled),
                      "scheduled_completion_steps": steps - scenario.event_step if scheduled and success else None,
                      "scheduled_censored": bool(scheduled and not success),
                      "post_schedule_steps": steps - scenario.event_step if scheduled else 0,
                      "recovery_steps": steps - scenario.event_step if exposed and success else None,
                      "recovery_censored": bool(exposed and not success),
                      "post_event_steps": steps - scenario.event_step if exposed else 0,
                      "intervention": "none", "intervention_step": scenario.event_step,
                      "environment_version": environment_version, **diagnostics}
            if request["trace"]:
                record["trace"] = trace
            records.append(record)
    if any(signature(model) != identities[team] for team, model in models.items()) or len(set(identities.values())) != 1:
        raise RuntimeError("Frozen evaluation or composition changed checkpoint tensors")
    report = {"checkpoint": request["checkpoint"], "checkpoint_sha256": request["checkpoint_sha256"],
              "model_signature": next(iter(identities.values())), "parameters_unchanged": True,
              "evaluation_version": 2, "evaluator_kind": "reconstruction-hetnet-v1",
              "evaluator": {"source": request["evaluator_source"], "runtime": {
                  "python": sys.version, "platform": platform.platform(), "executable": sys.executable,
                  "device": "cpu", "input_dtype": "float64",
                  "parameter_dtypes": sorted({str(t.dtype) for t in saved["policy_net"].values()}),
                  "torch_threads": torch.get_num_threads(),
                  "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
                  "imported_paths": imported}},
              "source_sha256": request["training_source_sha256"], "training_seed": saved["seed"],
              "checkpoint_progress": progress, "config": config, "model_config": model_spec,
              "environment_version": environment_version, "scenarios": request["scenarios"],
              "scenarios_sha256": request["scenarios_sha256"], "intervention": "none", "sham": request["sham"],
              "episodes": len(records), "per_episode": records,
              "success_rate": float(np.mean([r["success"] for r in records])),
              "mean_team_return": float(np.mean([r["team_return"] for r in records])),
              "mean_agent_return": float(np.mean([r["mean_agent_return"] for r in records])),
              "event_exposed_episodes": sum(r["event_exposed"] for r in records),
              "scheduled_event_exposed_episodes": sum(r["scheduled_event_exposed"] for r in records),
              "comparison": "Typed observations; original HetNet has no explicit current-health input"}
    if request["evaluation_protocol"] is not None:
        report["evaluation_protocol"] = request["evaluation_protocol"]
    return report


if __name__ == "__main__":
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = run(request)
    print(json.dumps(result, allow_nan=False))
