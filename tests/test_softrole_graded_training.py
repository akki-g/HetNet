"""B1: graded descriptors reach the unchanged shared learner and replay exactly."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing as mp

import numpy as np
import pytest
import torch

from softrole.config import Config
from softrole.env import make_env
from softrole.evaluate import evaluate_checkpoint, model_signature
from softrole.model import SoftRoleNet
from softrole.rollout import run_episode
from softrole.scenarios import GradedScenario, Scenario
from softrole.train import collect_batch, train, worker_collect


def graded_config(**overrides):
    # Small version of the intended shared/decomposed arm; no training-quality claim.
    settings = dict(task="pcp", model="shared", capabilities="graded", descriptor="capability",
                    sensing_p=(1, 3), sensing_a=(0,), held_out_sensing_p=(2,),
                    held_out_sensing_a=(1, 2, 3), num_p=2, num_a=1, dim=5, vision=2,
                    max_steps=3, pre_dim=8, hidden_dim=8, heads=2, head_dim=4, msg_dim=3,
                    comm_rounds=3, communication="real", independent_heads=True,
                    critic_mode="decomposed", credit_beta=1., optimizer="adam", lr=1e-3,
                    seed=23, epochs=1, updates_per_epoch=3, batch_steps=2,
                    nprocesses=1, save_every=1, detach_gap=2)
    settings.update(overrides)
    return Config(**settings)


def assert_state_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_state_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_state_equal(left, right)
    else:
        assert first == second


@pytest.mark.parametrize("critic_mode", ["team", "decomposed"])
def test_native_graded_rollout_matches_binary_logits_values_actions_and_metrics(critic_mode):
    binary = graded_config(capabilities="binary", sensing_p=(), sensing_a=(),
                           held_out_sensing_p=(), held_out_sensing_a=(),
                           communication="binary", comm_rounds=2, independent_heads=False,
                           critic_mode=critic_mode)
    graded = replace(binary, capabilities="graded")
    scenario = Scenario("native", 11, 12, 13, 2, 1, event_step=1, victim=0)
    graded_scenario = GradedScenario(**asdict(scenario), sensing=(3, 3, 0))
    torch.manual_seed(83)
    model = SoftRoleNet(**binary.model_kwargs()).double()
    captured = []

    def record_outputs(module, args, output):
        captured.append({key: output[key].detach().clone() for key in ("logits", "value")})

    handle = model.register_forward_hook(record_outputs)
    try:
        original = run_episode(model, make_env(binary), binary, scenario, training=False, trace=True)
        boundary = len(captured)
        native = run_episode(model, make_env(graded), graded, graded_scenario, training=False, trace=True)
    finally:
        handle.remove()
    assert_state_equal(captured[:boundary], captured[boundary:])
    assert original.traces == native.traces
    assert "sensing" not in original.metrics
    assert native.metrics["sensing"] == [3, 3, 0]
    assert original.metrics == {key: value for key, value in native.metrics.items() if key != "sensing"}
    assert original.rewards == native.rewards


@pytest.mark.parametrize("descriptor", ["capability", "type"])
def test_rollout_keeps_physical_failure_separate_from_actor_and_critic_descriptor(monkeypatch, descriptor):
    config = graded_config(descriptor=descriptor)
    # The victim is a capture agent that senses, outside the native binary roster.
    scenario = GradedScenario("capture-sensor", 11, 12, 13, 2, 1,
                              event_step=1, victim=2, sensing=(1, 3, 2))
    adapter = make_env(config)
    torch.manual_seed(92)
    model = SoftRoleNet(**config.model_kwargs()).double()
    inputs, critic_agents, critic_teams, logits, physical_steps = [], [], [], [], []
    handles = [
        model.register_forward_pre_hook(lambda module, args: inputs.append(
            (args[0].detach().clone(), args[1].detach().clone()))),
        model.critic_agent.register_forward_pre_hook(lambda module, args: critic_agents.append(
            args[0].detach().clone())),
        model.critic.register_forward_pre_hook(lambda module, args: critic_teams.append(
            args[0].detach().clone())),
        model.register_forward_hook(lambda module, args, output: logits.append(
            output["logits"].detach().clone())),
    ]
    original_step = adapter.step

    def record_step(actions, kappa):
        physical_steps.append(kappa.copy())
        return original_step(actions, kappa)

    def stay(logits, seed, step):
        # Agents start on distinct cells; staying guarantees exposure at t=1.
        actions = torch.full((len(logits),), 4, dtype=torch.long)
        return actions, torch.distributions.Categorical(logits=logits).log_prob(actions).sum()

    monkeypatch.setattr(adapter, "step", record_step)
    monkeypatch.setattr("softrole.rollout.sample_actions", stay)
    try:
        episode = run_episode(model, adapter, config, scenario, training=False)
    finally:
        for handle in handles:
            handle.remove()
    assert episode.metrics["steps"] == 3 and episode.metrics["event_exposed"]
    assert episode.metrics["sensing"] == [1, 3, 2]
    initial = np.array([[1 / 3, 0.], [1., 0.], [2 / 3, 1.]])
    failed = initial.copy()
    failed[2, 0] = 0.
    type_descriptor = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
    for t, (observation, actual_descriptor) in enumerate(inputs):
        physical = initial if t == 0 else failed
        np.testing.assert_array_equal(physical_steps[t], physical)
        expected = torch.as_tensor(physical) if descriptor == "capability" else type_descriptor
        assert torch.equal(actual_descriptor, expected)
        assert torch.equal(observation[:, -2:], expected)
        assert torch.equal(critic_agents[t][:, -2:], expected)
        # Decomposed critic tail is [N, sum(descriptor), remaining] for each agent.
        torch.testing.assert_close(critic_teams[t][:, -3:-1], expected.sum(0).expand(3, -1), rtol=0, atol=0)
        probabilities = logits[t].softmax(-1)
        assert torch.equal(probabilities[:2, 5], torch.zeros(2, dtype=torch.float64))
        assert probabilities[2, 5] > 0
        victim_sensory = observation[2, 25 * 25:-2]
        if t == 0:
            assert victim_sensory.count_nonzero() > 0  # Includes own occupied cell.
        else:
            assert victim_sensory.count_nonzero() == 0


def test_collector_uses_configured_non_native_levels_and_keeps_finite_gradients():
    config = graded_config(sensing_p=(1,), sensing_a=(2,), held_out_sensing_p=(),
                           held_out_sensing_a=(), max_steps=2, batch_steps=3)
    torch.manual_seed(config.seed)
    model = SoftRoleNet(**config.model_kwargs()).double()
    batch = collect_batch(model, config, collector=0, update=0)
    assert batch["num_episodes"] >= 2
    assert all(episode["sensing"] == [1, 1, 2] for episode in batch["episodes"])
    assert all(np.isfinite(gradient).all() for gradient in batch["gradients"])
    assert any(np.any(gradient != 0) for gradient in batch["gradients"])


@pytest.mark.parametrize("capabilities,changes,message", [
    ("binary", {}, "require graded capabilities"),
    ("graded", {"sensing": (1, 3)}, "one integer level"),
    ("graded", {"sensing": (True, 3, 2)}, "one integer level"),
    ("graded", {"sensing": (1., 3, 2)}, "one integer level"),
    ("graded", {"sensing": (-1, 3, 2)}, "one integer level"),
    ("graded", {"sensing": (1, 4, 2)}, "one integer level"),
    ("graded", {"sensing": (1, 3, 0)}, "victim must identify a sensing agent"),
    ("graded", {"victim": 3}, "victim must identify a sensing agent"),
    ("graded", {"sensing": (0, 0, 2)}, "at least two sensing agents"),
])
def test_invalid_graded_scenarios_fail_before_model_or_environment_use(capabilities, changes, message):
    config = graded_config(capabilities=capabilities, sensing_p=(), sensing_a=(),
                           held_out_sensing_p=(), held_out_sensing_a=())
    scenario = GradedScenario("invalid", 11, 12, 13, 2, 1,
                              event_step=1, victim=2, sensing=(1, 3, 2))
    # These sentinels have no model/adapter interface: validation must run first.
    with pytest.raises(ValueError, match=message):
        run_episode(object(), object(), config, replace(scenario, **changes))


@pytest.mark.parametrize("descriptor", ["capability", "type"])
def test_spawned_graded_collectors_match_serial_episode_records_and_gradients(descriptor):
    config = graded_config(descriptor=descriptor, nprocesses=2, max_steps=2, batch_steps=2)
    torch.manual_seed(config.seed)
    model = SoftRoleNet(**config.model_kwargs()).double()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    serial = [worker_collect(state, config.model_kwargs(), config.to_dict(), rank, 7)
              for rank in range(2)]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        pending = [pool.submit(worker_collect, state, config.model_kwargs(), config.to_dict(), rank, 7)
                   for rank in range(2)]
        spawned = [future.result(timeout=60) for future in pending]
    for expected, actual in zip(serial, spawned):
        assert expected.keys() == actual.keys()
        for key in expected.keys() - {"gradients"}:
            assert actual[key] == expected[key]
        assert all("sensing" in episode for episode in actual["episodes"])
        for expected_gradient, actual_gradient in zip(expected["gradients"], actual["gradients"]):
            np.testing.assert_array_equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize("descriptor", ["capability", "type"])
def test_graded_shared_training_and_partial_epoch_resume_are_exact(tmp_path, descriptor):
    config = graded_config(descriptor=descriptor)
    full_path = train(config, tmp_path / "full")
    partial_path = train(replace(config, total_steps=1), tmp_path / "partial")
    partial = torch.load(partial_path, map_location="cpu", weights_only=False)
    assert partial["updates"] == 1 and partial["completed_epochs"] == 0
    assert partial["updates_in_partial_epoch"] == 1
    resumed_path = train(config, tmp_path / "resumed", resume=partial_path)
    full = torch.load(full_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    for field in ("model_state", "optimizer_state", "updates", "total_steps", "total_episodes",
                  "completed_epochs", "updates_in_partial_epoch", "signature", "model_config", "config"):
        assert_state_equal(full[field], resumed[field])
    assert full["updates"] == 3
    assert Config(**full["config"]) == config
    assert full["model_config"] == config.model_kwargs()
    assert any(not torch.equal(partial["model_state"][name], value)
               for name, value in full["model_state"].items())
    records = [json.loads(line) for line in (tmp_path / "full" / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == full["total_episodes"]
    assert sum(row["steps"] for row in records) == full["total_steps"]
    assert all(len(row["sensing"]) == 3 and set(row["sensing"][:2]) <= {1, 3}
               and row["sensing"][2] == 0 for row in records)
    updates = [json.loads(line) for line in (tmp_path / "full" / "updates.jsonl").read_text().splitlines()]
    assert len(updates) == 3
    assert all(np.isfinite(row["gradient_norm_before_clip"]) and row["gradient_norm_before_clip"] > 0
               for row in updates)


@pytest.mark.parametrize("descriptor", ["capability", "type"])
def test_frozen_graded_checkpoint_replays_sham_and_capture_sensor_diagnostics(tmp_path, descriptor):
    config = graded_config(descriptor=descriptor, updates_per_epoch=1)
    checkpoint = train(config, tmp_path / "run")
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    frozen = SoftRoleNet(**saved["model_config"]).double()
    frozen.load_state_dict(saved["model_state"], strict=True)
    expected_signature = model_signature(frozen)
    scenario = GradedScenario("capture-sensor", 11, 12, 13, 2, 1,
                              event_step=1, victim=2, sensing=(1, 3, 2))
    failure = evaluate_checkpoint(checkpoint, [scenario], tmp_path / "failure.json", trace=True)
    sham = evaluate_checkpoint(checkpoint, [scenario], tmp_path / "sham.json", trace=True, sham=True)
    repeated = evaluate_checkpoint(checkpoint, [scenario], tmp_path / "sham-replay.json", trace=True, sham=True)
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert sham["per_episode"] == repeated["per_episode"]
    for report in (failure, sham, repeated):
        assert report["checkpoint_sha256"] == checkpoint_hash
        assert report["model_signature"] == expected_signature
        assert report["scenarios"] == [asdict(scenario)]
        assert report["scheduled_event_exposed_episodes"] == 1
        assert report["per_episode"][0]["sensing"] == [1, 3, 2]
    assert failure["event_exposed_episodes"] == 1 and sham["event_exposed_episodes"] == 0
    actual, control = failure["per_episode"][0], sham["per_episode"][0]
    assert actual["trace"][:1] == control["trace"][:1]

    # Check diagnostics against physical coordinates, independently of the
    # adapted channels and pcp_sensor_status used by frozen evaluation.
    adapter = make_env(config)
    _, kappa = adapter.reset(scenario.env_seed, sensing=scenario.sensing)

    def victim_sees_target():
        displacement = np.abs(adapter.raw.prey_loc - adapter.positions[scenario.victim])
        return bool(np.any(displacement.max(axis=1) <= scenario.sensing[scenario.victim] - 1))

    previously_seen = victim_sees_target()
    adapter.step(np.array(actual["trace"][0]["actions"]), kappa)
    expected = {"pre_event_victim_reached": bool(adapter.raw.reached_prey[scenario.victim]),
                "pre_event_victim_target_visible": victim_sees_target(),
                "pre_event_victim_target_seen": previously_seen}
    for name, value in expected.items():
        assert actual[name] is value and control[name] is value


def test_cli_loads_mixed_binary_and_graded_scenario_records(tmp_path):
    from softrole.__main__ import parser, scenarios_for

    binary = Scenario("binary", 11, 12, 13, 2, 1)
    graded = GradedScenario("graded", 21, 22, 23, 2, 1, sensing=(1, 3, 2))
    source = tmp_path / "scenarios.json"
    source.write_text(json.dumps([asdict(binary), asdict(graded)]))
    args = parser().parse_args(["evaluate", "--checkpoint", "unused.pt", "--output", "unused.json",
                               "--scenarios", str(source)])
    actual = scenarios_for(args, graded_config())
    assert actual == [binary, graded]
    assert type(actual[0]) is Scenario and type(actual[1]) is GradedScenario
    assert actual[1].sensing == (1, 3, 2)


@pytest.mark.parametrize("held_out", [False, True])
def test_cli_generated_graded_scenarios_use_declared_evaluation_support(held_out):
    from softrole.__main__ import parser, scenarios_for

    config = graded_config() if held_out else graded_config(held_out_sensing_p=(), held_out_sensing_a=())
    args = parser().parse_args(["evaluate", "--checkpoint", "unused.pt", "--output", "unused.json",
                               "--episodes", "10", "--seed", "123"])
    scenarios = scenarios_for(args, config)
    assert len(scenarios) == 10
    assert scenarios == scenarios_for(args, config)
    p_support, a_support = ({2}, {1, 2, 3}) if held_out else ({1, 3}, {0})
    for scenario in scenarios:
        assert type(scenario) is GradedScenario
        assert len(scenario.sensing) == 3
        assert set(scenario.sensing[:2]) <= p_support
        assert scenario.sensing[2] in a_support


def test_empty_graded_sensing_uses_native_behavior_and_preserves_record_identity():
    config = graded_config()
    torch.manual_seed(17)
    model = SoftRoleNet(**config.model_kwargs()).double()
    scenario = GradedScenario("native-default", 11, 12, 13, 2, 1)
    implicit = run_episode(model, make_env(config), config, scenario, training=False, trace=True)
    explicit = run_episode(model, make_env(config), config, replace(scenario, sensing=(3, 3, 0)),
                           training=False, trace=True)
    assert implicit.metrics["sensing"] == []
    assert explicit.metrics["sensing"] == [3, 3, 0]
    assert implicit.traces == explicit.traces
    assert {key: value for key, value in implicit.metrics.items() if key != "sensing"} == {
        key: value for key, value in explicit.metrics.items() if key != "sensing"}


def test_cli_training_dry_run_preserves_graded_fields_without_creating_run(tmp_path, capsys):
    from softrole.__main__ import main

    destination = tmp_path / "dry"
    main(["train", "--task", "pcp", "--model", "shared", "--output", str(destination),
          "--capabilities", "graded", "--descriptor", "type", "--sensing-p", "1", "3",
          "--sensing-a", "0", "--held-out-sensing-p", "2", "--held-out-sensing-a", "1", "2", "3",
          "--dry-run"])
    record = json.loads(capsys.readouterr().out)
    assert record["capabilities"] == "graded" and record["descriptor"] == "type"
    assert record["sensing_p"] == [1, 3] and record["sensing_a"] == [0]
    assert record["held_out_sensing_p"] == [2] and record["held_out_sensing_a"] == [1, 2, 3]
    assert not destination.exists()
