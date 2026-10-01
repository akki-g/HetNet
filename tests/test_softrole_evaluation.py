"""Replayable domain rollouts, frozen checkpoints, and event censoring."""
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from softrole.config import recipe
from softrole.env import ENVIRONMENT_VERSION, make_env
from softrole.evaluate import evaluate_checkpoint, model_signature
from softrole.model import SoftRoleNet
from softrole.rollout import run_episode, sample_actions
from softrole.scenarios import Scenario, make_scenarios


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def initial_memory(self, n):
        return {"H": torch.zeros(n, 1, dtype=self.bias.dtype),
                "a_prev": torch.full((n,), 4, dtype=torch.long)}

    @staticmethod
    def detach_memory(memory):
        return {key: value.detach() for key, value in memory.items()}

    def forward(self, obs, kappa, memory, adjacency, remaining, noise,
                gate_override=None, comm_off=False):
        n = len(obs)
        state = memory["H"] + 1
        score = state + kappa[:, :1] * 2 + self.bias
        gate = torch.softmax(torch.cat([score, -score], -1), -1)
        if gate_override is not None:
            mask, pinned = gate_override
            gate = torch.where(mask[:, None], pinned, gate)
        logits = torch.full((n, 6), -torch.inf, dtype=self.bias.dtype)
        logits[:, 4] = self.bias
        return {"logits": logits, "value": self.bias,
                "memory": {"H": state, "a_prev": memory["a_prev"]},
                "gate": gate, "alpha_null": [torch.full((n, 1), 1. if comm_off else .5)],
                "bits": [(part > 0).double() for part in noise]}


class ToyEnvironment:
    def __init__(self, finish_at=4, success=False):
        self.finish_at, self.success = finish_at, success

    def reset(self, seed, num_p, num_a):
        self.t = 0
        self.kappa = np.array([[1., 0.]] * num_p + [[0., 1.]] * num_a)
        self.positions = np.zeros((num_p + num_a, 2))
        return self.observe(self.kappa), self.kappa.copy()

    def observe(self, kappa):
        return np.asarray(kappa).copy()

    def step(self, actions, kappa):
        self.t += 1
        done = self.t == self.finish_at
        return self.observe(kappa), -np.ones(len(actions)), done, {
            "success": done and self.success, "environment_version": "toy"}


def toy_config():
    return SimpleNamespace(max_steps=4, comm_range=-1., msg_dim=2, detach_gap=2)


def test_pre_event_success_is_not_recovery_and_failures_are_censored():
    scenario = Scenario("event", 11, 12, 13, 2, 1, event_step=2, victim=0)
    early = run_episode(ToyModel(), ToyEnvironment(1, True), toy_config(), scenario, training=False)
    assert early.metrics["pre_event_success"]
    assert not early.metrics["event_exposed"]
    assert early.metrics["recovery_steps"] is None
    assert not early.metrics["recovery_censored"]
    failed = run_episode(ToyModel(), ToyEnvironment(), toy_config(), scenario, training=False)
    assert failed.metrics["event_exposed"] and failed.metrics["recovery_censored"]
    assert failed.metrics["recovery_steps"] is None
    assert failed.metrics["post_event_steps"] == 2
    recovered = run_episode(ToyModel(), ToyEnvironment(4, True), toy_config(), scenario, training=False)
    assert recovered.metrics["recovery_steps"] == 2


@pytest.mark.parametrize("intervention", ["freeze_affected", "freeze_all", "comm_off"])
@pytest.mark.parametrize("sham", [False, True])
def test_interventions_share_prefix_and_start_after_pre_event_gate(intervention, sham):
    scenario = Scenario("event", 11, 12, 13, 2, 1, event_step=2, victim=0)
    ordinary = run_episode(ToyModel(), ToyEnvironment(), toy_config(), scenario,
                           training=False, trace=True, sham=sham)
    changed = run_episode(ToyModel(), ToyEnvironment(), toy_config(), scenario, training=False,
                          intervention=intervention, trace=True, sham=sham)
    assert ordinary.traces[:2] == changed.traces[:2]
    if intervention.startswith("freeze"):
        agents = [0] if intervention == "freeze_affected" else range(3)
        for step in changed.traces[2:]:
            for agent in agents:
                assert step["gate"][agent] == changed.traces[1]["gate"][agent]
        if intervention == "freeze_affected":
            assert changed.traces[2]["gate"][1] != changed.traces[1]["gate"][1]
    else:
        assert all(value == 1 for step in changed.traces[2:]
                   for round_values in step["alpha_null"] for agent in round_values for value in agent)


def test_sham_replays_no_failure_behavior_on_the_matched_event_schedule():
    scenario = Scenario("paired", 11, 12, 13, 2, 1, event_step=2, victim=0)
    failure = run_episode(ToyModel(), ToyEnvironment(), toy_config(), scenario,
                          training=False, trace=True)
    sham = run_episode(ToyModel(), ToyEnvironment(), toy_config(), scenario,
                       training=False, trace=True, sham=True)
    replay = run_episode(ToyModel(), ToyEnvironment(), toy_config(), scenario,
                         training=False, trace=True, sham=True)
    no_event = run_episode(ToyModel(), ToyEnvironment(), toy_config(),
                           replace(scenario, event_step=-1), training=False, trace=True)
    assert sham.metrics == replay.metrics and sham.traces == replay.traces
    assert sham.traces[:2] == failure.traces[:2]
    assert sham.traces == no_event.traces
    assert sham.traces[2]["gate"][0] != failure.traces[2]["gate"][0]
    assert sham.metrics["scheduled_event_exposed"] and not sham.metrics["event_exposed"]
    assert sham.metrics["scheduled_censored"] and not sham.metrics["recovery_censored"]
    assert sham.metrics["post_schedule_steps"] == 2 and sham.metrics["post_event_steps"] == 0
    assert sham.metrics["recovery_steps"] is None
    completed = run_episode(ToyModel(), ToyEnvironment(4, True), toy_config(), scenario,
                            training=False, sham=True)
    assert completed.metrics["scheduled_completion_steps"] == 2
    assert not completed.metrics["pre_event_success"] and completed.metrics["recovery_steps"] is None
    early = run_episode(ToyModel(), ToyEnvironment(1, True), toy_config(), scenario,
                        training=False, sham=True)
    assert early.metrics["pre_event_success"] and not early.metrics["scheduled_event_exposed"]
    assert early.metrics["scheduled_completion_steps"] is None
    with pytest.raises(ValueError, match="requires a scheduled"):
        run_episode(ToyModel(), ToyEnvironment(), toy_config(), replace(scenario, event_step=-1), sham=True)


def test_freeze_control_without_failure_and_missing_pre_event_gate():
    scenario = Scenario("control", 11, 12, 13, 2, 1, victim=0)
    episode = run_episode(ToyModel(), ToyEnvironment(), toy_config(), scenario, training=False,
                          intervention="freeze_affected", intervention_step=2, trace=True)
    assert not episode.metrics["event_exposed"] and episode.metrics["intervention_exposed"]
    assert episode.traces[3]["gate"][0] == episode.traces[1]["gate"][0]
    with pytest.raises(ValueError, match="pre-intervention"):
        run_episode(ToyModel(), ToyEnvironment(), toy_config(), replace(scenario, event_step=0),
                    training=False, intervention="freeze_affected")


def test_roundoff_cannot_sample_masked_final_action(monkeypatch):
    logits = torch.tensor([[-1.1028719396565423, -1.1338311858823507,
                            -0.5691238190467616, -0.8556987240546314,
                            -0.19700617595002753, -torch.inf]], dtype=torch.float64)
    near_one = torch.nextafter(logits.new_tensor(1.), logits.new_tensor(0.))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: near_one.reshape(1))
    action, log_prob = sample_actions(logits, 1, 0)
    assert action.item() == 4 and torch.isfinite(log_prob)


@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
def test_domain_replay_is_independent_of_global_rng_and_weights(task):
    config = recipe(task, max_steps=3, pre_dim=8, hidden_dim=8, heads=1, head_dim=4, msg_dim=3)
    model = SoftRoleNet(**config.model_kwargs()).double()
    model.eval()
    signature = model_signature(model)
    scenario = make_scenarios(51, 1, config.training_compositions)[0]
    first = run_episode(model, make_env(config), config, scenario, training=False, trace=True)
    np.random.rand(100)
    torch.rand(100)
    second = run_episode(model, make_env(config), config, scenario, training=False, trace=True)
    assert first.metrics == second.metrics and first.traces == second.traces
    assert model_signature(model) == signature
    assert all(not value.requires_grad for value in first.values)


def test_checkpoint_evaluation_replays_and_never_overwrites(tmp_path):
    config = recipe("pcp", max_steps=3, pre_dim=8, hidden_dim=8, heads=1, head_dim=4, msg_dim=3)
    model = SoftRoleNet(**config.model_kwargs()).double()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": config.to_dict(), "model_config": config.model_kwargs(),
                "model_state": model.state_dict(), "format_version": 1,
                "environment_version": ENVIRONMENT_VERSION}, checkpoint)
    scenarios = make_scenarios(44, 2, config.training_compositions)
    first = evaluate_checkpoint(checkpoint, scenarios, tmp_path / "first.json", trace=True)
    second = evaluate_checkpoint(checkpoint, scenarios, tmp_path / "second.json", trace=True)
    assert first == second
    assert first["evaluation_version"] == 2
    assert first["source_sha256"] is None  # Do not invent missing training identity.
    assert first["evaluator"]["source"]["files"]["softrole/evaluate.py"]
    assert first["evaluator"]["runtime"]["packages"]["torch"] == torch.__version__
    assert first["evaluator"]["runtime"]["torch_threads"] == 1
    assert first["model_signature"] == model_signature(model)
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(checkpoint, scenarios, tmp_path / "first.json")
    with pytest.raises(ValueError, match="unique"):
        evaluate_checkpoint(checkpoint, [scenarios[0], scenarios[0]], tmp_path / "invalid.json")
    out_of_horizon = replace(scenarios[0], event_step=3, victim=0)
    with pytest.raises(ValueError, match="horizon"):
        evaluate_checkpoint(checkpoint, [out_of_horizon], tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()
    failure_scenarios = make_scenarios(44, 2, config.training_compositions,
                                       failure_prob=1., failure_window=(1, 1))
    sham_report = evaluate_checkpoint(checkpoint, failure_scenarios, tmp_path / "sham.json", sham=True)
    assert sham_report["sham"] and sham_report["event_exposed_episodes"] == 0
    assert sham_report["scheduled_event_exposed_episodes"] == 2
    assert all(record["event_step"] == 1 for record in sham_report["scenarios"])
    assert all(row["pre_event_victim_target_seen"] is not None
               for row in sham_report["per_episode"])


def test_unsupported_event_rejected_before_environment_reset():
    config = recipe("pp", max_steps=3)
    scenario = Scenario("unsupported", 11, 12, 13, 3, 0, event_step=2, victim=0)
    with pytest.raises(ValueError, match="only for PCP"):
        run_episode(ToyModel(), None, config, scenario, training=False)


@pytest.mark.parametrize("changed", ["checkpoint", "evaluator"])
def test_evaluation_rejects_provenance_changes_during_rollout(tmp_path, monkeypatch, changed):
    config = recipe("pcp", max_steps=1, pre_dim=8, hidden_dim=8, heads=1, head_dim=4, msg_dim=3)
    model = SoftRoleNet(**config.model_kwargs()).double()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": config.to_dict(), "model_config": config.model_kwargs(),
                "model_state": model.state_dict(), "format_version": 1,
                "environment_version": ENVIRONMENT_VERSION}, checkpoint)
    from softrole.train import source_snapshot
    original = run_episode
    if changed == "evaluator":
        calls = 0

        def changed_source():
            nonlocal calls
            calls += 1
            record = source_snapshot()
            if calls > 1:
                record["sha256"] = "changed-evaluator"
            return record

        monkeypatch.setattr("softrole.train.source_snapshot", changed_source)

    def changed_rollout(*args, **kwargs):
        result = original(*args, **kwargs)
        if changed == "checkpoint":
            checkpoint.write_bytes(b"replaced after loading")
        return result

    monkeypatch.setattr("softrole.evaluate.run_episode", changed_rollout)
    output = tmp_path / "invalid.json"
    with pytest.raises(RuntimeError, match="changed during"):
        evaluate_checkpoint(checkpoint, make_scenarios(44, 1, [(2, 1)]), output)
    assert not output.exists()


DIAGNOSTIC_FIELDS = ("pre_event_victim_reached", "pre_event_victim_target_visible",
                     "pre_event_victim_target_seen")


def scripted_diagnostic_episode(monkeypatch, starts, actions, event_step, sham=False):
    """Use physical PCP transitions with fixed starts and prescribed actions."""
    config = recipe("pcp", max_steps=4, vision=1, msg_dim=2)
    adapter = make_env(config)
    monkeypatch.setattr(adapter.raw, "_get_cordinates", lambda: np.array(starts))
    monkeypatch.setattr("softrole.rollout.sample_actions", lambda logits, seed, step: (
        torch.tensor(actions[min(step, len(actions) - 1)]), logits[:, 4].sum() * 0))
    scenario = Scenario("diagnostic", 11, 12, 13, 2, 1, event_step=event_step,
                        victim=0 if event_step >= 0 else -1)
    episode = run_episode(ToyModel(), adapter, config, scenario, training=False,
                          trace=True, sham=sham, event_diagnostics=True)
    return episode, adapter


@pytest.mark.parametrize("victim_start,actions,event_step,expected", [
    # At step zero there is no earlier delivered observation, even if visible.
    ([0, 1], [[1, 4, 4]], 0, (False, True, False)),
    # The first visible observation is at the event itself, before reaching it.
    ([0, 0], [[1, 4, 4]], 1, (False, True, False)),
    # Earlier direct observation remains recorded after leaving the view.
    ([0, 1], [[3, 4, 4]], 1, (False, False, True)),
    # Reached status includes the action just before the scheduled event.
    ([0, 0], [[1, 4, 4]], 2, (True, True, True)),
])
def test_pre_event_diagnostics_distinguish_visibility_history_and_reaching(
        monkeypatch, victim_start, actions, event_step, expected):
    starts = [victim_start, [4, 4], [4, 3], [0, 2]]
    failure, failed_adapter = scripted_diagnostic_episode(
        monkeypatch, starts, actions, event_step)
    sham, _ = scripted_diagnostic_episode(monkeypatch, starts, actions, event_step, sham=True)
    assert tuple(failure.metrics[key] for key in DIAGNOSTIC_FIELDS) == expected
    assert tuple(sham.metrics[key] for key in DIAGNOSTIC_FIELDS) == expected
    assert failure.traces[:event_step] == sham.traces[:event_step]
    assert failure.metrics["scheduled_event_exposed"] and sham.metrics["scheduled_event_exposed"]
    assert failed_adapter.kappa[0, 0] == 0


@pytest.mark.parametrize("event_step", [-1, 2])
def test_unexposed_diagnostic_fields_are_null(monkeypatch, event_step):
    episode, _ = scripted_diagnostic_episode(
        monkeypatch, [[1, 2], [2, 1], [2, 3], [2, 2]],
        [[2, 1, 3], [4, 4, 5]], event_step)
    assert episode.metrics["success"] and episode.metrics["steps"] == 2
    assert not episode.metrics["scheduled_event_exposed"]
    assert all(episode.metrics[key] is None for key in DIAGNOSTIC_FIELDS)
    assert episode.metrics["pre_event_success"] == (event_step >= 0)


def test_sensor_status_reads_cached_view_without_observation_calls(monkeypatch):
    adapter = make_env(recipe("pcp", max_steps=4))
    with pytest.raises(RuntimeError, match="reset"):
        adapter.pcp_sensor_status(0)
    adapter.reset(9)
    original = adapter.raw_observation
    before = adapter.pcp_sensor_status(0)
    monkeypatch.setattr(adapter.raw, "_get_obs", lambda: pytest.fail("generated another view"))
    monkeypatch.setattr(adapter, "observe", lambda *_: pytest.fail("adapted another view"))
    assert adapter.pcp_sensor_status(0) == before
    np.testing.assert_array_equal(adapter.raw_observation, original)
    with pytest.raises(ValueError, match="sensing agent"):
        adapter.pcp_sensor_status(adapter.num_p)
    with pytest.raises(ValueError, match="only for PCP"):
        make_env(recipe("fc")).pcp_sensor_status(0)


@pytest.mark.parametrize("sham", [False, True])
def test_diagnostics_do_not_change_policy_trajectory_rng_or_observation_count(monkeypatch, sham):
    config = recipe("pcp", max_steps=5, pre_dim=8, hidden_dim=8, heads=1, head_dim=4, msg_dim=3)
    model = SoftRoleNet(**config.model_kwargs()).double()
    model.eval()
    signature = model_signature(model)
    scenario = Scenario("noninterference", 11, 12, 13, 2, 1, event_step=2, victim=0)
    episodes, adapters, call_counts = [], [], []
    np_before, py_before, torch_before = np.random.get_state(), random.getstate(), torch.get_rng_state()
    for enabled in (False, True):
        adapter = make_env(config)
        counts = {"native": 0, "adapted": 0}
        native, adapted = adapter.raw._get_obs, adapter.observe

        def counted_native():
            counts["native"] += 1
            return native()

        def counted_adapted(kappa):
            counts["adapted"] += 1
            return adapted(kappa)

        monkeypatch.setattr(adapter.raw, "_get_obs", counted_native)
        monkeypatch.setattr(adapter, "observe", counted_adapted)
        episode = run_episode(model, adapter, config, scenario, training=False,
                              trace=True, sham=sham, event_diagnostics=enabled)
        episodes.append(episode)
        adapters.append(adapter)
        call_counts.append(counts)
    plain, diagnostic = episodes
    assert plain.metrics == {key: value for key, value in diagnostic.metrics.items()
                             if key not in DIAGNOSTIC_FIELDS}
    assert plain.traces == diagnostic.traces
    for left, right in zip(plain.values + plain.log_probs, diagnostic.values + diagnostic.log_probs):
        assert torch.equal(left, right)
    assert call_counts[0] == call_counts[1]
    assert call_counts[0] == {"native": plain.metrics["steps"] + 1,
                              "adapted": plain.metrics["steps"] + 1 + (not sham)}
    for attribute in ("positions", "raw_observation", "kappa"):
        np.testing.assert_array_equal(getattr(adapters[0], attribute), getattr(adapters[1], attribute))
    assert model_signature(model) == signature
    assert np_before[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(np_before[1], np.random.get_state()[1])
    assert np_before[2:] == np.random.get_state()[2:]
    assert py_before == random.getstate()
    assert torch.equal(torch_before, torch.get_rng_state())
