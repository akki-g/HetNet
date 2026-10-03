"""Frozen original-weight transfer, singleton repair, and evaluator isolation."""
from argparse import Namespace
from dataclasses import replace

import numpy as np
import pytest
import torch

from hetnet_ext.signatures import model_signature
from softrole.env import make_env
from softrole.hetnet import (_new_model, evaluate_hetnet, hetnet_logits,
                            initial_memory, load_hetnet_model)
from softrole.scenarios import make_scenarios
from softrole.report import summarize_reports


def config():
    return Namespace(task="pcp", dim=5, vision=1, max_steps=2,
                     num_p=2, num_a=1, msg_dim=16, comm_range=-1)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("num_p,num_a", [(3, 2), (1, 2)])
def test_strict_original_weights_transfer_without_changes(binary, num_p, num_a):
    settings = config()
    source = _new_model(settings, 2, 1, binary)
    signature = model_signature(source)["sha256"]
    target = load_hetnet_model(source.state_dict(), settings, num_p, num_a, binary)
    assert model_signature(target)["sha256"] == signature
    adapter = make_env(settings)
    adapter.reset(5, num_p, num_a)
    memory = initial_memory(target)
    torch_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        logits, _ = hetnet_logits(target, adapter, settings, memory, 92, 0)
    assert logits.shape == (num_p + num_a, 6)
    assert torch.isneginf(logits[:num_p, 5]).all()
    assert torch.isfinite(logits[:, :5]).all()
    assert model_signature(target)["sha256"] == signature
    assert torch.equal(torch.random.get_rng_state(), torch_before)


@pytest.mark.parametrize("binary", [False, True])
def test_singleton_hook_leaves_original_team_outputs_exactly_unchanged(binary):
    settings = config()
    original = _new_model(settings, 2, 1, binary)
    adapted = load_hetnet_model(original.state_dict(), settings, 2, 1, binary)
    adapter = make_env(settings)
    adapter.reset(11)
    with torch.no_grad():
        expected, expected_memory = hetnet_logits(original, adapter, settings,
                                                  initial_memory(original), 77, 0)
        actual, actual_memory = hetnet_logits(adapted, adapter, settings,
                                              initial_memory(adapted), 77, 0)
    assert torch.equal(actual, expected)
    for key in expected_memory:
        for first, second in zip(expected_memory[key], actual_memory[key]):
            assert torch.equal(first, second)


@pytest.mark.parametrize("binary", [False, True])
def test_frozen_evaluator_replays_and_refuses_overwrite(tmp_path, binary):
    settings = config()
    source = _new_model(settings, 2, 1, binary)
    path = tmp_path / "original.pt"
    torch.save({"policy_net": source.state_dict()}, path)
    scenarios = make_scenarios(14, 2, [(3, 2), (1, 2)])
    first = evaluate_hetnet(path, settings, scenarios, tmp_path / "first.json", binary, trace=True)
    torch.rand(100)
    np.random.default_rng(100).random(20)
    second = evaluate_hetnet(path, settings, scenarios, tmp_path / "second.json", binary, trace=True)
    assert first == second
    assert first["parameters_unchanged"] is True
    assert "richer P/A-typed" in first["comparison"]
    assert first["environment_version"] == "corrected-observation-v1"
    assert first["episodes"] == 2
    assert len(first["checkpoint_sha256"]) == 64
    assert all(e["trace"] for e in first["per_episode"])
    assert first["mean_agent_return"] == pytest.approx(np.mean([
        np.mean(e["per_agent_returns"]) for e in first["per_episode"]]))
    assert all(e["mean_agent_return"] == pytest.approx(np.mean(e["per_agent_returns"]))
               for e in first["per_episode"])
    with pytest.raises(FileExistsError):
        evaluate_hetnet(path, settings, scenarios, tmp_path / "first.json", binary)


def test_legacy_evaluator_rejects_sensor_events_and_checkpoint_mismatch(tmp_path):
    settings = config()
    source = _new_model(settings, 2, 1, True)
    path = tmp_path / "original.pt"
    torch.save({"policy_net": source.state_dict()}, path)
    scenario = make_scenarios(1, 1, [(2, 1)])[0]
    with pytest.raises(ValueError, match="sensor-failure"):
        evaluate_hetnet(path, settings, [replace(scenario, event_step=1, victim=0)], tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
    wrong = config()
    wrong.vision = 2
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_hetnet_model(source.state_dict(), wrong, 3, 2)


def test_model_construction_preserves_default_dtype_and_rng():
    dtype = torch.get_default_dtype()
    state = torch.random.get_rng_state().clone()
    _new_model(config(), 2, 1, True)
    assert torch.get_default_dtype() == dtype
    assert torch.equal(torch.random.get_rng_state(), state)


def test_legacy_report_uses_checkpoint_training_seed_and_can_be_summarized(tmp_path):
    settings = config()
    source = _new_model(settings, 2, 1, True)
    checkpoint = tmp_path / "original.pt"
    torch.save({"policy_net": source.state_dict(), "seed": 41}, checkpoint)
    scenarios = make_scenarios(987, 2, [(3, 2)])
    output = tmp_path / "evaluation.json"
    report = evaluate_hetnet(checkpoint, settings, scenarios, output)
    assert report["config"]["seed"] == report["training_seed"] == 41
    summary = summarize_reports([output], bootstrap_samples=20)
    assert len(summary["groups"]) == 1
    assert summary["groups"][0]["training_seeds"] == 1
    assert summary["groups"][0]["config"]["model"] == "HetNet-Binary"
    assert summary["groups"][0]["metrics"]["success_rate"]["ci95"] is None


def test_legacy_report_does_not_invent_missing_training_seed(tmp_path):
    settings = config()
    source = _new_model(settings, 2, 1, False)
    checkpoint = tmp_path / "original.pt"
    torch.save({"policy_net": source.state_dict()}, checkpoint)
    output = tmp_path / "evaluation.json"
    report = evaluate_hetnet(checkpoint, settings, make_scenarios(987, 1, [(2, 1)]), output, False)
    assert report["training_seed"] is None and "seed" not in report["config"]
    with pytest.raises(ValueError, match="seed/model/task/max_steps"):
        summarize_reports([output], bootstrap_samples=20)


@pytest.mark.parametrize("task", ["pp", "fc"])
@pytest.mark.parametrize("binary", [False, True])
def test_frozen_baseline_supports_other_released_domain_recipes(tmp_path, task, binary):
    settings = config()
    settings.task = task
    settings.num_p, settings.num_a = (3, 0) if task == "pp" else (2, 1)
    settings.vision = 2 if task == "pp" else 1
    model = _new_model(settings, settings.num_p, settings.num_a, binary)
    checkpoint = tmp_path / "original.pt"
    torch.save({"policy_net": model.state_dict(), "seed": 0}, checkpoint)
    scenarios = make_scenarios(0, 1, [(settings.num_p, settings.num_a)])
    report = evaluate_hetnet(checkpoint, settings, scenarios, tmp_path / "evaluation.json", binary)
    assert report["episodes"] == 1 and report["parameters_unchanged"]
    assert report["config"]["task"] == task
    assert report["config"]["vision"] == settings.vision
