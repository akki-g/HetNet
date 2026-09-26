from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from hetnet_ext.recording import TrainingRecorder, write_checkpoint_signature
from hetnet_ext.signatures import model_signature


def test_recorder_counts_each_fresh_batch_once_and_copies_rewards(tmp_path):
    model = torch.nn.Linear(2, 1).double()
    recorder = TrainingRecorder(Namespace(metrics_file=str(tmp_path / "metrics.jsonl"), seed=7), model)
    first = {"num_steps": 40, "num_episodes": 2, "success": 1, "steps_taken": 40,
             "reward": np.array([2., 4., 6.]), "action_loss": 8., "value_loss": 12.}
    second = {"num_steps": 50, "num_episodes": 5, "success": 2, "steps_taken": 50,
              "reward": np.array([3., 6., 9.]), "action_loss": 10., "value_loss": 15.}
    recorder.add_batch(first)
    first["reward"][:] = -999  # Later upstream normalization cannot change recorded sums.
    recorder.add_batch(second)
    metrics = recorder.finish_epoch(1, 2.0, model)
    assert metrics["steps"] == metrics["total_steps"] == 90
    assert metrics["episodes"] == metrics["total_episodes"] == 7
    assert metrics["success_rate"] == 3 / 7
    assert metrics["steps_taken"] == 90 / 7
    assert metrics["reward_per_agent"] == [5 / 7, 10 / 7, 15 / 7]
    assert metrics["policy_loss"] == 18 / 90
    assert metrics["value_loss"] == 27 / 90
    recorder.add_batch(second)
    metrics = recorder.finish_epoch(2, 3.0, model)
    assert metrics["steps"] == 50 and metrics["total_steps"] == 140
    assert metrics["episodes"] == 5 and metrics["total_episodes"] == 12
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
    epochs = [json.loads(line) for line in (tmp_path / "epoch_signatures.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in epochs] == [1, 2]
    assert epochs[-1]["signature"] == model_signature(model)
    with pytest.raises(FileExistsError):
        TrainingRecorder(Namespace(metrics_file=str(tmp_path / "metrics.jsonl")), model)


def test_signature_detects_parameter_and_buffer_mutations_without_rng_or_state_changes(tmp_path):
    model = torch.nn.Linear(2, 1).double()
    model.register_buffer("diagnostic_buffer", torch.tensor(0., dtype=torch.float64))
    rng = torch.random.get_rng_state().clone()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    signature = model_signature(model)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert signature == model_signature(model)
    write_checkpoint_signature(tmp_path / "model_ep1.pt", model)
    assert json.loads((tmp_path / "model_ep1.pt.signature.json").read_text()) == signature
    with torch.no_grad():
        model.weight.add_(1)
    after_weight = model_signature(model)
    assert signature["sha256"] != after_weight["sha256"]
    model.diagnostic_buffer.add_(1)
    assert after_weight["sha256"] != model_signature(model)["sha256"]
