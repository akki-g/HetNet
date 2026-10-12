"""Diagnostics must measure policy changes without changing the learned policy."""
import json
import math

import pytest
import torch

from softrole.config import Config
from softrole.lr_diagnostics import UpdateDiagnostics, policy_statistics, summarize_run
from softrole.model import SoftRoleNet
from softrole.train import collect_batch, train


def small_config(channel="real"):
    return Config(model="shared", optimizer="adam", lr=.001, communication=channel,
                  comm_rounds=3, independent_heads=True, msg_dim=4, heads=2, head_dim=4,
                  pre_dim=8, hidden_dim=8, max_steps=6, epochs=1, updates_per_epoch=2,
                  nprocesses=1, batch_steps=1, save_every=1)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_masked_kl_matches_two_action_reference():
    old = torch.tensor([[.25, .75, 0.]], dtype=torch.float64).log()
    new = torch.tensor([[.5, .5, 0.]], dtype=torch.float64).log()
    stats = policy_statistics(old, new)
    assert stats["kl"] == pytest.approx(.25 * math.log(.5) + .75 * math.log(1.5))
    assert stats["entropy_after"] == pytest.approx(math.log(2))
    assert policy_statistics(old, old)["kl"] == 0
    with pytest.raises(ValueError, match="feasible"):
        policy_statistics(old, torch.zeros_like(old))


@pytest.mark.parametrize("channel", ["real", "binary"])
def test_diagnostics_preserve_exact_training_and_optimizer(tmp_path, channel):
    config = small_config(channel)
    plain = train(config, tmp_path / "plain")
    measured = train(config, tmp_path / "measured", diagnostics_every=1)
    left = torch.load(plain, weights_only=False)
    right = torch.load(measured, weights_only=False)
    for key in ("model_state", "optimizer_state", "updates", "total_steps", "total_episodes"):
        assert_same(left[key], right[key])
    assert (tmp_path / "plain" / "episodes.jsonl").read_bytes() == (tmp_path / "measured" / "episodes.jsonl").read_bytes()
    assert not (tmp_path / "plain" / "diagnostics.jsonl").exists()
    rows = [json.loads(line) for line in (tmp_path / "measured" / "diagnostics.jsonl").open()]
    assert len(rows) == 2
    for row in rows:
        assert row["parameter_update_l2"] > 0
        assert row["parameter_update_l2"] ** 2 == pytest.approx(
            sum(v ** 2 for v in row["parameter_group_update_l2"].values()))
        for mode in ("recomputed", "teacher_forced"):
            assert row["policy_replay"][mode]["kl"] >= -1e-14
    summary = summarize_run(tmp_path / "measured")
    windows = summary["windows"]
    assert windows[0]["episodes"] == right["total_episodes"]
    assert windows[0]["steps_taken"] == right["total_steps"] / right["total_episodes"]
    assert all(row["episodes"] == 0 and row["steps_taken"] is None for row in windows[1:])
    assert not any(row["fully_reached"] for row in windows)


def test_recurrence_change_is_distinct_from_teacher_forced_output(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = small_config()
    model = SoftRoleNet(**config.model_kwargs()).double()
    batches = [collect_batch(model, config, 0, 0)]
    diagnostic = UpdateDiagnostics(config, tmp_path, every=5)
    rng = torch.get_rng_state().clone()
    gradients = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    diagnostic.before(model, batches, update=1)
    with torch.no_grad():
        model.lstm.bias_ih.add_(.4)
    diagnostic.after(model, 1, batches[0]["num_steps"], 1.)
    row = json.loads((tmp_path / "diagnostics.jsonl").read_text())
    assert row["policy_replay"]["recomputed"]["kl"] > 0
    assert row["policy_replay"]["teacher_forced"]["kl"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    for parameter, gradient in zip(model.parameters(), gradients):
        assert_same(parameter.grad, gradient)
    # Unsampled updates still measure displacement, with no extra replay.
    diagnostic.before(model, batches, update=2)
    diagnostic.after(model, 2, 2 * batches[0]["num_steps"], 1.)
    second = json.loads((tmp_path / "diagnostics.jsonl").read_text().splitlines()[1])
    assert "policy_replay" not in second and second["parameter_update_l2"] == 0


def test_policy_replay_covers_complete_batch_not_only_first_episode(tmp_path):
    torch.set_num_threads(1)
    config = small_config()
    config.batch_steps = 14
    model = SoftRoleNet(**config.model_kwargs()).double()
    batch = collect_batch(model, config, 0, 0)
    assert batch["num_episodes"] > 1
    diagnostic = UpdateDiagnostics(config, tmp_path, every=50)
    diagnostic.before(model, [batch], update=1)
    diagnostic.after(model, 1, batch["num_steps"], 0.)
    replay = json.loads((tmp_path / "diagnostics.jsonl").read_text())["policy_replay"]
    assert replay["episodes"] == batch["num_episodes"]
    assert replay["recomputed"]["agent_steps"] == 3 * batch["num_steps"]
    assert replay["recomputed"]["kl"] == 0


@pytest.mark.parametrize("every", [0, -1, 1.5, True])
def test_invalid_diagnostics_fail_before_creating_run(tmp_path, every):
    with pytest.raises(ValueError, match="positive integer"):
        train(small_config(), tmp_path / "invalid", diagnostics_every=every)
    assert not (tmp_path / "invalid").exists()


def test_dry_run_rejects_unsupported_diagnostics(tmp_path):
    from softrole.__main__ import main
    with pytest.raises(SystemExit) as error:
        main(["train", "--output", str(tmp_path / "invalid"), "--task", "pp",
              "--diagnostics-every", "1", "--dry-run"])
    assert error.value.code == 2
    assert not (tmp_path / "invalid").exists()
