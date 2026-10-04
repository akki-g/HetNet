"""Independent small-number references for the versioned paper learner."""
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_learning_under_test",
    ROOT / "publication_reconstruction/runtime/hetnet_ext/paper_learning.py")
learning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(learning)
DTYPE = torch.float64


def test_monte_carlo_critic_target_is_not_lambda_return():
    rewards = torch.tensor([[1.0], [2.0]], dtype=DTYPE, requires_grad=True)
    values = torch.tensor([0.3, 0.4], dtype=DTYPE, requires_grad=True)
    returns, advantages = learning.returns_and_advantages(rewards, values)
    torch.testing.assert_close(returns[:, 0], torch.tensor([3.0, 2.0], dtype=DTYPE), rtol=0, atol=0)
    torch.testing.assert_close(advantages[:, 0], torch.tensor([2.62, 1.6], dtype=DTYPE), rtol=0, atol=1e-15)
    assert advantages[0, 0] + values[0] != returns[0, 0]
    assert not returns.requires_grad and not advantages.requires_grad


def test_discounted_finite_horizon_gae_has_zero_terminal_bootstrap():
    rewards = torch.tensor([[1.0, 3.0], [2.0, -1.0]], dtype=DTYPE)
    values = torch.tensor([0.5, 0.25], dtype=DTYPE)
    returns, advantages = learning.returns_and_advantages(rewards, values, gamma=0.5, gae_lambda=0.25)
    torch.testing.assert_close(returns, torch.tensor([[2.0, 2.5], [2.0, -1.0]], dtype=DTYPE), rtol=0, atol=0)
    torch.testing.assert_close(advantages, torch.tensor([[0.84375, 2.46875], [1.75, -1.25]], dtype=DTYPE), rtol=0, atol=0)


def test_class_targets_total_team_weight_and_detached_actor_baseline():
    # One timestep makes the expected actor and critic derivatives elementary.
    rewards = torch.tensor([[1., 5., 9.]], dtype=DTYPE, requires_grad=True)
    p_log = torch.tensor([[-0.2, -0.4]], dtype=DTYPE, requires_grad=True)
    a_log = torch.tensor([[-0.7]], dtype=DTYPE, requires_grad=True)
    p_value = torch.tensor([2.], dtype=DTYPE, requires_grad=True)
    a_value = torch.tensor([4.], dtype=DTYPE, requires_grad=True)
    actor, critic = learning.episode_losses(rewards, [p_log, a_log], [p_value, a_value])
    assert actor.item() == pytest.approx(4.5 / 3)
    assert critic.item() == pytest.approx(2 / 3 + 25 / 3)
    gradients = torch.autograd.grad(actor, [p_log, a_log, p_value, a_value, rewards], allow_unused=True)
    torch.testing.assert_close(gradients[0], torch.tensor([[1 / 3, -1.]], dtype=DTYPE))
    torch.testing.assert_close(gradients[1], torch.tensor([[-5 / 3]], dtype=DTYPE))
    assert gradients[2:] == (None, None, None)
    value_gradients = torch.autograd.grad(critic, [p_value, a_value, rewards], allow_unused=True)
    torch.testing.assert_close(value_gradients[0], torch.tensor([-4 / 3], dtype=DTYPE))
    torch.testing.assert_close(value_gradients[1], torch.tensor([-10 / 3], dtype=DTYPE))
    assert value_gradients[2] is None


def test_absent_class_and_zero_padding_do_not_change_loss():
    rewards = torch.tensor([[1., 2., 3.]], dtype=DTYPE)
    logs = torch.tensor([[-0.2, -0.4, -0.6]], dtype=DTYPE)
    values = torch.tensor([0.5], dtype=DTYPE)
    expected = learning.episode_losses(rewards, [logs], [values])
    actual = learning.episode_losses(
        torch.cat([rewards, torch.zeros(5, 3, dtype=DTYPE)]),
        [torch.cat([logs, torch.zeros(5, 3, dtype=DTYPE)])],
        [torch.cat([values, torch.zeros(5, dtype=DTYPE)])])
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def _episode(model, index):
    length = (1, 3, 2, 4, 2)[index]
    observations = torch.arange(length * 3, dtype=DTYPE).reshape(length, 3) / 7 + index / 3
    features = model(observations)
    log_probs = features[:, :3].log_softmax(dim=1)
    rewards = torch.tensor([[1., -2., 3.]], dtype=DTYPE).repeat(length, 1) / (index + 1)
    return learning.episode_losses(rewards, [log_probs[:, :2], log_probs[:, 2:]],
                                   [features[:, 3], features[:, 4]])


def test_unequal_episode_partitions_average_once_and_clip_once(monkeypatch):
    torch.manual_seed(14)
    initial = torch.nn.Linear(3, 5).to(DTYPE).state_dict()
    clip = torch.nn.utils.clip_grad_norm_
    calls = []

    def spy(parameters, max_norm, **kwargs):
        parameters = list(parameters)
        calls.append([p.grad.clone() for p in parameters])
        return clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    results = []
    for partitions in ([[0, 1, 2, 3, 4]], [[0], [1, 2], [3], [4]]):
        learner = torch.nn.Linear(3, 5).to(DTYPE)
        learner.load_state_dict(initial)
        gradients = []
        for indices in partitions:
            worker = torch.nn.Linear(3, 5).to(DTYPE)
            worker.load_state_dict(initial)
            sum(sum(_episode(worker, index)) for index in indices).backward()
            gradients.append([p.grad.clone() for p in worker.parameters()])
        for parameter, pieces in zip(learner.parameters(), zip(*gradients)):
            parameter.grad = torch.stack(pieces).sum(dim=0)
        preclip = learning.normalize_and_clip_gradients(learner.parameters(), 5)
        assert preclip > 0.75
        torch.testing.assert_close(torch.cat([p.grad.flatten() for p in learner.parameters()]).norm(),
                                   torch.tensor(0.75, dtype=DTYPE), rtol=2e-6, atol=0)
        optimizer = torch.optim.Adam(learner.parameters(), lr=1e-3, foreach=False, fused=False)
        optimizer.step()
        results.append((learner.state_dict(), optimizer.state_dict()))
    assert len(calls) == 2
    for left, right in zip(calls[0], calls[1]):
        torch.testing.assert_close(left, right, rtol=1e-14, atol=1e-14)
    for name in results[0][0]:
        torch.testing.assert_close(results[0][0][name], results[1][0][name], rtol=0, atol=1e-15)
    for identifier in results[0][1]['state']:
        for name in results[0][1]['state'][identifier]:
            torch.testing.assert_close(results[0][1]['state'][identifier][name],
                                       results[1][1]['state'][identifier][name], rtol=1e-14, atol=1e-14)


def test_gradient_storage_and_unused_parameters_are_preserved():
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=DTYPE))
    unused = torch.nn.Parameter(torch.zeros(1, dtype=DTYPE))
    parameter.grad = torch.tensor([0.5], dtype=DTYPE)
    storage = parameter.grad.data_ptr()
    norm = learning.normalize_and_clip_gradients([parameter, unused], 2)
    assert norm.item() == 0.25
    assert parameter.grad.data_ptr() == storage
    assert unused.grad is None


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_reject_invalid_episode_denominators(count):
    with pytest.raises(ValueError, match='episode count'):
        learning.normalize_and_clip_gradients([], count)


def test_nonfinite_gradient_is_rejected_before_optimizer_step():
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=DTYPE))
    parameter.grad = torch.tensor([float('nan')], dtype=DTYPE)
    with pytest.raises(RuntimeError, match='non-finite'):
        learning.normalize_and_clip_gradients([parameter], 1)
