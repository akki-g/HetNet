"""Behavioral checks of local information, binary payloads, and convex banks."""

import pytest
import torch

from softrole.model import AffineBank, CommunicationRound, SoftRoleNet, straight_through_bits


def make_model(mode="banked", feedback=True):
    torch.manual_seed(19)
    return SoftRoleNet(base=2, n_squares=3, mode=mode, experts=3, pre_dim=8,
                       hidden_dim=8, heads=2, head_dim=3, msg_dim=5,
                       feedback=feedback).double()


def inputs(model, n=3):
    generator = torch.Generator().manual_seed(73)
    obs = torch.randn(n, model.obs_dim, generator=generator, dtype=torch.float64)
    kappa = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    kappa = kappa[torch.arange(n) % 3]
    obs[:, -2:] = kappa
    adjacency = ~torch.eye(n, dtype=torch.bool)
    noise = [torch.randn(n, model.msg_dim, generator=generator, dtype=torch.float64)
             for _ in range(2)]
    return obs, kappa, model.initial_memory(n), adjacency, noise


def forward(model, data, **kwargs):
    obs, kappa, memory, adjacency, noise = data
    return model(obs, kappa, memory, adjacency, remaining=0.7, noise=noise, **kwargs)


def test_affine_bank_matches_an_explicit_mixed_matrix_and_bias():
    bank = AffineBank(3, 2, 2).double()
    with torch.no_grad():
        bank.weight.copy_(torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]],
                                       [[2., -1.], [-3., 2.]]]))
        bank.bias.copy_(torch.tensor([[1., 2.], [3., 4.], [-1., 1.]]))
    x = torch.tensor([[2., -3.]], dtype=torch.float64)
    gate = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    expected_matrix = 0.2 * bank.weight[0] + 0.3 * bank.weight[1] + 0.5 * bank.weight[2]
    expected_bias = 0.2 * bank.bias[0] + 0.3 * bank.bias[1] + 0.5 * bank.bias[2]
    torch.testing.assert_close(bank(x, gate)[0], expected_matrix @ x[0] + expected_bias)


def test_hard_bit_forward_and_the_declared_surrogate_derivative():
    logits = torch.tensor([-1., 0., 2.], dtype=torch.float64, requires_grad=True)
    noise = torch.tensor([2., -0.5, -3.], dtype=torch.float64)
    bits = straight_through_bits(logits, noise)
    assert bits.tolist() == [1., 0., 0.]
    bits.sum().backward()
    relaxed = torch.sigmoid(logits.detach() + noise)
    torch.testing.assert_close(logits.grad, relaxed * (1 - relaxed))


def test_sampled_bits_have_bernoulli_marginals():
    torch.manual_seed(431)
    logits = torch.tensor([-1., 0., 1.], dtype=torch.float64).expand(40000, -1)
    bits = straight_through_bits(logits)
    torch.testing.assert_close(bits.mean(0), torch.sigmoid(logits[0]), atol=0.012, rtol=0)
    assert torch.all((bits == 0) | (bits == 1))


def test_receiver_not_sender_selects_the_decoder():
    layer = CommunicationRound(1, heads=1, head_dim=1, msg_dim=1, experts=2, final=True).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.encoder.bias.fill_(100.)
        layer.decoder.weight[:, 0, 0] = torch.tensor([2., 6.])
    gates = torch.eye(2, dtype=torch.float64)
    output, bits, null = layer(torch.zeros(2, 1, dtype=torch.float64), gates,
                               ~torch.eye(2, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.float64))
    # One real message and null have equal scores.  Receiver 0 decodes 2;
    # receiver 1 decodes 6, despite receiving the other agent's payload.
    torch.testing.assert_close(output[:, 0], torch.tensor([1., 3.], dtype=torch.float64))
    assert bits.tolist() == [[1.], [1.]]
    assert null.tolist() == [[0.5], [0.5]]


def test_empty_neighborhood_and_comm_off_remove_decoder_bias_exactly():
    layer = CommunicationRound(1, heads=1, head_dim=1, msg_dim=1, experts=1, final=True).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.encoder.bias.fill_(-100.)
        layer.decoder.bias.fill_(3.)
    h, gate, noise = (torch.zeros(2, 1, dtype=torch.float64),
                      torch.ones(2, 1, dtype=torch.float64), torch.zeros(2, 1, dtype=torch.float64))
    connected = ~torch.eye(2, dtype=torch.bool)
    listening, bits, _ = layer(h, gate, connected, noise)
    assert torch.all(bits == 0)
    assert torch.all(listening == 1.5)
    empty, _, empty_null = layer(h, gate, torch.zeros_like(connected), noise)
    off, _, off_null = layer(h, gate, connected, noise, comm_off=True)
    assert torch.count_nonzero(empty) == torch.count_nonzero(off) == 0
    assert torch.all(empty_null == 1) and torch.all(off_null == 1)


def test_directed_adjacency_is_receiver_by_sender():
    layer = CommunicationRound(1, heads=1, head_dim=1, msg_dim=1, experts=1, final=True).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.encoder.weight.fill_(1.)
        layer.decoder.weight.fill_(1.)
    h = torch.tensor([[0.], [100.], [-100.]], dtype=torch.float64)
    adjacency = torch.zeros(3, 3, dtype=torch.bool)
    adjacency[0, 1] = True
    output, _, null = layer(h, torch.ones(3, 1, dtype=torch.float64), adjacency,
                            torch.zeros(3, 1, dtype=torch.float64))
    torch.testing.assert_close(output[:, 0], torch.tensor([0.5, 0., 0.], dtype=torch.float64))
    assert null[:, 0].tolist() == [0.5, 1., 1.]


@pytest.mark.parametrize("n", [1, 3, 5])
def test_variable_roster_bit_budget_masks_and_shapes(n):
    model = make_model()
    data = inputs(model, n)
    output = forward(model, data)
    assert output["logits"].shape == (n, 6)
    assert output["value"].shape == ()
    assert output["gate"].shape == (n, 3)
    torch.testing.assert_close(output["gate"].sum(-1), torch.ones(n, dtype=torch.float64))
    assert sum(bits.numel() for bits in output["bits"]) == 2 * n * model.msg_dim
    assert all(bits.shape == (n, model.msg_dim) for bits in output["bits"])
    assert all(torch.all((bits == 0) | (bits == 1)) for bits in output["bits"])
    probabilities = output["logits"].softmax(-1)
    assert torch.all(probabilities[data[1][:, 1] == 0, 5] == 0)
    assert torch.isfinite(output["logits"][:, :5]).all()
    assert torch.isfinite(output["value"])
    if n == 1:
        assert all(torch.all(null == 1) for null in output["alpha_null"])


def test_permutation_equivariance_with_coupled_noise_and_memory():
    model = make_model()
    obs, kappa, memory, adjacency, noise = inputs(model)
    first = model(obs, kappa, memory, adjacency, 0.7, noise)
    memory = first["memory"]
    memory["a_prev"] = torch.tensor([0, 4, 5])
    permutation = torch.tensor([2, 0, 1])
    original = model(obs, kappa, memory, adjacency, 0.6, noise)
    permuted = model(obs[permutation], kappa[permutation],
                     {name: value[permutation] for name, value in memory.items()},
                     adjacency[permutation][:, permutation], 0.6,
                     [value[permutation] for value in noise])
    for name in ("logits", "gate"):
        torch.testing.assert_close(permuted[name], original[name][permutation])
    torch.testing.assert_close(permuted["value"], original["value"])
    for name in memory:
        torch.testing.assert_close(permuted["memory"][name], original["memory"][name][permutation])
    for name in ("bits", "alpha_null"):
        for before, after in zip(original[name], permuted[name]):
            torch.testing.assert_close(after, before[permutation])


def test_critic_parameters_and_remaining_time_cannot_change_actor():
    model = make_model()
    data = inputs(model)
    original = forward(model, data)
    with torch.no_grad():
        for parameter in list(model.critic_agent.parameters()) + list(model.critic.parameters()):
            parameter.add_(3.)
    obs, kappa, memory, adjacency, noise = data
    changed = model(obs, kappa, memory, adjacency, remaining=0.1, noise=noise)
    assert torch.equal(original["logits"], changed["logits"])
    assert torch.equal(original["gate"], changed["gate"])
    assert not torch.equal(original["value"], changed["value"])
    actor_gradient = torch.autograd.grad(changed["logits"][:, 0].sum(),
                                        tuple(model.critic.parameters()), allow_unused=True)
    assert all(gradient is None for gradient in actor_gradient)


def test_detach_cuts_all_memory_and_feedback_carries_received_information():
    model = make_model()
    data = inputs(model)
    communicating = forward(model, data)
    silent = forward(model, data, comm_off=True)
    detached = model.detach_memory(communicating["memory"])
    assert all(value.grad_fn is None for value in detached.values())
    assert all(not value.requires_grad for value in detached.values())
    obs, kappa, _, adjacency, noise = data
    with_messages = model(obs, kappa, communicating["memory"], adjacency, 0.6, noise)
    without_messages = model(obs, kappa, silent["memory"], adjacency, 0.6, noise)
    assert not torch.equal(with_messages["memory"]["H"], without_messages["memory"]["H"])


def test_no_feedback_ablation_ignores_previous_communicated_features():
    model = make_model(feedback=False)
    data = inputs(model)
    original = forward(model, data)
    data[2]["u_bar"].fill_(100.)
    changed = forward(model, data)
    assert torch.equal(original["logits"], changed["logits"])
    assert torch.equal(original["memory"]["H"], changed["memory"]["H"])


@pytest.mark.parametrize("mode", ["shared", "capability", "constant"])
def test_comparison_modes_have_the_declared_gate_dependencies(mode):
    model = make_model(mode)
    data = inputs(model)
    original = forward(model, data)
    data[0][:, :-2].add_(7.)
    changed = forward(model, data)
    assert torch.equal(original["gate"], changed["gate"])
    if mode == "shared":
        assert model.experts == 1
        assert model.gate_network is None and model.constant_logits is None
        assert all(layer.self_map.weight.shape[0] == 1 for layer in model.rounds)
        assert torch.all(original["gate"] == 1)
    if mode == "constant":
        assert torch.equal(original["gate"][0], original["gate"][1])


def test_partial_gate_freeze_preserves_other_agents_current_gates():
    model = make_model()
    data = inputs(model)
    original = forward(model, data)
    pinned = torch.eye(3, dtype=torch.float64)
    frozen = forward(model, data, gate_override=(torch.tensor([False, True, False]), pinned))
    assert torch.equal(frozen["gate"][1], pinned[1])
    assert torch.equal(frozen["gate"][[0, 2]], original["gate"][[0, 2]])


def test_dtype_aware_initial_memory_and_no_global_default_dtype_changes():
    before = torch.get_default_dtype()
    model = make_model()
    memory = model.initial_memory(2)
    assert torch.get_default_dtype() == before
    assert all(memory[name].dtype == torch.float64 for name in ("H", "C", "u_bar"))
    assert memory["a_prev"].dtype == torch.long and memory["a_prev"].tolist() == [4, 4]


def test_actor_and_value_losses_backpropagate_to_shared_recurrence():
    model = make_model()
    output = forward(model, inputs(model))
    actor_grads = torch.autograd.grad(output["logits"][:, 0].sum(), model.lstm.weight_ih,
                                     retain_graph=True)[0]
    value_grads = torch.autograd.grad(output["value"], model.lstm.weight_ih)[0]
    assert torch.isfinite(actor_grads).all() and torch.count_nonzero(actor_grads) > 0
    assert torch.isfinite(value_grads).all() and torch.count_nonzero(value_grads) > 0
