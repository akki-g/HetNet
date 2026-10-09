"""Channel controls: frozen legacy behavior and independent binary/real payloads."""

import hashlib

import pytest
import torch

from softrole.model import CommunicationRound, SoftRoleNet


def small_model(**kwargs):
    return SoftRoleNet(base=2, n_squares=3, experts=3, pre_dim=8,
                       hidden_dim=8, heads=2, head_dim=3, msg_dim=5,
                       **kwargs).double()


def inputs(model):
    generator = torch.Generator().manual_seed(73)
    obs = torch.randn(3, model.obs_dim, generator=generator, dtype=torch.float64)
    kappa = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    obs[:, -2:] = kappa
    shape = (3, model.heads, model.msg_dim) if model.independent_heads else (3, model.msg_dim)
    noise = ([torch.randn(shape, generator=generator, dtype=torch.float64)
              for _ in range(model.comm_rounds)] if model.communication == "binary" else None)
    return obs, kappa, model.initial_memory(3), ~torch.eye(3, dtype=torch.bool), noise


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()


def test_default_retains_pre_extension_state_initialization_and_forward_snapshot():
    # Captured from the original two-round shared-payload model before this
    # extension.  Fix float32 initialization, then use the CPU float64 forward.
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(194)
            model = small_model()
            assert state_hash(model.state_dict()) == (
                "4f1252da8b2b921200edc585f3545231e4d2af7f08dbcca5f62df15d9a3e96f3")
            assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest() == (
                "909f0ccddb78ae68c215767e11b65e4ac4f85c3e713847cc9089674ccf2fb783")
            obs, kappa, memory, adjacency, noise = inputs(model)
            result = model(obs, kappa, memory, adjacency, 0.7, noise)
        expected = torch.tensor([
            [-0.05515769298362636, 0.26924067265513807, -0.002819023557041796,
             0.29392129348452106, 0.2418135160093554, -float("inf")],
            [-0.08173075636416927, 0.4009697359648734, -0.04399140770424496,
             0.3746780222462355, 0.23361216000170126, 0.2676854325808391],
            [-0.06845446209957944, 0.3296809498356963, -0.022054982786637426,
             0.32953635193541075, 0.23031460804591847, 0.29992398963960953],
        ], dtype=torch.float64)
        torch.testing.assert_close(result["logits"], expected, atol=0, rtol=0)
        assert result["value"].item() == 0.021982999691623206
        assert state_hash(result["memory"]) == (
            "0e7bbd9271866a045372c9d60e090cb71f65b8f4fd10ee4ae64193d0452b75d8")
        assert result["messages"] is result["bits"]
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("communication", ["binary", "real"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_round_shapes_gate_reuse_and_payload_budget(communication, rounds):
    model = SoftRoleNet(base=2, n_squares=3, comm_rounds=rounds, heads=4,
                        msg_dim=64, independent_heads=True, communication=communication).double()
    obs, kappa, memory, adjacency, noise = inputs(model)
    seen = []
    handles = [layer.register_forward_hook(
        lambda layer, args, output: seen.append((args[1], output[0].shape)))
        for layer in model.rounds]
    try:
        result = model(obs, kappa, memory, adjacency, 0.7, noise)
    finally:
        for handle in handles:
            handle.remove()
    assert result["logits"].shape == (3, 6)
    assert result["memory"]["u_bar"].shape == (3, 16)
    assert result["value"].shape == ()
    assert len(result["messages"]) == rounds
    assert all(payload.shape == (3, 4, 64) for payload in result["messages"])
    assert sum(payload.numel() for payload in result["messages"]) == 3 * rounds * 256
    assert all(gate is result["gate"] for gate, _ in seen)
    assert [shape for _, shape in seen] == [(3, 64)] * (rounds - 1) + [(3, 16)]
    if communication == "binary":
        assert all(torch.all((payload == 0) | (payload == 1)) for payload in result["bits"])
    else:
        assert result["bits"] == []


def test_binary_and_real_use_identical_parameters_and_initialization():
    torch.manual_seed(17)
    binary = small_model(comm_rounds=3, independent_heads=True)
    torch.manual_seed(17)
    real = small_model(comm_rounds=3, independent_heads=True, communication="real")
    assert state_hash(binary.state_dict()) == state_hash(real.state_dict())
    real.load_state_dict(binary.state_dict(), strict=True)


def test_independent_head_encoding_and_receiver_decoding_do_not_mix_heads():
    layer = CommunicationRound(1, heads=2, head_dim=1, msg_dim=1, experts=2,
                               final=False, independent_heads=True).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.encoder.bias.fill_(100.)
        layer.decoder.weight[:, :, 0, 0] = torch.tensor([[2., 10.], [6., 14.]])
    h = torch.zeros(2, 1, dtype=torch.float64)
    gates = torch.eye(2, dtype=torch.float64)
    adjacency = ~torch.eye(2, dtype=torch.bool)
    noise = torch.zeros(2, 2, 1, dtype=torch.float64)
    before, payload, null = layer(h, gates, adjacency, noise)
    # The other agent supplies a one in each head. Equal real/null scores
    # halve the receiver's expert-specific decoder output in each head.
    torch.testing.assert_close(before, torch.tensor([[1., 5.], [3., 7.]], dtype=torch.float64))
    assert torch.all(payload == 1) and torch.all(null == 0.5)
    with torch.no_grad():
        layer.encoder.bias[:, 0].fill_(-100.)
    after, changed_payload, _ = layer(h, gates, adjacency, noise)
    assert torch.count_nonzero(after[:, 0]) == 0
    assert torch.equal(after[:, 1], before[:, 1])
    assert torch.count_nonzero(changed_payload[:, 0]) == 0
    assert torch.equal(changed_payload[:, 1], payload[:, 1])


def test_binary_noise_is_independent_per_head_and_shared_between_receivers(monkeypatch):
    layer = CommunicationRound(1, heads=2, head_dim=1, msg_dim=1, experts=1,
                               final=False, independent_heads=True).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.decoder.weight.fill_(1.)
    drawn = []

    def uniform(logits):
        drawn.append(logits.shape)
        return logits.new_tensor([[[0.9], [0.1]], [[0.1], [0.9]], [[0.1], [0.1]]])

    monkeypatch.setattr(torch, "rand_like", uniform)
    adjacency = torch.zeros(3, 3, dtype=torch.bool)
    adjacency[1:, 0] = True
    output, payload, _ = layer(torch.zeros(3, 1, dtype=torch.float64),
                               torch.ones(3, 1, dtype=torch.float64), adjacency)
    assert drawn == [(3, 2, 1)]
    assert payload.tolist() == [[[1.], [0.]], [[0.], [1.]], [[0.], [0.]]]
    torch.testing.assert_close(output, torch.tensor([[0., 0.], [0.5, 0.], [0.5, 0.]],
                                                    dtype=torch.float64))


def test_real_sends_raw_values_has_exact_gradient_and_draws_no_noise(monkeypatch):
    layer = CommunicationRound(1, heads=2, head_dim=1, msg_dim=1, experts=1,
                               final=True, independent_heads=True, communication="real").double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.encoder.weight[0, :, 0, 0] = torch.tensor([2., 3.])
        layer.decoder.weight[0, :, 0, 0] = torch.tensor([4., 5.])

    def forbid_noise(*args, **kwargs):
        raise AssertionError("Real communication must not draw bit noise")

    monkeypatch.setattr(torch, "rand_like", forbid_noise)
    h = torch.tensor([[0.25], [0.5]], dtype=torch.float64, requires_grad=True)
    output, payload, _ = layer(h, torch.ones(2, 1, dtype=torch.float64),
                               ~torch.eye(2, dtype=torch.bool))
    assert payload.tolist() == [[[0.5], [0.75]], [[1.0], [1.5]]]
    torch.testing.assert_close(output[:, 0], torch.tensor([2.875, 1.4375], dtype=torch.float64))
    output.sum().backward()
    torch.testing.assert_close(h.grad, torch.full_like(h, 5.75), atol=0, rtol=0)


@pytest.mark.parametrize("communication", ["binary", "real"])
def test_independent_channel_is_permutation_equivariant(communication):
    model = small_model(comm_rounds=3, independent_heads=True, communication=communication)
    obs, kappa, memory, adjacency, noise = inputs(model)
    original = model(obs, kappa, memory, adjacency, 0.7, noise)
    permutation = torch.tensor([2, 0, 1])
    permuted = model(obs[permutation], kappa[permutation],
                     {key: value[permutation] for key, value in memory.items()},
                     adjacency[permutation][:, permutation], 0.7,
                     None if noise is None else [value[permutation] for value in noise])
    for key in ("logits", "gate"):
        torch.testing.assert_close(permuted[key], original[key][permutation])
    for key in ("messages", "alpha_null"):
        for before, after in zip(original[key], permuted[key]):
            torch.testing.assert_close(after, before[permutation])
    for key in memory:
        torch.testing.assert_close(permuted["memory"][key], original["memory"][key][permutation])
    torch.testing.assert_close(permuted["value"], original["value"])


@pytest.mark.parametrize("communication", ["binary", "real"])
def test_independent_channel_empty_neighbors_remove_decoder_bias(communication):
    layer = CommunicationRound(1, heads=2, head_dim=1, msg_dim=1, experts=1,
                               final=True, independent_heads=True, communication=communication).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.decoder.bias.fill_(7.)
    h, gates = torch.zeros(2, 1, dtype=torch.float64), torch.ones(2, 1, dtype=torch.float64)
    empty, _, null = layer(h, gates, torch.zeros(2, 2, dtype=torch.bool))
    off, _, off_null = layer(h, gates, ~torch.eye(2, dtype=torch.bool), comm_off=True)
    assert torch.count_nonzero(empty) == torch.count_nonzero(off) == 0
    assert torch.all(null == 1) and torch.all(off_null == 1)


@pytest.mark.parametrize("kwargs", [
    {"comm_rounds": 0}, {"comm_rounds": -1}, {"comm_rounds": True}, {"comm_rounds": 1.5},
    {"independent_heads": 1}, {"communication": "continuous"},
])
def test_invalid_channel_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        small_model(**kwargs)


def test_noise_must_match_the_binary_channel_shape_and_round_count():
    model = small_model(comm_rounds=3, independent_heads=True)
    obs, kappa, memory, adjacency, noise = inputs(model)
    with pytest.raises(ValueError, match="each communication round"):
        model(obs, kappa, memory, adjacency, 0.7, noise[:2])
    with pytest.raises(ValueError, match="same shape"):
        model(obs, kappa, memory, adjacency, 0.7, [value[:, 0] for value in noise])
    real = small_model(comm_rounds=3, independent_heads=True, communication="real")
    with pytest.raises(ValueError, match="does not accept bit noise"):
        real(obs, kappa, memory, adjacency, 0.7, noise)
