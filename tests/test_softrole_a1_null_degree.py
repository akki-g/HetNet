"""A1: log(max(d,1)/2) correction of the receiver's null attention logit.

With equal sender scores the corrected null share is independent of the
in-degree d (the null candidate is replicated d times). The constant -log 2 is
the proposal's corrected-null initialization folded into the offset, so the
nominal two-sender case is bitwise identical to the baseline at initialization
and any corrected model equals a baseline whose null parameter is shifted by
log(d/2) at degree d.
"""
from dataclasses import replace
import json
import math

import pytest
import torch

from softrole.config import Config, recipe
from softrole.evaluate import evaluate_checkpoint
from softrole.model import CommunicationRound, SoftRoleNet
from softrole.scenarios import make_scenarios
from softrole.train import train


def unit_message_round(degree_correction, nu):
    """Zero scores, every decoded message equal to one, null logit nu."""
    layer = CommunicationRound(1, heads=1, head_dim=1, msg_dim=1, experts=1, final=True,
                               degree_correction=degree_correction).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.encoder.bias.fill_(100.)
        layer.decoder.weight.fill_(1.)
        layer.null_logits.fill_(nu)
    return layer


def triangular_adjacency(n):
    # Receiver i hears exactly the i senders with smaller indices: degrees 0..n-1.
    index = torch.arange(n)
    return index.unsqueeze(0) < index.unsqueeze(1)


@pytest.mark.parametrize("nu", [0., .3, -1.2])
def test_equal_score_null_share_is_degree_invariant_only_with_the_correction(nu):
    n = 5
    adjacency = triangular_adjacency(n)
    degrees = adjacency.sum(1).double()
    h, gate, noise = torch.zeros(n, 1, dtype=torch.float64), torch.ones(n, 1, dtype=torch.float64), torch.zeros(n, 1, dtype=torch.float64)
    corrected_out, _, corrected_null = unit_message_round(True, nu)(h, gate, adjacency, noise)
    baseline_out, _, baseline_null = unit_message_round(False, nu)(h, gate, adjacency, noise)
    expected_corrected = torch.full((n,), math.exp(nu) / (math.exp(nu) + 2), dtype=torch.float64)
    expected_corrected[0] = 1.  # empty neighbourhood
    torch.testing.assert_close(corrected_null[:, 0], expected_corrected, rtol=0, atol=1e-14)
    torch.testing.assert_close(baseline_null[:, 0], math.exp(nu) / (math.exp(nu) + degrees), rtol=0, atol=1e-14)
    # The aggregate of unit messages is the total non-null share.
    torch.testing.assert_close(corrected_out[:, 0], 1 - corrected_null[:, 0], rtol=0, atol=1e-14)
    torch.testing.assert_close(baseline_out[:, 0], 1 - baseline_null[:, 0], rtol=0, atol=1e-14)
    assert corrected_out[0, 0] == 0 and corrected_null[0, 0] == 1
    assert torch.all(baseline_null[2:, 0] < baseline_null[1, 0])


def test_directed_degree_counts_only_permitted_senders():
    adjacency = torch.tensor([[False, True, False], [True, False, True], [False, False, False]])
    layer = unit_message_round(True, .3)
    args = (torch.zeros(3, 1, dtype=torch.float64), torch.ones(3, 1, dtype=torch.float64))
    out, _, null = layer(*args, adjacency, torch.zeros(3, 1, dtype=torch.float64))
    share = math.exp(.3) / (math.exp(.3) + 2)
    torch.testing.assert_close(null[:, 0], torch.tensor([share, share, 1.], dtype=torch.float64), rtol=0, atol=1e-14)
    assert out[2, 0] == 0  # exact zero aggregate at degree zero
    off, _, off_null = layer(*args, adjacency, torch.zeros(3, 1, dtype=torch.float64), comm_off=True)
    assert torch.all(off == 0) and torch.all(off_null == 1)


def paired_models(seed=19):
    torch.manual_seed(seed)
    baseline = SoftRoleNet(base=2, n_squares=3, mode="shared", pre_dim=8, hidden_dim=8, heads=2,
                           head_dim=3, msg_dim=5, comm_rounds=3, independent_heads=True,
                           communication="real").double()
    torch.manual_seed(seed)
    corrected = SoftRoleNet(base=2, n_squares=3, mode="shared", pre_dim=8, hidden_dim=8, heads=2,
                            head_dim=3, msg_dim=5, comm_rounds=3, independent_heads=True,
                            communication="real", null_degree_correction=True).double()
    return baseline, corrected


def model_inputs(model, n):
    generator = torch.Generator().manual_seed(11)
    obs = torch.randn(n, model.obs_dim, generator=generator, dtype=torch.float64)
    kappa = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., 0.]], dtype=torch.float64)[:n]
    obs[:, -2:] = kappa
    return obs, kappa, model.initial_memory(n), ~torch.eye(n, dtype=torch.bool)


def test_fresh_corrected_model_is_bitwise_identical_to_the_baseline_at_degree_two():
    baseline, corrected = paired_models()
    for (name, before), after in zip(baseline.named_parameters(), corrected.parameters()):
        assert torch.equal(before, after), name
    for n, equal in ((3, True), (2, False), (4, False)):
        obs, kappa, memory, adjacency = model_inputs(corrected, n)
        left = baseline(obs, kappa, memory, adjacency, .5)
        right = corrected(obs, kappa, corrected.initial_memory(n), adjacency, .5)
        same = all(torch.equal(a, b) for a, b in zip(left["alpha_null"], right["alpha_null"]))
        assert same == equal, n
        if equal:
            assert torch.equal(left["logits"], right["logits"]) and torch.equal(left["value"], right["value"])
            assert all(torch.equal(a, b) for a, b in zip(left["messages"], right["messages"]))


@pytest.mark.parametrize("n", [2, 4])
def test_corrected_model_equals_a_baseline_with_null_shifted_by_log_half_degree(n):
    baseline, corrected = paired_models(seed=5)
    with torch.no_grad():
        for parameter in corrected.parameters():
            parameter.add_(torch.randn_like(parameter) * .3)
        baseline.load_state_dict(corrected.state_dict())
        for layer in baseline.rounds:
            layer.null_logits.add_(math.log((n - 1) / 2))
    obs, kappa, memory, adjacency = model_inputs(corrected, n)
    left = baseline(obs, kappa, memory, adjacency, .5)
    right = corrected(obs, kappa, corrected.initial_memory(n), adjacency, .5)
    torch.testing.assert_close(left["logits"], right["logits"], rtol=0, atol=1e-12)
    for a, b in zip(left["alpha_null"], right["alpha_null"]):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in corrected.parameters())
    gradient = torch.autograd.grad(right["logits"][:, :5].sum(), [l.null_logits for l in corrected.rounds])
    assert all(torch.count_nonzero(g) > 0 for g in gradient)


def test_configuration_layout_and_cli(tmp_path, capsys):
    from softrole.__main__ import main
    assert Config(null_degree_correction=True).model_kwargs()["null_degree_correction"] is True
    assert "null_degree_correction" not in Config().model_kwargs()
    with pytest.raises(ValueError, match="boolean"):
        Config(null_degree_correction=1)
    with pytest.raises(ValueError, match="boolean"):
        SoftRoleNet(base=2, n_squares=3, pre_dim=4, hidden_dim=4, heads=1, head_dim=2, msg_dim=2,
                    null_degree_correction="yes")
    main(["train", "--output", str(tmp_path / "dry"), "--dry-run", "--task", "pcp",
          "--study", "composition", "--null-degree-correction"])
    assert json.loads(capsys.readouterr().out)["null_degree_correction"] is True
    main(["train", "--output", str(tmp_path / "dry"), "--dry-run", "--task", "pcp", "--study", "composition"])
    assert json.loads(capsys.readouterr().out)["null_degree_correction"] is False
    assert not (tmp_path / "dry").exists()


def test_composition_training_evaluation_and_resume_guard(tmp_path):
    config = recipe("pcp", "composition", model="shared", optimizer="adam", communication="real",
                    comm_rounds=3, independent_heads=True, null_degree_correction=True,
                    pre_dim=8, hidden_dim=8, heads=2, head_dim=4, msg_dim=4, vision=1, max_steps=3,
                    epochs=1, updates_per_epoch=2, batch_steps=1, nprocesses=1, save_every=1)
    checkpoint = train(config, tmp_path / "run")
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["learner"]["null_degree_correction"] is True
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["model_config"]["null_degree_correction"] is True
    assert saved["model_state"]["rounds.0.null_logits"].shape == (1, 2)
    held_out = make_scenarios(31, 2, ((3, 2),))
    report = evaluate_checkpoint(checkpoint, held_out, tmp_path / "held_out.json")
    assert report["episodes"] == 2 and all(0 <= e["alpha_null"] <= 1 for e in report["per_episode"])
    partial = train(replace(config, total_steps=1), tmp_path / "partial")
    with pytest.raises(ValueError, match="resume may change only"):
        train(replace(config, null_degree_correction=False), tmp_path / "switched", resume=partial)
    assert not (tmp_path / "switched").exists()
