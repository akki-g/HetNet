"""Capability-conditioned baseline actors: architecture, contract and pipeline checks."""
import io
import json
from contextlib import redirect_stdout

import pytest
import torch

from hetnet_ext.signatures import model_signature as tensor_signature
from softrole.__main__ import main
from softrole.baselines import BASELINE_CLASSES, BASELINE_MODES
from softrole.config import Config
from softrole.evaluate import evaluate_checkpoint, model_signature
from softrole.factory import build_model
from softrole.model import SoftRoleNet
from softrole.scenarios import make_scenarios
from softrole.train import train

MODES = ("ca_mlp", "ca_cc_gnn", "ca_gnn", "rnn_exp", "cash")
GNN_MODES = ("ca_cc_gnn", "ca_gnn")
# Actor-only parameters at 5x5, vision 2 (obs_dim 702), H = hyper_dim = 64.
EXPECTED_ACTOR_PARAMETERS = {"ca_mlp": 57_862, "ca_cc_gnn": 61_830, "ca_gnn": 61_830,
                             "rnn_exp": 70_342, "cash": 202_566}
SMALL = dict(base=4, n_squares=9, hidden_dim=8)  # obs_dim 65
KAPPA = torch.tensor([[1., 0.], [1., 1.], [0., 1.], [1., 0.], [0., 0.]], dtype=torch.float64)


def small_model(mode, critic_mode="team", seed=0):
    torch.manual_seed(seed)
    settings = dict(mode=mode, critic_mode=critic_mode, **SMALL)
    if mode == "cash":
        settings["hyper_dim"] = 8
    return build_model(settings).double()


def adjacency_for(n, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(n, n, generator=generator) < 0.6) & ~torch.eye(n, dtype=torch.bool)


def random_memory(model, n, seed=2):
    generator = torch.Generator().manual_seed(seed)
    memory = model.initial_memory(n)
    for name, value in memory.items():
        if name == "a_prev":
            memory[name] = torch.randint(0, 6, (n,), generator=generator)
        else:
            memory[name] = torch.randn(value.shape, generator=generator, dtype=value.dtype)
    return memory


def random_obs(model, n, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, model.obs_dim, generator=generator, dtype=torch.float64)


def forward(model, obs, kappa, memory, adjacency, **options):
    return model(obs, kappa, memory, adjacency, remaining=0.5, **options)


# 1. Parameter counts -------------------------------------------------------

@pytest.mark.parametrize("critic_mode", ["team", "decomposed"])
@pytest.mark.parametrize("mode", MODES)
def test_actor_parameter_counts_match_the_specification(mode, critic_mode):
    config = Config(model=mode, communication="real", critic_mode=critic_mode)
    model = build_model(config.model_kwargs())
    assert model.obs_dim == 702
    total = sum(p.numel() for p in model.parameters())
    critic = sum(p.numel() for name, p in model.named_parameters() if name.startswith("critic_head."))
    assert total - critic == EXPECTED_ACTOR_PARAMETERS[mode]
    reference = SoftRoleNet(**Config(model="banked", critic_mode=critic_mode).model_kwargs())
    reference_critic = sum(p.numel() for name, p in reference.named_parameters()
                           if name.startswith("critic"))
    assert critic == reference_critic


def test_baseline_registry_is_consistent():
    assert set(BASELINE_MODES) == set(MODES) == set(BASELINE_CLASSES)


# 2. Contract: shapes, masks, critic ---------------------------------------

@pytest.mark.parametrize("critic_mode", ["team", "decomposed"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("mode", MODES)
def test_forward_contract(mode, n, critic_mode):
    model = small_model(mode, critic_mode)
    kappa, adjacency = KAPPA[:n], adjacency_for(n)
    memory = random_memory(model, n)
    out = forward(model, random_obs(model, n), kappa, memory, adjacency)
    assert out["logits"].shape == (n, 6)
    assert torch.equal(torch.isinf(out["logits"][:, 5]), kappa[:, 1] == 0)
    assert (out["logits"][:, 5][kappa[:, 1] == 0] < 0).all()
    finite = torch.ones_like(out["logits"], dtype=torch.bool)
    finite[:, 5] = kappa[:, 1] > 0
    assert torch.isfinite(out["logits"][finite]).all()
    assert out["value"].shape == () and torch.isfinite(out["value"])
    if critic_mode == "decomposed":
        assert out["agent_values"].shape == (n,) and torch.isfinite(out["agent_values"]).all()
        assert torch.equal(out["agent_values"].sum(), out["value"])
    else:
        assert out["agent_values"] is None
    assert torch.equal(out["memory"]["a_prev"], memory["a_prev"])
    assert set(out["memory"]) == set(model.initial_memory(n))
    assert torch.equal(out["gate"], torch.ones(n, 1, dtype=torch.float64))
    assert out["bits"] == []
    assert isinstance(out["messages"], list)
    assert len(out["alpha_null"]) == 1 and out["alpha_null"][0].shape == (n, 1)
    expected_null = 0. if mode in GNN_MODES else 1.
    assert torch.equal(out["alpha_null"][0], torch.full((n, 1), expected_null, dtype=torch.float64))
    initial = model.initial_memory(n)
    assert torch.equal(initial["a_prev"], torch.full((n,), 4))
    detached = model.detach_memory(out["memory"])
    assert all(not tensor.requires_grad for tensor in detached.values())


@pytest.mark.parametrize("mode", MODES)
def test_comm_off_marks_every_model_silent(mode):
    model = small_model(mode)
    out = forward(model, random_obs(model, 3), KAPPA[:3], model.initial_memory(3),
                  adjacency_for(3), comm_off=True)
    assert torch.equal(out["alpha_null"][0], torch.ones(3, 1, dtype=torch.float64))


@pytest.mark.parametrize("mode", MODES)
def test_noise_and_gate_overrides_and_bad_shapes_are_rejected(mode):
    model = small_model(mode)
    obs, adjacency, memory = random_obs(model, 2), adjacency_for(2), model.initial_memory(2)
    with pytest.raises(ValueError, match="noise"):
        forward(model, obs, KAPPA[:2], memory, adjacency, noise=[torch.zeros(2, 4)])
    with pytest.raises(ValueError, match="gate"):
        forward(model, obs, KAPPA[:2], memory, adjacency, gate_override=torch.ones(2, 1))
    with pytest.raises(ValueError, match="shapes"):
        forward(model, obs[:, :-1], KAPPA[:2], memory, adjacency)
    with pytest.raises(ValueError, match="Adjacency"):
        forward(model, obs, KAPPA[:2], memory, adjacency.to(torch.float64))
    with pytest.raises(ValueError, match="nonempty|least one"):
        model.initial_memory(0)


def test_payload_sizes_are_reported_by_the_models():
    sizes = {mode: small_model(mode).payload_values_per_agent_step for mode in MODES}
    assert sizes == {"ca_mlp": 0, "ca_cc_gnn": 8, "ca_gnn": 8, "rnn_exp": 0, "cash": 0}
    assert not hasattr(SoftRoleNet(mode="shared", communication="real"), "payload_values_per_agent_step")


# 3. Permutation equivariance ----------------------------------------------

@pytest.mark.parametrize("critic_mode", ["team", "decomposed"])
@pytest.mark.parametrize("mode", MODES)
def test_permuting_agents_permutes_logits_and_preserves_value(mode, critic_mode):
    n = 5
    model = small_model(mode, critic_mode)
    obs, kappa = random_obs(model, n), KAPPA[:n]
    memory, adjacency = random_memory(model, n), adjacency_for(n)
    perm = torch.tensor([3, 0, 4, 2, 1])
    base = forward(model, obs, kappa, memory, adjacency)
    permuted = forward(model, obs[perm], kappa[perm], {k: v[perm] for k, v in memory.items()},
                       adjacency[perm][:, perm])
    torch.testing.assert_close(permuted["logits"], base["logits"][perm], rtol=0, atol=1e-10)
    assert abs(float(permuted["value"] - base["value"])) <= 1e-10
    if critic_mode == "decomposed":
        torch.testing.assert_close(permuted["agent_values"], base["agent_values"][perm],
                                   rtol=0, atol=1e-10)
    for name in base["memory"]:
        if name != "a_prev":
            torch.testing.assert_close(permuted["memory"][name], base["memory"][name][perm],
                                       rtol=0, atol=1e-10)


# 4. GNN behaviour ----------------------------------------------------------

@pytest.mark.parametrize("mode", GNN_MODES)
def test_comm_off_equals_an_empty_adjacency(mode):
    model = small_model(mode)
    obs, memory = random_obs(model, 4), model.initial_memory(4)
    adjacency = adjacency_for(4)
    assert adjacency.any()
    silent = forward(model, obs, KAPPA[:4], memory, adjacency, comm_off=True)
    empty = forward(model, obs, KAPPA[:4], memory, torch.zeros(4, 4, dtype=torch.bool))
    assert torch.equal(silent["logits"], empty["logits"])
    assert torch.equal(silent["value"], empty["value"])
    connected = forward(model, obs, KAPPA[:4], memory, adjacency)
    assert not torch.equal(connected["logits"][:, :5], silent["logits"][:, :5])


def test_ca_gnn_encoder_ignores_the_tail_but_the_head_reads_it():
    model = small_model("ca_gnn")
    obs = random_obs(model, 3)
    changed = obs.clone()
    changed[1, -2:] += 1.0
    assert torch.equal(model._encode(obs), model._encode(changed))
    adjacency = adjacency_for(3)
    adjacency[0, 1] = adjacency[2, 1] = True
    base = forward(model, obs, KAPPA[:3], model.initial_memory(3), adjacency)
    after = forward(model, changed, KAPPA[:3], model.initial_memory(3), adjacency)
    assert not torch.equal(base["logits"][1, :5], after["logits"][1, :5])
    # A neighbour's tail never reaches the receiver in ca_gnn.
    assert torch.equal(base["logits"][0], after["logits"][0])
    assert torch.equal(base["logits"][2], after["logits"][2])


def test_ca_cc_gnn_neighbour_tail_changes_a_receivers_logits():
    model = small_model("ca_cc_gnn")
    obs = random_obs(model, 3)
    changed = obs.clone()
    changed[1, -2:] += 1.0
    adjacency = torch.zeros(3, 3, dtype=torch.bool)
    adjacency[0, 1] = True
    base = forward(model, obs, KAPPA[:3], model.initial_memory(3), adjacency)
    after = forward(model, changed, KAPPA[:3], model.initial_memory(3), adjacency)
    assert not torch.equal(base["logits"][0, :5], after["logits"][0, :5])
    # Agent 2 hears nobody and agent 1's own tail is not its receiver's concern.
    assert torch.equal(base["logits"][2], after["logits"][2])


# 5. CASH -------------------------------------------------------------------

def test_cash_logits_follow_the_tail_at_fixed_physical_observation():
    model = small_model("cash")
    obs = random_obs(model, 3)
    changed = obs.clone()
    changed[:, -2:] = 1.0 - changed[:, -2:]
    kappa, memory = torch.ones(3, 2, dtype=torch.float64), random_memory(model, 3)
    base = forward(model, obs, kappa, memory, adjacency_for(3))
    after = forward(model, changed, kappa, memory, adjacency_for(3))
    assert torch.equal(obs[:, :-2], changed[:, :-2])
    assert not torch.allclose(base["logits"], after["logits"])
    # The recurrent state sees the physical observation only.
    assert torch.equal(base["memory"]["H"], after["memory"]["H"])


def test_cash_initialisation_follows_the_released_code():
    model = small_model("cash")
    obs = random_obs(model, 4)
    assert torch.count_nonzero(model.hyper_b(obs)) == 0
    for layer in model.hyper_b.net:
        if isinstance(layer, torch.nn.Linear):
            assert torch.count_nonzero(layer.weight) == 0 and torch.count_nonzero(layer.bias) == 0
    linears = [layer for layer in model.hyper_w.net if isinstance(layer, torch.nn.Linear)]
    assert len(linears) == 4
    for layer in linears:
        assert torch.count_nonzero(layer.bias) == 0
        rows, cols = layer.weight.shape
        gram = (layer.weight @ layer.weight.T if rows <= cols else layer.weight.T @ layer.weight)
        # Orthogonalised in float32 at construction, then cast to float64.
        torch.testing.assert_close(gram, 0.04 * torch.eye(gram.shape[0], dtype=gram.dtype),
                                   rtol=0, atol=1e-6)
    assert sum(isinstance(m, torch.nn.LayerNorm) for m in model.hyper_w.net) == 3


# 6. Defaults are unchanged -------------------------------------------------

DEFAULT_CONFIG = {
    "task": "pcp", "env_version": "corrected-observation-v1", "num_p": 2, "num_a": 1, "dim": 5,
    "vision": 2, "max_steps": 80, "nfires": 1, "reward_type": 3, "model": "banked", "experts": 4,
    "pre_dim": 128, "hidden_dim": 64, "heads": 4, "head_dim": 16, "msg_dim": 16, "comm_rounds": 2,
    "independent_heads": False, "communication": "binary", "null_degree_correction": False,
    "critic_mode": "team", "credit_beta": 1.0, "objective": "team", "actor_capabilities": "explicit",
    "feedback": True, "comm_range": -1.0, "seed": 0, "epochs": 2000, "updates_per_epoch": 10,
    "batch_steps": 500, "nprocesses": 4, "total_steps": None, "lr": 0.0001, "optimizer": "rmsprop",
    "gamma": 1.0, "gae_lambda": 0.95, "actor_coeff": 50.0, "value_coeff": 1.0, "detach_gap": 5,
    "max_grad_norm": 0.75, "save_every": 50, "compositions": (), "held_out": (),
    "failure_prob": 0.0, "failure_window": (10, 30),
}
DEFAULT_MODEL_KWARGS = {"base": 25, "n_squares": 25, "mode": "banked", "experts": 4, "pre_dim": 128,
                        "hidden_dim": 64, "heads": 4, "head_dim": 16, "msg_dim": 16, "feedback": True}
DEFAULT_STATE_KEYS = [
    "pre.weight", "pre.bias", "lstm.weight_ih", "lstm.weight_hh", "lstm.bias_ih", "lstm.bias_hh",
    "gate_network.0.weight", "gate_network.0.bias", "gate_network.2.weight", "gate_network.2.bias",
    *[f"rounds.{r}.{name}" for r in (0, 1) for name in (
        "null_logits", "self_map.weight", "self_map.bias", "encoder.weight", "encoder.bias",
        "decoder.weight", "decoder.bias", "attention.weight", "attention.bias", "score.weight")],
    "output.weight", "output.bias", "critic_agent.0.weight", "critic_agent.0.bias",
    "critic.0.weight", "critic.0.bias", "critic.2.weight", "critic.2.bias"]


def test_default_configuration_and_model_layout_are_unchanged():
    assert Config().to_dict() == DEFAULT_CONFIG
    assert Config().model_kwargs() == DEFAULT_MODEL_KWARGS
    model = build_model(Config().model_kwargs())
    assert type(model) is SoftRoleNet
    assert list(model.state_dict()) == DEFAULT_STATE_KEYS
    assert "hyper_dim" not in Config(model="cash", communication="real").to_dict()
    assert Config(model="cash", communication="real", hyper_dim=32).to_dict()["hyper_dim"] == 32
    for name in ("shared", "capability", "constant"):
        assert type(build_model(Config(model=name).model_kwargs())) is SoftRoleNet


def test_baseline_model_kwargs_and_validation():
    for mode in MODES:
        config = Config(model=mode, communication="real", critic_mode="decomposed", feedback=False)
        expected = dict(mode=mode, base=25, n_squares=25, hidden_dim=64, critic_mode="decomposed")
        if mode == "cash":
            expected["hyper_dim"] = 64
        assert config.model_kwargs() == expected
        assert Config(**config.to_dict()).model_kwargs() == expected
        with pytest.raises(ValueError, match="communication"):
            Config(model=mode)
        with pytest.raises(ValueError, match="actor capabilities"):
            Config(model=mode, communication="real", actor_capabilities="physical_only")
    with pytest.raises(ValueError, match="hyper_dim"):
        Config(model="cash", communication="real", hyper_dim=0)


def test_command_line_exposes_baselines_hyper_dim_and_entropy(tmp_path):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main(["train", "--output", str(tmp_path / "unused"), "--dry-run", "--model", "cash",
              "--communication", "real", "--hyper-dim", "32", "--entropy-coeff", "0.003"])
    printed = json.loads(buffer.getvalue())
    assert printed["model"] == "cash" and printed["hyper_dim"] == 32 and printed["entropy_coeff"] == 0.003


# 7. End to end: train, checkpoint, frozen evaluation -----------------------

SENSING = dict(sensing_p=(1, 2), sensing_a=(0,), vision=1)


def tiny_config(mode):
    return Config(task="pcp", env_version="graded-observation-v2", capabilities="graded",
                  sensing_p=(1, 2), sensing_a=(0,), vision=1, compositions=((1, 1), (2, 1)),
                  max_steps=4, batch_steps=4, nprocesses=1, epochs=1, updates_per_epoch=2,
                  save_every=1, detach_gap=2, seed=11, hidden_dim=8, hyper_dim=8,
                  entropy_coeff=0.003, critic_mode="decomposed", communication="real", model=mode)


@pytest.fixture(scope="module", params=MODES)
def trained(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"baseline_{request.param}")
    config = tiny_config(request.param)
    checkpoint = train(config, root / "run")
    return request.param, config, checkpoint, root


def test_tiny_training_writes_a_valid_checkpoint(trained):
    mode, config, checkpoint, _ = trained
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["updates"] == 2
    assert saved["model_config"] == config.model_kwargs()
    assert saved["model_config"]["mode"] == mode
    assert all(t.dtype == torch.float64 and torch.isfinite(t).all() for t in saved["model_state"].values())
    model = build_model(saved["model_config"]).double()
    model.load_state_dict(saved["model_state"], strict=True)
    assert tensor_signature(model) == saved["signature"]
    assert saved["optimizer_state"]["state"]
    updates = [json.loads(line) for line in (checkpoint.parent.parent / "updates.jsonl").read_text().splitlines()]
    assert len(updates) == 2 and all(row["action_entropy_sum"] > 0 for row in updates)
    assert all(row["gradient_norm_before_clip"] > 0 for row in updates)


def test_frozen_evaluation_with_failure_and_comm_off(trained):
    mode, config, checkpoint, root = trained
    nominal = make_scenarios(5, 2, [(1, 1), (2, 1)], **SENSING)
    failing = make_scenarios(6, 2, [(2, 1)], failure_prob=1.0, failure_window=(1, 2), **SENSING)
    scenarios = nominal + failing
    assert all(s.event_step in (1, 2) for s in failing)
    plain = evaluate_checkpoint(checkpoint, scenarios, root / "evaluation_none.json")
    silent = evaluate_checkpoint(checkpoint, scenarios, root / "evaluation_comm_off.json",
                                 intervention="comm_off", intervention_step=1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(saved["model_config"]).double()
    model.load_state_dict(saved["model_state"])
    for report in (plain, silent):
        assert report["episodes"] == 4 and report["model_config"] == config.model_kwargs()
        assert report["model_signature"] == model_signature(model)
        assert tensor_signature(model) == saved["signature"]
        assert all(torch.isfinite(torch.tensor(e["team_return"])) for e in report["per_episode"])
        expected_payload = model.payload_values_per_agent_step
        for episode in report["per_episode"]:
            assert episode["communication"] == "real"
            assert episode["payload_values_per_agent_step"] == expected_payload
            assert episode["payload_bits_per_agent_step"] == expected_payload * 64
    assert plain["intervention"] == "none" and silent["intervention"] == "comm_off"
    assert all(e["intervention_exposed"] for e in silent["per_episode"])
    assert sum(e["scheduled_event_exposed"] for e in plain["per_episode"]) <= 2
    assert all(e["event_exposed"] == e["scheduled_event_exposed"] for e in plain["per_episode"])
    # The model is untouched by the frozen evaluation, and the unconnected
    # models ignore the intervention exactly.
    if mode not in GNN_MODES:
        assert [e["team_return"] for e in plain["per_episode"]] == [
            e["team_return"] for e in silent["per_episode"]]


def test_collector_workers_rebuild_baselines_from_the_model_config():
    from softrole.train import worker_collect
    for mode in MODES:
        config = tiny_config(mode)
        torch.manual_seed(config.seed)
        model = build_model(config.model_kwargs()).double()
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        batch = worker_collect(state, config.model_kwargs(), config.to_dict(), 0, 0)
        assert len(batch["gradients"]) == len(list(model.parameters()))
        assert batch["num_episodes"] >= 1
