"""Capability-conditioned baseline actors that train through the SoftRole pipeline.

Each baseline keeps the SoftRole environment, learner, critic arithmetic, rollout
and frozen evaluator; only the actor differs.  Sources are Howell et al. 2023
(``ca_mlp``, ``ca_cc_gnn``, ``ca_gnn``) and CASH, Fu et al. 2025 (``rnn_exp``, ``cash``).
The actor reads its own capabilities from the observation tail ``obs[:, -2:]``;
``kappa`` enters only the action mask and the critic, as in ``SoftRoleNet``.
No baseline receives the previous action or the remaining time in the actor.

Dense adjacency is indexed ``[receiver, sender]``.  Baselines always use the
real-valued channel and no gate, so noise and gate overrides are rejected.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from softrole.config import BASELINE_MODES  # noqa: F401  (re-exported)


ACTIONS = 6
CAPABILITY_WIDTH = 2


class CapabilityCritic(nn.Module):
    """The SoftRoleNet critic with a baseline feature vector in place of the LSTM state.

    Layers, sizes and arithmetic follow ``model.py`` (``critic_agent`` and
    ``critic``) for both the team and the decomposed value head.
    """

    def __init__(self, hidden_dim: int, critic_mode: str):
        super().__init__()
        if critic_mode not in ("team", "decomposed"):
            raise ValueError("critic_mode must be team or decomposed")
        self.critic_mode = critic_mode
        self.critic_agent = nn.Sequential(nn.Linear(hidden_dim + 2, 64), nn.ReLU())
        if critic_mode == "team":
            self.critic = nn.Sequential(nn.Linear(68, 64), nn.ReLU(), nn.Linear(64, 1))
        else:
            self.critic = nn.Sequential(nn.Linear(hidden_dim + 2 + 64 + 4, 64), nn.ReLU(),
                                        nn.Linear(64, 1))

    def forward(self, features: Tensor, obs: Tensor, kappa: Tensor, remaining: float):
        n = obs.shape[0]
        pooled = self.critic_agent(torch.cat((features, kappa), dim=-1)).mean(dim=0)
        remaining_tensor = torch.as_tensor(remaining, dtype=obs.dtype, device=obs.device).reshape(1)
        team = torch.cat((obs.new_tensor([n]), kappa.sum(dim=0), remaining_tensor))
        if self.critic_mode == "team":
            return self.critic(torch.cat((pooled, team))).squeeze(-1), None
        per_agent = torch.cat((features, kappa, pooled.unsqueeze(0).expand(n, -1),
                               team.unsqueeze(0).expand(n, -1)), dim=-1)
        agent_values = self.critic(per_agent).squeeze(-1)
        return agent_values.sum(), agent_values


class _BaselineNet(nn.Module):
    """Shared SoftRoleNet-compatible forward contract; subclasses define ``_act``."""

    mode: str = ""

    def __init__(self, mode: str, base: int = 25, n_squares: int = 25,
                 hidden_dim: int = 64, critic_mode: str = "team"):
        super().__init__()
        if mode != self.mode:
            raise ValueError(f"{type(self).__name__} implements mode {self.mode!r}, not {mode!r}")
        if any(type(value) is not int or value <= 0 for value in (base, n_squares, hidden_dim)):
            raise ValueError("All model dimensions must be positive")
        self.allow_stay = True
        self.obs_dim = n_squares * (base + 3) + CAPABILITY_WIDTH
        self.hidden_dim = hidden_dim
        self.critic_mode = critic_mode
        # Real values broadcast per agent and step; the rollout reports this size.
        self.payload_values_per_agent_step = 0
        self.critic_head = CapabilityCritic(hidden_dim, critic_mode)

    def initial_memory(self, n: int) -> dict[str, Tensor]:
        """Zero local state; previous action is stay (index 4)."""
        if n <= 0:
            raise ValueError("The roster must contain at least one active agent")
        reference = next(self.parameters())
        memory = self._initial_state(n, reference)
        memory["a_prev"] = torch.full((n,), 4, dtype=torch.long, device=reference.device)
        return memory

    def _initial_state(self, n: int, reference: Tensor) -> dict[str, Tensor]:
        return {}

    @staticmethod
    def detach_memory(memory: dict[str, Tensor]) -> dict[str, Tensor]:
        """Cut every recurrent gradient path at the same truncation boundary."""
        return {name: tensor.detach() for name, tensor in memory.items()}

    def _act(self, obs: Tensor, memory: dict[str, Tensor], adjacency: Tensor, comm_off: bool):
        """Return ``(logits, critic features, new local memory, sent payloads)``."""
        raise NotImplementedError

    def forward(self, obs: Tensor, kappa: Tensor, memory: dict[str, Tensor],
                adjacency: Tensor, remaining: float, noise=None, gate_override=None,
                comm_off: bool = False) -> dict:
        if noise is not None:
            raise ValueError("Baseline actors use the real channel and do not accept noise")
        if gate_override is not None:
            raise ValueError("Baseline actors have no gate to override")
        n = obs.shape[0]
        if n == 0 or obs.shape != (n, self.obs_dim) or kappa.shape != (n, 2):
            raise ValueError("Observation/capability shapes do not match this nonempty roster")
        if adjacency.shape != (n, n) or adjacency.dtype != torch.bool:
            raise ValueError("Adjacency must be a boolean [receiver, sender] matrix")
        logits, features, next_memory, messages = self._act(obs, memory, adjacency, comm_off)
        feasible = torch.ones_like(logits, dtype=torch.bool)
        feasible[:, 4] = self.allow_stay
        feasible[:, 5] = kappa[:, 1] > 0
        logits = logits.masked_fill(~feasible, -torch.inf)
        value, agent_values = self.critic_head(features, obs, kappa, remaining)
        next_memory["a_prev"] = memory["a_prev"]
        silent = comm_off or not messages
        alpha_null = obs.new_ones(n, 1) if silent else obs.new_zeros(n, 1)
        return {"logits": logits, "value": value, "agent_values": agent_values,
                "memory": next_memory, "gate": obs.new_ones(n, 1), "bits": [],
                "messages": messages, "alpha_null": [alpha_null]}


class CapabilityMLP(_BaselineNet):
    """``ca_mlp``: obs -> Linear -> 3 x Linear -> Linear(A), ReLU after the first four."""

    mode = "ca_mlp"

    def __init__(self, mode: str = "ca_mlp", **kwargs):
        super().__init__(mode, **kwargs)
        h = self.hidden_dim
        self.trunk = nn.Sequential(nn.Linear(self.obs_dim, h), nn.ReLU(),
                                   nn.Linear(h, h), nn.ReLU(),
                                   nn.Linear(h, h), nn.ReLU(),
                                   nn.Linear(h, h), nn.ReLU())
        self.output = nn.Linear(h, ACTIONS)

    def _act(self, obs, memory, adjacency, comm_off):
        features = self.trunk(obs)
        return self.output(features), features, {}, []


class _GraphNet(_BaselineNet):
    """Sum-aggregation message passing without self-loops (Howell et al. 2023)."""

    tail_after_message = False

    def __init__(self, mode: str, **kwargs):
        super().__init__(mode, **kwargs)
        h = self.hidden_dim
        self.payload_values_per_agent_step = h
        tail = CAPABILITY_WIDTH if self.tail_after_message else 0
        self.encoder = nn.Linear(self.obs_dim - tail, h)
        self.message = nn.Sequential(nn.Linear(h, h, bias=False), nn.ReLU(),
                                     nn.Linear(h, h, bias=False))
        self.fuse = nn.Linear(2 * h + tail, h)
        self.output = nn.Linear(h, ACTIONS)

    def _encode(self, obs: Tensor) -> Tensor:
        actor_obs = obs[:, :-CAPABILITY_WIDTH] if self.tail_after_message else obs
        return F.relu(self.encoder(actor_obs))

    def _act(self, obs, memory, adjacency, comm_off):
        encoded = self._encode(obs)
        # agg_i = sum_k adjacency[i, k] * enc_k; zero aggregate => zero message (no biases).
        aggregate = (torch.zeros_like(encoded) if comm_off
                     else adjacency.to(encoded.dtype) @ encoded)
        message = self.message(aggregate)
        parts = [encoded, message]
        if self.tail_after_message:
            parts.append(obs[:, -CAPABILITY_WIDTH:])
        features = F.relu(self.fuse(torch.cat(parts, dim=-1)))
        return self.output(features), features, {}, [encoded]


class CapabilityCommGNN(_GraphNet):
    """``ca_cc_gnn``: the encoder sees the whole observation, tail included."""

    mode = "ca_cc_gnn"

    def __init__(self, mode: str = "ca_cc_gnn", **kwargs):
        super().__init__(mode, **kwargs)


class CapabilityGNN(_GraphNet):
    """``ca_gnn``: the tail bypasses the encoder and joins after the message step."""

    mode = "ca_gnn"
    tail_after_message = True

    def __init__(self, mode: str = "ca_gnn", **kwargs):
        super().__init__(mode, **kwargs)


class _RecurrentNet(_BaselineNet):
    def _initial_state(self, n, reference):
        return {"H": reference.new_zeros(n, self.hidden_dim)}


class ExplicitRNN(_RecurrentNet):
    """``rnn_exp``: ReLU(Linear(obs)) -> GRUCell -> Linear(A)."""

    mode = "rnn_exp"

    def __init__(self, mode: str = "rnn_exp", **kwargs):
        super().__init__(mode, **kwargs)
        h = self.hidden_dim
        self.encoder = nn.Linear(self.obs_dim, h)
        self.cell = nn.GRUCell(h, h)
        self.output = nn.Linear(h, ACTIONS)

    def _act(self, obs, memory, adjacency, comm_off):
        state = self.cell(F.relu(self.encoder(obs)), memory["H"])
        return self.output(state), state, {"H": state}, []


class HyperNetwork(nn.Module):
    """``3 x [Linear -> LayerNorm -> ReLU]`` then ``Linear`` (CASH, ``HyperNetwork``)."""

    def __init__(self, in_dim: int, width: int, out_dim: int, zero_init: bool):
        super().__init__()
        layers, previous = [], in_dim
        for _ in range(3):
            layers += [nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, out_dim))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    if zero_init:
                        nn.init.zeros_(layer.weight)
                    else:
                        nn.init.orthogonal_(layer.weight, gain=0.2)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CashNet(_RecurrentNet):
    """``cash``: a GRU on the physical observation with a capability-hypernetwork head.

    ``z = GRUCell(ReLU(Linear(obs[:, :-2])))``; the hypernetworks read the full
    observation (tail included) and emit the per-agent output weights ``W`` and
    biases ``b`` so that ``logits_i = z_i W_i + b_i``.  Teammate capabilities are
    deliberately not an input (matched-information variant).
    """

    mode = "cash"

    def __init__(self, mode: str = "cash", hyper_dim: int = 64, **kwargs):
        super().__init__(mode, **kwargs)
        if type(hyper_dim) is not int or hyper_dim <= 0:
            raise ValueError("hyper_dim must be a positive integer")
        h = self.hidden_dim
        self.hyper_dim = hyper_dim
        self.encoder = nn.Linear(self.obs_dim - CAPABILITY_WIDTH, h)
        self.cell = nn.GRUCell(h, h)
        self.hyper_w = HyperNetwork(self.obs_dim, hyper_dim, h * ACTIONS, zero_init=False)
        self.hyper_b = HyperNetwork(self.obs_dim, hyper_dim, ACTIONS, zero_init=True)

    def _act(self, obs, memory, adjacency, comm_off):
        n = obs.shape[0]
        state = self.cell(F.relu(self.encoder(obs[:, :-CAPABILITY_WIDTH])), memory["H"])
        weights = self.hyper_w(obs).reshape(n, self.hidden_dim, ACTIONS)
        biases = self.hyper_b(obs)
        logits = torch.einsum("nh,nha->na", state, weights) + biases
        return logits, state, {"H": state}, []


BASELINE_CLASSES = {cls.mode: cls for cls in (CapabilityMLP, CapabilityCommGNN, CapabilityGNN,
                                              ExplicitRNN, CashNet)}
if set(BASELINE_CLASSES) != set(BASELINE_MODES):
    raise RuntimeError("BASELINE_MODES and BASELINE_CLASSES disagree")
