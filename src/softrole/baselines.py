"""Capability-conditioned baseline actors that train through the SoftRole pipeline.

Each baseline keeps the SoftRole environment, learner, critic arithmetic, rollout
and frozen evaluator; only the actor differs.  Sources are Howell et al. 2023
(``ca_mlp``, ``ca_cc_gnn``, ``ca_gnn``) and CASH, Fu et al. 2025 (``rnn_exp``, ``cash``).
The actor reads its own capabilities from the observation tail ``obs[:, -2:]``;
``kappa`` enters only the action mask and the critic, as in ``SoftRoleNet``.
No baseline receives the previous action or the remaining time in the actor.

Dense adjacency is indexed ``[receiver, sender]``.  Baselines always use the
real-valued channel and no gate, so noise and gate overrides are rejected.

``class_typed`` is the exception to "capability-conditioned": a controlled
class-typed variant of the shared CapCon actor in which every actor parameter is
indexed by agent class (``kappa[:, 1] > 0`` is capture, otherwise perception).
It keeps the SoftRole recurrence, attention communication and critic.

``capcon_hyper`` is the fourth way of representing agent ability on that same body:
the shared CapCon actor without its capability input, whose output layer is replaced
by the CASH hypernetwork head (weights generated from the agent's own capability
entries).  It differs from ``cash`` in the body (LSTM and attention communication
instead of a GRU) and from the shared CapCon actor only in where the capabilities
enter (the head, not the recurrent input).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from softrole.config import BASELINE_MODES  # noqa: F401  (re-exported)
from softrole.model import CommunicationRound


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


class ClassTypedNet(nn.Module):
    """``class_typed``: the shared CapCon actor with every parameter indexed by agent class.

    Follows ``SoftRoleNet.forward`` in its shared mode: ``LSTMCell`` input
    ``[ReLU(pre(obs)), u_bar (if feedback), one-hot previous action]``, then
    ``comm_rounds`` rounds of ``CommunicationRound`` and a linear head.  Agent i
    has class ``c_i = 1`` if ``kappa[i, 1] > 0`` (capture), else ``0``
    (perception) and uses the encoder, LSTM cell and output head of its class.
    The communication rounds are built with two experts and the one-hot class
    matrix as the gate, which makes the self map, sender encoder, receiver
    decoder, attention, score and null logit class-specific.  The critic is the
    shared ``CapabilityCritic`` on the LSTM state.  With both classes' weights
    set to a shared ``SoftRoleNet`` the two models agree exactly.

    The real channel carries no noise, there is no gate to override, and the
    rollout's own payload formula applies (so no ``payload_values_per_agent_step``).
    """

    mode = "class_typed"
    CLASSES = 2

    def __init__(self, mode: str = "class_typed", base: int = 25, n_squares: int = 25,
                 pre_dim: int = 128, hidden_dim: int = 64, heads: int = 4, head_dim: int = 16,
                 msg_dim: int = 16, feedback: bool = True, comm_rounds: int = 2,
                 independent_heads: bool = False, null_degree_correction: bool = False,
                 critic_mode: str = "team"):
        super().__init__()
        if mode != self.mode:
            raise ValueError(f"{type(self).__name__} implements mode {self.mode!r}, not {mode!r}")
        if any(type(value) is not int or value <= 0
               for value in (base, n_squares, pre_dim, hidden_dim, heads, head_dim, msg_dim)):
            raise ValueError("All model dimensions must be positive")
        if type(comm_rounds) is not int or comm_rounds <= 0:
            raise ValueError("comm_rounds must be a positive integer")
        if any(type(value) is not bool for value in (feedback, independent_heads, null_degree_correction)):
            raise ValueError("feedback, independent_heads and null_degree_correction must be boolean")
        self.allow_stay = True
        self.experts, self.communication = self.CLASSES, "real"
        self.obs_dim = n_squares * (base + 3) + CAPABILITY_WIDTH
        self.hidden_dim, self.head_dim, self.msg_dim = hidden_dim, head_dim, msg_dim
        self.heads, self.comm_rounds, self.feedback = heads, comm_rounds, feedback
        self.independent_heads, self.null_degree_correction = independent_heads, null_degree_correction
        self.critic_mode = critic_mode
        self.pre = nn.ModuleList([nn.Linear(self.obs_dim, pre_dim) for _ in range(self.CLASSES)])
        self.lstm = nn.ModuleList([nn.LSTMCell(pre_dim + (head_dim if feedback else 0) + 6, hidden_dim)
                                   for _ in range(self.CLASSES)])
        self.rounds = nn.ModuleList([
            CommunicationRound(hidden_dim if index == 0 else heads * head_dim, heads, head_dim, msg_dim,
                               self.CLASSES, final=index == comm_rounds - 1,
                               independent_heads=independent_heads, communication="real",
                               degree_correction=null_degree_correction)
            for index in range(comm_rounds)
        ])
        self.output = nn.ModuleList([nn.Linear(head_dim, ACTIONS) for _ in range(self.CLASSES)])
        self.critic_head = CapabilityCritic(hidden_dim, critic_mode)

    def initial_memory(self, n: int) -> dict[str, Tensor]:
        """Zero local state; previous action is stay (index 4)."""
        if n <= 0:
            raise ValueError("The roster must contain at least one active agent")
        reference = self.pre[0].weight
        return {"H": reference.new_zeros(n, self.hidden_dim),
                "C": reference.new_zeros(n, self.hidden_dim),
                "u_bar": reference.new_zeros(n, self.head_dim),
                "a_prev": torch.full((n,), 4, dtype=torch.long, device=reference.device)}

    @staticmethod
    def detach_memory(memory: dict[str, Tensor]) -> dict[str, Tensor]:
        """Cut every recurrent gradient path at the same truncation boundary."""
        return {name: tensor.detach() for name, tensor in memory.items()}

    def _by_class(self, classes: Tensor, apply, *inputs: Tensor) -> list[Tensor]:
        """Run ``apply(c, *rows)`` on the rows of each present class; scatter back in roster order.

        A class absent from the roster is never evaluated, so its parameters
        receive no gradient.
        """
        outputs = None
        for index in range(self.CLASSES):
            rows = classes == index
            if not rows.any():
                continue
            result = apply(index, *(value[rows] for value in inputs))
            result = result if isinstance(result, tuple) else (result,)
            if outputs is None:
                outputs = [part.new_zeros((len(classes),) + part.shape[1:]) for part in result]
            for output, part in zip(outputs, result):
                output[rows] = part
        return outputs

    def forward(self, obs: Tensor, kappa: Tensor, memory: dict[str, Tensor],
                adjacency: Tensor, remaining: float, noise=None, gate_override=None,
                comm_off: bool = False) -> dict:
        if noise is not None:
            raise ValueError("Class-typed actors use the real channel and do not accept noise")
        if gate_override is not None:
            raise ValueError("Class-typed actors have no gate to override")
        n = obs.shape[0]
        if n == 0 or obs.shape != (n, self.obs_dim) or kappa.shape != (n, 2):
            raise ValueError("Observation/capability shapes do not match this nonempty roster")
        if adjacency.shape != (n, n) or adjacency.dtype != torch.bool:
            raise ValueError("Adjacency must be a boolean [receiver, sender] matrix")
        classes = (kappa[:, 1] > 0).long()
        gate = F.one_hot(classes, num_classes=self.CLASSES).to(obs.dtype)
        previous_action = F.one_hot(memory["a_prev"].long(), num_classes=6).to(obs.dtype)
        (encoded,) = self._by_class(classes, lambda c, x: self.pre[c](x), obs)
        recurrent_inputs = [F.relu(encoded)]
        if self.feedback:
            recurrent_inputs.append(memory["u_bar"])
        recurrent_inputs.append(previous_action)
        hidden, cell = self._by_class(classes, lambda c, x, h, s: self.lstm[c](x, (h, s)),
                                      torch.cat(recurrent_inputs, dim=-1), memory["H"], memory["C"])
        features, messages, alpha_null = hidden, [], []
        for layer in self.rounds:
            features, message, null = layer(features, gate, adjacency, None, comm_off)
            messages.append(message)
            alpha_null.append(null)
        (logits,) = self._by_class(classes, lambda c, x: self.output[c](x), features)
        feasible = torch.ones_like(logits, dtype=torch.bool)
        feasible[:, 4] = self.allow_stay
        feasible[:, 5] = kappa[:, 1] > 0
        logits = logits.masked_fill(~feasible, -torch.inf)
        value, agent_values = self.critic_head(hidden, obs, kappa, remaining)
        next_memory = {"H": hidden, "C": cell, "u_bar": features, "a_prev": memory["a_prev"]}
        return {"logits": logits, "value": value, "agent_values": agent_values,
                "memory": next_memory, "gate": gate, "bits": [],
                "messages": messages, "alpha_null": alpha_null}


class CapConHyperNet(nn.Module):
    """``capcon_hyper``: the shared CapCon body with a capability-hypernetwork output head.

    The body is ``SoftRoleNet`` in its shared mode with ``actor_capabilities=
    "physical_only"``: ``LSTMCell`` input ``[ReLU(pre(obs[:, :-2])), u_bar (if
    feedback), one-hot previous action]``, then ``comm_rounds`` rounds of
    ``CommunicationRound`` built with one expert and an all-ones ``N x 1`` gate.
    The output layer is replaced by the ``cash`` head: ``hyper_w`` and ``hyper_b``
    read the full observation (the agent's own capability entries included) and
    emit the per-agent output weights ``W_i`` (``head_dim x 6``) and biases
    ``b_i``, so that ``logits_i = u_i W_i + b_i`` with ``u`` the final-round
    feature.  Capabilities therefore reach the logits only through the head; the
    LSTM state, messages and null attention do not depend on them.  Teammate
    capabilities are not an input to the hypernetworks (matched-information rule
    of ``cash``).  The critic is the shared ``CapabilityCritic`` on the LSTM state.
    With a constant hypernetwork map equal to a shared output layer the model
    agrees exactly with ``SoftRoleNet(mode="shared", actor_capabilities=
    "physical_only")``.

    The real channel carries no noise, there is no gate to override, and the
    rollout's own payload formula applies (so no ``payload_values_per_agent_step``).
    """

    mode = "capcon_hyper"

    def __init__(self, mode: str = "capcon_hyper", base: int = 25, n_squares: int = 25,
                 pre_dim: int = 128, hidden_dim: int = 64, heads: int = 4, head_dim: int = 16,
                 msg_dim: int = 16, feedback: bool = True, comm_rounds: int = 2,
                 independent_heads: bool = False, null_degree_correction: bool = False,
                 critic_mode: str = "team", hyper_dim: int = 64):
        super().__init__()
        if mode != self.mode:
            raise ValueError(f"{type(self).__name__} implements mode {self.mode!r}, not {mode!r}")
        if any(type(value) is not int or value <= 0
               for value in (base, n_squares, pre_dim, hidden_dim, heads, head_dim, msg_dim)):
            raise ValueError("All model dimensions must be positive")
        if type(comm_rounds) is not int or comm_rounds <= 0:
            raise ValueError("comm_rounds must be a positive integer")
        if type(hyper_dim) is not int or hyper_dim <= 0:
            raise ValueError("hyper_dim must be a positive integer")
        if any(type(value) is not bool for value in (feedback, independent_heads, null_degree_correction)):
            raise ValueError("feedback, independent_heads and null_degree_correction must be boolean")
        self.allow_stay = True
        self.experts, self.communication = 1, "real"
        self.obs_dim = n_squares * (base + 3) + CAPABILITY_WIDTH
        self.hidden_dim, self.head_dim, self.msg_dim = hidden_dim, head_dim, msg_dim
        self.heads, self.comm_rounds, self.feedback = heads, comm_rounds, feedback
        self.independent_heads, self.null_degree_correction = independent_heads, null_degree_correction
        self.hyper_dim = hyper_dim
        self.critic_mode = critic_mode
        self.pre = nn.Linear(self.obs_dim - CAPABILITY_WIDTH, pre_dim)
        self.lstm = nn.LSTMCell(pre_dim + (head_dim if feedback else 0) + 6, hidden_dim)
        self.rounds = nn.ModuleList([
            CommunicationRound(hidden_dim if index == 0 else heads * head_dim, heads, head_dim, msg_dim,
                               self.experts, final=index == comm_rounds - 1,
                               independent_heads=independent_heads, communication="real",
                               degree_correction=null_degree_correction)
            for index in range(comm_rounds)
        ])
        self.hyper_w = HyperNetwork(self.obs_dim, hyper_dim, head_dim * ACTIONS, zero_init=False)
        self.hyper_b = HyperNetwork(self.obs_dim, hyper_dim, ACTIONS, zero_init=True)
        self.critic_head = CapabilityCritic(hidden_dim, critic_mode)

    def initial_memory(self, n: int) -> dict[str, Tensor]:
        """Zero local state; previous action is stay (index 4)."""
        if n <= 0:
            raise ValueError("The roster must contain at least one active agent")
        reference = self.pre.weight
        return {"H": reference.new_zeros(n, self.hidden_dim),
                "C": reference.new_zeros(n, self.hidden_dim),
                "u_bar": reference.new_zeros(n, self.head_dim),
                "a_prev": torch.full((n,), 4, dtype=torch.long, device=reference.device)}

    @staticmethod
    def detach_memory(memory: dict[str, Tensor]) -> dict[str, Tensor]:
        """Cut every recurrent gradient path at the same truncation boundary."""
        return {name: tensor.detach() for name, tensor in memory.items()}

    def forward(self, obs: Tensor, kappa: Tensor, memory: dict[str, Tensor],
                adjacency: Tensor, remaining: float, noise=None, gate_override=None,
                comm_off: bool = False) -> dict:
        if noise is not None:
            raise ValueError("CapCon-hypernetwork actors use the real channel and do not accept noise")
        if gate_override is not None:
            raise ValueError("CapCon-hypernetwork actors have no gate to override")
        n = obs.shape[0]
        if n == 0 or obs.shape != (n, self.obs_dim) or kappa.shape != (n, 2):
            raise ValueError("Observation/capability shapes do not match this nonempty roster")
        if adjacency.shape != (n, n) or adjacency.dtype != torch.bool:
            raise ValueError("Adjacency must be a boolean [receiver, sender] matrix")
        gate = obs.new_ones(n, 1)
        previous_action = F.one_hot(memory["a_prev"].long(), num_classes=ACTIONS).to(obs.dtype)
        recurrent_inputs = [F.relu(self.pre(obs[:, :-CAPABILITY_WIDTH]))]
        if self.feedback:
            recurrent_inputs.append(memory["u_bar"])
        recurrent_inputs.append(previous_action)
        hidden, cell = self.lstm(torch.cat(recurrent_inputs, dim=-1), (memory["H"], memory["C"]))
        features, messages, alpha_null = hidden, [], []
        for layer in self.rounds:
            features, message, null = layer(features, gate, adjacency, None, comm_off)
            messages.append(message)
            alpha_null.append(null)
        weights = self.hyper_w(obs).reshape(n, self.head_dim, ACTIONS)
        biases = self.hyper_b(obs)
        logits = torch.einsum("nh,nha->na", features, weights) + biases
        feasible = torch.ones_like(logits, dtype=torch.bool)
        feasible[:, 4] = self.allow_stay
        feasible[:, 5] = kappa[:, 1] > 0
        logits = logits.masked_fill(~feasible, -torch.inf)
        value, agent_values = self.critic_head(hidden, obs, kappa, remaining)
        next_memory = {"H": hidden, "C": cell, "u_bar": features, "a_prev": memory["a_prev"]}
        return {"logits": logits, "value": value, "agent_values": agent_values,
                "memory": next_memory, "gate": gate, "bits": [],
                "messages": messages, "alpha_null": alpha_null}


BASELINE_CLASSES = {cls.mode: cls for cls in (CapabilityMLP, CapabilityCommGNN, CapabilityGNN,
                                              ExplicitRNN, CashNet, ClassTypedNet, CapConHyperNet)}
if set(BASELINE_CLASSES) != set(BASELINE_MODES):
    raise RuntimeError("BASELINE_MODES and BASELINE_CLASSES disagree")
