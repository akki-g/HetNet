"""Shared capability-conditioned recurrent actors with a binary message channel.

The gate is deterministic.  Gaussian roles and information-rate penalties are
deliberately absent.  Dense adjacency is indexed ``[receiver, sender]``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def straight_through_bits(logits: Tensor, noise: Tensor | None = None) -> Tensor:
    """Bernoulli(sigmoid(u)) forward; biased sigmoid(u + Logistic) backward.

    Injecting the logistic noise couples otherwise stochastic forward passes.
    The same sampled bits are broadcast to every receiver and attention head.
    """
    if noise is None:
        eps = torch.finfo(logits.dtype).eps
        uniform = torch.rand_like(logits).clamp(eps, 1 - eps)
        noise = uniform.log() - torch.log1p(-uniform)
    elif noise.shape != logits.shape:
        raise ValueError("Bit noise must have the same shape as the encoder logits")
    relaxed = torch.sigmoid(logits + noise)
    hard = ((logits + noise) > 0).to(logits.dtype)
    return hard + (relaxed - relaxed.detach())


class AffineBank(nn.Module):
    """B(x,g) = sum_c g_c (W_c x + b_c), including broadcast batch axes."""

    def __init__(self, experts: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(experts, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(experts, out_dim))
        with torch.no_grad():
            for weight in self.weight:
                nn.init.xavier_uniform_(weight)

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        outputs = torch.einsum("...i,koi->...ko", x, self.weight) + self.bias
        return (outputs * gate.unsqueeze(-1)).sum(-2)


class HeadBank(nn.Module):
    """An affine bank for each attention head, with one shared agent gate."""

    def __init__(self, experts: int, heads: int, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(experts, heads, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(experts, heads, out_dim)) if bias else None
        with torch.no_grad():
            for expert in self.weight:
                for weight in expert:
                    nn.init.xavier_uniform_(weight)

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        outputs = torch.einsum("...hi,khoi->...kho", x, self.weight)
        if self.bias is not None:
            outputs = outputs + self.bias
        return (outputs * gate.unsqueeze(-1).unsqueeze(-1)).sum(-3)


class CommunicationRound(nn.Module):
    """Receiver-gated GATv2 with a zero-message null softmax candidate.

    z_j=B_self(h_j,g_j), b_k=ST(B_enc(h_k,g_k)),
    m_jk=B_dec(b_k,g_j), h'_j=z_j+sum_k alpha_jk m_jk.
    Only hidden rounds apply ReLU and concatenate heads; the final round
    averages heads without a final nonlinearity.
    """

    def __init__(self, in_dim: int, heads: int, head_dim: int, msg_dim: int,
                 experts: int, final: bool):
        super().__init__()
        self.heads, self.head_dim, self.final = heads, head_dim, final
        self.self_map = AffineBank(experts, in_dim, heads * head_dim)
        self.encoder = AffineBank(experts, in_dim, msg_dim)
        self.decoder = AffineBank(experts, msg_dim, heads * head_dim)
        self.attention = HeadBank(experts, heads, 2 * head_dim, head_dim)
        self.score = HeadBank(experts, heads, head_dim, 1, bias=False)
        self.null_logits = nn.Parameter(torch.zeros(experts, heads))

    def forward(self, h: Tensor, gate: Tensor, adjacency: Tensor,
                noise: Tensor | None = None, comm_off: bool = False):
        n = h.shape[0]
        z = self.self_map(h, gate).reshape(n, self.heads, self.head_dim)
        bits = straight_through_bits(self.encoder(h, gate), noise)
        if comm_off:
            aggregate = torch.zeros_like(z)
            alpha_null = h.new_ones(n, self.heads)
        else:
            # Axes are receiver, sender, head, feature.  The receiver's gate
            # selects its decoder, even when the sender has a different gate.
            payload = bits.unsqueeze(0).expand(n, n, -1)
            receiver_gate = gate.unsqueeze(1)
            messages = self.decoder(payload, receiver_gate).reshape(
                n, n, self.heads, self.head_dim)
            query = z.unsqueeze(1).expand(n, n, -1, -1)
            pair = torch.cat((query, messages), dim=-1)
            keys = F.leaky_relu(self.attention(pair, receiver_gate), negative_slope=0.2)
            scores = self.score(keys, receiver_gate).squeeze(-1)
            scores = scores.masked_fill(~adjacency.unsqueeze(-1), -torch.inf)
            null = gate @ self.null_logits
            alpha = torch.softmax(torch.cat((scores, null.unsqueeze(1)), dim=1), dim=1)
            aggregate = (alpha[:, :-1].unsqueeze(-1) * messages).sum(dim=1)
            alpha_null = alpha[:, -1]
        updated = z + aggregate
        output = updated.mean(dim=1) if self.final else F.relu(updated).flatten(1)
        return output, bits, alpha_null


class SoftRoleNet(nn.Module):
    """One actor shared across a variable-size roster, plus a team-value critic.

    ``obs`` has shape N x (n_squares * (base + 3) + 2) after the observation
    adapter.  Its final two entries already contain own capabilities.  Separate
    ``kappa`` (N x 2) supplies gate conditioning and physical action feasibility.
    ``remaining`` and aggregate team capabilities enter only the critic.

    Modes: banked=gate(H,kappa), capability=gate(kappa), constant=learned fixed
    gate, shared=one actual expert with no gate parameters.  All modes retain
    capability-aware shared recurrence.  ``feedback=False`` removes the
    previous communicated features from the LSTM input.
    """

    def __init__(self, base: int = 25, n_squares: int = 25, mode: str = "banked",
                 experts: int = 4, pre_dim: int = 128, hidden_dim: int = 64,
                 heads: int = 4, head_dim: int = 16, msg_dim: int = 16,
                 feedback: bool = True, allow_stay: bool = True):
        super().__init__()
        if type(allow_stay) is not bool:
            raise ValueError("allow_stay must be boolean")
        self.allow_stay = allow_stay
        if mode not in {"banked", "shared", "capability", "constant"}:
            raise ValueError(f"Unknown gate mode: {mode}")
        dimensions = (base, n_squares, experts, pre_dim, hidden_dim, heads, head_dim, msg_dim)
        if any(value <= 0 for value in dimensions):
            raise ValueError("All model dimensions must be positive")
        self.mode, self.experts, self.feedback = mode, (1 if mode == "shared" else experts), feedback
        self.obs_dim = n_squares * (base + 3) + 2
        self.hidden_dim, self.head_dim, self.msg_dim = hidden_dim, head_dim, msg_dim
        self.pre = nn.Linear(self.obs_dim, pre_dim)
        self.lstm = nn.LSTMCell(pre_dim + (head_dim if feedback else 0) + 6, hidden_dim)
        self.gate_network = None
        self.register_parameter("constant_logits", None)
        if mode in {"banked", "capability"}:
            gate_in = hidden_dim + 2 if mode == "banked" else 2
            self.gate_network = nn.Sequential(nn.Linear(gate_in, 32), nn.ReLU(),
                                              nn.Linear(32, self.experts))
        elif mode == "constant":
            self.constant_logits = nn.Parameter(torch.zeros(self.experts))
        self.rounds = nn.ModuleList([
            CommunicationRound(hidden_dim, heads, head_dim, msg_dim, self.experts, final=False),
            CommunicationRound(heads * head_dim, heads, head_dim, msg_dim, self.experts, final=True),
        ])
        self.output = nn.Linear(head_dim, 6)
        self.critic_agent = nn.Sequential(nn.Linear(hidden_dim + 2, 64), nn.ReLU())
        self.critic = nn.Sequential(nn.Linear(68, 64), nn.ReLU(), nn.Linear(64, 1))

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

    def _gate(self, hidden: Tensor, kappa: Tensor) -> Tensor:
        if self.mode == "shared":
            return hidden.new_ones(hidden.shape[0], 1)
        if self.mode == "constant":
            return torch.softmax(self.constant_logits, dim=-1).expand(hidden.shape[0], -1)
        gate_input = torch.cat((hidden, kappa), dim=-1) if self.mode == "banked" else kappa
        return torch.softmax(self.gate_network(gate_input), dim=-1)

    def forward(self, obs: Tensor, kappa: Tensor, memory: dict[str, Tensor],
                adjacency: Tensor, remaining: float, noise: list[Tensor] | None = None,
                gate_override: Tensor | tuple[Tensor, Tensor] | None = None,
                comm_off: bool = False) -> dict:
        """Return actor logits, scalar team value, and next local memory.

        ``adjacency[j,k]`` permits sender k to reach receiver j.  The caller
        writes the sampled actions to returned memory['a_prev'] before the
        next step.  Injected noise contains two N x msg_dim logistic tensors.
        A gate override is either the full gate or ``(agent_mask, pinned_gate)``
        to freeze selected agents while all others continue adapting.
        """
        n = obs.shape[0]
        if n == 0 or obs.shape != (n, self.obs_dim) or kappa.shape != (n, 2):
            raise ValueError("Observation/capability shapes do not match this nonempty roster")
        if adjacency.shape != (n, n) or adjacency.dtype != torch.bool:
            raise ValueError("Adjacency must be a boolean [receiver, sender] matrix")
        if noise is not None and len(noise) != 2:
            raise ValueError("Supply one bit-noise tensor for each of the two rounds")
        previous_action = F.one_hot(memory["a_prev"].long(), num_classes=6).to(obs.dtype)
        recurrent_inputs = [F.relu(self.pre(obs))]
        if self.feedback:
            recurrent_inputs.append(memory["u_bar"])
        recurrent_inputs.append(previous_action)
        hidden, cell = self.lstm(torch.cat(recurrent_inputs, dim=-1), (memory["H"], memory["C"]))
        if isinstance(gate_override, tuple):
            agent_mask, pinned_gate = gate_override
            if agent_mask.shape != (n,) or agent_mask.dtype != torch.bool:
                raise ValueError("A partial gate override needs a boolean mask over agents")
            if pinned_gate.shape != (n, self.experts):
                raise ValueError("Pinned gate must have shape [agents, effective experts]")
            gate = torch.where(agent_mask.unsqueeze(-1), pinned_gate, self._gate(hidden, kappa))
        else:
            gate = self._gate(hidden, kappa) if gate_override is None else gate_override
        if gate.shape != (n, self.experts):
            raise ValueError("Gate override must have shape [agents, effective experts]")
        features, bits, alpha_null = hidden, [], []
        for index, layer in enumerate(self.rounds):
            features, message, null = layer(features, gate, adjacency,
                                            None if noise is None else noise[index], comm_off)
            bits.append(message)
            alpha_null.append(null)
        logits = self.output(features)
        feasible = torch.ones_like(logits, dtype=torch.bool)
        feasible[:, 4] = self.allow_stay
        feasible[:, 5] = kappa[:, 1] > 0
        logits = logits.masked_fill(~feasible, -torch.inf)

        # The critic can train shared H but its outputs never enter the actor.
        pooled = self.critic_agent(torch.cat((hidden, kappa), dim=-1)).mean(dim=0)
        remaining_tensor = torch.as_tensor(remaining, dtype=obs.dtype, device=obs.device).reshape(1)
        team = torch.cat((obs.new_tensor([n]), kappa.sum(dim=0), remaining_tensor))
        value = self.critic(torch.cat((pooled, team))).squeeze(-1)
        next_memory = {"H": hidden, "C": cell, "u_bar": features, "a_prev": memory["a_prev"]}
        return {"logits": logits, "value": value, "memory": next_memory,
                "gate": gate, "bits": bits, "alpha_null": alpha_null}
