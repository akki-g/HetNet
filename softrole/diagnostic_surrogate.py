"""D5: a bounded, conditional audit of Binary message gradients.

This is a one-step policy-score surrogate on a logged state, action and detached
advantage. Only 1--4 selected bits vary. All other payloads and the incoming
recurrent state are clamped. It is not an exact reference for an episode return,
the complete message vector, or the production recurrent policy gradient.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import SoftRoleNet


def _conditional_gradients(logits: Tensor, objective: Callable[[Tensor], Tensor],
                           mc_samples: int, seed: int) -> dict:
    """Enumerate f(b); average its production ST derivative over fresh noise."""
    if logits.ndim != 1 or not 1 <= logits.numel() <= 4:
        raise ValueError("The conditional audit requires one to four bit logits")
    if type(mc_samples) is not int or not 2 <= mc_samples <= 65536:
        raise ValueError("mc_samples must be an integer between 2 and 65536")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    fixed = logits.detach()
    if not torch.isfinite(fixed).all():
        raise ValueError("Conditional bit logits must be finite")
    patterns = fixed.new_tensor(list(itertools.product((0., 1.), repeat=fixed.numel())))
    values, derivatives = [], []
    for pattern in patterns:
        bits = pattern.clone().requires_grad_(True)
        value = objective(bits)
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError("The conditional objective must be a finite scalar")
        derivative = (torch.autograd.grad(value, bits, allow_unused=True)[0]
                      if value.requires_grad else None)
        derivative = torch.zeros_like(bits) if derivative is None else derivative.detach()
        if not torch.isfinite(derivative).all():
            raise ValueError("Conditional downstream derivatives must be finite")
        values.append(value.detach())
        derivatives.append(derivative)
    values, derivatives = torch.stack(values), torch.stack(derivatives)
    probabilities = (patterns * F.logsigmoid(fixed)
                     + (1 - patterns) * F.logsigmoid(-fixed)).sum(-1).exp()
    # Exact score-function sum over the selected Bernoulli variables. The
    # objective and every unselected payload are held fixed with respect to u.
    # A constant baseline is an exact zero-score expectation and makes an
    # identically constant objective produce an exactly zero reference.
    exact = (probabilities[:, None] * (values - values[0])[:, None]
             * (patterns - fixed.sigmoid())).sum(0)
    generator = torch.Generator(device=fixed.device).manual_seed(seed)
    eps = torch.finfo(fixed.dtype).eps
    uniform = torch.rand((mc_samples, fixed.numel()), generator=generator,
                         dtype=fixed.dtype, device=fixed.device).clamp(eps, 1 - eps)
    noisy = fixed + uniform.log() - torch.log1p(-uniform)
    powers = 2 ** torch.arange(fixed.numel() - 1, -1, -1, device=fixed.device)
    pattern_indices = ((noisy > 0).long() * powers).sum(-1)
    relaxed = noisy.sigmoid()
    # This is sigmoid'(u + Logistic), precisely the production surrogate.
    samples = derivatives[pattern_indices] * relaxed * (1 - relaxed)
    return {"exact": exact, "st": samples.mean(0),
            "st_standard_error": samples.std(0, correction=1) / mc_samples ** .5,
            "expected_objective": (probabilities * values).sum().item(),
            "patterns": [{"bits": pattern.tolist(), "probability": probability.item(),
                          "objective": value.item()}
                         for pattern, probability, value in zip(patterns, probabilities, values)]}


def audit_surrogate(model: SoftRoleNet, *, obs: Tensor, kappa: Tensor,
                    memory: dict[str, Tensor], adjacency: Tensor, remaining: float,
                    noise: Sequence[Tensor] | None, actions: Tensor,
                    advantages: Tensor | float, round_index: int = -1,
                    bit_indices: Sequence[int] | None = None, n_bits: int = 3,
                    mc_samples: int = 128, seed: int = 0) -> dict:
    """Compare conditional exact and expected-ST encoder parameter gradients.

    ``actions`` and scalar/common-team or per-agent ``advantages`` come from a
    logged step. The objective maximizes sum_i A_i log pi_i(a_i); a common actor
    coefficient may be included in the supplied advantages. Supplied ``noise``
    must reproduce the logged Binary payloads. Flat selected indices refer to
    ``[sender, bit]`` or ``[sender, head, bit]`` in the selected round. Selection
    uses a separate seeded generator and only senders with a permitted receiver.

    The original model, incoming memory, global RNG and parameter .grad buffers
    are unchanged. Temporary decoder hooks are removed even on error. Run this
    read-only probe sequentially, without concurrent forwards of the same model.
    """
    scope = ("Conditional one-step logged-action policy surrogate; incoming state and "
             "all unselected payloads are fixed. Not a full production gradient reference.")
    if model.communication != "binary":
        reasons = {"real": "Real communication has no Bernoulli bits",
                   "real_noisy": "Noisy real communication has no Bernoulli bits",
                   "binary_deterministic": "Deterministic threshold bits have no score function; "
                                           "the production surrogate is sigmoid'(u)"}
        return {"status": "not_applicable", "reason": reasons[model.communication], "scope": scope}
    if type(round_index) is not int or not -model.comm_rounds <= round_index < model.comm_rounds:
        raise ValueError("round_index must identify an existing communication round")
    round_index %= model.comm_rounds
    if type(n_bits) is not int or not 1 <= n_bits <= 4:
        raise ValueError("n_bits must be an integer between one and four")
    if type(mc_samples) is not int or not 2 <= mc_samples <= 65536:
        raise ValueError("mc_samples must be an integer between 2 and 65536")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if obs.device.type != "cpu" or obs.dtype != torch.float64:
        raise ValueError("D5 requires CPU float64 logged inputs and model")
    if noise is None or len(noise) != model.comm_rounds:
        raise ValueError("D5 requires the logged noise for every Binary round")
    if actions.shape != (obs.shape[0],) or actions.dtype != torch.long:
        raise ValueError("actions must be one long integer per agent")
    if torch.any((actions < 0) | (actions >= 6)):
        raise ValueError("Logged actions must be in [0, 5]")
    weights = torch.as_tensor(advantages, dtype=obs.dtype, device=obs.device).detach()
    if weights.ndim != 0 and weights.shape != (obs.shape[0],):
        raise ValueError("advantages must be scalar or one value per agent")
    if not torch.isfinite(weights).all():
        raise ValueError("advantages must be finite")
    obs, kappa = obs.detach(), kappa.detach()
    memory = {name: value.detach() for name, value in memory.items()}
    noise = [value.detach() for value in noise]
    captured = {}

    def capture_encoder(_module, arguments):
        captured["inputs"] = tuple(value.detach().clone() for value in arguments)

    encoder = model.rounds[round_index].encoder
    handle = encoder.register_forward_pre_hook(capture_encoder)
    try:
        with torch.no_grad():
            logged = model(obs, kappa, memory, adjacency, remaining, noise=noise)
    finally:
        handle.remove()
    logged_log_probs = F.log_softmax(logged["logits"], dim=-1).gather(1, actions[:, None]).squeeze(1)
    if not torch.isfinite(logged_log_probs).all():
        raise ValueError("The logged action is physically masked or has nonfinite probability")
    payloads = [value.detach().clone() for value in logged["messages"]]
    base = payloads[round_index]
    if bit_indices is None:
        eligible = torch.nonzero(adjacency.any(dim=0), as_tuple=False).flatten()
        if not len(eligible):
            return {"status": "not_applicable", "reason": "No sender has a permitted receiver",
                    "scope": scope, "round_index": round_index}
        width = base[0].numel()
        candidates = (eligible[:, None] * width + torch.arange(width)).flatten()
        generator = torch.Generator().manual_seed(seed)
        bit_indices = candidates[torch.randperm(len(candidates), generator=generator)
                                 [:min(n_bits, len(candidates))]].tolist()
    else:
        bit_indices = list(bit_indices)
        if (not 1 <= len(bit_indices) <= 4
                or any(type(index) is not int or not 0 <= index < base.numel()
                       for index in bit_indices)
                or len(set(bit_indices)) != len(bit_indices)):
            raise ValueError("bit_indices must contain one to four distinct valid flat indices")
    indices = torch.tensor(bit_indices, dtype=torch.long)
    # A copy supports genuinely frozen checkpoint parameters without changing
    # requires_grad on the user's model; all values are exactly the loaded ones.
    probe_encoder = copy.deepcopy(encoder).requires_grad_(True)
    selected_logits = probe_encoder(*captured["inputs"]).flatten()[indices]

    def objective(bits: Tensor) -> Tensor:
        replacements = list(payloads)
        replacements[round_index] = base.flatten().index_copy(0, indices, bits).reshape_as(base)
        handles = []
        try:
            for layer, payload in zip(model.rounds, replacements):
                def replace_payload(_module, arguments, payload=payload):
                    return (payload.unsqueeze(0).expand(obs.shape[0], *payload.shape),
                            *arguments[1:])
                handles.append(layer.decoder.register_forward_pre_hook(replace_payload))
            result = model(obs, kappa, memory, adjacency, remaining, noise=noise)
            log_probs = F.log_softmax(result["logits"], dim=-1).gather(1, actions[:, None]).squeeze(1)
            return (weights * log_probs).sum()
        finally:
            for registered in handles:
                registered.remove()

    gradients = _conditional_gradients(selected_logits, objective, mc_samples, seed)
    parameters = tuple(probe_encoder.parameters())
    exact_parts = torch.autograd.grad(selected_logits, parameters,
                                      grad_outputs=gradients["exact"], retain_graph=True)
    st_parts = torch.autograd.grad(selected_logits, parameters, grad_outputs=gradients["st"])
    exact, st = (torch.cat([part.flatten() for part in parts]).detach()
                 for parts in (exact_parts, st_parts))
    exact_norm, st_norm = exact.norm().item(), st.norm().item()
    cosine = ((torch.dot(exact, st) / (exact.norm() * st.norm())).clamp(-1, 1).item()
              if exact_norm > 0 and st_norm > 0 else None)
    return {"status": "ok", "scope": scope,
            "objective": "sum_i detached_advantage_i * log pi_i(logged_action_i)",
            "round_index": round_index, "payload_shape": list(base.shape),
            "selected_flat_indices": bit_indices, "selected_logits": selected_logits.detach().tolist(),
            "selected_hard_bits": base.flatten()[indices].tolist(),
            "encoder_parameter_names": [name for name, _ in probe_encoder.named_parameters()],
            "encoder_parameter_count": exact.numel(), "enumerated_patterns": gradients["patterns"],
            "expected_objective": gradients["expected_objective"],
            "score_encoder_gradient_norm": exact_norm, "st_encoder_gradient_norm": st_norm,
            "encoder_gradient_cosine": cosine,
            "relative_gradient_error": (st - exact).norm().item() / exact_norm if exact_norm > 0 else None,
            "score_selected_logit_gradient": gradients["exact"].tolist(),
            "st_selected_logit_gradient": gradients["st"].tolist(),
            "st_selected_logit_standard_error": gradients["st_standard_error"].tolist(),
            "mc_samples": mc_samples, "mc_seed": seed,
            "st_estimator": "Monte Carlo mean with derivative sigmoid'(u + Logistic(0,1))"}
