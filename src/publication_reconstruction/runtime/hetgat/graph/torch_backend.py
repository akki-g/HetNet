"""Small-team, unlimited-communication HetGAT aggregation in plain PyTorch.

Only topology is cached. Learned features, scores and aggregates are new tensors
on every call. Each relation has its own receiver-normalized softmax (paper
Eq. 2/4); combining all agent classes into one softmax would change HetNet.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


RELATIONS = (
    ("P", "p2p", "P"), ("P", "p2a", "A"),
    ("A", "a2p", "P"), ("A", "a2a", "A"),
    ("P", "p2s", "state"), ("A", "a2s", "state"),
)


@dataclass(frozen=True)
class StaticTopology:
    """Receiver-by-sender masks, deliberately outside model state_dict/buffers."""

    num_P: int
    num_A: int
    with_two_state: bool = True

    def __post_init__(self):
        if self.num_P < 0 or self.num_A < 0:
            raise ValueError("Agent counts must be nonnegative")
        counts = {"P": self.num_P, "A": self.num_A,
                  "state": 2 if self.with_two_state else 1}
        masks = {}
        for source, name, target in RELATIONS:
            mask = torch.ones(counts[target], counts[source], dtype=torch.bool)
            if source == target:
                mask.fill_diagonal_(False)
            if target == "state" and self.with_two_state:
                mask.zero_()
                mask[0 if source == "P" else 1, :] = True
            masks[name] = mask
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "nonempty", {
            name: bool(mask.any()) for name, mask in masks.items()})


def aggregate(messages, receiver, source_attention, target_attention, mask,
              negative_slope=0.2):
    """Apply one typed attention relation; inputs have shape (nodes,heads,width).

    Empty neighborhoods yield zero with no null/self candidate. Softmax is over
    permitted senders only. Empty rows are made finite before softmax so their
    backward pass cannot propagate NaNs from an all-minus-infinity row.
    """
    if not mask.numel() or not bool(mask.any()):
        return receiver.new_zeros(receiver.shape)
    source_score = (messages * source_attention).sum(-1)
    target_score = (receiver * target_attention).sum(-1)
    scores = F.leaky_relu(target_score[:, None, :] + source_score[None, :, :],
                          negative_slope=negative_slope)
    allowed = mask[:, :, None]
    has_sender = allowed.any(dim=1, keepdim=True)
    scores = scores.masked_fill(~allowed, -torch.inf)
    scores = torch.where(has_sender, scores, torch.zeros_like(scores))
    weights = scores.softmax(dim=1).masked_fill(~allowed, 0)
    return (weights[:, :, :, None] * messages[None, :, :, :]).sum(dim=1)
