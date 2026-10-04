"""Deviation B: seed all CPU RNGs before constructing environments or models."""

import random
import secrets

import numpy as np
import torch


def seed_everything(requested_seed: int) -> int:
    seed = secrets.randbelow(2**31) if requested_seed == -1 else requested_seed
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be -1 or an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
