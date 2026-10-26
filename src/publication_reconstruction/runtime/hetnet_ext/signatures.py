"""Value signatures for the unchanged model (feasibility report Eq. 6).

These inspect tensors without drawing random numbers or changing module state.
Names, shapes, dtypes and values all enter the aggregate digest.
"""
from __future__ import annotations

import hashlib
import json


def _entries(tensors):
    import torch

    entries = []
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        # Viewing bytes also supports dtypes NumPy cannot represent directly.
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        entries.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    return entries


def model_signature(model):
    signature = {"schema_version": 1, "parameters": _entries(model.named_parameters()),
                 "buffers": _entries(model.named_buffers())}
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    signature["sha256"] = hashlib.sha256(canonical).hexdigest()
    return signature


def tree_signature(value):
    """Stable value identity for model/optimizer/RNG trees, independent of pickle."""
    import numpy as np
    import torch

    def canonical(item):
        if torch.is_tensor(item):
            return {"tensor": _entries([("", item)])[0]}
        if isinstance(item, np.ndarray):
            return {"ndarray": str(item.dtype), "shape": list(item.shape),
                    "sha256": hashlib.sha256(item.tobytes(order="C")).hexdigest()}
        if isinstance(item, dict):
            return {"dict": [[canonical(k), canonical(v)] for k, v in
                             sorted(item.items(), key=lambda pair: (type(pair[0]).__name__, str(pair[0])))]}
        if isinstance(item, (tuple, list)):
            return {type(item).__name__: [canonical(v) for v in item]}
        if isinstance(item, np.generic):
            return canonical(item.item())
        if item is None or isinstance(item, (bool, int, float, str)):
            return {type(item).__name__: item}
        raise TypeError(f"Unsupported identity value: {type(item).__name__}")

    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
