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
