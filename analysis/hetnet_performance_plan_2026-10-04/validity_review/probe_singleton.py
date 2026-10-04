"""Diagnose the frozen learner's singleton advantage; no environment or update.

Runs eight synthetic one-step forward/backward probes against the hash-verified
pre-review source. Prints strict JSON. It never writes or saves model state.
"""
from pathlib import Path
import hashlib
import json
import sys
import warnings

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "baseline_manifest.json").read_text())
for name, expected in manifest["files"].items():
    if hashlib.sha256((HERE / "baseline" / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Frozen baseline changed: {name}")
runtime = HERE / "baseline/publication_reconstruction/runtime"
sys.path[:0] = [str(runtime), str(runtime / "envs")]
import numpy as np
import torch
from hetgat.policy import A2CPolicy

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
rows = []
for task, vision in (("pcp", 2), ("fc", 1)):
    for specification in ("public-code-v1", "supplement-v1"):
        for binary in (False, True):
            torch.manual_seed(941)
            policy = A2CPolicy(
                dict(vision=vision, P=29, A=25, state=4), dict(P=29, A=25, state=4),
                dict(P=16, A=16, state=16), dict(P=5, A=6, state=8), 2, 1,
                num_heads=4, device=torch.device("cpu"), use_real=not binary,
                use_CNN=False, per_class_critic=True, with_two_state=True,
                obs=(2 * vision + 1) ** 2, tensor_obs=False, action_vision=-1,
                model_spec=specification, total_state_action_in_batch=1)
            observation = torch.zeros(1, 3, 29 * (2 * vision + 1) ** 2)
            for agent in range(3):
                observation[0, agent, agent] = 1
            policy.batch_select_action_universal([observation, policy.model.init_hidden(1)], 0)
            policy.append_log_probs_properly(np.array([[0, 1, 2]]))
            policy.batch_rewards[0].append(np.array([-.05, -.05, -.05]))
            with warnings.catch_warnings(record=True) as observed_warnings:
                warnings.simplefilter("always")
                loss = policy.batch_finish_per_class(1, 1, num_P=2, num_A=1)
            gradients = [p.grad for p in policy.model.parameters() if p.grad is not None]
            row = dict(task=task, model_spec=specification, binary=binary,
                loss_finite={key: bool(np.isfinite(value)) for key, value in loss.items()},
                gradients_present=len(gradients),
                nonfinite_gradients=sum(not bool(torch.isfinite(g).all()) for g in gradients),
                warnings=[str(w.message) for w in observed_warnings])
            if row["loss_finite"]["total"] or not row["nonfinite_gradients"]:
                raise ValueError("Expected frozen singleton diagnosis was not reproduced")
            rows.append(row)
print(json.dumps({"scope": "synthetic one-step forward/backward; no environment or optimizer step",
                  "cases": rows}, indent=2, allow_nan=False))
