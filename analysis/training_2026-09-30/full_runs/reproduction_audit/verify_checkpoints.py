"""Read trusted user-owned reproduction checkpoints and verify tensor evidence.

No model is instantiated; no policy evaluation, training, or input mutation occurs.
Legacy files contain the project's LogField objects, so ordinary torch.load is
needed instead of tensor-only loading.
"""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from hetnet_ext.signatures import _entries

records = []
for path in sorted((ROOT / "stokes_runs/runs/reproduction-fast").rglob("*.pt")):
    data = torch.load(path, map_location="cpu")
    sidecar = json.loads(Path(str(path) + ".signature.json").read_text())
    assert _entries(data["policy_net"].items()) == sidecar["parameters"]
    assert all(torch.isfinite(value).all() for value in data["policy_net"].values())
    optimizer = data["trainer"]
    assert all(torch.isfinite(value).all() for state in optimizer["state"].values()
               for value in state.values() if isinstance(value, torch.Tensor))
    steps = [state["step"].item() if hasattr(state["step"], "item") else state["step"]
             for state in optimizer["state"].values()]
    groups = [{key: value for key, value in group.items() if key != "params"}
              for group in optimizer["param_groups"]]
    records.append({"path": str(path.relative_to(ROOT)), "tensors_match_sidecar": True,
                    "model_and_optimizer_tensors_finite": True,
                    "optimizer_step_min": min(steps), "optimizer_step_max": max(steps),
                    "optimizer_groups": groups, "seed": data["seed"]})
(Path(__file__).resolve().parent / "checkpoint_verification.json").write_text(
    json.dumps(records, indent=2) + "\n")
print(f"Verified {len(records)} reproduction checkpoints.")
