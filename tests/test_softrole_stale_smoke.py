"""Two local gate updates are an engineering check, never research evidence."""
from dataclasses import replace
import json

import torch

from softrole.stale_capability import gate_config
from softrole.train import train


def test_two_gate_updates_have_finite_weights_and_optimizer_state(tmp_path):
    # Only execution budget/collection fields change; architecture, optimizer,
    # sensing, horizon and scientific gate settings are those of coefficient .003.
    config = replace(gate_config(0.003, 0), nprocesses=1, batch_steps=16,
                     epochs=1, updates_per_epoch=2, total_steps=None, save_every=1)
    output = tmp_path / "engineering_gate_smoke"
    checkpoint = train(config, output, episode_log="chunked-gzip", checkpoint_commits=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["updates"] == 2
    assert all(t.dtype == torch.float64 and t.device.type == "cpu" and torch.isfinite(t).all()
               for t in saved["model_state"].values())
    assert saved["optimizer_state"]["state"]

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return True

    assert finite(saved["optimizer_state"])
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert len(updates) == 2
    assert all(row["action_entropy_sum"] > 0 for row in updates)
    assert json.loads((output / "finished.json").read_text())["updates"] == 2
