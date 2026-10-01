"""Check released feature extraction against the simulator's cell schema.

Run from the repository root. This is a diagnostic of the preserved release,
not a corrected model, and writes only a fresh adjacent JSON result.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from hetgat.uavnet import UAVNetA2CEasy


def main():
    torch.set_num_threads(1)
    rows = []
    for task, cells, num_p, num_a in [("pp", 25, 3, 0), ("pcp", 25, 2, 1), ("fc", 9, 2, 1)]:
        # Each simulator cell is 25 position entries plus four sensory entries.
        schema = SimpleNamespace(num_P=num_p, num_A=num_a,
                                 in_dim={"P": 29, "A": 25, "state": 4},
                                 P_s=25, obs_squares=cells)
        x = torch.arange((num_p + num_a) * cells * 29, dtype=torch.float64).reshape(1, num_p + num_a, -1)
        actual = UAVNetA2CEasy.get_obs_features(schema, x).reshape(num_p, cells, 4)
        expected = x.reshape(num_p + num_a, cells, 29)[:num_p, :, 25:]
        mismatches = int((actual[0] != expected[0]).sum())
        # Perturb only the physical target channel, one relative cell at a time.
        # The independent position branch must not recover discarded targets.
        target_cells = []
        for cell in range(cells):
            obs = torch.zeros_like(x)
            obs[0, 0, cell * 29 + 27] = 1
            sensory = UAVNetA2CEasy.get_obs_features(schema, obs)
            position, _ = UAVNetA2CEasy.remove_excess_action_features_from_all(schema, obs)
            assert not torch.any(position)
            if torch.any(sensory):
                target_cells.append(cell)
        selected = actual[0].long().flatten().tolist()
        rows.append({"task": task, "relative_cells": cells,
                     "sensory_width_per_P": cells * 4,
                     "misplaced_entries_per_P": mismatches,
                     "selected_entries_actually_from_sensory_channels": sum(i % 29 >= 25 for i in selected),
                     "relative_cells_whose_target_channel_reaches_model": target_cells,
                     "position_branch_does_not_recover_target": True,
                     "first_three_actual_index_groups": actual[0, :3].long().tolist(),
                     "first_three_expected_index_groups": expected[0, :3].long().tolist()})
        assert mismatches == (cells - 1) * 4
        assert len(target_cells) < cells
    local = ROOT / "hetgat/uavnet.py"
    upstream = Path(__file__).with_name("upstream_uavnet.py")
    assert local.read_bytes() == upstream.read_bytes()
    result = {"interpretation": "Confirmed active released feature-stride defect; performance causality requires controlled retraining.",
              "upstream_url": "https://raw.githubusercontent.com/CORE-Robotics-Lab/HetNet/main/hetgat/uavnet.py",
              "upstream_equals_local": True,
              "uavnet_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "domains": rows}
    with Path(__file__).with_name("feature_findings.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
