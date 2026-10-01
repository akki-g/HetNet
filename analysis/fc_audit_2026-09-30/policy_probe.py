"""Bounded, frozen FC diagnostic panel; never changes simulator or checkpoints.

Run from the repository root with .venv/bin/python. The common epoch 200 is
available for all six archived FC runs; this cannot diagnose the later collapse.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from softrole.config import Config
from softrole.env import EnvironmentAdapter
from softrole.evaluate import model_signature
from softrole.model import SoftRoleNet
from softrole.rollout import run_episode
from softrole.scenarios import make_scenarios


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ObservedAdapter(EnvironmentAdapter):
    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.initial_fire = self.raw.fire_loc.tolist()
        self.initial_positions = self.positions.tolist()
        self.records = []
        return result

    def step(self, actions, kappa):
        pre_pos = self.positions.tolist()
        pre_fires = self.raw.fire_loc.tolist()
        pre_front = self.raw.ign_points_all.tolist()
        result = super().step(actions, kappa)
        self.records.append({
            "step": len(self.records), "positions_before": pre_pos,
            "fire_cells_before": pre_fires, "front_before": pre_front,
            "actions": actions.tolist(), "fire_cells_after": self.raw.fire_loc.tolist(),
            "front_after": self.raw.ign_points_all.tolist(),
            "false_drop": self.raw.false_water_drop.tolist(),
            "extinguished": self.raw.fire_extinguished.tolist(),
            "rewards": result[1].tolist(), "success": result[3]["success"],
        })
        return result


def diagnostics(adapter, outputs):
    rows = adapter.records
    dumps = sum(r["actions"][-1] == 5 for r in rows)
    zero_steps = [r["step"] for r in rows if any(f == [0., 0., 0.] for f in r["front_after"])]
    source_steps = [r["step"] for r in rows if r["extinguished"][0] == 2]
    post_source = rows[source_steps[0]+1:] if source_steps else []
    return {
        "initial_fire": adapter.initial_fire,
        "initial_positions": adapter.initial_positions,
        "zero_placeholder_exposed": bool(zero_steps),
        "zero_placeholder_first_step": min(zero_steps) if zero_steps else None,
        "source_capture_count": len(source_steps),
        "source_capture_step": source_steps[0] if source_steps else None,
        "source_capture_at_origin": any(r["extinguished"][0] == 2 and r["positions_before"][-1] == [0, 0] for r in rows),
        "ordinary_fire_capture_count": sum(r["extinguished"][0] == 1 for r in rows),
        "false_drop_count": sum(r["false_drop"][0] for r in rows),
        "dump_count": dumps,
        "post_source_steps": len(post_source),
        "post_source_dump_count": sum(r["actions"][-1] == 5 for r in post_source),
        "post_source_origin_steps": sum(r["positions_before"][-1] == [0, 0] for r in post_source),
        "final_fire_count": len(rows[-1]["fire_cells_after"]),
        "max_fire_count": max(len(r["fire_cells_after"]) for r in rows),
        "mean_action_entropy_per_agent": np.mean([r["entropy"] for r in outputs], axis=0).tolist(),
        "mean_value_prediction": float(np.mean([r["value"] for r in outputs])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=260930)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    scenarios = make_scenarios(args.seed, args.episodes, [(2, 1)])
    paths = ["WildFire_Simulate_Original.py", "envs/ic3net_envs/fire_commander_env.py",
             "softrole/env.py", "softrole/model.py", "softrole/rollout.py",
             "softrole/scenarios.py", "softrole/config.py", "softrole/learning.py",
             str(Path(__file__).relative_to(ROOT))]
    source = {p: sha(ROOT / p) for p in paths}
    manifest = {"panel": [asdict(s) for s in scenarios], "source_sha256": source,
                "checkpoint_rule": "epoch0200.pt for every FC shared/banked seed 0,1,2; common archived epoch selected before outcomes",
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                "torch_threads": torch.get_num_threads(), "policy_results": []}
    (args.output / "panel.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for variant in ("shared", "banked"):
        for seed in range(3):
            checkpoint = ROOT / f"stokes_runs/runs/softrole_primary/fc_{variant}/seed{seed}/checkpoints/epoch0200.pt"
            before_bytes = sha(checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cfg = Config(**saved["config"])
            model = SoftRoleNet(**saved["model_config"]).double()
            model.load_state_dict(saved["model_state"], strict=True)
            model.eval()
            before = model_signature(model)
            adapter = ObservedAdapter(cfg)
            outputs = []

            def hook(_model, _inputs, out):
                p = torch.softmax(out["logits"], dim=-1)
                entropy = -(p * p.clamp_min(1e-300).log()).sum(-1)
                outputs.append({"entropy": entropy.detach().tolist(), "value": float(out["value"].detach())})

            handle = model.register_forward_hook(hook)
            rows = []
            for i, scenario in enumerate(scenarios):
                outputs.clear()
                episode = run_episode(model, adapter, cfg, scenario, training=False)
                row = {"metrics": episode.metrics, "diagnostics": diagnostics(adapter, outputs)}
                rows.append(row)
                # Retain full state/action evidence on this small diagnostic panel.
                with (args.output / f"{variant}_seed{seed}_traces.jsonl").open("a") as f:
                    f.write(json.dumps({"scenario_id": scenario.scenario_id, "steps": adapter.records}) + "\n")
                if i == 0:
                    plain = run_episode(model, EnvironmentAdapter(cfg), cfg, scenario, training=False)
                    assert plain.metrics == episode.metrics, "Instrumentation changed rollout metrics"
            handle.remove()
            assert before == model_signature(model)
            assert before_bytes == sha(checkpoint)
            assert source == {p: sha(ROOT / p) for p in paths}
            summary = {"variant": variant, "seed": seed, "checkpoint": str(checkpoint.relative_to(ROOT)),
                       "checkpoint_sha256": before_bytes, "model_signature": before,
                       "training_source_sha256": saved["source_sha256"],
                       "total_steps": saved["total_steps"], "epoch": saved["epoch"], "updates": saved["updates"],
                       "episodes": len(rows), "successes": sum(r["metrics"]["success"] for r in rows),
                       "mean_steps": float(np.mean([r["metrics"]["steps"] for r in rows])),
                       "mean_return": float(np.mean([r["metrics"]["team_return"] for r in rows])),
                       "zero_placeholder_episodes": sum(r["diagnostics"]["zero_placeholder_exposed"] for r in rows),
                       "source_captured_episodes": sum(r["diagnostics"]["source_capture_count"] > 0 for r in rows),
                       "source_capture_at_origin_episodes": sum(r["diagnostics"]["source_capture_at_origin"] for r in rows),
                       "failed_after_source_capture": sum(not r["metrics"]["success"] and r["diagnostics"]["source_capture_count"] > 0 for r in rows),
                       "mean_action_entropy_per_agent": np.mean([r["diagnostics"]["mean_action_entropy_per_agent"] for r in rows], axis=0).tolist(),
                       "instrumentation_first_episode_matches_plain": True, "weights_and_checkpoint_unchanged": True}
            (args.output / f"{variant}_seed{seed}.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n")
            manifest["policy_results"].append(summary)
            print(json.dumps(summary), flush=True)
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
