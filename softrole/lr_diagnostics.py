"""Read-only, opt-in update measurements for the nominal PCP learning-rate pilot.

D2 replays every complete episode in the logged batch at the declared cadence.
Observations, actions and Binary noise come from the actual pre-update trajectory.
Recomputed recurrence and teacher-forced final features are reported separately.
"""
import json
from pathlib import Path
import sys

import torch

from softrole.env import make_env
from softrole.rollout import run_episode
from softrole.scenarios import Scenario


def policy_statistics(old_logits, new_logits):
    """Sum KL(old || new) and entropies over agent-steps, respecting action masks."""
    feasible = torch.isfinite(old_logits)
    if not torch.equal(feasible, torch.isfinite(new_logits)):
        raise ValueError("Replay changed the feasible action set")
    old_log = old_logits.log_softmax(-1).masked_fill(~feasible, 0)
    new_log = new_logits.log_softmax(-1).masked_fill(~feasible, 0)
    old_p = old_logits.softmax(-1)
    new_p = new_logits.softmax(-1)
    return {"kl": float((old_p * (old_log - new_log)).sum()),
            "entropy_before": float(-(old_p * old_log).sum()),
            "entropy_after": float(-(new_p * new_log).sum()),
            "agent_steps": old_logits.shape[0]}


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(item) for item in value)
    return value


class UpdateDiagnostics:
    def __init__(self, config, output, every):
        self.config, self.output, self.every = config, Path(output), every
        from softrole.train import write_json
        write_json(self.output / "diagnostics_protocol.json", {
            "version": 1, "policy_replay_every_updates": every,
            "policy_replay_also_first_update": True,
            "sample": "all complete episodes in each diagnostic update; agent-step weighted",
            "geometry": "applied parameter displacement overall and by top-level module, every update",
            "recomputed": "fixed logged observations/actions/noise; recompute recurrence and communication",
            "teacher_forced": "fixed logged final communication features; apply updated action head only",
            "limits": ["D2 measured at a sparse update cadence", "no actor/value gradient decomposition or D4/D5",
                       "no held-out evaluation", "no automatic collapse or learning-rate selection"],
        })

    @torch.no_grad()
    def before(self, model, batches, update, environment_source=None):
        self.parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.replays = []
        if update != 1 and update % self.every:
            return
        for batch in batches:
            for record in batch["episodes"]:
                scenario = Scenario(**{key: record[key] for key in Scenario.__dataclass_fields__})
                steps = []

                def capture(module, args, kwargs, out):
                    steps.append((clone(args), clone(kwargs), clone(out["logits"]),
                                  clone(out["memory"]["u_bar"])))

                handle = model.register_forward_hook(capture, with_kwargs=True)
                try:
                    episode = run_episode(model, make_env(self.config, environment_source),
                                          self.config, scenario, training=False)
                finally:
                    handle.remove()
                if episode.metrics != record:
                    raise RuntimeError("Pre-update diagnostic replay differs from the collector episode")
                self.replays.append({"collector": batch["collector"], "scenario": scenario,
                                     "steps": steps})

    @torch.no_grad()
    def after(self, model, update, total_steps, gradient_norm):
        squared = {}
        for name, parameter in model.named_parameters():
            group = name.split(".")[0]
            squared[group] = squared.get(group, 0.) + float(
                (parameter - self.parameters[name]).square().sum())
        row = {"update": update, "total_steps": total_steps,
               "gradient_norm_before_clip": gradient_norm,
               "clip_active": gradient_norm > self.config.max_grad_norm,
               "parameter_update_l2": sum(squared.values()) ** .5,
               "parameter_group_update_l2": {key: value ** .5 for key, value in squared.items()}}
        if self.replays:
            sums = {name: dict(kl=0., entropy_before=0., entropy_after=0., agent_steps=0)
                    for name in ("recomputed", "teacher_forced")}
            for replay in self.replays:
                scenario = replay["scenario"]
                memory = model.initial_memory(scenario.num_p + scenario.num_a)
                for args, kwargs, old_logits, features in replay["steps"]:
                    obs, kappa, old_memory, adjacency = args
                    memory["a_prev"] = old_memory["a_prev"]
                    out = model(obs, kappa, memory, adjacency, **kwargs)
                    memory = out["memory"]
                    teacher_logits = model.output(features).masked_fill(~torch.isfinite(old_logits), -torch.inf)
                    for name, logits in (("recomputed", out["logits"]), ("teacher_forced", teacher_logits)):
                        for key, value in policy_statistics(old_logits, logits).items():
                            sums[name][key] += value
            row["policy_replay"] = {
                "episodes": len(self.replays),
                "identities": [{"collector": r["collector"], "scenario_id": r["scenario"].scenario_id,
                                "env_seed": r["scenario"].env_seed, "action_seed": r["scenario"].action_seed,
                                "message_seed": r["scenario"].message_seed} for r in self.replays],
                **{name: {key: value / stats["agent_steps"] if key != "agent_steps" else value
                          for key, value in stats.items()} for name, stats in sums.items()},
            }
        from softrole.train import append_json
        append_json(self.output / "diagnostics.jsonl", row)
        self.parameters, self.replays = {}, []


def summarize_run(output):
    """Summarize recorded pilot diagnostics; missing S1 windows remain absent."""
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    diagnostics = [json.loads(line) for line in (output / "diagnostics.jsonl").open()]
    epochs = [json.loads(line) for line in (output / "metrics.jsonl").open()]
    windows = {window: dict(episodes=0, steps=0, successes=0, team_return=0.)
               for window in ((0, 2_000_000), (4_000_000, 8_000_000), (8_000_000, 12_000_000))}
    # Episode-weight complete updates whose endpoint falls in each window; record
    # actual endpoints rather than fractionally assigning straddling updates.
    update_steps = {}
    for line in (output / "updates.jsonl").open():
        row = json.loads(line)
        update_steps[row["update"]] = row["total_steps"]
    from softrole.ledger import iter_episodes
    for episode in iter_episodes(output):
        for (lo, hi), records in windows.items():
            if lo < update_steps[episode["update"]] <= hi:
                records["episodes"] += 1
                records["steps"] += episode["steps"]
                records["successes"] += episode["success"]
                records["team_return"] += episode["team_return"]
    policy = [row["policy_replay"] for row in diagnostics if "policy_replay" in row]
    summary = {
        "config": config, "finished": json.loads((output / "finished.json").read_text()),
        "protocol": json.loads((output / "diagnostics_protocol.json").read_text()),
        "recorded_updates": len(diagnostics),
        "clip_fraction": sum(row["clip_active"] for row in diagnostics) / len(diagnostics),
        "mean_parameter_update_l2": sum(row["parameter_update_l2"] for row in diagnostics) / len(diagnostics),
        "max_parameter_update_l2": max(row["parameter_update_l2"] for row in diagnostics),
        "policy_change": {mode: {
            "measured_updates": len(policy),
            "mean_kl": sum(r[mode]["kl"] for r in policy) / len(policy) if policy else None,
            "max_kl": max(r[mode]["kl"] for r in policy) if policy else None,
            "max_entropy_drop": max(r[mode]["entropy_before"] - r[mode]["entropy_after"] for r in policy) if policy else None,
        } for mode in ("recomputed", "teacher_forced")},
        "epoch_regressions": {
            "largest_steps_increase": max([0.] + [b["steps_taken"] - a["steps_taken"]
                                                 for a, b in zip(epochs, epochs[1:])]),
            "largest_success_drop": max([0.] + [a["success_rate"] - b["success_rate"]
                                               for a, b in zip(epochs, epochs[1:])]),
            "worst_epoch_steps": max(row["steps_taken"] for row in epochs),
            "worst_epoch_success": min(row["success_rate"] for row in epochs),
        },
        "policy_replay": [{"update": r["update"], "total_steps": r["total_steps"], **r["policy_replay"]}
                          for r in diagnostics if "policy_replay" in r],
        "epoch_performance": [{key: row[key] for key in
                               ("epoch", "total_steps", "episodes", "steps_taken", "success_rate", "team_return")}
                              for row in epochs],
        "windows": [{"start_exclusive": lo, "end_inclusive": hi, "episodes": records["episodes"],
                     "steps_taken": records["steps"] / records["episodes"] if records["episodes"] else None,
                     "success_rate": records["successes"] / records["episodes"] if records["episodes"] else None,
                     "team_return": records["team_return"] / records["episodes"] if records["episodes"] else None,
                     "fully_reached": diagnostics[-1]["total_steps"] >= hi}
                    for (lo, hi), records in windows.items()],
        "interpretation": "Descriptive pilot. Inspect epoch performance and sampled KL together; no collapse threshold or rate selection.",
    }
    from softrole.train import write_json
    write_json(output / "diagnostics_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = summarize_run(Path(sys.argv[1]))
    print(json.dumps({key: result[key] for key in
                     ("finished", "recorded_updates", "clip_fraction", "mean_parameter_update_l2", "windows")}, indent=2))
