"""Verify the slide numbers against raw ledgers; export small presentation figures.

From the repository root:
uv run --no-project --with matplotlib==3.10.7 --with numpy==1.26.4 \
  --python .venv/bin/python python analysis/advisor_sync_2026-09-30/prepare.py
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT / "analysis/training_2026-09-30/full_runs"
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
    }
    return data.decode()


def mean(rows, key):
    return sum(r["episodes"] * r[key] for r in rows) / sum(r["episodes"] for r in rows)


summary = json.loads(read(ANALYSIS / "summary.json"))
old_hashes = json.loads(read(ANALYSIS / "structured_input_hashes.json"))
audits = json.loads(read(ANALYSIS / "softrole_audit/findings.json"))
exposure = json.loads(read(ANALYSIS / "softrole_audit/nominal_failure_exposure.json"))
runs, all_updates = [], []
checked_source_files = []
current_source_status = {}
for run in summary["runs"]:
    if not run["primary"]:
        continue
    path = ROOT / run["path"]
    metrics_path = path / "metrics.jsonl"
    rows = [json.loads(line) for line in read(metrics_path).splitlines() if line.strip()]
    assert inputs[str(metrics_path.relative_to(ROOT))] == old_hashes[str(metrics_path.relative_to(ROOT))]
    assert len(rows) == run["epochs"]
    assert [r["epoch"] for r in rows] == list(range(1, len(rows) + 1))
    assert sum(r["steps"] for r in rows) == run["steps"]
    assert sum(r["episodes"] for r in rows) == run["episodes"]
    for row in rows:
        if "team_return" not in row:
            row["team_return"] = sum(row["reward_per_agent"])
    if run["suite"] == "softrole_primary":
        config = json.loads(read(path / "config.json"))
        assert config["failure_prob"] == 0 and config["compositions"] == []
        assert config["held_out"] == [] and config["seed"] == run["seed"]
        assert all(r["event_exposed_episodes"] == 0 for r in rows)
        updates_path = path / "updates.jsonl"
        updates = [json.loads(line) for line in read(updates_path).splitlines() if line.strip()]
        assert inputs[str(updates_path.relative_to(ROOT))] == old_hashes[str(updates_path.relative_to(ROOT))]
        all_updates.extend(updates)
        manifest = json.loads(read(path / "source_manifest.json"))
        # These are the code paths cited in the slides, not a rerun of every archive audit.
        for name in ["softrole/model.py", "softrole/env.py", "softrole/config.py",
                     "softrole/rollout.py", "softrole/learning.py", "softrole/train.py"]:
            digest = hashlib.sha256(read(path / "source" / name).encode()).hexdigest()
            assert digest == manifest["files"][name], name
            local_digest = hashlib.sha256(read(ROOT / name).encode()).hexdigest()
            current_source_status[name] = {"sha256": local_digest,
                                          "matches_training_archive": local_digest == digest}
            if name in ("softrole/model.py", "softrole/config.py", "softrole/learning.py"):
                assert local_digest == digest, "Recheck architecture/loss slides after source changes"
            checked_source_files.append(name)
    selected = [r for r in rows if r["total_steps"] <= summary["budgets"][run["task"]]][-50:]
    assert len(selected) == 50
    for key in ("success_rate", "team_return", "steps_taken"):
        assert abs(mean(selected, key) - run["common"][key]) < 1e-10
        earlier = [r for r in rows if r["total_steps"] <= 4000000][-50:]
        assert abs(mean(earlier, key) - run["common_4m"][key]) < 1e-10
    runs.append({**run, "rows": rows})

assert len(runs) == 27
assert sum(r["epochs"] for r in runs) == 11375
assert sum(r["episodes"] for r in runs) == 10820962
assert sum(r["steps"] for r in runs) == 252821848
assert len(all_updates) == 63214
assert min(r["gradient_norm_before_clip"] for r in all_updates) > .75

# The separate pilot landed after the nominal-run audit. Verify its raw panel,
# rather than presenting the saved summary as an independent experiment.
pilot_root = ROOT / "runs/sensor_failure_validation_20260930"
pilot_hashes = json.loads(read(pilot_root / "artifact_manifest.json"))["files"]


def pilot_read(relative):
    path = pilot_root / relative
    value = json.loads(read(path))
    assert inputs[str(path.relative_to(ROOT))]["sha256"] == pilot_hashes[relative]
    return value


validation = pilot_read("validation.json")
pilot = pilot_read("pilot/pilot.json")
pilot_summary = pilot_read("pilot/summary.json")
panel = pilot_read("pilot/scenarios.json")
assert len(panel) == 20 and pilot["composition"] == [2, 1]
assert pilot["failure_probability"] == 1 and pilot["failure_window"] == [10, 30]
assert validation["tests"]["passed"] == 178
assert pilot["evaluator_source_sha256"] == pilot_summary["evaluator_source_sha256"]
pilot_rows = []
for mode in ("shared", "banked"):
    spec = next(p for p in pilot_summary["policies"] if p["model"] == mode)
    conditions = {}
    for condition in ("failure", "sham"):
        result = pilot_read(f"pilot/{mode}_seed0_{condition}.json")
        assert result["scenarios"] == panel and result["training_seed"] == 0
        assert result["config"]["failure_prob"] == 0
        assert result["checkpoint_progress"] == spec["checkpoint_progress"]
        assert result["checkpoint_sha256"] == spec["checkpoint_sha256"]
        rows = result["per_episode"]
        assert len(rows) == 20
        saved = spec[condition]
        assert sum(r["success"] for r in rows) / 20 == saved["success_rate"]
        assert sum(r["event_exposed"] for r in rows) == saved["actual_event_exposed"]
        assert sum(r["scheduled_event_exposed"] for r in rows) == saved["diagnostic_denominator"]
        for key, count in saved["diagnostic_true_counts"].items():
            assert sum(r[key] is True for r in rows) == count
        conditions[condition] = rows
    for failure, sham in zip(conditions["failure"], conditions["sham"]):
        assert failure["scenario_id"] == sham["scenario_id"]
        event = failure["event_step"]
        assert failure["trace"][:event] == sham["trace"][:event]
        for key in spec["failure"]["diagnostic_true_counts"]:
            assert failure[key] == sham[key]
    assert sum(f["success"] and not s["success"] for f, s in
               zip(conditions["failure"], conditions["sham"])) == spec["failure_only_successes"]
    assert sum(s["success"] and not f["success"] for f, s in
               zip(conditions["failure"], conditions["sham"])) == spec["sham_only_successes"]
    checkpoint = Path(spec["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == spec["checkpoint_sha256"]
    f = spec["failure"]
    exposed = f["diagnostic_denominator"]
    pilot_rows.append(mode.title() + " & " + " & ".join([
        f"{spec['sham']['success_rate'] * 100:.0f}\\%",
        f"{f['success_rate'] * 100:.0f}\\%", f"{exposed}/20",
        f"{f['diagnostic_true_counts']['pre_event_victim_reached']}/{exposed}",
        f"{f['diagnostic_true_counts']['pre_event_victim_target_seen']}/{exposed}",
    ]) + r" \\")
(HERE / "pilot_rows.tex").write_text(
    r"\begin{tabular}{lrrrrr}\toprule & Sham & Failure & Exposed & Reached$^*$ & Seen$^*$\\\midrule"
    + "\n" + "\n".join(pilot_rows) + "\n" + r"\bottomrule\end{tabular}" + "\n")

plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
colors = {"HetNet Real": "#687582", "SoftRole shared": "#1474A8", "SoftRole banked": "#CB7037"}
styles = ("-", "--", ":")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.25), sharey=True, layout="constrained")
for ax, task in zip(axes, ("pp", "pcp", "fc")):
    for model in colors:
        selected = [r for r in runs if r["task"] == task and r["model"] == model]
        ys = [r["common"]["success_rate"] * 100 for r in selected]
        x = list(colors).index(model)
        ax.scatter(x + np.array([-.08, 0, .08]), ys, color=colors[model], s=42, zorder=3)
        ax.scatter(x, np.mean(ys), marker="D", color="black", s=36, zorder=4)
    ax.set_xticks(range(3), ("HetNet\nReal", "Shared", "Banked"))
    ax.set_title(f"{task.upper()} / {summary['budgets'][task]/1e6:g}M steps")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=.18)
axes[0].set_ylabel("Training success (%)")
fig.savefig(HERE / "success_by_budget.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(11.3, 3.25), layout="constrained")
for r in runs:
    if r["task"] != "pcp" or r["suite"] != "softrole_primary":
        continue
    rows = r["rows"]
    y = [mean(rows[max(0, i-49):i+1], "success_rate") * 100 for i in range(len(rows))]
    ax.plot([z["total_steps"]/1e6 for z in rows], y, color=colors[r["model"]],
            linestyle=styles[r["seed"]], label=f"{r['model'].replace('SoftRole ', '')}, seed {r['seed']}", lw=1.8)
ax.axvline(5, color="#687582", lw=1, alpha=.7)
ax.set(xlabel="Recorded environment steps (millions)", ylabel="Training success (%)", ylim=(0, 103), xlim=(0, None))
ax.grid(alpha=.18)
ax.legend(ncol=3, loc="lower right", fontsize=10, frameon=False)
fig.savefig(HERE / "pcp_progress.pdf", bbox_inches="tight")
plt.close(fig)

groups = {(r["task"], r["model"]): r for r in summary["groups"]}
for (task, model), group in groups.items():
    chosen = [r for r in runs if r["task"] == task and r["model"] == model]
    assert len(chosen) == 3
    for window in ("common", "common_4m"):
        for metric in ("success_rate", "team_return", "steps_taken"):
            assert abs(np.mean([r[window][metric] for r in chosen]) - group[window][metric]["mean"]) < 1e-10
durations = {r["name"]: [0., 0] for r in exposure}
specs = {r["name"]: r for r in exposure}
for line in read(ANALYSIS / "softrole_audit/episode_epoch_sufficient_stats.jsonl").splitlines():
    row = json.loads(line)
    spec = specs[row["name"]]
    if spec["first_epoch"] <= row["epoch"] <= spec["last_epoch"]:
        for length, count in row["steps_hist"].items():
            durations[row["name"]][0] += count * max(0, min(21, int(length) - 10)) / 21
            durations[row["name"]][1] += count
for spec in exposure:
    numerator, denominator = durations[spec["name"]]
    assert denominator == spec["episodes"]
    assert abs(numerator / denominator - spec["nominal_uniform_10_30_event_exposure"]) < 1e-10
tex = []
for task in ("pp", "pcp", "fc"):
    values = [groups[(task, model)]["common"]["success_rate"]["mean"] * 100 for model in colors]
    tex.append(task.upper() + " & " + " & ".join(f"{v:.2f}\\%" for v in values) + r" \\")
(HERE / "result_rows.tex").write_text(
    r"\begin{tabular}{lrrr}\toprule Training success & HetNet Real & Shared & Banked\\\midrule" + "\n"
    + "\n".join(tex) + "\n" + r"\bottomrule\end{tabular}" + "\n")
tex = []
for mode in ("shared", "banked"):
    values = [next(r for r in exposure if r["name"] == f"pcp_{mode}/seed{s}")["nominal_uniform_10_30_event_exposure"] * 100 for s in range(3)]
    tex.append(mode.title() + " & " + " & ".join(f"{v:.2f}\\%" for v in values) + r" \\")
(HERE / "exposure_rows.tex").write_text(
    r"\begin{tabular}{lrrr}\toprule & Seed 0 & Seed 1 & Seed 2\\\midrule" + "\n"
    + "\n".join(tex) + "\n" + r"\bottomrule\end{tabular}" + "\n")

for relative in ["softrole/RESEARCH.md", "softrole/ARCHITECTURE_COMPARISON.md",
                 "analysis/training_2026-09-30/report.md", "SENSOR_FAILURE_HANDOFF.md",
                 "Capability-Conditioned HetNet Deck Review and Revised Outline.pdf"]:
    path = ROOT / relative
    data = path.read_bytes()
    inputs[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

result = {
    "snapshot": "2026-09-30", "primary_runs": len(runs), "completed_epochs": 11375,
    "completed_epoch_steps": 252821848, "completed_epoch_episodes": 10820962,
    "available_softrole_updates": len(all_updates), "all_updates_above_clip_threshold": True,
    "source_files_matched_to_all_18_manifests": sorted(set(checked_source_files)),
    "current_source_status": current_source_status,
    "common_budget_groups": summary["groups"],
    "pilot": pilot, "pilot_summary": pilot_summary,
    "latest_recorded_engineering_validation": validation,
    "scope": "Fresh verification of metrics/configs/update counts and cited source paths. Existing streaming episode/checkpoint audits are cited, not rerun.",
    "inputs": inputs,
}
(HERE / "verified_facts.json").write_text(json.dumps(result, indent=2) + "\n")
print("Verified 27 primary runs, 11,375 epochs, 63,214 update norms and six archived source files against 18 manifests.")
