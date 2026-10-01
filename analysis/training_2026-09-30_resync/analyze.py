"""Refresh descriptive training plots from resynced stdout, without running policies.

uv run --no-project --with matplotlib==3.10.7 --with numpy==1.26.4 \
  --python .venv/bin/python python analysis/training_2026-09-30_resync/analyze.py

Reuses the previous stdout parser/audit. New SoftRole sample comparisons pool
episodes within each run and weight training seeds equally. Reproduction has
rounded means and invalid cumulative counters, so cross-method comparisons
use equal-epoch windows, explicitly a different estimand.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / "analysis/training_2026-09-30"
spec = importlib.util.spec_from_file_location("previous_analysis", OLD / "analyze.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.OUT = OUT
TASKS, MODELS, COLORS, STYLES = previous.TASKS, previous.MODELS, previous.COLORS, previous.STYLES
METRICS = previous.METRICS
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return data.decode()


def average(rows, key, weighted):
    return float(np.average([r[key] for r in rows], weights=[r["episodes"] if weighted else 1 for r in rows]))


def window(rows, weighted):
    assert len(rows) == 50
    result = {"first_epoch": rows[0]["epoch"], "last_epoch": rows[-1]["epoch"],
              "weighting": "episodes" if weighted else "equal epochs",
              **{key: average(rows, key, weighted) for key in METRICS}}
    if weighted:
        result.update(episodes=sum(r["episodes"] for r in rows),
                      steps_start=rows[0]["total_steps"] - rows[0]["steps"],
                      steps_end=rows[-1]["total_steps"])
        for key in ("value_loss_per_episode", "gate_entropy", "alpha_null"):
            result[key] = average(rows, key, True)
    return result


def smooth(rows, key, weighted):
    return [average(rows[max(0, i-49):i+1], key, weighted) for i in range(len(rows))]


old_hashes = json.loads(read(OLD / "provenance.json"))["inputs"]
archive_hashes = json.loads(read(OLD / "full_runs/structured_input_hashes.json"))
old_summary = json.loads(read(OLD / "full_runs/summary.json"))
read(OLD / "analyze.py")
runs, extensions = [], []
for path in sorted([*ROOT.glob("logs_1/*.out"), *ROOT.glob("logs_sr/*.out")]):
    data = read(path)
    old = old_hashes[str(path.relative_to(ROOT))]
    assert hashlib.sha256(data.encode()[:old["bytes"]]).hexdigest() == old["sha256"], path
    sr = path.parent.name == "logs_sr"
    # Both parsers reject malformed/incomplete epoch records rather than guess.
    run = previous.parse_softrole(path) if sr else previous.parse_reproduction(path)
    previous.audit(run)
    task, model, seed = run["task"], run["model"], run["seed"]
    suite = "softrole_primary" if sr else "reproduction-fast"
    label = task + "_" + (model.split()[-1] if sr else "real")
    directory = ROOT / "stokes_runs/runs" / suite / label / f"seed{seed}"
    metrics = directory / "metrics.jsonl"
    archived = [json.loads(line) for line in read(metrics).splitlines()]
    assert inputs[str(metrics.relative_to(ROOT))] == archive_hashes[str(metrics.relative_to(ROOT))]
    config_path = directory / ("config.json" if sr else "resolved_args.json")
    config = json.loads(read(config_path))
    assert inputs[str(config_path.relative_to(ROOT))] == archive_hashes[str(config_path.relative_to(ROOT))]
    assert len(run["rows"]) >= len(archived)
    for raw, old_row in zip(run["rows"], archived):
        if sr:
            assert {k: v for k, v in raw.items() if k != "source_line"} == old_row
        else:
            for key in ("success_rate", "steps_taken"):
                assert abs(raw[key] - old_row[key]) <= .005000001
            assert all(abs(a-b) <= .005000001 for a, b in zip(raw["reward_per_agent"], old_row["reward_per_agent"]))
    if sr:
        assert config["model"] == model.split()[-1] and config["seed"] == seed and config["task"] == task
        assert config["failure_prob"] == 0 and not config["compositions"]
        run["identity_basis"] = "All archived epoch fields match stdout exactly; appended tail lacks a fresh source/checkpoint archive"
        run["config"] = config
    else:
        assert all(run["config"][k] == config[k] for k in run["config"] if k in config)
    run["archived_epochs"] = len(archived)
    run["archived_steps"] = archived[-1]["total_steps"]
    runs.append(run)
    extensions.append({"source": run["source"], "task": task, "model": model, "seed": seed,
                       "archived_epochs": len(archived), "new_epochs": len(run["rows"]),
                       "additional_epochs": len(run["rows"])-len(archived),
                       "old_stdout_byte_prefix_unchanged": True, "archive_metric_prefix_matches": True})
assert len(runs) == 27 and len({(r["task"], r["model"], r["seed"]) for r in runs}) == 27
runs.sort(key=lambda r: (TASKS.index(r["task"]), MODELS.index(r["model"]), r["seed"]))
stderr = []
for path in sorted([*ROOT.glob("logs_1/*.err"), *ROOT.glob("logs_sr/*.err")]):
    text = read(path)
    stderr.append({"path": str(path.relative_to(ROOT)), "unchanged_from_previous":
                   inputs[str(path.relative_to(ROOT))] == old_hashes[str(path.relative_to(ROOT))],
                   "has_traceback": "Traceback (most recent call last)" in text})

common_epochs = {t: min(len(r["rows"]) for r in runs if r["task"] == t) for t in TASKS}
budgets = {t: min(r["rows"][-1]["total_steps"] for r in runs if r["task"] == t and r["suite"] == "softrole") // 500000 * 500000 for t in TASKS}
summary = {"common_epoch_ends": common_epochs, "softrole_sample_budgets": budgets, "runs": [], "groups": []}
for run in runs:
    rows = run["rows"]; sr = run["suite"] == "softrole"
    end = common_epochs[run["task"]]
    record = {k: run[k] for k in ("source", "task", "model", "seed", "archived_epochs", "archived_steps", "identity_basis")}
    record.update(epochs=len(rows), configured_epochs=run["config"].get("epochs", run["config"].get("num_epochs")),
                  common_epoch=window(rows[end-50:end], False), latest=window(rows[-50:], sr),
                  previous=window(rows[-100:-50], sr), archive_latest=window(rows[run["archived_epochs"]-50:run["archived_epochs"]], sr))
    if sr:
        record.update(total_steps=rows[-1]["total_steps"], total_episodes=rows[-1]["total_episodes"],
                      event_exposed_episodes=sum(r["event_exposed_episodes"] for r in rows),
                      common_sample=window([r for r in rows if r["total_steps"] <= budgets[run["task"]]][-50:], True))
        # Same estimator at old and new budgets; never subtract unlike windows.
        record["prior_common_sample"] = window([r for r in rows if r["total_steps"] <= old_summary["budgets"][run["task"]]][-50:], True)
        reference = next(r for r in old_summary["runs"] if r["primary"] and r["task"] == run["task"] and r["model"] == run["model"] and r["seed"] == run["seed"])
        assert all(abs(record["prior_common_sample"][k]-reference["common"][k]) < 1e-10 for k in METRICS)
    summary["runs"].append(record)
for task in TASKS:
    for model in MODELS:
        chosen = [r for r in summary["runs"] if r["task"] == task and r["model"] == model]
        windows = ("common_epoch", "latest", "archive_latest") + (() if model == MODELS[0] else ("common_sample", "prior_common_sample"))
        summary["groups"].append({"task": task, "model": model, **{
            name: {key: {"mean": float(np.mean([r[name][key] for r in chosen])),
                         "min": min(r[name][key] for r in chosen), "max": max(r[name][key] for r in chosen)} for key in METRICS} for name in windows}})

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
pages = []
labels = ("Training success (%)", "Mean team return", "Mean episode length")
for suite in ("reproduction", "softrole"):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.8), sharex="col")
    for col, task in enumerate(TASKS):
        for run in [r for r in runs if r["suite"] == suite and r["task"] == task]:
            rows = run["rows"]; sr = suite == "softrole"
            x = [r["total_steps"]/1e6 if sr else r["epoch"] for r in rows]
            color = COLORS[run["model"]] if sr else previous.SEED_COLORS[run["seed"]]
            for i, key in enumerate(METRICS):
                scale = 100 if key == "success_rate" else 1
                y = np.array(smooth(rows, key, sr))*scale
                axes[i, col].plot(x, [r[key]*scale for r in rows], color=color, alpha=.10, lw=.5)
                axes[i, col].plot(x, y, color=color, ls=STYLES[run["seed"]] if sr else "-", lw=1.5)
                k = run["archived_epochs"]-1
                axes[i, col].scatter(x[k], y[k], color=color, s=25, zorder=5)
        axes[0, col].set_title(task.upper()); axes[0, col].set_ylim(-2, 102)
        axes[-1, col].set_xlabel("Recorded environment steps (millions)" if suite == "softrole" else "Completed epoch")
        for i in range(3): axes[i, col].grid(alpha=.18)
    for i, label in enumerate(labels): axes[i, 0].set_ylabel(label)
    handles = ([Line2D([], [], color=COLORS[m], ls=STYLES[s], label=f"{m.split()[-1]} seed {s}") for m in MODELS[1:] for s in range(3)] if suite == "softrole" else
               [Line2D([], [], color=previous.SEED_COLORS[s], label=f"Seed {s}") for s in range(3)])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(("SoftRole" if suite == "softrole" else "HetNet Real reproduction") + " — resynced training logs", fontsize=16)
    fig.text(.5, .935, "Lines: trailing 50 epochs, " + ("episode-weighted." if suite == "softrole" else "equal-epoch means of rounded stdout.") + " Dots: previous structured-archive endpoints.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .92)); pages.append((suite+"_training", fig))

fig, axes = plt.subplots(3, 3, figsize=(11.5, 9))
for col, task in enumerate(TASKS):
    for j, model in enumerate(MODELS[1:]):
        chosen = [r for r in summary["runs"] if r["task"] == task and r["model"] == model]
        for i, key in enumerate(METRICS):
            values = np.array([r["common_sample"][key] for r in chosen])*(100 if key == "success_rate" else 1)
            axes[i, col].scatter(j+np.array([-.07, 0, .07]), values, color=COLORS[model], s=45)
            axes[i, col].plot([j, j], [min(values), max(values)], color=COLORS[model], lw=1.4)
            axes[i, col].scatter(j, np.mean(values), marker="D", color="black", s=35, zorder=5)
    axes[0, col].set_title(f"{task.upper()} / {budgets[task]/1e6:g}M steps")
    axes[0, col].set_ylim(-2, 102)
    for i in range(3):
        axes[i, col].set_xticks([0, 1], ["Shared", "Banked"]); axes[i, col].set_xlim(-.4, 1.4); axes[i, col].grid(axis="y", alpha=.2)
for i, label in enumerate(labels): axes[i, 0].set_ylabel(label)
fig.suptitle("SoftRole at common sample budgets", fontsize=16)
fig.text(.5, .935, "Last 50 completed epochs at/before budget; episode weights within run, equal seed weights. Ranges are not CIs.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .92)); pages.append(("softrole_common_samples", fig))

fig, axes = plt.subplots(3, 3, figsize=(12.5, 9), sharex="col")
for col, task in enumerate(TASKS):
    for run in [r for r in runs if r["suite"] == "softrole" and r["task"] == task]:
        rows = run["rows"]; x = [r["total_steps"]/1e6 for r in rows]
        for i, key in enumerate(("value_loss_per_episode", "gate_entropy", "alpha_null")):
            if key == "gate_entropy" and run["model"] == MODELS[1]: continue
            axes[i, col].plot(x, smooth(rows, key, True), color=COLORS[run["model"]], ls=STYLES[run["seed"]], lw=1.3)
    axes[0, col].set_title(task.upper()); axes[0, col].set_yscale("log")
    axes[1, col].axhline(np.log(4), color="gray", ls=":", lw=1); axes[1, col].set_ylim(0, 1.45)
    axes[2, col].set_ylim(0, 1); axes[2, col].set_xlabel("Recorded environment steps (millions)")
    for i in range(3): axes[i, col].grid(alpha=.2)
for i, label in enumerate(("Value MSE (log scale)", "Banked gate entropy (nats)", "Mean null attention")): axes[i, 0].set_ylabel(label)
fig.suptitle("SoftRole diagnostics — descriptive, not causal", fontsize=16)
fig.text(.5, .935, "Episode-weighted 50-epoch windows. Entropy and attention do not establish roles or communication benefit.", ha="center")
fig.legend(handles=[Line2D([], [], color=COLORS[m], ls=STYLES[s], label=f"{m.split()[-1]} seed {s}") for m in MODELS[1:] for s in range(3)], loc="lower center", ncol=6, frameon=False)
fig.tight_layout(rect=(0, .055, 1, .92)); pages.append(("softrole_diagnostics", fig))
with PdfPages(OUT / "training_plots.pdf") as pdf:
    for name, fig in pages:
        fig.savefig(OUT / (name+".png"), dpi=170); fig.savefig(OUT / (name+".pdf")); pdf.savefig(fig); plt.close(fig)

previous.export_csv("epoch_metrics.csv", [{"task": r["task"], "model": r["model"], "seed": r["seed"], **row} for r in runs for row in r["rows"]])
previous.export_csv("run_summary.csv", [{k: v for k, v in r.items() if not isinstance(v, dict)} | {f"latest_{k}": v for k, v in r["latest"].items()} for r in summary["runs"]])
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
for name, entry in inputs.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == entry["sha256"], "Input changed during analysis: "+name
provenance = {"inputs": inputs, "extensions": extensions, "stderr": stderr,
              "completed_epochs": sum(len(r["rows"]) for r in runs), "archived_completed_epochs": 11375,
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "matplotlib": matplotlib.__version__, "numpy": np.__version__,
              "scope": "Resynced stdout with verified archived prefixes; no new checkpoint/episode-ledger/source audit; no policy execution"}
(OUT / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
print(json.dumps({"completed_epochs": provenance["completed_epochs"], "additional_epochs": sum(r["additional_epochs"] for r in extensions), "common_epoch_ends": common_epochs, "sample_budgets": budgets, "groups": summary["groups"]}, indent=2))
