"""Read the copied training logs, audit counts, and export descriptive plots/data.

Run without changing the project's training environment:
uv run --no-project --with matplotlib==3.10.7 --with numpy==1.26.4 \
  --python .venv/bin/python python analysis/training_2026-09-30/analyze.py

All comparisons use equal-epoch means, then equal-run means. SoftRole-only
episode-pooled summaries are exported separately, never imputed for HetNet.
"""
import ast
import csv
import hashlib
import json
import platform
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
WINDOW = 50
TASKS = ("pp", "pcp", "fc")
MODELS = ("HetNet Real", "SoftRole shared", "SoftRole banked")
COLORS = dict(zip(MODELS, ("#667085", "#0072B2", "#D55E00")))
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73")
STYLES = ("-", "--", ":")
METRICS = ("success_rate", "team_return", "steps_taken")
LABELS = ("Training success fraction", "Mean team return", "Mean episode length (steps)")
HEADER = re.compile(r"^Epoch (\d+)\s+Reward \[([^]]+)\]\s+Time ([\d.]+)s, "
                    r"Episodes (\d+), Total Steps (\d+),")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def namespace(text):
    line = next(line for line in text.splitlines() if line.startswith("Namespace("))
    node = ast.parse(line, mode="eval").body
    assert isinstance(node, ast.Call) and node.func.id == "Namespace"
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def parse_reproduction(path):
    text = path.read_text()
    config = namespace(text)
    p, a = config["nfriendly_P"], config["nfriendly_A"]
    task = "fc" if config["env_name"] == "fire_commander" else ("pp" if a == 0 else "pcp")
    assert not config["use_binary"]
    rows, row = [], None
    for number, line in enumerate(text.splitlines(), 1):
        match = HEADER.match(line)
        if match:
            assert row is None, "Incomplete previous epoch"
            epoch, reward, seconds, printed_episodes, printed_steps = match.groups()
            rewards = [float(x) for x in reward.split()]
            assert len(rewards) == p + a
            row = dict(epoch=int(epoch), team_return=sum(rewards),
                       reward_per_agent=rewards, wall_time_seconds=float(seconds),
                       reported_overcounted_episodes=int(printed_episodes),
                       reported_overcounted_steps=int(printed_steps),
                       return_nocap=float(np.mean(rewards[:p])),
                       return_cap=float(np.mean(rewards[p:])) if a else None,
                       source_line=number)
        elif row is not None and line.startswith("Success: "):
            row["success_rate"] = float(line.split(": ", 1)[1])
        elif row is not None and line.startswith("Steps-taken: "):
            row["steps_taken"] = float(line.split(": ", 1)[1])
            assert "success_rate" in row
            rows.append(row)
            row = None
    assert row is None
    return dict(suite="reproduction", model=MODELS[0], task=task, seed=config["seed"],
                identity_basis="Observed stdout Namespace", config=config,
                run=path.stem, source=str(path.relative_to(ROOT)), rows=rows)


def parse_softrole(path):
    index = int(path.stem.rsplit("_", 1)[1])
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["source_line"] = number
        rows.append(row)
    task, model, seed = TASKS[index // 6], MODELS[1 + (index // 3) % 2], index % 3
    return dict(suite="softrole", model=model, task=task, seed=seed,
                identity_basis="Inferred from current slurm/softrole.sbatch array mapping; config absent",
                run=path.stem, source=str(path.relative_to(ROOT)), rows=rows)


def audit(run):
    rows = run["rows"]
    assert [row["epoch"] for row in rows] == list(range(1, len(rows) + 1))
    for row in rows:
        assert all(np.isfinite(value) for value in row.values()
                   if isinstance(value, (int, float)))
        assert 0 <= row["success_rate"] <= 1 and row["steps_taken"] > 0
    if run["suite"] == "softrole":
        for count in ("steps", "episodes"):
            assert np.array_equal(np.cumsum([r[count] for r in rows]),
                                  [r["total_" + count] for r in rows])
        for row in rows:
            assert abs(row["steps_taken"] * row["episodes"] - row["steps"]) < 1e-7
            assert not row["partial_epoch"] and row["updates"] == 10 * row["epoch"]
            assert row["num_agents"] == 3
            assert 0 <= row["alpha_null"] <= 1
            assert 0 <= row["gate_entropy"] <= np.log(4) + 1e-12
            p, a = (3, 0) if run["task"] == "pp" else (2, 1)
            team = p * row["return_nocap"] + a * (row["return_cap"] or 0)
            assert abs(team - row["team_return"]) < 1e-8


def mean_window(run, start, end, weighted=False):
    rows = [r for r in run["rows"] if start <= r["epoch"] <= end]
    assert len(rows) == end - start + 1
    if weighted:
        assert run["suite"] == "softrole"
    weights = np.array([r["episodes"] if weighted else 1 for r in rows])
    names = (*METRICS, "wall_time_seconds", "return_nocap", "return_cap",
             "value_loss_per_episode", "policy_loss_per_episode", "gate_entropy", "alpha_null")
    values = {key: float(np.average([r[key] for r in rows], weights=weights))
              for key in names if all(r.get(key) is not None for r in rows)}
    return dict(epoch_start=start, epoch_end=end, weighting="episodes" if weighted else "equal_epoch",
                **values)


def rolling(rows, key):
    values = np.asarray([r[key] for r in rows], dtype=float)
    sums = np.r_[0, values.cumsum()]
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - WINDOW)
    return (sums[ends] - sums[starts]) / (ends - starts)


def export_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def style(fig, title, footer):
    fig.suptitle(title, fontsize=17, weight="bold", y=.985)
    fig.text(.5, .015, footer, ha="center", fontsize=9, color="#475467")
    for ax in fig.axes:
        ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.08, right=.98, top=.875, bottom=.08, hspace=.32, wspace=.25)


def save(fig, name, pdf):
    fig.savefig(OUT / f"{name}.png", dpi=170, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def curves(runs, suite, pdf):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), sharex="col")
    selected = [run for run in runs if run["suite"] == suite]
    for col, task in enumerate(TASKS):
        axes[0, col].set_title(task.upper(), fontsize=13, weight="bold")
        for run in [r for r in selected if r["task"] == task]:
            rows = run["rows"]
            x = np.array([r["epoch"] if suite == "reproduction" else r["total_steps"] / 1e6 for r in rows])
            color = SEED_COLORS[run["seed"]] if suite == "reproduction" else COLORS[run["model"]]
            for index, metric in enumerate(METRICS):
                ax = axes[index, col]
                ax.plot(x, [r[metric] for r in rows], color=color, alpha=.13, lw=.55)
                ax.plot(x, rolling(rows, metric), color=color, lw=1.7,
                        ls="-" if suite == "reproduction" else STYLES[run["seed"]])
        axes[0, col].set_ylim(-.025, 1.025)
        axes[-1, col].set_xlabel("Completed epoch" if suite == "reproduction" else "Recorded environment steps (millions)")
    for row, label in enumerate(LABELS):
        axes[row, 0].set_ylabel(label)
    if suite == "reproduction":
        handles = [Line2D([], [], color=c, label=f"Seed {i}") for i, c in enumerate(SEED_COLORS)]
        title = "HetNet Real reproduction training"
        footer = "Raw epochs faint; trailing up-to-50-epoch means bold. Rounded stdout; cumulative sample counters are invalid."
    else:
        handles = [Line2D([], [], color=COLORS[m], ls=STYLES[s], label=f"{m.replace('SoftRole ', '')} seed {s}")
                   for m in MODELS[1:] for s in range(3)]
        title = "SoftRole training by recorded environment steps"
        footer = "Raw epochs faint; trailing up-to-50-epoch means bold. Identities inferred from array mapping; training only, no confidence bands."
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .948), ncol=len(handles), frameon=False, fontsize=9)
    style(fig, title, footer)
    save(fig, f"{suite}_training", pdf)


def common_plot(summary, pdf):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex="col")
    for col, task in enumerate(TASKS):
        groups = [g for g in summary["common_window_groups"] if g["task"] == task]
        axes[0, col].set_title(f"{task.upper()} · epochs {groups[0]['start']}–{groups[0]['end']}", weight="bold")
        for x, group in enumerate(groups):
            color = COLORS[group["model"]]
            for y, metric in enumerate(METRICS):
                values = np.array([r[metric] for r in group["per_seed"]])
                ax = axes[y, col]
                ax.plot([x, x], [values.min(), values.max()], color=color, lw=2)
                ax.scatter(x + np.array([-.075, 0, .075]), values, s=24, color=color, alpha=.8)
                ax.scatter([x], [values.mean()], marker="D", s=60, color=color, edgecolor="white", zorder=4)
                ax.set_xlim(-.5, 2.5)
        axes[0, col].set_ylim(0, 1.04)
        axes[-1, col].set_xticks(range(3), ["HetNet\nReal", "SoftRole\nshared", "SoftRole\nbanked"])
    for row, label in enumerate(LABELS):
        axes[row, 0].set_ylabel(label)
    style(fig, "Common training windows for descriptive comparison",
          "Points: three seed means. Diamond: equal-seed mean. Line: observed seed range, NOT a confidence interval.\n"
          "All summaries weight epochs equally. Pipelines differ in observations, model and learner; this is not a causal architecture test.")
    fig.subplots_adjust(top=.91, bottom=.12)
    save(fig, "common_window_comparison", pdf)


def diagnostics(runs, pdf):
    keys = ("value_loss_per_episode", "policy_loss_per_episode", "gate_entropy", "alpha_null")
    labels = ("Value MSE per episode\n(log scale)", "Actor surrogate per episode\n(symmetric log scale)",
              "Banked gate entropy (nats)", "Mean null attention mass")
    fig, axes = plt.subplots(4, 3, figsize=(14, 12.5), sharex="col")
    for col, task in enumerate(TASKS):
        axes[0, col].set_title(task.upper(), weight="bold")
        for run in [r for r in runs if r["suite"] == "softrole" and r["task"] == task]:
            x = [r["total_steps"] / 1e6 for r in run["rows"]]
            for index, key in enumerate(keys):
                if key == "gate_entropy" and run["model"] == "SoftRole shared":
                    continue
                axes[index, col].plot(x, rolling(run["rows"], key), color=COLORS[run["model"]],
                                      ls=STYLES[run["seed"]], lw=1.5)
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("symlog", linthresh=.1)
        axes[2, col].axhline(np.log(4), color="#98A2B3", ls="--", lw=1)
        axes[2, col].set_ylim(0, 1.44)
        axes[3, col].set_ylim(0, 1)
        axes[-1, col].set_xlabel("Recorded environment steps (millions)")
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label)
    handles = [Line2D([], [], color=COLORS[m], ls=STYLES[s], label=f"{m.replace('SoftRole ', '')} seed {s}")
               for m in MODELS[1:] for s in range(3)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .948), ncol=6, frameon=False, fontsize=9)
    style(fig, "SoftRole optimization and communication diagnostics",
          "Trailing up-to-50-epoch means. Entropy reference: ln(4); shared K=1 entropy is identically zero and omitted.\n"
          "Loss, entropy and attention are diagnostics, not proofs of convergence, role specialization or communication benefit.")
    save(fig, "softrole_diagnostics", pdf)


def main():
    paths = sorted([*ROOT.glob("logs_1/*"), *ROOT.glob("logs_sr/*")])
    manifest = {str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in paths if p.is_file()}
    runs = [parse_reproduction(p) for p in sorted(ROOT.glob("logs_1/*.out"))]
    runs += [parse_softrole(p) for p in sorted(ROOT.glob("logs_sr/*.out"))]
    runs.sort(key=lambda r: (TASKS.index(r["task"]), MODELS.index(r["model"]), r["seed"]))
    assert len(runs) == 27 and len({(r["model"], r["task"], r["seed"]) for r in runs}) == 27
    for run in runs:
        audit(run)
    common = {t: min(len(r["rows"]) for r in runs if r["task"] == t) for t in TASKS}
    summary = dict(window=WINDOW, method="Equal epoch means within each run, equal run means across three seeds; no inferential confidence interval",
                   common_windows=common, runs=[], common_window_groups=[])
    for run in runs:
        last = len(run["rows"])
        record = {k: v for k, v in run.items() if k != "rows"}
        record.update(epochs=last, first50=mean_window(run, 1, WINDOW),
                      latest50=mean_window(run, last - WINDOW + 1, last),
                      previous50=mean_window(run, last - 2 * WINDOW + 1, last - WINDOW),
                      common50=mean_window(run, common[run["task"]] - WINDOW + 1, common[run["task"]]),
                      completed_epoch_hours=sum(r["wall_time_seconds"] for r in run["rows"]) / 3600)
        if run["suite"] == "softrole":
            record.update(total_steps=run["rows"][-1]["total_steps"], total_episodes=run["rows"][-1]["total_episodes"],
                          event_exposed_episodes=sum(r["event_exposed_episodes"] for r in run["rows"]),
                          latest50_episode_weighted=mean_window(run, last - WINDOW + 1, last, weighted=True))
        summary["runs"].append(record)
    for task in TASKS:
        for model in MODELS:
            records = [r for r in summary["runs"] if r["task"] == task and r["model"] == model]
            per_seed = [dict(seed=r["seed"], **r["common50"]) for r in records]
            summary["common_window_groups"].append(dict(task=task, model=model,
                start=common[task] - WINDOW + 1, end=common[task], per_seed=per_seed,
                **{m: {"mean": float(np.mean([r[m] for r in per_seed])),
                       "min": min(r[m] for r in per_seed), "max": max(r[m] for r in per_seed)} for m in METRICS}))
    export_csv("epoch_metrics.csv", [dict(suite=run["suite"], task=run["task"], model=run["model"],
               seed=run["seed"], source=run["source"], **row) for run in runs for row in run["rows"]])
    export_csv("run_summary.csv", [dict(run=r["run"], task=r["task"], model=r["model"], seed=r["seed"],
               epochs=r["epochs"], recorded_total_steps=r.get("total_steps"), **r["latest50"])
               for r in summary["runs"]])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    with PdfPages(OUT / "training_plots.pdf") as pdf:
        curves(runs, "reproduction", pdf)
        curves(runs, "softrole", pdf)
        common_plot(summary, pdf)
        diagnostics(runs, pdf)
    for relative, entry in manifest.items():
        assert sha(ROOT / relative) == entry["sha256"], "Input changed during analysis"
    provenance = dict(inputs=manifest, analysis_sha256=sha(Path(__file__)), window_epochs=WINDOW,
                      python=platform.python_version(), matplotlib=matplotlib.__version__, numpy=np.__version__,
                      source_note="Current local code explains metrics; copied SoftRole logs do not identify executed source",
                      code={name: sha(ROOT / name) for name in ("main.py", "trainer.py", "multi_processing.py",
                            "softrole/train.py", "softrole/rollout.py", "softrole/learning.py", "slurm/softrole.sbatch",
                            "envs/ic3net_envs/predator_capture_env.py", "envs/ic3net_envs/fire_commander_env.py")})
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"runs": len(runs), "epochs": sum(len(r["rows"]) for r in runs),
                      "common_epoch_end": common, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
