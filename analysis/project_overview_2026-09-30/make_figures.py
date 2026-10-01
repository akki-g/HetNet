"""Build compact, vector training figures from the audited resynced epoch table.

Run from the repository root:
uv run --no-project --with matplotlib==3.10.7 --with numpy==1.26.4 \
  --python .venv/bin/python python analysis/project_overview_2026-09-30/make_figures.py

No policies execute and no training inputs are modified. The figure manifest
records the exact input bytes, plot methods, software and output hashes.
"""

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "latest_snapshot"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TASKS = ("pp", "pcp", "fc")
MODELS = ("HetNet Real", "SoftRole shared", "SoftRole banked")
COLORS = {MODELS[0]: "#416A91", MODELS[1]: "#087F8C", MODELS[2]: "#C77B27"}
STYLES = ("-", (0, (5, 2)), (0, (1, 1.6)))
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
    }
    return data.decode()


summary = json.loads(read(SOURCE / "summary.json"))
source_provenance = json.loads(read(SOURCE / "provenance.json"))
groups = defaultdict(list)
for row in csv.DictReader(io.StringIO(read(SOURCE / "epoch_metrics.csv"))):
    parsed = {
        key: value if key in ("task", "model", "reward_per_agent", "partial_epoch") or value == ""
        else float(value)
        for key, value in row.items()
    }
    groups[(row["task"], row["model"], int(row["seed"]))].append(parsed)
assert len(groups) == 27
for (task, model, seed), rows in groups.items():
    assert [r["epoch"] for r in rows] == list(range(1, len(rows) + 1))
    assert all(np.isfinite(r[k]) for r in rows for k in
               ("team_return", "success_rate", "steps_taken"))
    if model != MODELS[0]:
        assert all(r["episodes"] > 0 for r in rows)
        assert all(a["total_steps"] < b["total_steps"] for a, b in zip(rows, rows[1:]))
    record = next(r for r in summary["runs"] if
                  (r["task"], r["model"], r["seed"]) == (task, model, seed))
    assert len(rows) == record["epochs"]


def smooth(rows, key, weighted, width=50):
    """Use all available epochs at the left edge, then full trailing windows."""
    weights = np.array([r["episodes"] if weighted else 1 for r in rows])
    values = np.array([r[key] for r in rows])
    # Explicit windows preserve the source report's summation interpretation.
    return np.array([
        np.average(values[max(0, i - width + 1):i + 1],
                   weights=weights[max(0, i - width + 1):i + 1])
        for i in range(len(rows))
    ])


endpoint_checks = 0
for record in summary["runs"]:
    rows = groups[(record["task"], record["model"], record["seed"])]
    weighted = record["model"] != MODELS[0]
    keys = ("team_return", "success_rate", "steps_taken")
    if weighted:
        keys += ("value_loss_per_episode", "gate_entropy", "alpha_null")
    for key in keys:
        assert np.isclose(smooth(rows[-50:], key, weighted)[-1],
                          record["latest"][key], rtol=1e-12, atol=1e-12)
        endpoint_checks += 1


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#7B8790",
    "axes.linewidth": .6,
    "text.color": "#263442",
    "axes.labelcolor": "#263442",
    "xtick.color": "#435360",
    "ytick.color": "#435360",
    "grid.color": "#CCD4DA",
    "grid.alpha": .65,
    "grid.linewidth": .5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

output_names = []


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", metadata={
        "Title": name.replace("_", " "),
        "Author": "HetNet / SoftRole project",
        "Subject": "Descriptive training logs; individual seeds, no confidence intervals",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(OUT / f"{name}.png", dpi=220)
    output_names.extend([f"{name}.pdf", f"{name}.png"])
    plt.close(fig)


def legend_handles(legacy=True, archive=False):
    selected = MODELS if legacy else MODELS[1:]
    result = [Line2D([], [], color=COLORS[model], lw=2,
                     label=model.removeprefix("SoftRole ").capitalize()
                     if model.startswith("SoftRole") else model)
              for model in selected]
    result.extend(Line2D([], [], color="#586775", ls=STYLES[s], lw=1.3,
                         label=f"Seed {s}") for s in range(3))
    if archive:
        result.append(Line2D([], [], color="#586775", ls="", marker="o",
                             markersize=3.5, label="Archive end"))
    return result


def format_axis(ax):
    ax.grid(axis="y")
    ax.xaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
    ax.tick_params(length=3, width=.6, pad=3)
    ax.margins(x=.025)


for task in TASKS:
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 4.0), sharex="col", sharey="row")
    fig.subplots_adjust(left=.105, right=.985, bottom=.19, top=.925,
                        wspace=.17, hspace=.16)
    for col, selected_models in enumerate(((MODELS[0],), MODELS[1:])):
        for model in selected_models:
            for seed in range(3):
                rows = groups[(task, model, seed)]
                weighted = model != MODELS[0]
                x = np.array([r["total_steps"] / 1e6 if weighted else r["epoch"]
                              for r in rows])
                record = next(r for r in summary["runs"] if
                              (r["task"], r["model"], r["seed"]) == (task, model, seed))
                for i, key in enumerate(("team_return", "success_rate", "steps_taken")):
                    y = smooth(rows, key, weighted) * (100 if key == "success_rate" else 1)
                    axes[i, col].plot(x, y, color=COLORS[model], ls=STYLES[seed],
                                      lw=1.15, alpha=.96)
                    k = record["archived_epochs"] - 1
                    axes[i, col].scatter(x[k], y[k], color=COLORS[model], s=12,
                                         zorder=5, edgecolor="white", linewidth=.25)
        axes[0, col].set_title("HetNet Real reproduction" if col == 0 else "SoftRole")
        axes[-1, col].set_xlabel("Completed epoch" if col == 0 else "Environment steps (millions)")
        for ax in axes[:, col]:
            format_axis(ax)
    for row, label in enumerate(("Mean team return", "Success (%)", "Episode length")):
        axes[row, 0].set_ylabel(label)
    axes[1, 0].set_ylim(-2, 102)
    axes[1, 0].set_yticks([0, 50, 100])
    handles = legend_handles(archive=True)
    fig.legend(handles=handles, ncol=7, loc="lower center", bbox_to_anchor=(.52, .003),
               frameon=False, columnspacing=.9, handlelength=2.2, handletextpad=.4)
    save(fig, f"{task}_training")


fig, axes = plt.subplots(3, 3, figsize=(7.1, 4.25), sharex="col")
fig.subplots_adjust(left=.10, right=.985, bottom=.19, top=.925,
                    wspace=.25, hspace=.23)
for col, task in enumerate(TASKS):
    for model in MODELS[1:]:
        for seed in range(3):
            rows = groups[(task, model, seed)]
            x = [r["total_steps"] / 1e6 for r in rows]
            for i, key in enumerate(("value_loss_per_episode", "gate_entropy", "alpha_null")):
                if key == "gate_entropy" and model == MODELS[1]:
                    continue
                axes[i, col].plot(x, smooth(rows, key, True), color=COLORS[model],
                                  ls=STYLES[seed], lw=1.1)
    axes[0, col].set_title(task.upper())
    for ax in axes[:, col]:
        format_axis(ax)
    axes[0, col].set_yscale("log")
    if task == "fc":
        axes[0, col].set_yticks([200, 1000, 5000], ["200", "1,000", "5,000"])
        axes[0, col].yaxis.set_minor_formatter(NullFormatter())
    axes[1, col].axhline(np.log(4), color="#85919A", ls=(0, (3, 2)), lw=.7)
    axes[1, col].set_ylim(0, 1.48)
    axes[1, col].set_yticks([0, .7, 1.4])
    axes[2, col].set_ylim(0, 1.0)
    axes[2, col].set_yticks([0, .5, 1.0])
    axes[2, col].set_xlabel("Steps (millions)")
    if col:
        axes[1, col].tick_params(labelleft=False)
        axes[2, col].tick_params(labelleft=False)
axes[0, 0].set_ylabel("Value MSE\n(log scale)")
axes[1, 0].set_ylabel("Banked gate\nentropy (nats)")
axes[2, 0].set_ylabel("Mean null\nattention")
fig.legend(handles=legend_handles(legacy=False), ncol=5, loc="lower center",
           bbox_to_anchor=(.52, .009), frameon=False, columnspacing=1.5, handlelength=2.5)
save(fig, "diagnostics")


rows = groups[("fc", MODELS[1], 2)]
indices = [i for i, r in enumerate(rows) if r["epoch"] >= 900]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.0))
fig.subplots_adjust(left=.10, right=.985, bottom=.265, top=.89, wspace=.30)
for ax, key, label, scale in zip(axes,
                               ("success_rate", "value_loss_per_episode"),
                               ("Success (%)", "Value MSE (log)"), (100, 1)):
    x = np.array([r["epoch"] for r in rows])
    raw = np.array([r[key] for r in rows]) * scale
    smoothed = smooth(rows, key, True, width=10) * scale
    ax.plot(x[indices], raw[indices], color=COLORS[MODELS[1]], alpha=.28, lw=.65)
    ax.plot(x[indices], smoothed[indices], color=COLORS[MODELS[1]], lw=1.4)
    ax.set_xlabel("Completed epoch")
    ax.set_ylabel(label)
    format_axis(ax)
    if key == "value_loss_per_episode":
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
save(fig, "fc_regression")


for relative, metadata in inputs.items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == metadata["sha256"], \
        f"Input changed while plotting: {relative}"
manifest = {
    "inputs": inputs,
    "script": str(Path(__file__).relative_to(ROOT)),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
    "method": {
        "main_panels": "Every seed; trailing 50 completed epochs; partial windows at the left edge.",
        "hetnet": "Equal weights across rounded stdout epoch means; x = completed epoch.",
        "softrole": "Episode-weighted epoch means; x = recorded environment steps in millions.",
        "uncertainty": "Individual seed trajectories only; no pooled confidence intervals or significance tests.",
        "archive_dots": "Last completed epoch represented in the earlier structured archive, by run.",
        "diagnostics": "SoftRole; episode-weighted trailing 50-epoch means; value MSE log y; gate-entropy reference = ln(4).",
        "fc_regression": "Shared FC seed 2, epoch 900 onward; faint raw epoch means plus episode-weighted trailing 10-epoch means, calculated with earlier epochs before trimming.",
        "comparability": "The two columns have different x measures. These are training curves, not a controlled architecture ranking or frozen evaluation.",
        "scope": "Only recorded metrics were plotted; no training, policy evaluation or input mutation.",
    },
    "rows": sum(map(len, groups.values())),
    "run_count": len(groups),
    "numeric_checks": {
        "plotted_final_window_values_match_existing_audited_summary": endpoint_checks,
        "all_epoch_sequences_contiguous": True,
        "all_softrole_sample_counts_strictly_increasing": True,
        "input_hashes_unchanged_during_plotting": True,
    },
    "outputs": {
        str((OUT / name).relative_to(ROOT)): {
            "bytes": (OUT / name).stat().st_size,
            "sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest(),
        }
        for name in output_names
    },
}
(HERE / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"runs": manifest["run_count"], "epochs": manifest["rows"],
                  "figures": output_names}, indent=2))
