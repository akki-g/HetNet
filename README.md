# HetNet and Capability-Conditioned HetNet

This is the sole active experimental checkout. Use this directory locally and
`~/HetNet` on Stokes. SoftRole, CapCom and CapCon share the implementation in
`src/softrole/`; they are names used across successive experiment protocols.

Start with the [code-to-experiment inventory](output/stokes_run_layout_2026-10-26/code_inventory.txt),
the [retained experiment references](old_logs/README.md),
the [local cleanup record](output/local_prune_2026-10-26/cleanup_report.txt),
and [AGENTS.md](AGENTS.md) for current experiment boundaries and repository rules.

## Current experiment: graded sensing

The current B1 implementation adds graded physical sensing and a descriptor control
to the shared CapCon MLP. The exploratory pilot compares G-cap, which receives
physical capability, with G-type, which receives a constant physical-class
descriptor. The descriptor enters both actor and critic.

Both arms use PCP 2P1A, seed 0, P sensing levels sampled from `(1, 3)`, blind A
sensing, real messages, three rounds, independent heads, Adam at 0.001 and a
2M-step target. There are no sensor failures in this pilot. It is a small
training check; the full B4 programme and held-out evaluation are separate work.

Both pilot runs completed their budgets. The
[26 October analysis](analysis/graded_sensing_2026-10-26/report.md) records their
training histories and the frozen saturation diagnostic. These one-seed results
do not establish generalization or architecture superiority.

Read [the graded pilot runbook](src/softrole/GRADED_PILOT.md) for the locked protocol,
source checks, output locations and Slurm-only preparation/submission route.
It describes fresh submissions using the current layout. [Amendment A-2](docs/graded_sensing_amendment_a2.tex)
provides the broader research specification.

## Code map

| Location | Responsibility |
|---|---|
| `src/softrole/config.py`, `env.py`, `scenarios.py` | Experimental configurations, observations, capabilities and reproducible scenarios. |
| `src/softrole/model.py`, `learning.py`, `rollout.py`, `train.py` | Shared actor/critic models, losses, collection and training. |
| `src/softrole/graded_pilot.py` | Locked G-cap/G-type pilot configuration, manifest and launch checks. |
| `src/softrole/evaluate.py`, `report.py`, `ledger.py` | Frozen evaluation, summaries and committed run evidence. |
| `scripts/`, `slurm/` | Experiment-specific launchers and cluster jobs; use the matching protocol document. |
| `tests/` | Mathematical, behavioral, provenance and launcher checks. |
| `src/*.py`, `src/hetgat/` | Original released HetNet learner and architecture. |
| `src/envs/`, `src/hetnet_ext/` | Released domains, compatibility, recording, seeding and progress helpers. |
| `src/publication_reconstruction/` | Separate public-code, supplement and paper-reconstruction baselines. |
| `test_config/` | Original evaluation initial-condition assets. |

`src/publication_reconstruction/runtime/` is a deliberate, independently versioned
runtime copy. Its files are checked against `ORIGINS.json`. Archived run source
copies are also required for replay and evaluation; they are not unused imports.

All active experimental implementations live under `src/`. Python module names
are unchanged: use `python -m softrole` and `python -m publication_reconstruction`.
The original reproduction entrypoint is now `python src/main.py` (or `python -m main`).
Launchers remain under `scripts/` and `slurm/`; tests, evaluation assets and outputs
remain at the repository root. Run launch commands from this root.

Archived source trees, old manifests and dated reports retain their recorded
layout and hashes. They describe the source used for those experiments. New runs
record `src/` paths and require fresh source-locked preparation manifests.

## Protocol and documentation index

| Study or reference | Read first |
|---|---|
| Current graded sensing pilot | [GRADED_PILOT.md](src/softrole/GRADED_PILOT.md) |
| Shared architecture and successive experimental extensions | [RESEARCH.md](src/softrole/RESEARCH.md) |
| CapCon critic comparison and 40M protocol | [CAPCON_L1_40M.md](src/softrole/CAPCON_L1_40M.md) |
| Additional PCP training seeds | [PCP_ADDITIONAL_SEEDS.md](src/softrole/PCP_ADDITIONAL_SEEDS.md) |
| Post-hoc PCP collapse recovery | [PCP_RECOVERY.md](src/softrole/PCP_RECOVERY.md) |
| Earlier Adam control | [ADAM_PCP.md](src/softrole/ADAM_PCP.md) |
| Publication reconstruction and declared deviations | [README](src/publication_reconstruction/README.md), [FIDELITY.md](src/publication_reconstruction/FIDELITY.md) |
| Original released-code reproduction | [Launcher](scripts/reproduce.sh), [retained runs](old_logs/README.md) |
| Architecture explanation and thesis | [Thesis sources and final documents](docs/CapCon/), [build and scope notes](docs/CapCon/README.txt) |

Sensor-failure, optimizer, message-channel and diagnostic modules remain part of
the shared implementation. Their protocol settings are not the current graded
pilot defaults. The [CCCA reference runs](old_logs/ccca_local_frontends/) record a
retired alternative; they are not a current launch route. Amendment A-2 specifies
which earlier requirements still apply to graded sensing.

## Running and checking

Use the pinned Python 3.12 environment defined by `pyproject.toml`, `uv.lock` and
`requirements.txt`. The existing `.venv` is the local environment. Environment
setup uses `bash scripts/setup_env.sh`, or `uv sync --locked` if uv is installed.
Both install this project and `src/envs` as editable packages. After pulling the
layout change into an existing environment, refresh those two editable installs;
third-party dependency versions are unchanged:

```bash
uv pip install --python .venv/bin/python --no-deps -e . -e src/envs
.venv/bin/python -m pytest -q
```

The Slurm launchers explicitly select `src/` and `src/envs/` for checkout imports.

On Stokes, inspect jobs before changing a checkout used by collectors:

```bash
cd ~/HetNet
squeue -u "$USER"
```

For a fresh graded pilot, follow the exact preparation and dependent-array commands
in [GRADED_PILOT.md](src/softrole/GRADED_PILOT.md). Keep the checkout fixed from
preparation until both jobs finish. The launcher rejects existing outputs and
protocol overrides; preserve completed or interrupted runs. It does not implement
continuation. No cluster jobs were submitted by this repository cleanup.

## One directory per Stokes experiment

Submit all jobs through `scripts/submit_stokes.py`. Give each experiment a clear
title and purpose. The helper creates a unique `runs/<UTC timestamp>_<test title>_<id>/`
directory before Slurm opens its logs. Related preparation, training and evaluation
jobs reuse that directory; model/arm/seed outputs remain separate children.

```bash
run_dir=$(.venv/bin/python scripts/submit_stokes.py --init-only \
  --title 'PCP graded sensing capability versus type' \
  --purpose 'Compare G-cap and G-type at matched sensing levels and a 2M-step budget.')

prep=$(.venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
  slurm/softrole_graded_pilot_prepare.sbatch)
.venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
  --dependency="afterok:${prep%%;*}" --kill-on-invalid-dep=yes \
  slurm/softrole_graded_pilot.sbatch
```

The container holds `README.md`, `run_info.json`, `submissions/`, `logs/`, protocol
manifests and experiment outputs such as `training/<arm>/seedN/`. Each submission
records the exact command, batch-script hash, output paths, scheduler response
and job ID. Training/evaluation records retain their existing source/configuration
snapshots. Add the experimental controls, budget and interpretation to the run's
description and protocol records; update its outcome when the jobs finish.

Use `--dry-run` instead of submission to preview paths and commands without
creating files. Existing prepared containers can be adopted with `--run-dir`,
`--title` and `--purpose`; subsequent stages use `--run-dir` alone. Explicit output
overrides must stay inside the container. Read-only checkpoint/archive inputs may
live elsewhere. The helper leaves trainer output directories fresh and preserves
their overwrite protection. Batch scripts reject execution without prepared run
metadata. Do not submit them directly with `sbatch` or create separate log roots.

Copy the same `runs/<identifier>/` directory back from Stokes, including logs and
source records. New source-locked manifests are required after launcher changes.
Do not replace archived manifests or change the checkout while its jobs run.

## Results and retained evidence

| Location | Contents |
|---|---|
| `old_logs/` | One selected run per model/configuration, matched logs and the required reference checkpoints. Start with its [index](old_logs/README.md). |
| `analysis/` | Latest relevant reports, figures, derived data and their analysis dependencies. |
| `docs/`, `research/` | Retained thesis/specification files, relevant references and supporting evidence. |
| `output/` | Inventories and cleanup/migration records. |
| `.archive/` | Git worktrees with historical or uncommitted work; not the active experimental checkout. |

The local cleanup keeps 19 run references with checkpoints and 10 log-only
references. Each selected run retains its available configuration, source archive,
metrics and raw episode/update ledgers. Signature and completion sidecars stay
beside checkpoints. Paired evaluation panels and eight older checkpoints required
for diagnostic tests, the sensor pilot or recovery are explicit exceptions.
Four latest sensor-development checks from an unmerged worktree are retained
separately as engineering references, one per model configuration.

"Latest" follows recorded run generations and progress where dates are absent;
it is not a selection by performance. Some logs extend beyond their local
checkpoints, and some runs are incomplete. The archive index identifies these
limits. Other replicates and intermediate checkpoints were removed locally, so
the retained reports cannot all be regenerated from this reduced archive.

Existing `runs/`, `stokes_runs/`, `logs_1/` and `logs_sr/` contents were consolidated
into `old_logs/`. New Stokes work uses the unified `runs/<identifier>/` layout above.
That pruning changed local files only; Stokes was not pruned again.

Archived source and manifest bytes are unchanged. Their embedded paths are
historical; the [relocation plan](output/local_prune_2026-10-26/run_plan.json) maps
retained files to their new locations. Use fresh preparation manifests for new
executions. Some retained files share read-only hardlinks: make an independent
copy before modifying evidence.
