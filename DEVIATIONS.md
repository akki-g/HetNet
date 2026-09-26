# Deviations from the released HetNet

Baseline: `upstream-bff9f7f` = `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`. Preserve `main`; changes are on `frozen-eval`. New experiment infrastructure belongs in `hetnet_ext/`, `tests/`, `configs/` and `slurm/`. Each upstream-behavior deviation has its own commit and row here. Commit hashes are backfilled after the commit exists; the unique subject resolves a row in the meantime.

| ID | Authorized scope | Status / commit | Scientific effect |
|---|---|---|---|
| A | Python 3.12 inspect aliases; Gym 0.26 checker disabling; CPU CUDA-memory guards; one Torch thread/collector; pinned environment | Implemented; commit subject `compat(A): pin Python 3.12 stack and preserve CPU execution` | Runtime compatibility; no architecture/learner change |
| B | Seed Python/NumPy/Torch before construction; retain worker offsets | Planned | Makes initial parameters and trajectories reproducible |
| C | JSONL metrics; correct external sample accounting; every-50/final checkpoints; provenance | Planned | Measurement and artifact cadence; no forward/loss change |
| D | Post-stack victim-only sensor-blinding hook, off by default | Planned for Phase B | Declared evaluation observation intervention; intact output preserved |
| E | Reset from hashed distinct-cell evaluation banks | Planned for Phase B | Controlled, paired initial states |
| F | Empty four agent-to-agent graph relations, off by default | Planned for Phase B | Declared message-removal evaluation intervention |

The following are **not authorized repairs**: changing layers/heads, reward/termination, the original observation feature parser, blanket repair of intact observation slicing, new critic/loss/optimizer, singleton-P handling, or silently changing the recipe to fit a budget. The Torch-2.2 cached-gradient risk is documented in `RESEARCH_NOTES.md`; execute a diagnostic and seek a decision before altering clearing/aggregation behavior if it fails.

The actual upstream `LICENSE` is GPLv3 while README/task prose says MIT. Preserve the file; no license replacement is performed.

## A: dependency verification

The requested core pins import successfully on macOS ARM64 with Python 3.12.13. Exact lock adds pandas 2.2.3, PyYAML 6.0.2 and Pydantic 2.10.6 because the DGL 2.1 GraphBolt import path requires them without declaring them in the installed dependency metadata. Visdom's legacy build imports `pkg_resources`; constrain isolated builds to setuptools 75.8.0. Initial setup attempts exposed these missing requirements and were not training runs. The final core pins remain Torch 2.2.1, DGL 2.1.0, torchdata 0.7.1, NumPy 1.26.4 and Gym 0.26.2. Linux uses the default PyPI Torch distribution; no unverified CPU-index variant is selected.
