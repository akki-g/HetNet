# Deviations from the released HetNet

Baseline: `upstream-bff9f7f` = `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`. Preserve `main`; changes are on `frozen-eval`. New experiment infrastructure belongs in `hetnet_ext/`, `tests/`, `configs/` and `slurm/`. Each upstream-behavior deviation has its own commit and row here. Commit hashes are backfilled after the commit exists; the unique subject resolves a row in the meantime.

| ID | Authorized scope | Status / commit | Scientific effect |
|---|---|---|---|
| A | Python 3.12 inspect aliases; Gym 0.26 checker disabling; CPU CUDA-memory guards; one Torch thread/collector; pinned environment | `2fadecf` | Runtime compatibility; no architecture/learner change |
| B | Seed Python/NumPy/Torch before construction; retain worker offsets | `7b334c1` | Makes initial parameters and trajectories reproducible |
| C | JSONL metrics; correct external sample accounting; every-50/final checkpoints; provenance | `50d0c37`; added orchestration in `be2d223`, validation fixes in `2db8a62` | Measurement and artifact cadence; no forward/loss change |
| D | Post-stack victim-only sensor-blinding hook, off by default | Planned for Phase B | Declared evaluation observation intervention; intact output preserved |
| E | Reset from hashed distinct-cell evaluation banks | Planned for Phase B | Controlled, paired initial states |
| F | Empty four agent-to-agent graph relations, off by default | Planned for Phase B | Declared message-removal evaluation intervention |
| G | Explicit in-place gradient clearing at exactly three selected-path sites | `0cfcea5`; user approved 26 September 2026 | Restores cached-gradient storage across updates; preserves clipping, fresh-gradient summation/division, RMSprop and architecture |
| H | Local macOS setup and short endpoint calibration launcher, with hardware/provenance/resource evidence | User selected short pilots then measured-cost review on 26 September 2026; commit subject `local(H): prepare bounded Mac calibration pilots` | Execution platform and orchestration only; same per-epoch recipe, four collectors, CPU, original model and learner; no full-study scope reduction |

The following are **not authorized repairs**: changing layers/heads, reward/termination, the original observation feature parser, blanket repair of intact observation slicing, new critic/loss/optimizer, singleton-P handling, or silently changing the recipe to fit a budget. The Torch-2.2 cached-gradient failure was diagnosed and explicitly approved for the narrow G repair below; no broader learner change is authorized.

The actual upstream `LICENSE` is GPLv3 while README/task prose says MIT. Preserve the file; no license replacement is performed.

## A: dependency verification

The requested core pins import successfully on macOS ARM64 with Python 3.12.13. Exact lock adds pandas 2.2.3, PyYAML 6.0.2 and Pydantic 2.10.6 because the DGL 2.1 GraphBolt import path requires them without declaring them in the installed dependency metadata. Visdom's legacy build imports `pkg_resources`; constrain isolated builds to setuptools 75.8.0. Initial setup attempts exposed these missing requirements and were not training runs. The final core pins remain Torch 2.2.1, DGL 2.1.0, torchdata 0.7.1, NumPy 1.26.4 and Gym 0.26.2. Linux uses the default PyPI Torch distribution; no unverified CPU-index variant is selected.

## G: user-approved gradient-storage repair

The user selected **“Apply explicit in-place clearing (Recommended)”** in response to the concrete three-line patch and preserved Gate A failure evidence. This approval covers `zero_grad(set_to_none=False)` in `MultiProcessWorker.run`, `MultiProcessTrainer.train_batch`, and the selected `A2CPolicy.batch_finish_per_class`. Upstream caches non-null gradient storage once; Torch 2.2.1's default clearing to `None` invalidates those pointers after the first update. Both a deterministic surrogate and actual HetNet loss reproduced the failure. In-place clearing retains storage while clearing values. Original failed evidence is retained under `evidence/gate_a/`; repaired evidence is separately named. These bounded checks establish the gradient contract, not learning or research outcomes.

## C: metrics and signatures

The optional `--metrics_file` records each fresh `train_batch` statistic before upstream merges or normalizes it. Episode means use the actual episode count; loss diagnostics retain upstream's summed-loss / joint-step reporting denominator. Cumulative counts sum each fresh batch once; upstream stdout remains unchanged, including its known cumulative-counter overcount. Epoch wall time is measured separately from checkpoint serialization and signature time, which are logged in `checkpoint_records.jsonl` with checkpoint size.

`resolved_args.json` includes hard-coded model critic/state choices in `resolved_model`; initial, every-epoch and checkpoint signatures cover names, shapes, dtypes, parameter values and buffers without consuming RNG. Checkpoint labels now mean completed epochs: 50, 100, …, 2000, with the final save performed once. Run evidence refuses overwrites. Resume has not been implemented; no claim of resumed-training reproducibility is made.

Recorder tests verify exact fresh-batch accounting, copying against later mutation, hash detection of parameter/buffer changes, absence of RNG/state effects, and refusal to overwrite evidence. Full smoke/determinism/load checks are recorded separately in Gate A.

## H: authorized Mac fallback for initial tests

After reporting a Stokes issue, Akki explicitly selected **short pilots first, then review measured cost**, on a separate Mac available all day. This authorizes preparing local setup and the two 20-epoch endpoint calibrations before deciding longer computation. It does not authorize silently reducing the 21-run final study or treating partially trained policies as its final checkpoints. The Stokes-only launcher and original scientific sources remain unchanged; a separate local path replaces scheduler-specific orchestration with actual host evidence and bounded sequential execution. Fresh Gate A is required for the new source inventory. The proposed 300-epoch learning pilots remain a decision after calibration. See `MAC_RUNBOOK.md`.
