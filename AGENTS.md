# HetNet agent instructions

Current as of 26 October 2026. Keep this file a concise guide to the active
experiment and repository rules. Replace obsolete instructions when the agreed
experiment changes; do not append an implementation diary or old result tables.

## Active experiment and authority

The active checkout is this repository locally and `~/HetNet` on Stokes.
SoftRole, CapCom and CapCon are successive protocol names sharing
`src/softrole/`. Read [README.md](README.md), the matching protocol document and
the current code before changing an experiment. Explicit user instructions take
precedence over older plans and dated reports.

The current direction is **graded sensing on the shared CapCon MLP**, specified
by [Amendment A-2](docs/graded_sensing_amendment_a2.tex). The implemented B1 pilot
compares G-cap (physical capability descriptor) with G-type (constant class
descriptor). Both use PCP 2P1A, seed 0, P sensing levels sampled from `(1, 3)`,
blind A sensing, real messages, three communication rounds, independent heads,
a decomposed critic with beta 1, Adam at 0.001 and a 2M-step target. There are no
sensor failures in this pilot. These are the locked pilot settings, not defaults
for every supported experiment.

Both pilot runs reached their budgets. The
[current analysis](analysis/graded_sensing_2026-10-26/report.md) identifies
training instability and action saturation; it does not establish held-out
transfer or architecture superiority. See the
[pilot runbook](src/softrole/GRADED_PILOT.md) for source checks and launch wiring.
B2–B4, class-specific weights and the full graded research programme remain
separate work. CCCA and its Gate 1/Gate 2 programme are retired. Do not revive
older sensor-failure, binary-bank or optimizer studies as the current experiment
without a user request.

## Scientific and implementation boundaries

- Preserve original HetNet reproduction entrypoints and default numerical
  behavior. New experimental options must be opt-in and keep existing binary
  configurations, scenario identities and manifests compatible.
- Graded sensing is supported in PCP with `corrected-observation-v1`. Physical
  sensing/capture capabilities drive observation masking, action legality and
  failure. Position encoding is unchanged; sensory channels are outside, target
  and total agent **counts**, with independent observation copies.
- G-cap receives physical capability; the explicitly declared G-type control
  receives a constant class descriptor. The descriptor enters both actor and
  critic. Do not describe this comparison as an actor-only intervention or add
  hidden class, identity or visibility information to the capability arm.
- Keep model, learner, reward, random streams and protocol parameters fixed
  unless they are the declared subject of the change. Use the matching protocol,
  not hyperparameters copied from an older study. CPU float64 and one Torch
  thread per collector are the current execution contract.
- Separate engineering smoke checks, exploratory training diagnostics and frozen
  research evaluation. Declare comparisons and checkpoint-selection rules before
  inspecting their outcomes. Keep training/development/final-test data roles
  separate; never tune on held-out panels. Equal seeds do not guarantee identical
  trajectories once policies differ.
- Keep claims proportional to evidence. Single-seed pilot outcomes, finite
  gradients and passing tests do not establish generalization or superiority.
  Preserve declared approximations and uncertainty in scientific reporting.
- Run appropriate behavioral, mathematical, provenance and launcher checks.
  Use the pinned `.venv`/`uv.lock` environment. Do not submit cluster jobs or full
  sweeps merely to validate an engineering change; launch them only when the
  user's task authorizes that work. Inspect active Stokes jobs before changing
  the checkout, and keep their source/environment fixed while they run.

## One directory for every Stokes run

Every new submission must have a meaningful title and one fresh directory under
`runs/` containing **all** of its artifacts. Use the submission helper
`scripts/submit_stokes.py` and the experiment's runbook. Do not put scheduler logs
in root-level `logs_sr/`, `logs_1/`, `slurm-*.out` or separate copied run trees.

Use a sortable UTC timestamp plus a short description of the test, for example
`runs/20261026T190000Z_pcp-graded-sensing-capability-vs-type_ab12cd/`. The identifier must be
unique; a date or job number alone does not explain the experiment. A related
preparation job and training array share this directory, with distinct arm/seed
subdirectories. Unrelated tests get separate directories. Preserve the same
relative directory name when copying results from Stokes to the local checkout.

The helper creates `README.md`, `run_info.json` and per-submission request/response
records in `submissions/`. Complete the run description and protocol records so they
identify its title, research question, changed variable, controls, arms,
seeds, budget, protocol, exact commands, source revision/hash and Slurm job IDs.
Record whether it is engineering, exploratory or confirmatory work. Record
completion/interruption and the location of its conclusions when known; never
call an interrupted run complete just because it has a checkpoint.

Keep preparation manifests, source/configuration snapshots, stdout/stderr,
metrics, episode/update ledgers, checkpoints and their sidecars, evaluation
panels/results and run-specific figures under this same directory. Create log
directories **before** `sbatch`, because Slurm opens output files before the job
script executes. Point every launcher output and environment override into the
run directory, including preparation and dependent jobs. Avoid duplicate source
or data copies unless isolation or reproducibility requires them.

Never overwrite an existing run, mutate locked scientific manifests, or delete
partial evidence to bypass a fresh-directory check. Continuations must follow
the protocol's supported recovery route, identify their parent checkpoint and
use a fresh segment. Keep submission/status records separate from immutable
source/configuration records.

## Repository layout and retention

- `src/`: active experimental Python implementations; module names remain
  `softrole` and `publication_reconstruction`. Original reproduction uses
  `python src/main.py`. `src/publication_reconstruction/runtime/` is a deliberate
  independently versioned copy checked against `ORIGINS.json`.
- `scripts/`, `slurm/`, `tests/`, `test_config/`: launchers, checks and fixed assets.
- `runs/`: current run directories, including their logs and generated evidence.
- `old_logs/`: selected historical references; start with its
  [index](old_logs/README.md). Retain the latest usable run per model/configuration,
  selected by recorded generation/progress rather than performance. Keep required
  replay/test/recovery anchors as labelled exceptions.
- `analysis/`, `docs/`, `research/`: latest directly relevant analyses,
  specifications, thesis documents and references, with required editable sources
  and build/data dependencies. Keep a clear canonical version and working links.
- `output/`: small inventories, migration/cleanup receipts and other maintenance
  records. It is not an alternative home for training logs or checkpoints.
- `.archive/`: protected historical/uncommitted worktrees, not an active launch
  route. Do not alter another worktree's work while cleaning this checkout.

Clean temporary files, redundant copies and generated intermediates after a
change. Do not create parallel `runs 2`, `runs 3`, copied-log or dated-checkout
folders. Before pruning completed evidence, check active jobs, reports, tests,
recovery parents and scientific replication requirements. Do not collapse a
multi-seed study into one seed while it is still needed for inference. Latest-only
retention applies to the historical reference collection, not an unfinished
research dataset. Record intentional removals and any resulting replay limits.

Archived source and manifest bytes are immutable. Some retained artifacts are
read-only hardlinks; make an independent copy before editing evidence. Historical
embedded paths may be obsolete after the
[local cleanup](output/local_prune_2026-10-26/cleanup_report.txt); use its relocation
records rather than rewriting old provenance. New runs require fresh source-locked
preparation after code changes.

Keep documentation and launch examples aligned with the implemented layout.
Put detailed run results in that run's records and relevant analysis; put change
and test details in the task/commit/PR record. Update this file only when current
scope, interfaces or durable rules change.
