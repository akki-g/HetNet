# HetNet reconstruction: training and frozen evaluation

## Paper-v1 reconciliation and PyTorch backend (4 October 2026)

The opt-in `--reconstruction-spec paper-v1` reconciles explicit paper/code
differences before comparing DGL with `--message-backend torch-v1`. Read the
[evidence and mathematical contract](FIDELITY.md) before selecting it. It changes
class preprocessing, Binary head channels, the learner and FC semantics, so it
requires fresh runs. Existing defaults/archives remain the legacy reconstruction
described below. Binary uses the selected 64 bits **per head**, 256 total per round;
this convention is not certified as the publication's 64-bit configuration.

```bash
# Dry runs create no training outputs and submit no jobs.
.venv/bin/python -m publication_reconstruction train \
  --reconstruction-spec paper-v1 --task pcp --variant binary \
  --message-backend torch-v1 --seed 0 --dry-run
.venv/bin/python -m publication_reconstruction benchmark \
  --output runs/paper_backend_benchmark --dry-run
.venv/bin/python -m publication_reconstruction paper-study-plan \
  --run-root runs/paper_study --output runs/paper_study_preparation --dry-run
```

The bounded benchmark uses 24 sequential 20-update runs in one allocation, four
collectors, floor 500/production horizons and three alternating pairs per workload.
`--diagnostics` adds separate paired three-update phase-timing runs. It records
hardware/affinity/source identities, startup, whole-process and update throughput,
CPU intervals and memory units. A 20-update run cannot pass the 100-update preflight
validator. Prior Mac speed estimates concern a different baseline.

Prepared Stokes commands, to run **after** local correctness/source checks:

```bash
mkdir -p logs_1 logs_sr
sbatch slurm/publication_backend_benchmark.sbatch
# After reviewing same-spec paired correctness/performance and resources:
PUBLICATION_MESSAGE_BACKEND=torch-v1 sbatch slurm/publication_paper_preflight.sbatch
# After all four official 100-update preflights pass:
.venv/bin/python -m publication_reconstruction paper-study-plan \
  --message-backend torch-v1 --run-root runs/paper_study \
  --output runs/paper_study_preparation
PUBLICATION_MESSAGE_BACKEND=torch-v1 PUBLICATION_RUN_ROOT=runs/paper_study/hetnet \
  sbatch slurm/publication_paper_train.sbatch
SOFTROLE_PAPER_RUN_ROOT=runs/paper_study/softrole_fc \
  sbatch slurm/softrole_paper_fc.sbatch
```

These files do not submit other jobs. The benchmark's 24-hour ceiling is a bound,
not a runtime estimate. Select one backend for the fresh HetNet wave; retain DGL
if Torch is slower, inconclusive or fails a gate. Cluster execution is not implied
by the presence of a script. `AGENTS.md` records the checks actually performed.

`paper-study-plan` prepares 12 HetNet protocols and 6 fresh SoftRole FC protocols
(shared/banked, seeds 0–2), plus their common 500-scenario nominal FC panel. Both use
the same archived paper-FC simulator. Old SoftRole FC results must stay separate.
The existing `resume` command inherits backend and source; there is no checkpoint
migration from old models to paper-v1. Profiling is off by default.

For the later matched FC evaluation, select each model/seed's first saved
complete-update checkpoint at or above 28M steps, in update order along its
retained continuation lineage. Use checkpoint counters, not epoch filenames:
HetNet stores `reconstruction.counts.env_steps`; SoftRole stores `total_steps`.
Before inspecting evaluation outcomes, record the selected checkpoint hashes and
verify task, seed, paper environment, model/backend and shared simulator identity
against the preparation. The existing `prepare-evaluation` command is the legacy
study route; automatic paper-specific selection and manifests are not implemented.

These manual templates use the prepared common panel for all three seeds and
both SoftRole models. `HETNET_RUN` is the archive owning the selected checkpoint,
including a retained parent when applicable; `MODEL` is `shared` or `banked`.
Use fresh result paths. The scenario file fixes all 500 cases, so no additional
scenario seed, episode count or composition override is needed.

```bash
.venv/bin/python -m publication_reconstruction evaluate \
  --run-dir "$HETNET_RUN" --checkpoint "$HETNET_CHECKPOINT" \
  --scenarios "$PREPARATION/fc_scenarios.json" \
  --output "$RESULTS/hetnet_real_seed${SEED}.json"
.venv/bin/python -m softrole evaluate \
  --checkpoint "$SOFTROLE_CHECKPOINT" \
  --scenarios "$PREPARATION/fc_scenarios.json" \
  --output "$RESULTS/softrole_${MODEL}_seed${SEED}.json"
```

## Legacy reconstruction

This separate source tree reconstructs the **public February 2022 environment
family**, with explicit public-code and supplement-aligned model specifications. It is
**not a verified publication training checkout**. The actual source and commands
behind the AAMAS results remain unavailable. `--model-spec public-code-v1` is the
compatibility default (two layers, RMSprop). `--model-spec supplement-v1` selects
three layers and the active trainer's Adam optimizer. Both retain the public learner.

Legacy entrypoints and their default numerical behavior are preserved. SoftRole's
new environment path is opt-in; existing archives and logs are unchanged.
The reconstruction launcher uses `runs/publication_reconstruction/` by default,
refuses existing output directories, and executes a source archive inside each
new run. Repository edits during training cannot change those archived files.

For proposed training speed work, see the [evidence-backed performance plan](PERFORMANCE_PLAN.md).
It ties the three opt-in graph/model changes to the original study reconstruction
and the frozen SoftRole comparison, with code references, hashed evidence, exact
equivalence checks and matched Stokes benchmarks. The flags and optimizations in
that plan are proposed; they have not yet demonstrated an end-to-end speedup.

## Two environment versions

| Version | Purpose | Behavior |
|---|---|---|
| `historical-2022` | Closest runnable public-source reference | February environment behavior, including observation interference and FC phantom-front behavior; only the runtime repairs below. |
| `corrected-v1` | Controlled environment corrections | Independent PCP/FC observation copies; FC front/coordinate repairs described below. |

Both versions restore the **29-entry sensory-cell stride**, which was correct
before the October 2022 regression. They retain typed observations, original
agent actions, individual rewards and per-class GAE.
They do not use SoftRole's merged channels, gates, team critic or gradient rule.
Real and Binary actors support PP, PCP and FC through the existing local zero-A
compatibility guards.

Environment source: `d57da0717d5564027df7e8ba75614feca8006960`. Learner/runtime
scaffold: local `0ee9ceb38b6133866e9c1db40bed2c6caa3b842d`, including seed-before-
construction, explicit preserved gradient storage, CPU float64 and one Torch
thread per collector. `ORIGINS.json` records each file's source and hashes;
the audit's `runtime_changes.patch` records differences against those sources.

Minimal environment repairs in both versions:

- Replace removed `np.int` aliases with `int`.
- Handle equal hotspot bounds as the requested single coordinate, avoiding the
  historical FC `randint(x,x)` initialization failure.
- Guard an empty FC discovered-fire list before NumPy containment, avoiding a
  modern-runtime broadcast error. Historical nonempty containment is preserved
  only in `historical-2022`.
- Remove FC diagnostic printing that does not affect environment state.

Additional `corrected-v1` choices:

- Copy each P/A sensory window before masking the blind agent's view.
- Preserve a boundary fire's actual source coordinate, with the historical
  stationary boundary behavior; never emit an unfilled zero row as a fire.
- If a proposed fire-front position leaves the grid, retain the previous valid
  front position. This bounded behavior is an explicit reconstruction choice.
- Compare complete coordinates for discovered-fire/source membership rather than
  NumPy's elementwise containment. The underlying individual reward formula stays
  the same.

**FC is still a public-code benchmark reconstruction.** Its original individual
reward formula and ability to extinguish undiscovered fires differ from the
paper's description. These have not been silently replaced with speculative
paper semantics. Corrected boundary behavior is not evidence of what the authors
trained. `historical-2022` intentionally retains known defects for comparison;
do not describe it as a corrected physical benchmark.

## Recipes, budgets and running

Run from the repository root using the existing pinned environment:

```bash
.venv/bin/python -m publication_reconstruction train \
  --task pcp --variant real --seed 0 --recipe june-2022 \
  --env-version historical-2022 --dry-run
```

The dry run prints resolved settings and a structured command without creating
files. Removing `--dry-run` starts training. No research training is launched as
part of implementation validation.

| Recipe/task | Collectors | Epochs | Updates/epoch | Batch floor/collector | Episode horizon | Sample floor |
|---|---:|---:|---:|---:|---:|---:|
| June PP/PCP | 1 | 2000 | 10 | 500 | 80 | 10M |
| October PP/PCP | 4 | 2000 | 10 | 500 | 80 | 40M |
| June/October FC | 4 | 1400 | 10 | 500 | 300 | 28M |

These are dated **postconference sample commands**, not certified study settings.
June PP uses `predator_prey`; October PP uses `predator_capture` with 3P0A. Both
are explicitly given three physical agents, repairing the June command's missing
`--nagents 3`. PP/PCP have radius2 sensing, FC radius1, on 5x5 grids. FC has one
initial fire and reward type3. Learning rate is `1e-4`, detach gap5, gamma1,
GAE lambda.95, local norm clip.75, RMSprop alpha.97/epsilon1e-6 in compatibility mode. The public
learner's episode averaging and subsequent step denominator remain intact.

To compare the corrected environment under the same recipe:

```bash
.venv/bin/python -m publication_reconstruction train \
  --task pcp --variant real --seed 0 --recipe june-2022 \
  --env-version corrected-v1 --dry-run
```

To specify a sample stopping threshold, add `--max-env-steps 10000000`. The run
ends after the first complete update reaching that threshold, or at the epoch
cap, whichever happens first. Whole episodes and collector batches can overshoot;
the exact counts and theoretical maximum overshoot are recorded. No trajectories
are truncated to force an exact budget. A 10M threshold is a comparison choice,
not a recovered publication budget. Changing collectors changes both samples per
update and update count at a fixed sample budget.
The last epoch record may contain only the updates collected before the threshold;
its epoch number is an index, not a claim that every configured update completed.
The checkpoint's actual update and sample counts identify this partial record.

The launcher supports explicit `--epochs`, `--collectors`, `--updates-per-epoch`,
`--batch-steps`, `--horizon`, `--save-every`, and `--output` for bounded validation
or a preregistered protocol. A small executable smoke example is:

```bash
.venv/bin/python -m publication_reconstruction train \
  --task pp --variant binary --seed 0 --env-version corrected-v1 \
  --epochs 1 --updates-per-epoch 1 --batch-steps 1 --horizon 4 \
  --output runs/publication_smoke_example
```

Every run includes `protocol.json`, resolved args, an environment/package record,
source archive and manifest, true epoch metrics, parameter signatures, checkpoints,
stdout and exit status. Checkpoints additionally contain the environment version,
resolved settings, actual step/episode/update/epoch counts and source-manifest
hash. New schema-2 checkpoints additionally support complete-update continuation.
The reconstruction-specific evaluator uses archived training imports in an isolated
process; it does not use the historical evaluator's missing-panel path.

## Locked first wave and submission

[STUDY.json](STUDY.json) is the fixed scientific manifest. `study-plan` expands it
into twelve auditable runtime protocols; `run-index` accepts no scientific overrides.

| Indices | Workload | Seeds | Stop target | Saved sample milestones |
|---|---|---|---:|---|
| 0–2 | PP Real, 3P0A | 0, 1, 2 | 40M | 10M, 20M, 30M, 40M |
| 3–5 | PCP Real, 2P1A | 0, 1, 2 | 40M | 10M, 20M, 30M, 40M |
| 6–8 | FC Real, 2P1A | 0, 1, 2 | 28M | 10M, 20M, 28M |
| 9–11 | PCP Binary-16, 2P1A | 0, 1, 2 | 40M | 10M, 20M, 30M, 40M |

All use `supplement-v1`, `corrected-v1`, October recipes, four collectors,
500-step floors, ten updates/epoch and one numerical thread/collector. PP/PCP
horizon80 and FC horizon300 are unchanged. The caps are 2000/1400 epochs.
CPU float64 remains the numeric mode; the public Real attention constructors
retain some float32 parameters. Frozen loading preserves these recorded dtypes
instead of silently casting the model. This is a documented public-code detail.

Submit from the **Stokes repository root**, after syncing this implementation into
an idle checkout and preserving any live evaluation checkout. First:

```bash
mkdir -p logs_1
sbatch slurm/publication_preflight.sbatch
```

This four-index array runs PP Real, PCP Real, FC Real and PCP Binary-16 for exactly
100 updates each, with production batch sizes/collectors/horizons. Outputs default
to `runs/hetnet_preflight_JOBID/`. Each `preflight.json` includes update/checkpoint
times, throughput, a budget-time projection, package versions, per-process RSS
with explicit units, CPU affinity/allocation and a validated checkpoint.
Per-process peaks are not simultaneous job memory; compare Slurm accounting too.
These are fresh timing policies, never research seeds or usable performance results.
Requests of 16GB/2h for preflight and 16GB/48h for training are provisional.

To run the same four preflights concurrently on the Mac, from the repository root:

```bash
caffeinate -i .venv/bin/python scripts/publication_preflight_local.py
```

This uses a fresh timestamped `runs/hetnet_preflight_mac_*` directory and four
collectors per job (16 total). `caffeinate -i` prevents idle sleep while the command
runs. Add `--dry-run` to inspect commands without starting jobs, or
`--concurrency 1` to run the same workloads sequentially. `--run-root PATH` selects
a fresh output directory; existing directories are rejected. Ctrl-C stops the
local jobs and their collectors.

The terminal prints every episode's steps, success, mean-agent return and measured
rollout seconds, labelled by workload and collector. Episode lines are buffered
until their update completes. Each update also prints its success rate, mean-agent
return, mean episode length and mean rollout seconds per episode. Separate
`logs/WORKLOAD.log`, `.episodes.jsonl` and `.progress.jsonl` files retain raw output,
individual episode records and these summaries. `local_launch.json` records the
commands; `local_summary.json` records status, elapsed time and final throughput.
Each workload also retains its usual `updates.jsonl` (losses and full update time),
`metrics.jsonl` (epoch metrics), checkpoints and `preflight.json` validation.

`rollout_wall_time_seconds` measures each collector's call to `get_episode`,
including reset, policy inference, environment steps and scheduling delays. It
excludes batch backpropagation, optimizer work and collector communication.
Parallel episodes overlap, so adding their durations does not give job wall time;
the existing update and run timers measure that separately. Timing reads do not
change the training recipe or random streams. Earlier archived Stokes runs lack
this new episode field and retain their original source identity.

**Compute-node readiness remains a gate:** all four preflights must finish, produce
usable checkpoints and show suitable memory/CPU allocation and throughput. Inspect
their JSON and Slurm accounting before submitting the long array. No engineering
validation script submits it automatically. After that gate:

```bash
sbatch slurm/publication_train.sbatch
```

The real Slurm file fixes `--array=0-11%3`, one task, four CPUs, 16GB and48h.
Set `PUBLICATION_RUN_ROOT` before submission to choose a fresh root (default
`runs/hetnet_supplement_v1`). Existing outputs fail rather than overwrite.
The Python runtime requests a graceful stop at46h and completes its current update.
`run_status.json` distinguishes `paused_wall_time` from `budget_completed`.
Two hours is a buffer, not a guarantee against cluster interruption. There is no
automatic requeue: periodic checkpoints remain the fallback for abrupt termination.

Inspect or save resolved commands without training:

```bash
.venv/bin/python -m publication_reconstruction run-index --index 9 \
  --run-root runs/hetnet_supplement_v1 --dry-run
.venv/bin/python -m publication_reconstruction study-plan \
  --run-root runs/hetnet_supplement_v1 --output runs/hetnet_first_wave_protocols.json
```

## Recovery and accounting contract

Continue from the **exact checkpoint path** in `run_status.json` or a verified
periodic/milestone checkpoint, into a fresh segment. Replace the example checkpoint
basename below with the recorded path; no filenames or checkpoint counts are inferred.

```bash
sbatch slurm/publication_resume.sbatch \
  --run-dir runs/hetnet_supplement_v1/pcp_real/seed0 \
  --checkpoint runs/hetnet_supplement_v1/pcp_real/seed0/checkpoints/pcp_real/run1/model_update00001234_paused.pt \
  --output runs/hetnet_supplement_v1_segments/pcp_real_seed0_segment2
```

Resume inherits science, source, collector count and budgets; its CLI rejects
scientific overrides. Old checkpoints without recovery state are rejected.
It restores model, active optimizer, parent/worker Python/NumPy/Torch RNG,
absolute counts, partial-epoch statistics, elapsed epoch time and completed
milestones, after constructing workers. Checkpoint names include absolute updates.
Atomic publication prevents a partial write from appearing as a valid checkpoint.
Extra checkpoints and episode-log destination changes do not draw random numbers.

`artifacts.lineage_metrics(final_segment)` retains parent epochs only through the
recovery point; preparation similarly trims abandoned parent checkpoint suffixes.
Plot against the retained absolute **epoch**; preserve actual updates/steps in
tables and compare prespecified sample milestones. A final budget-limited epoch
can be partial. Concatenating every parent and child log directly would double-count.

For C collectors, floor b and maximum episode length H, each update satisfies
`C*b <= S <= C*(b+H-1)`: the last episode starts before the floor and adds at most H.
At the first update crossing B, previous count is at most B−1. Therefore overshoot
is at most2315 PP/PCP or3195 FC steps. Actual counts are authoritative. The sample
floors come from dated recipes; they do not prove convergence or budget optimality.

The new RNG scheme combines library, initialization/collection purpose, collector
and run seed through NumPy SeedSequence. Compatibility mode retains the original
offset scheme. Exactly reproducible recovery is tested on the same pinned runtime;
cross-platform bitwise identity is not promised.

Episode records include individual returns, their sum, mean-agent return, steps,
success, collector and absolute episode/update identity. They are buffered through
each update, then written in one batch to JSONL or a tagged stdout record. Slurm
uses stdout; epoch/update files remain compact and structured.

## Frozen panels and reporting

Prepare on a compute node only after all twelve policies have reached the required
selection checkpoints. `PCP_PANEL_DIRECTORY` below is the existing output directory
from `scripts/prepare_pcp_frozen.py`, containing `scenarios_nominal.json` and
`scenarios_failure.json`. Its exact bytes are reused after content verification.

```bash
sbatch slurm/publication_prepare_evaluation.sbatch \
  --run-root runs/hetnet_supplement_v1 \
  --pcp-panel-dir PCP_PANEL_DIRECTORY \
  --output runs/hetnet_frozen_first_wave
# After successful preparation:
sbatch runs/hetnet_frozen_first_wave/submit.sbatch
```

For continued runs, add `--run-map PATH.json`; this is an explicit JSON object
mapping keys such as `pcp_real/seed0` to the final segment directory. Unlisted keys
use their original run directories. Preparation verifies source, model/protocol,
checkpoint bytes and actual counts. It selects the **first saved checkpoint at or
above30M** for PCP,40M PP and28M FC, never using outcomes to choose checkpoints.
It produces24 jobs:12 nominal and six PCP failure/sham pairs. PCP uses500 scenarios
for each of(2,1),(1,2),(2,2),(3,1),(3,2), seed2700; failures use100 native scenarios,
seed2701, event window10–30. PP and FC each use500 native scenarios with seeds2702
and2703. Checkpoint overshoots remain in every report.

The evaluator strictly loads recorded tensors from verified archived training code
in a separate `python -I` worker. Model and environment versions, actual imported
module paths, training/evaluator hashes, panel identity and actual progress are
reported. Actions are sampled stochastically using separate environment/action/
message streams. All-to-all physical-position graphs match training; range-limited
protocols are rejected. An immutable evaluator source copy accompanies each report.

At zero-based PCP event step t, cached pre-event diagnostics are read, then all
four victim sensory channels are persistently zeroed before actor inference.
Position, physical class/actions, rewards, recurrence and communication remain.
HetNet receives no health input. Sham retains the same victim, event schedule and
random streams, without masking. Failure/sham prefixes must agree **within a policy**;
different policies share assigned scenarios, not necessarily trajectories.
PP/FC sensor failures and gate interventions are unsupported and rejected.

Summaries report both `R_team = sum_i R_i` and `R_agent = R_team / N`, individual
seed results and equal-weight seed means. The released `print_plot_eval.py` averages
over episodes and agents; a team sum is therefore a different reward scale.
For matched failure/sham panels the summary first subtracts within each checkpoint/
seed, then averages seeds. Bootstrap intervals resample whole seeds and remain
limited by only three training seeds. They do not turn a diagnostic pilot into
strong generalization evidence.

```bash
.venv/bin/python -m softrole summarize runs/hetnet_frozen_first_wave/results/*.json \
  --output runs/hetnet_frozen_first_wave/summary.json
```

Explicit protocol metadata groups by declared budget/panel while retaining actual
epoch/update counts; old reports keep legacy grouping. To annotate already-created
SoftRole reports read-only, summarize each panel with `--protocol` pointing to
the corresponding prepared `pcp_nominal/protocol.json` or `pcp_failure/protocol.json`.
The sibling `scenarios.json` must exactly match all report scenarios. New SoftRole
evaluations also accept this sidecar through `evaluate --protocol --scenarios`.
Models and incompatible environment/learner protocols remain separate strata.
Corrected HetNet FC cannot establish a matched comparison against old SoftRole FC;
that adapter/training integration remains outside this implementation.

## Source and citation evidence map

| Decision | Code and behavioral evidence | Primary source or derivation |
|---|---|---|
| Three layers, four16-wide heads, hidden concatenation, final averaging, Adam1e-3 | `runtime/hetgat/uavnet.py`, `policy.py`, `trainer.py`; `test_publication_spec.py`, `test_publication_recovery.py` | [Authors' supplementary §2.1](https://github.com/CORE-Robotics-Lab/HetNet/blob/master/AAMAS_22___HetNet_Supplementary.pdf); local PDF SHA256 `22262845fe6f8e1b6aa6f66f970c176a27c9a9f0550b4fc62066c7d99a928696` |
| One active Adam step; explicit betas/epsilon/flags | `Trainer.optimizer`, multiprocessing parent's step; independent one-step checks | [Pinned PyTorch2.2.1 Adam](https://raw.githubusercontent.com/pytorch/pytorch/v2.2.1/torch/optim/adam.py), [Adam paper](https://arxiv.org/abs/1412.6980). Defaults beyond optimizer/lr are declared choices. |
| Public learner preserved | `runtime/hetgat/policy.py`, `trainer.py`, `multi_processing.py`; compatibility fixture tests | Individual-agent GAE, padded class normalization, local episode averaging/clip.75 and `sum_j clipped_gradient_j / sum_j steps_j`; [GAE](https://arxiv.org/abs/1506.02438) supports the estimator background, not this normalization's optimality. |
| Relation-wise incoming attention retained | `test_publication_spec.py` compares actual Real/Binary operators with independent per-relation destination sums | [DGL2.1.0 edge_softmax](https://raw.githubusercontent.com/dmlc/dgl/v2.1.0/python/dgl/ops/edge_softmax.py). No null node or GATv2 is added. |
| Independent deterministic seed namespaces | `runtime/hetnet_ext/recovery.py`; exact split/resume and stream tests | [NumPy1.26 parallel RNG](https://numpy.org/doc/1.26/reference/random/parallel.html) explicitly warns against root-seed-plus-worker offsets across runs. |
| Recovery at complete updates | `runtime/main.py`, recovery helper, `artifacts.py`; four-collector Binary/FC partial-epoch tests | [Checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [pinned reproducibility limits](https://raw.githubusercontent.com/pytorch/pytorch/v2.2.1/docs/source/notes/randomness.rst) |
| Archived frozen evaluation | `evaluation.py`, `evaluation_worker.py`; replay, strict loading, masking, sham and source mutation tests | [Python isolated mode](https://docs.python.org/3.12/using/cmdline.html#cmdoption-I), together with explicit import-path verification in this implementation |
| Budget and paired reporting | `study.py`, `softrole/report.py`; accounting/selection/grouping tests | Complete-episode overshoot derivation above; [RL evaluation](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) motivates independent-seed uncertainty. |
| Threads, array and manual continuation | `slurm/publication_*.sbatch`; shell/mapping/quoting tests | [OpenBLAS](https://www.openmathlib.org/OpenBLAS/docs/faq/), [Slurm sbatch](https://slurm.schedmd.com/sbatch.html), [arrays](https://slurm.schedmd.com/job_array.html) |

The legacy scientific label is **supplement-aligned HetNet architecture/optimizer with
documented public-code learner and corrected environment**. The state output
width8 remains a public-code choice; per-class critics are explicitly supported
by the paper and supplement. The learner does not
adopt SoftRole's team-credit, null attention, banks or gradient aggregation.
Failure training, mixed rosters during training, bank search, GPU migration and
corrected-FC integration into SoftRole remain excluded.

Current provenance is refreshed with `scripts/record_publication_sources.py` into
a fresh audit folder. Earlier dated audit files are preserved; new runtime helpers
are marked local additions with no invented upstream commit. See the additive
`AGENTS.md` record and `analysis/publication_submission_2026-10-03/` for validation.

## Validation status (3 October 2026)

The final repository suite passed **438 tests in 153.44 seconds**. The separate
pre-change audit matched actual Real/Binary updates with one/four collectors,
including all RNG states and subsequent draws. An archived four-collector pause/
resume run matched uninterrupted training; frozen failure/sham CLI execution,
source-archive replay, 39 mocked Slurm checks, shell syntax and the lock check
also passed. Exact commands, hashes, scope and records are in
[`analysis/publication_submission_2026-10-03/`](../analysis/publication_submission_2026-10-03/)
and the additive [`AGENTS.md`](../AGENTS.md) record. No full research array or
cluster job was launched. Stokes preflight is still required before long training.

## What longer training currently shows

Existing SoftRole PP/PCP training windows, episode-weighted within each seed and
then equally weighted across three seeds:

| Model/task | Mean episode length near 10M | Near 20M |
|---|---:|---:|
| PP shared | 4.840 | 4.784 |
| PP banked | 4.874 | 4.797 |
| PCP shared | 10.692 | 7.001 |
| PCP banked | 17.594 | 8.752 |

All twelve seed-level lengths improve. PP shows diminishing gains; PCP improves
substantially beyond10M. This does not diagnose generalization or overfitting:
those require frozen validation checkpoints and a predeclared selection rule.
One and four collectors both imply20,000 updates over2000epochs, despite10M
versus40M sample floors. Existing runs were not stopped, retuned or relabeled.

See the [historical audit](../analysis/upstream_history_2026-09-30/AUDIT.txt),
[author search](../analysis/publication_reconstruction_2026-09-30/author_search/FINDINGS.txt)
and [budget analysis](../analysis/publication_reconstruction_2026-09-30/budgets/FINDINGS.txt).

Validation on30September2026: **220 tests passed**, including all178 existing
regressions. Eighteen tiny training executions covered both environment versions,
allthree domains, Real/Binary policies and representative two-collector runs.
They collected190 total joint steps/48episodes/36updates, verified finite changed
weights and checkpoint/source identities, and exercised stopping in a partial
epoch. These executions do not estimate learned performance. See
[validation evidence](../analysis/publication_reconstruction_2026-09-30/validation.json).
