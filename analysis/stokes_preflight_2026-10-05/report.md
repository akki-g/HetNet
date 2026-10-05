# Completed paper-v1 preflights: evidence and next steps

5 October 2026. **Proceed with fresh `paper-v1` / `torch-v1` research runs.**
The requested first directory is actually `hetnet_paper_preflight_905404`.
Both it and `hetnet_paper_preflight_905118` pass the prescribed 100-update and
frozen-checkpoint gates. Another generic preflight is unnecessary. Keep the
reviewed implementation and scientific settings fixed while the study runs.

This analysis did not train, evaluate policies, change runtime code, submit jobs,
or modify the supplied results. All recommendations below are prepared actions.

## What passed, and what that establishes

All eight runs used Torch, seed 991, the `paper-v1` model and environment,
`paper-equations-v1` learner, four collectors, a 500-step floor per collector,
and horizons 80 (PP/PCP) or 300 (FC). Profiling was disabled. Each completed 100
updates / ten epochs, saved a usable checkpoint and completed one frozen probe
with unchanged weights. Together: **800 updates, 1,793,520 joint environment
steps and 20,318 training episodes**; the eight probe episodes are separate.

| Workload | Steps in each array | Episodes in each array | Probe outcome in each array |
|---|---:|---:|---|
| PP Real | 215,817 | 3,118 | Horizon, 80 steps |
| PCP Real | 216,285 | 2,947 | Success, 70 steps |
| FC Real | 247,718 | 1,098 | Horizon, 300 steps |
| PCP Binary | 216,940 | 2,996 | Success, 54 steps |

Actual checkpoint tensors, active Adam moments and steps, signatures, ledger
counts, RNG states, source/dependency bindings and probe archives all pass.
Across arrays, corresponding final models, Adam states and all four collectors'
RNG states are **bitwise identical**. All 100 numerical update rows, ordered
episode summaries, ten epoch signatures and the frozen episode also match;
only explicitly declared clocks are excluded from numerical-ledger comparison.
Full action trajectories and intermediate gradient tensors were not recorded.

All eight training archives have the same 53-file source manifest as benchmark
904547 and the current reviewed reconstruction:
`17143ed20265d29e321f96609d4999144a84e748dc0b29ad1bde059c61b49599`.
The initial model/optimizer/RNG and first 20 numerical updates also match that
benchmark's corresponding Torch runs exactly. Recorded Git HEAD is
`a17898917faeea999965c2bc272b95e73a1160e4`; Torch 2.2.1, DGL 2.1.0 and NumPy
1.26.4 are unchanged. ORIGINS validates all 43 runtime files.

Evidence: [independent provenance review](provenance_review/README.md),
[checkpoint/source results](provenance_review/review.json),
[independent learner/episode audit](learner_review/review.json), and the
[raw 905404](../../stokes_runs/hetnet_paper_preflight_905404/pcp_binary/seed991/preflight.json)
and [905118](../../stokes_runs/hetnet_paper_preflight_905118/pcp_binary/seed991/preflight.json)
receipts. Scripts and input hashes accompany these results.

These are reproducible engineering checks at one repeated seed, not eight
independent learning runs, a convergence result or a new DGL/Torch comparison.
The controlled backend evidence remains [benchmark 904547](../stokes_backend_2026-10-05/report.md):
median paired speedups 1.377× PP, 1.351× PCP Real, 1.401× FC, 1.282× PCP Binary.

## Measured speed and scheduling implications

Update throughput is `sum(steps) / sum(update wall seconds)` over all 100 updates.
Budget projections use the complete training-segment rate, including logging
and checkpointing, and exclude startup, frozen probes and queueing. They hold
the early rate constant; the two endpoints are observations, not confidence
intervals or guaranteed bounds.

| Workload | 905118 update steps/s | 905404 update steps/s | Fixed budget per seed | Projected active hours per seed | 46-hour HetNet segments at these rates |
|---|---:|---:|---:|---:|---:|
| PP Real | 335.2 | 342.1 | 40M | 32.7–33.3 | 1 |
| PCP Real | 190.0 | 263.5 | 40M | 42.3–58.6 | 1–2 |
| FC Real | 191.1 | 292.4 | 28M | 26.7–40.8 | 1 |
| PCP Binary | 128.4 | 197.4 | 40M | 56.5–86.8 | 2 |

The later array is faster by 2.0%, 38.7%, 53.0% and 53.7%, respectively. Since
the implementation and numerical work are identical, these are execution-time
differences, not backend gains or evidence that one policy learned better.
The artifacts contain no hostname, CPU model/topology or actual Slurm node/
constraint record. **We cannot identify the cause or certify that either array
ran on Sapphire Rapids.** Affinity indices alone cannot identify hardware.

905404 PCP improves from 233.9 steps/s in updates 1–20 to 272.1 in 21–100.
Its FC and Binary first-20 rates are within 0.6% of the earlier benchmark's
matching Torch run, but PP remains about 35% below its matching first-20 rate
(about 32% below the benchmark's post-first-update median). Thus the prior
22-hour PP projection is too optimistic for these preflight observations.
905118 FC falls from 205.7 to 184.5 steps/s from the first to last 20-update
window. Preserve this uncertainty in scheduling rather than choosing only the
fastest observation. [All window rates](metrics/update_windows.csv)

Aggregate measured CPU use is 3.385–3.600 cores, or 84.6–90.0% of four CPUs;
system CPU is only 0.405–0.686% of process CPU time. Slower runs also use more
CPU milliseconds per step. Checkpointing is only 0.009–0.0385% of segment time.
Summed individual lifetime peak RSS is 3.630–6.477 GiB; this is not a concurrent
job/cgroup peak and can double-count shared pages. No memory-growth, swap,
fault or pressure time series was saved. Keep four CPUs and 16 GiB; these
results do not justify more collectors, less memory or checkpoint suppression.
[Resource calculations and limitations](resource_review/results.json)

Across the twelve HetNet research runs, the two observed rate sets imply
**474.3–658.7 job-hours**, or **6.6–9.1 idealized days** with three continuously
occupied HetNet slots. This excludes queue delays, continuation delays,
startup, changing episode lengths, SoftRole training and evaluation. The HetNet
and SoftRole study arrays each permit three concurrent jobs; launching both
allows up to six jobs, 24 CPUs and 96 GiB in total.

## Validity and reporting cautions

- All recorded losses and preclip norms are finite; all 800 updates invoke the
  declared 0.75 global clip. FC's median preclip norm is 26,738.1 and its value
  loss spans 20,239–334,447. Its summed-time critic objective, 300-step horizon
  and extinction reward scale differ from PP/PCP. These magnitudes alone do not
  identify a defect. Do not change the learning rate, clipping, reward scale or
  loss to improve this preflight. Monitor them during the locked runs.
- Use `updates.jsonl` and `metrics.jsonl` for paper learner losses. Checkpoint
  legacy `log` losses retain per-step normalization; their conversion was
  reconciled with the structured per-episode values. Legacy entropy zeros are
  placeholders, not measurements. See [main.py](../../publication_reconstruction/runtime/main.py),
  [recording.py](../../publication_reconstruction/runtime/hetnet_ext/recording.py)
  and [paper_learning.py](../../publication_reconstruction/runtime/hetnet_ext/paper_learning.py).
- The single FC probe has positive mean return despite failure. Individual
  extinctions earn reward even when some fires remain; this is compatible with
  [the archived reward/termination implementation](../../publication_reconstruction/runtime/envs/ic3net_envs/fire_commander_env.py).
  Four single-scenario outcomes cannot assess task quality.
- The reconstruction is paper-aligned with declared choices. Binary is the
  agreed **64 bits per head / 256 total per sender per round** interpretation;
  these checks cannot certify the unavailable original publication checkout or
  its precise bandwidth convention. Preserve the [fidelity statement](../../publication_reconstruction/FIDELITY.md).
- Fresh paper-FC SoftRole and HetNet share physical simulator semantics, while
  actor information and learning methods differ deliberately. Old FC archives
  have different actions/rewards and must stay outside the matched comparison.
- Do not continue seed-991 preflight checkpoints into the research study. They
  exhausted their ten-epoch engineering budgets. Start seeds 0, 1 and 2 fresh.

## Exact next steps

**1. Freeze this implementation and prepare one fresh study root on Stokes.**
Use the same installed environment, with no package update. Run from the
repository root. The following block only prepares protocols and validates
source bytes; it does not submit or train. If either path exists, select a new
suffix instead of overwriting or deleting it.

```bash
set -euo pipefail
export STUDY_ROOT=runs/paper_study_20261005_v1
export PREPARATION=runs/paper_study_20261005_v1_preparation
test ! -e "$STUDY_ROOT"
test ! -e "$PREPARATION"
mkdir -p logs_1 logs_sr
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
from publication_reconstruction.__main__ import validate_source_origins
validate_source_origins()
reference = Path('runs/hetnet_paper_preflight_905404/pp_real/seed991/source_manifest.json')
raw = reference.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '17143ed20265d29e321f96609d4999144a84e748dc0b29ad1bde059c61b49599'
for name, digest in json.loads(raw)['files'].items():
    assert hashlib.sha256((Path('publication_reconstruction') / name).read_bytes()).hexdigest() == digest, name
print('Reviewed preflight reconstruction source matches exactly')
PY
.venv/bin/python -m publication_reconstruction paper-study-plan \
  --message-backend torch-v1 --run-root "$STUDY_ROOT" \
  --output "$PREPARATION"
```

The original Stokes run location above comes from the recorded protocol, not
the Mac's `stokes_runs/` copy. If the original archive was moved, change only
`reference` to its actual manifest path. A source mismatch requires review;
do not regenerate ORIGINS just to accept it. Reprepare if reviewed source
changes. The prepared identity also records dependencies and launchers.

**2. Launch the locked twelve fresh HetNet runs with the benchmark's explicit
CPU-family constraint.** No additional backend/profiling sweep is required.
The script otherwise defaults to DGL, so the backend environment variable is
essential. Each array entry has four CPUs, 16 GiB and 48 hours; the learner
pauses after 46 active hours, at a complete update.

```bash
HETNET_JOB=$(PUBLICATION_MESSAGE_BACKEND=torch-v1 \
  PUBLICATION_RUN_ROOT="$STUDY_ROOT/hetnet" \
  sbatch --parsable --constraint=sapphirerapids slurm/publication_paper_train.sbatch)
printf '%s\n' "$HETNET_JOB" > "$PREPARATION/hetnet_job_id.txt"
```

Indices 0–2 are PP Real seeds 0–2; 3–5 PCP Real; 6–8 FC Real; 9–11 PCP
Binary. Budgets are 40M/40M/28M/40M steps per seed. Keep all horizons, collector
floors, dtypes, random streams and optimizer settings unchanged. Source:
[study mapping and budgets](../../publication_reconstruction/study.py),
[training launcher](../../slurm/publication_paper_train.sbatch).

**3. Launch the six matched paper-FC SoftRole runs under the same prepared
root.** This supplies shared/banked seeds 0–2 at 28M each for the intended
comparison. The two preflight arrays audited here assess HetNet, not SoftRole
throughput; do not apply HetNet timing estimates to SoftRole.

```bash
SOFTROLE_JOB=$(SOFTROLE_PAPER_RUN_ROOT="$STUDY_ROOT/softrole_fc" \
  sbatch --parsable --constraint=sapphirerapids slurm/softrole_paper_fc.sbatch)
printf '%s\n' "$SOFTROLE_JOB" > "$PREPARATION/softrole_fc_job_id.txt"
```

Source: [matched preparation](../../publication_reconstruction/paper_study.py),
[SoftRole FC launcher](../../slurm/softrole_paper_fc.sbatch). The launcher has
no 46-hour graceful pause: it saves every ten epochs (100 updates) and at
completion. A 48-hour timeout requires recovery from its last valid saved
checkpoint into a fresh directory, using unchanged source; the unsaved suffix
is discarded. Validate the checkpoint and signature rather than choosing only
the highest filename: a timeout during a write can leave a partial file.
SoftRole resume executes the current learner/model code and checks simulator
identity; unlike HetNet, it does not automatically execute the full parent
archive. Compare source hashes and keep the checkout frozen. The original
SoftRole FC array launcher does not accept `--resume`; use the
[explicit recovery recipe](provenance_review/softrole_recovery.md).
Keep the parent and record this retained lineage. Its method and settings must
remain fixed. Do not pass SoftRole checkpoints to the HetNet resume command.

**4. Recover missing scheduler evidence and monitor sample counts.** This
read-only command can run alongside training; it need not delay submission.
It identifies historical nodes and requested constraints if accounting retained
them. It does not supply a CPU-model measurement by itself.

```bash
sacct -j 905118,905404 --parsable2 \
  --format=JobID,State,ExitCode,NodeList,Constraints,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS \
  > "$PREPARATION/preflight_accounting.psv"
```

Record the same accounting for the newly saved array IDs after their tasks
finish. While active, retain `scontrol show job <array_job_id>_<task_id>` and
`scontrol show node <recorded_node>` outputs. For a later diagnostic performance
run, capture hostname, `lscpu`, affinity and Slurm allocation metadata inside
its allocation, as the existing bounded benchmark already does. Requested
constraints and CPU masks must not be presented as measured CPU models.
The listed fields follow [official sacct documentation](https://slurm.schedmd.com/sacct.html).

At each completed epoch, inspect structured counts, finite losses/norms,
success and segment throughput, and verify the first saved checkpoint's source
and model/backend identity. Do not tune scientific settings from these early
outcomes. Slurm `COMPLETED` is not sufficient: HetNet's `run_status.json` must
say `scientific_budget_completed: true` **and** reach its intended step target.
For SoftRole, inspect `finished.json` and checkpoint `total_steps >= 28000000`.
A missing final status after timeout is an incomplete run, not zero performance.

**5. Continue HetNet pauses from their exact recorded checkpoint.** Plan this
for every Binary seed; PCP Real may need it too. At the faster observed rate,
Binary reaches roughly 32.58M steps in 46 hours and needs another 10.48 active
hours; at the slower rate it needs roughly 40.8 additional hours. Queue delay
is extra. The existing resume route restores source, backend, optimizer, RNG,
partial epoch and remaining scientific budget; never restart the seed or reduce
40M to fit one allocation.

This concrete example continues Binary seed 0 once its first segment has paused:

```bash
export PARENT="$STUDY_ROOT/hetnet/pcp_binary/seed0"
export SEGMENT="$STUDY_ROOT/continuations/pcp_binary/seed0/segment02"
CHECKPOINT=$(.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
s = json.loads((Path(os.environ['PARENT']) / 'run_status.json').read_text())
assert s['stop_reason'] in ('paused_wall_time', 'paused_signal'), s['stop_reason']
assert not s['scientific_budget_completed']
assert Path(s['checkpoint']).is_file()
print(s['checkpoint'])
PY
)
test ! -e "$SEGMENT"
SLURM_CPU_BIND=cores sbatch --export=ALL --constraint=sapphirerapids \
  slurm/publication_resume.sbatch \
  --run-dir "$PARENT" --checkpoint "$CHECKPOINT" --output "$SEGMENT"
```

Repeat for each paused run, updating parent and fresh segment paths along its
lineage. No automatic continuation is implemented. The resume script lacks an
explicit `srun --cpu-bind`; inherited `SLURM_CPU_BIND=cores` supplies that setting
without editing the reviewed script, per [official srun documentation](https://slurm.schedmd.com/srun.html).
An abrupt timeout without `run_status.json` instead requires selecting the last
valid saved checkpoint and documenting the discarded unsaved suffix.

**6. Lock the final comparison manifest while training proceeds.** Use the
prepared `fc_scenarios.json` common panel (500 cases, seed 2703). For each of
the three HetNet FC and six SoftRole FC runs, select the first saved checkpoint
at or above 28M environment steps along its retained continuation lineage,
before inspecting evaluation outcomes. Record its exact file hash, counts,
training seed, environment/model/backend, simulator identity and parent chain.
Use `reconstruction.counts.env_steps` for HetNet and `total_steps` for SoftRole,
not the epoch filename. Then use the [existing manual frozen-evaluation commands](../../publication_reconstruction/README.md)
on that one panel. Paper-specific automated selection/manifests remain a small
orchestration task; the legacy `prepare-evaluation` command targets the old
protocol and must not be used for these runs. Report independent training seeds
separately, then aggregate with equal seed weight; do not count the preflight
repeats as study seeds or choose checkpoints using evaluation returns.

## Reproduction of this analysis

```bash
.venv/bin/python analysis/stokes_preflight_2026-10-05/analyze.py
.venv/bin/python analysis/stokes_preflight_2026-10-05/provenance_review/audit.py
AUDIT_OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/hetnet-preflight-audit.XXXXXX")
.venv/bin/python analysis/stokes_preflight_2026-10-05/learner_review/audit.py \
  --output "$AUDIT_OUTPUT/learner.json"
.venv/bin/python analysis/stokes_preflight_2026-10-05/resource_review/analyze.py \
  --output "$AUDIT_OUTPUT/resources.json"
```

The learner and resource scripts require fresh `--output` JSON paths. Their
original results and input hashes are retained here. Root timing
calculations are in [summary.json](metrics/summary.json),
[throughput.csv](metrics/throughput.csv) and [update_windows.csv](metrics/update_windows.csv).
Input hashes and independent audit results distinguish measured facts from
planning projections. No numerical implementation test was rerun for this
analysis-only change; prior implementation validation remains attributed to
its original date in `AGENTS.md`.
