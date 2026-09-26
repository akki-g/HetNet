# Stokes runbook: original HetNet Phase A

This prepares original HetNet-Real/A2C PCP training. No cluster jobs have been submitted by this work. Calibration requires passing Gate A and verified live Stokes facts. Full Run 1 additionally requires Akki's confirmation of the measured budget. Run 2 is a later phase, with no submission script here. Consult the actual gate report; the existence of scripts does not establish a passed gate.

Official sources, wheel evidence, and live unknowns are documented in [STOKES_SOURCE_AUDIT.md](research/STOKES_SOURCE_AUDIT.md). Public documentation establishes neither today's partition time cap nor this user's balance, account requirement, available concurrency, or compute connectivity. The existing Newton wrapper's resources are not Stokes measurements.

## 1. Freeze the scientific checkout

Use a dedicated Stokes checkout at Phase A scientific commit **`2db8a62`**, matching the passing Gate A report. The initial infrastructure commit `be2d223` was followed by audited budget/provenance fixes; do not deploy the initial commit alone. Preserve the final checkout, `uv.lock`, and `.venv-stokes` while calibration or Run 1 tasks are queued/running. Do not pull new code into it. Develop Phase B in a **separate checkout/worktree with its own environment**: extension, test, config, or script changes invalidate queued jobs' Gate A hash.

The inventory hashes tracked Python/shell sources, `pyproject.toml`, `uv.lock`, and study source/config/test/script files, including new untracked study files. Documentation/results are excluded so an evidence-only commit does not invalidate equivalent code. Every run still records HEAD, dirty status, the code inventory and lock hash. Gate A separately verifies all reviewed deviations from the original baseline.

The reviewed original-file commits are `2fadecf`, `7b334c1`, `0cfcea5`, and `50d0c37`. Do not expand that allowlist automatically. The interrupted first attempt at `runs/gate_a/full_20260926_01` is preserved as incomplete evidence; the replacement target is `runs/gate_a/full_20260926_02`. Inspect its final report before using it. Starting a gate is not passing it.

## 2. Local gate and task mapping

From the root with the locked Python 3.12 environment:

```bash
.venv/bin/python -m unittest discover -s tests -p test_grid_slurm.py -v
.venv/bin/python -m hetnet_ext.grid --dry-run
.venv/bin/python -m hetnet_ext.grid --dry-run --mode calibration
.venv/bin/python -m hetnet_ext.gate_a --plan
```

Dry runs print commands and intended paths without creating run directories. Training has exactly 21 tasks:

| Indices | Composition | Seeds | Priority |
|---|---|---|---|
| 0–4 | 2P1A | 0–4 | P0 |
| 5–9 | 3P3A | 0–4 | P1 |
| 10–14 | 4P6A | 0–4 | P1 |
| 15–17 | 3P1A | 0–2 | P2 |
| 18–20 | 2P2A | 0–2 | P2 |

Calibration indices 0 and 1 select 2P1A and 4P6A, both seed 0, for exactly **20 epochs**. Training uses 2,000 epochs. Both use `epoch_size=10`, four processes, per-process batch target 500, horizon 80, map dimension 5, hidden size 128, learning rate 0.0001, detach gap 5, real-valued HetGAT/A2C, and checkpoint interval 50. Calibration explicitly saves epoch 20. Singleton-P is excluded; singleton-A is valid.

To reproduce Gate A after a reviewed change, choose a new output directory and supply the successful clean locked-environment evidence:

```bash
: "${CLEAN_ENV_EVIDENCE:?Set the successful clean uv sync/import evidence JSON}"
: "${GATE_A_OUTPUT:?Choose a new evidence directory}"
.venv/bin/python -m hetnet_ext.gate_a \
  --output "$GATE_A_OUTPUT" --clean-environment-evidence "$CLEAN_ENV_EVIDENCE" \
  --allowed-upstream-commit 2fadecf --allowed-upstream-commit 7b334c1 \
  --allowed-upstream-commit 0cfcea5 --allowed-upstream-commit 50d0c37 \
  --run-smokes
```

All eleven required checks must be true: environment, clean locked environment, baseline diff, unit contracts, all-composition smokes, shared weights, determinism, cross-composition load, grid/scripts, gradient storage, and fresh gradient aggregation. Missing, skipped, failed, or stale proof blocks cluster jobs. No production bypass flag is provided. Unit tests alone do not establish successful learning.

## 3. Live Stokes preflight

Preserve complete timestamped raw output on Stokes:

```bash
date -u
hostname
scontrol show partition normal
sinfo -p normal -o "%P %l %c %m %D"
sacctmgr show assoc user="$USER" format=Account,Partition,MaxJobs,MaxSubmit,GrpTRES
myusage
quota -s
module avail anaconda
squeue -u "$USER"
```

If necessary use `lfs quota -h -u "$USER" /lustre/fs1`. Inspect account/QoS restrictions as well as user limits. Blank fields do not establish unlimited use. Ensure all 21 pending/running elements and chosen concurrency fit the verified limits **after existing jobs and applicable shared-account use**. Runtime validates the array size against the declared limit but cannot reserve scheduler capacity beforehand.

After Gate A and scheduler checks, use a short verified compute allocation to check Python 3.12 and actual artifact access to PyPI/wheel hosts, GitHub releases, and `astral.sh`. Capture `module list`, interpreter version, `uname -a`, `/etc/os-release`, `lscpu`, `getconf GNU_LIBC_VERSION`, `pwd`, and `df -h .`. The [ARCC submission guide](https://arcc.ist.ucf.edu/docs/scheduler/scripts/) gives interactive allocation examples; their memory/time values are not HetNet requirements. Unresolved limits, Python, network or account identity require Akki's decision; do not silently switch module/cluster/partition or install packages on a login node.

Create verified preflight JSON. This incomplete template intentionally fails until its unknowns are replaced with evidence:

```json
{
  "schema_version": 1,
  "verified": false,
  "cluster": "stokes",
  "partition": "normal",
  "max_wall_time_seconds": null,
  "remaining_core_hours": null,
  "max_concurrent_jobs": null,
  "max_submit_jobs": null,
  "account_required": null,
  "account": null,
  "verified_at_utc": null,
  "evidence_paths": [],
  "python_major_minor": null,
  "compute_network_verified": false
}
```

Numeric limits must be positive; job-count limits require positive integers or the explicit evidence-backed string `"unlimited"`. `account_required` must be a boolean and a required account must be named. Set `verified=true`, Python `"3.12"`, and network verification true only after checks pass. Preserve referenced raw evidence; recheck balance and occupancy before full submission.

## 4. Bootstrap and endpoint calibration

Wrappers request `normal`, one node/task, four CPUs and no GPU. They have **no default memory or wall time**. All installs occur inside a compute allocation, following [ARCC Anaconda guidance](https://arcc.ist.ucf.edu/docs/software/anaconda/). Bootstrap clears Python overrides, loads `anaconda/anaconda-2024.10`, activates base, asserts Python 3.12, then validates preflight/Gate A before installation. An exclusive `.tools/stokes-setup.lock` serializes uv 0.12.5 setup and `uv sync --locked --no-python-downloads` into `.venv-stokes`. It verifies imported package versions. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `HETNET_TORCH_THREADS` are one; the CPU patch explicitly sets Torch threads to one too.

Choose and record calibration memory/time from local resource evidence and the live cap, with a documented margin. A local smoke or documentation example is not a guaranteed Stokes bound. In the submitting shell, use absolute compute-readable evidence paths:

```bash
export HETNET_STOKES_PREFLIGHT=/absolute/path/to/verified-stokes-preflight.json
export HETNET_GATE_A=/absolute/path/to/passing-gate-report.json
export HETNET_MEM_GB
export HETNET_TIME_SECONDS
export HETNET_ARRAY_CONCURRENCY
: "${HETNET_MEM_GB:?Set planned calibration memory in integer GiB}"
: "${HETNET_TIME_SECONDS:?Set explicit time as whole-minute seconds}"
: "${HETNET_ARRAY_CONCURRENCY:?Set verified available concurrency}"
mkdir -p logs
```

Before submission, use an available Python 3.12 interpreter to validate evidence and format resources. On a fresh Stokes checkout, the commands below use the verified module's `python3` for **stdlib-only** checks. Do not create a login-node environment just to run them.

```bash
python3 - <<'PY'
import os
from hetnet_ext.train_job import read_json, validate_preflight
p = read_json(os.environ['HETNET_STOKES_PREFLIGHT'])
validate_preflight(p)
assert p['max_submit_jobs'] == 'unlimited' or p['max_submit_jobs'] >= 2
PY
account_args=()
stokes_account=$(python3 - <<'PY'
import os
from hetnet_ext.train_job import read_json
p = read_json(os.environ['HETNET_STOKES_PREFLIGHT'])
print(p['account'] if p['account_required'] else '')
PY
)
if [[ -n "$stokes_account" ]]; then account_args=(--account="$stokes_account"); fi
slurm_time=$(python3 - <<'PY'
import os
s = int(os.environ['HETNET_TIME_SECONDS'])
assert s > 0 and s % 60 == 0
print(f'{s//86400}-{s//3600%24:02}:{s//60%60:02}:{s%60:02}')
PY
)
sbatch "${account_args[@]}" --export=ALL \
  --array="0-1%${HETNET_ARRAY_CONCURRENCY}" \
  --mem="${HETNET_MEM_GB}G" --time="$slurm_time" slurm/calibrate.sbatch
```

These are future operator commands, not recorded submissions. A bare integer for `sbatch --time` means minutes; conversion deliberately produces `days-HH:MM:SS`. Runtime checks actual partition, CPUs, throttle, memory, time and required account. Calibration writes `runs/calibration/<composition>/seed0/`; optional `HETNET_OUTPUT_ROOT` selects a new explicit root. Preserve its choice with the submitted command.

Each job launches `main.py` through its own `sys.executable`, tees stdout/stderr, forwards termination signals, and refuses existing output directories. It preserves config, fully resolved args, git/lock/code inventory, hardware/Slurm provenance, epoch metrics/signatures, checkpoint save records, and initial/final signatures. Successful completion requires contiguous metrics/signatures, a unique final checkpoint, and equality between its signature and the final training epoch's signature.

## 5. Measure costs and obtain budget confirmation

After both tasks finish, preserve scheduler/jobstats evidence, for example:

```bash
: "${CALIBRATION_ARRAY_ID:?Set the returned calibration array ID}"
sacct -j "$CALIBRATION_ARRAY_ID" \
  --format=JobID,State,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU
myusage
```

Use each task's allocated elapsed time and four CPUs; do not sum parent-array and child-step rows as separate allocations. `TotalCPU` is utilization rather than allocated core-hours. A single process/step `MaxRSS` may not establish whole-job peak; inspect available cgroup/jobstats measurements and record their scope. Upstream `tracemalloc` omits much native/multiprocess memory. If whole-job memory remains unknown, do not fabricate a budget.

Create this measurements JSON, replacing every null with measured evidence or an explicit justified reserve:

```json
{
  "jobs": {
    "2P1A": {
      "whole_job_peak_rss_gb": null,
      "peak_rss_source": null,
      "setup_seconds": null,
      "allocation_wall_seconds": null,
      "accounting_evidence": null
    },
    "4P6A": {
      "whole_job_peak_rss_gb": null,
      "peak_rss_source": null,
      "setup_seconds": null,
      "allocation_wall_seconds": null,
      "accounting_evidence": null
    }
  },
  "evaluation_reserve_core_hours": null,
  "other_reserve_core_hours": null
}
```

Use GiB consistently with Slurm's `G`. `setup_seconds` covers nonoverlapping setup/initialization, including allocated time waiting for uv's lock; document its derivation. Bootstrap and launcher elapsed times support that accounting. Epoch duration excludes checkpoint writes and does not directly include all per-epoch signature/logging overhead. The projection explicitly accounts for the remaining measured allocation time as described below; preserve its derivation and revisit the estimate if memory/timing trends invalidate the margin. Include earlier cluster probes/retries in the explicit other reserve where applicable. A zero evaluation reserve is a declared estimate, not proof that Run 2 is free.

Checkpoint-save duration comes directly from `checkpoint_records.jsonl`, whose checkpoint byte count/signature must match verified provenance and the checkpoint content hash. Do not manually guess save cost. Downloaded checkpoint trees can relocate, but hashes must still match. The budget requires complete epochs 1–20, matching endpoint composition/P/A/seed in config and provenance, and hashed resolved arguments matching the complete production recipe and original default PCP/model settings. Renaming a 2P1A artifact directory to 4P6A does not make it a valid endpoint measurement.

```bash
python3 -m hetnet_ext.budget \
  --calibration-root runs/calibration \
  --measurements /absolute/path/to/calibration-measurements.json \
  --preflight "$HETNET_STOKES_PREFLIGHT" \
  --out /absolute/path/to/new-budget-directory
```

The report separates the first epoch and reports mean/nearest-rank p90 over epochs 2–20, seconds/update and seconds/actual transition. It computes `residual = allocation_wall - setup - sum(epoch_durations) - recorded_save` and rejects negative residuals. All residual time is amortized as `residual/20` per epoch, including signature/logging overhead. This conservatively repeats any shutdown or unclassified fixed startup also included in the residual; it does not pretend to identify those components separately. Intermediate rosters use the largest combined endpoint slope and maximum setup, first-epoch excess, save cost and memory, explicitly labeled as **modeled assumptions, not measured slopes or proved bounds**.

One epoch contains ten updates. Under the original whole-episode collection bound, each full run collects 40,000,000–46,320,000 joint transitions, not 4,000,000. The report uses actual fresh step totals for throughput; the derivation is in the source audit.

The model is `wall = setup + max(0, first_epoch - p90_epoch) + 2000 × (p90_epoch + residual/20) + 40 × measured_save`. Requested time is 1.3 times that estimate rounded up to a complete minute; memory is 1.5 times measured whole-job peak rounded up to GiB. Guarded expected cost sums all 21 seeds at four CPUs, adds actual calibration core-hours, and adds explicit reserves. The single array requests the maximum time/memory across rows. JSON separately reports its full uniform reservation envelope: requested maximum and actual elapsed usage are different quantities.

Review every resource row, unmeasured-roster assumptions, total core-hours and percentage of a fresh remaining balance. Above **4,000 core-hours or 10% of remaining balance**, stop for Akki's explicit threshold decision. Above the partition cap, stop for a resume/recipe decision. No true-resume implementation exists here and epochs are not silently reduced.

The tool writes an **unapproved** `approval_template.json` tied to the budget file's SHA-256. Only after Akki confirms the concrete report should an operator copy it to an approval record, fill `approved=true`, approving identity and UTC timestamp, and, if explicitly authorized, `threshold_override_approved=true`. Editing the budget afterwards invalidates that record. Approval is also bound to the current preflight file hash and balance: a refreshed preflight needs a fresh matching budget, preventing approval against an older, larger balance. The record documents the user's decision; it cannot supply permission by itself.

## 6. Submit and monitor Run 1

Refresh balance/occupancy before submission; if material facts/costs changed, regenerate and reconfirm the budget. Verify 21-task submission capacity, available concurrency, disk/quota, passing gate, and unchanged scientific hash. Read uniform resources from the confirmed budget:

```bash
export HETNET_BUDGET_APPROVAL=/absolute/path/to/confirmed-approval.json
export HETNET_MEM_GB=$(python3 - <<'PY'
import os
from pathlib import Path
from hetnet_ext.grid import ROOT, code_sha256, file_sha256
from hetnet_ext.train_job import validate_approval
b = validate_approval(Path(os.environ['HETNET_BUDGET_APPROVAL']), code_sha256(), file_sha256(ROOT / 'uv.lock'), Path(os.environ['HETNET_STOKES_PREFLIGHT']))['budget']
print(b['array_uniform_memory_gb'])
PY
)
export HETNET_TIME_SECONDS=$(python3 - <<'PY'
import os
from pathlib import Path
from hetnet_ext.grid import ROOT, code_sha256, file_sha256
from hetnet_ext.train_job import validate_approval
b = validate_approval(Path(os.environ['HETNET_BUDGET_APPROVAL']), code_sha256(), file_sha256(ROOT / 'uv.lock'), Path(os.environ['HETNET_STOKES_PREFLIGHT']))['budget']
print(b['array_uniform_time_seconds'])
PY
)
python3 - <<'PY'
import os
from hetnet_ext.train_job import read_json, validate_preflight
p = read_json(os.environ['HETNET_STOKES_PREFLIGHT'])
validate_preflight(p)
assert p['max_submit_jobs'] == 'unlimited' or p['max_submit_jobs'] >= 21
PY
```

Recompute `slurm_time` using section 4, retain verified `account_args` and concurrency, and ensure no calibration-specific `HETNET_OUTPUT_ROOT` remains exported. Submit from the immutable checkout root:

```bash
mkdir -p logs
sbatch "${account_args[@]}" --export=ALL \
  --array="0-20%${HETNET_ARRAY_CONCURRENCY}" \
  --mem="${HETNET_MEM_GB}G" --time="$slurm_time" slurm/run1_train.sbatch
```

Default output is `runs/run1_train/<composition>/seed<seed>/`. Upstream checkpoint nesting is retained, typically `checkpoints/<composition>_s<seed>/run1/model_ep2000.pt`. Use the recorded `final_checkpoint` field rather than guessing a flattened path.

Monitor `squeue`, `sacct`, stdout, epoch JSONL and `myusage`. Around epoch 300, inspect source-policy success, steps taken, rewards, losses and timing. If learning is flat or tasks fail, preserve evidence and ask Akki before further expensive work. Engineering gates do not prove policy competence.

Do not delete failed output to make a retry appear original. Use a new explicit `HETNET_OUTPUT_ROOT`, preserve the original attempt/job IDs and reason, target only affected authorized indices, and account for retry cost. No automatic resume, overwrite or evaluation chaining is provided. Run 2 needs its later implementation, preregistration and Gate B.

## 7. Preserve and retrieve evidence

Keep configs, provenance, resolved args, epoch/checkpoint metrics and signatures, checkpoints, complete Slurm logs, raw cluster/accounting evidence, gate reports, measurements, budget and confirmation together. [ARCC storage policy](https://arcc.ist.ucf.edu/docs/data/files/) states user/group storage is not backed up. From the local machine, use verified login/path values:

```bash
: "${STOKES_LOGIN:?Set verified username and Stokes login hostname}"
: "${REMOTE_HETNET:?Set the immutable remote checkout path}"
: "${LOCAL_RESULTS:?Choose a local results destination}"
rsync -av --progress "$STOKES_LOGIN:$REMOTE_HETNET/runs/" "$LOCAL_RESULTS/runs/"
rsync -av --progress "$STOKES_LOGIN:$REMOTE_HETNET/logs/" "$LOCAL_RESULTS/logs/"
rsync -av --progress "$STOKES_LOGIN:$REMOTE_HETNET/evidence/" "$LOCAL_RESULTS/evidence/"
```

Also copy external output/evidence roots where actually used. Verify hashes after transfer; downloading does not authorize removing cluster copies.

### Newton or offline contingency

Newton and `preemptable` are not automatic fallbacks: launchers require Stokes/normal. A switch needs Akki's decision, new live limits/balance, separate environment/provenance and fresh runtime/calibration evidence. If networking fails, an approved staged wheelhouse can be considered, still installed on a compute node under ARCC policy. No silent login installation, GPU request, or Run 2 script is part of Phase A.
