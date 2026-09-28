# Stokes runbook: original HetNet Phase A

This prepares original HetNet-Real/A2C PCP training. No cluster jobs have been submitted by this work. The refreshed 28 September Gate A passed all eleven checks, including 78 pytest cases and twelve bounded smoke runs. See [GATE_A_REPORT.md](GATE_A_REPORT.md). Calibration still requires verified live Stokes facts. Full Run 1 additionally requires Akki's confirmation of the measured budget. Run 2 is a later phase, with no submission script here.

**Current target: Stokes, restored by Akki on 28 September 2026.** Run the full 21-task training grid. The Mac pilot plan is retained as historical fallback documentation. The refreshed submission/progress tooling is in scientific revision `084302b`; use matching current Gate A evidence, never the older snapshot's pass. The [current hardware and timing audit](research/STOKES_RUNTIME_20260928.md) gives conditional costs and identifies the remaining live facts.

The submission entry point is [`slurm/submit_run1.sh`](slurm/submit_run1.sh), which validates and calls [`slurm/run1_train.sbatch`](slurm/run1_train.sbatch). Section 6 gives the exact commands. It derives resources from the measured, confirmed budget and requires an explicit concurrency value.
Akki’s [live terminal evidence](evidence/stokes/preflight_20260928/user_terminal.txt) confirms `normal` has 153 nodes / 6,928 CPU slots and unlimited partition wall time; account `cenyioha` had all 80,000 September CPU-hours remaining and user MaxJobs=250. The [partial preflight](evidence/stokes/preflight_20260928/preflight_draft.json) remains unverified until inherited limits, occupancy, runtime/network and quota are resolved. No fake finite cap is substituted.

Official sources, wheel evidence, and live unknowns are documented in [STOKES_SOURCE_AUDIT.md](research/STOKES_SOURCE_AUDIT.md). The public pages alone did not establish these live facts; the supplied terminal evidence resolves the partition cap, associated account and balance, while effective limits and compute prerequisites still need verification. The existing Newton wrapper's resources are not Stokes measurements.

## 1. Freeze the scientific checkout

Use a dedicated Stokes checkout of the delivery commit containing this runbook and the current Gate A archive. Its scientific files must match `084302b`; a later documentation/evidence-only descendant is equivalent. Record its full `git rev-parse HEAD`. Preserve the checkout, `uv.lock`, and `.venv-stokes` while calibration or Run 1 tasks are queued/running. Do not pull new code into it. Develop Phase B in a **separate checkout/worktree with its own environment**: extension, test, config, or script changes invalidate queued jobs' Gate A hash.
The inventory hashes tracked Python/shell sources, `pyproject.toml`, `uv.lock`, and study source/config/test/script files, including new untracked study files. Documentation/results are excluded so an evidence-only commit does not invalidate equivalent code. Every run still records HEAD, dirty status, the code inventory and lock hash. Gate A separately verifies all reviewed deviations from the original baseline.

The reviewed original-file commits remain `2fadecf`, `7b334c1`, `0cfcea5`, and `50d0c37`. Do not expand that allowlist automatically. The current passing attempt is `runs/gate_a/full_20260928_stokes_02`, archived at `evidence/gate_a/full_20260928_stokes_02/gate_a_report.json`. See [GATE_A_REPORT.md](GATE_A_REPORT.md) for its status and source hash. Historical attempts, including the successful 26 September snapshot and interrupted first attempt, remain preserved.
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

Use a Bash shell for the command blocks in this runbook (run `bash` first if your login shell differs). Preserve complete timestamped raw output on Stokes:

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

The supplied user and account association listings were not requested with `WOPLimits`, so inherited association Max limits are included under [Slurm's documented listing semantics](https://slurm.schedmd.com/sacctmgr.html). Remaining aggregate submission constraints can be inspected once when finalizing preflight:

```bash
sacctmgr -nP show qos where name=normal \
  format=Name,MaxJobsPU,MaxJobsPA,MaxSubmitJobsPU,MaxSubmitJobsPA,GrpJobs,GrpSubmit,GrpTRES,MaxWall
sacctmgr -nP show assoc where account=root \
  format=Cluster,Account,User,GrpJobs,GrpSubmit,GrpTRES,MaxWall
lfs quota -h -u "$USER" /lustre/fs1
df -h "$PWD"
```

Blank limit fields mean unset at that scope; distinguish an unset per-user MaxSubmit from an aggregate GrpSubmit constraint. If the aggregate checks show no applicable submission restriction, record `max_submit_jobs="unlimited"`. The known MaxJobs=250 permits the planned concurrency of 21 at the user level; queue availability is still not guaranteed.

After Gate A and scheduler checks, use a short verified compute allocation to check Python 3.12 and actual artifact access to PyPI/wheel hosts, GitHub releases, and `astral.sh`. Capture `module list`, interpreter version, `uname -a`, `/etc/os-release`, `lscpu`, `getconf GNU_LIBC_VERSION`, `pwd`, and `df -h .`. The [ARCC submission guide](https://arcc.ist.ucf.edu/docs/scheduler/scripts/) gives interactive allocation examples; their memory/time values are not HetNet requirements. Unresolved limits, Python, network or account identity require Akki's decision; do not silently switch module/cluster/partition or install packages on a login node.

After the current local gate passes, this **lightweight compute probe** checks the module interpreter and actual artifact endpoints; it does not install dependencies or train. The one-CPU, 20-minute request is a provisional probe ceiling. Run from the checkout in Bash and preserve the output:

```bash
mkdir -p evidence/stokes/live
set -o pipefail
srun --account=cenyioha --partition=normal --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=2G --time=00:20:00 bash -l -s <<'BASH' \
  | tee "evidence/stokes/live/compute-probe-$(date -u +%Y%m%dT%H%M%SZ).log"
set -euo pipefail
date -u
hostname
lscpu
module purge
module load anaconda/anaconda-2024.10
conda activate base
module list
python3 - <<'PYPROBE'
import json, platform, sys, urllib.request
assert sys.version_info[:2] == (3, 12), sys.version
print('Python:', sys.version, sys.executable)
print('Platform:', platform.platform())
urls = ['https://astral.sh/uv/0.12.5/install.sh',
        'https://github.com/astral-sh/uv/releases/download/0.12.5/uv-x86_64-unknown-linux-gnu.tar.gz']
for package, version in [('torch', '2.2.1'), ('dgl', '2.1.0')]:
    endpoint = f'https://pypi.org/pypi/{package}/{version}/json'
    with urllib.request.urlopen(endpoint, timeout=30) as response:
        metadata = json.load(response)
    files = [f for f in metadata['urls'] if 'cp312' in f['filename']
             and 'manylinux' in f['filename'] and 'x86_64' in f['filename']]
    assert len(files) == 1, [f['filename'] for f in files]
    urls.append(files[0]['url'])
for url in urls:
    # Request headers only; avoids downloading large wheels for a connectivity check.
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        print('Artifact HEAD:', response.status, url)
        assert 200 <= response.status < 300
print('Compute Python/artifact-connectivity probe passed; locked imports are checked by bootstrap before training.')
PYPROBE
BASH
```

If the probe fails, preserve its output; do not mark the preflight verified. Bootstrap separately performs the real locked install and pinned imports inside the calibration allocation, before starting training. HEAD connectivity does not prove every subsequent download will succeed. Quota and scheduler capacity checks above still apply.

Create verified preflight JSON. This incomplete template intentionally fails until its unknowns are replaced with evidence:

```json
{
  "schema_version": 1,
  "verified": false,
  "cluster": "stokes",
  "partition": "normal",
  "max_wall_time_seconds": "unlimited",
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

A verified unlimited partition/effective time cap is encoded as the literal string `"unlimited"`; finite caps must be positive seconds. Each job still requests a finite calibrated time. Job-count limits require positive integers or the explicit evidence-backed string `"unlimited"`. `account_required` must be a boolean and a required account must be named. Set `verified=true`, Python `"3.12"`, and network verification true only after checks pass. Preserve referenced raw evidence; recheck balance and occupancy before full submission.

## 4. Bootstrap and endpoint calibration

Wrappers request `normal`, one node/task, four CPUs and no GPU. They have **no default memory or wall time**. All installs occur inside a compute allocation, following [ARCC Anaconda guidance](https://arcc.ist.ucf.edu/docs/software/anaconda/). Bootstrap clears Python overrides, loads `anaconda/anaconda-2024.10`, activates base, asserts Python 3.12, then validates preflight/Gate A before installation. An exclusive `.tools/stokes-setup.lock` serializes uv 0.12.5 setup and `uv sync --locked --no-python-downloads` into `.venv-stokes`. It verifies imported package versions. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `HETNET_TORCH_THREADS` are one; the CPU patch explicitly sets Torch threads to one too.

A concrete initial calibration request is **16 GiB, two hours, four CPUs per endpoint**, with at most two endpoints concurrent: a ceiling of 16 allocated CPU-hours if both use their complete time. These are conservative provisional allocation choices, **not measured peak memory or runtime**. They leave the full per-epoch recipe unchanged; if either allocation fails, preserve it and review resources. Full training resources come only from successful calibration. Choose and record calibration memory/time from local resource evidence and the live cap, with a documented margin. A local smoke or documentation example is not a guaranteed Stokes bound. In the submitting shell, use absolute compute-readable evidence paths:

```bash
export HETNET_STOKES_PREFLIGHT=/absolute/path/to/verified-stokes-preflight.json
export HETNET_GATE_A="$PWD/evidence/gate_a/full_20260928_stokes_02/gate_a_report.json"
export HETNET_MEM_GB=16
export HETNET_TIME_SECONDS=7200
export HETNET_ARRAY_CONCURRENCY=2
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
# Prevent inherited sbatch environment options from adding or overriding resources.
for stokes_option in "${!SBATCH_@}"; do unset "$stokes_option"; done
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

Stokes has heterogeneous CPU generations. Record calibration hardware and compare it with full-job provenance: the endpoint maximum and 1.3× time margin are planning assumptions, not bounds on slower nodes, contention, or later runtime changes. Reassess calibration and budget if observed timings invalidate those assumptions.

One epoch contains ten updates. Under the original whole-episode collection bound, each full run collects 40,000,000–46,320,000 joint transitions, not 4,000,000. The report uses actual fresh step totals for throughput; the derivation is in the source audit.

The model is `wall = setup + max(0, first_epoch - p90_epoch) + 2000 × (p90_epoch + residual/20) + 40 × measured_save`. Requested time is 1.3 times that estimate rounded up to a complete minute; memory is 1.5 times measured whole-job peak rounded up to GiB. Guarded expected cost sums all 21 seeds at four CPUs, adds actual calibration core-hours, and adds explicit reserves. The single array requests the maximum time/memory across rows. JSON separately reports its full uniform reservation envelope: requested maximum and actual elapsed usage are different quantities.

Review every resource row, unmeasured-roster assumptions, total core-hours and percentage of a fresh remaining balance. Above **4,000 core-hours or 10% of remaining balance**, stop for Akki's explicit threshold decision. Above the partition cap, stop for a resume/recipe decision. No true-resume implementation exists here and epochs are not silently reduced.

The tool writes an **unapproved** `approval_template.json` tied to the budget file's SHA-256. Only after Akki confirms the concrete report should an operator copy it to an approval record, fill `approved=true`, approving identity and UTC timestamp, and, if explicitly authorized, `threshold_override_approved=true`. Editing the budget afterwards invalidates that record. Approval is also bound to the current preflight file hash and balance: a refreshed preflight needs a fresh matching budget, preventing approval against an older, larger balance. The record documents the user's decision; it cannot supply permission by itself.

## 6. Submit and monitor Run 1

Refresh balance/occupancy before submission; if material facts/costs changed, regenerate and reconfirm the budget. Verify 21-task submission capacity, available concurrency, disk/quota, passing gate, and unchanged scientific hash. In the verified Python 3.12 shell, from the immutable checkout:

```bash
export HETNET_STOKES_PREFLIGHT=/absolute/path/to/verified-stokes-preflight.json
export HETNET_GATE_A="$PWD/evidence/gate_a/full_20260928_stokes_02/gate_a_report.json"
export HETNET_BUDGET_APPROVAL=/absolute/path/to/confirmed-approval.json
# Planned full parallelism: user MaxJobs=250 and the supplied queue is empty.
# Recheck effective limits/occupancy first; lower this if needed.
export HETNET_ARRAY_CONCURRENCY=21

# Review: validates evidence and prints the exact sbatch command; no submission.
bash slurm/submit_run1.sh \
  --preflight "$HETNET_STOKES_PREFLIGHT" --gate-a "$HETNET_GATE_A" \
  --budget-approval "$HETNET_BUDGET_APPROVAL" \
  --concurrency "$HETNET_ARRAY_CONCURRENCY"

# Submit the reviewed 21-task array.
bash slurm/submit_run1.sh \
  --preflight "$HETNET_STOKES_PREFLIGHT" --gate-a "$HETNET_GATE_A" \
  --budget-approval "$HETNET_BUDGET_APPROVAL" \
  --concurrency "$HETNET_ARRAY_CONCURRENCY" --submit
```

The driver supplies time, memory, account if required, `--array=0-20%N`, one node/task, four CPUs and the correct environment to `run1_train.sbatch`. It clears inherited `SBATCH_*` overrides, checks the connected Slurm cluster, creates `logs/` before submission and preserves the command, evidence hashes and returned job ID in `runs/run1_train/run1_submission.json`. No GPU is requested. Its default is a dry-run; explicit `--submit` invokes Slurm. A failed/ambiguous submission record is preserved and requires queue inspection before retrying. Do not delete it to bypass duplicate protection.

Default output is `runs/run1_train/<composition>/seed<seed>/`; `--output-root` selects a fresh alternative. This explicit option overrides any stale calibration environment variable. The existing nested checkpoint path is typically `checkpoints/<composition>_s<seed>/run1/model_ep2000.pt`; use provenance's `final_checkpoint` field. No automatic resume, overwrite or Run 2 chaining is provided.

Use the new **stdlib-only, read-only** progress command on the login node or on downloaded logs. It performs no training and loads no checkpoints:

```bash
python3 -m hetnet_ext.progress --runs runs/run1_train
# Optional terminal refresh, if watch is available:
watch -n 60 'python3 -m hetnet_ext.progress --runs runs/run1_train'
# Preserve a new snapshot for sharing; never overwrites an existing snapshot:
python3 -m hetnet_ext.progress --runs runs/run1_train \
  --out "runs/progress/$(date -u +%Y%m%dT%H%M%SZ)"
```

Each snapshot contains `progress.txt`, `report.json`, `summary.csv`, and a standalone `learning_curves.svg` with every seed's success, capped episode length and P/A rewards. No pooled-seed curve hides failures. Per-epoch evidence already includes policy/value loss, actual joint steps and episodes, cumulative counts, and training time; checkpoint evidence records save time, bytes and signatures. The JSON report adds throughput, most recent checkpoint, metrics age, loss diagnostics, per-run ETA and warnings.

Success, steps and class rewards use **episode-weighted** means within each seed over the last 50 completed epochs. P/A rewards first average agents of that class. Losses retain the recorder's upstream reporting denominator and are diagnostics, not an independent test that the policy learns. Curves use nonoverlapping 50-epoch blocks, with a final partial block. The reader ignores only an unfinished final JSONL line and reports malformed committed records; missing runs are not assumed queued, and stale logs alone do not prove a stall. ETA uses recent median epoch timing plus observed checkpoint cost and omits queue/startup, repeated signature/recorder overhead and future slowdowns. Allocation sizing uses calibration, not this live ETA.

At source epoch 300, the monitor reports each seed's success difference between epochs **251–300 and 1–50**. There is no recorded epoch-zero baseline. A prominent review becomes due once at least two source seeds reach 300; inspect 2–3 seeds promptly, and all five as they become available. If success stays flat across source seeds, preserve logs and contact Akki before consuming further budget, as required by task §4.4/§9. The monitor neither declares competence from a threshold nor automatically cancels jobs. Do not continue a clearly failed learning run solely because losses are finite. At assumed 30/60/120 seconds per epoch this checkpoint occurs around 2.5/5/10 hours after training starts, plus overhead—not a guaranteed 2–3 hours.

Compare diagnostics with the scheduler and balance:

```bash
: "${RUN1_ARRAY_ID:?Set the job ID printed by submit_run1.sh}"
squeue -j "$RUN1_ARRAY_ID"
sacct -j "$RUN1_ARRAY_ID" --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,TotalCPU
myusage
```

Never delete failed output to make a retry appear original. Preserve the attempt/job ID and reason, obtain the required decision, use a fresh output root and target only affected indices with the underlying wrapper; the full-sweep helper deliberately submits all 21 tasks and is not a selective retry tool. Run 2 requires its later implementation and Gate B; successful training alone does not answer frozen transfer or justify a SoftRole claim.

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
