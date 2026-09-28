# Full Stokes training sweep: submission, monitoring and expected cost

Prepared 28 September 2026. This restores Stokes as the execution target. **No cluster jobs have been submitted and no Stokes epoch timing has been measured.** The deliverable is the complete Run 1 workflow; frozen evaluation and the SoftRole architecture remain subsequent work.

## What will run

The [committed grid](configs/run1_grid.json) expands to 21 independent policies: 2P1A, 3P3A and 4P6A with five seeds each, plus 3P1A and 2P2A with three seeds each. Every policy uses the original HetNet-Real/A2C recipe: 2,000 epochs, ten updates/epoch, four collector processes, per-collector batch target 500, horizon 80, and a 5×5 map. This is approximately 840–973 million joint environment transitions across the training sweep. Each array element requests four CPU slots; all 21 running together would require 84. No GPU is requested.

The actual Slurm script is [`slurm/run1_train.sbatch`](slurm/run1_train.sbatch). The new [`slurm/submit_run1.sh`](slurm/submit_run1.sh) validates evidence, supplies the calibrated resources and chosen array throttle, and records the resulting job ID. It defaults to a review-only dry-run. It preserves failed attempts and refuses output reuse. The training/model sources and learner are unchanged by this handoff.

## Submission

Use the [runbook](STOKES_RUNBOOK.md) in order: current Gate A → live Stokes preflight → two 20-epoch endpoint calibrations → measured budget review → full array. This sequence comes from the original [task specification §4.3–4.4](research/TASK_SPEC.md), including confirmation of the concrete budget before full submission. The new request restores the full study; it does not supply unknown account limits or measured resource costs.

After those evidence files exist, run from the immutable Stokes checkout in the verified Python 3.12 shell:

```bash
bash slurm/submit_run1.sh \
  --preflight /absolute/path/to/verified-stokes-preflight.json \
  --gate-a "$PWD/evidence/gate_a/full_20260928_stokes_02/gate_a_report.json" \
  --budget-approval /absolute/path/to/confirmed-approval.json \
  --concurrency "$HETNET_ARRAY_CONCURRENCY" --submit
```

Set `HETNET_ARRAY_CONCURRENCY` to verified, currently available capacity from 1 to 21. Omit `--submit` to inspect the exact command first. The helper obtains memory and wall time from the approved calibration budget, creates log directories and invokes `sbatch`. These paths are deliberate placeholders for facts not yet available; neither synthetic calibration nor an assumed account balance is included in the handoff.

Calibration measures the unchanged full per-epoch workload at 2P1A and 4P6A. The budget calculator includes setup, first-epoch excess, p90 subsequent epoch time, allocation-time residual and checkpoint cost; it requests 30% time and 50% memory margins. Intermediate-team costs are modeled from the endpoint envelope, not proved bounds. If the result exceeds the partition time cap, the existing scripts refuse it: true resume is not implemented. If projected costs exceed 4,000 CPU-hours or 10% of the actual remaining balance, the task requires a specific budget decision.

## Learning and progress

```bash
python3 -m hetnet_ext.progress --runs runs/run1_train
python3 -m hetnet_ext.progress --runs runs/run1_train \
  --out "runs/progress/$(date -u +%Y%m%dT%H%M%SZ)"
```

The first command prints all 21 expected runs, including missing/failed runs. The second saves a new snapshot with a CSV summary, complete JSON diagnostics and an SVG of every seed's learning curves. It reads logs only and is suitable for a login node or downloaded results.

Tracked metrics include episode-weighted success rate and episode length, per-agent and P/A mean returns, policy/value losses, actual cumulative joint transitions and episodes, epoch time, throughput, checkpoint timing/size and per-run ETA. Source seeds receive a review flag at epoch 300 using fixed windows 1–50 versus 251–300. Inspect 2–3 source seeds promptly and all five when available. Flat success across seeds requires review before further expensive training; finite losses alone do not establish learning. No automatic statistical verdict or cancellation is performed.

Use `squeue`, `sacct` and `myusage` alongside these logs. A missing run is not assumed queued; an old metrics file is not proof of a hung job. The live ETA is a diagnostic extrapolation, not a resource request. Full details and exact commands are in runbook §6.

## Current Stokes hardware and timing estimate

Akki’s [live terminal output](evidence/stokes/preflight_20260928/user_terminal.txt) now confirms **normal: 153 nodes / 6,928 CPU slots; no partition wall-time cap; account `cenyioha`: 80,000 CPU-hours remaining; user MaxJobs: 250**. This supports planning for all 21 jobs at once subject to effective submission capacity and actual availability. Follow-up output shows an empty user queue, the required Anaconda module and no additional values in the requested account/QoS limit fields; blank quota output still leaves storage headroom unverified. The ten-minute default job time is too short, so the scripts still supply an explicit calibrated time.

The official [Stokes status inventory](https://arcc.ist.ucf.edu/status/stokesNodes.json), fetched 28 September 2026, lists 160 nodes and 7,336 advertised CPU slots spanning Skylake, Cascade Lake, Ice Lake and Sapphire Rapids feature labels. It does not identify the actual CPUs your jobs will receive, physical-core equivalence or your available concurrency. Static official descriptions give different aggregate counts. The [source audit](research/STOKES_RUNTIME_20260928.md) preserves the raw inventory and explains these differences.

There is **no defensible measured full-run forecast yet**. The previous ≈22 seconds/epoch was a one-process sandbox result; this sweep uses four processes and different hardware. The table below is a planning sensitivity calculation, not a confidence interval or a promised runtime range. Actual timing may lie outside it.

| Assumed seconds/epoch | Training hours per policy | Sweep hours if all 21 run together | Sweep hours at 7 concurrent jobs | Training CPU-hours, all 21 |
|---:|---:|---:|---:|---:|
| 30 | 16.7 | 16.7 | 50 | 1,400 |
| 60 | 33.3 | 33.3 | 100 | 2,800 |
| 120 | 66.7 | 66.7 | 200 | 5,600 |

These assume equal durations, continuous availability and no queue delay; they exclude startup, checkpoints, other overhead, calibration, retries and frozen evaluation. The formula is `hours/policy = 2000 × seconds/epoch / 3600`; total training CPU-hours are `21 × 4 × hours/policy`. For equal jobs at concurrency C, ideal wall time is `ceil(21/C) × hours/policy`. Heterogeneous durations and scheduler availability change the actual schedule.

At the middle scenario, the run is roughly **33 hours with 84 CPUs available**, or **100 hours (4.2 days) with 28 CPUs available**, plus the excluded costs. A 30% allocation margin alone would request about 43.3 hours per job before adding setup/checkpoint/residual costs; the live partition cap must permit this. Twenty-epoch calibration at the table's three speeds takes approximately 10, 20 or 40 minutes of epoch work per endpoint, plus installation/other overhead and queue time.

ARCC's [scheduler documentation](https://arcc.ist.ucf.edu/docs/scheduler/) describes a nominal shared faculty allocation of 80,000 CPU-hours/month. That is not the current remaining balance. The user-provided live output resolves the partition cap and user MaxJobs, while effective parent/QoS limits remain to be checked. At the observed 80,000-hour balance, the table uses 1.75%, 3.5% and 7% before overhead; with only the 30% time margin, those become 2.275%, 4.55% and 9.1%. The 4,000-hour threshold remains independent of the balance percentage. Because Stokes is heterogeneous, also compare the CPU models in calibration and full-job provenance; the endpoint envelope and 30% margin do not guarantee coverage of slower nodes or contention.

## Delivery validation

Scientific revision: `084302b`. [Gate A report](GATE_A_REPORT.md) records current full-gate status and the exact source/lock hashes. The full refreshed Gate A passed all eleven checks: 78 selected pytest cases, twelve smoke runs, 36 verified checkpoints, repeated-seed determinism and strict loading into all five compositions. The new submission/progress tools have dedicated tests, including malformed live logs and prevention of duplicate submissions. Historical failed/incomplete engineering evidence remains preserved. A successful Gate A validates engineering contracts; only actual training curves and the later frozen controls can establish experimental findings.

## Copy the prepared checkout to Stokes

The delivery includes `runs/stokes_handoff/frozen-eval-stokes-sweep.bundle`, a Git bundle containing the branch and archived Gate A text evidence. It avoids relying on an unpushed GitHub branch. On this Mac, use the existing verified Stokes login alias rather than guessing a hostname:

```bash
: "${STOKES_LOGIN:?Set your working Stokes username@host or SSH alias}"
scp runs/stokes_handoff/frozen-eval-stokes-sweep.bundle "$STOKES_LOGIN:~/"
```

On Stokes, choose a fresh directory and verify/clone the bundle:

```bash
git clone -b frozen-eval ~/frozen-eval-stokes-sweep.bundle ~/HetNet-stokes-20260928
cd ~/HetNet-stokes-20260928
git status --short
git rev-parse HEAD
```

The local companion `runs/stokes_handoff/manifest.json` records the bundle SHA-256, byte size, delivery commit and scientific hash. Keep this checkout fixed for calibration and training; prepare later frozen-evaluation changes in a separate worktree. Raw local smoke checkpoints are not duplicated in the bundle; their manifest and validated text evidence are included. Actual Stokes results will be separate run artifacts.
