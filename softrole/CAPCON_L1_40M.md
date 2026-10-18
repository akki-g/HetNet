# CapCon critic/credit study: protocol and runbook

Profile `capcon_l1_40m_v1`, declared 17 October 2026. This implementation delivers
a locally validated package. The commands for Git publication, Stokes readiness
and research submission below are later operational steps, not records of actions
performed during preparation. Preserve the running HetNet checkout and historical
staging/evidence directories.

The sole active checkout is `/Users/akshatguduru/Desktop/Thesis/HetNet` locally
and the existing `~/HetNet` on Stokes. The former sibling worktrees are backups
under local `.archive/consolidation_20261017/`, not execution directories.
Local admitted S1 evidence now resolves through
`runs/capcon_consolidation_20261017/lr_selection.json`; on Stokes use its existing
location-valid selector. Historical manifests keep their original path records.

## Simple Slurm submission

The dedicated `slurm/softrole_capcon_40m.sbatch` submits all twelve tasks with
up to twelve concurrent, using the past S1 allocation: account `cenyioha`,
partition `normal`, Sapphire Rapids nodes, four CPUs, 16 GB and 48 hours per
task. HetNet used the same resource sizes; S1 explicitly pinned this CPU family,
not a particular hostname.
The allocated node is recorded in each task's stdout.

The manifest is a generated run artifact and is intentionally excluded from
Git. Copying/pulling the source and running `sbatch` does not create it.
After deploying the complete validated source to Stokes, prepare it once from
the existing S1 evidence. This command verifies that evidence, writes a selector
with Stokes-local paths, prepares the twelve-entry manifest and checks every
entry on a compute node. Submission happens only if all preparation succeeds:

```bash
cd "$HOME/HetNet"
(
  set -e
  export HETNET_PYTHON="$PWD/.venv/bin/python"
  export SOFTROLE_PROGRAMME_MANIFEST="$PWD/runs/capcon_l1_40m_20261017/manifest.json"
  export SOFTROLE_PROGRAMME_RUN_ROOT="$PWD/runs/capcon_l1_40m_20261017/training"
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  mkdir -p logs_sr
  srun --account=cenyioha --partition=normal --constraint=sapphirerapids \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:30:00 \
    bash -l -s <<'PREPARE'
set -euo pipefail
cd "$HOME/HetNet"
module load anaconda/anaconda-2024.10
"$HETNET_PYTHON" analysis/capcom_programme_2026-10-12/integration/stage_evidence.py verify \
  --bundle "$PWD/runs/capcom_staging_20261014/s1_evidence" \
  --inventory-sha256 08826cbdedbf2a91dd57b27bedccded186bfd7490396c213c234a6bc3feaa527 \
  --output "$PWD/runs/capcon_l1_40m_20261017/lr_selection.json"
"$HETNET_PYTHON" -m softrole.programme prepare --profile capcon_l1_40m_v1 \
  --lr-selection "$PWD/runs/capcon_l1_40m_20261017/lr_selection.json" \
  --output "$SOFTROLE_PROGRAMME_MANIFEST"
for index in {0..11}; do
  bash scripts/softrole_programme.sh "$index" --dry-run
done
PREPARE
  sbatch slurm/softrole_capcon_40m.sbatch
)
```

The manifest must be prepared after this launcher and source documentation are
in place; the earlier local bounded manifest is validation evidence. The script
uses compressed episode logging and the existing source/configuration checks.
The evidence bundle path is the extracted bundle retained during the earlier
Stokes staging cleanup; no deleted archive is needed. Preparation refuses to
overwrite an existing selector or manifest. If an earlier array was submitted
without a manifest, cancel that array before preparation so its remaining tasks
cannot start while the manifest is being created.
Resume stopped, unfinished indices with the same manifest and run root:

```bash
sbatch --array=4,7 slurm/softrole_capcon_40m.sbatch --resume
```

Indices 4 and 7 are examples. The longer readiness procedure below remains
available; the user's simplified submission route does not require another
readiness job before these manifest checks and training submission.

## Locked experiment

| Indices | Arm | Seeds | Contrast |
|---|---|---|---|
| 0–2 | `c1_real64` | 0, 1, 2 | Current team-critic control |
| 3–5 | `l1b_decomposed_beta1` | 0, 1, 2 | Critic decomposition versus team critic |
| 6–8 | `l1a_decomposed_beta05` | 0, 1, 2 | Partial cross-agent credit versus beta 1 |
| 9–11 | `l1a_decomposed_beta0` | 0, 1, 2 | Own-agent credit versus beta 1 |

All runs use corrected-observation-v1 PCP, fixed 2P1A, a 5×5 map, vision 2,
horizon 80, no sensor failures, a shared actor, Real64 communication, three
rounds, four independent heads and explicit own capabilities. The optimizer is
the certified Adam **0.001**. Four CPU float64 collectors each use one Torch
thread and a 500-step batch floor. Retain ten updates per epoch, a 2,000-epoch
ceiling, checkpoint interval ten epochs and diagnostics interval 50 updates.
The remaining architecture, GAE, loss and clipping settings are inherited
unchanged from the existing C1/L1 protocol.

Each run targets **40,000,000 joint environment steps**: 480M across twelve
runs before complete-update overshoot. The epoch ceiling guarantees a 40M
floor if reached; actual collection commonly reaches the step target earlier.
The trainer already saves when the complete update crosses that target.

The historical `capcom_proposals_2026-10-12` profile remains 63×20M and is the
CLI default. Its exact 20M S1 configuration is unchanged. The new profile
requires existing non-bounded S1 admission selecting 0.001; it does not repeat
S1 training or count S1 runs as additional replicates. A runtime budget override
creates an engineering run and cannot substitute for a research manifest.

## Evaluation and selection

For every run select the first accepted committed checkpoint at or above 40M,
including accepted continuation attempts and excluding abandoned ancestor tails.
Record actual counts and the preceding checkpoint. Resume uses the latest
accepted checkpoint; evaluation uses the first accepted budget-crossing checkpoint.

The `nominal-selection` phase creates exactly one common nominal 2P1A panel:
500 scenarios, panel seed 2704, twelve reports and 6,000 policy episodes.
Weights remain frozen and evaluation uses separate reproducible streams.
No composition-transfer, sensor-failure or intervention panels enter this phase.

For arm a, seed s and scenario e, completion c is the successful episode's step
count, or 80 for an unsuccessful episode. Rank by

\[
Q_a=\frac{1}{3}\sum_{s=0}^{2}\frac{1}{500}\sum_{e=1}^{500}c_{a,s,e}.
\]

Lower is better. Exact rational arithmetic resolves equality; exact ties choose
the first arm in the table above. Success rate and team return are reported
alongside Q and are not additional tie breakers. All twelve reports must validate
before a research configuration is selected. Missing-report summaries and bounded
engineering panels have no scientific selected arm. Selection is a development
decision, not evidence of superiority on the future final comparison panel.

## Local checks

From `HetNet`, using its existing Python environment:

```bash
export HETNET_PYTHON=/Users/akshatguduru/Desktop/Thesis/HetNet/.venv/bin/python
"$HETNET_PYTHON" -m pytest -q
bash -n scripts/softrole_programme.sh slurm/softrole_programme.sbatch \
  slurm/softrole_capcon_readiness.sbatch
uv lock --check
git diff --check
```

The focused readiness driver also accepts `--updates 2 --skip-tests` for a short
local engineering execution. This still uses four collectors, production batch
floor/horizon, all four learner paths, and actual interrupted continuation. Its
reduced status cannot establish full Stokes readiness. The normal driver uses
60 updates per learner, full regressions and a separate interrupted run.

## Later Git deployment and optional Stokes readiness

Finish all source, tests and the additive AGENTS record before freezing a source
identity. Publish the complete intended change from `HetNet` through Git; the old
branch tip alone omits the previously uncommitted integration. Update the existing
Stokes `~/HetNet` to the resulting exact commit after jobs using that checkout
have stopped. Preserve local changes and run artifacts. Do not copy the Mac
virtual environment or create another experimental checkout.

```bash
# On Stokes, after publication and after jobs using this checkout have stopped:
export CAPCON_COMMIT=REPLACE_WITH_VALIDATED_COMMIT
cd "$HOME/HetNet"
git fetch origin
git checkout --detach "$CAPCON_COMMIT"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="$HOME/HetNet/.venv/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_sr runs

export CAPCON_READINESS="$PWD/runs/capcon_readiness_20261017"
sbatch slurm/softrole_capcon_readiness.sbatch \
  --baseline "$HOME/HetNet/runs/capcom_staging_20261014/baseline" \
  --evidence "$HOME/HetNet/runs/capcom_staging_20261014/s1_evidence" \
  --inventory-sha256 08826cbdedbf2a91dd57b27bedccded186bfd7490396c213c234a6bc3feaa527 \
  --output "$CAPCON_READINESS"
```

Confirm these existing extracted paths on Stokes before submission. The baseline
must be clean commit `644f8f17944d01c9d10e3a4fb670cefef32ceb3f`. If pinned pytest
tools are outside the shared environment, pass `--test-tools /absolute/test-tools`;
the driver does not install dependencies or modify the shared environment.

The driver verifies existing admitted evidence and creates a location-valid
selection wrapper without rewriting canonical receipts. No deleted tar archive
is required and no further large evidence upload is needed. Retain the completed
`checks_920409` validator; this focused driver does not rerun its broad 21-arm
validator.

If using this optional readiness route, require a complete passing
`readiness.json`, matching local/deployed source
fingerprints, finite audited four-collector updates, and successful interrupted
continuation with unchanged parent artifacts and exact uninterrupted-reference
model/Adam state. Inspect resource records and scheduler accounting:

```bash
sacct -j READINESS_JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocCPUS
```

The readiness job requests four CPUs, 16 GB and six hours. The training request
starts at four CPUs, 16 GB and 48 hours per task. Timing is bounded early-training
evidence, not a guarantee of whole-run duration; `time` memory measurements do
not prove the simultaneous total over all collectors. Use the recorded measurements
to assess the request before submission. Continuation handles unfinished runs.

## Later final preparation and training

After deploying the validated source, keep the checkout fixed. Any source correction
requires renewed relevant validation and a new manifest. The final preparation
must use a location-valid certified selector. This longer example uses the
selector produced by optional readiness; the simple route above reuses an
existing verified selector.

```bash
export SOFTROLE_PROGRAMME_MANIFEST="$PWD/runs/capcon_l1_40m_20261017/manifest.json"
export SOFTROLE_PROGRAMME_RUN_ROOT="$PWD/runs/capcon_l1_40m_20261017/training"

# Preparation and all twelve source/configuration checks run on a compute node.
srun --account=cenyioha --partition=normal --constraint=sapphirerapids \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:30:00 \
  bash -c '
    set -euo pipefail
    "$HETNET_PYTHON" -m softrole.programme prepare --profile capcon_l1_40m_v1 \
      --lr-selection "$CAPCON_READINESS/lr_selection.json" \
      --output "$SOFTROLE_PROGRAMME_MANIFEST"
    for index in {0..11}; do
      bash scripts/softrole_programme.sh "$index" --dry-run
    done
  '

# Only after every check succeeds:
sbatch --array=0-11%12 --job-name=capcon-l1-40m \
  slurm/softrole_programme.sbatch --episode-log chunked-gzip
```

Do not pass `--total-steps 40000000` to training: the budget is already locked
in the manifest. Preserve the same absolute run root for all continuations.
The historical Slurm file defaults to 0–62; the explicit 0–11 override above is
required for this profile. Store the returned job ID, inspect `squeue`/`sacct`,
and determine completion from accepted checkpoint counts, not scheduler state.

After confirming an unfinished task has stopped, resume only that task:

```bash
sbatch --array=4 --job-name=capcon-l1-resume \
  slurm/softrole_programme.sbatch --resume
```

Task 4 is an example. Keep the manifest, source checkout and run root unchanged;
do not add fresh runtime overrides. Completed runs are recognized from committed
counts even when a final completion receipt was lost.

## Later nominal evaluation

After all twelve runs reach 40M, prepare under a compute allocation using:

```bash
"$HETNET_PYTHON" -m softrole.programme_evaluation prepare \
  --manifest "$SOFTROLE_PROGRAMME_MANIFEST" \
  --phase nominal-selection --run-root "$SOFTROLE_PROGRAMME_RUN_ROOT" \
  --output runs/capcon_l1_40m_20261017/evaluation
```

Submit its generated evaluation array from the frozen checkout, then summarize
under a compute allocation after all tasks complete:

```bash
sbatch runs/capcon_l1_40m_20261017/evaluation/evaluate.sbatch
"$HETNET_PYTHON" -m softrole.programme_evaluation summarize \
  --evaluation runs/capcon_l1_40m_20261017/evaluation \
  --output runs/capcon_l1_40m_20261017/summary
```

The summary retains the declared contrasts and writes `selection.json` with
exact scores, the tie rule, per-seed results and source/panel/checkpoint/report
identities. Do not tune on the later final HetNet or frozen composition/failure
panels. Offline D1–D6 analyses and soft-gated-bank comparisons remain separate.
