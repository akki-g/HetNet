#!/bin/bash
#SBATCH --job-name=hetnet-run1
#SBATCH --partition=normal
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/run1-%A_%a.out
#SBATCH --error=logs/run1-%A_%a.err
# Explicit calibrated --time/--mem and a verified concurrency limit are required.
set -eo pipefail
if (( $# != 0 )); then
    echo "This wrapper accepts no positional arguments; use documented HETNET_* environment variables." >&2
    exit 2
fi
: "${SLURM_SUBMIT_DIR:?Submit from the HetNet repository root}"
: "${SLURM_ARRAY_TASK_ID:?Submit the 21-task training array}"
: "${HETNET_BUDGET_APPROVAL:?Set the path to explicit user budget confirmation JSON}"
test -r "$HETNET_BUDGET_APPROVAL" || { echo "Budget confirmation file missing." >&2; exit 2; }
cd "$SLURM_SUBMIT_DIR"
source slurm/bootstrap.sh
hetnet_bootstrap
job_args=(--mode train --index "$SLURM_ARRAY_TASK_ID"
          --preflight "$HETNET_STOKES_PREFLIGHT" --gate-a "$HETNET_GATE_A"
          --budget-approval "$HETNET_BUDGET_APPROVAL"
          --requested-memory-gb "$HETNET_MEM_GB" --requested-time-seconds "$HETNET_TIME_SECONDS"
          --array-concurrency "$HETNET_ARRAY_CONCURRENCY")
if [[ -n "${HETNET_OUTPUT_ROOT:-}" ]]; then job_args+=(--output-root "$HETNET_OUTPUT_ROOT"); fi
exec "$HETNET_PYTHON" -u -m hetnet_ext.train_job "${job_args[@]}"
