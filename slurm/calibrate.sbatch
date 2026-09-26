#!/bin/bash
#SBATCH --job-name=hetnet-calibration
#SBATCH --partition=normal
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/calibrate-%A_%a.out
#SBATCH --error=logs/calibrate-%A_%a.err
# --time and --mem must be supplied explicitly at submission after live preflight.
set -eo pipefail
if (( $# != 0 )); then
    echo "This wrapper accepts no positional arguments; use documented HETNET_* environment variables." >&2
    exit 2
fi
: "${SLURM_SUBMIT_DIR:?Submit from the HetNet repository root}"
: "${SLURM_ARRAY_TASK_ID:?Submit the two-task calibration array}"
cd "$SLURM_SUBMIT_DIR"
source slurm/bootstrap.sh
hetnet_bootstrap
job_args=(--mode calibration --index "$SLURM_ARRAY_TASK_ID"
          --preflight "$HETNET_STOKES_PREFLIGHT" --gate-a "$HETNET_GATE_A"
          --requested-memory-gb "$HETNET_MEM_GB" --requested-time-seconds "$HETNET_TIME_SECONDS"
          --array-concurrency "$HETNET_ARRAY_CONCURRENCY")
if [[ -n "${HETNET_OUTPUT_ROOT:-}" ]]; then job_args+=(--output-root "$HETNET_OUTPUT_ROOT"); fi
exec "$HETNET_PYTHON" -u -m hetnet_ext.train_job "${job_args[@]}"
