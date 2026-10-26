#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_diagnostics.sbatch
# Frozen D1-D6 diagnostics only; no training, checkpoint edits, or reserved panels.
# Set the manifest and
# output root explicitly, then override --array=0-(entry_count-1)%3 at submission.
# The 1-CPU / 16G / 12h request is an unmeasured starting point, not a benchmark.
#SBATCH --job-name=capcom-diagnostics
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-0
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
manifest=${CAPCOM_DIAGNOSTICS_MANIFEST:?Set CAPCOM_DIAGNOSTICS_MANIFEST to the locked JSON manifest}
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Expected a nonnegative array index' >&2; exit 2; }
job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?Slurm job ID required}}
run_root=${CAPCOM_DIAGNOSTICS_OUTPUT_ROOT:-$HETNET_RUN_DIR/diagnostics}
# Reject overrides here too: an array entry must keep its manifest and output identity.
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
  echo 'Only --dry-run may be passed to this Slurm script; use the manifest for settings.' >&2
  exit 2
fi
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun --cpu-bind=cores bash scripts/softrole_diagnostics.sh \
  --manifest "$manifest" --index "$index" --output "$run_root/task_$index" "$@"
