#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_stale_train.sbatch
# Gate only: entropy {0.003, 0.01} x seeds {0, 1, 2}, 2M steps each.
# Use an afterok dependency on the preparation job in the same titled run.
# These resources are unmeasured starting values, not a runtime guarantee.
#SBATCH --job-name=stale-capability-gate
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%2
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
  echo 'Usage: submit_stokes.py ... -- slurm/softrole_stale_train.sbatch [--dry-run]' >&2; exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^[0-5]$ ]] || { echo 'Expected index 0 through 5' >&2; exit 2; }
export SOFTROLE_STALE_MANIFEST="${SOFTROLE_STALE_MANIFEST:-$HETNET_RUN_DIR/manifest.json}"
run_root=${SOFTROLE_STALE_RUN_ROOT:-$HETNET_RUN_DIR/training}
stokes_run_output "$SOFTROLE_STALE_MANIFEST" "$run_root"
[[ -f $SOFTROLE_STALE_MANIFEST ]] || {
  echo "Manifest file not found: $SOFTROLE_STALE_MANIFEST" >&2; exit 2
}
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.stale_capability launch \
  --manifest "$SOFTROLE_STALE_MANIFEST" --index "$index" --run-root "$run_root" "$@"
