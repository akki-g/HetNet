#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_graded_pilot.sbatch
# Exploratory G-cap/G-type seed0 pilot: two fixed 2M-step, horizon-80 PCP runs.
# Prepare the source-locked manifest in the titled run directory before submission.
# These resources are an initial allocation, not a measured runtime guarantee.
#SBATCH --job-name=graded-sensing-pilot
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
  echo 'Usage: submit_stokes.py ... -- slurm/softrole_graded_pilot.sbatch [--dry-run]' >&2; exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^[01]$ ]] || { echo 'Expected index 0 or 1' >&2; exit 2; }
export SOFTROLE_GRADED_PILOT_MANIFEST="${SOFTROLE_GRADED_PILOT_MANIFEST:-$HETNET_RUN_DIR/manifest.json}"
[[ -f $SOFTROLE_GRADED_PILOT_MANIFEST ]] || {
  echo "Manifest file not found: $SOFTROLE_GRADED_PILOT_MANIFEST" >&2; exit 2
}
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
run_root=${SOFTROLE_GRADED_PILOT_RUN_ROOT:-$HETNET_RUN_DIR/training}
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.graded_pilot launch \
  --manifest "$SOFTROLE_GRADED_PILOT_MANIFEST" --index "$index" --run-root "$run_root" "$@"
