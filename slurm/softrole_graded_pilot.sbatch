#!/bin/bash -l
# Exploratory G-cap/G-type seed0 pilot: two fixed 2M-step, horizon-80 PCP runs.
# Prepare the source-locked manifest and create logs_sr before submission.
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
#SBATCH --output=logs_sr/graded-sensing-pilot-%A_%a.out
#SBATCH --error=logs_sr/graded-sensing-pilot-%A_%a.err
set -euo pipefail
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
  echo 'Usage: sbatch slurm/softrole_graded_pilot.sbatch [--dry-run]' >&2; exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^[01]$ ]] || { echo 'Expected index 0 or 1' >&2; exit 2; }
export SOFTROLE_GRADED_PILOT_MANIFEST="${SOFTROLE_GRADED_PILOT_MANIFEST:-runs/graded_sensing_pilot_20261024/manifest.json}"
[[ -f $SOFTROLE_GRADED_PILOT_MANIFEST ]] || {
  echo "Manifest file not found: $SOFTROLE_GRADED_PILOT_MANIFEST" >&2; exit 2
}
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
run_root=${SOFTROLE_GRADED_PILOT_RUN_ROOT:-runs/graded_sensing_pilot_20261024/training}
export PYTHONPATH="$PWD:$PWD/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.graded_pilot launch \
  --manifest "$SOFTROLE_GRADED_PILOT_MANIFEST" --index "$index" --run-root "$run_root" "$@"
