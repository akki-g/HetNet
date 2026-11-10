#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_mechanism_train.sbatch
# Capability-mechanism study training: 2 conditions x 7 arms x 5 seeds, 10M-step, horizon-80 PCP runs.
# Prepare the source-locked manifest in the titled run directory before submission.
# These resources are an initial allocation, not a measured runtime guarantee.
#SBATCH --job-name=mechanism-study
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-69%10
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
  echo 'Usage: submit_stokes.py ... -- slurm/softrole_mechanism_train.sbatch [--dry-run]' >&2; exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^([0-9]|[1-6][0-9])$ ]] || { echo 'Expected index 0 through 69' >&2; exit 2; }
export SOFTROLE_MECHANISM_MANIFEST="${SOFTROLE_MECHANISM_MANIFEST:-$HETNET_RUN_DIR/manifest.json}"
run_root=${SOFTROLE_MECHANISM_RUN_ROOT:-$HETNET_RUN_DIR/training}
stokes_run_output "$SOFTROLE_MECHANISM_MANIFEST" "$run_root"
[[ -f $SOFTROLE_MECHANISM_MANIFEST ]] || {
  echo "Manifest file not found: $SOFTROLE_MECHANISM_MANIFEST" >&2; exit 2
}
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.mechanism_study launch \
  --manifest "$SOFTROLE_MECHANISM_MANIFEST" --index "$index" --run-root "$run_root" "$@"
