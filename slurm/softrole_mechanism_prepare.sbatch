#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_mechanism_prepare.sbatch
# Prepare or validate the source-locked study manifest, then preview all 70 entries.
# Submit the training array with an afterok dependency on this preparation job.
#SBATCH --job-name=mechanism-study-prepare
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
(( $# == 0 )) || { echo 'Preparation accepts no arguments; use the documented environment paths' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
manifest=${SOFTROLE_MECHANISM_MANIFEST:-$HETNET_RUN_DIR/manifest.json}
run_root=${SOFTROLE_MECHANISM_RUN_ROOT:-$HETNET_RUN_DIR/training}
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
if [[ ! -e $manifest ]]; then
  srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.mechanism_study prepare --output "$manifest"
fi
# Existing manifests are never rewritten; all 70 previews verify their source lock.
for index in {0..69}; do
  srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.mechanism_study launch \
    --manifest "$manifest" --index "$index" --run-root "$run_root" --dry-run
done
