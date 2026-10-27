#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_stale_prepare.sbatch
# Prepare the source-locked stability-gate manifest, then preview all six runs.
# Resources are unmeasured starting values. No main-study stage is implemented.
#SBATCH --job-name=stale-capability-prepare
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
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
  echo 'Usage: submit_stokes.py ... -- slurm/softrole_stale_prepare.sbatch [--dry-run]' >&2; exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
manifest=${SOFTROLE_STALE_MANIFEST:-$HETNET_RUN_DIR/manifest.json}
run_root=${SOFTROLE_STALE_RUN_ROOT:-$HETNET_RUN_DIR/training}
stokes_run_output "$manifest" "$run_root"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
if [[ ! -e $manifest && ! -L $manifest ]]; then
  srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.stale_capability prepare \
    --stage gate --output "$manifest" "$@"
  # A preparation dry-run creates no manifest to preview.
  (( $# == 0 )) || exit 0
fi
# Existing manifests are never rewritten; all previews verify their source lock.
for index in 0 1 2 3 4 5; do
  srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.stale_capability launch \
    --manifest "$manifest" --index "$index" --run-root "$run_root" --dry-run
done
