#!/bin/bash -l
# Prepare or validate the source-locked pilot manifest, then preview both arms.
# Submit the training array with an afterok dependency on this preparation job.
#SBATCH --job-name=graded-sensing-prepare
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs_sr/graded-sensing-prepare-%j.out
#SBATCH --error=logs_sr/graded-sensing-prepare-%j.err
set -euo pipefail
(( $# == 0 )) || { echo 'Preparation accepts no arguments; use the documented environment paths' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
manifest=${SOFTROLE_GRADED_PILOT_MANIFEST:-runs/graded_sensing_pilot_20261024/manifest.json}
run_root=${SOFTROLE_GRADED_PILOT_RUN_ROOT:-runs/graded_sensing_pilot_20261024/training}
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD:$PWD/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
if [[ ! -e $manifest ]]; then
  srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.graded_pilot prepare --output "$manifest"
fi
# Existing manifests are never rewritten; both previews verify their source lock.
for index in 0 1; do
  srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.graded_pilot launch \
    --manifest "$manifest" --index "$index" --run-root "$run_root" --dry-run
done
