#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --run-dir RUN_DIR -- --dependency=afterok:PREPARE_JOB slurm/softrole_mechanism_eval.sbatch
# Frozen evaluation of one training run's selected checkpoint (index 0-69 = the training index).
# Set SOFTROLE_MECHANISM_TRAINING_RUN to the finished training run container (read only).
# Resources are initial allocations, not measured runtime guarantees.
#SBATCH --job-name=mechanism-eval
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-69%20
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
  echo 'Usage: submit_stokes.py ... -- slurm/softrole_mechanism_eval.sbatch [--dry-run]' >&2; exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^([0-9]|[1-6][0-9])$ ]] || { echo 'Expected index 0 through 69' >&2; exit 2; }
training=${SOFTROLE_MECHANISM_TRAINING_RUN:?Set SOFTROLE_MECHANISM_TRAINING_RUN to the finished training run directory}
[[ -d $training ]] || { echo "Training run directory not found: $training" >&2; exit 2; }
[[ ! $training -ef $HETNET_RUN_DIR ]] || {
  echo 'The evaluation needs its own run directory; the training run is read only' >&2; exit 2
}
eval_dir=${SOFTROLE_MECHANISM_EVAL_DIR:-$HETNET_RUN_DIR/evaluation}
stokes_run_output "$eval_dir"
[[ -f $eval_dir/manifest.json ]] || {
  echo "Evaluation manifest not found: $eval_dir/manifest.json" >&2; exit 2
}
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.mechanism_evaluation launch \
  --manifest "$eval_dir/manifest.json" --index "$index" --training-run "$training" "$@"
