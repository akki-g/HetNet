#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --run-dir RUN_DIR -- slurm/softrole_mechanism_eval_prepare.sbatch [--allow-missing]
# Select the first committed checkpoint at the target for every training run, write the frozen panels and the
# source-locked evaluation manifest, then preview all 70 entries. Set SOFTROLE_MECHANISM_TRAINING_RUN to the
# finished training run container (read only). Submit the evaluation array with an afterok dependency on this job.
# A study with fewer entries (the 10-run supplement) sets SOFTROLE_MECHANISM_EVAL_COUNT to its entry count.
# Resources are initial allocations, not measured runtime guarantees.
#SBATCH --job-name=mechanism-eval-prepare
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --allow-missing ]]; }; then
  echo 'Usage: submit_stokes.py ... -- slurm/softrole_mechanism_eval_prepare.sbatch [--allow-missing]' >&2; exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
training=${SOFTROLE_MECHANISM_TRAINING_RUN:?Set SOFTROLE_MECHANISM_TRAINING_RUN to the finished training run directory}
[[ -d $training ]] || { echo "Training run directory not found: $training" >&2; exit 2; }
count=${SOFTROLE_MECHANISM_EVAL_COUNT:-70}
[[ $count =~ ^[1-9][0-9]?$ ]] && (( count <= 70 )) || {
  echo 'SOFTROLE_MECHANISM_EVAL_COUNT must be an entry count from 1 through 70' >&2; exit 2
}
[[ ! $training -ef $HETNET_RUN_DIR ]] || {
  echo 'The evaluation needs its own run directory; the training run is read only' >&2; exit 2
}
eval_dir=${SOFTROLE_MECHANISM_EVAL_DIR:-$HETNET_RUN_DIR/evaluation}
stokes_run_output "$eval_dir"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
if [[ ! -e $eval_dir && ! -L $eval_dir ]]; then
  srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.mechanism_evaluation prepare \
    --training-run "$training" --output "$eval_dir" "$@"
fi
# An existing evaluation directory is never rewritten; the previews of every entry verify its source lock,
# checkpoint hashes and panels.
for (( index = 0; index < count; index++ )); do
  srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.mechanism_evaluation launch \
    --manifest "$eval_dir/manifest.json" --index "$index" --training-run "$training" --dry-run
done
