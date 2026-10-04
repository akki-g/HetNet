#!/bin/bash -l
# Fresh matched FC shared/banked seeds0..2; old FC archives are different environments.
#SBATCH --job-name=softrole-paper-fc
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs_sr/softrole-paper-fc-%A_%a.out
#SBATCH --error=logs_sr/softrole-paper-fc-%A_%a.err
set -euo pipefail
if (( $# )) && [[ "$*" != "--dry-run" ]]; then
  echo "Only optional --dry-run is accepted; scientific settings are fixed." >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
case "${SLURM_ARRAY_TASK_ID:?}" in
  0|1|2) model=shared; seed="$SLURM_ARRAY_TASK_ID" ;;
  3|4|5) model=banked; seed="$((SLURM_ARRAY_TASK_ID - 3))" ;;
  *) echo 'Expected array index0..5' >&2; exit 2 ;;
esac
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole train --task fc --env-version paper-v1 \
  --model "$model" --seed "$seed" --nprocesses 4 --batch-steps 500 --epochs 1400 \
  --updates-per-epoch 10 --total-steps 28000000 --save-every 10 \
  --output "${SOFTROLE_PAPER_RUN_ROOT:-runs/softrole_paper_fc_${SLURM_ARRAY_JOB_ID}}/$model/seed$seed" "$@"
