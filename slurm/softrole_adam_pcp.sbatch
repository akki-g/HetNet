#!/bin/bash -l
# Three fresh shared PCP seeds; compare with archived RMSprop history.
#SBATCH --job-name=softrole-adam-pcp
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs_sr/softrole-adam-pcp-%A_%a.out
#SBATCH --error=logs_sr/softrole-adam-pcp-%A_%a.err
set -euo pipefail
if (( $# )) && [[ "$*" != "--dry-run" ]]; then
  echo 'Only optional --dry-run is accepted; scientific settings are fixed.' >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the checkout root}"
case "${SLURM_ARRAY_TASK_ID:?Array index required}" in
  0|1|2) seed=$SLURM_ARRAY_TASK_ID ;;
  *) echo 'Expected array index 0..2' >&2; exit 2 ;;
esac
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun --cpu-bind=cores bash scripts/softrole_adam_pcp.sh "$seed" "$@"
