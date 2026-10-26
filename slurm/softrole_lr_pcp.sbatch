#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_lr_pcp.sbatch
# S1 learning-rate training + diagnostics; shared Real, three seeds/rate, 20M.
# Resources follow the existing Stokes CPU launcher.
#SBATCH --job-name=capcon-lr-pcp
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5%3
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
export SOFTROLE_LR_PCP_RUN_ROOT="${SOFTROLE_LR_PCP_RUN_ROOT:-$HETNET_RUN_DIR/training}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^[0-5]$ ]] || { echo 'Expected array index 0..5' >&2; exit 2; }
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun --cpu-bind=cores bash scripts/softrole_lr_pcp.sh "$index" "$@"
