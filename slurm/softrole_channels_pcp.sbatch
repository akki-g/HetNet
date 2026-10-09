#!/bin/bash -l
# Fresh shared PCP: Binary seeds 0/1/2, then Real seeds 0/1/2.
# Resources are starting requests; the larger communication model is unbenchmarked.
#SBATCH --job-name=softrole-channels-pcp
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs_sr/softrole-channels-pcp-%A_%a.out
#SBATCH --error=logs_sr/softrole-channels-pcp-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^[0-5]$ ]] || { echo 'Expected array index 0..5' >&2; exit 2; }
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun --cpu-bind=cores bash scripts/softrole_channels_pcp.sh "$index" "$@"
