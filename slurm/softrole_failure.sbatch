#!/bin/bash -l
# PCP failure only: shared seeds 0/1/2, then banked seeds 0/1/2.
# Resource settings are unmeasured starting points, as in softrole.sbatch.
#SBATCH --job-name=softrole-failure
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs_sr/softrole-failure-%A_%a.out
#SBATCH --error=logs_sr/softrole-failure-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^[0-5]$ ]] || { echo 'Expected index 0..5' >&2; exit 2; }
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun bash scripts/softrole_failure.sh "$index" "$@"
