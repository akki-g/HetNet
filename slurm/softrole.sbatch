#!/bin/bash -l
# 3 tasks x 2 primary models x 3 screening seeds. No jobs submitted automatically.
#SBATCH --job-name=softrole
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-17%3
#SBATCH --output=logs_sr/softrole-%A_%a.out
#SBATCH --error=logs_sr/softrole-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^[0-9]+$ ]] && (( index <= 17 )) || { echo 'Expected index 0..17' >&2; exit 2; }
tasks=(pp pcp fc)
models=(shared banked)
task=${tasks[index / 6]}
model=${models[(index / 3) % 2]}
seed=$((index % 3))
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun bash scripts/softrole.sh "$task" "$model" "$seed" "$@"
