#!/bin/bash -l
# One frozen evaluation, without moving checkpoint files off Stokes.
# Memory/time requests are unmeasured starting values; override with sbatch.
#SBATCH --job-name=softrole-evaluate
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs_sr/softrole-evaluate-%j.out
#SBATCH --error=logs_sr/softrole-evaluate-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun bash scripts/softrole_evaluate.sh "$@"
