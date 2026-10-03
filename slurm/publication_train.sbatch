#!/bin/bash -l
# Submit from the repository root. Resource limits require compute-node preflight.
#SBATCH --job-name=hetnet-train
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-11%3
#SBATCH --output=logs_1/hetnet-train-%A_%a.out
#SBATCH --error=logs_1/hetnet-train-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun "$HETNET_PYTHON" -u -m publication_reconstruction run-index --index "${SLURM_ARRAY_TASK_ID:?}" --run-root "${PUBLICATION_RUN_ROOT:-runs/hetnet_supplement_v1}"
