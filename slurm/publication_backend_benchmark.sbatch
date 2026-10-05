#!/bin/bash -l
# One allocation, sequential paired runs. Submit only after local correctness gates.
#SBATCH --job-name=hetnet-backends
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs_1/hetnet-backends-%j.out
#SBATCH --error=logs_1/hetnet-backends-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m publication_reconstruction benchmark \
  --output "${PUBLICATION_BENCHMARK_ROOT:-runs/hetnet_backend_benchmark_${SLURM_JOB_ID:?}}" "$@"
