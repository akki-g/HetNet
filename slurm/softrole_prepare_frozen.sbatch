#!/bin/bash -l
# Validate checkpoints and prepare the frozen PCP array on a compute node.
# Memory/time requests are unmeasured starting values; override with sbatch.
# Submit from the repository root with prepare_pcp_frozen.py CLI arguments.
#SBATCH --job-name=softrole-prepare-frozen
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs_sr/pcp-prepare-%j.out
#SBATCH --error=logs_sr/pcp-prepare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun "$HETNET_PYTHON" -u scripts/prepare_pcp_frozen.py "$@"
