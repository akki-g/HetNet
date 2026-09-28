#!/bin/bash -l
# Install once on a compute node, then reuse .venv for the training array.
#SBATCH --job-name=hetnet-setup
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the HetNet repository root}"
module load anaconda/anaconda-2024.10
exec bash scripts/setup_env.sh
