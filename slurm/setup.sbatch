#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/setup.sbatch
# Install once on a compute node, then reuse .venv for the training array.
#SBATCH --job-name=hetnet-setup
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the HetNet repository root}"
module load anaconda/anaconda-2024.10
exec bash scripts/setup_env.sh
