#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/reproduce.sbatch
# First tranche: the authors' three README Real recipes x three seeds.
# Binary remains available in scripts/reproduce.sh for the published PCP ablations.
# These resource defaults are starting requests, not measured runtime bounds.
# Override resource and array options through the submission helper.
#SBATCH --job-name=hetnet-repro
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
export HETNET_RUN_ROOT="${HETNET_RUN_ROOT:-$HETNET_RUN_DIR/training}"
cd "${SLURM_SUBMIT_DIR:?Submit from the HetNet repository root}"
module load anaconda/anaconda-2024.10
index=${SLURM_ARRAY_TASK_ID:?This script requires an array index}
[[ $index =~ ^[0-9]+$ ]] && (( index <= 8 )) || { echo 'Expected array index 0..8' >&2; exit 2; }
tasks=(pp pcp fc)
task=${tasks[index / 3]}
seed=$((index % 3))
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun bash scripts/reproduce.sh "$task" real "$seed" "$@"
