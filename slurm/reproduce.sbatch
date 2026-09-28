#!/usr/bin/env bash
# First release-reproduction tranche: 3 tasks x 2 variants x 3 seeds.
# These resource defaults are starting requests, not measured runtime bounds.
# Override with ordinary sbatch --mem / --time / --array options as needed.
#SBATCH --job-name=hetnet-repro
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-17%6
#SBATCH --output=logs/reproduce-%A_%a.out
#SBATCH --error=logs/reproduce-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the HetNet repository root}"
index=${SLURM_ARRAY_TASK_ID:?This script requires an array index}
[[ $index =~ ^[0-9]+$ ]] && (( index <= 17 )) || { echo 'Expected array index 0..17' >&2; exit 2; }
tasks=(pp pcp fc)
variants=(real binary)
task=${tasks[index / 6]}
variant=${variants[(index / 3) % 2]}
seed=$((index % 3))
export HETNET_PYTHON="$PWD/.venv/bin/python"
exec srun bash scripts/reproduce.sh "$task" "$variant" "$seed" "$@"
