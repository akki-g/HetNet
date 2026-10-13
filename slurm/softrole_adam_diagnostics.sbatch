#!/bin/bash -l
# Submit from the repository root. Prepare and validate both Adam checkpoint
# stages on this allocation, then submit their Sapphire Rapids diagnostic arrays.
# Preparation logs use the repository root so no log directory is needed first.
#SBATCH --job-name=capcom-adam-prepare
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=capcom-adam-prepare-%j.out
#SBATCH --error=capcom-adam-prepare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
job_id=${SLURM_JOB_ID:?Submit this script with sbatch}
[[ $job_id =~ ^[1-9][0-9]*$ ]] || { echo 'Expected a positive Slurm job ID' >&2; exit 2; }
(( $# == 0 )) || {
  echo 'No arguments accepted; use CAPCOM_ADAM_RUN_ROOT and CAPCOM_ADAM_OUTPUT_ROOT for paths.' >&2
  exit 2
}
run_root=${CAPCOM_ADAM_RUN_ROOT:-runs/softrole_adam_pcp_911488}
output_root=${CAPCOM_ADAM_OUTPUT_ROOT:-runs/capcom_adam_diagnostics_$job_id}
module load anaconda/anaconda-2024.10
export PYTHON="${HETNET_PYTHON:-${PYTHON:-$PWD/.venv/bin/python}}"
export HETNET_PYTHON="$PYTHON"
echo "Results root: $output_root"
exec srun --cpu-bind=cores bash scripts/softrole_adam_diagnostics.sh \
  --run-root "$run_root" --output-root "$output_root" --submit
