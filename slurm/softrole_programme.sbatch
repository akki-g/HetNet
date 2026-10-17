#!/bin/bash -l
# Locked CapCom proposal programme: 63 fresh 20M-step PCP runs, twelve concurrent.
# Create logs_sr, export SOFTROLE_PROGRAMME_MANIFEST (prepared on this exact
# checkout), then sbatch from the repository root. Resources follow the S1
# launcher and are unmeasured for the new learner variants.
# For capcon_l1_40m_v1, submit explicitly with --array=0-11%12; the manifest
# carries its 40M budget. The historical array default below is preserved.
#SBATCH --job-name=capcom-programme
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-62%12
#SBATCH --output=logs_sr/capcom-programme-%A_%a.out
#SBATCH --error=logs_sr/capcom-programme-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Expected a nonnegative manifest index without leading zeros' >&2; exit 2; }
if [[ -z ${SOFTROLE_PROGRAMME_MANIFEST:-} ]]; then
  echo 'Set SOFTROLE_PROGRAMME_MANIFEST to the locked programme manifest' >&2; exit 2
fi
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun --cpu-bind=cores bash scripts/softrole_programme.sh "$index" "$@"
