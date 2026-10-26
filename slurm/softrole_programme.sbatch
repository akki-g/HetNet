#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_programme.sbatch
# Locked CapCom proposal programme: 63 fresh 20M-step PCP runs, twelve concurrent.
# Export SOFTROLE_PROGRAMME_MANIFEST prepared on this exact checkout.
# Submit from the repository root through the titled run helper. Resources follow the S1
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
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
export SOFTROLE_PROGRAMME_RUN_ROOT="${SOFTROLE_PROGRAMME_RUN_ROOT:-$HETNET_RUN_DIR/training}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
index=${SLURM_ARRAY_TASK_ID:?Array index required}
[[ $index =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Expected a nonnegative manifest index without leading zeros' >&2; exit 2; }
if [[ -z ${SOFTROLE_PROGRAMME_MANIFEST:-} ]]; then
  echo 'Set SOFTROLE_PROGRAMME_MANIFEST to the locked programme manifest' >&2; exit 2
fi
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
exec srun --cpu-bind=cores bash scripts/softrole_programme.sh "$index" "$@"
