#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --run-dir RUN_DIR -- --dependency=... slurm/softrole_stale_gate_summary.sbatch
# Summarize the six stability-gate runs into one fresh JSON file. Evidence only:
# it never selects a coefficient. Takes no arguments.
# Resources are unmeasured starting values.
#SBATCH --job-name=stale-capability-gate-summary
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
if (( $# != 0 )); then
  echo 'Usage: submit_stokes.py ... -- slurm/softrole_stale_gate_summary.sbatch (no arguments)' >&2; exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
manifest=${SOFTROLE_STALE_MANIFEST:-$HETNET_RUN_DIR/manifest.json}
run_root=${SOFTROLE_STALE_RUN_ROOT:-$HETNET_RUN_DIR/training}
summary=${SOFTROLE_STALE_SUMMARY:-$HETNET_RUN_DIR/gate_summary.json}
stokes_run_output "$manifest" "$run_root" "$summary"
[[ -f $manifest ]] || { echo "Manifest file not found: $manifest" >&2; exit 2; }
if [[ -e $summary || -L $summary ]]; then
  echo "Summary exists; set a fresh SOFTROLE_STALE_SUMMARY: $summary" >&2; exit 2
fi
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.stale_capability gate-summary \
  --manifest "$manifest" --run-root "$run_root" --output "$summary"
