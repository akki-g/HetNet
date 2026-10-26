#!/bin/bash -l
# Submit with: python scripts/submit_stokes.py --title SLUG --purpose TEXT -- slurm/softrole_capcon_40m.sbatch
# Prepare the admitted research manifest first; see softrole.programme prepare --help.
# Submit from the frozen checkout through scripts/submit_stokes.py.
# Same resources as S1/HetNet and S1 CPU family; twelve 40M runs concurrently.
# Resume a stopped subset with the same run directory and array indices 4,7.
#SBATCH --job-name=capcon-l1-40m
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-11%12
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/scripts/stokes_run.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$HOME/HetNet/.venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export SOFTROLE_PROGRAMME_MANIFEST="${SOFTROLE_PROGRAMME_MANIFEST:-$HETNET_RUN_DIR/manifest.json}"
export SOFTROLE_PROGRAMME_RUN_ROOT="${SOFTROLE_PROGRAMME_RUN_ROOT:-$HETNET_RUN_DIR/training}"

if [[ ! -f $SOFTROLE_PROGRAMME_MANIFEST ]]; then
  printf 'Missing research manifest: %s\n' "$SOFTROLE_PROGRAMME_MANIFEST" >&2
  echo 'Prepare capcon_l1_40m_v1 on this checkout using the existing certified S1 evidence before submitting.' >&2
  echo 'Preparation options: python -m softrole.programme prepare --help' >&2
  exit 2
fi

# Reject the historical or engineering manifest before starting a research task.
"$HETNET_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["SOFTROLE_PROGRAMME_MANIFEST"]).read_text())
if manifest.get("programme") != "capcon_l1_40m_v1" or manifest.get("bounded_check") is not False:
    raise SystemExit("Expected a non-bounded capcon_l1_40m_v1 research manifest")
PY

printf 'Allocated node(s): %s\n' "${SLURM_JOB_NODELIST:-unknown}"
exec srun --cpu-bind=cores bash scripts/softrole_programme.sh \
  "${SLURM_ARRAY_TASK_ID:?Array index required}" --episode-log chunked-gzip "$@"
