#!/bin/bash -l
# Submit from the frozen checkout: sbatch slurm/softrole_capcon_40m.sbatch
# Same resources as S1/HetNet and S1 CPU family; twelve 40M runs concurrently.
# Resume a stopped subset: sbatch --array=4,7 slurm/softrole_capcon_40m.sbatch --resume
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
#SBATCH --output=logs_sr/capcon-l1-40m-%A_%a.out
#SBATCH --error=logs_sr/capcon-l1-40m-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$HOME/HetNet/.venv/bin/python}"
export SOFTROLE_PROGRAMME_MANIFEST="${SOFTROLE_PROGRAMME_MANIFEST:-$PWD/runs/capcon_l1_40m_20261017/manifest.json}"
export SOFTROLE_PROGRAMME_RUN_ROOT="${SOFTROLE_PROGRAMME_RUN_ROOT:-$PWD/runs/capcon_l1_40m_20261017/training}"

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
