#!/bin/bash -l
# Focused bounded checks only; forward explicit evidence/baseline/output arguments.
#SBATCH --job-name=capcon-readiness
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=capcon-readiness-%j.out
#SBATCH --error=capcon-readiness-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the revised checkout}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$HOME/HetNet/.venv/bin/python}"
[[ -x $HETNET_PYTHON ]] || { echo 'Set HETNET_PYTHON to the existing pinned interpreter' >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/envs"
# Short paths avoid Unix multiprocessing socket-length limits in test fixtures.
export TMPDIR
TMPDIR=$(mktemp -d "/tmp/capcon-${SLURM_JOB_ID:?}.XXXXXX")
# Preserve this private directory for failure diagnosis; its path is recorded.
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u scripts/softrole_capcon_readiness.py --checkout "$PWD" "$@"
