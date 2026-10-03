#!/usr/bin/env bash
# One frozen checkpoint; relative paths are resolved from the repository root.
set -euo pipefail
usage() {
  echo 'Usage: bash scripts/softrole_evaluate.sh CHECKPOINT OUTPUT [evaluate options...]' >&2
  echo 'Example options: --compositions 2,1 --episodes 500 --seed 1700' >&2
  echo 'OUTPUT must be a fresh JSON path. Use --trace only when step traces are needed.' >&2
}
if (( $# == 1 )) && [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
if (( $# < 2 )); then usage; exit 2; fi
checkpoint=$1 output=$2
shift 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
[[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
python=${HETNET_PYTHON:-$root/.venv/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec "$python" -u -m softrole evaluate "$@" --checkpoint "$checkpoint" --output "$output"
