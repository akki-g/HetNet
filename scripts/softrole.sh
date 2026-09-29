#!/usr/bin/env bash
# Same released domain recipes, new deterministic binary actor and team learner.
set -euo pipefail
if (( $# < 3 )); then
  echo 'Usage: bash scripts/softrole.sh {pp|pcp|fc} {shared|banked|capability|constant} SEED [train options...]' >&2
  exit 2
fi
task=$1 model=$2 seed=$3
shift 3
case "$task" in pp|pcp|fc) ;; *) echo 'Unknown task' >&2; exit 2 ;; esac
case "$model" in shared|banked|capability|constant) ;; *) echo 'Unknown model' >&2; exit 2 ;; esac
[[ $seed =~ ^[0-9]+$ ]] || { echo 'SEED must be nonnegative' >&2; exit 2; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${HETNET_PYTHON:-$root/.venv/bin/python}
run=${SOFTROLE_RUN_ROOT:-runs/softrole}/${task}_${model}/seed${seed}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec "$python" -u -m softrole train --task "$task" --model "$model" --seed "$seed" "$@" --output "$run"
