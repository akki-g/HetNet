#!/usr/bin/env bash
# Historical fixed-team shared PCP recipe; only the optimizer changes.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
  echo 'Usage: bash scripts/softrole_adam_pcp.sh {0|1|2} [--dry-run]' >&2
  exit 2
fi
seed=$1
case "$seed" in 0|1|2) ;; *) echo 'Expected seed 0, 1 or 2' >&2; exit 2 ;; esac
shift
if (( $# )) && [[ $1 != --dry-run ]]; then
  echo 'Only --dry-run is accepted; historical scientific settings are fixed.' >&2
  exit 2
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${HETNET_PYTHON:-$root/.venv/bin/python}
run_root=${SOFTROLE_ADAM_PCP_RUN_ROOT:-runs/softrole_adam_pcp_${SLURM_ARRAY_JOB_ID:-manual}}
export PYTHONPATH="$root:$root/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec "$python" -u -m softrole train \
  --task pcp --study fixed --env-version corrected-observation-v1 --model shared --seed "$seed" \
  --optimizer adam --lr 0.0001 --num-p 2 --num-a 1 --dim 5 --vision 2 --max-steps 80 \
  --epochs 2000 --updates-per-epoch 10 --batch-steps 500 --nprocesses 4 --save-every 50 \
  --experts 4 --pre-dim 128 --hidden-dim 64 --heads 4 --head-dim 16 --msg-dim 16 \
  --comm-range -1 --gamma 1 --gae-lambda 0.95 --actor-coeff 50 --value-coeff 1 \
  --detach-gap 5 --max-grad-norm 0.75 --failure-prob 0 --failure-window 10 30 \
  --episode-log file --output "$run_root/pcp_shared/seed$seed" "$@"
