#!/usr/bin/env bash
# S1 pilot: current shared CapConNet, two new Adam rates, matched seed coordinates.
set -euo pipefail
usage() {
  echo 'Usage: bash scripts/softrole_lr_pcp.sh INDEX [--dry-run] [runtime options...]' >&2
  echo '0..2=Real/3e-4 seeds0..2; 3..5=Real/1e-3 seeds0..2.' >&2
  echo 'Options: --total-steps, --epochs, --updates-per-epoch, --batch-steps, --nprocesses, --save-every, --diagnostics-every POSITIVE_INTEGER.' >&2
}
if (( $# == 0 )); then usage; exit 2; fi
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
index=$1
shift
[[ $index =~ ^[0-5]$ ]] || { usage; exit 2; }
options=("$@")
dry_run=false
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --total-steps|--epochs|--updates-per-epoch|--batch-steps|--nprocesses|--save-every|--diagnostics-every)
      (( $# >= 2 )) && [[ $2 =~ ^[1-9][0-9]*$ ]] || {
        echo "$1 requires a positive integer" >&2; exit 2;
      }
      shift 2 ;;
    *) echo "Unsupported LR-pilot option: $1" >&2; usage; exit 2 ;;
  esac
done
seed=$((index % 3))
channel=real
if (( index < 3 )); then lr=0.0003; rate=3e-4; else lr=0.001; rate=1e-3; fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${HETNET_PYTHON:-$root/.venv/bin/python}
run_root=${SOFTROLE_LR_PCP_RUN_ROOT:-runs/softrole_lr_pcp_${SLURM_ARRAY_JOB_ID:-manual}}
run=$run_root/pcp_shared/$channel/adam_lr$rate/seed$seed
[[ ! -e $run ]] || { echo "Run directory already exists: $run" >&2; exit 2; }
export PYTHONPATH="$root:$root/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
"$python" -u -m softrole train \
  --task pcp --study fixed --env-version corrected-observation-v1 --model shared --seed "$seed" \
  --optimizer adam --lr "$lr" --num-p 2 --num-a 1 --dim 5 --vision 2 --max-steps 80 \
  --epochs 2000 --updates-per-epoch 10 --batch-steps 500 --nprocesses 4 \
  --total-steps 20000000 --save-every 10 --diagnostics-every 50 \
  --experts 4 --pre-dim 128 --hidden-dim 64 --heads 4 --head-dim 16 --msg-dim 64 \
  --comm-rounds 3 --independent-heads --communication "$channel" \
  --comm-range -1 --gamma 1 --gae-lambda 0.95 --actor-coeff 50 --value-coeff 1 \
  --detach-gap 5 --max-grad-norm 0.75 --failure-prob 0 --failure-window 10 30 \
  --episode-log file --output "$run" "${options[@]}"
if [[ $dry_run == false ]]; then
  "$python" -u -m softrole.lr_diagnostics "$run"
fi
