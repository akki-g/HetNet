#!/usr/bin/env bash
# Matched native PCP communication experiment: Binary seeds 0/1/2, then Real 0/1/2.
set -euo pipefail
usage() {
  echo 'Usage: bash scripts/softrole_channels_pcp.sh INDEX [runtime options...]' >&2
  echo 'INDEX: 0=binary/0 1=binary/1 2=binary/2 3=real/0 4=real/1 5=real/2' >&2
  echo 'Options: --dry-run; --episode-log file|stdout; --epochs, --updates-per-epoch, --batch-steps, --nprocesses, --total-steps, --save-every POSITIVE_INTEGER' >&2
  echo 'Output root: SOFTROLE_CHANNELS_PCP_RUN_ROOT (default runs/softrole_channels_pcp_${SLURM_ARRAY_JOB_ID:-manual}).' >&2
}
if (( $# == 0 )); then usage; exit 2; fi
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
index=$1
shift
[[ $index =~ ^[0-5]$ ]] || { echo 'Expected index 0..5' >&2; exit 2; }

# Keep architecture, task, learner and seed fixed across the six-job study.
# Runtime overrides support small execution checks; they are recorded in config.
validate_options() {
  while (( $# )); do
    case "$1" in
      --dry-run) shift ;;
      --episode-log)
        (( $# >= 2 )) && [[ $2 == file || $2 == stdout ]] || {
          echo '--episode-log needs file or stdout' >&2; exit 2;
        }
        shift 2 ;;
      --epochs|--updates-per-epoch|--batch-steps|--nprocesses|--total-steps|--save-every)
        (( $# >= 2 )) && [[ $2 =~ ^[1-9][0-9]*$ ]] || {
          echo "$1 needs a positive integer value" >&2; exit 2;
        }
        shift 2 ;;
      *) echo "Unsupported communication-launch option: $1" >&2; usage; exit 2 ;;
    esac
  done
}
validate_options "$@"
channels=(binary binary binary real real real)
seeds=(0 1 2 0 1 2)
channel=${channels[index]}
seed=${seeds[index]}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${HETNET_PYTHON:-$root/.venv/bin/python}
run_root=${SOFTROLE_CHANNELS_PCP_RUN_ROOT:-runs/softrole_channels_pcp_${SLURM_ARRAY_JOB_ID:-manual}}
run=$run_root/pcp_shared/$channel/seed$seed
[[ ! -e $run ]] || { echo "Run directory already exists: $run" >&2; exit 2; }
export PYTHONPATH="$root:$root/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec "$python" -u -m softrole train \
  --task pcp --study fixed --env-version corrected-observation-v1 --model shared --seed "$seed" \
  --optimizer adam --lr 0.0001 --num-p 2 --num-a 1 --dim 5 --vision 2 --max-steps 80 \
  --epochs 2000 --updates-per-epoch 10 --batch-steps 500 --nprocesses 4 \
  --total-steps 40000000 --save-every 50 \
  --experts 4 --pre-dim 128 --hidden-dim 64 --heads 4 --head-dim 16 --msg-dim 64 \
  --comm-rounds 3 --independent-heads --communication "$channel" \
  --comm-range -1 --gamma 1 --gae-lambda 0.95 --actor-coeff 50 --value-coeff 1 \
  --detach-gap 5 --max-grad-norm 0.75 --failure-prob 0 --failure-window 10 30 \
  --episode-log file --output "$run" "$@"
