#!/usr/bin/env bash
# Released HetNet recipes. No preflight files or custom scheduler framework.
set -eo pipefail
if (( $# < 3 )); then
  echo 'Usage: bash scripts/reproduce.sh {pp|pcp|fc} {real|binary} SEED [--dry-run] [main.py options...]' >&2
  exit 2
fi
task=$1 variant=$2 seed=$3
shift 3
[[ $seed =~ ^[0-9]+$ ]] || { echo 'SEED must be a nonnegative integer' >&2; exit 2; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${HETNET_PYTHON:-$root/.venv/bin/python}
run=${HETNET_RUN_ROOT:-runs/reproduction}/${task}_${variant}/seed${seed}
case "$task" in
  pp)  domain=(--env_name predator_capture --nfriendly_P 3 --nfriendly_A 0 --num_epochs 2000 --max_steps 80) ;;
  pcp) domain=(--env_name predator_capture --nfriendly_P 2 --nfriendly_A 1 --num_epochs 2000 --max_steps 80) ;;
  fc)  domain=(--env_name fire_commander --nfriendly_P 2 --nfriendly_A 1 --num_epochs 1400 --max_steps 300 --vision 1 --nfires 1 --reward_type 3) ;;
  *) echo 'Unknown task: choose pp, pcp or fc' >&2; exit 2 ;;
esac
case "$variant" in
  real) method=() ;;
  binary) method=(--use_binary --msg_dim 16) ;;
  *) echo 'Unknown variant: choose real or binary' >&2; exit 2 ;;
esac
dry_run=false
extra=()
for arg in "$@"; do
  if [[ $arg == --dry-run ]]; then dry_run=true; else extra+=("$arg"); fi
done
command=("$python" -u main.py "${domain[@]}" --hetgat --hetgat_a2c
  --nprocesses 4 --epoch_size 10 --batch_size 500 --hid_size 128
  --detach_gap 5 --lrate 0.0001 --dim 5 --seed "$seed" --save_every 50
  "${method[@]}" "${extra[@]}"
  --experiment_name "${task}_${variant}_seed${seed}" --save_dir "$run/checkpoints"
  --metrics_file "$run/metrics.jsonl")
if $dry_run; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
if [[ ! -x $python ]]; then
  printf 'Python environment missing or not executable: %s\n' "$python" >&2
  echo 'On Stokes, first run: mkdir -p logs; sbatch slurm/setup.sbatch' >&2
  echo 'Elsewhere with Python 3.12: bash scripts/setup_env.sh' >&2
  echo 'An existing environment can be selected with HETNET_PYTHON=/absolute/path/bin/python.' >&2
  exit 2
fi
mkdir -p "$(dirname "$run")"
# Atomic creation prevents an accidental rerun from replacing previous evidence.
mkdir "$run" || { echo "Run already exists: $run (choose a new HETNET_RUN_ROOT)" >&2; exit 2; }
trap 'code=$?; printf "%s\n" "$code" > "$run/exit_code.txt"' EXIT
printf '%q ' "${command[@]}" > "$run/command.txt"
printf '\n' >> "$run/command.txt"
{
  date -u
  hostname
  git rev-parse HEAD
  git status --short
  "$python" --version
  "$python" -c 'import importlib.metadata as m; print("Installed packages:"); print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in m.distributions())))'
  printf 'Slurm job: %s; array task: %s\n' "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-none}"
} > "$run/environment.txt" 2>&1
git diff --no-ext-diff HEAD > "$run/source.patch"
cp uv.lock "$run/uv.lock"
cp requirements.txt "$run/requirements.txt"
export DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"${command[@]}" 2>&1 | tee "$run/stdout.log"
