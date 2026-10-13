#!/usr/bin/env bash
# Historical Adam job 911488: midway and final D1-D6 panels for seeds 0--2.
set -euo pipefail
usage() {
  echo 'Usage: bash scripts/softrole_adam_diagnostics.sh [--run-root DIR] [--output-root DIR] [--submit]' >&2
  echo 'Default: prepare and validate both stages; --submit additionally submits two three-task arrays.' >&2
  echo 'Relative paths are resolved from the repository root. PYTHON overrides .venv/bin/python.' >&2
}
run_root= output_root= submit=false
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --run-root|--output-root)
      (( $# >= 2 )) && [[ -n $2 && $2 != --* ]] || {
        echo "$1 requires a value" >&2; usage; exit 2;
      }
      case "$1" in
        --run-root) [[ -z $run_root ]] || { echo 'Duplicate --run-root' >&2; exit 2; }; run_root=$2 ;;
        --output-root) [[ -z $output_root ]] || { echo 'Duplicate --output-root' >&2; exit 2; }; output_root=$2 ;;
      esac
      shift 2 ;;
    --submit)
      [[ $submit == false ]] || { echo 'Duplicate --submit' >&2; exit 2; }
      submit=true; shift ;;
    *) echo "Unsupported option: $1" >&2; usage; exit 2 ;;
  esac
done
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
run_root=${run_root:-runs/softrole_adam_pcp_911488}
output_root=${output_root:-runs/capcom_adam_diagnostics_$(date +%Y%m%d_%H%M%S)_$$}
python=${PYTHON:-$root/.venv/bin/python}
[[ ! -e $output_root && ! -L $output_root ]] || {
  echo "Output already exists: $output_root" >&2; exit 2;
}
for epoch in 1000 2000; do
  for seed in 0 1 2; do
    checkpoint="$run_root/pcp_shared/seed$seed/checkpoints/epoch$epoch.pt"
    [[ -f $checkpoint ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
  done
done
export HETNET_PYTHON="$python"
export PYTHONPATH="$root:$root/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
mkdir -p "$(dirname "$output_root")"
mkdir "$output_root"
mkdir "$output_root/manifests"
for stage in midway final; do
  if [[ $stage == midway ]]; then
    epoch=1000
    mode=logged-next-update
    rule='Historical Adam job 911488; seeds 0-2; first archived checkpoint reaching 20000000 environment steps per seed (epoch1000); next logged update'
  else
    epoch=2000
    mode=checkpoint-probe
    rule='Historical Adam job 911488; seeds 0-2; final archived checkpoint (epoch2000, approximately 40000000 environment steps); fresh checkpoint probe'
  fi
  checkpoints=()
  for seed in 0 1 2; do
    checkpoints+=("$run_root/pcp_shared/seed$seed/checkpoints/epoch$epoch.pt")
  done
  "$python" -u -m softrole.diagnostics prepare \
    --runtime archived-adam --batch-mode "$mode" \
    --checkpoints "${checkpoints[@]}" --checkpoint-rule "$rule" \
    --output "$output_root/manifests/$stage.json"
done
# Both manifests deliberately use the same default panel, seed and D5 settings.
# Validate every array entry before allowing either stage to be submitted.
for stage in midway final; do
  for index in 0 1 2; do
    bash scripts/softrole_diagnostics.sh \
      --manifest "$output_root/manifests/$stage.json" --index "$index" \
      --output "$output_root/$stage/task_$index" --dry-run
  done
done
if [[ $submit == true ]]; then
  mkdir -p logs_sr
  for stage in midway final; do
    CAPCOM_DIAGNOSTICS_MANIFEST="$output_root/manifests/$stage.json" \
    CAPCOM_DIAGNOSTICS_OUTPUT_ROOT="$output_root/$stage" \
      sbatch --export=ALL --array=0-2%3 slurm/softrole_diagnostics.sbatch
  done
fi
echo "Results root: $output_root"
echo "Midway: $output_root/manifests/midway.json -> $output_root/midway/task_{0,1,2}"
echo "Final: $output_root/manifests/final.json -> $output_root/final/task_{0,1,2}"
if [[ $submit == false ]]; then
  echo 'Prepared and validated only; no jobs submitted.'
fi
