#!/usr/bin/env bash
# Two locked graded-sensing pilot arms: G-cap/seed0, then G-type/seed0.
set -euo pipefail
usage() {
  echo 'Usage: SOFTROLE_GRADED_PILOT_MANIFEST=manifest.json bash scripts/softrole_graded_pilot.sh INDEX [--dry-run]' >&2
  echo 'INDEX: 0=G-cap/seed0 1=G-type/seed0. All training settings come from the locked manifest.' >&2
  echo 'Output root: SOFTROLE_GRADED_PILOT_RUN_ROOT (default runs/graded_sensing_pilot_20261024/training).' >&2
}
if (( $# == 0 )); then usage; exit 2; fi
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
index=$1
shift
[[ $index =~ ^[01]$ ]] || { echo 'Expected index 0 or 1' >&2; exit 2; }
options=()
while (( $# )); do
  case "$1" in
    --dry-run) options+=("$1"); shift ;;
    *) echo "Unsupported graded-pilot option: $1" >&2; usage; exit 2 ;;
  esac
done
if [[ -z ${SOFTROLE_GRADED_PILOT_MANIFEST:-} ]]; then
  echo 'Set SOFTROLE_GRADED_PILOT_MANIFEST to the locked pilot manifest' >&2; exit 2
fi
manifest=$SOFTROLE_GRADED_PILOT_MANIFEST
[[ -f $manifest ]] || { echo "Manifest file not found: $manifest" >&2; exit 2; }
# Resolve a caller-relative manifest before moving to the repository checkout.
manifest=$(cd "$(dirname "$manifest")" && pwd)/$(basename "$manifest")
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${HETNET_PYTHON:-$root/.venv/bin/python}
run_root=${SOFTROLE_GRADED_PILOT_RUN_ROOT:-runs/graded_sensing_pilot_20261024/training}
export PYTHONPATH="$root:$root/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec "$python" -u -m softrole.graded_pilot launch \
  --manifest "$manifest" --index "$index" --run-root "$run_root" ${options[@]+"${options[@]}"}
