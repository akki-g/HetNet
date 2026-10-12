#!/usr/bin/env bash
# D1-D6 checkpoint diagnostics: protocol and checkpoint selection live in the manifest.
set -euo pipefail
usage() {
  echo 'Usage: bash scripts/softrole_diagnostics.sh --manifest FILE --index N --output DIR [--dry-run]' >&2
  echo 'N is the zero-based manifest entry index; DIR must not exist.' >&2
  echo 'Only these options are accepted. Configure diagnostic settings in the manifest.' >&2
}
manifest= index= output= dry_run=false
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --manifest|--index|--output)
      (( $# >= 2 )) && [[ -n $2 && $2 != --* ]] || {
        echo "$1 requires a value" >&2; usage; exit 2;
      }
      case "$1" in
        --manifest) [[ -z $manifest ]] || { echo 'Duplicate --manifest' >&2; exit 2; }; manifest=$2 ;;
        --index) [[ -z $index ]] || { echo 'Duplicate --index' >&2; exit 2; }; index=$2 ;;
        --output) [[ -z $output ]] || { echo 'Duplicate --output' >&2; exit 2; }; output=$2 ;;
      esac
      shift 2 ;;
    --dry-run)
      [[ $dry_run == false ]] || { echo 'Duplicate --dry-run' >&2; exit 2; }
      dry_run=true; shift ;;
    *) echo "Unsupported diagnostics option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n $manifest && -n $index && -n $output ]] || { usage; exit 2; }
[[ $index =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Index must be a nonnegative integer' >&2; exit 2; }
[[ -f $manifest ]] || { echo "Manifest file not found: $manifest" >&2; exit 2; }
[[ ! -e $output && ! -L $output ]] || { echo "Output already exists: $output" >&2; exit 2; }
# Preserve paths relative to the caller while importing this checkout explicitly.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${HETNET_PYTHON:-$root/.venv/bin/python}
export PYTHONPATH="$root:$root/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
options=(--manifest "$manifest" --index "$index" --output "$output")
if [[ $dry_run == true ]]; then options+=(--dry-run); fi
exec "$python" -u -m softrole.diagnostics "${options[@]}"
