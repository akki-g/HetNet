#!/usr/bin/env bash
# Train one entry of a locked, versioned CapCom/CapCon programme manifest.
set -euo pipefail
usage() {
  echo 'Usage: SOFTROLE_PROGRAMME_MANIFEST=manifest.json bash scripts/softrole_programme.sh INDEX [--dry-run] [runtime options...]' >&2
  echo 'INDEX follows the manifest: legacy 0..62; capcon_l1_40m_v1 0..11 (arm order times seeds 0,1,2).' >&2
  echo 'Options: --epochs, --updates-per-epoch, --batch-steps, --nprocesses, --total-steps, --save-every,' >&2
  echo '         --diagnostics-every POSITIVE_INTEGER; --episode-log file|chunked-gzip; --resume. Scientific settings come from the manifest.' >&2
}
if (( $# == 0 )); then usage; exit 2; fi
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
index=$1
shift
[[ $index =~ ^(0|[1-9][0-9]*)$ ]] || { usage; exit 2; }
options=()
operation=launch
while (( $# )); do
  case "$1" in
    --resume) operation=resume; shift ;;
    --dry-run) options+=("$1"); shift ;;
    --episode-log)
      (( $# >= 2 )) && [[ $2 == file || $2 == chunked-gzip ]] || { echo '--episode-log requires file or chunked-gzip' >&2; exit 2; }
      options+=("$1" "$2"); shift 2 ;;
    --epochs|--updates-per-epoch|--batch-steps|--nprocesses|--total-steps|--save-every|--diagnostics-every)
      (( $# >= 2 )) && [[ $2 =~ ^[1-9][0-9]*$ ]] || { echo "$1 requires a positive integer" >&2; exit 2; }
      options+=("$1" "$2"); shift 2 ;;
    *) echo "Unsupported programme option: $1" >&2; usage; exit 2 ;;
  esac
done
if [[ -z ${SOFTROLE_PROGRAMME_MANIFEST:-} ]]; then
  echo 'Set SOFTROLE_PROGRAMME_MANIFEST to the locked programme manifest' >&2; exit 2
fi
manifest=$SOFTROLE_PROGRAMME_MANIFEST
[[ -f $manifest ]] || { echo "Manifest file not found: $manifest" >&2; exit 2; }
manifest=$(cd "$(dirname "$manifest")" && pwd)/$(basename "$manifest")
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${HETNET_PYTHON:-$root/.venv/bin/python}
run_root=${SOFTROLE_PROGRAMME_RUN_ROOT:-runs/capcom_programme_${SLURM_ARRAY_JOB_ID:-manual}}
export PYTHONPATH="$root:$root/envs${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
exec "$python" -u -m softrole.programme "$operation" \
  --manifest "$manifest" --index "$index" --run-root "$run_root" ${options[@]+"${options[@]}"}
