#!/usr/bin/env bash
# Locked PCP failure screening: shared seeds 0/1/2, then banked seeds 0/1/2.
set -euo pipefail
usage() {
  echo 'Usage: bash scripts/softrole_failure.sh INDEX [runtime options...]' >&2
  echo 'INDEX: 0=shared/0 1=shared/1 2=shared/2 3=banked/0 4=banked/1 5=banked/2' >&2
  echo 'Options: --dry-run; --epochs, --updates-per-epoch, --batch-steps, --nprocesses, --total-steps, --save-every INTEGER' >&2
  echo 'Output root: SOFTROLE_FAILURE_RUN_ROOT (default runs/softrole_failure).' >&2
}
if (( $# == 0 )); then usage; exit 2; fi
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
index=$1
shift
[[ $index =~ ^[0-5]$ ]] || { echo 'Expected index 0..5' >&2; exit 2; }

# Restrict overrides so all six runs retain the declared failure protocol.
# In particular, resume cannot silently turn this into nominal continuation.
validate_options() {
  while (( $# )); do
    case "$1" in
      --dry-run) shift ;;
      --epochs|--updates-per-epoch|--batch-steps|--nprocesses|--total-steps|--save-every)
        (( $# >= 2 )) && [[ $2 =~ ^[0-9]+$ ]] || {
          echo "$1 needs an integer value" >&2; exit 2;
        }
        shift 2 ;;
      *) echo "Unsupported failure-launch option: $1" >&2; usage; exit 2 ;;
    esac
  done
}
validate_options "$@"
models=(shared shared shared banked banked banked)
seeds=(0 1 2 0 1 2)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SOFTROLE_RUN_ROOT=${SOFTROLE_FAILURE_RUN_ROOT:-runs/softrole_failure}
exec bash "$root/scripts/softrole.sh" pcp "${models[index]}" "${seeds[index]}" \
  --study failure "$@"
