#!/usr/bin/env bash
# Prepare a fresh locked protocol, then reconstruct both boundaries before recovery.
set -Eeuo pipefail
usage() {
  echo 'Usage: bash scripts/softrole_pcp_recovery.sh ARCHIVE_ROOT NEW_OUTPUT_ROOT [ADDITIONAL_STEPS] [--prepare-only]' >&2
}
(( $# >= 2 && $# <= 4 )) || { usage; exit 2; }
archive_root=$1
output_root=$2
shift 2
additional_steps=5000000
prepare_only=false
if (( $# )) && [[ $1 != --prepare-only ]]; then
  additional_steps=$1
  shift
fi
if (( $# )); then
  [[ $# == 1 && $1 == --prepare-only ]] || { usage; exit 2; }
  prepare_only=true
fi
[[ $additional_steps =~ ^[1-9][0-9]*$ ]] || { echo 'ADDITIONAL_STEPS must be positive' >&2; exit 2; }
# Resolve relative arguments before changing to the frozen checkout.
[[ $archive_root == /* ]] || archive_root="$PWD/$archive_root"
[[ $output_root == /* ]] || output_root="$PWD/$output_root"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export HETNET_PYTHON="${HETNET_PYTHON:-$root/.venv/bin/python}"
[[ -x $HETNET_PYTHON ]] || { echo 'Pinned interpreter is not executable' >&2; exit 2; }
"$HETNET_PYTHON" -u -m softrole.pcp_recovery prepare \
  --archive-root "$archive_root" --output "$output_root" --additional-steps "$additional_steps"
manifest="$output_root/manifest.json"
[[ -f $manifest ]] || { echo 'Preparation did not produce a manifest' >&2; exit 2; }
if $prepare_only; then
  printf 'Prepared recovery manifest: %s\n' "$manifest"
  exit 0
fi
# A successful reconstruction of BOTH boundaries is required for every branch.
reconstruct_submission=$(env -u BASH_ENV -u ENV sbatch --parsable --array=0-1%2 \
  "$root/slurm/softrole_pcp_recovery.sbatch" reconstruct "$manifest")
reconstruct_job=${reconstruct_submission%%;*}
[[ $reconstruct_job =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid reconstruction sbatch job ID' >&2; exit 2; }
printf 'Reconstruction job: %s\n' "$reconstruct_submission"
run_submission=$(env -u BASH_ENV -u ENV sbatch --parsable \
  "--dependency=afterok:$reconstruct_job" --array=0-5%6 \
  "$root/slurm/softrole_pcp_recovery.sbatch" run "$manifest")
run_job=${run_submission%%;*}
[[ $run_job =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid recovery sbatch job ID' >&2; exit 2; }
printf 'Recovery job: %s\nManifest: %s\n' "$run_submission" "$manifest"
