#!/bin/bash
# Sourced by batch jobs. The submission helper creates the container before Slurm
# opens stdout/stderr; trainer leaf directories remain fresh.
if [[ -z ${HETNET_RUN_DIR:-} || -z ${HETNET_RUN_TITLE:-} ||
      ! -f ${HETNET_RUN_DIR:-/nonexistent}/run_info.json ||
      ! -d ${HETNET_RUN_DIR:-/nonexistent}/logs ]]; then
  echo 'Submit through python scripts/submit_stokes.py --title TITLE --purpose PURPOSE -- JOB.sbatch; reuse --run-dir for later stages.' >&2
  return 2
fi

stokes_run_output() {
  "${HETNET_SUBMISSION_PYTHON:-python3}" - "$HETNET_RUN_DIR" "$@" <<'PY'
import os
import sys
root = os.path.realpath(sys.argv[1])
for value in sys.argv[2:]:
    path = os.path.realpath(value)
    if os.path.commonpath([root, path]) != root:
        raise SystemExit('Run output must stay inside {}: {}'.format(root, value))
PY
}
