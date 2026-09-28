#!/bin/bash
# Review only by default. Pass --submit explicitly after calibrated budget approval.
# This driver uses only the Python standard library; it installs nothing.
set -euo pipefail
submission_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$submission_script_dir/.."
exec "${HETNET_SUBMIT_PYTHON:-python3}" -m hetnet_ext.submit_run1 "$@"
