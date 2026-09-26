#!/bin/bash
# Source from a Phase A sbatch script. Installation only inside an allocation.
hetnet_bootstrap() {
    : "${SLURM_JOB_ID:?Submit inside a Stokes compute allocation}"
    : "${SLURM_SUBMIT_DIR:?Submit from the HetNet repository root}"
    : "${HETNET_STOKES_PREFLIGHT:?Set the path to verified Stokes preflight JSON}"
    : "${HETNET_GATE_A:?Set the path to passing Gate A JSON}"
    : "${HETNET_MEM_GB:?Set the explicitly requested memory in GiB}"
    : "${HETNET_TIME_SECONDS:?Set the explicitly requested time in seconds}"
    : "${HETNET_ARRAY_CONCURRENCY:?Set a verified explicit array concurrency}"
    case "${SLURM_CLUSTER_NAME:-}" in
        stokes|Stokes) ;;
        *) echo "This study is authorized for Stokes only; ask Akki before switching." >&2; return 2 ;;
    esac
    cd "$SLURM_SUBMIT_DIR"
    test -f uv.lock || { echo "Missing uv.lock; finish the locked local environment gate." >&2; return 2; }
    test -r "$HETNET_STOKES_PREFLIGHT" && test -r "$HETNET_GATE_A" || {
        echo "Preflight/Gate A report missing; no installation or training started." >&2; return 2;
    }
    local setup_started python_bin uv_dir
    setup_started=$(date +%s)
    export HETNET_BOOTSTRAP_STARTED_AT
    HETNET_BOOTSTRAP_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
    module purge
    module load anaconda/anaconda-2024.10
    conda activate base
    set -u
    module list
    python_bin=$(python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Stokes Anaconda must provide Python 3.12; ask Akki before changing modules"; print(sys.executable)')
    # Fail stale evidence before dependency installation. These modules use only stdlib.
    "$python_bin" - <<'PY'
import os
from pathlib import Path
from hetnet_ext.grid import ROOT, code_sha256, file_sha256
from hetnet_ext.train_job import read_json, validate_gate, validate_preflight
validate_preflight(read_json(os.environ['HETNET_STOKES_PREFLIGHT']))
validate_gate(read_json(os.environ['HETNET_GATE_A']), code_sha256(), file_sha256(ROOT / 'uv.lock'))
PY
    command -v flock >/dev/null || { echo "flock is unavailable; do not run concurrent setup." >&2; return 2; }
    uv_dir="$PWD/.tools/uv-0.12.5"
    export UV_PROJECT_ENVIRONMENT="$PWD/.venv-stokes"
    mkdir -p .tools
    (
        flock --exclusive 9
        if [[ ! -x "$uv_dir/uv" ]]; then
            curl --fail --location --silent --show-error https://astral.sh/uv/0.12.5/install.sh \
                | env UV_UNMANAGED_INSTALL="$uv_dir" sh
        fi
        "$uv_dir/uv" --version | awk '$1 == "uv" && $2 == "0.12.5" { ok=1 } END { exit !ok }'
        "$uv_dir/uv" sync --locked --python "$python_bin" --no-python-downloads
    ) 9>.tools/stokes-setup.lock
    export HETNET_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
    "$HETNET_PYTHON" - <<'PY'
import sys
import torch, dgl, torchdata, gym, numpy
assert sys.version_info[:2] == (3, 12)
assert torch.__version__.split('+')[0] == '2.2.1'
assert dgl.__version__ == '2.1.0'
assert torchdata.__version__.split('+')[0] == '0.7.1'
assert gym.__version__ == '0.26.2'
assert int(numpy.__version__.split('.')[0]) < 2
print('Pinned Stokes imports:', sys.version, torch.__version__, dgl.__version__, torchdata.__version__, gym.__version__, numpy.__version__)
PY
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HETNET_TORCH_THREADS=1
    export HETNET_BOOTSTRAP_SECONDS=$(( $(date +%s) - setup_started ))
}
