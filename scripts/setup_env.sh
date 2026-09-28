#!/usr/bin/env bash
# Python 3.12 + standard venv/pip; uv is not needed to use this script.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
base_python=${HETNET_BASE_PYTHON:-python3}
venv=${1:-.venv}
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "HetNet requires Python 3.12; load anaconda/anaconda-2024.10 or set HETNET_BASE_PYTHON"'
"$base_python" -m venv "$venv"
python="$venv/bin/python"
# Also handles an existing uv-created environment that has no pip installed.
"$python" -m ensurepip --upgrade
"$python" -m pip install 'pip==25.0.1' 'setuptools==75.8.0' 'wheel==0.45.1'
# Match uv.lock's setuptools build constraint for legacy Gym/Visdom sources.
"$python" -m pip install --no-build-isolation -r requirements.txt
"$python" -m pip install --no-deps --no-build-isolation -e ./envs
"$python" -m pip check
export DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" - <<'PY'
import sys
import torch, dgl, torchdata, gym, numpy, ic3net_envs
print('Python:', sys.version)
for package in (torch, dgl, torchdata, gym, numpy):
    print(package.__name__, package.__version__)
graph = dgl.graph(([0], [1]))
assert graph.in_degrees().tolist() == [0, 1]
print('HetNet environment imports and a CPU DGL operation passed.')
PY
printf 'Environment ready: %s/bin/python\n' "$venv"
