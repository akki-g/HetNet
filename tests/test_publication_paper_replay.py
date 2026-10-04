"""The fixed-rollout gate exercises full losses/updates, not layer timing alone."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('task,variant,collectors', [('pcp', 'binary', 4), ('fc', 'real', 1)])
def test_recorded_replay_checks_complete_updates_and_keeps_timing_separate(tmp_path, task, variant, collectors):
    output = tmp_path / 'replay'
    command = [sys.executable, '-m', 'publication_reconstruction.runtime.hetnet_ext.paper_replay',
        '--output', str(output), '--task', task, '--variant', variant,
        '--collectors', str(collectors), '--updates', '3', '--batch-steps', '2',
        '--horizon', '6', '--timing-repeats', '1']
    env = {**os.environ, 'DGLBACKEND': 'pytorch', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-7000:]
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['correctness_passed']
    assert summary['collectors'] == collectors and summary['updates'] == 3
    assert 'serial collector compute' in summary['semantics']
    assert 'no diagnostic tensor copies' in summary['timing_overhead']
    assert summary['fixture_generation_seconds'] > 0
    assert summary['fixture_sha256'] == hashlib.sha256((output / 'fixture.pt').read_bytes()).hexdigest()
    assert len(summary['pairs']) == 1
    pair = summary['pairs'][0]
    assert pair['order'] == ['dgl', 'torch-v1']
    assert pair['speedup'] > 0  # Functional smoke: deliberately no performance claim.
    for run in pair['runs'].values():
        assert len(run['update_seconds']) == 3 and run['wall_seconds'] > 0
        assert run['first_update_seconds'] == run['update_seconds'][0]
        assert run['max_rss_unit'] in ('bytes', 'KiB')
    fixture = torch.load(output / 'fixture.pt', map_location='cpu')
    assert len(fixture['updates']) == 3
    assert len(fixture['updates'][0]) == collectors
    assert fixture['initial_optimizer']['state'] == {}
    assert fixture['final_optimizer']['state']
    first = fixture['updates'][0][0][0][0]
    assert set(first) == {'observation', 'action', 'reward', 'rng_before', 'rng_after_forward'}
    assert first['observation'].shape[1] == 3
    assert set(first['rng_before']) == {'python', 'numpy', 'torch'}
    # Output protection is checked before importing/generating fresh trajectories.
    repeat = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert repeat.returncode != 0 and 'FileExistsError' in repeat.stderr
    assert hashlib.sha256((output / 'fixture.pt').read_bytes()).hexdigest() == summary['fixture_sha256']


def test_replay_shared_weights_keep_separate_gradients_and_finite_gate(tmp_path):
    # Isolate the historical runtime import names from root/SoftRole modules.
    code = r'''
from types import SimpleNamespace
import numpy as np
import torch
from publication_reconstruction.runtime.hetnet_ext.paper_replay import (
    compare, finish_update, share_parameter_storage)
torch.set_default_dtype(torch.float64)
models = [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
with torch.no_grad():
    models[0].weight.fill_(1)
    models[1].weight.fill_(9)

class Policy:
    def __init__(self, model, gradient):
        self.model, self.gradient = model, gradient
    def batch_finish_per_class(self, *args):
        self.model.weight.grad = torch.full_like(self.model.weight, self.gradient)
        return dict(total=np.array(1.), policy=np.array(.5), critic=np.array(.5))

trainers = [SimpleNamespace(policy_net=model, params=list(model.parameters()),
    policy=Policy(model, gradient), optimizer=torch.optim.SGD(model.parameters(), lr=1),
    args=SimpleNamespace(max_steps=1, nfriendly_P=2, nfriendly_A=1))
    for model, gradient in zip(models, (1., 3.))]
share_parameter_storage(trainers)
assert models[0].weight is not models[1].weight
assert models[0].weight.data_ptr() == models[1].weight.data_ptr()
# No learned forward runs in this test: a clone here would be replay diagnostic
# bookkeeping inside the timed aggregation/optimizer path.
original_clone = torch.Tensor.clone
def forbid_clone(*args, **kwargs):
    raise AssertionError('Diagnostic clone in uncaptured replay update')
torch.Tensor.clone = forbid_clone
try:
    assert finish_update(trainers, [[None], [None, None]], capture=False) is None
finally:
    torch.Tensor.clone = original_clone
assert models[0].weight.grad.data_ptr() != models[1].weight.grad.data_ptr()
assert models[1].weight.grad.item() == 3
assert abs(models[0].weight.grad.item() - .75) < 1e-6
assert .24 < models[0].weight.item() < .26
assert torch.equal(models[0].weight, models[1].weight)
for exact in (False, True):
    for invalid in (float('inf'), -float('inf'), float('nan')):
        try:
            compare(torch.tensor([invalid]), torch.tensor([invalid]), exact=exact)
        except AssertionError as error:
            assert 'Non-finite tensor' in str(error)
        else:
            raise AssertionError('Matching non-finite tensors escaped the gate')
'''
    environment = {**os.environ, 'DGLBACKEND': 'pytorch', 'OMP_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
