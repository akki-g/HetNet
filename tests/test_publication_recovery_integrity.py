"""Continuation accepts recorded bytes and valid optimizer state, without training."""
import copy
import hashlib
import json
import shutil

import pytest
import torch

from test_publication_artifacts import archived_run, resume_args
from publication_reconstruction import __main__ as launcher
from publication_reconstruction.artifacts import _validate_optimizer


def test_resume_rejects_finite_weight_change_against_recorded_checkpoint(archived_run, tmp_path):
    archive = archived_run
    archive.saved['policy_net']['weight'][0] += 1
    torch.save(archive.saved, archive.checkpoint)
    destination = tmp_path / 'continued'
    with pytest.raises(ValueError, match='recorded bytes'):
        launcher.resolve_resume(resume_args(archive, destination))
    assert not destination.exists()


@pytest.mark.parametrize('problem', ['missing_ledger', 'unrecorded_copy', 'recorded_counts', 'bad_prefix'])
def test_resume_requires_checkpoint_ledger_binding(archived_run, tmp_path, problem):
    archive = archived_run
    ledger = archive.run / 'checkpoint_records.jsonl'
    if problem == 'missing_ledger':
        ledger.unlink()
    elif problem == 'unrecorded_copy':
        archive.checkpoint = tmp_path / 'unrecorded.pt'
        saved = copy.deepcopy(archive.saved)
        saved['policy_net']['weight'][0] += 1
        torch.save(saved, archive.checkpoint)
    elif problem == 'recorded_counts':
        row = json.loads(ledger.read_text())
        row['counts']['env_steps'] += 1
        ledger.write_text(json.dumps(row) + '\n')
    else:
        ledger.write_text('{"incomplete":\n' + ledger.read_text())
    with pytest.raises(ValueError, match='recorded|ledger'):
        launcher.resolve_resume(resume_args(archive, tmp_path / 'continued'))
    assert not (tmp_path / 'continued').exists()


@pytest.mark.parametrize('location', ['moved_archive', 'external_copy'])
def test_resume_accepts_recorded_bytes_after_copy(archived_run, tmp_path, location):
    archive = archived_run
    expected = hashlib.sha256(archive.checkpoint.read_bytes()).hexdigest()
    if location == 'moved_archive':
        copied = tmp_path / 'copied run'
        shutil.copytree(archive.run, copied)
        archive.run = copied
        archive.checkpoint = copied / 'checkpoint.pt'
    else:
        copied = tmp_path / 'copied checkpoint.pt'
        shutil.copy2(archive.checkpoint, copied)
        archive.checkpoint = copied
    result = launcher.resolve_resume(resume_args(archive, tmp_path / 'continued'))
    assert result['continuation']['checkpoint_sha256'] == expected
    assert not (tmp_path / 'continued').exists()


def test_resume_ignores_abandoned_ledger_suffix_after_selected_checkpoint(archived_run, tmp_path):
    ledger = archived_run.run / 'checkpoint_records.jsonl'
    with ledger.open('a') as stream:
        stream.write('{"interrupted":')
    result = launcher.resolve_resume(resume_args(archived_run, tmp_path / 'continued'))
    assert result['continuation']['counts']['updates'] == 1


@pytest.mark.parametrize('problem', ['wrong_step', 'missing_step', 'nonscalar_step',
    'wrong_shape', 'wrong_dtype', 'negative_variance', 'parameter_inventory'])
def test_optimizer_validation_rejects_invalid_existing_state(archived_run, problem):
    saved = copy.deepcopy(archived_run.saved)
    state = saved['trainer']['state'][0]
    if problem == 'wrong_step':
        state['step'].fill_(999)
    elif problem == 'missing_step':
        del state['step']
    elif problem == 'nonscalar_step':
        state['step'] = torch.ones(1)
    elif problem == 'wrong_shape':
        state['exp_avg'] = torch.zeros(2, dtype=torch.float64)
    elif problem == 'wrong_dtype':
        state['exp_avg'] = state['exp_avg'].float()
    elif problem == 'negative_variance':
        state['exp_avg_sq'].fill_(-1)
    else:
        saved['trainer']['param_groups'][0]['params'].append(1)
    with pytest.raises(ValueError, match='optimizer'):
        _validate_optimizer(saved, archived_run.protocol, 1)


@pytest.mark.parametrize('optimizer_kind', ['Adam', 'RMSprop'])
def test_optimizer_validation_allows_legitimately_unused_parameters(optimizer_kind):
    active = torch.nn.Parameter(torch.tensor([.25], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([.75], dtype=torch.float32))
    if optimizer_kind == 'Adam':
        optimizer = torch.optim.Adam([active, unused], lr=.001, foreach=False, fused=False)
        specification = dict(name='Adam', lr=.001, betas=[.9, .999], epsilon=1e-8,
            weight_decay=0, amsgrad=False, foreach=False, fused=False)
    else:
        optimizer = torch.optim.RMSprop([active, unused], lr=.0001, alpha=.97, eps=1e-6)
        specification = dict(name='RMSprop', lr=.0001, alpha=.97, epsilon=1e-6)
    for _ in range(3):
        active.grad = torch.ones_like(active)
        optimizer.step()
    saved = {'policy_net': {'active': active.detach(), 'unused': unused.detach()},
             'trainer': optimizer.state_dict()}
    assert len(saved['trainer']['state']) == 1
    _validate_optimizer(saved, {'optimizer': specification}, 3)
