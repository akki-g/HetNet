"""Bounded recovery continuation and fail-closed provenance checks."""
from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest
import torch

from softrole.config import Config
from softrole.ledger import canonical, sha
from softrole import pcp_recovery as recovery
from softrole.train import train, write_json


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_state(x, y)
    else:
        assert a == b


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@pytest.fixture
def tiny_boundary(tmp_path):
    config = Config(max_steps=4, vision=1, model='shared', pre_dim=8, hidden_dim=8,
                    heads=1, head_dim=4, msg_dim=4, seed=4, epochs=1,
                    updates_per_epoch=1, batch_steps=4, nprocesses=4, save_every=1,
                    optimizer='adam', lr=.001, critic_mode='decomposed', credit_beta=0.)
    path = train(config, tmp_path / 'boundary', episode_log='chunked-gzip')
    return config, Path(path), load(path)


@pytest.mark.parametrize('variant', recovery.VARIANTS, ids=[v[0] for v in recovery.VARIANTS])
def test_four_collector_branches_preserve_adam_and_exact_interrupted_continuation(
        tmp_path, monkeypatch, tiny_boundary, variant):
    config, boundary_path, boundary = tiny_boundary
    branch_config = replace(config, lr=variant[1], entropy_coeff=variant[2], epochs=3)
    root = tmp_path / variant[0]
    root.mkdir()
    initial = root / 'initial.pt'
    source = recovery.source_snapshot()['sha256']
    provenance = {'source_sha256': source}
    recovery.initialize_branch(boundary_path, branch_config, initial, provenance)
    saved = load(initial)
    equal_state(saved['model_state'], boundary['model_state'])
    equal_state(saved['optimizer_state']['state'], boundary['optimizer_state']['state'])
    expected_groups = deepcopy(boundary['optimizer_state']['param_groups'])
    for group in expected_groups:
        group['lr'] = variant[1]
    equal_state(saved['optimizer_state']['param_groups'], expected_groups)
    assert Config(**saved['config']).entropy_coeff == variant[2]
    for key in ('updates', 'total_steps', 'total_episodes'):
        assert saved[key] == boundary[key]
    manifest_hash = 'toy-manifest'
    write_json(root / 'initial.json', {'manifest_sha256': manifest_hash, 'checkpoint_sha256': sha(initial)})
    entry = {'output': str(root), 'config': recovery.normalized(branch_config)}
    assert recovery.accepted_branch(entry, manifest_hash, source, boundary=boundary_path,
                                    provenance=provenance)[0] == initial
    uninterrupted = train(branch_config, tmp_path / 'uninterrupted', resume=initial,
                          episode_log='chunked-gzip')
    for row in map(json.loads, (tmp_path / 'uninterrupted' / 'updates.jsonl').read_text().splitlines()):
        if variant[2]:
            assert row['action_entropy_sum'] > 0
            assert row['objective_entropy_loss_sum'] < 0
        else:
            assert 'action_entropy_sum' not in row and 'objective_entropy_loss_sum' not in row
    attempts = root / 'attempts'
    attempts.mkdir()
    write_json(attempts / '0000.json', {'number': 0, 'manifest_sha256': manifest_hash,
               'parent_checkpoint': str(initial.resolve()), 'parent_sha256': sha(initial)})
    training = importlib.import_module('softrole.train')
    original_apply = training.apply_gradients
    calls = 0

    def interrupt_second_update(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('bounded simulated interruption')
        return original_apply(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(training, 'apply_gradients', interrupt_second_update)
        with pytest.raises(RuntimeError, match='simulated interruption'):
            train(branch_config, attempts / '0000', resume=initial, episode_log='chunked-gzip')
    # Simulate a killed append after the committed prefix, without a newline.
    with (attempts / '0000' / 'updates.jsonl').open('a') as stream:
        stream.write('{"update":')
    parent_path, parent, records = recovery.accepted_branch(entry, manifest_hash, source,
                                                          boundary=boundary_path, provenance=provenance)
    assert parent['updates'] == 2 and len(records) == 1
    write_json(attempts / '0001.json', {'number': 1, 'manifest_sha256': manifest_hash,
               'parent_checkpoint': str(parent_path.resolve()), 'parent_sha256': sha(parent_path)})
    train(branch_config, attempts / '0001', resume=parent_path, episode_log='chunked-gzip')
    final_path, final, records = recovery.accepted_branch(entry, manifest_hash, source)
    expected = load(uninterrupted)
    assert final['updates'] == 3 and len(records) == 2
    equal_state(final['model_state'], expected['model_state'])
    equal_state(final['optimizer_state'], expected['optimizer_state'])
    for key in ('updates', 'total_steps', 'total_episodes'):
        assert final[key] == expected[key]
    assert sha(boundary_path) != sha(final_path)
    # Mutation of committed ledger evidence must invalidate acceptance.
    with (final_path.parent.parent / 'updates.jsonl').open('a') as stream:
        stream.write('{}\n')
    # Appended unpublished tails are explicitly permitted; changing committed bytes is not.
    updates = final_path.parent.parent / 'updates.jsonl'
    updates.write_bytes(updates.read_bytes().replace(b'"update": 3', b'"update": 9', 1))
    with pytest.raises(ValueError):
        recovery.accepted_branch(entry, manifest_hash, source)


def test_branch_cannot_change_architecture_or_overwrite(tmp_path, tiny_boundary):
    config, boundary, _ = tiny_boundary
    with pytest.raises(ValueError, match='only LR'):
        recovery.initialize_branch(boundary, replace(config, gamma=1., hidden_dim=16),
                                   tmp_path / 'bad.pt', {'source_sha256': 'toy'})
    destination = tmp_path / 'occupied.pt'
    destination.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        recovery.initialize_branch(boundary, config, destination, {'source_sha256': 'toy'})
    assert destination.read_bytes() == b'preserve'
    root = tmp_path / 'substitution'
    root.mkdir()
    initial = root / 'initial.pt'
    provenance = {'source_sha256': 'toy'}
    recovery.initialize_branch(boundary, config, initial, provenance)
    saved = load(initial)
    next(iter(saved['model_state'].values())).add_(1.)
    torch.save(saved, initial)
    write_json(root / 'initial.json', {'manifest_sha256': 'toy', 'checkpoint_sha256': sha(initial)})
    with pytest.raises(ValueError, match='admitted onset state'):
        recovery.accepted_branch({'output': str(root), 'config': recovery.normalized(config)},
                                 'toy', 'toy', boundary=boundary, provenance=provenance)


@pytest.fixture
def toy_manifest(tmp_path, monkeypatch):
    parents = [dict(seed=seed, update=row[0], total_steps=row[1], config=recovery.normalized(Config(seed=seed)))
               for seed, row in recovery.ONSETS.items()]
    monkeypatch.setattr(recovery, 'parent_record', lambda archive, seed: deepcopy(parents[seed - 4]))
    monkeypatch.setattr(recovery, 'source_snapshot', lambda: {'sha256': 'locked-toy-source'})
    root = tmp_path / 'prepared'
    manifest = recovery.prepare(tmp_path / 'archive', root)
    return root / 'manifest.json', manifest


def test_manifest_exact_protocol_and_no_overwrite(toy_manifest):
    path, manifest = toy_manifest
    assert recovery.load_manifest(path) == manifest
    assert [recovery.launch(path, i, dry_run=True)['variant'] for i in range(6)] == [
        'control', 'lower_lr', 'entropy', 'control', 'lower_lr', 'entropy']
    with pytest.raises(FileExistsError):
        recovery.prepare(path.parent, path.parent)
    with pytest.raises(ValueError, match='5M'):
        recovery.prepare(path.parent, path.parent / 'wrong', additional_steps=123)


@pytest.mark.parametrize('index', [-1, 6, True, 1.5])
def test_invalid_indices_fail_before_read(index):
    with pytest.raises(ValueError, match='index'):
        recovery.launch('does-not-exist', index, dry_run=True)


@pytest.mark.parametrize('field', ['source_sha256', 'entry', 'onset', 'panel'])
def test_manifest_mutations_rejected(toy_manifest, field):
    path, manifest = toy_manifest
    if field == 'source_sha256':
        manifest[field] = 'substitution'
    elif field == 'entry':
        manifest['entries'][0]['config']['lr'] = .9
    elif field == 'onset':
        manifest['parents'][0]['update'] += 1
    else:
        (path.parent / 'panel.json').write_text('[]')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        recovery.load_manifest(path)


def test_existing_evaluation_cannot_substitute_checkpoint_or_panel(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'toy-checkpoint')
    entry = {'output': str(tmp_path)}
    manifest = {'panel_sha256': canonical(recovery.panel())}
    for report in ({'checkpoint_sha256': 'wrong', 'scenarios_sha256': manifest['panel_sha256']},
                   {'checkpoint_sha256': sha(checkpoint), 'scenarios_sha256': 'wrong'}):
        (tmp_path / 'nominal.json').write_text(json.dumps(report))
        with pytest.raises(ValueError, match='different checkpoint/panel'):
            recovery.evaluate_entry(manifest, entry, checkpoint)


def test_branch_budget_is_incremental_and_extends_epoch_ceiling():
    config = Config(epochs=2000, updates_per_epoch=10, batch_steps=500, nprocesses=4)
    parent = {'config': recovery.normalized(config), 'update': 19588, 'total_steps': 39432334}
    for variant in recovery.VARIANTS:
        branch = recovery.branch_config(parent, variant)
        assert branch.total_steps == parent['total_steps'] + 5_000_000
        assert branch.epochs * branch.updates_per_epoch > parent['update'] + 2500
        assert branch.nprocesses == 4 and branch.updates_per_epoch == 10
        assert branch.lr == variant[1] and branch.entropy_coeff == variant[2]


def test_reconstruct_invalid_index_fails_before_read():
    with pytest.raises(ValueError, match='index'):
        recovery.reconstruct('missing', 2)
