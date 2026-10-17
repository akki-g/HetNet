"""Archive admission is pinned and private modules never replace live modules."""
import json
from pathlib import Path
import sys

import pytest

from softrole.diagnostic_archive import archived_adam_runtime, ARCHIVED_SOURCE_SHA256


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'stokes_runs/softrole_adam_pcp_911488/pcp_shared/seed0/checkpoints/epoch0050.pt'


def private_modules():
    return {name for name in sys.modules if name.startswith('_softrole_adam_archive_')}


@pytest.fixture
def checkpoint():
    if not CHECKPOINT.is_file():
        pytest.skip('The audited research archive is not distributed with repository source')
    return CHECKPOINT


def test_unknown_archive_refused_before_import(tmp_path):
    checkpoint = tmp_path / 'checkpoints' / 'epoch0050.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'not loaded by the source admission check')
    (tmp_path / 'source_manifest.json').write_text(json.dumps({'files': {}, 'sha256': 'unrecognized'}))
    before = private_modules()
    with pytest.raises(ValueError, match='Unknown archived Adam source profile'):
        with archived_adam_runtime(checkpoint):
            pytest.fail('Unknown source must not execute')
    assert private_modules() == before


def test_private_archived_symbols_and_metadata_preserve_current_modules(checkpoint):
    from softrole import config, env, learning, model, rollout, train
    modules = (config, env, learning, model, rollout, train)
    before = private_modules()
    originals = {module.__name__: dict(module.__dict__) for module in modules}
    with archived_adam_runtime(checkpoint) as runtime:
        cfg = runtime['Config']()
        assert cfg.communication == 'binary' and cfg.comm_rounds == 2
        assert cfg.critic_mode == 'team' and cfg.objective == 'team'
        assert cfg.credit_beta == 1 and cfg.actor_capabilities == 'explicit'
        assert not cfg.null_degree_correction and not cfg.independent_heads
        # Compatibility metadata is not added to the archived serialized config.
        assert 'communication' not in cfg.to_dict()
        assert runtime['SoftRoleNet'].forward is runtime['SoftRoleNet'].__bases__[0].forward
        assert runtime['run_episode'].__globals__['Episode'] is not learning.Episode
        assert runtime['collect_batch'].__globals__['run_episode'] is runtime['run_episode']
        assert runtime['collect_batch'].__globals__['make_env'] is runtime['make_env']
        adapter = runtime['make_env'](cfg)
        assert type(adapter.raw).__module__.startswith('_softrole_adam_archive_')
        assert adapter._rng is None
        assert runtime['provenance']['source_sha256'] == ARCHIVED_SOURCE_SHA256
        assert len(runtime['provenance']['module_paths']) == 11
        assert private_modules() - before
        for module in modules:
            assert sys.modules[module.__name__] is module
            assert module.__dict__.keys() == originals[module.__name__].keys()
            assert all(module.__dict__[key] is value for key, value in originals[module.__name__].items())
    assert private_modules() == before


def test_nested_contexts_isolated_and_cleaned_on_exception(checkpoint):
    before = private_modules()
    with archived_adam_runtime(checkpoint) as first:
        outer = private_modules()
        with pytest.raises(RuntimeError, match='test cleanup'):
            with archived_adam_runtime(checkpoint) as second:
                assert first['run_episode'] is not second['run_episode']
                assert first['run_episode'].__module__ != second['run_episode'].__module__
                assert private_modules() > outer
                raise RuntimeError('test cleanup')
        assert private_modules() == outer
    assert private_modules() == before


def test_changed_archived_bytes_refused_before_import(checkpoint, monkeypatch):
    original = Path.read_bytes
    target = checkpoint.resolve().parent.parent / 'source/softrole/model.py'

    def altered(path):
        data = original(path)
        return data + b'\n# changed\n' if path == target else data

    monkeypatch.setattr(Path, 'read_bytes', altered)
    before = private_modules()
    with pytest.raises(ValueError, match='Archived source hash mismatch: softrole/model.py'):
        with archived_adam_runtime(checkpoint):
            pytest.fail('Changed source must not execute')
    assert private_modules() == before


def test_full_archived_diagnostic_probe_on_synthetic_fixture(checkpoint, tmp_path):
    """Old bits/Episode interfaces complete D1-D6 without research training."""
    import hashlib
    import shutil

    import numpy as np
    import torch

    from hetnet_ext.signatures import model_signature
    from softrole import diagnostics

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    original = checkpoint.parent.parent
    original_checkpoint_hash = digest(checkpoint)
    original_manifest_hash = digest(original / 'source_manifest.json')
    run = tmp_path / 'synthetic'
    run.mkdir()
    shutil.copytree(original / 'source', run / 'source')
    shutil.copyfile(original / 'source_manifest.json', run / 'source_manifest.json')
    shutil.copyfile(original / 'config.json', run / 'config.json')
    (run / 'checkpoints').mkdir()
    synthetic_checkpoint = run / 'checkpoints/epoch0001.pt'
    synthetic_checkpoint.write_bytes(b'placeholder; source admission does not load tensors')
    torch.set_num_threads(1)
    with archived_adam_runtime(synthetic_checkpoint) as runtime:
        cfg = runtime['Config'](model='shared', optimizer='adam', max_steps=2,
                                nprocesses=1, batch_steps=1, epochs=1, updates_per_epoch=1)
        with torch.random.fork_rng():
            torch.manual_seed(77)
            model = runtime['SoftRoleNet'](**cfg.model_kwargs()).double()
        optimizer = runtime['make_optimizer'](model.parameters(), cfg)
        before = model_signature(model)
        # Populate complete finite Adam state for the synthetic fixture. These
        # zero gradients neither train the policy nor change its parameters.
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        assert model_signature(model) == before
        saved = {
            'format_version': 1, 'config': cfg.to_dict(), 'model_config': cfg.model_kwargs(),
            'environment_version': cfg.env_version, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'optimizer':
                runtime['make_optimizer'].__globals__['optimizer_spec'](cfg),
            'signature': before, 'updates': 1, 'total_steps': 1, 'total_episodes': 1,
            'source_sha256': ARCHIVED_SOURCE_SHA256,
        }
        torch.save(saved, synthetic_checkpoint)
        (run / 'config.json').write_text(json.dumps(cfg.to_dict()))
        (run / 'run.json').write_text(json.dumps({
            'source_sha256': ARCHIVED_SOURCE_SHA256, 'environment_version': cfg.env_version,
            'torch': torch.__version__, 'numpy': np.__version__,
        }))
        Path(str(synthetic_checkpoint) + '.signature.json').write_text(json.dumps(before))
    fixture_hash = digest(synthetic_checkpoint)
    manifest = tmp_path / 'manifest.json'
    output = tmp_path / 'results'
    globals_before = diagnostics.Config, diagnostics.SoftRoleNet, diagnostics.NUMERICAL_ROOT
    namespaces_before = private_modules()
    diagnostics.prepare([synthetic_checkpoint], manifest,
        'Synthetic initialized fixture; bounded execution only', runtime='archived-adam',
        batch_mode='checkpoint-probe', probe_episodes=1, d5_instances=1, d5_bits=2, d5_samples=2)
    plan = diagnostics.execute(manifest, 0, output)
    report = json.loads((output / 'report.json').read_text())
    assert report['D1']['historical_update_verified'] is False
    assert report['D1']['parameter_update_l2'] > 0
    assert len(report['D6']['strata']) == 9 and plan['D6_policy_episodes'] == 9
    assert report['D5']['instances'][0]['status'] == 'ok'
    assert len(report['D5']['instances'][0]['enumerated_patterns']) == 4
    assert private_modules() == namespaces_before
    assert globals_before == (diagnostics.Config, diagnostics.SoftRoleNet, diagnostics.NUMERICAL_ROOT)
    assert digest(synthetic_checkpoint) == fixture_hash
    assert digest(checkpoint) == original_checkpoint_hash
    assert digest(original / 'source_manifest.json') == original_manifest_hash
    # Admission checks every original archived file again, rather than checking
    # only the subset copied into the numerical runtime.
    with archived_adam_runtime(checkpoint):
        pass
