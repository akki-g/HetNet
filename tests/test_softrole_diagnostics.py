"""End-to-end offline replay: compare a real next update, preserve all inputs."""
import json
from pathlib import Path

import pytest
import torch

from softrole.config import Config
from softrole.diagnostics import (audit_archive, batch_diagnostics, d6_panels, execute,
                                   load_checkpoint, prepare, sha)
from softrole.train import train


def fixture_run(tmp_path, channel='real', optimizer='adam'):
    cfg = Config(model='shared', optimizer=optimizer, lr=.001, communication=channel,
                 comm_rounds=3, independent_heads=True, msg_dim=3, heads=2, head_dim=3,
                 pre_dim=5, hidden_dim=5, max_steps=4, epochs=2, updates_per_epoch=1,
                 nprocesses=1, batch_steps=5, save_every=1)
    train(cfg, tmp_path / 'run')
    return tmp_path / 'run' / 'checkpoints' / 'epoch0001.pt'


@pytest.mark.parametrize('channel,optimizer', [('real', 'adam'), ('binary', 'rmsprop')])
def test_d1_d2_reconstruct_actual_next_update_without_changing_inputs(tmp_path, channel, optimizer):
    checkpoint = fixture_run(tmp_path, channel, optimizer)
    manifest = prepare([checkpoint], tmp_path / 'manifest.json', 'first saved epoch; engineering fixture',
                       probe_episodes=1, d5_instances=1, d5_bits=2, d5_samples=32)
    before = {p: sha(p) for p in (tmp_path / 'run').rglob('*') if p.is_file()}
    plan = execute(tmp_path / 'manifest.json', 0, tmp_path / 'out', dry_run=True)
    assert plan['training_steps_persisted'] == 0 and not (tmp_path / 'out').exists()
    execute(tmp_path / 'manifest.json', 0, tmp_path / 'out')
    report = json.loads((tmp_path / 'out' / 'report.json').read_text())
    actual = torch.load(checkpoint.with_name('epoch0002.pt'), weights_only=False)
    assert report['D1']['disposable_post_update_signature'] == actual['signature']['sha256']
    assert report['D1']['historical_update_verified']
    assert report['D1']['parameter_update_l2'] > 0
    assert report['D2']['recomputed']['agent_steps'] == 3 * sum(e['steps'] for e in report['episodes'])
    assert report['D2']['recomputed']['kl'] > 0
    assert report['D3']['overall']['episodes'] == len(report['episodes'])
    assert report['D3']['overall']['proxy_identity_max_abs_error'] < 1e-12
    assert report['D4']['variants']['allocated_gae_beta_1']['mean_difference_from_team_gae_l2'] < 1e-10
    assert len(report['D6']['strata']) == 9
    assert {tuple(r['composition']) for r in report['D6']['strata']} == {(1, 2), (2, 2), (3, 1)}
    assert all(r['scheduled_exposed'] == 1 for r in report['D6']['strata'] if r['condition'] != 'nominal')
    assert bool(report['D5']['instances']) == (channel == 'binary')
    assert before == {p: sha(p) for p in (tmp_path / 'run').rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        execute(tmp_path / 'manifest.json', 0, tmp_path / 'out')


def test_terminal_checkpoint_requires_explicit_probe_mode(tmp_path):
    checkpoint = fixture_run(tmp_path).with_name('epoch0002.pt')
    with pytest.raises(ValueError, match='No complete next-update'):
        prepare([checkpoint], tmp_path / 'missing.json', 'terminal fixture')
    assert not (tmp_path / 'missing.json').exists()
    prepare([checkpoint], tmp_path / 'manifest.json', 'terminal fixture',
            batch_mode='checkpoint-probe', probe_episodes=1, d5_instances=1)
    execute(tmp_path / 'manifest.json', 0, tmp_path / 'out')
    report = json.loads((tmp_path / 'out' / 'report.json').read_text())
    assert not report['D1']['historical_update_verified']
    assert report['batch_mode'] == 'checkpoint-probe'


def test_changed_locked_ledger_and_reserved_panel_rejected_before_output(tmp_path):
    checkpoint = fixture_run(tmp_path)
    manifest_path = tmp_path / 'manifest.json'
    manifest = prepare([checkpoint], manifest_path, 'fixture')
    ledger = checkpoint.parent.parent / 'episodes.jsonl'
    original = ledger.read_bytes()
    ledger.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='Locked input hash changed'):
        execute(manifest_path, 0, tmp_path / 'out', dry_run=True)
    ledger.write_bytes(original)
    manifest['compositions'] = [[3, 2]]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsupported diagnostic protocol'):
        execute(manifest_path, 0, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_archive_tamper_and_signature_tamper_rejected(tmp_path):
    checkpoint = fixture_run(tmp_path)
    saved, cfg, _, _ = load_checkpoint(checkpoint)
    source = checkpoint.parent.parent / 'source' / 'softrole' / 'model.py'
    original = source.read_bytes()
    source.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='Archived source hash mismatch'):
        audit_archive(checkpoint, saved, cfg)
    source.write_bytes(original)
    sidecar = Path(str(checkpoint) + '.signature.json')
    sidecar.write_text('{}')
    with pytest.raises(ValueError, match='signature sidecar'):
        load_checkpoint(checkpoint)


def test_d6_panels_replay_and_only_one_sensor_loss_is_explicit():
    panels = d6_panels(971, 10)
    assert panels == d6_panels(971, 10)
    for composition, nominal, event in panels:
        for n, e in zip(nominal, event):
            assert n.env_seed == e.env_seed and n.action_seed == e.action_seed and n.message_seed == e.message_seed
            assert e.event_step == 1 and 0 <= e.victim < composition[0]
            assert n.event_step == -1
        assert len(nominal) == 10
    assert all(s.victim == 0 for s in panels[0][2])


@pytest.mark.parametrize('samples', [1, 65537])
def test_invalid_d5_sample_count_rejected_before_archive_reads(tmp_path, samples):
    with pytest.raises(ValueError, match='D5 samples'):
        prepare(['missing.pt'], tmp_path / 'manifest.json', 'fixture', d5_samples=samples)


def test_nonfinite_and_missing_optimizer_moments_fail_preparation(tmp_path):
    checkpoint = fixture_run(tmp_path)
    original = checkpoint.read_bytes()
    saved = torch.load(checkpoint, weights_only=False)
    first = next(iter(saved['optimizer_state']['state'].values()))
    first['exp_avg'].flatten()[0] = float('nan')
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='finite tensors'):
        load_checkpoint(checkpoint)
    checkpoint.write_bytes(original)
    saved = torch.load(checkpoint, weights_only=False)
    saved['optimizer_state']['state'].pop(next(iter(saved['optimizer_state']['state'])))
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='Incomplete or misaligned'):
        load_checkpoint(checkpoint)


def test_forged_episode_metrics_fail_exact_historical_replay(tmp_path):
    checkpoint = fixture_run(tmp_path)
    saved, cfg, model, _ = load_checkpoint(checkpoint)
    manifest = prepare([checkpoint], tmp_path / 'manifest.json', 'fixture', probe_episodes=1)
    ledger = checkpoint.parent.parent / 'episodes.jsonl'
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    for row in rows:
        if row['update'] == 2:
            row['team_return'] += 1
    ledger.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='Exact archived episode replay failed'):
        batch_diagnostics(saved, cfg, model, manifest, checkpoint.parent.parent)


def test_evaluator_archive_must_match_locked_source(tmp_path, monkeypatch):
    import softrole.diagnostics as diagnostics
    checkpoint = fixture_run(tmp_path)
    manifest = tmp_path / 'manifest.json'
    prepare([checkpoint], manifest, 'fixture')
    original = diagnostics.source_snapshot

    def changed_at_archive(output=None):
        record = original(output)
        return {**record, 'sha256': '0' * 64} if output is not None else record

    monkeypatch.setattr(diagnostics, 'source_snapshot', changed_at_archive)
    with pytest.raises(RuntimeError, match='before archive capture'):
        execute(manifest, 0, tmp_path / 'out')
    assert not (tmp_path / 'out/report.json').exists()
