"""Replay admission uses the archived learner and proves an intermediate state."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from softrole.config import Config
from softrole.recovery_replay import exact, sha, verify_source
from softrole.train import train

WORKER = Path(__file__).resolve().parents[1] / 'softrole/recovery_replay.py'


@pytest.fixture
def archive(tmp_path, request):
    run = tmp_path / 'historical'
    train(Config(model='shared', optimizer='adam', lr=.001, communication='real',
                 msg_dim=4, heads=2, head_dim=4, pre_dim=8, hidden_dim=8,
                 max_steps=2, epochs=3, updates_per_epoch=1, nprocesses=getattr(request, "param", 1),
                 batch_steps=1, save_every=1), run)
    return run


def execute(run, output):
    return subprocess.run([sys.executable, '-B', str(WORKER), '--run-dir', str(run),
        '--output', str(output), '--boundary-update', '2', '--anchor-before',
        str(run / 'checkpoints/epoch0001.pt'), '--anchor-after',
        str(run / 'checkpoints/epoch0003.pt')], text=True, capture_output=True)


@pytest.mark.parametrize("archive", [1, 4], indirect=True)
def test_exact_boundary_and_archive_immutable(archive, tmp_path):
    before = {str(p): sha(p) for p in archive.rglob('*') if p.is_file()}
    output = tmp_path / 'replayed'
    result = execute(archive, output)
    assert result.returncode == 0, result.stderr
    boundary = torch.load(output / 'boundary.pt', weights_only=False)
    original = torch.load(archive / 'checkpoints/epoch0002.pt', weights_only=False)
    assert exact(boundary['model_state'], original['model_state'])
    assert exact(boundary['optimizer_state'], original['optimizer_state'])
    assert boundary['total_steps'] == original['total_steps']
    assert json.loads((output / 'admission.json').read_text())['admitted']
    assert before == {str(p): sha(p) for p in archive.rglob('*') if p.is_file()}
    assert execute(archive, output).returncode != 0


@pytest.mark.parametrize('failure', ['runtime', 'episode', 'optimizer'])
def test_rejects_unprovable_replay(archive, tmp_path, failure):
    if failure == 'runtime':
        path = archive / 'run.json'
        record = json.loads(path.read_text())
        record['torch'] = '2.2.1+different-build'
        path.write_text(json.dumps(record))
    elif failure == 'episode':
        path = archive / 'episodes.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[1]['team_return'] += 1
        path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    else:
        path = archive / 'checkpoints/epoch0003.pt'
        payload = torch.load(path, weights_only=False)
        next(iter(payload['optimizer_state']['state'].values()))['exp_avg'].add_(1)
        torch.save(payload, path)
    output = tmp_path / 'rejected'
    result = execute(archive, output)
    assert result.returncode != 0
    assert not output.exists()
    assert {'runtime': 'exact recorded Torch/NumPy', 'episode': 'episode replay failed',
            'optimizer': 'later-anchor model/Adam/count'}[failure] in result.stderr


def test_source_mutation_and_tensor_type_rejected(archive):
    assert not exact(torch.ones(1), torch.ones(1, dtype=torch.float64))
    (archive / 'source/softrole/model.py').write_text('changed')
    with pytest.raises(ValueError, match='source hash mismatch'):
        verify_source(archive)
