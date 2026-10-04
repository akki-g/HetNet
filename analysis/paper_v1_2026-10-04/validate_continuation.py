"""Actual archived outer-CLI continuation checks; no performance/quality claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from publication_reconstruction.artifacts import load_checkpoint, verify_run
from publication_reconstruction.runtime.hetnet_ext.signatures import tree_signature


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(left, right, path='root'):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            same(left[key], right[key], path + '.' + str(key))
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, f'{path}[{index}]')
    else:
        assert left == right, path


def run(command, log):
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'DGLBACKEND': 'pytorch'}
    begin = time.monotonic()
    with log.open('x') as stream:
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    if completed.returncode:
        raise RuntimeError(f'Command failed with {completed.returncode}; see {log}')
    return time.monotonic() - begin


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    output, report_path = args.output.resolve(), args.report.resolve()
    if report_path.exists():
        raise FileExistsError(report_path)
    output.mkdir(parents=True, exist_ok=False)
    script_hash = digest(__file__)
    results = []
    for backend in ('dgl', 'torch-v1'):
        full_run, resumed_run = output / (backend + '_full'), output / (backend + '_resumed')
        train_command = [sys.executable, '-m', 'publication_reconstruction', 'train',
            '--reconstruction-spec', 'paper-v1', '--task', 'pcp', '--variant', 'binary',
            '--message-backend', backend, '--seed', '991', '--epochs', '1',
            '--updates-per-epoch', '3', '--collectors', '4', '--batch-steps', '7',
            '--horizon', '6', '--save-every', '1', '--milestones', '1', '--output', str(full_run)]
        full_seconds = run(train_command, output / (backend + '_full.log'))
        full_status = json.loads((full_run / 'run_status.json').read_text())
        expected = load_checkpoint(full_run, full_status['checkpoint'])
        milestones = list((full_run / 'checkpoints').rglob('*_steps1.pt'))
        assert len(milestones) == 1
        selected = milestones[0]
        selected_digest = digest(selected)
        partial = load_checkpoint(full_run, selected, require_recovery=True)
        assert partial['recovery']['counts']['updates'] == 1
        assert partial['recovery']['completed_epochs'] == 0
        assert partial['recovery']['updates_in_epoch'] == 1
        full_protocol, full_manifest = verify_run(full_run)
        resume_command = [sys.executable, '-m', 'publication_reconstruction', 'resume',
            '--run-dir', str(full_run), '--checkpoint', str(selected), '--output', str(resumed_run)]
        resume_seconds = run(resume_command, output / (backend + '_resumed.log'))
        resumed_status = json.loads((resumed_run / 'run_status.json').read_text())
        actual = load_checkpoint(resumed_run, resumed_status['checkpoint'])
        resumed_protocol, resumed_manifest = verify_run(resumed_run)
        verify_run(full_run)
        assert digest(selected) == selected_digest
        assert (full_run / 'source_manifest.json').read_bytes() == (resumed_run / 'source_manifest.json').read_bytes()
        same(full_manifest, resumed_manifest, 'source_manifest')
        for field in ('message_backend', 'reconstruction_spec', 'model_spec', 'env_version',
                      'learner_spec', 'seed', 'collectors', 'optimizer'):
            same(full_protocol[field], resumed_protocol[field], 'protocol.' + field)
        assert resumed_protocol['message_backend'] == backend
        assert actual['reconstruction']['message_backend'] == backend
        assert resumed_protocol['continuation']['checkpoint_sha256'] == selected_digest
        for field in ('policy_net', 'trainer', 'log'):
            same(expected[field], actual[field], field)
        for field in ('counts', 'rng_states', 'recorder_state', 'milestones_reached',
                      'completed_epochs', 'updates_in_epoch'):
            same(expected['recovery'][field], actual['recovery'][field], 'recovery.' + field)
        assert len(actual['recovery']['rng_states']) == 4
        assert actual['recovery']['counts']['updates'] == 3
        same(full_status['counts'], resumed_status['counts'], 'status.counts')
        resumed_updates = rows(resumed_run / 'updates.jsonl')
        assert [row['update'] for row in resumed_updates] == [2, 3]
        assert [row['update_in_epoch'] for row in resumed_updates] == [2, 3]
        reference_episodes = [row for row in rows(full_run / 'episodes.jsonl') if row['update'] > 1]
        continued_episodes = rows(resumed_run / 'episodes.jsonl')
        for left, right in zip(reference_episodes, continued_episodes):
            excluded = {'rollout_wall_time_seconds'}
            same({key: value for key, value in left.items() if key not in excluded},
                 {key: value for key, value in right.items() if key not in excluded}, 'episode')
        assert len(reference_episodes) == len(continued_episodes)
        assert actual['trainer']['state']
        for state in actual['trainer']['state'].values():
            for value in state.values():
                assert torch.isfinite(value).all()
        result = dict(backend=backend, full_run=str(full_run), continued_run=str(resumed_run),
            training_command=train_command, continuation_command=resume_command,
            full_command_seconds=full_seconds, continuation_command_seconds=resume_seconds,
            selected_checkpoint=str(selected), selected_checkpoint_sha256=selected_digest,
            full_checkpoint=full_status['checkpoint'], full_checkpoint_sha256=digest(full_status['checkpoint']),
            continued_checkpoint=resumed_status['checkpoint'], continued_checkpoint_sha256=digest(resumed_status['checkpoint']),
            counts=actual['recovery']['counts'], selected_counts=partial['recovery']['counts'],
            final_model=tree_signature(actual['policy_net']), final_active_optimizer=tree_signature(actual['trainer']),
            source_manifest_sha256=digest(full_run / 'source_manifest.json'),
            source_files=len(full_manifest['files']), continued_episode_records=len(continued_episodes),
            exact_model_optimizer_log_counts_recorder_all_rng=True,
            inherited_backend_and_source=True, selected_checkpoint_bytes_unchanged=True,
            excluded_episode_fields=['rollout_wall_time_seconds'])
        results.append(result)
        print('passed archived continuation', backend, actual['recovery']['counts'], flush=True)
    assert results[0]['source_manifest_sha256'] == results[1]['source_manifest_sha256']
    assert digest(__file__) == script_hash
    report = dict(passed=True, script=str(Path(__file__).resolve()), script_sha256=script_hash,
        specification='paper-v1 / paper-equations-v1', workload='PCP Binary; seed991; four collectors; horizon6; floor7',
        full_runs=2, continued_segments=2, executed_updates=10, rows=results,
        limitation='Bounded correctness and source-inheritance checks; no learning-quality or throughput estimate.')
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
