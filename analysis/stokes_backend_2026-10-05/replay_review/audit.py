"""Read-only audit of copied Stokes replay evidence; does not rerun training."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
        return value.numel()
    if isinstance(value, np.ndarray):
        assert np.isfinite(value).all()
        return value.size
    if isinstance(value, dict):
        return sum(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        assert math.isfinite(value)
    return 0


def rng_equal(left, right):
    return (left['python'] == right['python'] and
            left['numpy'][0] == right['numpy'][0] and
            np.array_equal(left['numpy'][1], right['numpy'][1]) and
            left['numpy'][2:] == right['numpy'][2:] and
            torch.equal(left['torch'], right['torch']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    assert not args.output.exists()
    torch.set_num_threads(1)
    combined = json.loads((root / 'summary.json').read_text())
    results = []
    inputs = {str(root / 'summary.json'): digest(root / 'summary.json')}
    for workload in ('pp_real', 'pcp_real', 'fc_real', 'pcp_binary'):
        task, variant = workload.split('_')
        replay_root = root / 'replay' / workload
        summary_path, fixture_path = replay_root / 'summary.json', replay_root / 'fixture.pt'
        summary = json.loads(summary_path.read_text())
        fixture_hash = digest(fixture_path)
        assert fixture_hash == summary['fixture_sha256']
        inputs[str(fixture_path)] = fixture_hash
        inputs[str(summary_path)] = digest(summary_path)
        archive = root / 'throughput' / workload / 'pair1/dgl/source'
        source_mismatches = []
        for name, expected in summary['source_sha256'].items():
            actual = digest(archive / 'runtime' / name)
            if actual != expected:
                source_mismatches.append(name)
        assert not source_mismatches
        signature_file = archive / 'runtime/hetnet_ext/signatures.py'
        spec = importlib.util.spec_from_file_location('audited_signatures', signature_file)
        signatures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(signatures)
        fixture = torch.load(fixture_path, map_location='cpu')
        assert len(fixture['updates']) == summary['updates'] == 3
        assert summary['collectors'] == fixture['args']['nprocesses'] == 4
        assert summary['batch_step_floor_per_collector'] == fixture['args']['batch_size'] == 500
        assert summary['horizon'] == fixture['args']['max_steps'] == (300 if task == 'fc' else 80)
        assert summary['seed'] == fixture['args']['seed'] == 991
        assert fixture['args']['model_spec'] == fixture['args']['publication_env_version'] == 'paper-v1'
        assert fixture['args']['learner_spec'] == 'paper-equations-v1'
        assert not fixture['args']['profile_phases'] and not fixture['args']['profile_memory']
        assert fixture['args']['detach_gap'] == 5 and fixture['args']['lrate'] == .001
        assert fixture['initial_optimizer']['state'] == {}
        assert fixture['final_optimizer']['state']
        finite_model_values = finite_tree(fixture['initial_model']) + finite_tree(fixture['final_model'])
        finite_optimizer_values = finite_tree(fixture['initial_optimizer']) + finite_tree(fixture['final_optimizer'])
        assert fixture['initial_model'].keys() == fixture['final_model'].keys()
        changed = []
        for name, initial in fixture['initial_model'].items():
            final = fixture['final_model'][name]
            assert initial.dtype == final.dtype and initial.shape == final.shape
            if not torch.equal(initial, final):
                changed.append(name)
        assert changed
        for state in fixture['final_optimizer']['state'].values():
            assert state['step'].item() == 3
        initial_identity = dict(model=signatures.tree_signature(fixture['initial_model']),
                                optimizer=signatures.tree_signature(fixture['initial_optimizer']))
        for pair in range(1, 4):
            for backend in ('dgl', 'torch-v1'):
                identity_path = root / 'throughput' / workload / f'pair{pair}' / backend / 'initial_training_identity.json'
                identity = json.loads(identity_path.read_text())
                assert all(identity[key] == value for key, value in initial_identity.items())
                inputs[str(identity_path)] = digest(identity_path)
        update_rows, torch_changed = [], 0
        all_rewards, total_episodes = set(), 0
        for update in fixture['updates']:
            assert len(update) == 4
            collectors = []
            for episodes in update:
                lengths = [len(episode) for episode in episodes]
                assert lengths and min(lengths) > 0 and max(lengths) <= summary['horizon']
                count = sum(lengths)
                assert 500 <= count <= 500 + summary['horizon'] - 1
                collectors.append(dict(steps=count, episodes=len(episodes), minimum_length=min(lengths),
                    maximum_length=max(lengths), episodes_crossing_detach_gap=sum(length > 5 for length in lengths)))
                total_episodes += len(episodes)
                for episode in episodes:
                    for step in episode:
                        assert set(step) == {'observation', 'action', 'reward', 'rng_before', 'rng_after_forward'}
                        obs = step['observation']
                        assert obs.shape == (1, 3, 29 * (9 if task == 'fc' else 25))
                        assert obs.dtype == torch.float64 and torch.isfinite(obs).all()
                        reward = np.asarray(step['reward'])
                        assert reward.shape == (3,) and np.isfinite(reward).all()
                        all_rewards.update(map(float, reward))
                        action = np.asarray(step['action'])
                        assert action.shape == (1, 3)
                        assert np.isfinite(action).all() and np.equal(action, np.floor(action)).all()
                        dimensions = ([4, 4, 5] if task == 'fc' else [5, 5, 5] if task == 'pp' else [5, 5, 6])
                        assert ((action[0] >= 0) & (action[0] < dimensions)).all()
                        before, after = step['rng_before'], step['rng_after_forward']
                        assert set(before) == set(after) == {'python', 'numpy', 'torch'}
                        assert before['python'] == after['python']
                        assert before['numpy'][0] == after['numpy'][0]
                        assert np.array_equal(before['numpy'][1], after['numpy'][1])
                        assert before['numpy'][2:] == after['numpy'][2:]
                        assert before['torch'].dtype == after['torch'].dtype == torch.uint8
                        is_changed = not torch.equal(before['torch'], after['torch'])
                        assert is_changed == (variant == 'binary')
                        torch_changed += is_changed
            update_rows.append(dict(steps=sum(row['steps'] for row in collectors), collectors=collectors))
        total_steps = sum(row['steps'] for row in update_rows)
        assert len(fixture['final_environment_streams']) == 4
        replay_pairs = []
        for index, pair in enumerate(summary['pairs']):
            assert pair['repetition'] == index + 1
            assert pair['order'] == (['dgl', 'torch-v1'] if index % 2 == 0 else ['torch-v1', 'dgl'])
            a, b = pair['runs']['dgl'], pair['runs']['torch-v1']
            ratio = a['wall_seconds'] / b['wall_seconds']
            assert ratio == pair['speedup'] and ratio > 1
            for run in (a, b):
                assert run['steps'] == total_steps
                assert len(run['update_seconds']) == 3
                assert run['first_update_seconds'] == run['update_seconds'][0]
                assert run['wall_seconds'] >= sum(run['update_seconds']) > 0
                assert run['steps_per_second'] == total_steps / run['wall_seconds']
                assert run['user_cpu_seconds'] > 0 and run['system_cpu_seconds'] >= 0
            replay_pairs.append(dict(pair=index + 1, reported_speedup=ratio,
                speedup_excluding_first_update=sum(a['update_seconds'][1:]) / sum(b['update_seconds'][1:]),
                process_cpu_ratio=(a['user_cpu_seconds'] + a['system_cpu_seconds']) /
                                  (b['user_cpu_seconds'] + b['system_cpu_seconds']),
                maximum_interupdate_unaccounted_seconds=max(
                    run['wall_seconds'] - sum(run['update_seconds']) for run in (a, b))))
        assert len(replay_pairs) == 3
        assert summary['correctness_passed'] is True
        assert summary['tolerances'] == {'float32': {'atol': 1e-6, 'rtol': 1e-5},
                                         'float64': {'atol': 1e-10, 'rtol': 1e-8}}
        log_path = root / 'logs' / f'replay_{workload}.log'
        log = log_path.read_text()
        assert 'Traceback' not in log and 'AssertionError' not in log
        receipt = json.loads(next(line for line in reversed(log.splitlines()) if line.startswith('{')))
        assert receipt['correctness_passed'] is True
        assert receipt['speedups'] == [pair['speedup'] for pair in summary['pairs']]
        inputs[str(log_path)] = digest(log_path)
        combined_row = next(row for row in combined['comparisons'] if row['task'] == task and row['variant'] == variant)
        assert combined_row['compute_gate'] == all(pair['speedup'] > 1 for pair in summary['pairs'])
        assert combined_row['throughput_gate'] == (all(pair['update_throughput_ratio'] > 1 for pair in combined_row['pairs']) and
                                                 combined_row['ratio_of_median_segment_throughput'] > 1)
        assert combined_row['compute_replay']['speedups'] == receipt['speedups']
        median_speedup = statistics.median(pair['reported_speedup'] for pair in replay_pairs)
        results.append(dict(workload=workload, fixture_sha256=fixture_hash, fixture_bytes=fixture_path.stat().st_size,
            fixture_steps=total_steps, fixture_episodes=total_episodes, update_partitions=update_rows,
            all_observations_rewards_actions_finite_and_actions_legal=True,
            initial_model_and_optimizer_match_all_six_throughput_runs=True,
            final_model_tensors_changed=len(changed), active_optimizer_parameters=len(fixture['final_optimizer']['state']),
            finite_model_values_checked=finite_model_values, finite_optimizer_values_checked=finite_optimizer_values,
            all_active_optimizer_steps=3, within_forward_torch_rng_changed_count=torch_changed,
            within_forward_python_numpy_unchanged=True, observed_reward_values=sorted(all_rewards),
            archived_python_source_files_hash_checked=len(summary['source_sha256']),
            reported_maximum_absolute_differences=summary['maximum_absolute_differences'],
            replay_pairs=replay_pairs, median_replay_speedup=median_speedup,
            median_replay_wall_time_reduction_percent=100 * (1 - 1 / median_speedup),
            throughput_gate_reconciled=combined_row['throughput_gate'], compute_gate_reconciled=combined_row['compute_gate'],
            summary_path=str(summary_path), log_path=str(log_path), archive_runtime=str(archive / 'runtime')))
        assert digest(fixture_path) == fixture_hash
        del fixture
        gc.collect()
        print('audited', workload, total_steps, total_episodes, median_speedup, flush=True)
    report = dict(passed=True, source=str(root), workloads=results, input_sha256=inputs,
        execution='Loaded/copied artifact values and inspected archived validation code; did not rerun model forward/backward or training.',
        assertion_coverage=dict(exact=['DGL replay final model/Adam versus recorded collection', 'per-forward Python/NumPy/Torch RNG',
            'gradient presence, tensor shapes/dtypes and dictionary schema'], tolerance_based=['all forward actor/state/critic/hidden tensors',
            'collector losses and raw gradients', 'globally clipped gradients and norm', 'model and active Adam state after each of three updates'],
            not_directly_checked=['new action samples/environment outcomes during replay (fixed inputs)', '20-update trajectory identity',
                'input gradients in this replay', 'per-backend replay RSS (process-lifetime high-water mark)',
                'long-training quality or original historical learner equivalence']),
        limitations=['The four Stokes correctness flags/maxima were not numerically recomputed on another architecture; archived code shows checks precede report publication.',
            'Replay is serial collector compute, including RNG restoration/Python iteration; it excludes environment/action sampling/IPC and is separate from four-process throughput.',
            'ru_maxrss per timing pass includes earlier fixture/correctness peaks in the same process; equal readings cannot establish equal backend memory.',
            'A three-update correctness fixture and three same-seed repetitions do not establish task convergence or between-seed uncertainty.'])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
