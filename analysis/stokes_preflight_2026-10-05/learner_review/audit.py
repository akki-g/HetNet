"""Read-only reconciliation of eight copied 100-update Stokes paper preflights.

Run with .venv/bin/python; no training or evaluator execution is performed.
Numerical ledgers compare exactly across duplicate arrays, except declared clocks.
Floating aggregation checks allow 1e-10 absolute/1e-12 relative summation roundoff.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from publication_reconstruction.artifacts import load_checkpoint, verify_run, _validate_optimizer
from publication_reconstruction.runtime.hetnet_ext.signatures import _entries, tree_signature
from publication_reconstruction.study import _validate_preflight_ledger


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def finite(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
        return value.numel()
    if isinstance(value, np.ndarray):
        assert np.isfinite(value).all()
        return value.size
    if isinstance(value, dict):
        return sum(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        assert math.isfinite(value)
    return 0


def close(a, b):
    assert np.allclose(a, b, atol=1e-10, rtol=1e-12), (a, b)


def strip_clock(records, clock):
    return [{k: v for k, v in row.items() if k != clock} for row in records]


def model_signature(state):
    result = dict(schema_version=1, parameters=_entries(state.items()), buffers=[])
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def probe_signature(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def diff_files(left, right):
    return {key: [left.get(key), right.get(key)] for key in sorted(left.keys() | right.keys())
            if left.get(key) != right.get(key)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    results, comparisons, stored, hashes = [], [], {}, {}
    for array in ('905118', '905404'):
        for workload in ('pp_real', 'pcp_real', 'fc_real', 'pcp_binary'):
            run = ROOT / 'stokes_runs' / ('hetnet_paper_preflight_' + array) / workload / 'seed991'
            protocol, manifest = verify_run(run)
            runtime_args = read(run / 'resolved_args.json')
            updates, epochs = rows(run / 'updates.jsonl'), rows(run / 'metrics.jsonl')
            batches = [json.loads(line) for line in (run / 'stdout.log').read_text().splitlines()
                       if line.startswith('{') and '"publication_episode_batch"' in line]
            episodes = [row for batch in batches for row in batch['episodes']]
            status, preflight, segment = [read(run / f) for f in
                ('run_status.json', 'preflight.json', 'training_segment.json')]
            checkpoint, = run.glob('checkpoints/**/*.pt')
            checkpoint_hash = digest(checkpoint)
            saved = load_checkpoint(run, checkpoint)
            _validate_optimizer(saved, protocol, 100)
            _validate_preflight_ledger(updates, status, segment, protocol, saved)
            counts = saved['reconstruction']['counts']
            assert counts == saved['recovery']['counts'] == status['counts'] == preflight['counts']
            assert counts['updates'] == 100 and counts['epoch'] == 10
            assert status['scientific_budget_completed'] and status['stop_reason'] == 'epoch_cap_completed'
            assert (run / 'exit_code.txt').read_text().strip() == '0'
            assert protocol['message_backend'] == 'torch-v1'
            assert protocol['reconstruction_spec'] == protocol['model_spec'] == protocol['env_version'] == 'paper-v1'
            assert protocol['learner_spec'] == 'paper-equations-v1'
            assert protocol['collectors'] == 4 and protocol['batch_step_floor_per_collector'] == 500
            assert runtime_args['gamma'] == 1 and runtime_args['detach_gap'] == 5
            assert runtime_args['lrate'] == .001 and runtime_args['entr'] == 0
            assert not runtime_args['profile_phases'] and not runtime_args['profile_memory']
            horizon = 300 if workload == 'fc_real' else 80
            assert protocol['episode_horizon'] == horizon
            assert len(updates) == len(batches) == 100 and len(epochs) == 10
            assert len(episodes) == counts['episodes'] and sum(e['steps'] for e in episodes) == counts['env_steps']
            finite(updates); finite(epochs); finite(episodes)
            assert [e['episode'] for e in episodes] == list(range(1, len(episodes) + 1))
            collector_intervals, minimum_gradient = [], float('inf')
            for index, (update, batch) in enumerate(zip(updates, batches), 1):
                assert update['update'] == batch['update'] == index
                records = batch['episodes']
                assert len(records) == update['episodes']
                assert sum(e['steps'] for e in records) == update['steps']
                assert update['loss_units'] == 'global completed-episode mean of total-N weighted sums'
                assert update['value_loss'] >= 0 and update['gradient_norm_preclip'] > 0
                minimum_gradient = min(minimum_gradient, update['gradient_norm_preclip'])
                by_collector = defaultdict(list)
                for e in records:
                    assert e['update'] == index and e['epoch'] == update['epoch']
                    assert e['num_agents'] == len(e['reward_per_agent']) == 3
                    assert 1 <= e['steps'] <= horizon
                    assert e['success'] == e['terminated']
                    assert e['steps'] == horizon or e['terminated']
                    close(sum(e['reward_per_agent']), e['team_return'])
                    close(e['team_return'] / 3, e['mean_agent_return'])
                    if workload == 'fc_real':
                        close(e['reward_per_agent'][0], e['reward_per_agent'][1])
                        assert e['reward_per_agent'][2] <= e['reward_per_agent'][0] + 1e-10
                        delta = e['reward_per_agent'][0] - e['reward_per_agent'][2]
                        close(delta / .1, round(delta / .1))
                        assert delta <= .1 * e['steps'] + 1e-10
                    else:
                        for reward in e['reward_per_agent']:
                            assert -.05 * e['steps'] - 1e-10 <= reward <= 1e-10
                            close(reward / .05, round(reward / .05))
                    by_collector[e['collector']].append(e)
                assert set(by_collector) == {0, 1, 2, 3}
                for collector, part in by_collector.items():
                    assert [e['collector_episode'] for e in part] == list(range(len(part)))
                    steps = sum(e['steps'] for e in part)
                    assert 500 <= steps <= 500 + horizon - 1
                    assert steps - part[-1]['steps'] < 500
                    collector_intervals.append(steps)
                reward = np.mean([e['reward_per_agent'] for e in records], axis=0)
                close(reward, update['reward_per_agent'])
                close(reward.sum(), update['team_return'])
                close(reward.mean(), update['mean_agent_return'])
                close(sum(e['success'] for e in records) / len(records), update['success_rate'])
            for index, epoch in enumerate(epochs, 1):
                block = updates[(index - 1) * 10:index * 10]
                n = sum(u['episodes'] for u in block)
                assert epoch['epoch'] == index and epoch['updates'] == index * 10
                assert epoch['steps'] == sum(u['steps'] for u in block)
                assert epoch['episodes'] == n and epoch['total_steps'] == block[-1]['total_steps']
                assert epoch['total_episodes'] == block[-1]['total_episodes']
                for key in ('reward_per_agent', 'team_return', 'mean_agent_return', 'success_rate', 'policy_loss', 'value_loss'):
                    close(sum(np.asarray(u[key]) * u['episodes'] for u in block) / n, epoch[key])
                close(epoch['steps_taken'], epoch['steps'] / n)
                # The retained public stdout/checkpoint log uses per-step losses.
                close(saved['log']['value_loss'].data[index - 1], epoch['value_loss'] * n / epoch['steps'])
                close(saved['log']['action_loss'].data[index - 1], epoch['policy_loss'] * n / epoch['steps'])
                for source_key, target_key in [('reward', 'reward_per_agent'), ('success', 'success_rate'),
                        ('num_steps', 'steps'), ('num_episodes', 'episodes'), ('steps_taken', 'steps_taken')]:
                    close(saved['log'][source_key].data[index - 1], epoch[target_key])
            model_values, optimizer_values = finite(saved['policy_net']), finite(saved['trainer'])
            assert all(state['step'].item() == 100 for state in saved['trainer']['state'].values())
            epoch_signatures = rows(run / 'epoch_signatures.jsonl')
            initial_signature = read(run / 'initial_signature.json')
            signature = model_signature(saved['policy_net'])
            assert signature == read(str(checkpoint) + '.signature.json') == epoch_signatures[-1]['signature']
            assert all(a['signature']['sha256'] != b['signature']['sha256'] for a, b in zip(epoch_signatures, epoch_signatures[1:]))
            initial_by_name = {p['name']: p['sha256'] for p in initial_signature['parameters']}
            changed = [p['name'] for p in signature['parameters'] if p['sha256'] != initial_by_name[p['name']]]
            assert changed and len(saved['recovery']['rng_states']) == 4
            assert saved['recovery']['completed_epochs'] == 10 and saved['recovery']['updates_in_epoch'] == 0
            assert saved['recovery']['epoch_stat'] == {}
            recorder = saved['recovery']['recorder_state']
            assert recorder['total_steps'] == counts['env_steps'] and recorder['total_episodes'] == counts['episodes']
            assert recorder['last_epoch'] == 10 and recorder['steps'] == recorder['episodes'] == 0
            checkpoint_record, = rows(run / 'checkpoint_records.jsonl')
            assert checkpoint_record['counts'] == counts and checkpoint_record['checkpoint_sha256'] == checkpoint_hash
            assert checkpoint_record['bytes'] == checkpoint.stat().st_size
            assert checkpoint_record['parameter_sha256'] == signature['sha256']
            probe = read(run / 'checkpoint_probe/report.json')
            assert probe['checkpoint_sha256'] == checkpoint_hash and probe['parameters_unchanged']
            assert probe['model_signature'] == probe_signature(saved['policy_net'])
            assert probe['episodes'] == len(probe['per_episode']) == 1
            assert probe['config']['message_backend'] == 'torch-v1' and probe['training_seed'] == 991
            assert probe['checkpoint_progress'] == dict(epoch=10, updates=100, total_steps=counts['env_steps'], total_episodes=counts['episodes'])
            finite(probe['per_episode'])
            per_episode, = probe['per_episode']
            assert 1 <= per_episode['steps'] <= horizon
            close(per_episode['team_return'], sum(per_episode['agent_returns']))
            close(probe['mean_team_return'], per_episode['team_return'])
            close(probe['success_rate'], float(per_episode['success']))
            evaluation_archive = run / 'checkpoint_probe/report.json.sources'
            evaluation_manifest = read(evaluation_archive / 'manifest.json')
            assert evaluation_manifest['source'] == probe['evaluator']['source']
            for name, expected in evaluation_manifest['source']['files'].items():
                assert digest(evaluation_archive / name) == expected
            assert digest(evaluation_archive / 'manifest.json') == probe['evaluator']['source_archive']['manifest_sha256']
            assert probe['source_sha256'] == digest(run / 'source_manifest.json')
            prior = ROOT / 'stokes_runs/hetnet_backend_benchmark_904547/throughput' / workload / 'pair1/torch-v1'
            prior_manifest = read(prior / 'source_manifest.json')
            source_differences = diff_files(manifest['files'], prior_manifest['files'])
            assert not source_differences
            initial = read(run / 'initial_training_identity.json')
            prior_initial = read(prior / 'initial_training_identity.json')
            assert all(initial[k] == prior_initial[k] for k in ('model', 'optimizer', 'rng'))
            prior_updates = rows(prior / 'updates.jsonl')
            prior_numerical = strip_clock(prior_updates, 'wall_time_seconds')
            prefix_equal = strip_clock(updates[:len(prior_updates)], 'wall_time_seconds') == prior_numerical
            assert prefix_equal
            final_identity = {key: tree_signature(saved[key]) for key in ('policy_net', 'trainer', 'log')}
            final_identity.update({key: tree_signature(saved['recovery'][key]) for key in
                ('counts', 'rng_states', 'recorder_state', 'milestones_reached', 'completed_epochs', 'updates_in_epoch')})
            gradients = [u['gradient_norm_preclip'] for u in updates]
            result = dict(array=array, workload=workload, run=str(run.relative_to(ROOT)), counts=counts,
                backend='torch-v1', archived_source_files_verified=len(manifest['files']),
                source_manifest_sha256=digest(run / 'source_manifest.json'), source_differences_from_benchmark=source_differences,
                exact_initial_model_optimizer_rng_and_20_update_benchmark_prefix=True,
                ledger_updates=len(updates), ledger_epochs=len(epochs), ledger_episodes=len(episodes),
                all_counts_rewards_loss_aggregation_reconciled=True,
                collector_steps_min=min(collector_intervals), collector_steps_max=max(collector_intervals),
                all_recorded_losses_and_gradient_norms_finite=True,
                gradient_norm_min=min(gradients), gradient_norm_median=float(np.median(gradients)),
                gradient_norm_max=max(gradients), updates_clipped=sum(g > .75 for g in gradients),
                policy_loss_range=[min(u['policy_loss'] for u in updates), max(u['policy_loss'] for u in updates)],
                value_loss_range=[min(u['value_loss'] for u in updates), max(u['value_loss'] for u in updates)],
                episode_return_range=[min(e['mean_agent_return'] for e in episodes), max(e['mean_agent_return'] for e in episodes)],
                epoch_metrics=[{key: e[key] for key in ('epoch', 'steps', 'episodes', 'success_rate', 'mean_agent_return', 'policy_loss', 'value_loss')} for e in epochs],
                model_tensor_count=len(saved['policy_net']), changed_model_tensor_count=len(changed),
                active_adam_parameters=len(saved['trainer']['state']), all_active_adam_steps=100,
                finite_model_values=model_values, finite_optimizer_values=optimizer_values,
                parameter_dtypes=sorted({str(t.dtype) for t in saved['policy_net'].values()}),
                checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=checkpoint_hash,
                final_value_identities=final_identity,
                frozen_probe=dict(episodes=1, parameters_unchanged=True, checkpoint_hash_matches=True,
                    source_files_verified=len(evaluation_manifest['source']['files']),
                    success=per_episode['success'], steps=per_episode['steps'], mean_agent_return=probe['mean_agent_return']))
            results.append(result)
            stored[(array, workload)] = dict(updates=strip_clock(updates, 'wall_time_seconds'),
                epochs=strip_clock(epochs, 'wall_time_seconds'), episodes=strip_clock(episodes, 'rollout_wall_time_seconds'),
                signatures=epoch_signatures, identity=final_identity, initial={k: initial[k] for k in ('model', 'optimizer', 'rng')},
                probe=probe['per_episode'], manifest=manifest)
            for name in ('protocol.json', 'source_manifest.json', 'updates.jsonl', 'metrics.jsonl', 'stdout.log',
                    'epoch_signatures.jsonl', 'initial_signature.json', 'initial_training_identity.json',
                    'run_status.json', 'preflight.json', 'checkpoint_records.jsonl', 'checkpoint_probe/report.json'):
                hashes[str((run / name).relative_to(ROOT))] = digest(run / name)
            hashes[str(checkpoint.relative_to(ROOT))] = checkpoint_hash
            assert digest(checkpoint) == checkpoint_hash
            print('audited', array, workload, counts, flush=True)
    for workload in ('pp_real', 'pcp_real', 'fc_real', 'pcp_binary'):
        left, right = stored[('905118', workload)], stored[('905404', workload)]
        exact = {key: left[key] == right[key] for key in left}
        assert all(exact.values()), (workload, exact)
        comparisons.append(dict(workload=workload, exact_equal=exact))
    report = dict(passed=True, script_sha256=digest(__file__), runs=results, duplicate_array_comparisons=comparisons,
        input_sha256=hashes, numerical_comparison_exclusions=['updates/metrics.wall_time_seconds', 'episodes.rollout_wall_time_seconds',
            'checkpoint paths/source/runtime/timing fields are not compared as numerical state; model/Adam/log/all RNG/count/recorder values are compared exactly'],
        reporting_cautions=['Both arrays use torch-v1; neither is an additional DGL versus Torch performance comparison.',
            'Arrays repeat seed991 and exact data/model trajectories; they are not independent training seeds.',
            'Public checkpoint log/stdout loss fields retain per-step normalization; use paper-labelled updates.jsonl/metrics.jsonl for global-episode-mean losses.',
            'Legacy log entropy is a placeholder zero, not measured policy entropy.',
            'One frozen scenario establishes load/execution/immutability, not task quality.',
            'No actions or intermediate gradient tensors are persisted in these 100-update ledgers; finite global preclip norms and checkpoint/optimizer finiteness are checked, not independent per-update gradient recomputation.',
            'Large FC loss/gradient magnitudes reflect a distinct reward scale and unaveraged time sums; these data alone neither establish a defect nor validate eventual convergence.'],
        execution='Only artifact loading, hashes, value comparisons, and recorded-source inspection; no training/evaluation or Slurm jobs executed.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
