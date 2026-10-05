"""Recompute the Stokes backend benchmark from raw update ledgers.

Reads copied inputs without changing archived paths/checkpoints. Three timing
repetitions share one training seed; ranges are descriptive, not confidence
intervals. Projections hold this early-run segment throughput constant.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


WORKLOADS = (("pp", "real", 40_000_000), ("pcp", "real", 40_000_000),
             ("fc", "real", 28_000_000), ("pcp", "binary", 40_000_000))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def read(path, jsonl=False):
        payload = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(payload).hexdigest()
        return [json.loads(line) for line in payload.splitlines()] if jsonl else json.loads(payload)

    recorded = read(args.input / 'summary.json')
    plan = read(args.input / 'plan.json')
    hardware = read(args.input / 'hardware.json')
    rows = []
    for index, job in enumerate(plan['jobs']):
        result = read(args.input / f'result_{index:02d}.json')
        assert all(result[key] == job[key] for key in job)
        directory = args.input / ('diagnostics' if job['profiled'] else 'throughput')
        directory /= f"{job['task']}_{job['variant']}"
        if not job['profiled']:
            directory /= f"pair{job['pair']}"
        directory /= job['backend']
        updates = read(directory / 'updates.jsonl', True)
        status = read(directory / 'run_status.json')
        protocol = read(directory / 'protocol.json')
        assert len(updates) == job['updates'] == status['counts']['updates']
        assert [row['update'] for row in updates] == list(range(1, len(updates) + 1))
        for field, count_key in [('steps', 'env_steps'), ('episodes', 'episodes')]:
            assert sum(row[field] for row in updates) == status['counts'][count_key]
        assert result['counts'] == status['counts']
        assert protocol['collectors'] == 4 and protocol['batch_step_floor_per_collector'] == 500
        assert protocol['episode_horizon'] == (300 if job['task'] == 'fc' else 80)
        assert protocol['seed'] == 991
        assert protocol['message_backend'] == job['backend']
        assert protocol['reconstruction_spec'] == protocol['model_spec'] == protocol['env_version'] == 'paper-v1'
        assert protocol['learner_spec'] == 'paper-equations-v1'
        assert all(('phases' in row) == job['profiled'] for row in updates)
        steps = status['counts']['env_steps']
        derived = dict(
            post_first_update_steps_per_second=sum(row['steps'] for row in updates[1:]) /
                sum(row['wall_time_seconds'] for row in updates[1:]),
            all_update_steps_per_second=steps / sum(row['wall_time_seconds'] for row in updates),
            segment_steps_per_second=steps / status['segment_wall_time_seconds'],
            process_steps_per_second=steps / result['process_seconds'])
        for key, value in derived.items():
            assert math.isclose(result[key], value, rel_tol=1e-12, abs_tol=1e-12), (index, key)
        rows.append({**result, **derived, 'local_directory': str(directory)})

    workloads = []
    for task, variant, budget in WORKLOADS:
        selected = [row for row in rows if not row['profiled'] and
                    (row['task'], row['variant']) == (task, variant)]
        replay = read(args.input / 'replay' / f'{task}_{variant}' / 'summary.json')
        pairs = []
        backends = {}
        for backend in ('dgl', 'torch-v1'):
            group = [row for row in selected if row['backend'] == backend]
            assert len(group) == 3
            rates = [row['post_first_update_steps_per_second'] for row in group]
            segment_rates = [row['segment_steps_per_second'] for row in group]
            hours = [budget / rate / 3600 for rate in segment_rates]
            backends[backend] = dict(post_first_update_steps_per_second=rates,
                median_post_first_update_steps_per_second=statistics.median(rates),
                segment_steps_per_second=segment_rates,
                median_segment_steps_per_second=statistics.median(segment_rates),
                projected_budget_hours=budget / statistics.median(segment_rates) / 3600,
                projected_budget_hours_range=[min(hours), max(hours)],
                projected_100_update_segment_minutes=statistics.median(row['segment_seconds'] for row in group) * 5 / 60,
                startup_seconds=[row['startup_seconds'] for row in group],
                first_update_seconds=[row['first_update_seconds'] for row in group])
        for repetition in (1, 2, 3):
            pair = {row['backend']: row for row in selected if row['pair'] == repetition}
            a, b = pair['dgl'], pair['torch-v1']
            assert a['initial_identity'] == b['initial_identity']
            assert a['source_manifest_sha256'] == b['source_manifest_sha256']
            speed = b['post_first_update_steps_per_second'] / a['post_first_update_steps_per_second']
            pairs.append(dict(pair=repetition, update_speedup=speed,
                update_time_reduction_fraction=1 - 1 / speed,
                segment_speedup=b['segment_steps_per_second'] / a['segment_steps_per_second'],
                same_episode_ledger=a['episode_ledger_identity'] == b['episode_ledger_identity'],
                same_final_rng=a['final_identity']['rng'] == b['final_identity']['rng'],
                same_counts=a['counts'] == b['counts'],
                full_trajectory_verified=a['trajectory_identity_complete'] and b['trajectory_identity_complete']))
        ratios = [pair['update_speedup'] for pair in pairs]
        median_ratio = statistics.median(ratios)
        segment_ratio = (backends['torch-v1']['median_segment_steps_per_second'] /
                         backends['dgl']['median_segment_steps_per_second'])
        expected = next(item for item in recorded['comparisons'] if
                        (item['task'], item['variant']) == (task, variant))
        for actual, reference in zip(pairs, expected['pairs']):
            assert math.isclose(actual['update_speedup'], reference['update_throughput_ratio'], rel_tol=1e-12)
        assert math.isclose(segment_ratio, expected['ratio_of_median_segment_throughput'], rel_tol=1e-12)
        assert min(ratios) > 1 and segment_ratio > 1 and expected['throughput_gate']
        assert replay['correctness_passed'] and all(pair['speedup'] > 1 for pair in replay['pairs'])
        workloads.append(dict(task=task, variant=variant, budget=budget, backends=backends,
            pairs=pairs, median_paired_update_speedup=median_ratio,
            update_speedup_range=[min(ratios), max(ratios)],
            median_paired_update_time_reduction_fraction=1 - 1 / median_ratio,
            ratio_of_median_segment_throughput=segment_ratio,
            replay_speedups=[pair['speedup'] for pair in replay['pairs']],
            median_replay_speedup=statistics.median(pair['speedup'] for pair in replay['pairs']),
            replay_correctness_passed=replay['correctness_passed'],
            replay_maximum_absolute_differences=replay['maximum_absolute_differences']))

    totals = {label: {field: sum(row['counts'][field] for row in group)
                     for field in ('updates', 'env_steps', 'episodes')}
              for label, group in [('throughput', [r for r in rows if not r['profiled']]),
                                   ('diagnostics', [r for r in rows if r['profiled']]), ('all', rows)]}
    result = dict(schema_version=1, analysis_date='2026-10-05', input=str(args.input),
        training_runs=len(rows), counts=totals, workloads=workloads,
        hostname=hardware['hostname'], affinity=hardware['affinity'],
        throughput_and_diagnostic_process_hours=sum(row['process_seconds'] for row in rows) / 3600,
        replay_process_hours=sum(row['compute_replay']['process_seconds'] for row in recorded['comparisons']) / 3600,
        projected_12_run_job_hours={backend: 3 * sum(w['backends'][backend]['projected_budget_hours'] for w in workloads)
                                    for backend in ('dgl', 'torch-v1')},
        limitations=['one machine/allocation; one training seed repeated for timing',
                     'no full trajectory digests; equal episode summaries are not trajectory proofs',
                     'three timing repetitions are not independent training seeds or a confidence interval',
                     'constant early-run throughput projections exclude queueing and later learning changes',
                     'this is the new paper-v1 baseline, not the old supplement-v1 preflight'],
        raw_ledger_rates_recomputed=True, recorded_summary_agrees=True)
    for filename, payload in [('summary.json', result), ('inputs.json', inputs)]:
        with (args.output / filename).open('x') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    with (args.output / 'throughput.csv').open('x', newline='') as stream:
        fields = ['task', 'variant', 'backend', 'pair', 'profiled', 'post_first_update_steps_per_second',
                  'all_update_steps_per_second', 'segment_steps_per_second', 'process_steps_per_second',
                  'startup_seconds', 'first_update_seconds', 'segment_seconds', 'process_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
