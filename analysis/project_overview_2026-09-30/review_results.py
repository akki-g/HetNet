"""Independent, read-only verification of the supplied latest training snapshot.

Uses only the Python standard library; does not import prior analysis code.
Run: .venv/bin/python analysis/project_overview_2026-09-30/review_results.py
"""
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix('.json')
BASE = ROOT / 'analysis/project_overview_2026-09-30/latest_snapshot'
summary = json.loads((BASE / 'summary.json').read_text())
provenance = json.loads((BASE / 'provenance.json').read_text())
csv_rows = list(csv.DictReader((BASE / 'epoch_metrics.csv').open()))
csv_by_key = {(r['task'], r['model'], int(r['seed']), int(r['epoch'])): r for r in csv_rows}
assert len(csv_rows) == len(csv_by_key)
checked_hashes = {}
for name, expected in provenance['inputs'].items():
    data = (ROOT / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert len(data) == expected['bytes'] and digest == expected['sha256'], name
    checked_hashes[name] = digest

HEADER = re.compile(r'^Epoch (\d+)\s+Reward \[([^]]+)\]')
metrics = ('success_rate', 'team_return', 'steps_taken')
raw_by_run = {}
raw_csv_fields = 0
summary_fields = 0


def equal(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def mean(rows, key, weighted):
    weights = [r['episodes'] if weighted else 1 for r in rows]
    return math.fsum(r[key] * w for r, w in zip(rows, weights)) / sum(weights)


for record in summary['runs']:
    data = (ROOT / record['source']).read_text()
    sr = record['model'].startswith('SoftRole')
    if sr:
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        total_steps = total_episodes = 0
        for row in rows:
            total_steps += row['steps']
            total_episodes += row['episodes']
            assert total_steps == row['total_steps']
            assert total_episodes == row['total_episodes']
            equal(row['steps_taken'] * row['episodes'], row['steps'])
            assert row['num_agents'] == 3 and row['event_exposed_episodes'] == 0
    else:
        namespace_line = next(x for x in data.splitlines() if x.startswith('Namespace('))
        call = ast.parse(namespace_line, mode='eval').body
        config = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        assert config['seed'] == record['seed'] and not config['use_binary']
        task = 'fc' if config['env_name'] == 'fire_commander' else ('pp' if config['nfriendly_A'] == 0 else 'pcp')
        assert task == record['task']
        rows, current = [], None
        for line in data.splitlines():
            m = HEADER.match(line)
            if m:
                assert current is None
                current = {'epoch': int(m[1]), 'team_return': math.fsum(float(x) for x in m[2].split())}
            elif current is not None and line.startswith('Success: '):
                current['success_rate'] = float(line.split(':')[1])
            elif current is not None and line.startswith('Steps-taken: '):
                current['steps_taken'] = float(line.split(':')[1])
                rows.append(current)
                current = None
        assert current is None
    assert len(rows) == record['epochs']
    assert [r['epoch'] for r in rows] == list(range(1, len(rows) + 1))
    key = (record['task'], record['model'], record['seed'])
    raw_by_run[key] = rows
    for raw in rows:
        exported = csv_by_key[key + (raw['epoch'],)]
        for metric in metrics:
            equal(raw[metric], float(exported[metric]))
            raw_csv_fields += 1
        if sr:
            for field in ('total_steps', 'total_episodes', 'steps', 'episodes', 'gate_entropy', 'alpha_null', 'value_loss_per_episode'):
                equal(raw[field], float(exported[field]))
                raw_csv_fields += 1
    for name, window in record.items():
        if not isinstance(window, dict) or 'first_epoch' not in window:
            continue
        chosen = rows[window['first_epoch'] - 1:window['last_epoch']]
        assert len(chosen) == 50
        weighted = window['weighting'] == 'episodes'
        for metric in (*metrics, 'value_loss_per_episode', 'gate_entropy', 'alpha_null'):
            if metric in window:
                equal(mean(chosen, metric, weighted), window[metric])
                summary_fields += 1
        if weighted:
            assert sum(r['episodes'] for r in chosen) == window['episodes']
            assert chosen[-1]['total_steps'] == window['steps_end']
            assert chosen[0]['total_steps'] - chosen[0]['steps'] == window['steps_start']
        if name == 'common_sample':
            budget = summary['softrole_sample_budgets'][record['task']]
            assert chosen[-1]['total_steps'] <= budget
            assert rows[chosen[-1]['epoch']]['total_steps'] > budget

assert sum(map(len, raw_by_run.values())) == len(csv_rows) == provenance['completed_epochs']
group_fields = 0
for group in summary['groups']:
    selected = [r for r in summary['runs'] if r['task'] == group['task'] and r['model'] == group['model']]
    assert len(selected) == 3
    for window, by_metric in group.items():
        if not isinstance(by_metric, dict):
            continue
        for metric, statistics in by_metric.items():
            values = [r[window][metric] for r in selected]
            for stat, actual in [('mean', math.fsum(values) / 3), ('min', min(values)), ('max', max(values))]:
                equal(actual, statistics[stat])
                group_fields += 1

first_latest = []
for task, model in sorted({(k[0], k[1]) for k in raw_by_run}):
    selected = [v for k, v in raw_by_run.items() if k[:2] == (task, model)]
    weighted = model.startswith('SoftRole')
    first_latest.append({'task': task, 'model': model,
        'first50': {m: math.fsum(mean(r[:50], m, weighted) for r in selected) / 3 for m in metrics},
        'latest50': {m: math.fsum(mean(r[-50:], m, weighted) for r in selected) / 3 for m in metrics},
        'weighting_within_seed': 'episodes' if weighted else 'equal epochs',
        'note': 'Latest endpoints differ by seed and method; progress illustration only.'})

regression = next(r for r in summary['runs'] if r['task'] == 'fc' and r['model'] == 'SoftRole shared' and r['seed'] == 2)
rows = raw_by_run[('fc', 'SoftRole shared', 2)]
blocks = []
for start in range(998, 1058, 10):
    chosen = rows[start - 1:start + 9]
    blocks.append({'first_epoch': start, 'last_epoch': start + 9,
                   'success_rate': mean(chosen, 'success_rate', True),
                   'value_loss_per_episode': mean(chosen, 'value_loss_per_episode', True)})

fc_base = ROOT / 'analysis/fc_audit_2026-09-30'
fc_simulator = json.loads((fc_base / 'simulator_probe.json').read_text())
for path, digest in fc_simulator['source_sha256'].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
passive = fc_simulator['passive_panel']
times = [r['zero_front_at_step'] for r in passive['episodes']]
assert len(times) == passive['count'] == 1000
assert sum(t is not None for t in times) == passive['origin_placeholder_by_300'] == 945
assert sum(t is not None and t <= 10 for t in times) == passive['origin_placeholder_by_10'] == 716
fc_panel = json.loads((fc_base / 'policy_panel/summary.json').read_text())
fc_totals = {key: sum(r[key] for r in fc_panel['policy_results']) for key in
             ('episodes', 'successes', 'zero_placeholder_episodes', 'source_captured_episodes', 'failed_after_source_capture')}

pilot_base = ROOT / 'runs/sensor_failure_validation_20260930/pilot'
pilot_summary = json.loads((pilot_base / 'summary.json').read_text())
pilot_results = []
for policy in pilot_summary['policies']:
    model = policy['model']
    assert hashlib.sha256(Path(policy['checkpoint']).read_bytes()).hexdigest() == policy['checkpoint_sha256']
    conditions = {}
    for condition in ('failure', 'sham'):
        report = json.loads((pilot_base / f'{model}_seed0_{condition}.json').read_text())
        rows = report['per_episode']
        assert len(rows) == 20
        expected = policy[condition]
        computed = {'episodes': len(rows), 'success_rate': sum(r['success'] for r in rows) / len(rows),
                    'team_return': math.fsum(r['team_return'] for r in rows) / len(rows),
                    'completion_steps_horizon_capped': sum(r['steps'] for r in rows) / len(rows),
                    'actual_event_exposed': sum(r['event_exposed'] for r in rows),
                    'scheduled_event_exposed': sum(r['scheduled_event_exposed'] for r in rows)}
        for field, value in computed.items():
            equal(value, expected[field])
        computed['diagnostic_true_counts'] = {field: sum(r[field] is True for r in rows) for field in
            ('pre_event_victim_reached', 'pre_event_victim_target_seen', 'pre_event_victim_target_visible')}
        assert computed['diagnostic_true_counts'] == expected['diagnostic_true_counts']
        conditions[condition] = computed
    pilot_results.append({'model': model, **conditions, 'checkpoint_progress': policy['checkpoint_progress']})

report = {'scope': 'New independent standard-library parse, count reconciliation, hash verification and summary recalculation. No policies or training executed.',
    'input_hashes_verified': len(checked_hashes), 'completed_epochs': len(csv_rows),
    'runs': len(summary['runs']), 'raw_csv_numeric_fields_checked': raw_csv_fields,
    'window_numeric_fields_checked': summary_fields, 'group_numeric_fields_checked': group_fields,
    'softrole_total_steps': sum(r['total_steps'] for r in summary['runs'] if 'total_steps' in r),
    'softrole_total_episodes': sum(r['total_episodes'] for r in summary['runs'] if 'total_episodes' in r),
    'all_softrole_event_exposure_counts_zero': True,
    'common_epoch_ends': summary['common_epoch_ends'], 'softrole_sample_budgets': summary['softrole_sample_budgets'],
    'groups': summary['groups'], 'first50_vs_latest50': first_latest,
    'fc_shared_seed2_blocks': blocks, 'fc_shared_seed2_previous_latest': {k: regression[k] for k in ('previous', 'latest')},
    'prior_fc_audit_rechecked': {'source_hashes': len(fc_simulator['source_sha256']),
         'passive_resets': len(times), 'placeholder_by300': 945, 'placeholder_by10': 716,
         'older_epoch200_policy_panel': fc_totals,
         'scope': 'Recomputed saved diagnostic records; did not rerun simulator or policies.'},
    'prior_pcp_pilot_rechecked': pilot_results,
    'verified_input_sha256': checked_hashes,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(BASE / 'comparison_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('groups', 'first50_vs_latest50', 'verified_input_sha256', 'fc_shared_seed2_previous_latest')}, indent=2))
