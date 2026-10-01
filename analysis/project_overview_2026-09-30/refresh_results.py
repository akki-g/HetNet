"""Refresh the report snapshot from current logs, preserving all older analyses.

Parser fields and estimators follow training_2026-09-30/analyze.py and its resync
analysis. Uses the standard library; never trains or executes a policy.
Run: .venv/bin/python analysis/project_overview_2026-09-30/refresh_results.py
"""
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'latest_snapshot'
OLD = ROOT / 'analysis/training_2026-09-30_resync'
TASKS = ('pp', 'pcp', 'fc')
MODELS = ('HetNet Real', 'SoftRole shared', 'SoftRole banked')
METRICS = ('success_rate', 'team_return', 'steps_taken')
HEADER = re.compile(r'^Epoch (\d+)\s+Reward \[([^]]+)\]\s+Time ([\d.]+)s, Episodes (\d+), Total Steps (\d+),')
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return data.decode()


def mean(values):
    return math.fsum(values) / len(values)


def parse_reproduction(text):
    call = ast.parse(next(line for line in text.splitlines() if line.startswith('Namespace(')), mode='eval').body
    config = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    assert not config['use_binary']
    p, a = config['nfriendly_P'], config['nfriendly_A']
    task = 'fc' if config['env_name'] == 'fire_commander' else ('pp' if a == 0 else 'pcp')
    rows, row = [], None
    for number, line in enumerate(text.splitlines(), 1):
        match = HEADER.match(line)
        if match:
            assert row is None, 'Incomplete previous epoch'
            epoch, reward, seconds, episodes, steps = match.groups()
            rewards = [float(x) for x in reward.split()]
            assert len(rewards) == p + a
            row = dict(epoch=int(epoch), team_return=sum(rewards), reward_per_agent=rewards,
                       wall_time_seconds=float(seconds), reported_overcounted_episodes=int(episodes),
                       reported_overcounted_steps=int(steps), return_nocap=mean(rewards[:p]),
                       return_cap=mean(rewards[p:]) if a else None, source_line=number)
        elif row is not None and line.startswith('Success: '):
            row['success_rate'] = float(line.split(': ', 1)[1])
        elif row is not None and line.startswith('Steps-taken: '):
            row['steps_taken'] = float(line.split(': ', 1)[1])
            assert 'success_rate' in row
            rows.append(row)
            row = None
    assert row is None, 'An epoch header exists without its completed metrics'
    return task, MODELS[0], config['seed'], config, rows


def average(rows, key, weighted):
    weights = [r['episodes'] if weighted else 1 for r in rows]
    return math.fsum(r[key] * w for r, w in zip(rows, weights)) / sum(weights)


def window(rows, weighted):
    assert len(rows) == 50
    result = {'first_epoch': rows[0]['epoch'], 'last_epoch': rows[-1]['epoch'],
              'weighting': 'episodes' if weighted else 'equal epochs',
              **{key: average(rows, key, weighted) for key in METRICS}}
    if weighted:
        result.update(episodes=sum(r['episodes'] for r in rows),
                      steps_start=rows[0]['total_steps']-rows[0]['steps'], steps_end=rows[-1]['total_steps'])
        result.update({key: average(rows, key, True) for key in ('value_loss_per_episode', 'gate_entropy', 'alpha_null')})
    return result


old_provenance = json.loads(read(OLD / 'provenance.json'))
old_summary = json.loads(read(OLD / 'summary.json'))
full_summary = json.loads(read(ROOT / 'analysis/training_2026-09-30/full_runs/summary.json'))
archive_hashes = json.loads(read(ROOT / 'analysis/training_2026-09-30/full_runs/structured_input_hashes.json'))
read(ROOT / 'analysis/training_2026-09-30/analyze.py')
read(OLD / 'analyze.py')
runs, extensions, stderr = [], [], []
for path in sorted([*ROOT.glob('logs_1/*.out'), *ROOT.glob('logs_sr/*.out')]):
    source = str(path.relative_to(ROOT))
    text = read(path)
    old_identity = old_provenance['inputs'][source]
    assert hashlib.sha256(text.encode()[:old_identity['bytes']]).hexdigest() == old_identity['sha256'], source
    sr = path.parent.name == 'logs_sr'
    if sr:
        index = int(path.stem.rsplit('_', 1)[1])
        task, model, seed = TASKS[index // 6], MODELS[1 + (index // 3) % 2], index % 3
        rows = [dict(json.loads(line), source_line=number) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
        raw_config = None
    else:
        task, model, seed, raw_config, rows = parse_reproduction(text)
    directory = ROOT / 'stokes_runs/runs' / ('softrole_primary' if sr else 'reproduction-fast') / (task + '_' + (model.split()[-1] if sr else 'real')) / f'seed{seed}'
    archive_path = directory / 'metrics.jsonl'
    archived = [json.loads(line) for line in read(archive_path).splitlines()]
    config_path = directory / ('config.json' if sr else 'resolved_args.json')
    config = json.loads(read(config_path))
    for file in (archive_path, config_path):
        relative = str(file.relative_to(ROOT))
        assert inputs[relative] == archive_hashes[relative], relative
    assert len(rows) >= len(archived)
    for row, archived_row in zip(rows, archived):
        if sr:
            assert {k: v for k, v in row.items() if k != 'source_line'} == archived_row
        else:
            for key in ('success_rate', 'steps_taken'):
                assert abs(row[key] - archived_row[key]) <= .005000001
            assert all(abs(x-y) <= .005000001 for x, y in zip(row['reward_per_agent'], archived_row['reward_per_agent']))
    assert [r['epoch'] for r in rows] == list(range(1, len(rows)+1))
    for row in rows:
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
        assert 0 <= row['success_rate'] <= 1 and row['steps_taken'] > 0
    if sr:
        assert (config['task'], config['model'], config['seed']) == (task, model.split()[-1], seed)
        assert config['failure_prob'] == 0 and not config['compositions']
        cumulative_steps = cumulative_episodes = 0
        for row in rows:
            cumulative_steps += row['steps']; cumulative_episodes += row['episodes']
            assert row['total_steps'] == cumulative_steps and row['total_episodes'] == cumulative_episodes
            assert abs(row['steps_taken'] * row['episodes'] - row['steps']) < 1e-7
            assert not row['partial_epoch'] and row['updates'] == 10 * row['epoch']
            assert row['num_agents'] == 3 and row['event_exposed_episodes'] == 0
            assert 0 <= row['alpha_null'] <= 1 and 0 <= row['gate_entropy'] <= math.log(4)+1e-12
            p, a = (3,0) if task == 'pp' else (2,1)
            assert abs(p*row['return_nocap']+a*(row['return_cap'] or 0)-row['team_return']) < 1e-8
    else:
        assert all(raw_config[k] == config[k] for k in raw_config if k in config)
    old_run = next(r for r in old_summary['runs'] if r['source'] == source)
    run = dict(source=source, task=task, model=model, seed=seed, rows=rows, config=config,
               archived_epochs=len(archived), archived_steps=archived[-1]['total_steps'],
               identity_basis=('All archived epoch fields match stdout exactly; appended tail lacks a fresh source/checkpoint archive' if sr else 'Observed stdout Namespace'))
    runs.append(run)
    extensions.append(dict(source=source, task=task, model=model, seed=seed, archived_epochs=len(archived),
        previous_stdout_epochs=old_run['epochs'], new_epochs=len(rows), additional_epochs=len(rows)-old_run['epochs'],
        previous_stdout_byte_prefix_unchanged=True, archive_metric_prefix_matches=True,
        incomplete_metric_header_blocks=0, incomplete_json_lines=0,
        trailing_in_progress_batch_lines=(0 if sr else sum(line.startswith('[Epoch] batch ') for line in text.splitlines()[rows[-1]['source_line']+2:]))))
for path in sorted([*ROOT.glob('logs_1/*.err'), *ROOT.glob('logs_sr/*.err')]):
    text = read(path); name = str(path.relative_to(ROOT))
    stderr.append(dict(path=name, unchanged_from_previous=inputs[name] == old_provenance['inputs'][name], has_traceback='Traceback (most recent call last)' in text))
assert len(runs) == len({(r['task'],r['model'],r['seed']) for r in runs}) == 27
runs.sort(key=lambda r: (TASKS.index(r['task']), MODELS.index(r['model']), r['seed']))
common_epochs = {task: min(len(r['rows']) for r in runs if r['task'] == task) for task in TASKS}
budgets = {task: min(r['rows'][-1]['total_steps'] for r in runs if r['task'] == task and r['model'] != MODELS[0]) // 500000 * 500000 for task in TASKS}
summary = dict(common_epoch_ends=common_epochs, softrole_sample_budgets=budgets, runs=[], groups=[])
for run in runs:
    rows = run['rows']; sr = run['model'] != MODELS[0]; end = common_epochs[run['task']]
    record = {k: run[k] for k in ('source','task','model','seed','archived_epochs','archived_steps','identity_basis')}
    record.update(epochs=len(rows), configured_epochs=run['config'].get('epochs', run['config'].get('num_epochs')),
        common_epoch=window(rows[end-50:end],False), latest=window(rows[-50:],sr), previous=window(rows[-100:-50],sr),
        archive_latest=window(rows[run['archived_epochs']-50:run['archived_epochs']],sr))
    if sr:
        record.update(total_steps=rows[-1]['total_steps'], total_episodes=rows[-1]['total_episodes'],
            event_exposed_episodes=sum(r['event_exposed_episodes'] for r in rows),
            common_sample=window([r for r in rows if r['total_steps'] <= budgets[run['task']]][-50:],True),
            prior_common_sample=window([r for r in rows if r['total_steps'] <= full_summary['budgets'][run['task']]][-50:],True))
        old_run = next(r for r in old_summary['runs'] if r['source'] == run['source'])
        for key in METRICS:
            assert abs(record['prior_common_sample'][key]-old_run['prior_common_sample'][key]) < 1e-10
    summary['runs'].append(record)
for task in TASKS:
    for model in MODELS:
        selected = [r for r in summary['runs'] if r['task'] == task and r['model'] == model]
        windows = ('common_epoch','latest','archive_latest') + (() if model == MODELS[0] else ('common_sample','prior_common_sample'))
        summary['groups'].append(dict(task=task, model=model, **{name: {key: dict(mean=mean([r[name][key] for r in selected]),
            min=min(r[name][key] for r in selected),max=max(r[name][key] for r in selected)) for key in METRICS} for name in windows}))
for path, entry in inputs.items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == entry['sha256'], 'Input changed during refresh: '+path
provenance = dict(inputs=inputs, extensions=extensions, stderr=stderr,
    completed_epochs=sum(len(r['rows']) for r in runs), archived_completed_epochs=11375,
    previous_stdout_completed_epochs=old_provenance['completed_epochs'],
    analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Latest resynced stdout with verified prior stdout and archived prefixes; no new checkpoint/episode-ledger/source audit; no policy execution')
OUT.mkdir(exist_ok=True)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
flat = [dict(task=run['task'],model=run['model'],seed=run['seed'],**row) for run in runs for row in run['rows']]
with (OUT/'epoch_metrics.csv').open('w',newline='') as stream:
    fields = list(dict.fromkeys(key for row in flat for key in row))
    writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
    for row in flat:
        writer.writerow({k: json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
print(json.dumps({'completed_epochs':provenance['completed_epochs'],'additional_epochs':sum(r['additional_epochs'] for r in extensions),
    'common_epoch_ends':common_epochs,'sample_budgets':budgets,'softrole_steps':sum(r.get('total_steps',0) for r in summary['runs']),
    'softrole_episodes':sum(r.get('total_episodes',0) for r in summary['runs']),
    'at_configured_epoch_target':sum(r['epochs']==r['configured_epochs'] for r in summary['runs'])},indent=2))
