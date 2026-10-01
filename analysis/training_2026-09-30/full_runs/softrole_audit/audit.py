"""Read-only, streaming audit of the copied SoftRole run ledgers."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "stokes_runs/runs/softrole_primary"
OUTPUT = Path(__file__).parent


def entropy(values):
    return -sum(x * math.log(x) for x in values if x > 0)


def fresh(k):
    return dict(episodes=0, steps=0, success=0, team_return=0., gate_entropy=0.,
                alpha_null=0., gate_entropy_agent_steps=0., alpha_null_steps=0.,
                agent_steps=0, expert_load=[0.] * k,
                expert_load_agent_steps=[0.] * k, event_exposed=0,
                scheduled_event_exposed=0, intervention_exposed=0,
                early_unsuccess=0, at_horizon=0, payload_bits_generated=0,
                return_nocap=0., return_cap=0.,
                steps_hist=collections.Counter(), success_steps_hist=collections.Counter(),
                failure_steps_hist=collections.Counter(), return_hist=collections.Counter())


def update(a, row, horizon):
    t, n = row['steps'], row['num_agents']
    a['episodes'] += 1
    for key in ['steps', 'success', 'team_return', 'gate_entropy', 'alpha_null',
                'event_exposed', 'scheduled_event_exposed', 'intervention_exposed',
                'payload_bits_generated']:
        a[key] += row[key]
    for key in ['return_nocap', 'return_cap']:
        a[key] += row[key] or 0.
    a['agent_steps'] += n * t
    a['gate_entropy_agent_steps'] += row['gate_entropy'] * n * t
    a['alpha_null_steps'] += row['alpha_null'] * t
    a['early_unsuccess'] += (not row['success'] and t < horizon)
    a['at_horizon'] += t == horizon
    for k, val in enumerate(row['expert_load']):
        a['expert_load'][k] += val
        a['expert_load_agent_steps'][k] += val * n * t
    a['steps_hist'][t] += 1
    a['success_steps_hist' if row['success'] else 'failure_steps_hist'][t] += 1
    a['return_hist'][round(row['team_return'], 6)] += 1


def combine(rows, k):
    total = fresh(k)
    for a in rows:
        for key, val in a.items():
            if isinstance(val, list):
                total[key] = [x + y for x, y in zip(total[key], val)]
            elif isinstance(val, dict):
                total[key].update(val)
            else:
                total[key] += val
    return total


def quantile_hist(hist, quantiles=(.1, .25, .5, .75, .9, .95, .99)):
    # Empirical inverse CDF: min{x: F_n(x)>=q}; no interpolation.
    n = sum(hist.values())
    result = {}
    for q in quantiles:
        if not n:
            result[str(q)] = None
            continue
        threshold = math.ceil(q * n)
        count = 0
        for value, freq in sorted(hist.items()):
            count += freq
            if count >= threshold:
                result[str(q)] = value
                break
    return result


def describe(a):
    n = a['episodes']
    g = [v/n for v in a['expert_load']]
    gt = [v/a['agent_steps'] for v in a['expert_load_agent_steps']]
    h = a['gate_entropy']/n
    ht = a['gate_entropy_agent_steps']/a['agent_steps']
    return dict(episodes=n, steps=a['steps'], success_rate=a['success']/n,
                team_return=a['team_return']/n, mean_length=a['steps']/n,
                gate_entropy=h, pooled_gate=g, pooled_gate_entropy=entropy(g),
                gate_heterogeneity_gap=entropy(g)-h,
                token_gate_entropy=ht, token_pooled_gate=gt,
                token_gate_heterogeneity_gap=entropy(gt)-ht,
                alpha_null=a['alpha_null']/n,
                step_weighted_alpha_null=a['alpha_null_steps']/a['steps'],
                at_horizon_fraction=a['at_horizon']/n,
                early_unsuccess=a['early_unsuccess'],
                event_exposed=a['event_exposed'],
                scheduled_event_exposed=a['scheduled_event_exposed'],
                length_quantiles=quantile_hist(a['steps_hist']),
                success_length_quantiles=quantile_hist(a['success_steps_hist']),
                failure_length_quantiles=quantile_hist(a['failure_steps_hist']),
                return_quantiles=quantile_hist(a['return_hist']))


def grad_summary(rows, threshold):
    norms = sorted(r['gradient_norm_before_clip'] for r in rows)
    n = len(norms)
    return dict(updates=n, clipped_fraction=sum(v > threshold for v in norms)/n,
                min=min(norms), median=statistics.median(norms), max=max(norms),
                p10=norms[math.ceil(.1*n)-1], p90=norms[math.ceil(.9*n)-1],
                median_clip_multiplier=statistics.median(min(1., threshold/(v+1e-6)) for v in norms))


def main():
    summaries = []
    epoch_records = []
    for configfile in sorted(INPUT.glob('*/*/config.json')):
        directory = configfile.parent
        config = json.loads(configfile.read_text())
        runmeta = json.loads((directory/'run.json').read_text())
        name = str(directory.relative_to(INPUT))
        metrics = [json.loads(line) for line in (directory/'metrics.jsonl').open()]
        updates = [json.loads(line) for line in (directory/'updates.jsonl').open()]
        k = 1 if config['model'] == 'shared' else config['experts']
        epochs = {}
        update_counts = collections.defaultdict(lambda: dict(episodes=0, steps=0))
        previous_update = 0
        seen_current_update = set()
        anomalies = collections.Counter()
        versions, compositions, interventions = set(), set(), set()
        filehash = hashlib.sha256()
        sourcefile = directory/'episodes.jsonl'
        before = sourcefile.stat()
        for line_no, line in enumerate(sourcefile.open('rb'), 1):
            filehash.update(line)
            row = json.loads(line)
            assert all(math.isfinite(row[key]) for key in ['team_return','gate_entropy','alpha_null'])
            assert 0 <= row['alpha_null'] <= 1
            assert -1e-12 <= row['gate_entropy'] <= math.log(k) + 1e-12
            assert len(row['expert_load']) == k and abs(sum(row['expert_load'])-1) < 1e-12
            assert abs(sum(row['agent_returns'])-row['team_return']) < 1e-9
            assert 0 < row['steps'] <= config['max_steps']
            assert row['payload_bits_generated'] == row['steps'] * row['num_agents'] * 2 * config['msg_dim']
            u = row['update']
            if u < previous_update:
                anomalies['nonmonotone_update'] += 1
            if u != previous_update:
                seen_current_update.clear()
            scenario = row['scenario_id']
            if scenario in seen_current_update:
                anomalies['duplicate_scenario_within_update'] += 1
            seen_current_update.add(scenario)
            previous_update = u
            epoch = (u - 1) // config['updates_per_epoch'] + 1
            if epoch not in epochs:
                epochs[epoch] = fresh(k)
            update(epochs[epoch], row, config['max_steps'])
            update_counts[u]['episodes'] += 1
            update_counts[u]['steps'] += row['steps']
            versions.add(row['environment_version'])
            compositions.add(tuple(row['composition']))
            interventions.add(row['intervention'])
        after = sourcefile.stat()
        assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
        ledger = dict(episode_records=line_no, episode_first_update=min(update_counts),
                      episode_last_update=max(update_counts),
                      metrics_last_epoch=metrics[-1]['epoch'],
                      metrics_last_update=metrics[-1]['updates'],
                      updates_last_update=updates[-1]['update'],
                      anomalies=dict(anomalies), update_mismatches=[], epoch_mismatches=[])
        for u in updates:
            actual = update_counts.get(u['update'])
            if actual is None or any(actual[key] != u[key] for key in ['episodes','steps']):
                ledger['update_mismatches'].append(dict(update=u['update'], episodes_ledger=actual, update_ledger=u))
        for m in metrics:
            e = epochs.get(m['epoch'])
            diffs = {}
            if e is not None:
                for key in ['episodes','steps']:
                    if e[key] != m[key]:
                        diffs[key] = [e[key],m[key]]
                for key, numerator in [('success_rate','success'),('team_return','team_return'),
                                       ('gate_entropy','gate_entropy'),('alpha_null','alpha_null')]:
                    if abs(e[numerator]/e['episodes']-m[key]) > 1e-9:
                        diffs[key] = [e[numerator]/e['episodes'],m[key]]
            else:
                diffs['missing_episode_epoch'] = True
            if diffs:
                ledger['epoch_mismatches'].append(dict(epoch=m['epoch'],differences=diffs))
        matched_epochs = [m['epoch'] for m in metrics if m['epoch'] in epochs and epochs[m['epoch']]['episodes'] == m['episodes'] and epochs[m['epoch']]['steps'] == m['steps']]
        end = max(matched_epochs)
        window = list(range(end-49,end+1))
        latest = describe(combine([epochs[e] for e in window],k))
        first = describe(combine([epochs[e] for e in range(1,51)],k))
        latest_grads = [r for r in updates if (end-50)*config['updates_per_epoch'] < r['update'] <= end*config['updates_per_epoch']]
        summaries.append(dict(name=name, config=config, run=runmeta,
                              file_sha256=filehash.hexdigest(), file_bytes=before.st_size,
                              ledger=ledger, versions=sorted(versions),compositions=sorted(compositions),
                              interventions=sorted(interventions), matched_end_epoch=end,
                              first50=first, latest50=latest,
                              gradient_all=grad_summary(updates,config['max_grad_norm']),
                              gradient_first500=grad_summary(updates[:500],config['max_grad_norm']),
                              gradient_latest50epochs=grad_summary(latest_grads,config['max_grad_norm'])))
        for epoch,e in sorted(epochs.items()):
            epoch_records.append(dict(name=name,epoch=epoch,**e))
        print(json.dumps(dict(run=name,episodes=line_no,ledger=ledger,latest50=latest,
                              gradient=grad_summary(latest_grads,config['max_grad_norm']))),flush=True)
    (OUTPUT/'findings.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (OUTPUT/'episode_epoch_sufficient_stats.jsonl').open('w') as f:
        for row in epoch_records:
            f.write(json.dumps(row,sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
