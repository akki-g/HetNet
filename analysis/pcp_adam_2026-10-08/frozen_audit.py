"""Read-only refresh of prior PCP frozen results and geometric reference.

No policy execution. Recompute from raw saved reports, retain failures, and give
each independent training seed equal weight. Run from the repository root.
"""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = ROOT / 'stokes_runs/frozen_pcp_30m_20261002_slurm'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    inputs = {}
    def read(p):
        inputs[str(p.relative_to(ROOT))] = sha(p)
        return json.loads(p.read_text())
    manifest = read(INPUT / 'manifest.json')
    rows, reports, pairs, checkpoints = [], {}, [], []
    for condition in ('nominal', 'failure', 'sham'):
        panel = read(INPUT / ('scenarios_nominal.json' if condition == 'nominal' else 'scenarios_failure.json'))
        for model, seed in itertools.product(('shared', 'banked'), range(3)):
            path = INPUT / 'results' / f'{model}_seed{seed}_{condition}.json'
            r = read(path)
            reports[model, seed, condition] = r
            es = r['per_episode']
            assert r['scenarios'] == panel
            assert r['episodes'] == len(es) == len(panel)
            assert len({e['scenario_id'] for e in es}) == len(es)
            for e, s in zip(es, panel):
                assert all(e[k] == v for k, v in s.items())
                assert math.isclose(sum(e['agent_returns']), e['team_return'], abs_tol=1e-10)
                assert math.isclose(e['mean_agent_return'], e['team_return'] / e['num_agents'], abs_tol=1e-10)
                assert e['success'] or e['steps'] == 80
            if condition == 'nominal':
                for comp in sorted({tuple(e['composition']) for e in es}):
                    group = [e for e in es if tuple(e['composition']) == comp]
                    rows.append(dict(model=model, seed=seed, num_p=comp[0], num_a=comp[1], episodes=len(group),
                                     success_rate=mean(e['success'] for e in group),
                                     steps_taken=mean(e['steps'] for e in group),
                                     team_return=mean(e['team_return'] for e in group),
                                     mean_agent_return=mean(e['mean_agent_return'] for e in group)))
    for model, seed in itertools.product(('shared', 'banked'), range(3)):
        failure = reports[model, seed, 'failure']['per_episode']
        sham = reports[model, seed, 'sham']['per_episode']
        assert [e['scenario_id'] for e in failure] == [e['scenario_id'] for e in sham]
        pairs.append(dict(model=model, seed=seed, pairs=len(failure), exposed=sum(e['event_exposed'] for e in failure),
                          success_delta=mean(int(a['success'])-int(b['success']) for a,b in zip(failure,sham)),
                          steps_delta=mean(a['steps']-b['steps'] for a,b in zip(failure,sham)),
                          team_return_delta=mean(a['team_return']-b['team_return'] for a,b in zip(failure,sham))))
    for policy in manifest['policies']:
        relative = Path(policy['checkpoint'].split('/runs/', 1)[1])
        local = ROOT / 'stokes_runs' / relative
        exists = local.exists()
        actual = sha(local) if exists else None
        if exists:
            assert actual == policy['checkpoint_sha256']
            inputs[str(local.relative_to(ROOT))] = actual
        checkpoints.append(dict(model=policy['model'], seed=policy['training_seed'],
                                local_path=str(local.relative_to(ROOT)), local_exists=exists,
                                sha256=actual, recorded_sha256=policy['checkpoint_sha256'],
                                progress=policy['checkpoint_progress']))
    groups = []
    for model, p, a in sorted({(r['model'],r['num_p'],r['num_a']) for r in rows}):
        group = [r for r in rows if (r['model'],r['num_p'],r['num_a']) == (model,p,a)]
        assert len(group) == 3
        groups.append(dict(model=model, num_p=p, num_a=a, seeds=3,
                           metrics={key: dict(mean=mean(r[key] for r in group), minimum=min(r[key] for r in group),
                                              maximum=max(r[key] for r in group))
                                    for key in ('success_rate','steps_taken','team_return','mean_agent_return')}))
    # Uniform, distinct agent/target starts; an optimistic fully informed geometric
    # lower bound, not a decentralized implementable policy or per-panel bound.
    cells = list(itertools.product(range(5), repeat=2))
    count = total_time = total_penalty = 0
    for tx, ty in cells:
        distances = [abs(x-tx)+abs(y-ty) for x,y in cells if (x,y)!=(tx,ty)]
        for p0,p1,a in itertools.permutations(distances, 3):
            total_time += max(p0,p1,a+1)
            total_penalty += p0+p1+a+1-3
            count += 1
    assert count == 303600
    oracle = dict(placements=count, optimistic_mean_steps=total_time/count,
                  optimistic_mean_team_return=-.05*total_penalty/count,
                  scope='Population geometric reference; not a policy evaluation or the exact common-panel lower bound.')
    summary = dict(raw_reports=18, total_evaluated_episodes=sum(r['episodes'] for r in reports.values()),
                   nominal_seed_rows=rows, nominal_groups=groups, failure_sham_pairs=pairs,
                   checkpoint_byte_checks=checkpoints, geometric_reference=oracle,
                   caveats=['These are prior RMSprop evaluations, not evaluations of the new Adam checkpoints.',
                            'No new frozen-policy evaluation was run.', 'Seed ranges are descriptive, not confidence intervals.'])
    (HERE/'frozen_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (HERE/'frozen_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    with (HERE/'frozen_compositions.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    assert all(sha(ROOT/p) == h for p,h in inputs.items())
    print(json.dumps(dict(reports=18, episodes=summary['total_evaluated_episodes'],
                         checkpoint_files_verified=sum(c['local_exists'] for c in checkpoints),
                         exposed_pairs=sum(p['exposed'] for p in pairs), oracle=oracle),indent=2))


if __name__ == '__main__':
    main()
