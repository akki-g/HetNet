"""Join independently audited epoch series into fair, descriptive PCP comparisons.

Complete epochs whose end lies in (lower,upper] enter each sample window.
Episode-weight within a seed, then weight the three training seeds equally.
There is no interpolation/extrapolation and no best-epoch/checkpoint selection.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
LABELS = {'shared_adam':'CapCom shared / Adam', 'shared_rmsprop':'CapCom shared / RMSprop',
          'banked_rmsprop':'CapCom banked / RMSprop', 'pcp_real':'HetNet paper-v1 Real',
          'pcp_binary':'HetNet paper-v1 Binary'}


def csv_read(p):
    result=[]
    for row in csv.DictReader(p.open()):
        r={}
        for k,v in row.items():
            if v=='': continue
            try: r[k]=float(v)
            except ValueError:r[k]=v
        result.append(r)
    return result


def aggregate(rs):
    if not rs: return None
    n=sum(r['episodes'] for r in rs)
    return dict(epochs=len(rs),first_epoch=int(rs[0]['epoch']),last_epoch=int(rs[-1]['epoch']),
                start_steps=int(rs[0]['total_steps']-rs[0]['steps']),end_steps=int(rs[-1]['total_steps']),
                episodes=int(n),steps=int(sum(r['steps'] for r in rs)),
                steps_taken=sum(r['steps'] for r in rs)/n,
                success_rate=sum(r['episodes']*r['success_rate'] for r in rs)/n,
                mean_agent_return=sum(r['episodes']*r['mean_agent_return'] for r in rs)/n,
                epoch_hours=sum(r['wall_time_seconds'] for r in rs)/3600)


def save_csv(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)


def main():
    series=defaultdict(list)
    for file in ('softrole_epochs.csv','paper_pcp_epochs.csv'):
        for r in csv_read(HERE/'speed_review'/file):
            run=r['run']
            if 'mean_agent_return' not in r:
                if 'team_return' in r:r['mean_agent_return']=r['team_return']/3
                else:r['mean_agent_return']=mean(json.loads(r['reward_per_agent'].replace("'",'"')))
            series[run].append(r)
    seed_rows=[]
    for run,rs in series.items():
        group,seed=run.rsplit('_seed',1)
        for lower,upper in ((0,1),(1,2),(2,4),(4,8),(8,10),(10,20),(20,30),(26,27),(29,30),(30,33),(39,40)):
            # A window is not complete merely because it contains some epochs.
            if rs[-1]['total_steps'] < upper*1e6:continue
            a=aggregate([r for r in rs if lower*1e6<r['total_steps']<=upper*1e6])
            if a:seed_rows.append(dict(group=group,seed=int(seed),window=f'{lower}-{upper}M',**a))
        seed_rows.append(dict(group=group,seed=int(seed),window='last100',**aggregate(rs[-100:])))
    groups=[]
    for group,win in sorted({(r['group'],r['window']) for r in seed_rows}):
        xs=[r for r in seed_rows if (r['group'],r['window'])==(group,win)]
        d=dict(group=group,label=LABELS[group],window=win,seeds=len(xs))
        for key in ('steps_taken','success_rate','mean_agent_return'):
            d[key]=mean(r[key] for r in xs)
            d[key+'_min']=min(r[key] for r in xs);d[key+'_max']=max(r[key] for r in xs)
        groups.append(d)
    # Independent agreement with the separate trial audit.
    remap={'shared_adam':'adam_shared','shared_rmsprop':'rmsprop_shared','banked_rmsprop':'rmsprop_banked'}
    reference=csv_read(HERE/'trial_review/group_windows.csv')
    checks=0
    for d in groups:
        if d['group'] not in remap:continue
        matched=[r for r in reference if r['group']==remap[d['group']] and r['window']==d['window']]
        assert len(matched)==1
        for key in ('steps_taken','success_rate','mean_agent_return'):
            assert math.isclose(d[key],matched[0][key],abs_tol=1e-10),(d,matched)
            checks+=1
    bins=[]
    for run,rs in series.items():
        group,seed=run.rsplit('_seed',1)
        for hi in range(1,int(rs[-1]['total_steps']//1e6)+1):
            a=aggregate([r for r in rs if (hi-1)*1e6<r['total_steps']<=hi*1e6])
            if a:bins.append(dict(group=group,seed=int(seed),budget_m=hi,**a))
    save_csv('comparison_windows.csv',seed_rows);save_csv('comparison_groups.csv',groups);save_csv('curve_bins.csv',bins)
    result=dict(method='Episode-weighted complete-epoch windows, then equal independent seed weights; all failures retained.',
                limitations=['Descriptive observed seed ranges, not confidence intervals.',
                            'Latest tails have different sample budgets; matched26-27M is the common six-HetNet-seed window.',
                            'RMSprop shared39-40M has only seeds1 and2; seed0 timed out.',
                            'Training curves are not frozen-policy evaluation.'],groups=groups,
                cross_checked_trial_values=checks)
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in groups:
        if r['window'] in ('26-27M','29-30M','last100'):
            print(r['label'],r['window'],r['seeds'],round(r['steps_taken'],5),round(r['success_rate'],6),round(r['mean_agent_return'],6))


if __name__ == '__main__':main()
