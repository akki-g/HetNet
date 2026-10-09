#!/usr/bin/env python3
"""Independently reconcile selected paper stdout epoch windows against episodes."""
from pathlib import Path
import collections,csv,hashlib,json,math,re
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
runs=json.loads((OUT/'paper_pcp_timing.json').read_text());allrows=list(csv.DictReader((OUT/'paper_pcp_epochs.csv').open()))
results=[]
for run in runs:
 key=f"{run['task']}_seed{run['seed']}"
 rows=[{k:(float(v) if k not in ('run','record_type','reward_per_agent') else v) for k,v in r.items()} for r in allrows if r['run']==key]
 subsets={'26m-27m':[r for r in rows if 26e6<r['total_steps']<=27e6],'last100':rows[-100:]}
 if key=='pcp_real_seed0':
  worst=max((r for r in rows if r['total_steps']>20e6),key=lambda r:r['steps_taken'])
  subsets['worst_epoch_after20m']=[worst]
  (OUT/'real_seed0_regression_epochs.csv').write_text('epoch,total_steps,episodes,steps_taken,success_rate,mean_agent_return\n'+'\n'.join(','.join(str(r[k]) for k in ('epoch','total_steps','episodes','steps_taken','success_rate','mean_agent_return')) for r in rows if 23e6<r['total_steps']<31e6)+'\n')
 update_sets={label:{u for r in rs for u in range(int(r['updates']-r['updates_in_epoch'])+1,int(r['updates'])+1)} for label,rs in subsets.items()}
 update_labels=collections.defaultdict(list)
 for label,us in update_sets.items():
  for u in us:update_labels[u].append(label)
 agg={label:dict(episodes=0,steps=0,successes=0,mean_agent_return_sum=0.0,updates=0,first_episode=None,last_episode=None) for label in subsets}
 with (ROOT/run['path']).open('rb') as f:
  for line in f:
   if not line.startswith(b'{"episodes": ['):continue
   match=re.search(rb'"update": (\d+)\}\s*$',line);u=int(match.group(1))
   if u not in update_labels:continue
   batch=json.loads(line);episodes=batch['episodes'];assert batch['update']==u
   assert {e['collector'] for e in episodes}=={0,1,2,3}
   for e in episodes:assert e['update']==u and e['terminated']==e['success']
   for label in update_labels[u]:
    a=agg[label];a['updates']+=1;a['episodes']+=len(episodes);a['steps']+=sum(e['steps'] for e in episodes);a['successes']+=sum(e['success'] for e in episodes);a['mean_agent_return_sum']+=sum(e['mean_agent_return'] for e in episodes)
    if a['first_episode'] is None:a['first_episode']=episodes[0]['episode']
    a['last_episode']=episodes[-1]['episode']
 for label,a in agg.items():
  rs=subsets[label];n=sum(r['episodes'] for r in rs)
  assert a['episodes']==n and a['steps']==sum(r['steps'] for r in rs)
  assert a['updates']==len(update_sets[label])
  assert math.isclose(a['successes'],sum(r['success_rate']*r['episodes'] for r in rs),abs_tol=1e-7)
  assert math.isclose(a['mean_agent_return_sum'],sum(r['mean_agent_return']*r['episodes'] for r in rs),abs_tol=1e-7)
  a.update(first_epoch=int(rs[0]['epoch']),last_epoch=int(rs[-1]['epoch']),start_steps=int(rs[0]['total_steps']-rs[0]['steps']),end_steps=int(rs[-1]['total_steps']),steps_taken=a['steps']/n,success_rate=a['successes']/n,mean_agent_return=a['mean_agent_return_sum']/n,passed=True)
 results.append(dict(run=key,windows=agg));print(key,'passed',flush=True)
(OUT/'episode_window_validation.json').write_text(json.dumps(results,indent=2)+'\n')
