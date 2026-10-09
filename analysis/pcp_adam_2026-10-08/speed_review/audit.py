#!/usr/bin/env python3
"""Read-only run timing/epoch audit; outputs only alongside this script.

Uses stored monotonic elapsed durations; never interprets filesystem mtimes as
job starts/finishes. Huge publication logs are streamed; only epoch/status JSON
is decoded. Episode batches are counted but not re-audited here.
"""
from pathlib import Path
import csv, hashlib, json, math, re
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
HASHES={}
def read(path):
    p=ROOT/path if not isinstance(path,Path) else path
    b=p.read_bytes(); HASHES[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest()
    return b

def load(path):return json.loads(read(path))
def rows(path):return [json.loads(l) for l in read(path).splitlines() if l]
def window(rs):
    if not rs:return None
    n=sum(r['episodes'] for r in rs); t=sum(r['wall_time_seconds'] for r in rs); s=sum(r['steps'] for r in rs)
    return dict(first_epoch=rs[0]['epoch'],last_epoch=rs[-1]['epoch'],epochs=len(rs),steps=s,episodes=n,updates=sum(r.get('updates_in_epoch',10) for r in rs),timed_seconds=t,hours=t/3600,steps_per_second=s/t,episodes_per_second=n/t,seconds_per_epoch=t/len(rs),steps_taken=s/n,success_rate=sum(r['success_rate']*r['episodes'] for r in rs)/n,mean_agent_return=sum((r.get('mean_agent_return',sum(r['reward_per_agent'])/len(r['reward_per_agent']) if 'reward_per_agent' in r else r['team_return']/3))*r['episodes'] for r in rs)/n)

def describe(rs):
    assert [r['epoch'] for r in rs]==list(range(1,len(rs)+1))
    assert sum(r['steps'] for r in rs)==rs[-1]['total_steps']
    assert sum(r['episodes'] for r in rs)==rs[-1]['total_episodes']
    for r in rs:
        assert math.isclose(r['steps']/r['episodes'],r['steps_taken'],abs_tol=1e-11)
    return dict(all=window(rs),first100=window(rs[:100]),last100=window(rs[-100:]),windows={f'{a}m-{b}m':window([r for r in rs if a*1e6<r['total_steps']<=b*1e6]) for a,b in [(0,1),(4,5),(9,10),(19,20),(26,27),(29,30),(35,36),(39,40)]})

def save_csv(name,rs):
    keys=sorted(set().union(*(r.keys() for r in rs)))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)

soft=[]; epoch_out=[]
for group,model in [('softrole_adam_pcp_911488','shared'),('softrole_primary','shared'),('softrole_primary','banked')]:
 for seed in range(3):
    path=ROOT/'stokes_runs'/group/f'pcp_{model}'/f'seed{seed}'
    cfg=load(path/'config.json')
    run_path=path/'run.json'
    if not run_path.exists():run_path=ROOT/'stokes_runs/runs'/group/f'pcp_{model}'/f'seed{seed}/run.json'
    run=load(run_path)
    if group=='softrole_primary':
        rs=rows(ROOT/f'logs_sr/softrole-898818_{6+(3 if model=="banked" else 0)+seed}.out')
        if (path/'metrics.jsonl').exists():
            saved=rows(path/'metrics.jsonl');assert saved==rs[:len(saved)]
    else:rs=rows(path/'metrics.jsonl')
    opt=cfg.get('optimizer','rmsprop'); key=f'{model}_{opt}_seed{seed}'
    entry=dict(key=key,path=str(path.relative_to(ROOT)),config=cfg,run=run,summary=describe(rs),finished=load(path/'finished.json') if (path/'finished.json').exists() else None)
    if group=='softrole_adam_pcp_911488':
        stdout=rows(ROOT/f'logs_sr/softrole-adam-pcp-911488_{seed}.out')
        assert len(rs)==len(stdout)
        entry['stdout_mismatches']=[dict(epoch=a['epoch'],fields={k:{'metric':a.get(k),'stdout':b.get(k)} for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}) for a,b in zip(rs,stdout) if a!=b]
        updates=rows(path/'updates.jsonl')
        assert len(updates)==rs[-1]['updates']==20000
        assert sum(r['steps'] for r in updates)==rs[-1]['total_steps']
        assert sum(r['episodes'] for r in updates)==rs[-1]['total_episodes']
        entry['hypothetical_old_episode_file_opens']=rs[-1]['total_episodes']
        entry['new_episode_file_opens']=len(updates)
        entry['episode_file_open_reduction_factor']=rs[-1]['total_episodes']/len(updates)
    soft.append(entry)
    epoch_out.extend(dict(run=key,**r) for r in rs)
save_csv('softrole_epochs.csv',epoch_out)
(OUT/'softrole_timing.json').write_text(json.dumps(soft,indent=2)+'\n')
print('SOFTROLE',[(r['key'],round(r['summary']['all']['hours'],3),round(r['summary']['all']['steps_per_second'],2)) for r in soft],flush=True)

# Inventory all 12 headers and status receipts efficiently, then fully stream six PCP inputs.
inventory=[];paper=[];paper_epochs=[]
for p in sorted((ROOT/'logs_1').glob('hetnet-paper-908316_*.out'),key=lambda p:int(p.stem.rsplit('_',1)[1])):
 with p.open('rb') as f:
    head=f.read(16000).decode(errors='replace');f.seek(max(0,p.stat().st_size-500000));tail=f.read().decode(errors='replace')
    namespace=next(l for l in head.splitlines() if l.startswith('Namespace('))
    task=re.search(r"experiment_name='([^']+)'",namespace).group(1);seed=int(re.search(r', seed=(\d+),',namespace).group(1))
    status=[json.loads(l) for l in tail.splitlines() if l.startswith('{"active_time_seconds":')][-1]
    entry=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,task=task,seed=seed,namespace=namespace,status=status)
    inventory.append(entry)
    if not task.startswith('pcp'):continue
    h=hashlib.sha256();rs=[];batch_count=0;last_batch_update=None;last_complete_offset=0;offset=0
    with p.open('rb') as f:
      for line in f:
        h.update(line);offset+=len(line)
        if line.startswith(b'{') and b'"record_type": "publication_epoch"' in line:
          rs.append(json.loads(line));last_complete_offset=offset
        elif line.startswith(b'{"episodes": ['):
          batch_count+=1
          match=re.search(rb'"update": (\d+)\}\s*$',line)
          assert match
          last_batch_update=int(match.group(1))
    HASHES[entry['path']]=h.hexdigest()
    assert batch_count==last_batch_update==status['counts']['updates']
    assert rs[-1]['total_steps']<=status['counts']['env_steps']
    entry.update(sha256=h.hexdigest(),summary=describe(rs),episode_batch_count=batch_count,last_batch_update=last_batch_update,complete_epoch_updates=rs[-1]['updates'],last_complete_epoch_byte_end=last_complete_offset,partial_suffix=dict(updates=status['counts']['updates']-rs[-1]['updates'],steps=status['counts']['env_steps']-rs[-1]['total_steps'],episodes=status['counts']['episodes']-rs[-1]['total_episodes']))
    paper.append(entry);paper_epochs.extend(dict(run=f'{task}_seed{seed}',**r) for r in rs)
    print('PAPER',task,seed,len(rs),rs[-1]['total_steps'],round(entry['summary']['last100']['steps_taken'],6),flush=True)
save_csv('paper_pcp_epochs.csv',paper_epochs)
(OUT/'paper_pcp_timing.json').write_text(json.dumps(paper,indent=2)+'\n')
(OUT/'paper_wave_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')

# Hash source paths read in the implementation comparison, retaining both snapshots.
for group in ('softrole_primary','softrole_adam_pcp_911488'):
 for rel in ('softrole/train.py','softrole/model.py','softrole/learning.py','softrole/rollout.py','softrole/env.py','softrole/config.py','scripts/softrole.sh','slurm/softrole.sbatch'):
    p=ROOT/'stokes_runs'/group/'pcp_shared/seed0/source'/rel
    if p.exists():read(p)
for rel in ('scripts/softrole_adam_pcp.sh','slurm/softrole_adam_pcp.sbatch','softrole/optimizers.py'):
 read(ROOT/'stokes_runs/softrole_adam_pcp_911488/pcp_shared/seed0/source'/rel)
(OUT/'input_hashes.json').write_text(json.dumps(HASHES,indent=2,sort_keys=True)+'\n')
print('DONE',flush=True)
