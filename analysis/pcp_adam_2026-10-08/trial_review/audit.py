#!/usr/bin/env python3
"""Read-only audit of the three PCP Adam archives and training-window comparators.

Run from any directory: .venv/bin/python analysis/pcp_adam_2026-10-08/trial_review/audit.py
Only this script's directory is written. --episodes adds the complete ~14GB episode audit.
Window membership: lower < epoch-end total_steps <= upper. Metrics weight complete
included epochs by their episode counts; actual retained sample boundaries are exported.
"""
import argparse, csv, hashlib, json, math, statistics, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
NEW=ROOT/'stokes_runs/softrole_adam_pcp_911488/pcp_shared'
OLD=ROOT/'stokes_runs/softrole_primary'
WINDOWS=[(0,1),(1,2),(2,4),(4,8),(8,10),(10,20),(20,30),(26,27),(29,30),(30,33),(39,40)]
FIELDS=['steps_taken','success_rate','team_return','mean_agent_return','return_cap','return_nocap','gate_entropy','alpha_null','policy_loss_per_episode','value_loss_per_episode']
HASHES={}

def rel(p):return str(p.relative_to(ROOT))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 HASHES[rel(p)]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
 return h.hexdigest()
def read(p):
 digest(p);return json.loads(p.read_text())
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(y) for y in x.values())
 if isinstance(x,list):return all(finite(y) for y in x)
 return True
def rows(p):
 digest(p);rs=[json.loads(line) for line in p.read_text().splitlines() if line.startswith('{')]
 assert all(finite(r) for r in rs),p
 return rs
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvsave(name,rs):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rs))));w.writeheader();w.writerows(rs)
def window(rs,label,group,seed,bounds):
 n=sum(r['episodes'] for r in rs)
 if not n:return {'group':group,'seed':seed,'window':label,'episodes':0}
 x={'group':group,'seed':seed,'window':label,'requested_lo':bounds[0],'requested_hi':bounds[1],
    'epochs':len(rs),'first_epoch':rs[0]['epoch'],'last_epoch':rs[-1]['epoch'],
    'actual_start_steps':rs[0]['total_steps']-rs[0]['steps'],'actual_end_steps':rs[-1]['total_steps'],
    'episodes':n,'steps':sum(r['steps'] for r in rs),'recorded_epoch_seconds':sum(r['wall_time_seconds'] for r in rs)}
 for k in FIELDS:x[k]=sum(r.get(k,r['team_return']/3 if k=='mean_agent_return' else 0)*r['episodes'] for r in rs)/n
 x['successes']=round(x['success_rate']*n);x['failures']=n-x['successes']
 x['steps_per_second']=x['steps']/x['recorded_epoch_seconds']
 return x

def checkpoint(p,config,manifest,epochs):
 import torch
 digest(p); side=read(Path(str(p)+'.signature.json'));c=torch.load(p,map_location='cpu',weights_only=False)
 for k in ['config','epoch','updates','total_steps','total_episodes','source_sha256','environment_version','completed_epochs','updates_in_partial_epoch']:
  assert k in c,(p,k)
 assert json.loads(json.dumps(c['config']))==config and c['source_sha256']==manifest['sha256']
 expected_model={'base':config['dim']**2,'n_squares':(2*config['vision']+1)**2,'mode':config['model'],**{k:config[k] for k in ['experts','pre_dim','hidden_dim','heads','head_dim','msg_dim','feedback']}}
 assert c['model_config']==expected_model and c['environment_version']==config['env_version'] and c['format_version']==1
 r=epochs[c['epoch']-1]
 for k in ['updates','total_steps','total_episodes']:assert c[k]==r[k],(p,k)
 assert c['completed_epochs']==c['epoch'] and c['updates_in_partial_epoch']==0
 m=c['model_state'];os=c['optimizer_state'];entries=[]
 for name,t in sorted(m.items()):
  assert torch.isfinite(t).all() and t.dtype==torch.float64
  entries.append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()})
 signature={'schema_version':1,'parameters':entries,'buffers':[]}
 signature['sha256']=hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 assert signature==c['signature']==side
 expected={'lr':1e-4,'betas':(.9,.999),'eps':1e-8,'weight_decay':0,'amsgrad':False,'maximize':False,'foreach':False,'capturable':False,'differentiable':False,'fused':False}
 assert len(os['param_groups'])==1
 g=os['param_groups'][0]
 assert {k:v for k,v in g.items() if k!='params'}==expected
 assert list(g['params'])==list(range(len(m))) and set(os['state'])==set(g['params'])
 for (name,t),idx in zip(m.items(),g['params']):
  state=os['state'][idx];assert set(state)=={'step','exp_avg','exp_avg_sq'}
  assert float(state['step'])==c['updates']
  for k in ['exp_avg','exp_avg_sq']:
   a=state[k];assert a.shape==t.shape and a.dtype==t.dtype and torch.isfinite(a).all()
  assert (state['exp_avg_sq']>=0).all()
 stats={'parameter_l2':sum(float(t.square().sum()) for t in m.values())**.5, 'exp_avg_l2':sum(float(v['exp_avg'].square().sum()) for v in os['state'].values())**.5,'exp_avg_sq_sum':sum(float(v['exp_avg_sq'].sum()) for v in os['state'].values())}
 return {'optimizer_tensor_stats':stats,'path':rel(p),'file_sha256':HASHES[rel(p)]['sha256'],'epoch':c['epoch'],'updates':c['updates'],'total_steps':c['total_steps'],'total_episodes':c['total_episodes'],'model_sha256':signature['sha256'],'parameters':sum(t.numel() for t in m.values()),'parameter_tensors':len(m),'optimizer_steps':c['updates'],'model_and_optimizer_finite':True,'source_sha256':c['source_sha256']}

def audit_episodes(seed):
 """Stream all JSON once, accumulating update/epoch summaries and hashing input bytes."""
 start=time.monotonic();d=NEW/f'seed{seed}';ms=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()];us=[json.loads(x) for x in (d/'updates.jsonl').read_text().splitlines()]
 acc=[Counter() for _ in range(len(us))];epoch_acc=[Counter() for _ in ms];lengths=Counter();h=hashlib.sha256();count=0;prev_update=0;prev_col=-1;scenarios=set();checked_scenarios=0
 floatkeys=['team_return','mean_agent_return','return_cap','return_nocap','gate_entropy','alpha_null']
 with (d/'episodes.jsonl').open('rb') as f:
  for raw in f:
   h.update(raw);r=json.loads(raw);u=r['update'];col=r['collector']
   assert prev_update<=u<=len(us) and 0<=col<4
   if u!=prev_update:
    assert u==prev_update+1
    scenarios=set();prev_col=-1;prev_update=u
   assert col>=prev_col;prev_col=col
   key=(col,r['scenario_id']);assert key not in scenarios;scenarios.add(key);checked_scenarios+=1
   assert r['composition']==[2,1] and r['num_p']==2 and r['num_a']==1 and r['num_agents']==3
   assert r['environment_version']=='corrected-observation-v1' and r['event_step']==-1 and not r['event_exposed'] and not r['scheduled_event_exposed'] and not r['sham']
   assert 1<=r['steps']<=80 and (r['success'] or r['steps']==80)
   assert r['expert_load']==[1.0] and r['gate_entropy']==0 and 0<=r['alpha_null']<=1
   assert r['payload_bits_generated']==r['steps']*96 and r['payload_bits_per_agent_step']==32
   assert all(math.isfinite(r[k]) for k in floatkeys) and all(math.isfinite(x) for x in r['agent_returns'])
   assert math.isclose(sum(r['agent_returns']),r['team_return'],abs_tol=1e-12)
   assert math.isclose(r['mean_agent_return']*3,r['team_return'],abs_tol=1e-12)
   a=acc[u-1];a['episodes']+=1;a['steps']+=r['steps'];a['successes']+=r['success'];a[f'collector_{col}_steps']+=r['steps']
   e=epoch_acc[(u-1)//10]
   e['episodes']+=1;e['steps']+=r['steps'];e['success_rate']+=r['success']
   for k in floatkeys:e[k]+=r[k]
   lengths[r['steps']]+=1;count+=1
   if count%1000000==0:print(f'episode audit seed{seed}: {count:,}',flush=True)
 for u,a in zip(us,acc):
  assert a['episodes']==u['episodes'] and a['steps']==u['steps']
  assert all(a[f'collector_{col}_steps']>=500 for col in range(4))
 for m,a in zip(ms,epoch_acc):
  assert m['episodes']==a['episodes'] and m['steps']==a['steps']
  for k in floatkeys+['success_rate']:assert math.isclose(a[k]/a['episodes'],m[k],rel_tol=1e-10,abs_tol=1e-11),(seed,m['epoch'],k)
 assert count==ms[-1]['total_episodes'] and sum(a['steps'] for a in acc)==ms[-1]['total_steps']
 result={'seed':seed,'path':rel(d/'episodes.jsonl'),'bytes':(d/'episodes.jsonl').stat().st_size,'sha256':h.hexdigest(),'episodes':count,'steps':sum(a['steps'] for a in acc),'updates_reconciled':len(us),'epochs_reconciled':len(ms),'successes':sum(a['successes'] for a in acc),'length_histogram':dict(sorted(lengths.items())),'within_update_scenario_identities_checked':checked_scenarios,'final_epoch_recomputed':{k:epoch_acc[-1][k]/epoch_acc[-1]['episodes'] for k in floatkeys+['success_rate']},'all_validations_pass':True,'audit_seconds':time.monotonic()-start}
 save(f'episodes_seed{seed}.json',result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--episodes',action='store_true');args=parser.parse_args()
 allruns=[];windows=[];cps=[];sourcecomparison={};all_epochs=[]
 for seed in range(3):
  d=NEW/f'seed{seed}';cfg=read(d/'config.json');run=read(d/'run.json');fin=read(d/'finished.json');manifest=read(d/'source_manifest.json');initial=read(d/'initial_signature.json')
  assert manifest['sha256']==hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()
  for path,sha in manifest['files'].items():assert digest(d/'source'/path)==sha,(d,path)
  ms=rows(d/'metrics.jsonl');stdout=rows(ROOT/f'logs_sr/softrole-adam-pcp-911488_{seed}.out');us=rows(d/'updates.jsonl');assert len(ms)==len(stdout)
  stdout_discrepancies=[{'epoch':i+1,'field':k,'structured_value':a.get(k),'stdout_value':b.get(k)} for i,(a,b) in enumerate(zip(ms,stdout)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)]
  digest(ROOT/f'logs_sr/softrole-adam-pcp-911488_{seed}.err')
  assert len(ms)==2000 and [r['epoch'] for r in ms]==list(range(1,2001)) and [r['update'] for r in us]==list(range(1,20001))
  assert all(not r['partial_epoch'] and r['event_exposed_episodes']==0 for r in ms)
  cumsteps=cumeps=0
  for i,r in enumerate(ms):
   seg=us[i*10:(i+1)*10];assert sum(x['steps'] for x in seg)==r['steps'] and sum(x['episodes'] for x in seg)==r['episodes']
   cumsteps+=r['steps'];cumeps+=r['episodes'];assert (cumsteps,cumeps,(i+1)*10)==(r['total_steps'],r['total_episodes'],r['updates'])
   assert math.isclose(r['steps']/r['episodes'],r['steps_taken'],abs_tol=1e-12)
  assert fin['total_steps']==cumsteps and fin['total_episodes']==cumeps and fin['updates']==20000
  assert run['resume'] is None and run['optimizer_initialization']=='fresh' and run['source_sha256']==manifest['sha256']
  cs=sorted((d/'checkpoints').glob('*.pt'));assert len(cs)==40
  for p in cs:cps.append(checkpoint(p,cfg,manifest,ms))
  assert cps[-1]['model_sha256']!=initial['sha256']
  a={'group':'adam_shared','seed':seed,'run_path':rel(d),'config':cfg,'run':run,'source_files_verified':len(manifest['files']),'git_commit':manifest['git_commit'],'git_status_lines':len(manifest['git_status'].splitlines()),'final':ms[-1], 'epochs':len(ms),'all_counts_reconciled':True,'stdout_discrepancies':stdout_discrepancies,'checkpoint_count':len(cs),'recorded_epoch_seconds':sum(r['wall_time_seconds'] for r in ms),'gradient_norm_min':min(r['gradient_norm_before_clip'] for r in us),'gradient_norm_median':statistics.median(r['gradient_norm_before_clip'] for r in us),'gradient_norm_max':max(r['gradient_norm_before_clip'] for r in us),'clipped_updates':sum(r['gradient_norm_before_clip']>.75 for r in us),'initial_model_sha256':initial['sha256'],'clipped_update_fraction':sum(r['gradient_norm_before_clip']>.75 for r in us)/len(us)}
  allruns.append(a)
  for lo,hi in WINDOWS:windows.append(window([r for r in ms if lo*1e6<r['total_steps']<=hi*1e6],f'{lo}-{hi}M','adam_shared',seed,(lo*1000000,hi*1000000)))
  windows.append(window(ms[-100:],'last100','adam_shared',seed,(None,None)))
  all_epochs.extend(dict(r,group='adam_shared',seed=seed) for r in ms)
  print('checkpoint/source/metric audit complete',seed,flush=True)
 for model,offset in [('shared',6),('banked',9)]:
  for seed in range(3):
   d=OLD/f'pcp_{model}'/f'seed{seed}';cfg=read(d/'config.json');ms=rows(ROOT/f'logs_sr/softrole-898818_{offset+seed}.out');assert cfg['model']==model and cfg['seed']==seed
   prefix=None
   if (d/'metrics.jsonl').exists():
    native=rows(d/'metrics.jsonl');assert native==ms[:len(native)];prefix=len(native)
   assert [r['epoch'] for r in ms]==list(range(1,len(ms)+1))
   assert sum(r['steps'] for r in ms)==ms[-1]['total_steps'] and sum(r['episodes'] for r in ms)==ms[-1]['total_episodes']
   allruns.append({'group':f'rmsprop_{model}','seed':seed,'run_path':rel(d),'config':cfg,'final':ms[-1],'epochs':len(ms),'matching_archive_epoch_prefix':prefix,'recorded_epoch_seconds':sum(r['wall_time_seconds'] for r in ms)})
   for lo,hi in WINDOWS:windows.append(window([r for r in ms if lo*1e6<r['total_steps']<=hi*1e6],f'{lo}-{hi}M',f'rmsprop_{model}',seed,(lo*1000000,hi*1000000)))
   windows.append(window(ms[-100:],'last100',f'rmsprop_{model}',seed,(None,None)))
   all_epochs.extend(dict(r,group=f'rmsprop_{model}',seed=seed) for r in ms)
 old_terminal=[]
 import torch
 for model in ['shared','banked']:
  for seed in range(3):
   d=OLD/f'pcp_{model}'/f'seed{seed}';ps=sorted((d/'checkpoints').glob('*.pt'))
   if not ps:
    old_terminal.append({'model':model,'seed':seed,'available':False});continue
   p=ps[-1];digest(p);c=torch.load(p,map_location='cpu',weights_only=False);os=c['optimizer_state'];g=os['param_groups'][0]
   assert g['lr']==1e-4 and g['alpha']==.97 and g['eps']==1e-6 and g['momentum']==0 and not g['centered']
   assert all(torch.isfinite(t).all() for t in c['model_state'].values())
   for t,state in zip(c['model_state'].values(),os['state'].values()):
    assert set(state)=={'step','square_avg'} and float(state['step'])==c['updates']
    assert state['square_avg'].shape==t.shape and state['square_avg'].dtype==t.dtype
    assert torch.isfinite(state['square_avg']).all() and (state['square_avg']>=0).all()
   match=next(r for r in all_epochs if r['group']==f'rmsprop_{model}' and r['seed']==seed and r['epoch']==c['epoch'])
   assert all(match[k]==c[k] for k in ['updates','total_steps','total_episodes'])
   old_terminal.append({'model':model,'seed':seed,'available':True,'path':rel(p),'sha256':HASHES[rel(p)]['sha256'],'epoch':c['epoch'],'updates':c['updates'],'steps':c['total_steps'],'episodes':c['total_episodes'],'actual_optimizer':'RMSprop','finite_model_and_optimizer':True})
 save('old_terminal_checkpoints.json',old_terminal)
 oldsource=read(OLD/'pcp_shared/seed0/source_manifest.json');newsource=read(NEW/'seed0/source_manifest.json')
 for p in sorted(newsource['files']):
  if p.startswith(('softrole/','envs/')):sourcecomparison[p]={'new':newsource['files'][p],'old':oldsource['files'].get(p),'same':newsource['files'][p]==oldsource['files'].get(p)}
 # Same-seed initialization is checked using saved signatures, not inferred from seed labels.
 for seed in range(3):
  p=OLD/f'pcp_shared/seed{seed}'/'initial_signature.json'
  if p.exists():
   allruns[seed]['initial_signature_matches_old_shared']=read(p)==read(NEW/f'seed{seed}'/'initial_signature.json')
  else:allruns[seed]['initial_signature_matches_old_shared']=None
 grouped=[]
 for group in sorted({w['group'] for w in windows}):
  for label in [f'{a}-{b}M' for a,b in WINDOWS]+['last100']:
   ws=[w for w in windows if w['group']==group and w['window']==label and w['episodes']]
   grouped.append({'group':group,'window':label,'n_seeds':len(ws),'seeds':[w['seed'] for w in ws],**{k:statistics.mean(w[k] for w in ws) if ws else None for k in FIELDS},**{k+'_seed_min':min(w[k] for w in ws) if ws else None for k in ['steps_taken','success_rate','team_return']},**{k+'_seed_max':max(w[k] for w in ws) if ws else None for k in ['steps_taken','success_rate','team_return']}})
 result={'scope':'Training records; no frozen evaluations or training were run. Within-window epoch means are episode weighted, then seeds equally weighted. New archives fully audited; old comparisons use matching stdout and available source/config prefixes.','runs':allruns,'group_windows':grouped,'new_runtime_source_vs_old':sourcecomparison,'checkpoints_verified':len(cps),'window_membership':'lower < epoch-end total_steps <= upper; complete epoch records retained; use actual_start_steps and actual_end_steps for realized coverage','limitations':['Epoch wall times exclude checkpoint save and some setup/shutdown; no absolute job elapsed time inferred.','Old shared seed0 stopped early;39-40M RMSprop shared group has two seeds.','Old shared seed2 has no copied structured metrics; stdout plus configuration support its mapping.','Source bytes include later logging and inactive domain options; observational optimizer comparison, not a same-checkout controlled ablation.']}
 thresholds=[]
 for run in allruns:
  rs=[r for r in all_epochs if r['group']==run['group'] and r['seed']==run['seed']]
  for metric,threshold in [('steps_taken',20),('steps_taken',10),('steps_taken',7),('steps_taken',6),('success_rate',.99)]:
   found=None
   for i in range(99,len(rs)):
    rr=rs[i-99:i+1];val=sum(r[metric]*r['episodes'] for r in rr)/sum(r['episodes'] for r in rr)
    if (val<=threshold if metric=='steps_taken' else val>=threshold):
     found={'first_epoch':rr[0]['epoch'],'last_epoch':rr[-1]['epoch'],'end_steps':rr[-1]['total_steps'],'value':val};break
   thresholds.append({'group':run['group'],'seed':run['seed'],'metric':metric,'threshold':threshold,'first_100epoch_crossing':found})
 save('thresholds.json',thresholds)
 save('review.json',result);save('input_hashes.json',HASHES);save('checkpoints.json',cps);csvsave('windows.csv',windows);csvsave('epoch_metrics.csv',all_epochs);csvsave('group_windows.csv',grouped)
 if args.episodes:
  with ProcessPoolExecutor(max_workers=3) as pool:res=list(pool.map(audit_episodes,range(3)))
  save('episode_audit.json',res)
 print(json.dumps({'runs':len(allruns),'checkpoints':len(cps),'new_total_steps':sum(x['final']['total_steps'] for x in allruns[:3]),'new_total_episodes':sum(x['final']['total_episodes'] for x in allruns[:3]),'output':str(OUT)},sort_keys=True),flush=True)
if __name__=='__main__':main()
