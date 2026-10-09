"""Read-only PCP source and paper audit. No model/environment execution."""
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path):
    return json.loads(path.read_text())

def methods(path, cls='PredatorCaptureEnv'):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return {n.name: ast.dump(n, include_attributes=False) for n in node.body if isinstance(n, ast.FunctionDef)}

new = ROOT/'stokes_runs/softrole_adam_pcp_911488/pcp_shared/seed0/source'
old = ROOT/'stokes_runs/softrole_primary/pcp_shared/seed0/source'
rel = Path('envs/ic3net_envs/predator_capture_env.py')
paths = {
    'adam_pcp_simulator': new/rel,
    'historical_softrole_pcp_simulator': old/rel,
    'working_pcp_simulator': ROOT/rel,
    'paper_v1_pcp_simulator': ROOT/'publication_reconstruction/runtime'/rel,
    'upstream_pcp_simulator_fetched_20261008': OUT/'upstream_predator_capture_env.py',
    'upstream_reward_reporting_fetched_20261008': OUT/'upstream_print_plot_eval.py',
    'paper_pdf': OUT/'hetnet_paper.pdf',
    'supplement_pdf': OUT/'hetnet_supplement.pdf',
    'local_paper_text': ROOT/'research/papers/HetNet.txt',
    'local_supplement_text': ROOT/'research/papers/HetNet_Supplementary.txt',
    'adam_adapter':new/'softrole/env.py',
    'historical_softrole_adapter':old/'softrole/env.py',
    'fidelity_contract':ROOT/'publication_reconstruction/FIDELITY.md',
}
base = methods(paths['adam_pcp_simulator'])
comparisons = {}
for label in ('historical_softrole_pcp_simulator', 'working_pcp_simulator', 'paper_v1_pcp_simulator', 'upstream_pcp_simulator_fetched_20261008'):
    other = methods(paths[label])
    comparisons[label] = {
        'same_bytes': sha(paths[label]) == sha(paths['adam_pcp_simulator']),
        'identical_method_AST': sorted(k for k in base if base[k] == other.get(k)),
        'different_method_AST': sorted(k for k in base if base[k] != other.get(k)),
    }

critical = ['step', 'reset', '_get_cordinates', '_take_action', '_get_reward', 'reward_terminal']
assert all(all(n in comparisons[l]['identical_method_AST'] for n in critical) for l in comparisons)
assert comparisons['historical_softrole_pcp_simulator']['same_bytes']
assert comparisons['working_pcp_simulator']['same_bytes']

configs=[]
for seed in range(3):
    oldp=ROOT/f'stokes_runs/softrole_primary/pcp_shared/seed{seed}/config.json'
    newp=ROOT/f'stokes_runs/softrole_adam_pcp_911488/pcp_shared/seed{seed}/config.json'
    a,b=read_json(oldp),read_json(newp)
    paths[f'old_config_seed{seed}']=oldp
    paths[f'adam_config_seed{seed}']=newp
    changes={k:{'old':a.get(k,'MISSING'),'adam':b.get(k,'MISSING')} for k in sorted(a.keys() | b.keys()) if a.get(k,'MISSING') != b.get(k,'MISSING')}
    configs.append({'seed':seed,'delta':changes,'adam_environment': {k:b[k] for k in ('task','dim','vision','num_p','num_a','max_steps','env_version','failure_prob','compositions','comm_range') }})
    assert b['task']=='pcp' and b['num_p']==2 and b['num_a']==1 and b['max_steps']==80
    assert b['vision']==2 and b['dim']==5 and b['env_version']=='corrected-observation-v1'
    assert b['failure_prob']==0 and b['compositions']==[]

# Static exhaustive reset-layout count; original aliasing erases target channels
# within blind A's radius-2 window for every overlapping observer.
cells=list(itertools.product(range(5),repeat=2))
counter=Counter(); total_T=0; total_penalty_steps=0
cheb=lambda a,b:max(abs(a[0]-b[0]),abs(a[1]-b[1]))
man=lambda a,b:abs(a[0]-b[0])+abs(a[1]-b[1])
for target,p0,p1,a in itertools.permutations(cells,4):
    counter['layouts']+=1
    visible = cheb(target,p0)<=2 or cheb(target,p1)<=2
    erased = cheb(target,a)<=2
    counter['corrected_any_P_sees_target_initially']+=visible
    counter['historical_any_P_sees_target_initially']+=visible and not erased
    counter['restored_visibility_initially']+=visible and erased
    ds=(man(target,p0),man(target,p1),man(target,a)+1)
    total_T += max(ds)
    total_penalty_steps += sum(ds)-3
assert counter['layouts']==303600
# Separate closed-form count by target cell, without enumerating all layouts.
corrected_count=historical_count=0
for target in cells:
    visible_cells=sum(cheb(target,p)<=2 for p in cells if p!=target)
    corrected_count+=(24*23-(24-visible_cells)*(23-visible_cells))*22
    historical_count+=(24-visible_cells)*(23*22-(23-visible_cells)*(22-visible_cells))
assert corrected_count==counter['corrected_any_P_sees_target_initially']
assert historical_count==counter['historical_any_P_sees_target_initially']
geom = dict(counter)
geom.update({k+'_fraction':counter[k]/counter['layouts'] for k in counter if k!='layouts'})
geom.update(omniscient_mean_episode_steps=total_T/counter['layouts'],
            omniscient_mean_team_return=-.05*total_penalty_steps/counter['layouts'],
            interpretation='Static uniform-distinct layout population; no policy/environment rollouts. Independent-copy correction restores intended P target channel; effect on learned performance is unmeasured.')

data={
 'review_date':'2026-10-08',
 'scope':'Read-only archive/paper/code analysis; no training or frozen-policy execution.',
 'files':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for k,p in paths.items()},
 'method_AST_comparisons':comparisons,
 'critical_method_AST_equality_asserted':critical,
 'adam_vs_historical_shared_configs':configs,
 'static_initial_visibility_and_oracle':geom,
 'publication_table_1':{
    'sample':'50 evaluation trials; final policy at convergence; exact historical checkout/panel and pooling unavailable',
    'pcp':{'steps_mean':9.98,'steps_SE':.36,'return_mean':-.364,'return_SE':.017},
    'pp':{'steps_mean':8.30,'steps_SE':.25,'return_mean':-.232,'return_SE':.010},
    'fc':{'steps_mean':46.40,'steps_SE':2.90,'return_mean':-9.862,'return_SE':2.77},
    'reward_scale':'Current upstream print_plot_eval averages across episodes and agents. Historical Table 1 implementation not established.'
 },
 'publication_figure_3':{'metric':'training episode steps, means and SE, 3 seeds','x_axis':'epochs, roughly through 2000','endpoint_interpretation':'PCP HetNet visually around 10; no exact digitized values recovered','unverified':'number of samples/updates behind every publication epoch'},
 'paper_supplement_protocol':{'pcp_team':[2,1],'grid':[5,5],'max_steps':80,'prey':'stationary','P_actions':'4 movement + stay','A_actions':'4 movement + stay + capture','reward':-.05,'reward_rule':'per-agent until reaching target (P) or capturing at target (A)','vision_radius':'not numerically specified in paper/supplement; 2 from public code','training_seeds':[0,1,2],'optimizer':{'name':'Adam','lr':.001}},
 'interpretation':{
    'physical_dynamics_changed_for_Adam':False,
    'observation_model_change_from_historical_release':True,
    'copy_correction_cause_of_paper_gap':'plausible but unproven: publication-producing source unknown',
    'recommended_scope':'Keep native benchmark unchanged; report paper scores as contextual references, matched frozen comparison for our trained methods; any harder variant is separately preregistered.'
 }
}
(OUT/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'checks':'passed','config_deltas':configs,'geometry':geom,'source_comparison':comparisons},indent=2))
