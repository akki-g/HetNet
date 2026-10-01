"""Generate exact report tables and a local evidence index; no policy execution."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAP = HERE / 'latest_snapshot'
summary = json.loads((SNAP / 'summary.json').read_text())

def write(name, text):
    (HERE / name).write_text(text + '\n')

rows = []
for task in ('pp', 'pcp', 'fc'):
    end = summary['common_epoch_ends'][task]
    for g in [g for g in summary['groups'] if g['task'] == task]:
        m = g['common_epoch']
        model = g['model'].replace('SoftRole ', '')
        rows.append(f"{task.upper()} / {end-49}--{end} & {model} & {100*m['success_rate']['mean']:.2f} & {m['team_return']['mean']:.3f} & {m['steps_taken']['mean']:.2f} " + r'\\')
    if task != 'fc': rows.append(r'\addlinespace[3pt]')
write('common_epoch_rows.tex', '\n'.join(rows))

for task in ('pp', 'pcp', 'fc'):
    rows = []
    for g in [g for g in summary['groups'] if g['task'] == task and g['model'] != 'HetNet Real']:
        m = g['common_sample']
        precision = 3 if task != 'fc' else 2
        rows.append(f"{g['model'].split()[-1].title()} & {100*m['success_rate']['mean']:.{precision}f} & {m['team_return']['mean']:.3f} & {m['steps_taken']['mean']:.2f} " + r'\\')
    write(f'{task}_sample_rows.tex', '\n'.join(rows))

rows = []
for task in ('pp','pcp','fc'):
    for model in ('HetNet Real','SoftRole shared','SoftRole banked'):
        runs = [r for r in summary['runs'] if r['task']==task and r['model']==model]
        epochs = f"{min(r['epochs'] for r in runs)}--{max(r['epochs'] for r in runs)}"
        steps = '--' if model == 'HetNet Real' else f"{min(r['total_steps'] for r in runs)/1e6:.2f}--{max(r['total_steps'] for r in runs)/1e6:.2f}"
        rows.append(f"{task.upper()} & {model} & {epochs} & {steps} " + r'\\')
write('coverage_rows.tex', '\n'.join(rows))

pilot_path = ROOT / 'runs/sensor_failure_validation_20260930/pilot/summary.json'
pilot = json.loads(pilot_path.read_text())
rows = []
for p in sorted(pilot['policies'], key=lambda p:p['model'], reverse=True):
    f,s = p['failure'],p['sham']
    rows.append(f"{p['model'].title()} & {round(s['success_rate']*s['episodes'])}/20 & {round(f['success_rate']*f['episodes'])}/20 & {s['team_return']:.4f} / {f['team_return']:.4f} & {f['actual_event_exposed']}/20 " + r'\\')
write('pilot_rows.tex','\n'.join(rows))

evidence = {
    'E1': {'claim':'Current log inventory, common-epoch and common-sample comparisons, window endpoints and three-seed statistics',
           'files':['analysis/project_overview_2026-09-30/latest_snapshot/summary.json','analysis/project_overview_2026-09-30/latest_snapshot/epoch_metrics.csv','analysis/project_overview_2026-09-30/latest_snapshot/provenance.json','analysis/project_overview_2026-09-30/refresh_results.py','analysis/training_2026-09-30_resync/summary.json','analysis/training_2026-09-30_resync/provenance.json','analysis/training_2026-09-30_resync/analyze.py','analysis/training_2026-09-30/analyze.py'],
           'lookup':'summary.groups[task,model].common_epoch/common_sample/prior_common_sample/latest; summary.runs includes source paths and windows. CSV source_line maps to raw stdout.'},
    'E2': {'claim':'Implemented observation, architecture, training and task rules',
           'files':['softrole/RESEARCH.md','softrole/model.py','softrole/env.py','softrole/learning.py','softrole/train.py','softrole/rollout.py','softrole/config.py','softrole/scenarios.py','hetgat/uavnet.py','envs/ic3net_envs/predator_capture_env.py','envs/ic3net_envs/fire_commander_env.py','main.py','trainer.py','multi_processing.py'],
           'lookup':'Named classes/functions in report; current source describes implementation. Archived training source provenance is separately recorded in E1.'},
    'E3': {'claim':'FC late regression, simulator defect and older frozen-panel diagnostics',
           'files':['analysis/fc_audit_2026-09-30/log_findings.json','analysis/fc_audit_2026-09-30/simulator_probe.json','analysis/fc_audit_2026-09-30/policy_panel/summary.json','analysis/fc_audit_2026-09-30/learner_findings.json','analysis/fc_audit_2026-09-30/artifact_manifest.json','analysis/fc_audit_2026-09-30/simulator_probe.py','analysis/fc_audit_2026-09-30/log_probe.py','analysis/fc_audit_2026-09-30/policy_probe.py','WildFire_Simulate_Original.py'],
           'lookup':'simulator_probe.deterministic_checks.immediate_origin_additions_of_25_cells; passive_panel; policy_panel summary; log_findings.softrole.'},
    'E4': {'claim':'Separate one-seed paired PCP sensor-loss pilot',
           'files':['runs/sensor_failure_validation_20260930/pilot/summary.json','runs/sensor_failure_validation_20260930/pilot/pilot.json','runs/sensor_failure_validation_20260930/pilot/scenarios.json','runs/sensor_failure_validation_20260930/validation.json'],
           'lookup':'summary.policies[model].failure/sham/failure_minus_sham; checkpoint_progress; diagnostic_true_counts.'},
    'E5': {'claim':'Previously recorded engineering validations and scope', 'files':['AGENTS.md'],
           'lookup':'2026-09-30 reproducible native PCP sensor-failure pilot: 178 tests; FC audit: targeted 40 tests. Historical records, not rerun for PDF.'}
}
write('evidence_index.json',json.dumps(evidence,indent=2))
inputs = set()
for e in evidence.values(): inputs.update(e['files'])
inputs.update(str(p.relative_to(ROOT)) for d in ('logs_1','logs_sr') for p in (ROOT/d).glob('*') if p.suffix in ('.out','.err'))
# AGENTS is deliberately updated after writing the report; hash it at final validation.
manifest = {p:{'bytes':(ROOT/p).stat().st_size,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sorted(inputs) if p!='AGENTS.md'}
write('input_manifest.json',json.dumps(manifest,indent=2))
with (HERE/'comparison_values.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['task','model','window','metric','mean','seed_min','seed_max'])
    for g in summary['groups']:
        for window in ('common_epoch','common_sample','prior_common_sample','latest'):
            for metric, values in g.get(window,{}).items():
                w.writerow([g['task'],g['model'],window,metric,values['mean'],values['min'],values['max']])
print(json.dumps({'runs':len(summary['runs']),'epochs':sum(r['epochs'] for r in summary['runs']),'softrole_steps':sum(r.get('total_steps',0) for r in summary['runs']),'inputs':len(manifest)},indent=2))
