"""Analyze copied structured logs; never execute a policy or modify run artifacts.

uv run --no-project --with matplotlib==3.10.7 --with numpy==1.26.4 \
  --python .venv/bin/python python analysis/training_2026-09-30/full_runs/analyze_full.py

Primary comparisons: last 50 completed epochs ending before a common task step
budget; episode weights within runs, equal weights across three training runs.
The epoch boundary prevents exact equality of consumed samples; bounds are saved.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
INPUT = ROOT / 'stokes_runs/runs'
TASKS = ('pp', 'pcp', 'fc')
MODELS = ('HetNet Real', 'SoftRole shared', 'SoftRole banked')
COLORS = dict(zip(MODELS, ('#667085', '#0072B2', '#D55E00')))
STYLES = ('-', '--', ':')
METRICS = ('success_rate', 'team_return', 'steps_taken')
LABELS = ('Training success fraction', 'Mean team return', 'Episode length (steps)')


def read_jsonl(path):
    rows = []
    for n, line in enumerate(path.open(), 1):
        row = json.loads(line)
        row['source_line'] = n
        rows.append(row)
    return rows


def weighted(rows, key):
    return float(np.average([r[key] for r in rows], weights=[r['episodes'] for r in rows]))


def window(rows):
    return {'epoch_start': rows[0]['epoch'], 'epoch_end': rows[-1]['epoch'],
            'steps_start': rows[0]['total_steps'] - rows[0]['steps'],
            'steps_end': rows[-1]['total_steps'],
            'episodes': sum(r['episodes'] for r in rows),
            **{k: weighted(rows, k) for k in METRICS}}


def rolling(rows, key):
    return [weighted(rows[max(0, i-49):i+1], key) for i in range(len(rows))]


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_runs():
    runs, all_epochs = [], []
    for path in sorted(INPUT.glob('*/*/seed*/metrics.jsonl')):
        suite, label, seed = path.parts[-4:-1]
        primary = suite in ('reproduction-fast', 'softrole_primary')
        sr = suite == 'softrole_primary'
        config = json.loads((path.parent / ('config.json' if sr else 'resolved_args.json')).read_text())
        task = label.split('_')[0]
        model = 'SoftRole ' + config['model'] if sr else MODELS[0]
        rows = read_jsonl(path)
        episodes = steps = 0
        for i, row in enumerate(rows, 1):
            assert row['epoch'] == i
            steps += row['steps']; episodes += row['episodes']
            assert (steps, episodes) == (row['total_steps'], row['total_episodes'])
            if not sr:
                row['team_return'] = sum(row['reward_per_agent'])
            assert 0 <= row['success_rate'] <= 1
            assert abs(row['steps_taken'] * row['episodes'] - row['steps']) < 1e-6
            assert all(np.isfinite(row[k]) for k in METRICS)
            assert abs(row['success_rate'] * row['episodes'] - round(row['success_rate'] * row['episodes'])) < 1e-7
            all_epochs.append(dict(suite=suite, task=task, model=model, seed=int(seed[4:]),
                                   primary=primary, source=str(path.relative_to(ROOT)), **row))
        run = dict(suite=suite, task=task, model=model, seed=int(seed[4:]), primary=primary,
                   path=str(path.parent.relative_to(ROOT)), config=config, rows=rows)
        if sr:
            run['updates'] = read_jsonl(path.parent / 'updates.jsonl')
            assert [r['update'] for r in run['updates']] == list(range(1, len(run['updates'])+1))
            run['metadata'] = json.loads((path.parent/'run.json').read_text())
        runs.append(run)
    csv_write(OUT/'epoch_metrics.csv', all_epochs)
    return runs


def summaries(runs):
    primary = [r for r in runs if r['primary']]
    assert len(primary) == 27
    budgets = {t: min(r['rows'][-1]['total_steps'] for r in primary if r['task']==t)//500000*500000 for t in TASKS}
    result = {'method': 'Episode-weighted last 50 completed epochs at/before common task sample budget; equal-run group means',
              'budgets': budgets, 'runs': [], 'groups': []}
    for r in runs:
        rows = r['rows']
        item = {k: r[k] for k in ('suite','task','model','seed','primary','path')}
        item.update(epochs=len(rows), steps=rows[-1]['total_steps'], episodes=rows[-1]['total_episodes'],
                    epoch_seconds=sum(x['wall_time_seconds'] for x in rows),
                    first50=window(rows[:50]), latest50=window(rows[-50:]),
                    configured_epochs=r['config'].get('epochs',r['config'].get('num_epochs')))
        if r['primary']:
            common = [x for x in rows if x['total_steps'] <= budgets[r['task']]][-50:]
            assert len(common)==50
            item['common'] = window(common)
            item['before_latest50'] = window(rows[-100:-50])
            item['common_4m'] = window([x for x in rows if x['total_steps'] <= 4000000][-50:])
        if 'updates' in r:
            cap=r['config']['max_grad_norm']
            norms=np.array([u['gradient_norm_before_clip'] for u in r['updates']])
            assert np.isfinite(norms).all()
            item['gradient']={name: {'updates':len(a),'clip_fraction':float(np.mean(a>cap)),
                'norm_median':float(np.median(a)), 'norm_p95':float(np.quantile(a,.95)),
                'scale_median':float(np.median(np.minimum(1,cap/(a+1e-6))))}
                for name,a in [('all',norms),('latest500',norms[-500:])]}
            item['parameters']=r['metadata']['parameters']
            item['event_exposed']=sum(x['event_exposed_episodes'] for x in rows)
            item['updates_last']=r['updates'][-1]['update']
            item['value_first50']=weighted(rows[:50],'value_loss_per_episode')
            item['value_latest50']=weighted(rows[-50:],'value_loss_per_episode')
        result['runs'].append(item)
    for t in TASKS:
        for m in MODELS:
            members=[r for r in result['runs'] if r['primary'] and r['task']==t and r['model']==m]
            assert len(members)==3
            group={'task':t,'model':m,'budget':budgets[t]}
            for w in ['common','latest50','common_4m']:
                group[w]={k:{'mean':float(np.mean([r[w][k] for r in members])),
                             'min':min(r[w][k] for r in members),'max':max(r[w][k] for r in members)} for k in METRICS}
            result['groups'].append(group)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    csv_write(OUT/'run_summary.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} |
              {f'{w}_{k}':v for w in ['first50','latest50','common'] for k,v in r.get(w,{}).items()} for r in result['runs']])
    return result


def figures(runs, summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    primary=[r for r in runs if r['primary']]
    pages=[]
    fig,axes=plt.subplots(3,3,figsize=(12,9.2),sharex='col')
    for col,t in enumerate(TASKS):
        for row,(k,label) in enumerate(zip(METRICS,LABELS)):
            ax=axes[row,col]
            for r in primary:
                if r['task']!=t:continue
                x=np.array([v['total_steps'] for v in r['rows']])/1e6
                ax.plot(x,[v[k] for v in r['rows']],color=COLORS[r['model']],alpha=.10,lw=.45)
                ax.plot(x,rolling(r['rows'],k),color=COLORS[r['model']],ls=STYLES[r['seed']],lw=1.25)
            ax.axvline(summary['budgets'][t]/1e6,color='#555555',ls=':',lw=.8)
            if row==0:ax.set_title(t.upper());ax.set_ylim(-.025,1.025)
            if col==0:ax.set_ylabel(label)
            if row==2:ax.set_xlabel('Recorded environment steps (millions)')
            ax.grid(alpha=.15)
    legend=[Line2D([],[],color=COLORS[m],label=m) for m in MODELS]+[Line2D([],[],color='k',ls=STYLES[s],label=f'Seed {s}') for s in range(3)]
    fig.legend(handles=legend,loc='lower center',ncol=6,bbox_to_anchor=(.5,.015))
    fig.suptitle('Full run artifacts: training outcomes against actual environment samples',fontsize=14)
    fig.text(.5,.935,'Lines: trailing 50 epochs, weighted by episodes. Faint: raw epochs. Vertical: common-budget comparison.',ha='center')
    fig.tight_layout(rect=(0,.065,1,.91));pages.append(('sample_learning_curves',fig))

    fig,axes=plt.subplots(3,3,figsize=(12,8.5))
    for col,t in enumerate(TASKS):
        for row,(key,label) in enumerate(zip(METRICS,LABELS)):
            ax=axes[row,col]
            for i,m in enumerate(MODELS):
                vals=[r['common'][key] for r in summary['runs'] if r['primary'] and r['task']==t and r['model']==m]
                ax.plot([i,i],[min(vals),max(vals)],color=COLORS[m],lw=2)
                ax.scatter(i+np.array([-.08,0,.08]),vals,color=COLORS[m],s=30)
                ax.scatter(i,np.mean(vals),color='k',marker='D',s=23,zorder=5)
            ax.set_xticks(range(3),['HetNet\nReal','SoftRole\nshared','SoftRole\nbanked'])
            if row==0:ax.set_title(f'{t.upper()} near {summary["budgets"][t]/1e6:g}M steps');ax.set_ylim(-.025,1.025)
            if col==0:ax.set_ylabel(label)
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('Common sample-budget windows: three run outcomes and equal-run means',fontsize=13)
    fig.text(.5,.943,'Last 50 completed epochs at/before the budget; boundary shortfalls recorded. Ranges are not confidence intervals.',ha='center')
    fig.tight_layout(rect=(0,0,1,.92));pages.append(('common_sample_comparison',fig))

    fig,axes=plt.subplots(3,3,figsize=(12,8.5),sharex='col')
    for col,t in enumerate(TASKS):
        for r in primary:
            if r['task']!=t or 'updates' not in r:continue
            color,style=COLORS[r['model']],STYLES[r['seed']]
            us=r['updates'];x=np.array([u['total_steps'] for u in us])/1e6
            norms=np.array([u['gradient_norm_before_clip'] for u in us])
            smooth=lambda v: [float(np.median(v[max(0,i-99):i+1])) for i in range(len(v))]
            axes[0,col].plot(x,smooth(norms),color=color,ls=style,lw=1)
            scales=np.minimum(1,r['config']['max_grad_norm']/(norms+1e-6))
            axes[1,col].plot(x,smooth(scales),color=color,ls=style,lw=1)
            axes[2,col].plot(np.array([v['total_steps'] for v in r['rows']])/1e6,
                              rolling(r['rows'],'value_loss_per_episode'),color=color,ls=style,lw=1)
        for row in range(3):axes[row,col].set_yscale('log');axes[row,col].grid(alpha=.2)
        axes[0,col].set_title(t.upper())
        axes[0,col].axhline(.75,color='k',ls=':',lw=.8)
        axes[2,col].set_xlabel('Recorded environment steps (millions)')
    for row,label in enumerate(['Pre-clip gradient norm','Applied gradient multiplier','Value-target MSE']):axes[row,0].set_ylabel(label)
    fig.suptitle('SoftRole optimization diagnostics',fontsize=14)
    fig.text(.5,.945,'Gradient panels: trailing 100-update medians. Value loss: 50-epoch episode-weighted means. Logarithmic axes.',ha='center')
    fig.legend(handles=legend[1:3]+legend[3:],loc='lower center',ncol=5)
    fig.tight_layout(rect=(0,.05,1,.925));pages.append(('optimization_diagnostics',fig))
    with PdfPages(OUT/'full_run_plots.pdf') as pdf:
        for name,fig in pages:
            fig.savefig(OUT/f'{name}.png');fig.savefig(OUT/f'{name}.pdf');pdf.savefig(fig);plt.close(fig)


def main():
    runs=load_runs(); summary=summaries(runs); figures(runs,summary)
    paths=[p for r in runs for p in Path(ROOT/r['path']).iterdir() if p.is_file() and p.name!='episodes.jsonl']
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    (OUT/'structured_input_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'runs':len(runs),'primary_epochs':sum(len(r['rows']) for r in runs if r['primary']),
                      'budgets':summary['budgets'],'groups':summary['groups']},indent=2))


if __name__=='__main__':main()
