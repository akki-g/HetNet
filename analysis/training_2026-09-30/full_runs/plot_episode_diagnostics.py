"""Plot audited episode sufficient statistics without re-reading large input logs."""
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

OUT=Path(__file__).resolve().parent
TASKS=('pp','pcp','fc')
COLORS={'shared':'#0072B2','banked':'#D55E00'}
STYLES=('-', '--', ':')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
findings=json.loads((OUT/'softrole_audit/findings.json').read_text())
groups=defaultdict(list)
for line in (OUT/'softrole_audit/episode_epoch_sufficient_stats.jsonl').open():
    row=json.loads(line);groups[row['name']].append(row)
summary=json.loads((OUT/'summary.json').read_text())


def entropy(p):
    p=np.array(p);return float(-np.sum(p[p>0]*np.log(p[p>0])))


def aggregate(rows):
    n=sum(r['episodes'] for r in rows)
    gate=sum((np.array(r['expert_load']) for r in rows))/n
    h=sum(r['gate_entropy'] for r in rows)/n
    hist=defaultdict(int)
    for r in rows:
        for t,count in r['steps_hist'].items():hist[int(t)]+=count
    return {'episodes':n,'gate':gate.tolist(),'H':h,'pooled_H':entropy(gate),
            'gap':max(0,entropy(gate)-h),'null':sum(r['alpha_null'] for r in rows)/n,
            'hist':dict(sorted(hist.items())),
            'nominal_exposure_10_30':sum(min(21,max(0,t-10))*c for t,c in hist.items())/(21*n)}


pages=[];derived={}
fig,axes=plt.subplots(3,3,figsize=(12,8.5),sharex='col')
for item in findings:
    name=item['name'];config=item['config'];task,mode,seed=config['task'],config['model'],config['seed']
    rows=[r for r in groups[name] if r['epoch']<=item['matched_end_epoch']]
    sr=next(r for r in summary['runs'] if r['path'].endswith(name) and r['suite']=='softrole_primary')
    sums=np.cumsum([r['steps'] for r in rows])/1e6
    stats=[aggregate(rows[max(0,i-49):i+1]) for i in range(len(rows))]
    last=aggregate(rows[-50:]);derived[name]=last
    col=TASKS.index(task)
    if mode=='banked':
        axes[0,col].plot(sums,[v['H'] for v in stats],color=COLORS[mode],ls=STYLES[seed])
        axes[1,col].plot(sums,[max(1e-7,v['gap']) for v in stats],color=COLORS[mode],ls=STYLES[seed])
    axes[2,col].plot(sums,[v['null'] for v in stats],color=COLORS[mode],ls=STYLES[seed])
for col,task in enumerate(TASKS):
    axes[0,col].set_title(task.upper());axes[0,col].axhline(np.log(4),color='k',ls=':',lw=.8)
    axes[0,col].set_ylim(0,1.44);axes[1,col].set_yscale('log');axes[1,col].set_ylim(1e-5,1)
    axes[2,col].set_ylim(0,1);axes[2,col].set_xlabel('Recorded environment steps (millions)')
    for row in range(3):axes[row,col].grid(alpha=.2)
for row,label in enumerate(['Banked mean gate entropy (nats)','Banked gate variability J (nats)','Mean null attention mass']):axes[row,0].set_ylabel(label)
legend=[Line2D([],[],color=COLORS[m],label=m) for m in COLORS]+[Line2D([],[],color='k',ls=STYLES[s],label=f'Seed {s}') for s in range(3)]
fig.legend(handles=legend,loc='lower center',ncol=5)
fig.suptitle('Communication diagnostics from audited episode records',fontsize=14)
fig.text(.5,.945,'50-epoch episode-weighted windows. J = H(pooled gate) − mean H(gate); includes policy drift within windows.',ha='center')
fig.tight_layout(rect=(0,.045,1,.92));pages.append(('gate_diagnostics',fig))

fig,axes=plt.subplots(1,3,figsize=(12,4.6))
for col,task in enumerate(TASKS):
    for item in findings:
        config=item['config']
        if config['task']!=task:continue
        stat=derived[item['name']];hist=stat['hist'];h=config['max_steps']
        survival=[sum(count for length,count in hist.items() if length>t)/stat['episodes'] for t in range(h+1)]
        axes[col].step(range(h+1),survival,where='post',color=COLORS[config['model']],ls=STYLES[config['seed']],lw=1.2)
    if task=='pcp':axes[col].axvspan(10,30,color='#56B4E9',alpha=.15)
    axes[col].set_title(task.upper());axes[col].set_xlabel('Episode step t');axes[col].set_ylim(0,1.02);axes[col].grid(alpha=.2)
axes[0].set_xlim(0,15);axes[0].set_ylabel('Fraction of episodes with length T > t')
fig.suptitle('Latest 50 epochs: episode survival curves',fontsize=14)
fig.text(.5,.905,'Run endpoints differ: distributions describe current trajectories, not matched-budget performance. PP shown through step 15.',ha='center')
fig.legend(handles=legend,loc='lower center',ncol=5)
fig.tight_layout(rect=(0,.1,1,.87));pages.append(('episode_length_survival',fig))

fig,axes=plt.subplots(1,3,figsize=(12,4.4))
for col,task in enumerate(TASKS):
    names=[f'{task}_banked/seed{s}' for s in range(3)]
    data=np.array([derived[name]['gate'] for name in names])
    im=axes[col].imshow(data,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(3):
        for j in range(4):axes[col].text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',color='white' if data[i,j]>.55 else 'black')
    axes[col].set_title(task.upper());axes[col].set_xticks(range(4),[1,2,3,4]);axes[col].set_yticks(range(3),['Seed 0','Seed 1','Seed 2']);axes[col].set_xlabel('Expert index within this run')
fig.suptitle('Banked pooled expert weights in each run’s latest 50 epochs',fontsize=14)
fig.text(.5,.08,'Expert indices have no shared semantic identity across independently trained runs; rows are not pooled across seeds.',ha='center')
fig.tight_layout(rect=(0,.13,1,.92));pages.append(('expert_weights',fig))
with PdfPages(OUT/'episode_diagnostic_plots.pdf') as pdf:
    for name,fig in pages:
        fig.savefig(OUT/f'{name}.png');fig.savefig(OUT/f'{name}.pdf');pdf.savefig(fig);plt.close(fig)
(OUT/'episode_derived.json').write_text(json.dumps(derived,indent=2)+'\n')
for name,item in derived.items():
    print(name,'gate_gap',item['gap'],'nominal_exposure',item['nominal_exposure_10_30'])
