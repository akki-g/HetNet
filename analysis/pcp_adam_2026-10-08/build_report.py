"""Build the PCP Adam analysis PDF and publication-quality standalone figures.

uv run --no-project --python .venv/bin/python --with numpy==1.26.4 \
  --with matplotlib==3.10.7 --with reportlab==4.4.4 --with pypdf==6.1.1 \
  python analysis/pcp_adam_2026-10-08/build_report.py
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUTPUT=ROOT/'output/pdf/pcp_adam_analysis_2026-10-08.pdf'
ORDER=['shared_adam','shared_rmsprop','banked_rmsprop','pcp_real','pcp_binary']
LABELS=['CapCom shared / Adam','CapCom shared / RMSprop','CapCom banked / RMSprop','HetNet paper-v1 Real','HetNet paper-v1 Binary']
COLORS=['#007F86','#71B6AF','#DA8F22','#5465B5','#9A4697']


def load(path):return json.loads((HERE/path).read_text())
def csvrows(path):
    rows=[]
    for r in csv.DictReader((HERE/path).open()):
        for k,v in r.items():
            try:r[k]=float(v)
            except (ValueError,TypeError):pass
        rows.append(r)
    return rows


def figures():
    data=csvrows('curve_bins.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','axes.titleweight':'bold'})
    fig,ax=plt.subplots(2,1,figsize=(10,7.5),sharex=True,gridspec_kw={'height_ratios':[2.2,1]})
    for group,label,color in zip(ORDER,LABELS,COLORS):
        subset=[r for r in data if r['group']==group]
        xx=[];ys=[[],[]];low=[[],[]];high=[[],[]]
        for budget in sorted({r['budget_m'] for r in subset}):
            es=[r for r in subset if r['budget_m']==budget]
            if len(es)!=3:continue
            xx.append(budget)
            for i,k in enumerate(('steps_taken','success_rate')):
                vals=[r[k] for r in es];ys[i].append(mean(vals));low[i].append(min(vals));high[i].append(max(vals))
        for i in range(2):
            ax[i].plot(xx,ys[i],color=color,lw=2,label=label)
            ax[i].fill_between(xx,low[i],high[i],color=color,alpha=.13,lw=0)
    ax[0].set_yscale('log');ax[0].set_ylim(4.7,85);ax[0].set_yticks([5,6,8,10,20,40,80],labels=['5','6','8','10','20','40','80'])
    ax[0].axhline(5.085059,color='#687783',ls=':',lw=1)
    ax[0].set_ylabel('Episode steps (log scale; lower is better)')
    ax[0].set_title('PCP training: performance and temporary instability',loc='left',pad=12)
    ax[0].annotate('Real seed 0 collapses and recovers',xy=(26,14),xytext=(17,43),fontsize=9,
                   arrowprops={'arrowstyle':'->','color':'#5465B5'},color='#394479')
    ax[0].legend(loc='upper right',fontsize=8.6,frameon=False)
    ax[1].set_ylim(.24,1.025);ax[1].set_ylabel('Success fraction');ax[1].set_xlabel('Environment joint steps (millions)')
    ax[1].set_xlim(0,41)
    for a in ax:a.grid(alpha=.16)
    fig.text(.09,.015,'1M-step windows; episode-weighted within seed, then equal seed means. Shading: observed seed range, not CI.\nTraining episodes. Curves require three seeds. Dotted line: omniscient population reference (5.085 steps).',fontsize=8.8,color='#405462')
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ('png','svg'):fig.savefig(HERE/f'training_curves.{ext}',dpi=200)
    plt.close(fig)
    # Each point represents one independently trained policy, at a common budget.
    rows=csvrows('comparison_windows.csv')
    fig,ax=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={'width_ratios':[1.25,1]})
    for i,(group,color) in enumerate(zip(ORDER,COLORS)):
        rs=[r for r in rows if r['group']==group and r['window']=='26-27M']
        for r in rs:
            y=i+(r['seed']-1)*.13
            ax[0].scatter(r['steps_taken'],y,s=38,color=color,marker=['o','s','^'][int(r['seed'])])
            rate=r['steps']/r['epoch_hours']/3600
            ax[1].scatter(rate,y,s=38,color=color,marker=['o','s','^'][int(r['seed'])])
    names=['Shared Adam','Shared RMSprop','Banked RMSprop','HetNet Real','HetNet Binary']
    ax[0].set_yticks(range(5),names);ax[1].set_yticks(range(5),['']*5)
    ax[0].set_xlabel('Episode steps');ax[1].set_xlabel('Joint environment steps / timed second')
    ax[0].set_title('Learning at 26-27M',loc='left');ax[1].set_title('Observed throughput at 26-27M',loc='left')
    ax[0].axvline(5.085059,color='#687783',ls=':',lw=1)
    for a in ax:a.invert_yaxis();a.grid(axis='x',alpha=.2)
    fig.text(.2,.015,'Circle / square / triangle: training seed 0 / 1 / 2. Hardware and historical logging differ.\nTiming covers recorded epoch loops, not queue, full job elapsed time or a controlled speed benchmark.',fontsize=8.5,color='#405462')
    fig.tight_layout(rect=(0,.1,1,1))
    for ext in ('png','svg'):fig.savefig(HERE/f'matched_performance_speed.{ext}',dpi=200)
    plt.close(fig)


FONT=Path('/System/Library/Fonts/Supplemental')
for n,f in [('R','Arial.ttf'),('RB','Arial Bold.ttf'),('RI','Arial Italic.ttf')]:
    pdfmetrics.registerFont(TTFont(n,str(FONT/f)))
pdfmetrics.registerFontFamily('R',normal='R',bold='RB',italic='RI')
INK=colors.HexColor('#193447');BLUE=colors.HexColor('#126C83');PALE=colors.HexColor('#EEF4F7')
ST={
 'title':ParagraphStyle('title',fontName='RB',fontSize=23,leading=26,textColor=INK,spaceAfter=9),
 'sub':ParagraphStyle('sub',fontName='R',fontSize=9,leading=12,textColor=colors.HexColor('#526675'),spaceAfter=12),
 'head':ParagraphStyle('head',fontName='RB',fontSize=12,leading=15,textColor=BLUE,spaceBefore=11,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='R',fontSize=9.7,leading=13.2,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='R',fontSize=8.4,leading=11.2,textColor=INK,spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='R',fontSize=8.8,leading=11.3,textColor=INK),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=7.5,leading=10,textColor=INK,spaceAfter=6),
}
def p(t,style='body'):return Paragraph(t,ST[style])
def table(rows,widths):
    t=Table([[p(str(x),'cell') for x in row] for row in rows],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BACKGROUND',(0,0),(-1,0),PALE),
                           ('LINEBELOW',(0,0),(-1,0),.7,BLUE),('LINEBELOW',(0,-1),(-1,-1),.4,BLUE),
                           ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                           ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return t
def footer(c,d):
    c.saveState();c.setFillColor(colors.HexColor('#61727C'));c.setFont('R',7.5)
    c.drawString(44,25,'PCP ADAM TRIAL REVIEW  |  8 OCTOBER 2026  |  Observed results, not a causal architecture ranking')
    c.drawRightString(568,25,str(d.page));c.restoreState()


def build():
    figures()
    groups=load('comparison.json')['groups'];trial=load('trial_review/review.json')
    frozen=load('frozen_summary.json');timing=load('speed_review/softrole_timing.json')
    def g(group,window):return next(r for r in groups if r['group']==group and r['window']==window)
    adam=[r for r in timing if '_adam_' in r['key']]
    seedtable=[['<b>Adam shared seed</b>','<b>Final steps</b>','<b>Last 100 epochs: steps</b>','<b>Mean-agent return</b>','<b>Timed hours</b>']]
    for seed,r in enumerate(adam):
        a=r['summary']['all'];t=r['summary']['last100']
        seedtable.append([seed,f"{r['finished']['total_steps']/1e6:.3f}M",f"{t['steps_taken']:.3f}",f"{t['mean_agent_return']:.4f}",f"{a['hours']:.2f}"])
    common=[['<b>Method (three seeds)</b>','<b>Mean steps</b>','<b>Seed range</b>','<b>Success</b>','<b>Return / agent</b>']]
    for name in ['pcp_binary','shared_rmsprop','shared_adam','pcp_real','banked_rmsprop']:
        r=g(name,'26-27M')
        common.append([LABELS[ORDER.index(name)],f"{r['steps_taken']:.3f}",f"{r['steps_taken_min']:.3f}-{r['steps_taken_max']:.3f}",f"{100*r['success_rate']:.4f}%",f"{r['mean_agent_return']:.4f}"])
    story=[p('PCP Adam trials: what changed,<br/>what improved, and what did not','title'),
      p('Analysis of the copied 911488 Adam runs, 908316 HetNet paper-version logs, older CapCom runs, saved checkpoints and prior frozen evaluations. Runtime source code and input artifacts were not changed.','sub'),
      p('<b>The new shared Adam runs finish strongly and run substantially faster.</b> Their last-100-epoch mean is <b>5.401 steps</b>, with 3 failures in 1,116,258 episodes across three seeds. The latest HetNet Real and Binary tails are about <b>5.12 steps</b>. This is an observed nominal training-performance gap, but its size depends on budget and seed stability. [A,B]'),
      p('<b>These are shared-model trials, not new banked CapCom results.</b> All three runs use one effective transform, 171,087 parameters, fixed 2P1A, no sensor failures, and fresh Adam at 1e-4. The banked comparison below remains the older RMSprop experiment. [A]'),
      table(seedtable,[103,78,122,119,102]),
      p('Last 100 means: episode-weight within each seed. Timed hours sum epoch-loop measurements and exclude checkpoint writes, some setup/shutdown, and queue time. All three finish 2,000 epochs / 20,000 updates, slightly over 40M steps. [A,B]','small'),
      p('A fair common-budget comparison','head'),
      p('The <b>26-27M</b> window is completely available for every one of the 15 policies below. Mean steps include failures at the 80-step horizon; larger mean-agent return is better. We average seeds equally. Seed ranges are descriptive, not confidence intervals. [A,B]','small'),
      table(common,[181,69,97,86,91]),
      p('<b>HetNet Binary leads this window. HetNet Real is not uniformly better:</b> seed 0 temporarily regressed, lifting its mean above shared CapCom. At 29-30M, after recovery, Real averages 5.114 steps versus shared Adam 5.820 and shared RMSprop 5.860. Binary had not reached 30M in the copied logs. [B]'),
      p('<b>Do not change the environment to make scores resemble the paper.</b> The audited PCP physical transitions and rewards are unchanged. The important historical difference is an observation correction, alongside nonidentical training and reporting protocols; page 3 explains it. [C]'),
      PageBreak(),p('Learning curves and the optimizer question','title'),
      p('Sustained sample-budget windows reveal learning speed, seed instability, and saturation.','sub'),
      Image(str(HERE/'training_curves.png'),width=524,height=393),
      p('<b>Adam has no demonstrated general sample-efficiency advantage over RMSprop.</b> At 4-8M, shared Adam takes 25.550 steps versus RMSprop 22.351; success is 96.57% versus 97.80%. At 26-27M they are effectively close (6.119 versus 6.095); at 29-30M Adam is slightly ahead (5.820 versus 5.860). These small late differences do not establish an optimizer ranking. Initial model signatures match for seeds 0/1; the old seed-2 initial signature is missing. [A]'),
      p('<b>HetNet Real seed 0 has a material transient failure.</b> Its 24-25M and 25-26M windows deteriorate to 17.05 / 34.59 steps and 84.09% / 62.48% success. Several epochs contain no successful episode. It recovers to 5.18 steps at 27-28M and 5.13 at 28-29M. A good final mean does not erase this instability, and the local scientific-run checkpoint/update archives are absent, so its cause is unresolved. [B]'),
      p('<b>Avoid comparing unmatched tails to claim Adam wins.</b> Older shared RMSprop seed 0 stopped at 33.670M; seeds 1/2 and all new Adam runs reached about 40.6M. An equal-seed average of their final windows mixes different training budgets. The shared 39-40M RMSprop result therefore has only two seeds and is secondary. [A,B]'),
      p('All curves describe changing stochastic policies during training. No best epoch was selected; no new checkpoint was evaluated in this analysis. Epoch counts alone are not matched sample budgets. The old release reproduction is a separate source/model/learner stratum and must not replace the newer paper-v1 baseline.','small'),
      PageBreak(),p('Why can our numbers beat the paper?','title'),
      p('The comparison requires separating physical dynamics, observations, reporting, and experimental protocol.','sub'),
      p('Physical PCP dynamics were not made easier for Adam','head'),
      p('The new and old CapCom raw PCP simulator files are <b>byte-identical</b>. The reset, movement/capture, reward, termination and terminal-reward methods also match the public upstream and current paper-v1 runtime after parsing away formatting. Native configuration remains 5x5, 2P1A, stationary target, sensing radius 2, unlimited communication and horizon 80. Reached agents remain at the target; A needs a capture action. There is no newly shortened horizon, moving-target change, or reward rescaling inside the simulator. [C]'),
      p('The observation correction is substantial','head'),
      p('In the historical release, observations can share array storage: masking the blind A agent\'s sensory view with -1 can also erase target information from overlapping P views. CapCom and the paper-v1 path use independent observation copies. CapCom additionally merges typed counts and supplies each agent\'s own capabilities. Thus physical dynamics match, but the observation model is not identical to the buggy legacy release. [C]'),
      p('An independent enumeration of all <b>303,600 distinct native layouts</b> finds that at least one P initially sees the target in <b>78.72%</b> with correct copies versus <b>33.36%</b> with the historical aliasing path. This demonstrates a large information difference; it does <b>not</b> measure the learned-performance effect or prove that the publication used that buggy checkout. The exact historical experiment source remains unknown. [C]'),
      table([['<b>Evidence</b>','<b>What it actually measures</b>'],
             ['Paper Table 1: PCP 9.98 +/- 0.36 steps; return -0.364 +/- 0.017','50 trials of a frozen final policy; reported uncertainty is standard error.'],
             ['Paper Figure 3','Training curves over three seeds, plotted against epochs; exact samples per paper epoch are not recovered.'],
             ['New Adam logs: 5.401 steps','Last 100 training epochs, three seeds, about 40.6M samples; not a new frozen evaluation.'],
             ['New paper-v1 HetNet: about 5.12 steps','Training tails of a declared reconstruction, not the recovered publication checkpoint.']],[228,296]),
      Spacer(1,7),
      p('<b>Use the same return units.</b> The public evaluation script averages over episodes <i>and agents</i>. CapCom\'s team return is a sum, so for native N=3 report R<sub>agent</sub>=R<sub>team</sub>/3. Adam\'s last 100 team return -0.4496 corresponds to -0.1499 per agent. This arithmetic resolves a scale mismatch, but does not make different evaluation protocols interchangeable. [C,D]'),
      p('<b>The task has little nominal headroom left.</b> With omniscient shortest paths, T*=max(d<sub>P0</sub>,d<sub>P1</sub>,d<sub>A</sub>+1). Over the uniform distinct-start population, E[T*]=<b>5.0851</b> and E[R*<sub>team</sub>]=-0.4000. These are physical population references, not decentralized-policy guarantees or exact bounds for a finite sampled panel. Scores near 5.12 are plausible under these dynamics; the paper\'s 9.98 is not a lower bound. [C,D]'),
      PageBreak(),p('Training is faster; the cause is not isolated','title'),
      p('The logs support a large observed throughput improvement, not a pure Adam speedup claim.','sub'),
      Image(str(HERE/'matched_performance_speed.png'),width=524,height=220),
      p('<b>The under-24-hour observation is consistent with recorded compute time.</b> The three Adam runs log <b>15.31-15.37 hours</b> of epoch loops and 733-737 joint steps/second over the full run. Completed old shared RMSprop seeds 1/2 log 24.67/24.87 hours and 453-457 steps/second: about <b>1.62x</b> higher throughput now. Old shared seed 0 is a much slower outlier at 195 steps/second and a 48-hour timeout; using it alone would exaggerate the representative gain. [B]'),
      p('What concretely changed','head'),
      p('<b>Episode logging:</b> old code opens/closes the JSONL file for each episode; new code opens it once per update. For the new runs that is <b>20,000 file opens</b> instead of 4.92-5.21 million (246-260x fewer). The same episode records are retained. This can matter on Lustre, especially once episodes shorten to 5-6 steps, but it is not a measured 246x runtime gain. [B]'),
      p('<b>Launch configuration:</b> the new script adds explicit CPU binding and OPENBLAS/NUMEXPR thread limits. Four collectors, one Torch thread per collector and CPU float64 remain. Both old and new already used persistent worker pools. A CUDA-enabled Torch package name does not mean these jobs ran on a GPU. Node model/load, contention and filesystem timing are not recorded for the new Adam runs. [B]'),
      p('<b>Optimizer and learning trajectory:</b> Adam replaces RMSprop, while learning rate stays 1e-4 and actor/value coefficients remain 50/1. Episode lengths change work per sample through reset, recurrence and logging overhead. There is no same-host factorial benchmark that separates optimizer, logging, affinity and hardware contributions. Do not ascribe the full 1.62x gain to any one of them. [A,B]'),
      p('HetNet has its own cost and incomplete runs','head'),
      p('The paper-v1 Real seeds 1/2 reached 40M at 43.80/43.85 recorded active hours; Real seed 0 paused at 46.00h and 36.767M. All three Binary jobs paused at 46.00h and 27.295-27.585M. These stdout receipts describe the copied segment, not present cluster state or later continuations. The paper model has three rounds, independent head channels and a different learner; its speed is not a controlled architectural ablation against CapCom. [B]'),
      p('Recorded Adam epoch durations omit checkpoint writes and some setup/shutdown. No scheduler start/end record was supplied, so exact total job turnaround, queue duration and hardware-normalized speed remain unverified.','small'),
      PageBreak(),p('What the frozen evidence does and does not say','title'),
      p('The supplied frozen reports belong to the older RMSprop policies selected near 30M, not the new Adam checkpoints.','sub'),
      p('The prior native and transfer results remain relevant','head'),
      table([['<b>Team</b>','<b>Shared steps</b>','<b>Banked steps</b>','<b>Shared success</b>','<b>Banked success</b>']]+
            [[f'{comp[0]}P{comp[1]}A']+[f"{next(r for r in frozen['nominal_groups'] if r['model']==model and (r['num_p'],r['num_a'])==comp)['metrics'][key]['mean']*(100 if key=='success_rate' else 1):.2f}"+('%' if key=='success_rate' else '') for key,model in [('steps_taken','shared'),('steps_taken','banked'),('success_rate','shared'),('success_rate','banked')]]
             for comp in [(2,1),(1,2),(2,2),(3,1),(3,2)]],[68,114,114,114,114]),
      p('Recomputed from saved raw outcomes: three training seeds, 500 scenarios per team per seed, identical assigned panels. Steps include horizon-capped failures. These are the previous 30M selection, not 40M endpoints; checkpoint overshoot is retained in the manifests. [D]','small'),
      p('<b>Reliable banked transfer across these teams is not established.</b> Banked seed 1 remains unstable specifically on two-A teams; its success is 49.8%, 61.4% and 64.4% on 1P2A, 2P2A and 3P2A. This negative result must remain in the comparison. New shared Adam nominal training cannot repair or explain it. No new banked Adam run is present. [D]'),
      p('<b>Sensor adaptation remains untested by this nominal batch.</b> All 15.2M new training episodes have no scheduled sensor loss. In the older frozen failure/sham panel, only 1 of 600 policy-scenario assignments actually reaches its scheduled loss. Near-zero exposure cannot establish robustness or meaningful adaptation. [A,D]'),
      p('A defensible next comparison','head'),
      p('<b>1. Recover and finish the declared baseline.</b> Copy the 908316 scientific run directories, archived source, checkpoints, update/optimizer diagnostics and scheduler receipts. Retain the Real seed 0 collapse and paused Binary suffixes. Resume only valid retained lineages to the declared budget; do not replace difficult seeds. This review launches nothing.'),
      p('<b>2. Freeze checkpoint selection before evaluating Adam.</b> Use the previously declared study budget rule consistently, with actual counts and hashes; keep 30M and 40M comparisons separate. Evaluate all seeds on the same fresh nominal panel, stochastic action/message settings and horizon. Include failures and report mean-agent return, success and capped steps. Avoid selecting a checkpoint using evaluation outcomes.'),
      p('<b>3. Ask a targeted scientific question.</b> The matched evidence does not show Adam solving sample efficiency. It does show faster execution and strong shared performance. If the claim concerns capability-conditioned banks, it needs a matched banked comparison and exposed perturbation/transfer tests. Any harder environment or earlier sensor-loss condition is a new, preregistered experiment, not a retroactive adjustment to make this benchmark favor a model.'),
      p('HetNet and CapCom also differ in actor information, rounds, payload and objective: paper-v1 is typed and uses individual policy credit and Adam 1e-3; the new shared CapCom trials use capability inputs, team credit, actor coefficient 50 and Adam 1e-4. Binary paper-v1 is explicitly 64 bits per head (256 per sender/round), whereas CapCom shares 16 bits across heads (32 across two rounds). A score difference does not isolate any one of these choices. [A,C]','small'),
      PageBreak(),p('Audit trail and interpretation limits','title'),
      p('Evidence checks distinguish valid local artifacts from what has only been reported in stdout.','sub'),
      p('New Adam artifacts were checked beyond their final log lines','head'),
      p('<b>Complete reconciliation:</b> 15,224,881 raw episode records; 121,822,429 joint steps; 60,000 updates; 6,000 epochs. Episode counts, rewards, success, capabilities, horizons and attention bounds reconcile to update/epoch ledgers. All 120 saved checkpoints were loaded: finite model/Adam tensors, moment shapes, nonnegative second moments, optimizer steps, configuration/source IDs and exact tensor signatures match their sidecars. All 918 archived source files match their manifests (306 per seed). [A]'),
      p('<b>One stdout scalar is invalid and disagrees with the run artifacts.</b> Seed 2 epoch 2000 has alpha_null=-1.3112684324133872 in stdout but +0.3112684324133873 in the run metrics. The negative value is impossible for attention mass. Reaggregated episode records agree with the positive metric; every other epoch field matches. This report uses the structured run artifacts, preserves both inputs, and does not infer why the stdout differs. Performance metrics are unaffected. [A,B]'),
      p('<b>Optimization diagnostics:</b> all 60,000 pre-clip norms exceed 0.75 (roughly 14.7-1,687.5); clipping is active throughout. This is consistent with the existing summed episode objective/actor coefficient 50. It does not by itself prove instability or quantify Adam parameter-step size. Shared gate entropy is identically zero because it has one effective transform; it is not learned expert specialization. [A]'),
      p('<b>Provenance limits:</b> the new archived source digest is 4c5dfbd6...9e876e, with a recorded dirty Git checkout. Archived bytes, not commit name alone, identify the run. Latest 908316 PCP evidence is stdout with per-episode batches and status receipts; scientific-run weight bytes are not available locally. Older frozen selection now has five of six matching checkpoint files; shared seed 2 remains absent. No fresh model replay was performed. [A-D]'),
      p('Sources and reproducibility','head'),
      p('<b>[A] New and historical CapCom:</b> <font name="Courier">logs_sr/softrole-adam-pcp-911488_{0,1,2}.out</font>; <font name="Courier">stokes_runs/softrole_adam_pcp_911488/pcp_shared/seed*/</font>; older 898818 PCP logs and <font name="Courier">stokes_runs/softrole_primary/</font>. Audit outputs: <font name="Courier">trial_review/</font> (full episodes, 120 checkpoints, source hashes, windows, thresholds).','small'),
      p('<b>[B] Timing and latest HetNet:</b> <font name="Courier">logs_1/hetnet-paper-908316_{3,4,5,9,10,11}.out</font>. <font name="Courier">speed_review/</font> contains streamed hashes, epoch tables, resource/status receipts, slowdown/regression evidence and source comparisons. Whole-episode records are checked for selected Real0 windows and the main comparative windows; all 60,000 new Adam updates receive full reconciliation.','small'),
      p('<b>[C] Published reference and environment:</b> <link href="https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf" color="#126C83">Seraj et al., AAMAS 2022, Table 1/Figure 3</link>; <link href="https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/AAMAS_22___HetNet_Supplementary.pdf" color="#126C83">authors\' supplementary protocol</link>; <link href="https://github.com/CORE-Robotics-Lab/HetNet/blob/main/print_plot_eval.py" color="#126C83">public reward reporting</link>. <font name="Courier">paper_review/</font> preserves references, code hashes, structural comparisons and layout enumeration. Current reconstruction choices are in <font name="Courier">publication_reconstruction/FIDELITY.md</font>.','small'),
      p('<b>[D] Prior frozen evaluation:</b> <font name="Courier">stokes_runs/frozen_pcp_30m_20261002_slurm/</font>; freshly recomputed 18 reports / 16,200 outcomes, scenario identities, return arithmetic, event exposure and available selected-checkpoint byte hashes in <font name="Courier">frozen_summary.json</font>. This is an analysis of recorded evaluations, not a new evaluation.','small'),
      p('From the repository root:','small'),
      p('python3 analysis/pcp_adam_2026-10-08/frozen_audit.py<br/>python3 analysis/pcp_adam_2026-10-08/compare.py<br/># Full independent audits / PDF build: see README.txt','code'),
      p('The accompanying CSVs retain exact included sample boundaries and denominators. Figures show observed seed ranges, never episode-count-based claims about between-seed certainty. No training, evaluation, runtime edit, cluster action, dependency-file change, commit or push was performed.','small')]
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=(612,792),leftMargin=44,rightMargin=44,topMargin=37,bottomMargin=47,
                         title='PCP Adam trials: results, dynamics and throughput',author='HetNet research analysis')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    pdf=PdfReader(OUTPUT)
    text='\n\n'.join(page.extract_text() for page in pdf.pages)
    (HERE/'report_text.txt').write_text(text)
    record=dict(pdf=str(OUTPUT.relative_to(ROOT)),pages=len(pdf.pages),sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    (HERE/'pdf_build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':build()
