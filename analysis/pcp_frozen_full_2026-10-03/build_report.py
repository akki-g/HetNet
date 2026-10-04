"""Build a two-page full-record update; preserve the earlier stdout-only PDF."""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / "output/pdf/pcp_frozen_full_results_2026-10-03.pdf"
FONT = Path("/System/Library/Fonts/Supplemental")
for name, filename in (("Report", "Arial.ttf"), ("ReportBold", "Arial Bold.ttf"), ("ReportItalic", "Arial Italic.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold", italic="ReportItalic")
INK, BLUE, PALE, AMBER = [colors.HexColor(x) for x in ("#142F44", "#17698A", "#EFF5F8", "#FFF0D9")]
STYLES = {
    "title": ParagraphStyle("title", fontName="ReportBold", fontSize=20, leading=23, textColor=INK, spaceAfter=7),
    "sub": ParagraphStyle("sub", fontName="Report", fontSize=8.3, leading=10.7, textColor=colors.HexColor("#526472"), spaceAfter=8),
    "heading": ParagraphStyle("heading", fontName="ReportBold", fontSize=10.8, leading=13.5, textColor=BLUE, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Report", fontSize=9.1, leading=12.0, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Report", fontSize=7.9, leading=10.0, textColor=INK, spaceAfter=4),
    "table": ParagraphStyle("table", fontName="Report", fontSize=8.7, leading=10.8, textColor=INK),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.8, leading=10.3, textColor=INK, spaceAfter=5),
}


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def table(rows, widths, highlight=None):
    obj = Table([[p(str(v), "table") for v in row] for row in rows], colWidths=widths, repeatRows=1)
    rules = [("VALIGN", (0,0),(-1,-1),"MIDDLE"), ("BACKGROUND",(0,0),(-1,0),PALE),
             ("LINEBELOW",(0,0),(-1,0),.7,BLUE), ("LINEBELOW",(0,-1),(-1,-1),.5,BLUE),
             ("TOPPADDING",(0,0),(-1,-1),4.5), ("BOTTOMPADDING",(0,0),(-1,-1),4.5),
             ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6)]
    if highlight is not None:
        rules.append(("BACKGROUND",(0,highlight),(-1,highlight),AMBER))
    obj.setStyle(TableStyle(rules))
    return obj


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D1DDE4")); canvas.line(44,37,568,37)
    canvas.setFont("Report",7.3); canvas.setFillColor(colors.HexColor("#526472"))
    canvas.drawString(44,25,"FULL-RECORD UPDATE  |  3 OCT 2026  |  Descriptive results from three training seeds")
    canvas.drawRightString(568,25,f"{doc.page} / 2")
    canvas.restoreState()


def build():
    d=json.loads((HERE/"summary.json").read_text())
    comps=[(2,1),(1,2),(2,2),(3,1),(3,2)]
    rows=[["<b>Policy / seed</b>"]+[f"<b>{a}P{b}A</b>" for a,b in comps]]
    for model in ("shared","banked"):
        for seed in range(3):
            vals=[next(r for r in d["composition_seed_rows"] if
                       (r["model"],r["seed"],r["num_p"],r["num_a"])==(model,seed,*c)) for c in comps]
            rows.append([f"{model.title()} / {seed}"]+[f"{r['success_rate']*100:.1f}%" for r in vals])
    means={(r["model"],tuple(r["composition"])):r["metrics"] for r in d["composition_means"]}
    perf=[["<b>Team</b>","<b>Shared steps</b>","<b>Banked steps</b>","<b>Shared return</b>","<b>Banked return</b>"]]
    for c in comps:
        s,b=means['shared',c],means['banked',c]
        perf.append([f"{c[0]}P{c[1]}A",f"{s['mean_capped_steps']['equal_seed_mean']:.2f}",
                     f"{b['mean_capped_steps']['equal_seed_mean']:.2f}",
                     f"{s['mean_agent_return']['equal_seed_mean']:.4f}",f"{b['mean_agent_return']['equal_seed_mean']:.4f}"])
    story=[p("PCP frozen evaluation<br/>Full-record update","title"),
        p("18 synced reports | 16,200 episode evaluations | 30M checkpoint-selection threshold","sub"),
        p("<b>Main finding:</b> both models perform strongly on native 2P1A. Banked seed 1 loses reliability specifically on the tested two-capture-agent teams. The full JSONs confirm the stdout averages and resolve the composition breakdown; they do not establish a mechanism or failure robustness. [1]"),
        p("1. Success by composition and training seed","heading"),
        p("Each cell contains <b>500 scenarios</b>, shared across policies; only 2P1A was used in these policies' nominal training. Frozen weights were evaluated stochastically. Other columns test changed composition and/or team size. [1,2]","small"),
        table(rows,[124,80,80,80,80,80],highlight=5),
        Spacer(1,6),
        p("<b>Positive:</b> native success averages 99.8% for shared and 100% for banked. Shared seeds 1 and 2 succeed on all five panels. Banked seeds 0 and 2 retain high transfer success; banked seed 0 matches or exceeds shared seed 0 in every column."),
        p("<b>Negative:</b> all 662 banked failures occur with two A agents. Seed 1 contributes <b>622 failures</b>: 1P2A (251), 2P2A (193), 3P2A (178); it has none in 2P1A or 3P1A. Across the four transfer teams, success averages <b>99.70% shared versus 88.97% banked</b> (all five teams: 99.72% / 91.17%). Retain all seeds; these held-out outcomes should not select a revised bank count."),
        p("2. Completion time and per-agent return","heading"),
        p("Equal-weight means over three training seeds; each seed has 500 trials per team. Steps include failures capped at H=80. Return is mean<sub>episodes</sub>(R<sub>team</sub>/N); larger (less negative) is better. Team returns and all seed values are in the accompanying CSV. [1,3]","small"),
        table(perf,[76,112,112,112,112]),
        Spacer(1,6),
        p("Shared has fewer mean steps and better mean-agent return in every composition, while banked has slightly higher native success. These are different objectives: a few failures can coexist with faster typical completion. The native returns (-0.1687 / -0.1813) are <b>not yet evidence of superiority to published HetNet</b>; a matched HetNet evaluation is still required."),
        p("<b>Reward-based diagnosis:</b> this PCP path gives -0.05 per unfinished agent-step, then zero after P reaches or A captures. Thus R<sub>i</sub> = -0.05 U<sub>i</sub>, where U<sub>i</sub> is its penalized step count. In all 622 banked seed 1 failures, at least one A has R<sub>i</sub>=-4 and never completes capture; in 618, both A agents do. All P agents reach in 620/622. This localizes the unresolved objective to the A agents, but cannot distinguish failure to navigate from failure to execute capture. Nominal reports lack trajectories/positions. [3]"),
        PageBreak(),
        p("Failure evidence and next actions","title"),
        p("What the full records verify, and what remains unresolved","sub"),
        p("3. Failure/sham pairing is valid; exposure is inadequate","heading"),
        p("The 100 native scenarios are reused for six policies. Only <b>1/600 policy-scenario assignments</b> reaches its scheduled sensor loss; 599 finish before their own event. All 600 pairs have exact recorded pre-event agreement, and the 599 unexposed pairs have identical entire traces. Each seed's paired mean failure-minus-sham difference is zero for success, steps and return. [1,4]"),
        p("The sole exposed case is shared seed 0, scenario <b>2106772258:37</b>, victim 1 at zero-based step 29. The victim had not reached, was not seeing the target, and had never directly seen it. Failure and sham both end unsuccessfully at step 80 with team return -12. After loss, sampled actions differ on 44/51 steps and null-attention values on all 51. <b>Equal outcomes do not mean unchanged behavior.</b> [4]"),
        p("With matched pre-event streams: mean(failure - sham) = (n<sub>exposed</sub>/M) x mean(exposed paired differences), for nonzero exposure. Shared seed 0 has n/M=1/100; all other policies have zero exposure. This panel tests almost no sustained loss, so it supports neither robustness nor a banked adaptation claim. Exposed subsets also differ by policy."),
        p("4. Provenance and the HetNet execution bug","heading"),
        p("<b>Verified:</b> 122 archived evaluator files hash-match; both scenario panels regenerate exactly; all 18 job mappings, 16,200 episode assignments and stdout summaries reconcile. Reports agree with the preparation manifest on checkpoint identity, counts and configuration. The common evaluator digest is <font name='Courier'>a666e191...d944f</font>. [1,2]"),
        p("<b>Still missing:</b> selected epoch-1500 checkpoint files and up-to-date training ledgers. We cannot independently rehash/replay those weights or reprove the first-saved-at-or-above-30M rule. Recorded progress is 15,000 updates and 30.508-30.729M steps. Report/source consistency passed; this is not a fresh policy replay. [2]"),
        p("<b>HetNet bug, unchanged by this sync:</b> all four new preflights stop before training at source inventory validation. The reproduced Git package omits the ignored <font name='Courier'>runtime/envs/LICENSE.md</font> (37 files versus 38 required). The prior fix makes it distributable and adds precise diagnostics plus a read-only Slurm audit. The exact Stokes inventory still needs confirmation. No new training failure or throughput measurement is established. [5]"),
        p("5. Immediate priorities","heading"),
        p("<b>Write now:</b> native PCP performance and frozen composition transfer, with all seeds and explicit limitations. <b>Run next:</b> sync the fix including the license; submit <font name='Courier'>slurm/publication_source_audit.sbatch</font>, then <font name='Courier'>slurm/publication_preflight.sbatch</font> after a passing audit. <b>Investigate:</b> obtain the selected weights and inspect A navigation versus capture on a separate diagnostic/development panel. Preserve this held-out panel; any model revision needs fresh confirmatory scenarios."),
        p("<b>Failure study:</b> preserve the current reference result; prerecord earlier loss on native/training teams, with fresh scenarios, matched shams and corresponding HetNet conditions. Select timing to achieve event exposure, not to favor a model. Three training seeds constrain inference: scenario count does not replace independent training runs. [6]"),
        p("Evidence and reproducibility","heading"),
        p("<b>[1]</b> Input root: <font name='Courier'>stokes_runs/frozen_pcp_30m/frozen_pcp_30m_20261002_slurm/</font>; <font name='Courier'>results/*_{nominal,failure,sham}.json</font>, <font name='Courier'>manifest.json</font> and scenario files. Derived tables, 147 input hashes and independent audits: <font name='Courier'>analysis/pcp_frozen_full_2026-10-03/</font>.","small"),
        p("<b>[2]</b> <font name='Courier'>provenance_audit.json</font>; archived <font name='Courier'>source/softrole/evaluate.py:53-115</font>. <b>[3]</b> <font name='Courier'>independent_metrics.json</font>; archived <font name='Courier'>envs/ic3net_envs/predator_capture_env.py:487-530</font> and <font name='Courier'>softrole/rollout.py:169-201</font>. <b>[4]</b> <font name='Courier'>failure_audit.json</font>; archived <font name='Courier'>softrole/rollout.py:111-168</font>.","small"),
        p("<b>[5]</b> Earlier <font name='Courier'>analysis/pcp_frozen_2026-10-03/preflight_failure_audit.json</font> and <font name='Courier'>logs_1/hetnet-preflight-902621_{0..3}.err</font>. Prior fix validation: 447 tests passed; no learner changes in this analysis. <b>[6]</b> <link href='https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html' color='#17698A'>Agarwal et al., NeurIPS 2021</link>; <link href='https://arxiv.org/abs/1709.06560' color='#17698A'>Henderson et al., AAAI 2018</link> (few-run uncertainty and reproducible RL evaluation).","small")]
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=(612,792),leftMargin=44,rightMargin=44,
                         topMargin=37,bottomMargin=47,title="PCP frozen evaluation: full-record update",
                         author="HetNet research analysis",pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(OUTPUT)
    assert len(reader.pages)==2, f"Expected 2 pages, got {len(reader.pages)}"
    text="\n".join(page.extract_text() for page in reader.pages)
    for expected in ("49.8%","61.4%","64.4%","622","618","1/600","122"):
        assert expected in text,expected
    (HERE/"report_text.txt").write_text(text)
    record={"pdf":str(OUTPUT.relative_to(ROOT)),"pages":len(reader.pages),
            "sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            "builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python":platform.python_version(),"packages":{n:version(n) for n in ("reportlab","pypdf")}}
    (HERE/"pdf_build.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    build()
