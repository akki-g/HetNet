"""Build the requested two-page stdout-only research note with embedded fonts."""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / "output/pdf/pcp_frozen_and_hetnet_preflight_2026-10-03.pdf"
FONT = Path("/System/Library/Fonts/Supplemental")
for name, filename in (("Report", "Arial.ttf"), ("ReportBold", "Arial Bold.ttf"), ("ReportItalic", "Arial Italic.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold", italic="ReportItalic")
INK, BLUE, PALE, AMBER = colors.HexColor("#142F44"), colors.HexColor("#17698A"), colors.HexColor("#EFF5F8"), colors.HexColor("#FFF0D9")
STYLES = {
    "title": ParagraphStyle("title", fontName="ReportBold", fontSize=20, leading=23, textColor=INK, spaceAfter=7),
    "sub": ParagraphStyle("sub", fontName="Report", fontSize=8.4, leading=11, textColor=colors.HexColor("#526472"), spaceAfter=9),
    "heading": ParagraphStyle("heading", fontName="ReportBold", fontSize=11, leading=14, textColor=BLUE, spaceBefore=9, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Report", fontSize=9.2, leading=12.2, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Report", fontSize=8.0, leading=10.3, textColor=INK, spaceAfter=4),
    "table": ParagraphStyle("table", fontName="Report", fontSize=9, leading=11, textColor=INK),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8, leading=11, textColor=INK, spaceAfter=5),
    "equation": ParagraphStyle("equation", fontName="Report", fontSize=10, leading=14, textColor=INK, alignment=TA_CENTER, spaceAfter=5),
}


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def table(rows, widths, highlights=()):
    data = [[p(str(value), "table") for value in row] for row in rows]
    obj = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    rules = [("VALIGN", (0,0),(-1,-1), "MIDDLE"), ("BACKGROUND",(0,0),(-1,0), PALE),
             ("LINEBELOW",(0,0),(-1,0),.7, BLUE), ("LINEBELOW",(0,-1),(-1,-1),.5, BLUE),
             ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
             ("LEFTPADDING",(0,0),(-1,-1),7), ("RIGHTPADDING",(0,0),(-1,-1),7)]
    for row in highlights:
        rules.append(("BACKGROUND", (0,row),(-1,row), AMBER if row==5 else PALE))
    obj.setStyle(TableStyle(rules))
    return obj


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D1DDE4")); canvas.line(44, 37, 568, 37)
    canvas.setFont("Report", 7.5); canvas.setFillColor(colors.HexColor("#526472"))
    canvas.drawString(44, 25, "STDOUT-ONLY AUDIT  |  3 OCT 2026 (ET)  |  Full JSON verification pending")
    canvas.drawRightString(568, 25, f"{doc.page} / 2")
    canvas.restoreState()


def build():
    data = json.loads((HERE / "summary.json").read_text())
    failure = json.loads((HERE / "preflight_failure_audit.json").read_text())
    rows = [["<b>Policy</b>","<b>Seed</b>","<b>Success</b>","<b>Team return</b>","<b>Agent return</b>"]]
    for model in ("shared", "banked"):
        for row in [r for r in data["records"] if r["model"]==model and r["condition"]=="nominal"]:
            rows.append([model.title(), row["training_seed"], f"{row['success_rate']*100:.2f}%",
                         f"{row['mean_team_return']:.4f}", f"{row['mean_agent_return']:.4f}"])
    for model in ("shared", "banked"):
        m = data["nominal"][model]["metrics"]
        rows.append([f"<b>{model.title()} mean</b>", "0-2", f"<b>{m['success_rate']['equal_seed_mean']*100:.2f}%</b>",
                     f"<b>{m['mean_team_return']['equal_seed_mean']:.4f}</b>", f"<b>{m['mean_agent_return']['equal_seed_mean']:.4f}</b>"])
    story = [p("PCP frozen evaluation<br/>and HetNet preflight", "title"),
        p("Preliminary research note | Summary stdout from arrays 902506 and 902621 | Source commit e8bca43", "sub"),
        p("<b>Main finding:</b> shared policies have strong pooled success; banked performance varies substantially by seed. The sensor-loss panel has almost no event exposure, so equal failure/sham scores do not demonstrate fault tolerance."),
        p("1. What the frozen runs show", "heading"),
        p("All 18 jobs printed complete evaluation summaries, covering 16,200 episode evaluations: 15,000 nominal, 600 failure and 600 sham. Their stderr contains the Gym maintenance warning, with no traceback. This supports application completion; scheduler exit states were not supplied. [1]"),
        p("Nominal results below are pooled across the preparer's five declared compositions, 500 episodes each: 2P1A, 1P2A, 2P2A, 3P1A and 3P2A. The manifest is outside this stdout-only audit, so the panel mapping remains to be verified. Higher return (less negative) is better. [1,2]"),
        table(rows, [130,45,100,120,129], highlights=(5,7,8)),
        Spacer(1,5),
        p("<b>Positive:</b> shared averages 99.72% success; its seed range is 99.16-100%. Banked seeds 0 and 2 also perform well, and banked seed 0 exceeds shared seed 0. These are promising frozen-policy results on the assigned aggregate panel."),
        p("<b>Negative:</b> banked averages 91.17%, with a 75.12-99.92% seed range. Seed 1 accounts for 622 of its 662 failures (93.96%). Under the declared 500-episode-per-composition panel, at least 122 of those 622 failures must be in transferred compositions, since the native block contains only 500. Which composition fails is not identifiable from stdout. Retain this seed; the data do not diagnose gate collapse or an unsuitable bank count. [1,2]"),
        p("2. Sensor-loss result: an exposure limitation", "heading"),
        p("Only <b>1 of 600 policy-scenario assignments</b> reached a sensor-loss event (shared seed 0: 1/100; the other five policies: 0/100). The other 599 ended before their own scheduled event in steps 10-30, not necessarily before step 10. All six failure/sham pairs have identical aggregate success and returns; the sole exposed trial fails in both conditions. [1,2]"),
        p("For nonzero exposure and verified pre-event coupling, unexposed outcomes match and:", "small"),
        p("mean(failure - sham) = (exposed / assigned) x mean(exposed-pair differences)", "equation"),
        p("Thus shared seed 0's success-rate effect can be at most 1 percentage point on this panel; the zero-exposure pairs never test physical sensor loss. Full JSON traces are required to verify coupling. Exposure is policy-dependent, so exposed-only results are not a common cross-policy test population. [3]", "small"),
        PageBreak(),
        p("Failure diagnosis and next actions", "title"),
        p("Engineering cause, scientific limits, and immediate priorities", "sub"),
        p("3. Why HetNet did not start", "heading"),
        p("All four <b>new preflight</b> jobs stopped in <font name='Courier'>validate_source_origins()</font> with <i>Runtime file inventory differs from ORIGINS.json</i>. All four stdout files are empty. The guard runs before run-directory creation and training, so this is not evidence of an optimizer, model or memory failure; it yields no Stokes throughput measurement. [4]"),
        p("<b>Reproduced packaging bug:</b> the manifest expects 38 runtime files, but Git commit e8bca43 contains 37. The missing file is <font name='Courier'>runtime/envs/LICENSE.md</font>, excluded by the blanket Markdown ignore rule. A clean tracked export reproduces the failure; restoring only the existing, hash-matching license makes validation pass. This explains a Git-based deployment. The copied remote error did not identify its missing file, so the Stokes inventory still needs confirmation. [4,5]"),
        p("<b>Fix:</b> the license now has a narrow ignore exception; diagnostics name missing, unexpected and modified files. A read-only Slurm source audit was added. Runtime/model/optimizer code and ORIGINS hashes were preserved. The focused fix tests passed (28 tests). After syncing the fix <i>including the license</i>, run:"),
        p("mkdir -p logs_1<br/>sbatch slurm/publication_source_audit.sbatch<br/># After the audit passes:<br/>sbatch slurm/publication_preflight.sbatch", "code"),
        p("Separate legacy PP logs under <font name='Courier'>logs_1/prev/</font> end at epochs 1018/972/1321 after SIGTERM cancellation. Those newer logs extend the root copies. They do not state who cancelled the jobs or why; they must not be conflated with the new preflight inventory error. The Gym warning in frozen-evaluation stderr is not its cause. [4,5]", "small"),
        p("4. What can be written now, and what must wait", "heading"),
        p("<b>Write now:</b> the all-seed aggregate table, banked variability, low failure exposure and this execution diagnosis. Use descriptive seed means/ranges: 7,500 evaluations per model do not provide 7,500 independently trained models. There are three training seeds per model. [6]"),
        p("<b>Do not claim yet:</b> native PCP superiority to the paper, faster capture, failure robustness or a gate mechanism. The nominal table mixes team sizes. Reported agent return is mean<sub>episodes</sub>(R<sub>team</sub>/N), not mean(R<sub>team</sub>)/mean(N). Average steps and composition-specific scores are absent. All checkpoints are epoch 1500/update 15,000, with 30.508-30.729M actual steps; stdout does not verify the first-saved selection or evaluator source identity. [1-3]"),
        p("<b>Next:</b> (1) audit the incoming manifest, panels and 18 full JSONs; split results by composition and inspect banked seed 1. (2) Pass the source audit and four-workload preflight before long HetNet training. (3) Preserve this reference failure panel; prerecord an earlier supplemental event on training/native compositions with fresh scenarios and matched HetNet/sham controls. Choose timing for measurable exposure, not for which model wins."),
        p("Traceable evidence", "heading"),
        p("<b>[1]</b> logs_sr/pcp-frozen-902506_{0..17}.out, lines 11-28; nominal indices 0,3,6,9,12,15; preparation: pcp-prepare-902504.out. Copied inputs, hashes and exact calculations: <font name='Courier'>analysis/pcp_frozen_2026-10-03/summary.json</font> and <font name='Courier'>summary_rows.csv</font>.", "small"),
        p("<b>[2]</b> scripts/prepare_pcp_frozen.py:193-194,203-210,224-240 (panels, mapping, selector). <b>[3]</b> softrole/evaluate.py:111-123; softrole/__main__.py:162-163 (stdout omissions); softrole/rollout.py:111-127,169-193 (event and paired-stream semantics).", "small"),
        p("<b>[4]</b> logs_1/hetnet-preflight-902621_{0..3}.err:1-18; original guard: commit e8bca43, publication_reconstruction/__main__.py:177-186,290. <b>[5]</b> <font name='Courier'>analysis/pcp_frozen_2026-10-03/preflight_failure_audit.json</font>: clean-export reproduction, input/code hashes and legacy endpoint references.", "small"),
        p("<b>[6]</b> <link href='https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html' color='#17698A'>Agarwal et al., NeurIPS 2021, Deep RL at the Edge of the Statistical Precipice</link>; <link href='https://arxiv.org/abs/1709.06560' color='#17698A'>Henderson et al., AAAI 2018, Deep RL that Matters</link>. Both motivate careful treatment of variation across training runs; neither establishes significance of this three-seed comparison.", "small")]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(612,792), leftMargin=44, rightMargin=44,
                           topMargin=40, bottomMargin=48, title="PCP frozen evaluation and HetNet preflight: stdout audit",
                           author="HetNet research analysis", pageCompression=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(OUTPUT)
    assert len(reader.pages)==2, f"Expected exactly two pages, got {len(reader.pages)}"
    text = "\n".join(page.extract_text() for page in reader.pages)
    for expected in ("99.72%", "91.17%", "LICENSE.md", "1 of 600", "622"):
        assert expected in text, expected
    (HERE / "report_text.txt").write_text(text)
    record = {"pdf":str(OUTPUT.relative_to(ROOT)),"pages":len(reader.pages),
              "sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              "builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "python":platform.python_version(),"packages":{name:version(name) for name in ("reportlab","pypdf")}}
    (HERE / "pdf_build.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps(record))


if __name__ == "__main__":
    build()
