"""Render the short resync report without altering the training environment."""
from pathlib import Path
import subprocess
import pypandoc

HERE = Path(__file__).resolve().parent
text = (HERE / "quick_report.md").read_text().split("\n", 1)[1]
latex = pypandoc.convert_text(text, "latex", format="markdown", extra_args=[
    "--standalone", "--metadata=title:HetNet / SoftRole training update",
    "--metadata=date:Resynced logs, 30 September 2026",
    "--variable=fontsize:10pt", "--variable=geometry:margin=0.75in",
    "--variable=linkcolor:blue", "--variable=urlcolor:blue",
    "--variable=header-includes:\\setlength{\\emergencystretch}{3em}",
])
(HERE / "quick_report.tex").write_text(latex)
subprocess.run(["tectonic", "--keep-logs", "quick_report.tex"], cwd=HERE, check=True)
