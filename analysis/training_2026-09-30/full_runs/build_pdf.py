"""Render the amended Markdown report with actual LaTeX mathematics.

uv run --no-project --with pypandoc-binary==1.15 --python .venv/bin/python \
  python analysis/training_2026-09-30/full_runs/build_pdf.py
Requires the installed tectonic executable; training dependencies are untouched.
"""
import json
import re
import subprocess
from pathlib import Path
import pypandoc

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent
header=HERE/'pdf_header.tex'
header.write_text(r'''\usepackage{microtype}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage[section]{placeins}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small HetNet and SoftRole Training Review}
\fancyhead[R]{\small 30 September 2026}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{3em}
''')
text=(REPORT/'report.md').read_text().split('\n',1)[1]
text=text.replace('Full run artifact amendment, 30 September 2026\n','',1)
text=re.sub(r'!\[Figure \d+\. ', '![', text)
# PDF uses vector counterparts of the six figures for clean print scaling.
for name in ['sample_learning_curves','common_sample_comparison','optimization_diagnostics',
             'gate_diagnostics','expert_weights','episode_length_survival']:
 text=text.replace(f'full_runs/{name}.png',f'full_runs/{name}.pdf')
latex=pypandoc.convert_text(text,'latex',format='markdown+tex_math_single_backslash',extra_args=[
 '--standalone','--toc','--toc-depth=2','--shift-heading-level-by=-1','--resource-path='+str(REPORT),
 '--metadata=title:HetNet and SoftRole Training Review',
 '--metadata=subtitle:Full run artifact amendment',
 '--metadata=date:30 September 2026',
 '--variable=fontsize:10pt','--variable=geometry:margin=0.8in',
 '--variable=linkcolor:blue','--variable=urlcolor:blue',
 '--include-in-header='+str(header)])
# Keep the compact complete inventory together instead of leaving its first row
# on the preceding prose page. The final table is the primary run inventory.
inventory=latex.rfind(r'\begin{longtable}')
assert inventory>=0
latex=latex[:inventory]+r'\clearpage'+'\n'+latex[inventory:]
(REPORT/'training_report.tex').write_text(latex)
result=subprocess.run(['tectonic','--keep-logs','--keep-intermediates','training_report.tex'],
                      cwd=REPORT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(HERE/'pdf_build.log').write_text(result.stdout)
print(result.stdout[-12000:])
result.check_returncode()
(HERE/'pdf_tools.json').write_text(json.dumps({'pypandoc':pypandoc.__version__,
 'pandoc':str(pypandoc.get_pandoc_version()),
 'tectonic':subprocess.check_output(['tectonic','--version'],text=True).strip()},indent=2)+'\n')
print('Published',REPORT/'training_report.pdf')
