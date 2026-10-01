"""Build the report with Tectonic; keep authored and expanded TeX separately."""
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
source = (HERE / 'project_overview.tex').read_text()
expanded = re.sub(r'\\input\{([^}]+)\}', lambda m: (HERE / m[1]).read_text(), source)
(HERE / 'project_overview_rendered.tex').write_text(expanded)
subprocess.run(['tectonic','--keep-logs','project_overview_rendered.tex'],cwd=HERE,check=True)
shutil.copyfile(HERE/'project_overview_rendered.pdf',HERE/'project_overview.pdf')
(HERE/'project_overview_rendered.pdf').unlink()
