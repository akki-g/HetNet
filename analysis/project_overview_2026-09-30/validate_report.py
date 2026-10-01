"""Validate deliverable structure, embedded fonts, fingerprints and page bounds."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def run(*args):
    return subprocess.check_output(args, cwd=HERE, text=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

pdf = HERE / 'project_overview.pdf'
info = run('pdfinfo',str(pdf))
assert re.search(r'Pages:\s+10\b',info), info
text = run('pdftotext','-layout',str(pdf),'-')
pages = [p for p in text.split('\f') if p.strip()]
assert len(pages) == 10
page_labels = ['01 / FOUNDATIONS','02 / ARCHITECTURE OVERVIEW','03 / COMPONENTS',
               '04 / READING THE EVIDENCE','05 / PREDATOR','06 / PREDATOR',
               '07 / FIRECOMMANDER','08 / LEARNING DIAGNOSTICS',
               '09 / ADAPTATION AND STATUS','10 / TRACEABILITY']
for page,label in zip(pages,page_labels):
    assert label in page, label
    assert 'Evidence.' in page
for token in ['37,566','631,310,527','10,688','20.0%','33.5 million','37 million','20 million']:
    assert token in text,token
log = (HERE/'project_overview_rendered.log').read_text()
assert not re.search(r'Overfull|Underfull|Missing character|^!',log,re.M), 'TeX layout warning'
fonts = run('pdffonts',str(pdf))
font_rows = [line.split() for line in fonts.splitlines()[2:] if line.strip()]
assert font_rows and all(row[-5]=='yes' for row in font_rows),fonts
bbox = run('pdftotext','-bbox-layout',str(pdf),'-')
tree = ET.fromstring(bbox)
word_count = 0
for page in tree.iter('{http://www.w3.org/1999/xhtml}page'):
    width,height = float(page.attrib['width']),float(page.attrib['height'])
    for word in page.iter('{http://www.w3.org/1999/xhtml}word'):
        word_count += 1
        assert 0 <= float(word.attrib['xMin']) < float(word.attrib['xMax']) <= width+.1
        assert 0 <= float(word.attrib['yMin']) < float(word.attrib['yMax']) <= height+.1
assert word_count > 3000
input_count = 0
for name,entry in json.loads((HERE/'input_manifest.json').read_text()).items():
    assert sha(ROOT/name)==entry['sha256'],name
    input_count += 1
figure_manifest = json.loads((HERE/'figure_manifest.json').read_text())
for name,entry in figure_manifest['inputs'].items():
    assert sha(ROOT/name)==entry['sha256'],name
raw_provenance = json.loads((HERE/'latest_snapshot/provenance.json').read_text())
for name,entry in raw_provenance['inputs'].items():
    assert sha(ROOT/name)==entry['sha256'],name
audit = json.loads((HERE/'review_results.json').read_text())
assert audit['completed_epochs']==37566 and audit['runs']==27
assert audit['softrole_total_steps']==631310527
for p in HERE.glob('*.py'):
    compile(p.read_text(),str(p),'exec')
references = re.findall(r'\\(?:href|refnum)\{(?:[^}]+)\}\{(https?://[^}]+)\}',(HERE/'project_overview.tex').read_text())
# Hyperref's direct href form has its URL in the first argument.
references += re.findall(r'\\href\{(https?://[^}]+)\}',(HERE/'project_overview.tex').read_text())
validation = {
    'pages':10,'required_first_three_pages':page_labels[:3],
    'all_pages_have_evidence_notes':True,'all_fonts_embedded':True,
    'font_count':len(font_rows),'tex_layout_warnings':0,
    'all_extracted_words_within_page_bounds':True,'extracted_words':word_count,
    'local_input_hashes_rechecked':input_count,
    'snapshot_input_hashes_rechecked':len(raw_provenance['inputs']),
    'raw_audit':{k:audit[k] for k in ['completed_epochs','runs','raw_csv_numeric_fields_checked','window_numeric_fields_checked','group_numeric_fields_checked','softrole_total_steps']},
    'primary_reference_urls':sorted(set(references)),
    'python_source_compilation':'passed',
    'pdf_sha256':sha(pdf),'pdf_bytes':pdf.stat().st_size,
    'work_log_sha256':sha(ROOT/'AGENTS.md'),
    'visual_review':'All ten rendered pages inspected; final changed pages re-inspected by the authoring agent.',
    'scope':'Document build and read-only analysis; no training, evaluation, simulator changes, dependency edits or runtime tests.'
}
(HERE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(HERE/'project_overview.txt').write_text(text)
artifacts = {}
for path in sorted(HERE.rglob('*')):
    relative = path.relative_to(HERE)
    if (not path.is_file() or 'previews' in relative.parts or '__pycache__' in relative.parts
            or path.suffix in {'.aux','.log','.out','.xdv'} or path.name=='artifact_manifest.json'):
        continue
    artifacts[str(relative)]={'bytes':path.stat().st_size,'sha256':sha(path)}
(HERE/'artifact_manifest.json').write_text(json.dumps({'files':artifacts,
    'excludes':'This manifest, transient compiler files, bytecode and visual-review previews.'},indent=2)+'\n')
print(json.dumps(validation,indent=2))
