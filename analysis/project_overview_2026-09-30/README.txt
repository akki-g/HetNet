CAPABILITY-CONDITIONED HETNET: PROJECT OVERVIEW AND CURRENT EVIDENCE
30 September 2026

Start with project_overview.pdf. Its first three pages explain foundations,
the architecture overview, and individual components. The remaining pages
compare current training, show diagnostics, explain the separate sensor pilot,
and give an evidence map. The results incorporate the logs resynced during
document preparation, superseding the earlier same-day stdout snapshot.

Evidence and audit
------------------
latest_snapshot/summary.json: exact per-run and grouped window results.
latest_snapshot/epoch_metrics.csv: all parsed epochs, with source-line numbers.
latest_snapshot/provenance.json: current input fingerprints and prefix checks.
review_results.json: independent raw-log and numerical checks, FC and pilot checks.
comparison_values.csv: unrounded values used for report comparisons.
evidence_index.json: claims mapped to repository evidence and lookup fields.
input_manifest.json: input identities used for this document.
figure_manifest.json: plotting method, input/output hashes and endpoint checks.
validation.json: PDF structure, fonts, input stability and rendering checks.
artifact_manifest.json: exact deliverable file list and SHA256 hashes.

Rebuild from the repository root
-------------------------------
These commands use isolated plotting packages and do not alter project dependencies.
Refreshing analysis requires the identified raw logs and archived repository inputs.
Review narrative conclusions if any data changes; the text is intentionally authored,
not an automatic interpretation engine.

.venv/bin/python analysis/project_overview_2026-09-30/refresh_results.py
.venv/bin/python analysis/project_overview_2026-09-30/review_results.py
uv run --no-project --with matplotlib==3.10.7 --with numpy==1.26.4 \
  --python .venv/bin/python python analysis/project_overview_2026-09-30/make_figures.py
.venv/bin/python analysis/project_overview_2026-09-30/prepare_report.py
.venv/bin/python analysis/project_overview_2026-09-30/build_pdf.py
.venv/bin/python analysis/project_overview_2026-09-30/validate_report.py

Tectonic builds LaTeX; Poppler supplies pdfinfo, pdftotext and pdffonts.
The authored source is project_overview.tex; generated table files and the expanded
project_overview_rendered.tex make every number inspectable. figures/ contains
vector PDFs and PNGs, including an optional FC regression detail not embedded in
the compact report. No policies are executed by these scripts.

Interpretation
--------------
HetNet Real reproduction and SoftRole use different observation/learning pipelines.
Reproduction stdout sample counters overcount, so its new tail uses epochs.
SoftRole comparisons use equal sample budgets, episode weights within each seed,
and equal weights across three seeds. Curves are changing-policy training records,
not frozen-policy or held-out results. The new tails extend verified archived
prefixes but do not supply newer source/checkpoint archives. FC simulator defects
and the separate, one-seed sensor pilot are explicitly distinguished.
