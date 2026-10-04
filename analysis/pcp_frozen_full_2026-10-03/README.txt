PCP frozen evaluation: full synced results, 3 October 2026

This extends, and does not overwrite, analysis/pcp_frozen_2026-10-03/.
Final two-page PDF: output/pdf/pcp_frozen_full_results_2026-10-03.pdf

From the repository root:
  python3 analysis/pcp_frozen_full_2026-10-03/analyze.py
  .venv/bin/python analysis/pcp_frozen_full_2026-10-03/provenance_audit.py
  python3 analysis/pcp_frozen_full_2026-10-03/failure_audit.py
  python3 analysis/pcp_frozen_full_2026-10-03/independent_metrics.py
  uv run --no-project --python .venv/bin/python --with reportlab==4.4.4 --with pypdf==6.1.1 python analysis/pcp_frozen_full_2026-10-03/build_report.py

The provenance audit needs NumPy, available in the existing pinned virtual
environment. The main metric and failure scripts use only the standard library.
The PDF builder uses embedded macOS Arial fonts. Poppler renders pages for QA.
No dependency or environment was changed for this analysis.

input_manifest.json hashes every supplied bundle file except .DS_Store.
summary.json and composition_seed_metrics.csv contain all recomputed metrics.
Independent audits cover source identity, scenarios, reward arithmetic and all
failure/sham traces. validation.json records cross-checks and PDF inspection.
artifact_manifest.json lists output hashes without attempting to hash itself.

The selected epoch1500 checkpoint files are absent and the local training
ledgers are old. Recorded checkpoint identity agrees across supplied records;
actual selected weight bytes, a new replay, and first-available checkpoint
selection cannot be independently verified. No model was run or tuned here.

The earlier HetNet packaging diagnosis is cited, not claimed to have been
reconfirmed on Stokes. No source-audit or training job was submitted remotely.
