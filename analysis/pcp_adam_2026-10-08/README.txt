PCP Adam trial analysis - 8 October 2026

Read the six-page report:
  output/pdf/pcp_adam_analysis_2026-10-08.pdf

The new 911488 batch is three shared CapCom/SoftRole Adam trials. It is not a
banked Adam batch. This report also analyzes the latest 908316 paper-v1 PCP
HetNet stdout, older RMSprop PCP training, and existing frozen PCP outcomes.
All analysis is read-only with respect to scientific inputs and runtime code.
No training, policy evaluation, cluster job, commit or push was performed.

Main findings
  * All three Adam runs finish 2000 epochs / 20000 updates, about40.6M steps.
    Last100 training epochs average5.401 steps, versus latest HetNet tails~5.12.
    These tails use different budgets and are not frozen-policy results.
  * At common26-27M: Binary5.116, shared RMSprop6.095, shared Adam6.119,
    Real6.655 and banked RMSprop7.534. Real seed0 regressed then recovered.
  * Adam does not show a consistent improvement in learning per sample.
    Recorded epoch loops take15.31-15.37h vs24.67-24.87h for completed older
    shared seeds1/2. Full Slurm elapsed time/hardware attribution are unavailable.
  * Physical PCP dynamics/rewards remain unchanged. Corrected observation
    copies differ materially from historical buggy release observations.
    Published Table1 and Figure3 are different experimental summaries.
  * Full audit:120 new checkpoints,918 archived source files,15,224,881 episodes,
    121,822,429 steps,60000 updates and6000 epochs. One final stdout attention
    scalar is invalid; structured metrics and episode reaggregation agree.

Reproduce from repository root (existing .venv for Torch checkpoint reading):
  .venv/bin/python analysis/pcp_adam_2026-10-08/trial_review/audit.py --episodes
  python3 analysis/pcp_adam_2026-10-08/speed_review/audit.py
  python3 analysis/pcp_adam_2026-10-08/speed_review/episode_window_audit.py
  python3 analysis/pcp_adam_2026-10-08/speed_review/build_notes.py
  python3 analysis/pcp_adam_2026-10-08/paper_review/audit.py
  python3 analysis/pcp_adam_2026-10-08/frozen_audit.py
  python3 analysis/pcp_adam_2026-10-08/compare.py
  python3 analysis/pcp_adam_2026-10-08/trial_review/verify.py
  uv run --no-project --python .venv/bin/python --with numpy==1.26.4 \
    --with matplotlib==3.10.7 --with reportlab==4.4.4 --with pypdf==6.1.1 \
    python analysis/pcp_adam_2026-10-08/build_report.py

Inspect each subfolder README/notes for audit scope and any supplementary
commands. Audits write derived files in this fresh analysis folder only.
The full episode audit streams approximately14GB of JSON; paper stdout adds
several more gigabytes. Rebuilds need the copied local input directories.
PDF dependencies are isolated with uv --no-project, not added to the project.

Primary evidence files
  comparison_windows.csv: every per-seed window with exact retained boundaries.
  comparison_groups.csv: equal-seed aggregate comparisons and observed ranges.
  curve_bins.csv: complete1M windows used in the learning-curve figure.
  frozen_summary.json:18 saved reports/16200 outcomes, failure exposure,
      oracle enumeration and current checkpoint-byte availability.
  trial_review/: complete new-run episode/update/checkpoint/source audit;
      older terminal-checkpoint checks and historical training windows.
  speed_review/: source/timing comparison, latest paper epoch/status data,
      selected raw paper episode reconciliation and all12-wave status inventory.
  paper_review/: primary references, source-level dynamics/observation checks,
      independent initial-visibility and geometric-oracle enumeration.
  training_curves.{png,svg}, matched_performance_speed.{png,svg}: standalone plots.
  report_text.txt: extracted report text; build_report.py: editable report source.
  validation.json and artifact_manifest.json: checks and exact output hashes.

Interpretation rules
  A sample window keeps complete epochs with lower<epoch-end steps<=upper;
  an epoch crossing the lower boundary is included whole. Retained boundaries
  are explicit, so these are approximately1M rather than exact1M subsets.
  Weight episodes within each training seed and then weight seeds equally.
  Do not pool millions of episodes as independent training replications.
  Seed shading/ranges are descriptive, not confidence intervals.
  Training metrics are not frozen evaluations or isolated architecture effects.
  New Adam run metrics take precedence over the isolated invalid stdout scalar.
  Prior frozen reports are RMSprop, not evaluations of the new Adam policies.
  Paper-v1 scientific checkpoints/source are not available in the copied root;
  older public-code/reproduction-fast artifacts are not substitutes for them.
  Five of six selected older frozen checkpoint files are now present and match;
  shared seed2 remains missing. No new replay was performed.
