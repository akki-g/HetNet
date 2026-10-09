# Proposed route to an experiment-complete paper draft by 12 October

Prepared 5 October 2026 from checkout `ddcd972`. This is a scoped proposal,
not an executed experiment, a preregistration of a new panel, or confirmation
of live Stokes job status. The recommendation assumes the core paper is a
paper-aligned HetNet comparison, frozen composition transfer, and a focused
within-policy mechanism test. Reproducing every original HetNet baseline and
ablation, or establishing the benefits of failure-trained adaptation, is a
larger scope and cannot be promised by next week from the current evidence.

## Current evidence and implications

- The new HetNet backend and official preflights passed. At twelve concurrent
  starts, the slowest observed workload projects to 56.5–86.8 active hours per
  Binary seed, including a continuation after the 46-hour pause. This permits
  a Thursday/Friday completion target for Monday starts, but queueing, staggered
  admission and changing rates can move it. The prior 6.6–9.1-day estimate was
  for three occupied slots. [Preflight evidence](../stokes_preflight_2026-10-05/report.md)
- Existing PCP frozen evaluation is already complete: 15,000 nominal/transfer
  episodes plus 1,200 failure/sham episodes. Reuse its reports if checkpoint,
  physical-domain and common-panel checks pass; do not rerun merely for coverage.
  Shared transfer success is 99.70% versus 88.97% banked across the four changed
  teams. Keep every seed, including the unstable banked seed. The current
  evidence does not establish banked superiority.
  [Full record](../pcp_frozen_full_2026-10-03/report_text.txt)
- Only 1/600 assigned failure episodes reached the event in the old steps10–30
  panel. It does not establish comparative robustness. Earlier native-team
  failure should be a separately declared supplemental condition, with fresh
  scenarios and preserved old results, rather than an unmarked replacement.
  [Exposure audit](../pcp_frozen_full_2026-10-03/failure_audit.json)
- The copied primary SoftRole archive is incomplete: PCP banked0/1/2 and shared1
  have completed ledgers; shared0's newer copy stops around33.67M; shared2 and
  all six PP checkpoint directories in that copy lack weights. This is an
  archival dependency, not proof of remote training failure. Retrieve selected
  checkpoints, signatures, source manifests, complete ledgers and job status.
  Five selected PCP epoch1500 weights now exist and match the frozen manifest;
  shared2 remains absent, superseding the old report's blanket statement that
  selected weights were absent. All six PP stdout logs reach2000 epochs/about40.3M
  steps, so retrieve those completed runs rather than retraining.
- PCP shared seed0 has a confirmed48h timeout in
  [its stderr](../../logs_sr/softrole-898818_6.err). Its last valid local
  epoch1650 checkpoint contains33,569,089 steps; the copied ledger continues to
  epoch1655/33,669,601. Prepare a source-verified continuation from the valid
  checkpoint to finish40M; its recent50–200epoch rates project11.8–11.9 active
  hours, conditional on similar hardware/rate. Preserve and exclude the abandoned
  post-checkpoint suffix from combined counts. Its existing30M frozen comparison
  remains usable. [Structured ledger](../../stokes_runs/softrole_primary/pcp_shared/seed0/metrics.jsonl)
- Six fresh paper-FC SoftRole runs remain necessary for matched FC semantics.
  Locally available completed FC SoftRole runs use the older environment. No
  production paper-FC results are present in this snapshot. Legacy FC timing
  is not a measured paper-FC forecast; derive an ETA from its first production
  updates. [Matched launcher](../../slurm/softrole_paper_fc.sbatch)

## Lock the comparison before new evaluation results

Retain the already-declared PCP **30M evaluation selection** rule, while allowing
all research training to finish its full40M budget. Select the first saved
checkpoint at or above30M along the retained continuation lineage for both
methods. The new HetNet launcher saves a30M milestone. Comparing existing30M
SoftRole against new40M HetNet without labeling the mismatch is not acceptable.
Any40M comparison should be a separately declared matched-budget analysis.
PP uses40M and FC28M. [Existing thresholds](../../publication_reconstruction/STUDY.json),
[milestone implementation](../../publication_reconstruction/study.py),
[paper-FC preparation](../../publication_reconstruction/paper_study.py).

| Primary comparison | Policies | Teams and scenarios | Complete panel size |
|---|---|---|---:|
| PP nominal | HetNet Real, SoftRole shared/banked; seeds0–2 | Native3P0A;500 common cases | 4,500 |
| PCP nominal and frozen transfer | HetNet Real/Binary, SoftRole shared/banked; seeds0–2 | (2,1),(1,2),(2,2),(3,1),(3,2);500 each | 30,000 |
| FC nominal | HetNet Real, fresh paper-FC SoftRole shared/banked; seeds0–2 | Native2P1A;500 common cases | 4,500 |

The nominal matrix is39,000 policy episodes, of which15,000 PCP SoftRole
episodes already exist. Nominal scenario seeds/panels are2700 PCP,2702 PP,
2703 FC. Preserve existing PCP panel bytes, evaluator identities and selection
receipts; do not falsely label reused results as produced by a new evaluator.
Held-out compositions already examined remain historical test results, not
unseen data for selecting a new architecture.

## Engineering work to finish while training runs

1. **Build the paper-specific evaluation preparer.** Reuse validated checkpoint
   candidates and retained-lineage accounting; support moved archive/run maps.
   Bind task, seed, model, backend, environment, budget, actual overshoot,
   source, checkpoint hash and scenario hash. Generate explicit protocol
   sidecars and Slurm evaluation commands without automatically submitting.
   The legacy `prepare-evaluation` constructs old reconstruction protocols;
   `prepare_pcp_frozen.py` is PCP-only and is not a general SoftRole continuation
   selector. `paper-study-plan` currently prepares FC scenarios and a textual
   selection rule, not final checkpoint manifests.
2. **Complete paired reporting and input validation.** Existing equal-seed
   summaries and within-policy failure-minus-sham comparisons can be reused.
   Add between-method descriptive contrasts and the gate-intervention interaction
   below, keeping seed-level values. Reject inconsistent composition, sham and
   success types (a deferred parser-hardening issue). Use explicit common
   protocol sidecars; otherwise differing epochs/updates at matched step budgets
   can split seeds into separate summary strata. Do not remove source/environment
   strata simply to force aggregation.
3. **Prepare recovery and inventory checks.** Track each seed's actual sample
   budget, most recent valid checkpoint, paused/completed status and retained
   parent chain. Arrange HetNet continuation promptly after its46h pause.
   SoftRole requires its separate periodic-checkpoint recovery route; it has no
   equivalent graceful stop. [Recovery guidance](../stokes_preflight_2026-10-05/provenance_review/softrole_recovery.md)
4. **Generate paper-ready tables and plots from structured records.** Include
   per-seed success, mean-agent and team returns, horizon-capped completion,
   learning curves versus true environment steps, model/bandwidth/information
   differences, and event exposure. Build the figure pipeline now using audited
   existing reports; replace inputs with final manifests without hand editing
   numbers. Start the architecture, methods and limitations text now.

Work in a separate evaluation/orchestration checkout. HetNet executes archived
source, while SoftRole training/recovery uses the current checkout's code. Do not
change active training source, dependencies or hyperparameters to meet the date.
New tooling needs focused checks for real failure modes: missing/duplicate seeds,
incorrect budget/backend/environment, abandoned continuation suffixes, mismatched
panels, changed checkpoints, and invalid paired arithmetic.

## Focused supplemental experiment, if included in the core paper

Propose native PCP2P1A sensor loss at **zero-based step1**, on a fresh common
500-scenario panel, with failure and scheduled-event sham for each of the12
PCP policies:12,000 episodes. Fix the timing and panel before inspecting these
outcomes. Step1 permits a pre-event gate; freezing at step0 is unsupported.
Measure actual exposure rather than assuming every episode reaches the event.
This studies frozen nominal policies under a new perturbation, not policies
trained for failure. Earlier loss is chosen to address the old unexposed panel,
not to maximize a favorable model difference.

For banked seeds0–2, add `freeze_all` and `comm_off`, each under failure and
sham on that same panel:6,000 additional episodes. Together these require18,000
new episode evaluations and no new training. Use a tiny execution/timing probe
before scheduling the full panel; execution checks must not select a condition
based on return. Estimate completion from measured rollout throughput rather
than treating a Slurm time limit as an ETA.

For each seed and each metric, report the paired gate contrast
`(normal_failure - frozen_failure) - (normal_sham - frozen_sham)`.
For success, a positive contrast means gate changes help more under failure;
interpret the direction appropriately for completion time. Pair scenario IDs
within a checkpoint first, then give seeds equal weight. Keep every assigned
episode and report exposure/censoring. Do not infer that visible gate motion
proves useful adaptation or semantic roles. These frozen interventions cannot
replace capability-only/constant-gate training controls for broader causal
claims about the architectural choice.

The operators are already implemented in [SoftRole rollout](../../softrole/rollout.py)
and [HetNet evaluation](../../publication_reconstruction/evaluation_worker.py).
The latter supports PCP sensor masking; old SoftRole `evaluate-hetnet` guidance
must not be confused with this newer reconstruction evaluator. The current
[summary implementation](../../softrole/report.py) already has within-policy
failure-minus-sham inference, despite stale wording in `RESEARCH.md`; the new
gate interaction and between-method contrasts remain analysis work.

## Calendar and completion criteria

| Date | Work that should complete |
|---|---|
| Mon5 Oct | Lock scope/selection/panels; confirm live run inventory; launch matched FC if missing; resume timed-out PCP shared0 from its verified source/checkpoint; obtain existing SoftRole weights; implement selector and protocol manifest. |
| Tue6 Oct | Validate preparation/reporting; evaluate available SoftRole policies; time frozen evaluation; build tables and paired mechanism analysis. |
| Wed7 Oct | Resume paused HetNet seeds around46h after their actual start; evaluate eligible PP/FC/PCP checkpoints as they arrive; inspect resource and count evidence. |
| Thu8–Fri9 Oct | Complete remaining training and frozen jobs if queues/rates permit; audit missing seeds/cells and source/budget compatibility. |
| Sat10–Sun11 Oct | Run only repairs for genuinely missing/failed cells; regenerate final tables/figures; independent result/provenance check; archive the complete reproducibility bundle. |
| Mon12 Oct | Freeze the result manifest and claims; writing proceeds from those fixed artifacts. |

“Experiments complete” means every required seed reaches its unchanged training
budget; every planned evaluation cell has the full assigned panel; no silently
missing runs or abandoned ledger suffixes remain; checkpoints/scenarios/code are
hash-bound; tables can be regenerated by one documented command; and claims
match the results, including negative findings. Three training seeds are a
limitation:500 scenarios per seed do not supply500 independent training runs.

Do not add bank-count searches, hyperparameter tuning, new communication-width
sweeps, new membership/fault semantics or a GPU migration to this deadline.
Full original-study replication still lacks baseline implementations and
several ablations; [the coverage table](../../README.md#full-original-study-coverage-still-required)
makes that boundary explicit. Dedicated composition/failure training and
capacity-matched conditioning controls need their own compute and protocol if
the paper's central claim requires them. The existing late-event failure-training
preset is not an established solution to the exposure problem.

This plan creates no guarantee of favorable results, queue availability or paper
acceptance. It provides a feasible critical path for the scoped comparison and
frozen-policy study without changing the trained methods after seeing outcomes.
