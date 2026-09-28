# Work log: original HetNet frozen-transfer study

## 26 September 2026 — research gate

- Read the complete revision-2 task attachment, preserved at `research/TASK_SPEC.md`.
- Verified the existing local `akki-g/HetNet` fork was clean and at upstream `bff9f7f`; added `upstream`, tagged `upstream-bff9f7f`, and created `frozen-eval`. No new repository or changes to `marl-comm`.
- Read primary HetNet, supplementary, IC3Net, Howell, CASH, Agarwal and Lowe evidence; checked exact author source and official DGL/ARCC/package sources. Wrote `RESEARCH_NOTES.md` before implementation, with three independent supporting audits and decisions table.
- Identified missing explicit Fig.5c map/horizon details, Real/Binary attribution, paper/code layer/optimizer discrepancies, incorrect upstream cumulative counters, existing observation aliasing, and a potential Torch-2.2 multiprocessing gradient-storage failure.
- Requested live Stokes preflight/account/balance/quota/module/access facts from Akki. They remain unknown; no remote connection or submission has occurred.
- Next: isolated compatibility/lock commit, early-seeding commit, logging/checkpoint commit, Phase-A infrastructure and scientific Gate A tests. No algorithm changes are authorized to hide a gate failure.

## Gate status

| Gate / deliverable | Status | Evidence needed |
|---|---|---|
| Research before implementation | Complete | RESEARCH_NOTES.md + primary/source audits |
| A1 clean lock/import | Passed locally | `evidence/gate_a/clean_environment_20260926T035949Z/` |
| A2 baseline diff | Passed | A/B/G/C commit inventory and preserved exact diff |
| A3 five compositions × one/four-process smoke | Passed | Twelve runs including repeat/alternate seed, 36 epochs/checkpoints; `GATE_A_REPORT.md` |
| A4 shared weights and gradient aggregation | Passed bounded diagnostics within full gate | Actual 2P1A/4P6A, three updates, exact fresh gradient sums and worker storage |
| A5 single-process determinism | Passed | Identical initial/all three epoch signatures and non-timing metrics; alternate seed initialization differs |
| A6 strict cross-composition load | Passed | Source epoch3 tensor names/shapes/dtypes/values preserved in all five targets; singleton-P rejected |
| A7 scripts/dry-run bijection | Passed locally | 21 mappings, script syntax, fail-closed infrastructure tests |
| A8 resume | Conditional | Only implement if calibrated time exceeds cap; then full kill/resume test |
| Stokes preflight | Awaiting user evidence | Limits/account/balance/modules/network/quota/access |
| Stokes calibration and budget | Pending after A | Two 20-epoch jobs; resource/timing table; explicit user confirmation |
| Run1 / early learning gate | Pending | 21 final policies, all source learning curves, review around epoch300 |
| Phase B / Gate B | Not started | Frozen runner, banks, signatures, pairing, sensor/message/permutation/oracle tests |
| Analysis preregistration | Committed before any Run2 outcomes | `4b4c0a6`, clarified by `0f86729`; `research/PREREGISTRATION.json` |
| Run2 | Pending after Run1 and B | Complete checkpoint preflight, frozen evaluation logs |
| Analysis / paper handback | Pending after Run2 | Reproducible tables/figures, RESULTS.md, evidence-mapped PAPER_SKELETON.md |

Full objective remains active; a prepared workflow or smoke pass is not completion of the cluster experiments.

Current steering (28 September 2026): Stokes is working again and Akki requests the complete 21-task sweep with progress monitoring and a calibrated time estimate. The Mac pilot target is superseded. No actual Stokes calibration or study training has run; current-source Gate A is being refreshed.

## 26 September 2026 — local compatibility and scientific gate preparation

- Committed A (`2fadecf`): Python 3.12/Gym/CPU compatibility and locked requested runtime, including missing declared-import dependencies and reproducible legacy build constraint.
- Committed B (`7b334c1`): seed before parser environment, model and trainer construction; retain worker offsets and add Python RNG seeding.
- Diagnosed actual stale cached gradients under Torch 2.2.1 after update 1. Preserved both surrogate and genuine HetNet failures. Worker weights were shared correctly, demonstrating why that check alone is insufficient.
- Akki explicitly approved the reviewable three-call `zero_grad(set_to_none=False)` repair. Committed G (`0cfcea5`); genuine 2P1A and 4P6A three-update checks now have exactly zero aggregation error and current shared weights/storage. This does not establish learning.
- Committed C (`50d0c37`): opt-in fresh-statistic metrics, accurate external cumulative counters, resolved model choices, initial/epoch/checkpoint signatures, completed-epoch cadence and checkpoint timing/size. Upstream stdout and forward/loss math remain as documented.
- Clean scratch lock synchronization and pinned imports passed; preserved evidence at `evidence/gate_a/clean_environment_20260926T035949Z/`. Recorder and infrastructure unit contracts passed locally.
- Corrected the direct-load test fixture before running smokes: recreate float64 defaults but preserve the explicit float32 attention parameters. No upstream model precision change was made.
- Phase-A scripts, budget projection and provenance tooling are being finalized with fail-closed live-cluster, Gate A and budget checks. Full Gate A smoke evidence is still pending. No cluster access, submissions or research outcomes exist yet.

## 26 September 2026 — fixed source gate run and prospective interpretation

- Completed source infrastructure at `2db8a62`: bind budget confirmation to current preflight evidence, verify calibrated endpoint identity/recipe, require four total allocated CPUs, reject source/lock drift, and include first-epoch/residual overhead in cost projections. Local non-training validation: 19 passed, three artifact checks deferred until their runs exist.
- Intentionally interrupted `full_20260926_01` when final gate-only edits raced its recorded source fingerprint. Preserved its incomplete report, interruption reason, raw logs, and artifact manifest. No algorithm failure is inferred from that interruption. Relaunched from fixed scientific sources as `full_20260926_02`; code changes are frozen for its duration.
- Added `STOKES_RUNBOOK.md` with concrete preflight/calibration/budget/submission/retrieval commands. Live Stokes facts still require the user's evidence; no limits, account, balance, or SSH destination were guessed.
- Wrote `research/RELATION_SUPPORT_AUDIT.md` before frozen outcomes. Eight A→A tensors are unused and twelve singleton-neighborhood attention tensors have zero learning gradients in source training. Existing diagnostics and the 30-update source smoke confirm the expected unchanged tensors. Larger teams activate different previously unsupported computations; this limits causal interpretation of a future transfer gap without changing any condition.
- Copied and SHA-verified the four already-read protocol papers (Howell, CASH, Agarwal, Lowe) into this project's `research/papers/`, with a manifest. Original `marl-comm` library files were read only.
- Committed `ANALYSIS_PLAN.md` before any frozen evaluation or bank outcomes: initial `4b4c0a6`, pre-outcome clarification `0f86729`. Fixed the 54,000 required episode inventory, final checkpoint selection, seed-level estimands/intervals, event timing, crash rules, and interpretation limits. The clarification preserves the task's mandatory fixed-oracle Gate B check on intact final banks while keeping learned-policy development on separate smoke banks. Production Phase B code remains unimplemented until Run1 proceeds.

## 26 September 2026 — full local Gate A passed

- Attempt `full_20260926_02` finished at 04:45:33 UTC with all eleven gate flags true. All twelve smoke runs completed: 36 epochs, 360 updates, 32,548 joint transitions and 36 checkpoints. The 22 selected pytest cases passed; the three direct gradient diagnostics passed separately.
- Actual 2P1A/4P6A fresh-gradient reconstruction had exactly zero aggregation error on all three updates, with current storage and shared weights. Independent read-only review recomputed the six saved NPZ reductions (432 named-parameter/update checks), verified snapshot hashes, and found all 36 checkpoint model states finite.
- Source seed0 single-process repeats matched initial/all three epoch signatures and all non-timing metrics. Source seed1 initialization differed. All 36 checkpoint states matched their sidecars; the source epoch3 state loaded identically into all five compositions, preserving mixed dtypes. These checks do not establish frozen forward behavior or performance.
- Archived unchanged text artifacts and a SHA-256 manifest for all 168 raw files at `evidence/gate_a/full_20260926_02/`. Raw checkpoint/gradient binaries remain in the ignored original run directory. Wrote `GATE_A_REPORT.md`; marked the historical gradient blocker resolved without removing failed evidence.
- Scientific source hash stayed `ebd295d990637b9b6184dbf8bad846c8ec70621743aeadedba31379b41c96db4` throughout the successful attempt and archival. Subsequent documentation/evidence commits retain equivalent scientific files to `2db8a62`.
- Next dependency is the previously requested live Stokes evidence. Calibration and its concrete budget cannot proceed without those facts. No Stokes jobs, full training, Gate B or frozen outcomes exist; the overall goal remains active.

## 26 September 2026 — Mac fallback assessment

- Responded to the proposed M4 Pro fallback by checking the current host: Apple M2, 8GiB, four performance/four efficiency cores. Preserved raw hardware command outputs at `evidence/local_mac/current_host_20260926.json`; existing Gate A timings are not M4 benchmarks.
- Verified the pinned PyTorch v2.2.1 MPS source rejects float64; recommended the already-tested CPU path, preserving original mixed precision and four collectors. No MPS port, dtype conversion, process-count or learner change was made.
- Prepared `MAC_RUNBOOK.md`: target validation, full-recipe endpoint calibration, a proposed three-seed source learning pilot, frozen diagnostic development, local launcher requirements, prospective budget formulas and interpretation limits. The full 21-run/2,000-epoch protocol and preregistration remain unchanged.
- Akki selected short pilots followed by measured-cost review and confirmed the separate machine is available all day. Reported base M4 Pro/16GB is inconsistent with Apple's listed Pro24GB/M4-16GB configurations, so target hardware detection is required. Started separate local setup/calibration launcher work; no original model or training-recipe change is needed. The proposed 300-epoch learning pilots remain for decision after measured-cost review. No long-running job was launched on the M2 host.
- Completed separate `local_setup.py` and `local_job.py`, plus their tests. Setup preserves a fresh locked environment's import/hardware/hash evidence. The launcher supports only the two 20-epoch endpoints, enforces one job per user and current target-bound Gate A, records process/memory/timing observations, terminates the child process group on interruption/timeout, and verifies final model metadata against fresh Gate A with exact checkpoint signatures.
- Independent review caught and corrected an initial validator's incorrect all-float64 assumption before use, plus setup signal cleanup and target-evidence binding. The actual original float64/float32 mixture is preserved. No original training/model source was changed for the Mac fallback.
- Combined validation: 47 local/setup/recorder/grid/preflight tests passed; three full-gate artifact checks intentionally deselected because new target smokes do not exist. Six separate gradient regression tests passed in 20.86 s. An actual import/interface probe in the existing M2 environment confirmed setup/launcher agreement on runtime paths, library hashes and hardware identity; this is not fresh-environment or M4 evidence. Validation evidence is archived under `evidence/local_mac/tooling_validation_01/`.
- Prepared a copyable Git bundle and `MAC_RUNBOOK.md` with target setup, tests, full Gate A and sequential endpoint commands. Each calibration has a declared two-hour child-runtime limit plus cleanup/validation time; this is a planning cap, not a runtime prediction. Longer learning runs, frozen-study results and any full-study scope amendment remain deferred to measured-cost review.

## 28 September 2026 — Stokes restored; full Run 1 handoff

- Akki restored Stokes as the target and requested the complete 21-task sweep, progress metrics, submission script and a current hardware-based timing estimate. This supersedes the Mac pilot target; the original recipe, calibrated-budget review and scientific gates remain in force.
- Added a read-only progress command with all 21 expected seeds, episode-weighted success/steps/P and A rewards, losses/counts/throughput in JSON, per-run ETA, fixed source review windows, and exportable SVG/CSV/JSON snapshots. It neither constructs a policy nor changes training state. Missing, failed, stale and malformed evidence are reported explicitly; no learning verdict or automatic cancellation is inferred.
- Added `slurm/submit_run1.sh` and its stdlib driver: review by default, explicit submission, resources derived from genuine confirmed calibration evidence, explicit verified concurrency, preservation of submission records and rejection of duplicate/output reuse. No scheduler submission has been executed here.
- Independent reviews checked the submission path and diagnostic weighting. Corrected wrong-type JSON isolation before freezing sources. The broader pre-gate suite passed 68 tests (3 artifact tests deliberately deselected); all 25 monitor/submission tests then passed after four additional malformed-JSON cases. Gate A now includes the new and Mac infrastructure contracts. A fresh full Gate A will bind the delivery to the complete current inventory; old passes are not relabeled.

- During the refreshed gate, Akki supplied live Stokes terminal output: normal has 153 nodes/6,928 CPU slots, MaxTime=UNLIMITED, account cenyioha has 80,000 September CPU-hours remaining and the user MaxJobs is 250. Saved raw output and a deliberately incomplete preflight under `evidence/stokes/preflight_20260928/`; inherited QoS/account limits, current occupancy and setup prerequisites remain unresolved.
- Deliberately interrupted `full_20260928_stokes_01` after the live facts exposed numeric-only partition-cap validation. The attempt and its interruption reason are preserved as INCOMPLETE, not an algorithm failure. Added exact `"unlimited"` partition-cap support across preflight, allocation, budget and submission; finite explicit job time remains mandatory. Twenty-six focused tests passed. Commit subject `infra(C): accept verified unlimited Stokes partition time` contains only this orchestration compatibility change and tests. Restart full Gate A from its frozen sources as `full_20260928_stokes_02`.
