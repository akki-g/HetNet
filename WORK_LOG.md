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
| A2 baseline diff | Passed precheck; included in full gate | A/B/G/C commit inventory and preserved exact diff |
| A3 five compositions × one/four-process smoke | In progress | `runs/gate_a/full_20260926_02/`; no overall pass yet |
| A4 shared weights and gradient aggregation | Passed bounded diagnostics | Actual 2P1A/4P6A, three updates, exact fresh gradient sums and worker storage |
| A5 single-process determinism | Pending | Initial/first-epoch signatures and deterministic metric projection; different seed differs |
| A6 strict cross-composition load | Pending | Five signatures, singleton-P rejection |
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
