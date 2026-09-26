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
| A1 clean lock/import | Pending | Clean uv sync and pinned-version import log |
| A2 baseline diff | Pending | Per-deviation commits and exact diff inventory |
| A3 five compositions × one/four-process smoke | Pending | Ten unique three-epoch run directories, finite metrics/checkpoints |
| A4 shared weights and gradient aggregation | Pending | Worker hashes after update; ≥2-update gradient storage/contribution diagnostic |
| A5 single-process determinism | Pending | Initial/first-epoch signatures and deterministic metric projection; different seed differs |
| A6 strict cross-composition load | Pending | Five signatures, singleton-P rejection |
| A7 scripts/dry-run bijection | Pending | Shell syntax checks and all 21 mappings |
| A8 resume | Conditional | Only implement if calibrated time exceeds cap; then full kill/resume test |
| Stokes preflight | Awaiting user evidence | Limits/account/balance/modules/network/quota/access |
| Stokes calibration and budget | Pending after A | Two 20-epoch jobs; resource/timing table; explicit user confirmation |
| Run1 / early learning gate | Pending | 21 final policies, all source learning curves, review around epoch300 |
| Phase B / Gate B | Not started | Frozen runner, banks, signatures, pairing, sensor/message/permutation/oracle tests |
| Analysis preregistration | Pending before Run2 outcomes | Committed ANALYSIS_PLAN.md and recorded hash |
| Run2 | Pending after Run1 and B | Complete checkpoint preflight, frozen evaluation logs |
| Analysis / paper handback | Pending after Run2 | Reproducible tables/figures, RESULTS.md, evidence-mapped PAPER_SKELETON.md |

Full objective remains active; a prepared workflow or smoke pass is not completion of the cluster experiments.
