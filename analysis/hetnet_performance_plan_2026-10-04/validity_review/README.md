# Coordination and experimental-validity review

Date: 4 October 2026. This is an implementation review of the existing
[performance plan](../../../publication_reconstruction/PERFORMANCE_PLAN.md), not
a competing acceleration proposal or a new timing experiment.

The repository search found one current performance plan. The older
`analysis/hetnet_restart_2026-10-03/RESTART_PLAN.txt` is deployment history;
`docs/plans/PLAN.md` and `docs/legacy_reproduction/ANALYSIS_PLAN.md` have different,
older scientific scopes. The current plan already contained contributions from
both earlier work streams. Only the current thread's three reviewers were
reachable through the agent tools. Direct contact with the other session's
author has not been established; the existing plan and this record provide the
shared handoff. No claim of approval by that other author is made.

## Reproduced issues and changes

These are validation failures under deliberately altered inputs, except for the
small-configuration NaN below. They are not evidence that the supplied research
archives are corrupted. All changes sit outside the archived training runtime.

| Issue | Evidence before the fix | Implemented boundary and regression evidence |
|---|---|---|
| Direct continuation accepted altered checkpoint bytes | Temporary copies of a genuine paused checkpoint were accepted after changing a finite weight, an Adam step from 1 to 999, a moment shape, or deleting an active parameter's optimizer state. The prior validator checked source identity but did not bind checkpoint bytes to the run's checkpoint ledger. | [`artifacts.py`](../../../publication_reconstruction/artifacts.py), `_validate_recorded_checkpoint`, binds continuation and preflight readiness to recorded bytes, length and progress; `_validate_optimizer` checks existing moment shapes/dtypes, second-moment sign and step counters. Unused relation parameters are allowed to lack optimizer state. [`test_publication_artifacts.py`](../../../tests/test_publication_artifacts.py) and [`test_publication_recovery_integrity.py`](../../../tests/test_publication_recovery_integrity.py) cover these boundaries. |
| Saved scientific defaults could disagree with the declared recipe | In-memory copies of the genuine PCP Real resolved arguments passed after changing `gamma`, `second_reward_scheme`, `nenemies` or communication range because those settings were absent from the command's explicit flags. | `_validate_scientific_protocol` checks the recipe's relevant saved defaults as well as explicit flags. The recipe remains unchanged. [`test_publication_validity.py`](../../../tests/test_publication_validity.py) covers domain defaults and drift. |
| Parent checkpoint could be labeled with the requested seed | A dependency-stub test returned a parent checkpoint with seed 2 for a requested seed 0. The selector checked the terminal run but not the selected parent before assigning its label. | [`study.py`](../../../publication_reconstruction/study.py), `select_checkpoint`, validates the selected run against the locked protocol and checks the checkpoint's actual seed. Valid parent selections still work. The tests separate rejection of a mismatched parent from acceptance of a matching parent. |
| A 100-row ledger was enough to pass the readiness summary | With the genuine Mac PP checkpoint and an in-memory ledger containing 100 copies of update 1, the previous summary accepted 216,500 ledger steps against 213,241 status steps and reported 291.91 steps/s instead of the actual 392.08. Probe execution and output writes were mocked. | `_validate_preflight_ledger` reconciles absolute update order, epoch positions, integer work counts, cumulative totals, finite timing and checkpoint/status progress against the segment's starting counts. [`test_publication_study.py`](../../../tests/test_publication_study.py) covers malformed and valid ledgers, including a continuation segment. |
| A supported-looking tiny smoke configuration produced NaNs | PCP/FC with horizon 1 and batch floor 1 give each collector's single A agent one advantage value. The preserved sample standard deviation is undefined for that singleton; the actual `batch_finish_per_class` path produced a nonfinite loss and gradients. | [`__main__.py`](../../../publication_reconstruction/__main__.py), `resolve`, now rejects that combination before creating a run. [`test_publication_launcher.py`](../../../tests/test_publication_launcher.py) checks both specifications and one/four collectors, plus neighboring valid configurations. The public learner's standard deviation and learning formulas are preserved. |

Before the default/parent-selection fixes, the new focused validity tests produced
8 failures and 4 passes. All twelve passed after the fixes. Final test commands,
results and actual-archive checks are recorded in [validation.json](validation.json);
passing synthetic tests alone does not certify the original study. The full
suite passed 524 tests in 150.64 seconds. After adding the final preflight
checkpoint-byte binding call/test, the affected guard suite passed 171 tests in
1.45 seconds. The two scopes and their logs are kept separate.

[check_archives.py](check_archives.py) accepted 33 existing checkpoints across
the two specifications and all three domains, all eight Mac/Stokes preflight
ledgers at exactly their original rates, and two genuine paused checkpoint
resume selections. It checks byte identities before and after loading; it does
not execute a policy. [probe_singleton.py](probe_singleton.py) independently
reproduced the nonfinite loss/gradients in all eight synthetic PCP/FC ×
Real/Binary × model-specification cases using the frozen source. These probes
perform forward/backward but no environment or optimizer steps. The full test
suite includes bounded engineering training/evaluation fixtures. No research
run, performance benchmark, dependency change or cluster submission was made.

## Preserved baseline and provenance

`baseline_manifest.json` identifies 54 source/test files copied before any of
these guard changes, with every hash matched to the earlier performance-plan
inventory. This includes all 38 runtime files. `prior_evidence.json` and
`prior_performance_plan.md` preserve the previous evidence and plan. The explicit
source-revision receipt links changed inventory entries to their old bytes and
new hashes. The verifier checks this transition; rehashing is not treated as
evidence that an unexplained change is acceptable.

The environment, wrappers, policy/model/graph code, learner, mixed tensor dtypes,
random draws, collector partitioning and active optimizer settings are unchanged
by this review. The pre-existing episode timers and local launcher are retained.
`ORIGINS.json` covers the runtime and still validates; no runtime provenance
refresh is needed for these outer guards. New run manifests already capture
`artifacts.py` and `study.py`, and new evaluator snapshots capture `artifacts.py`,
so new evaluations correctly acquire a distinct evaluator identity. Existing
archives and checkpoints are untouched.

## Remaining limits and next action

The speed solution remains the existing exact-first plan: profile PCP Real and
Binary on the same compute-node allocation, select worthwhile graph/model
preprocessing candidates, and require exact component/update/recovery checks
against an immutable reference before a paired unprofiled throughput comparison.
`cached-v1` is still a proposed interface; this review implements no accelerator
and supplies no new speedup estimate. The Stokes Binary slowdown remains
unexplained. The original-study comparison retains its budgets, frozen panel and
checkpoint rules, richer typed HetNet inputs, and documented reconstruction
limits, especially the unmatched FC environments.

The plan's three candidates have a sound source-level rationale: the selected
unlimited, non-lossy graph has position-independent connectivity; the layer
forwards repeatedly obtain the same relation views; and the feature loops have
an explicit reshape/slice index identity. None of those facts measures its
fraction of update time. Keep the profiler-first selection, fresh feature scopes
and unsupported-layout/range fallbacks. In particular, caching topology does
not make position-dependent distance features constant; the fast path must
exclude consumers of those features or refresh them.

One interface detail needs an explicit future regression: the current artifact
validator regenerates the training command with the current `resolve` function
and compares scientific flags. Once `resolve` adds `--execution_mode`, older
commands lacking that flag must still canonicalize to `reference` without
passing a new flag into old archived sources. Test actual old checkpoints and
continuations, not only new mode-aware fixtures. Profiling flags likewise need
a declared runtime-metadata boundary. Otherwise an apparently harmless new
default can break frozen loading even when the model's constructor is backward
compatible.

The proposed benchmark's work is explicit: four workloads × three pairs × two
modes × twenty updates = 480 updates, or 960,000–1,217,280 environment steps with
the production collector floors and horizons. Profiled runs and the four full
readiness preflights are additional work. Three alternating pairs retain an
order imbalance; report it and keep inconclusive outcomes. This design can
support a workload-specific implementation throughput result on that allocation,
not a claim that the large Binary slowdown has been explained or that the
scientific methods have equal learning performance.

Separate report-input hardening remains deferred. In
[`softrole/report.py`](../../../softrole/report.py), `summarize` trusts an
episode's composition for roster normalization, permits incomplete scenario
cross-checks and boolean-coerces success. Synthetic malformed inputs admitted
composition `[99,1]` under a declared `[2,1]` scenario (changing mean-agent return
from 1 to .03), a conflicting episode/report sham flag, and string success
`"False"`. All four supplied Stokes checkpoint-probe reports pass those specific
consistency checks. No report parser or research result was changed here.

Operationally, the local preflight launcher can run four workloads concurrently
(16 collectors); that is unsuitable for an isolated four-CPU timing comparison.
Its parent-recorded progress timestamps include stdout drain time. The planned
paired harness therefore remains sequential and uses producer-recorded update
times, with startup and profiling overhead reported separately.
