# Independent provenance and checkpoint review — Stokes 904547

Audited on 5 October 2026 with the copied archive read-only. Reproduce from the
repository root:

```bash
.venv/bin/python analysis/stokes_backend_2026-10-05/provenance_review/audit.py
```

`audit.py` writes `review.json` and `input_hashes.json` here. It maps recorded
Stokes paths relative to each run's recorded output; it does not rewrite any
archive. The audit checks actual checkpoint tensors and optimizer states, not
only benchmark summary fields. It reuses the reviewed read-only artifact
validators, adds independent cross-file bindings, and recomputes parameter,
model, optimizer and RNG signatures.

## Result

All 32 runs passed: 24 unprofiled 20-update runs and eight separately profiled
diagnostic runs. No substantive source, configuration or checkpoint-integrity
failure was found.

- All 68 entries in the benchmark `source_identity.json` match the reviewed
  checkout byte-for-byte, including model/learner/environment implementations,
  the benchmark harness, ORIGINS, fidelity documentation, dependencies and
  Slurm launchers.
- Each of the 32 run archives has the same 53-file source manifest, SHA256
  `17143ed20265d29e321f96609d4999144a84e748dc0b29ad1bde059c61b49599`.
  Every archived payload matches its declared hash and the benchmark identity.
- Every checkpoint's bytes, recorded byte size, parameter-signature sidecar,
  source-manifest binding, saved arguments, scientific protocol, completed
  counts and optimizer settings agree. Model and optimizer tensors are finite;
  Adam moments have the correct shapes/dtypes, second moments are nonnegative,
  and every active Adam parameter has the correct update count. Unused
  parameters legitimately lack Adam state (especially PP's empty A class).
- All runs use `paper-v1` model/environment, `paper-equations-v1`, seed 991,
  four collectors, 500-step collector floors, one Torch thread per collector,
  and horizons 80 for PP/PCP or 300 for FC. Within a pair, the only scientific
  argument difference is `message_backend`. Dependency inventories and Python
  records agree across all runs and with the recorded host.
- The recorded initial model, optimizer and collective RNG identities are
  identical across all eight executions of each workload. The three unprofiled
  repetitions end with bitwise-identical model, Adam and RNG states within
  each backend. Every cross-backend pair has exactly equal final RNG state.
- Progress stdout matches every planned job in order. The hash-verified harness
  waits for each child before starting the next and checks unchanged source
  between jobs. Throughput order is DGL first, Torch first, DGL first; diagnostic
  runs follow all throughput runs. No independent absolute process start/end
  timestamps were recorded, so chronology is supported by control flow and
  progress records rather than a separate scheduler timeline.

## Cross-backend numerical differences after 20 updates

These maxima compare the actual final model and active Adam state. All three
paired repetitions yield the same numbers, because each backend repeats exactly.
All float32 attention parameters are exactly equal. Listed model discrepancies
are float64; Adam has additionally negligible float32-moment discrepancies.

| Workload | Maximum model absolute difference | Maximum Adam first-moment difference | Maximum Adam second-moment difference |
|---|---:|---:|---:|
| PP Real | 7.782e-15 | 1.256e-17 | 5.082e-21 |
| PCP Real | 1.665e-13 | 3.123e-16 | 9.758e-19 |
| FC Real | 3.246e-14 | 9.714e-17 | 6.505e-19 |
| PCP Binary | 3.303e-14 | 1.167e-14 | 3.301e-17 |

All Adam step counters match exactly. These discrepancies are below the
declared local backend tolerances even at the end of 20 updates. This is
additional bounded numerical evidence; it is not a guarantee that all future
training trajectories remain identical. The separately reviewed fixed-rollout
gate remains the appropriate per-operation correctness test.

## Boundaries and remaining gates

1. This benchmark compares two backends of the **new paper-v1 reconstruction**.
   It does not measure a drop-in speedup of the older supplement-v1 learner or
   explain the earlier original preflight's Binary slowdown by itself.
2. The copied training records do not include full action-trajectory digests.
   Equal episode summaries and final RNG states are strong reproducibility
   evidence but do not independently establish every sampled trajectory.
   Initial model/optimizer/RNG values are preserved as signatures rather than
   separate initial checkpoints. The replay fixtures supply separate evidence.
3. The official 100-update preflight and established fidelity/recovery gates
   still precede research rollout. A 20-update benchmark checkpoint is not a
   training continuation or research checkpoint. Fresh runs should explicitly
   select the intended backend; preserve existing archives/continuations.
4. This is one seed on one node (`ec169`, Intel Xeon Gold 5418Y), with shared
   affinity `[0,2,4,6]` for all four collectors. The benchmark requests
   `sapphirerapids`; the generic preflight/train launchers do not. Request the
   same constraint in subsequent Slurm invocations if using this measurement
   to project wall time, and retain recorded hardware/resource evidence.
5. Three repeated runs of one seed assess engineering timing repeatability;
   they are not independent training seeds or evidence of experimental quality.
   Binary bandwidth remains the explicitly chosen 64 bits per head / 256 bits
   total interpretation documented in the reviewed fidelity record.

`input_hashes.json` records 3,118 audited input files, including the 32 complete
run trees, benchmark control records and job stdout/stderr. Replay evidence is
reviewed separately by the learner audit. `review.json` records every run,
parameter/optimizer inventory, per-dtype discrepancy and paired identity.
