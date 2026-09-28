# Gate A: current Stokes delivery validation

**PASS**, completed 2026-09-28T18:17:22.881588+00:00. All eleven flags in the [machine-readable report](evidence/gate_a/full_20260928_stokes_02/gate_a_report.json) are true. This refresh validates the Stokes submission/progress tools, unlimited partition-cap handling and retained Mac tooling with the original reviewed A/B/G/C training changes. It is engineering evidence, not successful learning or a frozen-transfer result. No Stokes jobs have been submitted.

## Current source and environment

- Scientific revision: `084302bc697a3498b914866369bcf2fceb16e989`.
- Scientific code/config/test/script SHA-256: `bc4822dda32aa75a4e3f8c1cef389599224002d72327ba436d387a692cc407dd`, identical at start, finish and archival.
- Dependency lock SHA-256: `ba4a086080686baca8f7d490c01c63a4ad1a95619148b39d42412246341bd051`.
- Local environment: macOS ARM64, Python 3.12.13, Torch 2.2.1, DGL 2.1.0, torchdata 0.7.1, NumPy 1.26.4 and Gym 0.26.2, CPU only.
- Raw attempt: `runs/gate_a/full_20260928_stokes_02/`, from 2026-09-28T17:35:13.211208+00:00 to 2026-09-28T18:17:22.881588+00:00 (approximately 42 minutes). The dirty flag records concurrent documentation/evidence work; the scientific source inventory stayed unchanged. Documentation-only delivery commits retain the same tested inventory.
- Clean-environment evidence uses the preserved successful clean synchronization/import for the identical lock and pyproject. The current gate also rechecked installed versions. This is not a Stokes Linux installation test; bootstrap verifies those imports on the compute node before training.

## Checks and scope

| Check | Evidence |
|---|---|
| Runtime, clean lock and baseline diff | Passed; original-file changes remain exactly the reviewed commits `2fadecf`, `7b334c1`, `0cfcea5`, `50d0c37` |
| Infrastructure and scripts | 71 recorder/grid/local/submission/progress contracts plus 4 preflight contracts passed; scripts parse; singleton-P rejected |
| Complete smoke matrix | Twelve runs: five compositions × one/four processes, plus source repeated/alternate seeds; 36 epochs, 360 updates, 32,548 actual joint transitions |
| Metrics and saved states | All epoch metrics finite and counters consistent; all 36 checkpoint states match saved names/shapes/dtypes/value hashes and epoch signatures |
| Multiprocessing gradients | Both actual-model endpoints passed three updates with current shared weights/storage and exactly zero fresh-gradient aggregation error |
| Independent gradient audit | Six NPZ snapshots, 432 tensor reductions / 2,982,672 scalar entries independently reconstructed exactly; shared-weight signatures and clip-norm bounds checked |
| Repeated seed | Source seed0 single-process initial/all three epoch signatures and all non-timing metrics match; alternate-seed initialization differs |
| Cross-composition load | One source epoch3 checkpoint loads strictly with identical tensor names/shapes/dtypes/values into all five compositions |
| Artifact tests | 3 passed; total selected pytest cases across the gate: **78**, with no skips |

The independent audit is [preserved separately](evidence/gate_a/full_20260928_stokes_02/INDEPENDENT_GRADIENT_AUDIT.json). Actual-model snapshots contain already-clipped local gradients: they establish aggregation and post-clip norm bounds, not reconstruction of the original unclipped loss gradients. The analytic surrogate additionally checks its clipping calculation. No multiprocess or cross-platform bitwise determinism claim is made. The sole excluded repeated-run metric is wall time.

The released mixed float64/float32 state, class-wise attention, model, observations, rewards, loss, clipping and stepping RMSprop are preserved. Strict loading is not a frozen-policy forward test or proof of transfer. No resume contract is claimed.

## Reproduce and preserve

```bash
DGLBACKEND=pytorch PYTHONUNBUFFERED=1 uv run --locked --python 3.12 \
  python -m hetnet_ext.gate_a \
  --output runs/gate_a/NEW_UNIQUE_ATTEMPT --run-smokes \
  --clean-environment-evidence evidence/gate_a/clean_environment_20260926T035949Z/evidence.json \
  --allowed-upstream-commit 2fadecf --allowed-upstream-commit 7b334c1 \
  --allowed-upstream-commit 0cfcea5 --allowed-upstream-commit 50d0c37
```

Choose a new output directory. The [current archive](evidence/gate_a/full_20260928_stokes_02/) preserves unchanged text evidence and a manifest of all 170 raw files. Raw checkpoint/gradient binaries remain in the ignored local attempt; their sizes and SHA-256 values are recorded. [ARCHIVE_SUMMARY.json](evidence/gate_a/full_20260928_stokes_02/ARCHIVE_SUMMARY.json) contains per-run counts and local timings. Small-batch Mac timing is not a Stokes cost calibration.

The [26 September pass](GATE_A_REPORT_20260926.md) remains historical evidence for its earlier inventory. The original gradient failure and narrowly approved G repair remain in [GATE_A_BLOCKER.md](GATE_A_BLOCKER.md). The first 28 September refresh was deliberately interrupted when supplied live Stokes facts required explicit unlimited-cap handling; its [incomplete evidence and reason](evidence/gate_a/full_20260928_stokes_01_interrupted/) are preserved. No algorithm failure is inferred from that interruption.

## Next operational step

Use [STOKES_FULL_SWEEP.md](STOKES_FULL_SWEEP.md) and [STOKES_RUNBOOK.md](STOKES_RUNBOOK.md) to finish the remaining compute/storage/capacity preflight, run the two 20-epoch full-recipe calibrations, inspect the measured budget and submit the 21-task array. The supplied account has 80,000 CPU-hours remaining and user MaxJobs=250; those facts do not replace workload measurement. Frozen evaluation, Gate B and study results remain pending. The preregistered analysis at `0f867292b498a637465cc110d4fe5520dcff6709` is unchanged.
