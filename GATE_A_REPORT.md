# Gate A: local engineering validation

**PASS**, completed 26 September 2026 at 04:45:33 UTC. The approved three-call gradient-storage repair is applied as deviation G (`0cfcea5`). All eleven required checks in the [machine-readable report](evidence/gate_a/full_20260926_02/gate_a_report.json) passed. Stokes calibration, full training, frozen evaluation, and Gate B have not run.

This pass applies to the source inventory recorded below (delivery snapshot `8e8c56e`). Subsequent Mac fallback launcher/tests require a fresh Gate A on that revision and target. See [MAC_RUNBOOK.md](MAC_RUNBOOK.md); this historical report is not relabeled as validation of later source changes or M4 hardware.

## Scope and provenance

This is engineering evidence for the original released HetNet-Real/A2C path with the separately recorded A/B/G/C deviations. It does not establish successful learning, frozen transfer, or an experimental result for the paper. The original architecture, class-wise attention, loss, clipping, and stepping RMSprop remain as recorded in the [reproduction ledger](REPRODUCTION_LEDGER.md).

- Scientific source revision: `2db8a627b0ee0a0265caef389f47673c6b7bdcf2`.
- Scientific code/config/test/script SHA-256: `ebd295d990637b9b6184dbf8bad846c8ec70621743aeadedba31379b41c96db4`, identical at gate start and finish and at evidence archival.
- Lock SHA-256: `ba4a086080686baca8f7d490c01c63a4ad1a95619148b39d42412246341bd051`.
- Environment: macOS ARM64, Python 3.12.13, Torch 2.2.1, DGL 2.1.0, torchdata 0.7.1, NumPy 1.26.4, Gym 0.26.2; CPU only.
- Raw attempt: `runs/gate_a/full_20260926_02/`, 04:16:49–04:45:33 UTC. The recorded dirty flag is true because documentation/evidence work was in progress. The exact scientific inventory stayed unchanged; no uncommitted upstream file changes were present. Later documentation commits do not change that inventory.

The complete invocation was:

```bash
DGLBACKEND=pytorch PYTHONUNBUFFERED=1 uv run --locked --python 3.12 \
  python -m hetnet_ext.gate_a \
  --output runs/gate_a/full_20260926_02 --run-smokes \
  --clean-environment-evidence evidence/gate_a/clean_environment_20260926T035949Z/evidence.json \
  --allowed-upstream-commit 2fadecf --allowed-upstream-commit 7b334c1 \
  --allowed-upstream-commit 0cfcea5 --allowed-upstream-commit 50d0c37
```

That output directory already exists and must not be reused. The [runbook](STOKES_RUNBOOK.md) gives the reproduction command with a new output directory.

## Checks and limits

| Requirement | Observed evidence | Scope |
|---|---|---|
| Clean locked environment | Separate clean synchronization and pinned imports passed; evidence hash is embedded in the gate report | Local macOS; Stokes installation/import still requires verification |
| Baseline diff | Exact preserved patch and commit inventory contain only reviewed A/B/G/C changes to original files | Original baseline `bff9f7f`; separate experimental infrastructure is inventoried too |
| Smoke matrix | Five compositions × one/four processes, plus seed-repeat and alternate-seed runs; all twelve completed three epochs | Ten updates/epoch, batch target 40/collector, horizon20; not the full training recipe |
| Metrics/checkpoints | All 36 epoch records finite and complete; cumulative counts consistent; all 36 saved model states match their epoch signatures and checkpoint sidecars, including dtypes | 360 updates and 32,548 joint environment transitions across the twelve runs |
| Shared weights | Main/worker post-step parameter hashes agree in both actual-model endpoint probes | Three updates each at 2P1A and 4P6A, four total processes |
| Current gradient storage | Parent and worker cached gradients retain current storage across all probe updates | Verifies the specific approved G repair |
| Fresh gradient aggregation | Actual-model endpoint probes match independently reconstructed clipped-gradient sums/division with maximum absolute error **0** at each update | Surrogate probe also passes within floating-point tolerance; preserved snapshots permit auditing |
| Single-process determinism | Same seed gives identical initial and all three epoch signatures, plus identical non-timing metrics; alternate seed gives different initialization | `wall_time_seconds` is the sole excluded metric. Raw JSONL files differ in timing; multiprocess and cross-platform bitwise determinism are not claimed |
| Cross-composition load | Final 2P1A single-process smoke checkpoint loads with `strict=True` into all five compositions; every name, shape, dtype and value matches | One source checkpoint tested across five targets; this check is loading, not a frozen forward/evaluation test |
| Scripts/grid | Exactly 21 unique training tasks; calibration mapping and launcher contracts pass; singleton-P rejected; shell syntax passes | `shellcheck` was unavailable; no Slurm allocation or submission was performed |
| Unit contracts | 4 preflight + 15 recorder/infrastructure + 3 artifact tests passed, without skips in the selected tests | Direct gradient probes run separately; resume is not implemented or claimed |

All twelve resolved argument files were also checked to retain dimension5, vision2, observation width725, real-valued CPU execution, per-class critic, and two state nodes. The float64 defaults and explicitly float32 attention parameters remain as released; no whole-model dtype conversion was used for loading.

### Local smoke timings

Each cell lists the three epoch durations in seconds, rounded to one decimal. These are audit details from this Mac, with small batches and horizon20. They must not be used as Stokes budget measurements or as evidence that four processes improve throughput.

| Composition, seed0 | One process | Four processes |
|---|---|---|
| 2P1A | 27.8, 31.2, 30.5 | 117.0, 68.7, 55.6 |
| 3P3A | 32.8, 33.4, 35.2 | 54.8, 67.7, 64.7 |
| 4P6A | 36.0, 34.8, 35.4 | 74.8, 67.3, 66.8 |
| 3P1A | 28.3, 30.7, 28.7 | 56.4, 55.0, 51.7 |
| 2P2A | 31.6, 31.9, 32.6 | 61.0, 61.1, 57.5 |

The source seed0 repeat took 27.4/27.3/27.5 seconds per epoch; source seed1 took 27.5/28.3/28.3. Complete values and counts are in [ARCHIVE_SUMMARY.json](evidence/gate_a/full_20260926_02/ARCHIVE_SUMMARY.json).

## Preserved attempts and evidence

The [original gradient failure](GATE_A_BLOCKER.md) remains documented: shared weights passed, but cached gradient storage and fresh aggregation failed after update1. The user explicitly approved the exact three-call repair. Those failed measurements are retained separately from repaired probes and this full gate.

The first full attempt, `full_20260926_01`, was deliberately interrupted because final gate-only edits raced its source fingerprint. Its [incomplete report and interruption record](evidence/gate_a/full_20260926_01_interrupted/) remain preserved. This is not a failed learning outcome and is not counted as a passed gate.

The [archive](evidence/gate_a/full_20260926_02/) contains unchanged copies of every text artifact: reports, logs, metrics, resolved arguments, tensor signatures, test XML, and exact upstream diff. Its [raw manifest](evidence/gate_a/full_20260926_02/RAW_ARTIFACT_MANIFEST.json) records paths, sizes and SHA-256 for all 168 raw files. All 36 `.pt` checkpoints and six gradient `.npz` snapshots remain in the original ignored run directory; these binary files are manifested rather than duplicated in Git. The raw attempt is approximately 324 MB; archived text is approximately 2.9 MB. Absolute local paths inside evidence are intentionally preserved rather than rewritten.

## Next gate

Gate A now permits Stokes calibration **once live preflight facts are supplied and verified**. The already-requested partition cap, account requirements, available allocation, quota, concurrency, modules and cluster access remain unresolved. Follow [STOKES_RUNBOOK.md](STOKES_RUNBOOK.md) to run the two 20-epoch endpoint calibrations, measure allocation time/whole-job memory/checkpoint size, and present the resulting budget for Akki's confirmation before the 21-task array. No cluster costs have been measured or authorized by this local pass.

The [analysis plan](ANALYSIS_PLAN.md) is preregistered before Run2 outcomes at `0f867292b498a637465cc110d4fe5520dcff6709` (initial commit `4b4c0a6`). Production Phase B remains scheduled for Run1, in a separate checkout. The relation-support limitation identified before outcomes is documented in [RELATION_SUPPORT_AUDIT.md](research/RELATION_SUPPORT_AUDIT.md); Gate A does not resolve it or establish a causal explanation for future transfer gaps.
