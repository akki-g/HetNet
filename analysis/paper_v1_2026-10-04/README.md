# Paper-v1 implementation evidence — 4 October 2026

This folder records engineering validation of the new paper-aligned HetNet
reconstruction and its optional plain-PyTorch message aggregation. It contains
no Stokes timing result, learned-performance claim, or recovered publication
checkpoint. See [the mathematical contract](../../publication_reconstruction/FIDELITY.md)
and [AGENTS.md](../../AGENTS.md) for decisions and the implementation record.

## Reference and source identity

- `initial_status.txt` preserves the pre-existing dirty tree. Earlier agents'
  timers, local launcher, reports and analysis outputs were preserved.
- `baseline/` and `baseline_manifest.json` freeze all 38 pre-change runtime files
  and five outer source/provenance files. This is the legacy behavioral reference.
- `provenance_01/` records the first integrated runtime; `provenance_02/` records
  the replay-timing and non-finite-comparison fix. Both use the existing
  `scripts/record_publication_sources.py` audit mechanism, retaining old origins
  and a reviewable patch. Final ORIGINS SHA256:
  `d79894723efbf40812232e53e5cf684eacecbdf8916ad4012fb1cfe20410589a`.

## Completed evidence

| Record | Meaning and limit |
|---|---|
| `pytest_full_01.txt` | 610 tests passed in 278.26 s before final replay/benchmark hardening; all existing regressions included |
| `pytest_final_focused.txt` | 173 tests passed in 10.72 s after that hardening; replay, benchmark, launchers, artifacts and study checks |
| `legacy_environment_replay.json` | 30 fixed-action traces, 600 steps in each version; exact observations, rewards, physical state and NumPy RNG on old paths |
| `integration_validation.json` | 24 fresh archived CLI runs, 72 updates and 24 frozen evaluations; PP/PCP/FC × Real/Binary × one/four collectors × both backends; horizon 6, floor 7 |
| `integration_numerical_recheck.json` | All 12 pairs pass final model/Adam tolerances with exact counts and final RNG; largest model absolute difference 4.659640728821302e-13 in float64, zero in float32 |
| `benchmark_ledger_validation.json` | 12 real archived four-collector episode/update/checkpoint/resource ledgers passed; missing trajectory digests correctly require replay |
| `continuation_validation.json` | Exact archived same-backend recovery for DGL/Torch PCP Binary, four collectors; model, Adam, recorder, episode order and all RNG streams; selected checkpoints unchanged |
| `continuation_artifact_manifest.json` | 298 hashes covering the continuation script, report, generated archives, logs and checkpoints |
| `slurm_validation.json` | 39 real batch-script/CLI dry runs with mocked module/srun commands; no submission or training output |
| `production_shape_02.json` | Eight production-horizon, four-collector, floor-500 updates: 18,766 steps / 206 episodes; two additional tiny shared/banked SoftRole FC runs and frozen evaluations |
| `production_shape_02_numerical_comparison.json` | Four production-shaped pairs pass model/Adam tolerances with identical final RNG, counts and episode summaries; complete trajectories are not recorded |

The full suite and final focused suite overlap; their counts must not be added.
The three additional study-provenance tests passed separately in 0.76 s.
The continuation checks executed ten updates, 480 steps and 80 episodes across
four segments; final continuation counters include their retained parent prefix.
Training and validation clocks here were observed with possible concurrent test
activity and are not backend speed measurements.

The first production-shape attempt completed one valid PP Real DGL update
(2,194 steps / 32 episodes), then its harness incorrectly requested continuation
eligibility for an exhausted epoch budget. `production_shape_01.json` preserves
that harness error; `production_shape_01_completed_run_audit.json` independently
validates the checkpoint. The corrected script and a fresh attempt were used;
no trainer/runtime fix was necessary. Include this extra update when accounting
for all engineering work. Each tiny SoftRole check executed one update / eight
steps / two episodes and one frozen evaluation, with identical archived simulator
bytes across methods. These checks do not establish task quality.

## Reproducible bounded checks

Run from the repository root with fresh output paths:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python analysis/paper_v1_2026-10-04/validate_integration.py \
  --output runs/paper_v1_validation_recheck
.venv/bin/python analysis/paper_v1_2026-10-04/compare_integration.py \
  --runs runs/paper_v1_validation_recheck --output numerical_recheck.json
.venv/bin/python analysis/paper_v1_2026-10-04/validate_slurm.py \
  --output slurm_recheck.json
.venv/bin/python analysis/paper_v1_2026-10-04/validate_continuation.py --help
.venv/bin/python analysis/paper_v1_2026-10-04/validate_production_shape.py --help
```

The fixed-rollout replay's executable module is
`publication_reconstruction.runtime.hetnet_ext.paper_replay`; it records actual
DGL rollouts, verifies replay including final optimizer state, compares Torch,
then times alternating serial compute passes. It includes RNG restoration and
Python iteration and excludes environment and interprocess work.

The Stokes command and rollout gates are in the
[reconstruction README](../../publication_reconstruction/README.md).
The benchmark fixes four collectors, 500-step floors, production horizons,
three alternating pairs and 20 updates per run. Optional phase diagnostics are
separate. The official 100-update preflight remains a separate gate.
