# Independent preflight provenance and checkpoint audit

Audited the eight copied runs in `hetnet_paper_preflight_905118` and
`hetnet_paper_preflight_905404` on 5 October 2026. Both groups use **torch-v1**;
these are repeated backend preflights, not a DGL-versus-Torch comparison.

All eight pass the prescribed 100-update engineering integrity gates. Their
source matches the previously reviewed benchmark 904547 and current runtime.
No substantive configuration, checkpoint or frozen-probe validity failure was
found. Reproduce this read-only audit from the repository root:

```bash
.venv/bin/python analysis/stokes_preflight_2026-10-05/provenance_review/audit.py
```

The script writes `review.json` and `input_hashes.json` here, maps absolute Stokes
paths relative to each run's recorded output, and checks all 898 input files
again before finishing. None changed. No training or frozen evaluation was run
locally: the recorded frozen reports were checked against their checkpoint and
archived executable source.

## Gates checked

- Every run completed exactly 100 updates / 10 epochs with four collectors,
  500-step floors, one Torch thread per collector, seed 991 and production
  horizons 80/300. All are fresh `paper-v1` model/environment runs using
  `paper-equations-v1`; profiling was disabled. Update, epoch, episode and
  checkpoint progress reconcile. Episode ordering, collector partitioning,
  floors and horizon bounds pass the full recorded ledger check.
- All eight 53-file training manifests are identical to benchmark 904547:
  `17143ed20265d29e321f96609d4999144a84e748dc0b29ad1bde059c61b49599`.
  Every archived byte hash matches its manifest, the benchmark source identity
  and current reviewed source. Current ORIGINS audit is valid. All installed
  package inventories and lock hashes match the benchmark.
- All actual checkpoint model tensors and active Adam states are finite.
  Moments have correct shapes/dtypes; second moments are nonnegative; every
  active Adam step is 100. Checkpoint SHA256/size, parameter-signature sidecar,
  record ledger, source binding, resolved arguments and protocol agree. All
  recorded numerical losses/metrics/probe outcomes are finite. All four
  collectors' Python, NumPy and Torch RNG states can be restored into isolated
  RNG objects.
- Each recorded frozen probe binds the exact checkpoint bytes and model tensor
  signature, training source, scenario-file bytes and progress. Its 12-file
  evaluator archive and manifest match the report and current evaluator.
  Imported model/environment paths resolve inside the training archive. Each
  probe completed one native-composition episode with parameters unchanged;
  the fail-closed archived evaluator checks this before producing the report.
- Both groups' first 20 update records, episode summaries and epoch-2 parameter
  signatures exactly match the Torch pair-1 prefix of benchmark 904547, excluding
  only the explicit wall-time fields. Their initial model/optimizer/RNG identities
  also match that benchmark.

## Comparison between the two groups

For every workload, the two groups have bitwise-identical final model, active
Adam and all collector RNG states. They also match at every epoch parameter
signature, all 100 update records, all ten epoch metrics, all episode-summary
records and their frozen probe episode. The only excluded numerical-record
fields are `wall_time_seconds` in update/epoch ledgers and
`rollout_wall_time_seconds` in episode summaries.

| Workload | Steps in each group | Episodes in each group | Active Adam parameters | Frozen probe steps / outcome |
|---|---:|---:|---:|---|
| PP Real | 215,817 | 3,118 | 158 | 80 / horizon reached |
| PCP Real | 216,285 | 2,947 | 334 | 70 / success |
| FC Real | 247,718 | 1,098 | 334 | 300 / horizon reached |
| PCP Binary | 216,940 | 2,996 | 382 | 54 / success |

Each group collected 896,760 joint environment steps and 10,159 episodes.
Unused parameters, including PP's empty A-class paths, legitimately lack Adam
state. Different checkpoint byte hashes are expected because saved paths and
timing metadata differ; all numerical training-state identities match exactly.

The frozen episodes are engineering loading/execution checks. The two successful
and two horizon-capped outcomes do not rank methods or establish convergence.
No complete action-trajectory digests were recorded, so equal summary ledgers
and RNG states are not described as a full trajectory proof. These two same-seed
groups are not independent scientific seed replicates.

The 100-update and strict frozen-checkpoint gates are now satisfied for Torch.
Research runs should start fresh with the locked study budgets. These preflight
checkpoints exhausted their ten-epoch engineering budgets and are not the start
of the full research runs. Hardware and long-run resource planning are assessed
separately; timing differences here cannot be attributed to a backend change.
