Independent PCP Adam training-artifact audit, 8 October 2026

Reproduce from repository root:
  .venv/bin/python analysis/pcp_adam_2026-10-08/trial_review/audit.py --episodes
The script writes only this directory. It reads training artifacts without
training, policy evaluation, checkpoint mutation, or installing dependencies.
The optional full episode pass streams approximately 14 GB and uses three
analysis workers. These are JSON-audit workers, not RL collectors.

FILES
  review.json: run mapping/configuration, ledger/checkpoint/source reconciliation,
    clipping counts, source-file comparisons and equal-seed window summaries.
  windows.csv: episode-weighted per-seed training windows, exact retained epoch
    and sample bounds, successes/failures, times and diagnostic metrics.
  group_windows.csv: equal-weight training-seed means and descriptive seed ranges.
  epoch_metrics.csv: normalized source epoch rows for plots; new Adam rows use
    the structured metrics, not the one malformed stdout scalar below.
  thresholds.json: first crossing of a trailing-100-epoch episode-weighted mean;
    post hoc descriptive learning thresholds, not a model-selection criterion.
  checkpoints.json: all 120 new snapshots; actual model signatures, complete Adam
    moment/step/shape/dtype/finiteness checks, scalar optimizer statistics and hashes.
  old_terminal_checkpoints.json: newest copied checkpoint per older PCP run;
    actual RMSprop moment/count checks. Shared seed2 has no copied checkpoint.
  input_hashes.json: exact bytes for small ledgers, logs, configs, checkpoint files,
    signatures and source archives. Full episode hashes are in episode_audit.json.
  episode_audit.json and episodes_seed*.json: every new raw episode validated,
    all 60,000 update and 6,000 epoch aggregates reconciled, length histograms,
    final-epoch independently recomputed metrics and exact raw-file hashes.
  console.txt and metrics_console.txt: audit execution receipts.

KEY FINDINGS
The three 911488 files are fresh PCP SHARED seeds 0,1,2, not banked CapCom trials.
They use Adam lr=1e-4, betas=(.9,.999), eps=1e-8, four collectors, floor500 per
collector, 2,000 epochs x10 updates, fixed2P1A and zero configured sensor failure.
All finish at20,000 optimizer steps. Together they contain121,822,429 environment
steps and15,224,881 episodes. There are40 verified checkpoints/seed. All918
archived source files match their manifests, and all three manifests share
source ID4c5dfbd6b9d519d99c999c638abb2d9f0679293e90d7ad235a099ea18a9e876e.
The source records commit692b88d3634920bf7c4b348840344c0ef21b410d with542
analysis-file deletions in the recorded worktree status; runtime modifications
are not present in that status. Exact archived bytes, not commit alone, identify it.

All raw episodes obey the fixed roster, horizon, no-event settings and payload
accounting; finite returns sum to recorded team return. Scenario identities are
unique within each update/collector. All updates contain four collector batches
meeting the500-step minimum. Every attention-null probability is in[0,1].
All60,000 update gradient norms exceed.75 (range14.6857 to1687.5245), so every
update is clipped. The optimizer receives clipped gradients; raw norm does not
measure update magnitude. Every model/optimizer tensor is finite at every saved
snapshot; terminal optimizer counters are20,000. Last100 epochs contain3/0/0
failed episodes across368,302/372,525/375,431 episodes in seeds0/1/2.

ONE INPUT DISCREPANCY
Seed2 stdout epoch2000 alpha_null is-1.3112684324133872, an impossible probability.
Its structured metrics value is+0.3112684324133873. Independently summing every
raw episode in that epoch gives+0.3112684324133872 (ordinary rounding difference).
All other fields and rows match stdout exactly. The raw and structured values
are authoritative for this analysis; inputs are preserved. The artifacts do not
identify the cause of the altered stdout scalar. It is not evidence of negative
attention in the trained model.

COMPARISON
The older RMSprop mapping is logs_sr/softrole-898818_6..8 shared seeds0..2 and
_9..11 banked seeds0..2. Available structured ledgers exactly match stdout.
Shared seed2 has config/source but no structured metrics or checkpoint locally;
its comparison uses stdout with the archived launcher mapping. Shared seed0 stops
at epoch1655/33,669,601 steps; its final saved checkpoint is epoch1650. Other older
PCP logs reach2000epochs. Never treat the incomplete shared0 last100 as a40M tail.

Window membership is lower < epoch-end total_steps <= upper. Entire selected
epochs are retained, so actual_start_steps/actual_end_steps are exported and can
straddle the nominal lower boundary slightly. Means weight episodes within each
seed, then weight seeds equally. Seed ranges are descriptive, not confidence
intervals. The39-40M RMSprop shared group has only seeds1/2; use matched seeds if
contrasting it to Adam, and use26-27M or29-30M for full three-seed comparisons.

At26-27M, mean steps are Adam shared6.1188, RMSprop shared6.0953, RMSprop banked
7.5339. At29-30M they are5.8202,5.8598,7.1047. Adam last100 is5.4012steps with
99.9997285% equal-seed mean success. Adam does not show a general sample-efficiency
gain: its first trailing100-epoch mean <=10steps occurs at12.737/11.448/13.220M,
versus11.260/10.641/11.556M for RMSprop shared. First >=99% success is also later
in all three Adam seeds. The <=6step crossing is mixed. These are training-policy
statistics with changing weights, not held-out frozen evaluation or significance.

ENVIRONMENT AND CAUSAL LIMITS
The physical PCP simulator, learning.py and scenarios.py are byte-identical to
the older source. Inspected diffs retain the same PCP sensing/action/reward/reset
path: new model allow_stay defaults true, new paper-FC paths are inactive, and
optional event diagnostics default off for collection. New mean_agent_return
is a derived metric. The intended scientific change is the fresh Adam optimizer.
Same-seed initial signatures match older shared seeds0/1; seed2's prior signature
is unavailable. Later source also batches episode-log writes per update instead
of opening the file per episode. This is a historical cohort comparison with
other implementation/hardware/launch-era differences, not an isolated wall-time
benchmark of Adam versus RMSprop. Epoch wall sums are15.323/15.371/15.314hours;
they exclude checkpoint saves and some setup/shutdown. No absolute scheduler
elapsed time or CPU model can be recovered from these files alone.

REPRODUCIBILITY OF VALIDATION AND FINAL REVIEW
The main audit.py generates old_terminal_checkpoints.json and thresholds.json.
Its --episodes pass generates final_epoch_recomputed in every episodes_seed*.json
and episode_audit.json by summing raw episode metrics, independent of stdout.
To reproduce the final input-hash recheck and compare every root-report SoftRole
window against this independently computed reference, after running compare.py:
  .venv/bin/python analysis/pcp_adam_2026-10-08/trial_review/verify.py
This reads/hashes the inputs and existing derived records; it does not repeat
checkpoint tensor loading or episode JSON aggregation. It regenerates
validation.json and comparison_verification.json. The prior report_review.json
is a dated human-style review receipt bound to the explicitly listed report
hashes; its numerical comparison is reproducible through verify.py. Its wording
recommendations describe that reviewed snapshot, not necessarily later revisions.
