# PCP timing and latest HetNet training evidence — 8 October 2026

The new shared-Adam run records support about **15.3 hours of timed epoch-loop work per seed**. They do not save a job start/end clock, Slurm accounting, hostname, CPU model, resource profile, or total process clock. The stored times exclude source/setup before the loop, epoch summary writes/stdout, checkpoint writes, shutdown and queueing. Therefore they support the reported sub-day experience but do not independently prove total elapsed submission-to-results time. All three have a final checkpoint and `finished.json` after 20,000 updates; root checkpoint/episode auditing is separate.

## Recorded throughput

Rows use sums of the same epoch timer. Old shared seed 0 is incomplete. The latest older RMSprop ledgers come from stdout; available structured metrics are checked as exact prefixes. Old shared seed 2 lacks its newer run ledger locally, so its saved configuration and original nested run metadata are used with the stdout mapping.

| Model/optimizer | Seed | Epochs | Steps | Episodes | Timed hours | Steps/s | Last 100 epochs: steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| shared_adam | 0 | 2,000 | 40,626,850 | 4,922,687 | 15.323 | 736.5 | 5.4564 |
| shared_adam | 1 | 2,000 | 40,578,698 | 5,205,340 | 15.371 | 733.3 | 5.3945 |
| shared_adam | 2 | 2,000 | 40,616,881 | 5,096,854 | 15.314 | 736.8 | 5.3526 |
| shared_rmsprop | 0 | 1,655 | 33,669,601 | 3,802,925 | 47.964 | 195.0 | 5.8434 |
| shared_rmsprop | 1 | 2,000 | 40,557,314 | 5,274,501 | 24.670 | 456.7 | 5.4370 |
| shared_rmsprop | 2 | 2,000 | 40,586,013 | 5,191,834 | 24.870 | 453.3 | 5.4223 |
| banked_rmsprop | 0 | 2,000 | 40,789,296 | 4,064,676 | 39.126 | 289.6 | 6.1153 |
| banked_rmsprop | 1 | 2,000 | 40,741,476 | 4,228,149 | 40.143 | 281.9 | 5.9004 |
| banked_rmsprop | 2 | 2,000 | 40,695,846 | 4,268,674 | 32.675 | 346.0 | 6.1269 |

Compared with completed RMSprop shared seeds 1/2, new Adam shared seeds 1/2 have **1.606× / 1.625× higher recorded steps/s**, at nearly equal total steps. The much slower incomplete older shared seed 0 is a separate runtime outlier, not evidence of a universal 3.8× change. The shared/banked timings include different architectures and must not be attributed to optimizer choice.

## What changed, and what is not measured

- **Episode logging is batched:** old `train.py` calls `append_json` once per episode, opening and closing the Lustre file each time; new `log_episode_batch` opens it once per update. For the actual new datasets this means 20,000 opens instead of 4,922,687 / 5,205,340 / 5,096,854 hypothetical old-style opens, reductions of 246.1× / 260.3× / 254.8× in this operation count. This is a strong plausible contributor to faster throughput, not a measured 246× speedup or a demonstrated share of wall time.
- **The optimizer changed:** RMSprop lr 1e-4, alpha .97, epsilon 1e-6 became Adam lr 1e-4, betas (.9,.999), epsilon 1e-8. The rollout/loss/gradient aggregation and float64 precision remain the same for this configured path. The comparison is historical rather than a same-host paired profiler experiment; Adam cannot be credited with the whole runtime change.
- **The shell configuration changed:** the archived new Slurm route explicitly uses `srun --cpu-bind=cores` and adds OPENBLAS/NUMEXPR thread limits. Both routes request four CPUs/16 GiB and use four collectors/one Torch thread; no recorded node identity establishes identical processors or placement. The older route had no explicit CPU binding, which does not establish its actual binding.
- **Persistent workers are not new:** both archived versions allocate `ProcessPoolExecutor` outside the epoch loop, and both construct a fresh model within each submitted collection job. There is no switch from per-update process spawning to persistent workers in this comparison.
- **The work budget did not shrink:** all new runs finish 2,000 epochs × 10 updates, four collectors with at least 500 environment steps per collector per update, and about 40.6M steps. Shorter episodes create more episode resets and records per fixed-step batch; they do not by themselves reduce the required environment steps. New first-100-epoch throughput is 935–942 steps/s, while last-100 throughput is 720–729 despite episode lengths falling from about 65–67 to 5.35–5.46. More successful policies increased episodes/s, not steps/s.
- Archived `learning.py` is byte-identical. Model changes add default-true `allow_stay`, which leaves this PCP configuration unchanged. Optional paper-environment routing and event diagnostics are inactive here. Parent environment audit checks physical transitions independently. There is no new GPU execution: saved dtype is CPU float64 even though the Torch package version has a CUDA build suffix.

## Latest paper-v1 HetNet wave

These are the **new** `logs_1/hetnet-paper-908316_{3,4,5,9,10,11}.out` runs, not the old reproduction-fast baseline. Their namespace declares paper-v1 model/environment, paper-equations-v1 learner and torch-v1 communication backend. It records native 2P1A, 5×5 grid, vision 2, static prey, horizon 80, four collectors, batch floor 500, Adam lr .001 and a 40M-step target. Namespace source paths point to `runs/paper_study_20261005_v1/hetnet/...`; that research-wave run directory/checkpoints/source manifests are not present in the supplied `stokes_runs` inventory. The log bytes are hashed. Source/configuration identity is therefore a stdout declaration, not an independently verified research-wave archive binding. Earlier preflight/benchmark archives are separate evidence.

| Paper model | Seed | Status | Recorded steps | Updates | Active hours | Latest 100 epoch steps |
|---|---:|---|---:|---:|---:|---:|
| pcp_real | 0 | paused_wall_time | 36,766,831 | 18,117 | 46.001 | 5.1171 |
| pcp_real | 1 | budget_completed | 40,001,350 | 19,842 | 43.797 | 5.1145 |
| pcp_real | 2 | budget_completed | 40,000,054 | 19,821 | 43.848 | 5.1170 |
| pcp_binary | 0 | paused_wall_time | 27,294,921 | 13,330 | 46.002 | 5.1187 |
| pcp_binary | 1 | paused_wall_time | 27,532,726 | 13,496 | 46.002 | 5.1182 |
| pcp_binary | 2 | paused_wall_time | 27,584,667 | 13,686 | 46.001 | 5.1193 |

The 46-hour stops are the declared graceful wall-time pause, not completed 40M experiments or unexplained crashes. Real seed 0 needs about 3.23M more steps; Binary seeds need about 12.42–12.71M more. Resume requires recovering their actual validated paused checkpoints. The final status counts include a partial epoch where present; complete-epoch tables retain their exact separately reconciled endpoint and exclude unreported partial epochs. Real seeds 1/2 have a reported final partial epoch because budget completion records it.

## Performance comparison at a common budget

Primary descriptive window: whole epochs whose cumulative step endpoint lies in (26M,27M], episode-weighted within a seed and equal-weight across independent seeds. Every included method/seed reaches this window. This is training behavior, not fresh frozen evaluation. Endpoints below are not scientifically interchangeable with Table 1 or the existing older 30M frozen panel.

| Method | Seed 0 steps | Seed 1 steps | Seed 2 steps | Equal-seed mean |
|---|---:|---:|---:|---:|
| Shared Adam | 6.3945 | 6.0522 | 5.9098 | 6.1188 |
| Shared RMSprop | 6.3429 | 5.9618 | 5.9812 | 6.0953 |
| Banked RMSprop | 7.6900 | 7.3941 | 7.5175 | 7.5339 |
| HetNet paper Real | 9.7273 | 5.1185 | 5.1185 | 6.6548 |
| HetNet paper Binary | 5.1129 | 5.1154 | 5.1194 | 5.1159 |

**Preserve the Real seed-0 collapse.** Its training was about 5.12 steps with 100% success through 23–24M, then declined at epoch 1198 (24,253,689 steps). Epoch 1200 had 280 episodes × 80 steps and zero successes; epochs 1201–1208 and 1210 also report 80 steps/zero success. The 24–25M, 25–26M and 26–27M windows respectively yield 17.0464 / 34.5862 / 9.7273 steps and 84.0885% / 62.4819% / 95.2573% success. By 27–28M it recovers to 5.1769 steps/99.9964%, and 29–30M gives 5.1131/100%. Value loss rises from .00107 at epoch1197 to 7.3747 at1198 and72.8772 at1199; this coincides with the collapse but does not establish cause. No scientific-wave update/optimizer checkpoints were supplied for causal diagnosis. The last epoch above six steps after20M is1314 at26,767,811 steps.

Raw episode batches independently reproduce all six models’ common-window and latest-100-window episode counts, steps, successes and mean agent returns, plus the worst Real0 epoch1200. Each selected update contains all four collectors. Thus the collapse is not merely a plotted rounding artifact. Latest endpoints show Real mean5.1162 and Binary mean5.1187 versus Adam shared5.4012, but the unmatched training budgets and Real0 instability preclude a universal ranking.

## Integrity finding

Adam stdout seed2 epoch2000 contains `alpha_null=-1.3112684324133872`; its saved `metrics.jsonl` says `+0.3112684324133873`. All other fields/epochs match, and seeds0/1 stdout match the structured ledger entirely. The negative value is impossible for attention mass. This review records the mismatch and uses structured metrics; it does not modify either input. Root episode auditing supplies the independent recomputation.

## Reproduction and evidence boundaries

Run from the repository root:

```bash
.venv/bin/python analysis/pcp_adam_2026-10-08/speed_review/audit.py
.venv/bin/python analysis/pcp_adam_2026-10-08/speed_review/episode_window_audit.py
.venv/bin/python analysis/pcp_adam_2026-10-08/speed_review/build_notes.py
```

`audit.py` hashes six complete PCP paper logs while streaming, parses epoch/status summaries, counts episode-batch update IDs, reconciles cumulative counters and new SoftRole update totals, checks stored mean lengths and archived metric/stdout prefixes, and writes CSV/JSON evidence. `episode_window_audit.py` independently decodes the selected raw batches. `input_hashes.json` covers all inputs consumed by the main timing/source comparison. All outputs are confined to this folder; no input, source, checkpoint, job, training or policy evaluation was modified/launched. No timing sweep was run.
