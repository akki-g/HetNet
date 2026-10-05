# Stokes backend results and next steps — 5 October 2026

**Recommendation: select `torch-v1` as the candidate for fresh paper-v1 training,
run the separate 100-update preflights on Sapphire Rapids, then launch the locked
study if those pass.** All twelve unprofiled timing pairs favor Torch, all four
fixed-rollout compute gates pass, and independent source/checkpoint/resource
reviews found no substantive blocker. Further message-passing optimization is
not needed before that next gate. This is engineering evidence for the explicitly
declared reconstruction, not evidence of published-score reproduction or policy
quality. The selected Binary convention remains 64 bits per head, 256 per round.

The input is [job 904547](../../stokes_runs/hetnet_backend_benchmark_904547/summary.json).
It completed 24 unprofiled runs (480 updates / 1,074,342 steps / 12,330 episodes),
eight separate diagnostic runs (24 updates / 52,100 steps / 610 episodes), and
four fixed-rollout replay workloads. The training runs total 504 updates and
1,126,442 steps. The four replay fixtures contain another 26,050 recorded steps
and 305 episodes; replay timings repeatedly compute those same fixtures and must
not be counted as newly collected environment samples. Summed reported child
process times, including replay preparation/checks/timing, are 2.32 hours; this
does not include Slurm queueing or measure the complete job's accounting time.
[Recomputed metrics](metrics/summary.json), [replay audit](replay_review/review.json).

| Workload | DGL steps/s | Torch steps/s | Median paired throughput gain | Paired range | Less update time |
|---|---:|---:|---:|---:|---:|
| PP Real | 366.3 | 506.0 | 37.7% | 37.7–39.2% | 27.4% |
| PCP Real | 203.8 | 275.3 | 35.1% | 34.4–35.6% | 26.0% |
| FC Real | 212.7 | 298.3 | 40.1% | 40.0–40.2% | 28.6% |
| PCP Binary | 155.8 | 199.6 | 28.2% | 27.7–28.3% | 22.0% |

Rates are the median of three runs, each using `sum(steps)/sum(update_seconds)`
over updates 2–20. Gains are the median of the three **paired ratios**, which need
not equal the ratio of the displayed median rates. Time saved is `1 - 1/speedup`;
40% more throughput is approximately 29% less time, not 40% less time. Ranges are
observed timing ranges, not confidence intervals. The three repetitions use the
same seed and trajectories as far as the recorded evidence can establish; they
are not three independent training seeds. The ratio of median full-segment rates
also favors Torch: 1.383× / 1.353× / 1.406× / 1.283× respectively.
[Raw-ledger recomputation](analyze.py), [per-run values](metrics/throughput.csv).

**Why this comparison is credible.** All jobs ran sequentially in one allocation
on `ec169`, an Intel Xeon Gold 5418Y machine. The recorded affinity `[0,2,4,6]`
covers four distinct physical cores on NUMA node 0, with one Torch thread per
collector. The allocation was four CPUs and 16 GiB. All 68 benchmark source
identities, all 43 runtime origins and all twelve direct dependency pins match
the reviewed implementation. Each of the 32 archived source trees contains the
same 53 files. Paired scientific arguments differ only in backend. The planned
DGL/Torch, Torch/DGL, DGL/Torch order agrees with stdout and the hash-verified
blocking subprocess implementation; separate absolute process start/end
timestamps were not saved. [Hardware evidence](hardware_review/results.json),
[source/checkpoint audit](provenance_review/README.md).

All 32 checkpoints pass actual tensor/Adam finiteness and structure checks,
completed-update counters, byte/signature/sidecar binding and archived-source
validation. Paired initial model/optimizer/RNG signatures match. Each backend's
three throughput repetitions finish with bitwise identical model, optimizer and
RNG states. All cross-backend episode-summary ledgers and final RNG states match;
the largest parameter difference after 20 updates is 1.665e-13 in float64.
Float32 model parameters are exact. Complete action/observation trajectories
were not logged, so these facts are not called a complete trajectory proof.
[Detailed numerical checks](provenance_review/review.json).

The independent fixed-rollout checks address that gap for controlled computation.
Reported median replay speedups are 1.491× PP Real, 1.422× PCP Real, 1.479× FC
Real and 1.334× PCP Binary. Their validation code checks all forward/hidden/critic
tensors, collector losses/raw gradients, globally clipped gradients, model and
active Adam state, plus exact per-forward RNG and gradient-presence masks.
Recorded maximum float64 differences are at most 7.28e-12, below the declared
1e-10 absolute tolerance; float32 maxima are at most 9.74e-15. The four fixtures
were independently loaded and hashed, their counts/floors/action legality/source
and initial model/optimizer identities checked. This analysis did not rerun the
Stokes numerical kernels on the Mac; recorded pass flags are bound to the
fail-closed code and logs. Replay is serial compute with RNG restoration and
Python iteration, excluding environment work, action sampling and IPC. It is
separate from the four-collector throughput measurement.
[Replay coverage and limits](replay_review/review.json).

**What became faster.** Three-update diagnostic collector wall time, summed over
collectors and divided by their total joint environment steps:

| Workload | Graph preparation, ms/step | Model forward, ms/step | Loss/backward, ms/step |
|---|---:|---:|---:|
| PP Real | 1.039 → 0.001 | 4.462 → 2.956 | 3.513 → 3.191 |
| PCP Real | 1.061 → 0.001 | 7.317 → 4.117 | 7.796 → 7.228 |
| FC Real | 1.135 → 0.001 | 6.492 → 3.388 | 6.713 → 6.149 |
| PCP Binary | 0.994 → 0.001 | 9.548 → 5.976 | 10.429 → 9.842 |

Graph preparation plus forward computation account for 87.8–88.6% of the reduction
in these measured collector phases. Loss/backward is now the largest measured
collector phase in all four Torch diagnostics. These phases overlap across
collectors and cannot be added to parent wait/optimizer intervals or treated as
a direct decomposition of elapsed speedup. Parent waiting is 9.2–25.2% of update
wall time and includes residual collector synchronization; it is not a pure IPC
cost. Checkpointing contributes only 0.024–0.226% of segment time. There is no
case here for changing the environment, research budgets or checkpoint frequency
to gain speed. [Recomputed phase/resource values](hardware_review/results.json).

Across the 24 unprofiled throughput runs, training consumed about 3.41–3.66
CPU-seconds per segment second, or 85–91% of the
four-core allocation. The sum of each collector's individual lifetime peak RSS
is 3.46–6.10 GiB for DGL and 3.60–6.41 GiB for Torch. These sums can double-count
shared pages and need not represent simultaneous peaks; they are not Slurm job
peak memory. Replay RSS is a cumulative process high-water mark after earlier
fixture/correctness work and cannot compare backend memory. Keep 16 GiB for the
preflights. No swap, page-fault pressure or cgroup peak counters were saved, so
absence of a recorded resource problem is not a full memory-pressure diagnosis.

The first PP DGL process had 146.68 seconds of pre-training startup; subsequent
throughput startups were 10.58–11.41 seconds. Its first update was also unusually
slow. The primary measurement excludes update one and startup. Do not use that
first whole-process ratio as the backend speedup; the startup cause was not
measured. [Per-run timing fields](metrics/throughput.csv).

**Scope and interpretation.** This is a controlled DGL-versus-Torch comparison
of the new paper-v1 architecture/learner/environment. It does not establish the
cause of the old supplement-v1 PCP Binary slowdown. Old Binary cost about 2.52×
Real per step, versus 1.31× DGL / 1.38× Torch in this benchmark. Architecture,
learner and payload settings changed, and the earlier runs lack equivalent
hardware evidence. Do not credit the entire difference from the old 38.8 steps/s
to this port. Speedup is not evidence of convergence, successful learning, or
SoftRole superiority. The paper fidelity choices and limits remain those in
[FIDELITY.md](../../publication_reconstruction/FIDELITY.md).

**Next steps, in order:**

1. Run the four official **paper-v1 / Torch 100-update preflights**. Keep four
   collectors, floor 500, horizons 80/300, the pinned environment and 16 GiB.
   The benchmark alone has `#SBATCH --constraint=sapphirerapids`; the preflight
   and training scripts currently lack it. Pass the constraint explicitly.
   The current rates suggest about 52 minutes of active segments across the
   serial `%1` preflight array, plus startup/probes/queueing; this is a planning
   estimate, not a time guarantee. Require the existing complete ledger,
   finite-optimizer and frozen-checkpoint probe gates to pass. Inspect resource
   growth over 100 updates and record Slurm accounting when available.

   ```bash
   PUBLICATION_MESSAGE_BACKEND=torch-v1 \
     sbatch --constraint=sapphirerapids slurm/publication_paper_preflight.sbatch
   ```

2. If all four pass, freeze this implementation and launch the prescribed fresh
   seeds 0–2 with Torch; retain DGL as the reference. Carry the same CPU constraint
   into training. Use the fresh paper-study preparation and common panel; existing
   old-model archives remain their own stratum. Avoid another optimization round
   before starting these research runs.

   ```bash
   # Only after the four official preflights pass:
   # Choose fresh paths and use this same root for both methods.
   STUDY_ROOT=runs/paper_study_20261005
   .venv/bin/python -m publication_reconstruction paper-study-plan \
     --message-backend torch-v1 --run-root "$STUDY_ROOT" \
     --output "${STUDY_ROOT}_preparation"
   PUBLICATION_MESSAGE_BACKEND=torch-v1 PUBLICATION_RUN_ROOT="$STUDY_ROOT/hetnet" \
     sbatch --constraint=sapphirerapids slurm/publication_paper_train.sbatch
   ```

3. Plan checkpointed continuations, especially for PCP Binary. The launcher pauses
   after 46 hours, below its 48-hour Slurm limit. At these early rates Binary
   projects to 55.5 hours, approximately 33.2M steps in the first 46 hours, then
   about 9.5 further active hours. Resume the recorded checkpoint into a fresh
   segment with its inherited backend/source and unchanged 40M budget. Record
   the retained lineage; do not restart a seed or shorten the budget. Ensure the
   continuation requests the same CPU family and explicit core binding—the
   existing resume script does not explicitly set `--cpu-bind=cores`.

4. Run the matched paper-FC SoftRole shared/banked seeds under the same physical
   simulator contract, keeping their own method unchanged. Final FC evaluation
   uses the common 500-scenario panel and the predeclared rule: select the first
   saved checkpoint at or above 28M steps. Before results are interpreted, prepare the paper-specific selected-
   checkpoint manifest/commands; that automation remains absent and the legacy
   `prepare-evaluation` command is not the paper-v1 route. Existing manual frozen
   commands are documented in the reconstruction README. This can proceed while
   research training runs; it need not delay backend preflight.

   ```bash
   # Same STUDY_ROOT and preparation as step 2:
   SOFTROLE_PAPER_RUN_ROOT="$STUDY_ROOT/softrole_fc" \
     sbatch --constraint=sapphirerapids slurm/softrole_paper_fc.sbatch
   ```

| Workload / per-seed budget | DGL projected hours | Torch projected hours |
|---|---:|---:|
| PP Real / 40M | 30.4 | 22.0 |
| PCP Real / 40M | 54.5 | 40.3 |
| FC Real / 28M | 36.7 | 26.1 |
| PCP Binary / 40M | 71.2 | 55.5 |

These projections use median **segment** throughput, including recorded logging
and checkpoints but excluding startup/queueing. Rates can change during learning.
The twelve HetNet runs project to 431.4 job-hours versus 578.2 for DGL: about
146.8 job-hours saved. With three runs continuously active that is an idealized
144 hours, approximately six days, before queueing, interruptions and evaluation;
SoftRole runs are not included. This makes the study substantially more practical,
without promising completion by a fixed date. [Projection formulas and data](metrics/summary.json).

Analysis changed no runtime, simulator, dependency, launcher, original input or
checkpoint. No training, model evaluation, cluster submission, commit or push was
performed. Files added are the reproducible analysis and independent audit records
in this folder; `.gitignore` and `AGENTS.md` record and retain the work. Source
inputs and generated reports are hashed in the audit manifests.

To reproduce the central arithmetic into a fresh output directory:

```bash
.venv/bin/python analysis/stokes_backend_2026-10-05/analyze.py \
  --input stokes_runs/hetnet_backend_benchmark_904547 \
  --output analysis/stokes_backend_2026-10-05/recheck
```
