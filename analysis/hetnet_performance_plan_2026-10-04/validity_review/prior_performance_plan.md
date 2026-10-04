# HetNet training speed plan

Date: 4 October 2026. Status: proposed engineering work; no optimization or speedup has been demonstrated by this plan.

The objective is to make the original HetNet study reconstruction practical to train and evaluate as a traceable comparison with our Capability-Conditioned HetNet (SoftRole). Speed is an engineering means to complete the locked scientific protocol, not a reason to weaken that baseline. Preserve the audited executable behavior, profile before selecting the opt-in graph/model candidates below, and accept changes only after exact behavioral checks and an unprofiled Stokes timing comparison. The completed preflights establish that training executes on Stokes; they do not identify the simulator as its bottleneck. [E1, C1, C3]

This document distinguishes **observed evidence**, **unmeasured candidates**, and **proposed lab acceptance rules**. The rules are engineering choices made for this project, not hyperparameters or guarantees supplied by the authors. No claim of a particular speedup, unchanged long-run learning quality, or exact reproduction of the published experiments is justified yet.

## What original HetNet means here

There are two obligations. Preserve the selected paper/supplement mechanisms while documenting existing reconstruction deviations, and preserve the exact behavior of the selected executable baseline. Neither obligation substitutes for the other. The paper and supplement describe the method; the public source, archived protocols and tests specify details that the paper does not fully determine. Our checkout is a **supplement-aligned architecture and optimizer with a documented public-code learner and corrected environment**. In particular, its FC individual rewards and ability to extinguish undiscovered fires already differ from the paper's description; this plan does not resolve that existing fidelity limit. The exact source and commands that produced the publication results have not been established by the repository's provenance audit. Do not describe a faster version as certified identical to that unavailable checkout. [P1, P2, C2]

The optimization baseline is the current `publication_reconstruction` runtime, using `supplement-v1`, `corrected-v1`, the locked October recipes and pinned packages. The public-code compatibility variant, historical environment variant, root reproduction code and SoftRole remain distinct. Do not switch between them to obtain an apparent speedup. The environment corrections already present in `corrected-v1` are documented reconstruction decisions; this plan adds no new environment semantics. [C2, C3]

| Preserve | Traceable basis |
|---|---|
| Typed agents and directed communication relations, preprocessing and recurrence, class policies/critics, and separation of actor information from centralized critic information | Paper §§4.2–5.2; active model and policy. [P1, C4] |
| Three communication layers, four heads, 16-wide hidden heads, hidden concatenation and final averaging, active Adam at learning rate 0.001 | Supplement §2.1; `supplement-v1` model construction and active trainer optimizer. These are not the compatibility variant's two layers/RMSprop. [P2, C4] |
| Actual Binary message generation, temperature, hard-forward/surrogate behavior, shared payload decoding, and random draw order | Active `fastbinary.py`, especially lines 48–60 and 184–220. Do not replace the preserved public-code Gumbel path with SoftRole bits or a threshold ablation. [P2, C5] |
| Observations, masks, counts, agent ordering, transitions, rewards, termination, horizon handling and FC discovery bookkeeping | The selected environment version and its executable branches. [C6] |
| Per-class padded advantage normalization, critic targets, episode averaging, actor/value weights 50:1, local clipping, ordered collector-gradient sum, subsequent step denominator, and one active optimizer step | Active `policy.py` lines 788–867 and `multi_processing.py` lines 133–160. These are executable-baseline requirements; do not attribute every detail to the paper. [C7] |
| Four collectors, 500-step floor per collector, ten updates per epoch, horizons 80 for PP/PCP and 300 for FC, complete episodes, prescribed sample budgets and seed streams | The selected study manifest and launcher. Public dated recipes are not certified publication experiment settings. [C3, C8] |
| Existing CPU numerical behavior, including float64 mode and explicit float32 attention parameters; parameter names, shapes, buffers and gradient storage | Runtime initialization, graph constructors and checkpoint/signature code. Do not silently cast the entire model. [C4, C5, C8, C9] |

Keep the executable learner even where another estimator appears cleaner. Replacing its aggregation with SoftRole's team objective, changing clipping/normalization, or reducing batch size is a different experiment, not an implementation speedup. [C7]

## Existing downstream study context

This section records the existing scientific protocols; writing this speed plan neither launches nor newly authorizes the twelve research runs.

The first-wave deliverable remains twelve fresh HetNet runs: PP Real, PCP Real, FC Real and PCP Binary-16, each with training seeds 0, 1 and 2. Preserve the existing 40M PP/PCP and 28M FC stopping targets, epoch caps and checkpoint milestones. These budgets come from the selected public recipes and our locked study, not a recovered publication-producing command. Completing a speed benchmark is not completing this scientific study. [C3]

For comparison with SoftRole, retain the existing frozen evaluation contract rather than selecting checkpoints or environments after seeing outcomes:

| Comparison requirement | Recorded implementation and interpretation |
|---|---|
| Outcome-independent checkpoint selection | First saved checkpoint at or above 30M steps for PCP, 40M for PP and 28M for FC; retain actual overshoot, update count and checkpoint hash. Match the declared budget of a SoftRole comparison, or report a budget mismatch explicitly. [C3, C13] |
| Common assigned PCP scenarios | Reuse the verified SoftRole panel bytes: 500 nominal scenarios per composition, seed 2700, and 100 native failure/sham scenarios, seed 2701, window 10–30. Shared assignments do not imply identical trajectories across models. [C13] |
| Honest information and learning differences | Original HetNet retains typed sensory channels, P/A policies and typed relations. SoftRole merges P/A counts and knows its own sensing/actuation capabilities; HetNet receives no health input in failure evaluation. The learners and optimizers also differ. Results compare complete methods under declared settings; they do not isolate soft gates as the sole causal difference. [C7, C13] |
| Compatible environments | Do not pool corrected HetNet FC with the existing old-environment SoftRole FC runs as a matched architecture comparison. An aligned FC comparison requires a separately declared environment/training integration. PCP physical/action/reward settings and observation-version differences must be recorded too. [C2, C13] |
| Comparable reporting | Report success, horizon-capped completion, both team sum and mean-agent return, actual exposure/censoring, each training seed and equal-weight seed means. Failure-minus-sham is paired within a policy/seed before averaging; held-out compositions do not choose settings. No superiority or robustness conclusion follows from a timing preflight. [C13] |

The reproduction claim is therefore explicit: **supplement-aligned HetNet, public-code learner, corrected-v1 environment, pinned runtime and fully recorded deviations**. Reproducing published score tables exactly remains unverified; speeding up this implementation cannot remove that evidence gap. [P1, P2, C2]

## What the existing measurements establish

Both platforms completed all four 100-update preflights with valid checkpoints and frozen probes. The table uses actual environment steps divided by summed update time. It is evidence about these executions, not an isolated hardware comparison. [E1, E2]

| Workload | Stokes steps per second | Mac steps per second | Observed Mac to Stokes ratio | Stokes full-budget projection in hours |
|---|---:|---:|---:|---:|
| PP Real | 168.63 | 392.08 | 2.33 | 66.25 |
| PCP Real | 97.72 | 221.23 | 2.26 | 114.15 |
| FC Real | 101.61 | 230.11 | 2.26 | 76.67 |
| PCP Binary | 38.81 | 212.63 | 5.48 | 287.25 |

The projections use the recorded segment throughput and 40M PP/PCP or 28M FC budgets; they are not runtime promises. The Mac run completed 893,647 steps and 10,950 episodes in about 18m24s. All four launchers overlapped for about 9m18s, after which contention declined. The cross-platform initial signatures differ in 36 float32 attention tensors per workload; initial float64 hashes match. Therefore the Mac comparison cannot identify the cause of the Slurm slowdown or serve as a paired optimization control. [E1, E2, W2]

In the Stokes records, summed `train_batch` time accounts for more than 99.4% of each active segment. The entire checkpoint path takes only 0.22–0.47 seconds per 100-update run. These are timer-scope observations, not a decomposition of compute inside `train_batch`, but they rule out checkpoint-frequency reduction as a material solution for these runs. Binary costs about 2.52 times as much wall time per environment step as PCP Real. The evidence inventory independently recalculates these quantities from `updates.jsonl`, `preflight.json` and `run_status.json`. [E1, C1]

Exploratory probes without retained scripts and raw timing samples cannot establish an accepted speedup or exact equivalence. The component and whole-update checks below must generate independently inspectable records. No conversation-only timing claim is used as acceptance evidence.

**Recommendation R1 — profile PCP Real and PCP Binary first.** They share the selected domain/team/horizon and show a substantial Stokes cost difference; this makes them useful diagnostic cases. This is prioritization, not evidence that Binary sampling, DGL, or the simulator causes the difference. Include PP and FC in the final correctness and throughput checks. [E1, C3]

Current timers bracket `get_episode` and `train_batch`. They do not separately time environment observation construction, model preprocessing, graph construction, attention, backward work or communication between collectors. Mean episode duration is not environment-only latency, and adding overlapping collector durations is not job wall time. [C1]

Several frequently suggested fixes are already present: one Torch thread and one configured numerical-library thread per collector, allocation tracing disabled by default, and episode records buffered per update. Static base grids are already constructed and then copied. Do not spend effort claiming these as new optimizations. Also, `policy_entropy.py` and the unused alternative observation helpers are not the active training path identified here. [C10]

## Establish a trustworthy profile on Stokes

**Recommendation R2 — freeze the baseline before instrumentation.** Record Git HEAD and dirty diff, exact runtime/source-manifest hashes, resolved scientific arguments, installed package versions, interpreter, model/optimizer signatures and RNG states. Hashes are necessary because the current working tree has intentional uncommitted changes. Use separate fresh output directories and archived source for baseline and candidate. After each understood instrumentation or candidate edit, retain the prior manifest and exact diff, then refresh that checkout's `ORIGINS.json` with the existing audited tool before launcher-based validation. The launcher rejects unexplained runtime drift. Never rewrite an old archive or refresh provenance merely to conceal unexplained drift. [C9]

**Recommendation R3 — record the actual CPU allocation and load context before changing scheduling.** Keep one Slurm task, four collectors and four allocated CPUs initially. Record node name, CPU model, socket/core/thread topology, process affinity, Slurm version, thread settings and accounting. Four CPU IDs alone do not prove four distinct physical cores. Slurm distinguishes allocation from binding, and its behavior depends on site configuration. An experiment with core binding or no SMT is conditional on observing the topology and site support; it is not an assumed cure. Do not set `--ntasks=4` while retaining four internal collectors per launcher. [C3, W3]

Useful read-only commands for a future compute-node diagnostic allocation are:

```bash
hostname
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE
scontrol show job --details "$SLURM_JOB_ID"
srun --cpu-bind=verbose .venv/bin/python -c 'import os, torch; print("affinity", sorted(os.sched_getaffinity(0))); print("torch_threads", torch.get_num_threads()); print(torch.__config__.show())'
```

The last command reports a separate diagnostic process, not the trainer's inherited thread count. Preserve the existing collector resource records as the authoritative check on the running collectors. Do not infer allocator or BLAS behavior from the requested CPU count alone. After the job, collect:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,ElapsedRaw,AllocCPUS,NodeList,TotalCPU,MaxRSS,MaxRSSTask
```

Replace `JOBID` with the actual job. Record units and the site accounting plugin where available. `MaxRSS` is an accounting maximum over tasks, not automatically simultaneous whole-job physical memory; neither it nor the sum of individual process peaks substitutes for a system memory trace. These commands collect evidence and do not change the recipe. [W3, W4, C8]

**Recommendation R4 — add opt-in profiling at existing computation boundaries.** Use coarse monotonic timing in every collector and a short PyTorch CPU trace in a designated collector. Attribute reset/observation and wrapper conversion, feature extraction, graph construction, policy forward/message sampling, action handling/environment step, GAE/loss/backward/clipping, parent waiting, gradient aggregation/optimizer, and logging/checkpointing. Where blocks nest, record inclusive/exclusive scope explicitly so totals are not double-counted. Never obtain another observation just to profile it: FC observation construction changes discovery state. [C1, C6, C7]

Proposed initial diagnostic size: five unprofiled warmup updates, then ten measured updates for each PCP variant, retaining four collectors, 500-step floors and horizon 80. Capture detailed operator traces for only two of those updates; retain coarse timing for all collectors. These sizes are lab choices, intended to bound tracing cost. The trace from one collector describes that collector, not the whole job. Use parent waits and per-collector times to identify the critical path. Expand to a straggling collector or FC only when the first profile indicates a reason. [C1, C3, W1]

Keep shape/stack/memory tracing off initially and enable it only to answer a specific unresolved question. PyTorch documents additional overhead and tensor-reference retention from some profiler options. Run a profiling-disabled control and verify the same trajectories/RNG states. Profiled throughput is diagnostic evidence, never the accepted speedup measurement. [W1, W2]

Deliver `profile_summary.json` with all scope definitions, counts, source identities, per-collector timings, parent wait/aggregation time, and trace paths. An observed environment bottleneck is a finding for a separate proposal; this patch preserves environment/wrapper bytes. As a fixed-work bound, if a replaceable critical-path portion occupies fraction f and is accelerated by r, the resulting time ratio is `(1-f)+f/r`. This arithmetic assumes everything else is unchanged; a tiny component cannot explain a large end-to-end gain. Estimate f from the measured critical path, not summed overlapping CPU time.

## Proposed execution interface

Add `--execution-mode reference|cached-v1` to `train`, `run-index`, `study-plan` and the local preflight launcher. Default to `reference`, including when reading older protocols without the field. The runtime receives `--execution_mode`; policy/model constructors receive a nonparameter setting that neither consumes RNG nor changes parameter/buffer names, shapes or initialization. Record the mode in the resolved command, protocol and checkpoint configuration. These flags are **planned additions**, not currently available commands. [C9, C14]

Preserve the locked scientific settings in `STUDY.json`; the new option selects execution of the same model/learner. Add an explicit execution-mode override to the fresh-run Slurm launchers, with `reference` as the default. Continuation inherits the recorded mode and archived source with no mode override. Old source archives remain executable without receiving a flag they do not support. Frozen evaluation records the training mode and verifies exact checkpoint loading; model-level optimization is used only when its archived constructor supports that recorded mode. [C3, C9, C13, C14]

Add optional `--profile-timing` (runtime `--profile_timing`), disabled by default. Its fields are separate from rewards/losses and have documented inclusive/exclusive scopes. Preserve the existing episode-timing field and local launcher rather than replacing their reporting. Benchmark controls must verify that profiling changes neither trajectories nor RNG. [C1, C14]

## Candidate changes and their obligations

The initial candidate set contains three **source-observed opportunities whose whole-update benefit is unmeasured**: relation-view reuse, exact feature copying, and guarded topology reuse. Use the profile to select worthwhile candidates, implement/test each separately, then validate any justified combination under `cached-v1`. Preserve a directly executable `reference` path. The source establishes repeated work; only controlled measurements establish its practical cost and whether a replacement helps. No environment file or learner formula is part of this proposed first pass. [C4–C7, C12]

### Reuse relation views within each layer

**R5.** Both active graph layers repeatedly index the same relation for edge counting, source/destination features, `apply_edges`, `edge_softmax` and `update_all`. In `cached-v1`, bind each of the six communication relation views once per layer forward, inside the current graph feature scope, and use those handles for the existing calls. Preserve relation order, empty-edge branches, tensor operators and reduction order. Do not retain a relation view across steps or cache learned features. In `reference`, retain the original view lookup behavior. This removes repeated graph-wrapper construction without changing the intended attention computation; exact backward checks must substantiate the implementation. [C5]

### Replace repeated feature-copy loops with exact selections

**R6.** The active forward calls `remove_excess_action_features_from_all` and `get_obs_features`, which allocate per-agent tensors, copy fixed cell slices, and concatenate repeatedly. Test reshape/slice or precomputed-index extraction that produces exactly the same tensor values, layout requirements, dtype, ownership and gradient behavior. Retain the 29-entry cell stride for these 5x5 recipes, including all typed sensory channels; do not replace it with a new observation representation. Handle the existing zero-A outputs deliberately rather than silently changing their shape. [C4]

Implement reshape/slice assignment into fresh `torch.zeros` allocations using the original default dtype/device, rather than returning input views or using input-derived `new_zeros`. Preserve PP's unused one-row A placeholder, independent storage, contiguity and ignored trailing columns. Guard the canonical CPU flat-observation path (`P=29`, `A=25`, sensory width 4); unsupported layouts use the existing extraction. Check float32 and float64 inputs, strided inputs, trailing columns, and gradients with respect to input as well as model parameters. [C4; proposed implementation]

This is a tensor-layout refactor, not permission to remove `clone`, `detach`, masks or recurrent truncation because they look redundant. Compare the preprocessed inputs, next hidden/cell states and backward results with the archived implementation, including a sequence that crosses the detach boundary. If exact equality fails, keep the reference path and investigate before timing a candidate as equivalent. [C4, C8]

The source supplies an explicit indexing argument. For cell index c, position width s, sensory width q, original cell width d=s+q, and action-agent position width a, the original loops compute `P_stat[i,c*s+j]=x[0,i,c*d+j]`, `P_obs[i,c*q+k]=x[0,i,c*d+s+k]`, and `A_stat[v,c*a+j]=x[0,num_P+v,c*d+j]`. Bulk copies must implement those same maps and the same casts/ownership. This is an index identity, not a proof that an arbitrary vectorized implementation preserves autograd or all accepted layouts. [C4]

### Reuse communication topology under explicit conditions

**R7.** The active action path builds a heterograph each step. Its helper decodes positions, computes distances, constructs typed edges, creates the graph and writes distance features. With a fixed team, unlimited communication ranges and no loss, the permitted agent-edge topology does not depend on positions; this follows from the helper's range predicates. A cache for that restricted configuration is therefore a reasonable candidate, not a measured optimization. [C12]

Preserve node types/counts, relation names, edge IDs/order, self-edge rules and state-node edges. Cache topology and immutable indexing only. Keep position-dependent edge data correct if any consumer reads it; otherwise explicitly restrict the fast path to consumers proven not to use it. Finite-range, lossy or changed-team calls must use the original path until separately tested. Do not merge typed relations, replace DGL attention with dense attention, change softmax groups, or skip stochastic Binary operations. [C5, C12]

DGL layers assign node and edge features during forward. Reusing a populated graph can retain stale features or connect different steps' autograd graphs. Evaluate a fresh local graph scope for each forward and demonstrate that outputs remain valid until backward. DGL's `local_scope` protects out-of-place feature assignments, but explicitly does not isolate in-place mutation. The API is not an equivalence proof: test repeated forward/backward, empty relations, cleanup and memory across several complete updates. [C5, W5]

The helper also evaluates the default `np.linalg.norm(...)` expression of `dict.get` even when the key exists. This is source evidence of redundant work, but a separate lookup refactor stays deferred with R8; it is not a fourth change hidden inside this patch. [C12]

### Keep secondary changes outside the first patch

**R8.** Leave the discarded duplicate `ma.log_prob(a_idx)` call, GAE list/scalar bookkeeping, constant allocations and simulator masking loops unchanged in this first patch. They are inspectable but have no recorded whole-update benefit justifying extra scope. Even a nonpersistent registered buffer changes the set considered by `named_buffers()` signatures. [C4, C7, C9, C11]

Environment sources remain byte-identical. In particular, FC `_get_obs` updates discovery state; observation copies, sentinels, count accumulation, agent ordering and coincident-agent handling must remain intact. If subsequent profiling identifies an environment bottleneck, document it as a separate proposal instead of silently expanding this implementation. [C6]

## Correctness gate before a performance claim

**Recommendation R9 — compare complete frozen baseline and candidate implementations in isolated processes on the same pinned platform.** A single reference model file that imports the candidate graph modules is not an independent oracle for graph changes. Capture all reference runtime files and hashes before edits. Use identical initial named model/optimizer tensors and Python/NumPy/Torch RNG states, and ensure construction of caches consumes no randomness. PyTorch does not promise bitwise replay across platforms; the Mac versus Stokes runs are not this gate. [T1, C8, C9, W2]

For this strict first pass, require exact equality on the same pinned CPU runtime. All discrete traces and RNG states must match. A tolerance-only result is useful diagnostic evidence, but does not silently qualify a refactor for the unchanged-behavior route. Investigate any mismatch; numerical reassociation, precision changes or a new kernel backend require a separately scoped proposal. This is a conservative lab acceptance rule.

Define exact tensor identity as equal names, shapes, dtypes and contiguous tensor bytes, with matching gradient `None` masks. `torch.equal` alone does not check every part of this contract, including signed zero. Use the project's named-tensor hashing pattern alongside direct comparisons; preserve persistent gradient storage because collectors exchange and retain gradient pointers. Explicitly exclude only clocks, resource measurements, paths and declared source/execution-mode metadata from whole-record comparisons; do not broadly drop diagnostic or numerical fields. [C7, C9]

| Gate | Required comparison | Existing evidence and missing coverage |
|---|---|---|
| Observation and transition replay | Fixed initial states and action sequences; exact observations, physical state, rewards, done/info, FC discovery arrays/flags and RNG after each step. Include overlapping views/coincident agents, borders, capture/fire actions, success and horizon termination. | Existing version-specific environment tests are useful regression cases, not an optimization equivalence oracle. [T2, C6] |
| Graph and recurrent model | Canonical relation edge order and distances where used; all forward outputs, critic values, hidden/cell states, Binary draws and next RNG state; gradients for every named parameter after multiple steps spanning detach gap 5. | Existing exact compatibility comparisons target `public-code-v1`; the active three-layer path needs its own complete before/after comparison. [T1, C4, C5] |
| Complete optimizer update | Per-collector episode order/counts, losses, named gradients before/after local clipping, aggregated/divided gradients, active Adam moments and resulting parameters. Repeat at least three updates to expose stale shared-gradient storage. | Current pipeline retains gradient pointers and uses `set_to_none=False`. Root-runtime gradient probes do not establish this isolated reconstruction's behavior. [C7, T3] |
| Lifecycle and persistence | Save/load and same-candidate pause/resume reproduce model, optimizer, counters, every collector RNG and non-timing episode records; source and evaluator identities validate; frozen probe leaves parameters unchanged. Retain interrupt/output-failure cleanup of launcher, collectors and probe descendants. | Existing recovery tests provide model/optimizer/RNG checks over short runs, but their workload matrix is limited. Extend for affected variants and retain process-cleanup regressions. [T3, C9] |

Apply short behavioral probes to both model specifications, Real/Binary and PP/PCP/FC, including complete updates with one and four collectors. Cover several forwards before a backward, detach-gap crossing, episode reset, zero-A/empty relations, changed positions, cache cleanup, input gradients and unsupported-path fallback. Retain the four actual study workloads for the production-shaped performance gate. A purported optimization must not broaden or silently repair unrelated behavior. [C3, T1–T3]

Passing these checks is finite implementation evidence, not a theorem about every possible trajectory or long-run convergence. Keep adversarial fixtures and source-level reasoning alongside test results. The existing suite alone cannot certify a change that has not yet been implemented.

## Slurm benchmark and acceptance decision

**Recommendation R10 — benchmark the equivalent candidate against a matched baseline on Stokes.** Run baseline and candidate sequentially in the same allocation and CPU set, with the same numerical libraries, thread settings, logging/checkpoint policy and identical initial tensor/RNG states. Verify those identities rather than assuming the same seed is sufficient. Do not run competing implementations concurrently when measuring an implementation speedup; concurrency is a separate experiment. [E2, C8, W2, W3]

The bounded comparison harness runs **three paired repetitions of 20 complete updates per workload**, using seed 991, four collectors, the 500-step floor, ten updates per epoch and the production horizon (80 PP/PCP; 300 FC). Each repetition restarts from the same verified initial tensors/RNG. Use reference→cached, cached→reference, reference→cached order. Run workload pairs sequentially in the same allocation and CPU set; never compare concurrently competing implementations. Report all 20 updates, first-update timing and process startup separately; use updates 2–20 for the prespecified post-first-update statistic. Do not claim this short window measures long-run steady state. These are **diagnostics**, not the official 100-update preflight. [C3, W2, W3; proposed lab design]

Implement this as a dedicated bounded comparison harness, sharing command/provenance logic with the existing local launcher without changing its four official preflight routes. The harness must not feed a 20-update run to the existing 100-update readiness validator or label it a completed research run. Produce raw per-run records, paired comparisons and a summary; include a dry run with no outputs/jobs and fresh-directory protection. Prepare an optional Stokes batch invocation with one task, four CPUs, one thread per collector and the same pinned environment, but do not submit it as part of document/engineering work. Four collectors and a sequential pair must not accidentally become eight or sixteen competing collectors in one timing comparison. [C3, C14]

Microbenchmarks can attribute individual changes, but cannot replace this whole-update comparison. PyTorch's timer supports warmups and explicit thread control for component measurements. Three repeats and alternating order are descriptive controls; they do not establish a statistical confidence interval or erase frequency/load drift. Record node/allocation identity and every result, including slower outcomes. [W3, W6]

Record, at minimum:

- Actual environment steps, completed episodes, updates and source/configuration hashes.
- `sum(steps) / sum(update_seconds)` for the full and prespecified post-first-update windows; total segment and full process time separately.
- Median/p95 update and episode rollout duration, per-collector wait information, checkpoint time and available memory/accounting evidence with units.
- Initial and final model/optimizer/RNG identities and the exact equivalence result, excluding only clocks and other explicitly nonsemantic metadata.

The fields and distinctions follow the existing recorder and resource contracts; do not average per-update rates or silently drop startup/checkpoint costs. [C1, C8, C9]

**Proposed rollout rule:** exact correctness is mandatory independently of timing. For each intended workload, require higher update throughput in all three matched pairs, higher median segment throughput, and no unexplained resource regression before recommending `cached-v1`. Publish the magnitude and individual ratios rather than promising a fixed speedup. If timing signs disagree, order effects dominate, or resource observations are insufficient, label that workload inconclusive and retain `reference` for it. These are conservative engineering decisions, not statistical significance tests or literature-derived constants; prespecify any follow-up before collecting further results.

Finally run the existing full 100-update preflight once per study workload for the selected final candidate, with profiler disabled and unchanged production collector/batch/horizon settings. Validate the saved model, active optimizer, source identity and frozen probe. This is a readiness check, not another claim of a paired speedup. Do not launch the 12-run study merely because unit tests or a component benchmark passed. [C3, C9]

## Changes outside this plan

GPU/MPS migration, all-float32 or mixed precision, dependency/compiler upgrades, `torch.compile`, dense replacements for typed DGL message passing, head/layer/width reductions, changed Binary estimators, reward/critic/GAE/gradient-rule changes, shorter horizons and smaller sample budgets are outside the strict optimization route. They either alter method/numerics or introduce a larger equivalence problem. No source or measurement reviewed here establishes them as safe performance fixes for this baseline. [P1, P2, C3–C8, W2]

Increasing Slurm array concurrency changes study scheduling without reducing each run's scientific work; resource contention can change per-run wall time and requires a separate measurement. Changing the internal collector count changes samples per update and thus the training protocol. Keep these separate from code optimization and do not use either to disguise a reduced scientific budget. The existing 46-hour graceful pause/continuation route addresses allocation duration while preserving the chosen stopping target; it is not an acceleration. [E2, C3, C7, C9]

## Implementation sequence and deliverables

1. Freeze the baseline and evidence inventory; retain current archives and checkpoints. Produce a written map of active call paths and invariants. [R2, R9]
2. Profile the two PCP variants on Stokes under observed CPU topology/allocation. Produce the component/critical-path summary and profiling-overhead control. [R1–R4]
3. Select candidates from R5–R7 using the measured profile, and implement each selected change separately with the recorded execution interface, retaining `reference` as the default. R8 stays deferred. Archive the diff/prior manifest and refresh intentional provenance before launcher-based checks. Keep root reproduction entrypoints and environment/wrapper sources unchanged. [R2, R5–R9]
4. Run the exact component, trajectory, gradient, update and recovery gates. Extend tests where the existing matrix is incomplete; reject unexplained differences. [R9]
5. Perform the paired unprofiled Slurm benchmark. Publish every measurement and the pass/fail/inconclusive decision against the prespecified rules. [R10]
6. Combine only individually justified changes, record the combined source identity, recheck their combination, then run the four official candidate preflights. Use the same audited provenance process before these checks, retaining the prior manifest and patch evidence. [C9]
7. Use the validated candidate only for fresh study runs. Complete the existing research budgets, apply the fixed frozen-checkpoint/panel rules, and report HetNet alongside SoftRole with the comparison qualifications above. Speed results and scientific results are separate deliverables. [C3, C13]

Do not bypass a source guard to resume an old checkpoint with different runtime code. Existing recovery explicitly checks source identity. An accelerated implementation gets a new run/source record; continuation across implementations would require a separately designed migration contract. Keep an unchanged checkout available for resuming existing runs. [C9]

For each candidate, the review record must contain the source diff, hypothesis and measured profile fraction, baseline/candidate hashes, equivalence results, timing/accounting records, actual counts, and the decision. Record every changed file and exact work in the existing uppercase `AGENTS.md`. No new core implementation, training job, or cluster submission is performed by writing this plan.

## Evidence register

Local paths are relative to this document. Line numbers identify the inspected snapshot; exact file hashes, input identities and verified quantitative values are recorded in [the plan evidence inventory](../analysis/hetnet_performance_plan_2026-10-04/evidence.json). The inventory is the stable anchor if line numbers move. Code establishes what is executed; it does not establish how much time that code consumes.

Recheck file identities, local links, table arithmetic and initial signature metadata with `python3 analysis/hetnet_performance_plan_2026-10-04/verify_evidence.py` from the repository root. This read-only check requires the listed local run archives; it neither executes a policy nor repeats the earlier checkpoint tensor audit. The large archives and local paper copies remain separate inputs, not bundled into the plan. `check_external_sources.py` in the same folder re-fetches primary references and prints byte identities without writing files. Its [recorded check](../analysis/hetnet_performance_plan_2026-10-04/external_sources.json) confirms P1/P2 remote PDF bytes match the prior paper manifest; dynamic documentation hashes identify the retrieval rather than guaranteeing future byte identity.

[External retrieval records](../analysis/hetnet_performance_plan_2026-10-04/external_sources.json) identify the cited primary-source bytes. Hashes and content markers support source identity; they do not replace reading the cited sections or prove a performance claim.

- **P1:** Seraj et al., [Learning Efficient Diverse Communication for Cooperative Heterogeneous Teaming](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf), AAMAS 2022, §§4.2–5.2, pp.1175–1178. Available [local PDF](../docs/papers/Seraj_2022_HetNet.pdf) and [extracted text](../research/papers/HetNet.txt). The local PDF was rehashed on 4 October and matches the [training paper manifest](../research/papers/TRAINING_PAPER_MANIFEST.json); its current path/hash is recorded in the plan evidence inventory.
- **P2:** Authors' [supplement at pinned repository commit](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/AAMAS_22___HetNet_Supplementary.pdf), §2.1 architecture/training and §2.2 Binary ablations. Available [local PDF](../docs/papers/Seraj_2022_HetNet_Supplementary.pdf) and [extracted text](../research/papers/HetNet_Supplementary.txt), especially lines 109–122. The local PDF was rehashed on 4 October and matches the prior manifest; the plan inventory records that verification.
- **E1:** [Stokes preflight audit](../analysis/hetnet_restart_2026-10-03/preflight_903502_audit.json), backed by [the four archived Stokes runs](../stokes_runs/hetnet_preflight_903502/), each `preflight.json`, `updates.jsonl`, checkpoint and source archive.
- **E2:** [Mac reconciled metrics](../runs/hetnet_preflight_mac_20261003_234455_727228_analysis/metrics/summary.json), [checkpoint/source audit](../runs/hetnet_preflight_mac_20261003_234455_727228_analysis/checkpoint_audit.json), [timing audit](../runs/hetnet_preflight_mac_20261003_234455_727228_analysis/timing_audit.json), and [639-file input inventory](../runs/hetnet_preflight_mac_20261003_234455_727228_analysis/metrics/input_hashes.json). Existing evidence; not rerun performance experiments.
- **C1:** [main.py](runtime/main.py), lines 561–576, and [trainer.py](runtime/trainer.py), lines 529–539: actual timing scopes. [Recorder](runtime/hetnet_ext/recording.py), lines 84–120: recorded quantities.
- **C2:** [Reconstruction README](README.md), opening provenance statement, environment-version differences, recipe table and final interpretation boundary; [ORIGINS.json](ORIGINS.json) records researcher environment commit `d57da0717d5564027df7e8ba75614feca8006960` and local scaffold `0ee9ceb38b6133866e9c1db40bed2c6caa3b842d`. Prior bounded [author-source search findings](../analysis/publication_reconstruction_2026-09-30/author_search/FINDINGS.txt) explain the unverified publication checkout; this is not a claim that no such checkout exists anywhere.
- **C3:** [STUDY.json](STUDY.json), [study.py](study.py), lines 35–46 and 62–105, [preflight batch file](../slurm/publication_preflight.sbatch), [training batch file](../slurm/publication_train.sbatch): locked workload shape, budgets and actual preflight/probe contract.
- **C4:** [uavnet.py](runtime/hetgat/uavnet.py), lines 110–130, 189–242, 307–412; [trainer.py](runtime/trainer.py), lines 55–58 and 164–175: architecture, feature extraction, active optimizer and recurrence truncation.
- **C5:** [fastbinary.py](runtime/hetgat/graph/fastbinary.py), lines 48–60, 70–87, 147–220 and 227–329; [fastreal.py](runtime/hetgat/graph/fastreal.py), forward feature assignments and repeated relation lookups from line 198: actual parameters, messages, graph mutation and view reuse candidates.
- **C6:** [PCP environment](runtime/envs/ic3net_envs/predator_capture_env.py), lines 268–314; [FC environment](runtime/envs/ic3net_envs/fire_commander_env.py), lines 312–357; [wrapper](runtime/env_wrappers.py), lines 81–109: counts, copies, sentinels, FC side effects and tensor conversion.
- **C7:** [policy.py](runtime/hetgat/policy.py), lines 788–911; [multi_processing.py](runtime/multi_processing.py), lines 119–160: active loss/GAE/gradient pipeline. Do not substitute `policy_entropy.py` or SoftRole's learner.
- **C8:** [main.py](runtime/main.py), lines 20–55 and 561–635; [recovery.py](runtime/hetnet_ext/recovery.py), lines 31–72: thread/default dtype/spawn, counts and per-collector RNG/resources.
- **C9:** [artifacts.py](artifacts.py), [recovery.py](runtime/hetnet_ext/recovery.py), especially source validation at lines 100–108, [signatures.py](runtime/hetnet_ext/signatures.py), lines 25–29, [launcher](__main__.py), [provenance recorder](../scripts/record_publication_sources.py): source, checkpoint, buffer and continuation contracts.
- **C10:** [main.py](runtime/main.py), lines 20–35 and 77–78; [trainer.py](runtime/trainer.py), lines 649–669; [recorder](runtime/hetnet_ext/recording.py), lines 92–101; environment `_set_grid`/`_get_obs` pairs in C6: optimizations already present and actual imported policy.
- **C11:** [policy.py](runtime/hetgat/policy.py), lines 569–589: discarded duplicate action-agent log probability.
- **C12:** [policy.py](runtime/hetgat/policy.py), lines 518–528; [graph helper](runtime/hetgat/utils.py), lines 49–159: repeated construction, range predicates, distance calculation and typed edge order.
- **C13:** [Frozen preparation](study.py), `select_checkpoint` and `prepare_evaluation`; [locked SoftRole panel preparation](../scripts/prepare_pcp_frozen.py); [native evaluator](evaluation_worker.py); [SoftRole research contract](../softrole/RESEARCH.md), especially observation/capability, evaluation and comparison sections; [SoftRole observation adapter](../softrole/env.py) and [reporting](../softrole/report.py); reconstruction README sections Frozen panels and reporting and source limitations. These establish panel/budget rules, information differences, reporting strata and the FC comparison boundary.
- **C14:** [Training CLI](__main__.py), `parser`, `resolve` and `resolve_resume`; [study launcher](study.py), `add_commands` and `train_arguments`; [local preflight launcher](../scripts/publication_preflight_local.py). These are the current extension points; `--execution-mode` and `--profile-timing` are proposed rather than existing interfaces.
- **T1:** [Model specification tests](../tests/test_publication_spec.py), lines 106–145 and 264–277; [model tests](../tests/test_publication_model.py), lines 117–140; [reference model fixture](../tests/fixtures/publication_uavnet_public_v1.py): existing compatibility coverage and its limits.
- **T2:** [Environment tests](../tests/test_publication_envs.py): original/corrected branch behaviors and observation fixtures.
- **T3:** [Recovery tests](../tests/test_publication_recovery.py), lines 87–143; [artifact tests](../tests/test_publication_artifacts.py); [source audit tests](../tests/test_publication_source_audit.py); [lifecycle tests](../tests/test_publication_lifecycle.py), from line 26; [local-launcher cleanup tests](../tests/test_publication_preflight_local.py), from line 170. [Root gradient probe](../tests/helpers/hetnet_gradient_probe.py), lines 20–42, imports the root runtime and is not proof for a candidate reconstruction.
- **W1:** PyTorch **v2.2.1** [profiler source/documentation](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/profiler/profiler.py), including overhead/reference-retention notes. Checked against the installed 2.2.1 source; no upgrade is proposed.
- **W2:** PyTorch **v2.2.1** [reproducibility documentation](https://github.com/pytorch/pytorch/blob/v2.2.1/docs/source/notes/randomness.rst): platform/version limits and RNG control.
- **W3:** Official Slurm [CPU management documentation](https://slurm.schedmd.com/cpu_management.html), allocation versus affinity, topology, `scontrol` and verbose binding. Current web documentation is guidance, not evidence of Stokes configuration; record the installed version and allocation.
- **W4:** Official Slurm [sacct documentation](https://slurm.schedmd.com/sacct.html), accounting fields and their task/plugin limitations.
- **W5:** DGL **2.1.0** [`local_scope` documentation](https://www.dgl.ai/dgl_docs/en/2.1.x/generated/dgl.DGLGraph.local_scope.html): out-of-place isolation and in-place exception.
- **W6:** PyTorch **v2.2.1** [benchmark timer source/documentation](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/utils/benchmark/utils/timer.py): warmups, replicated measurements and thread control.
