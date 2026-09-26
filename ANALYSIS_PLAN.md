# Prospective analysis plan: original HetNet frozen-policy evaluation

**Prospective protocol fixed for preregistration, 26 September 2026.** Its initial commit fixes this plan before any Run 2 outcomes are generated or inspected in this project. Record that commit hash, the bank and event-manifest hashes, and the eventual analysis-code revision in `RESULTS.md`; the registration record is maintained in `research/PREREGISTRATION.json`. No Run 2 outcome informed this plan. Local training and runner smoke tests are engineering evidence and cannot supply research results. This document specifies the analysis; it does not implement or authorize bypassing Phase B's gates.

The governing protocol is [TASK_SPEC.md §§0.1, 5–6](research/TASK_SPEC.md). Numerical choices introduced here are our declared choices, not claims about the HetNet paper's evaluation procedure. Amendments made before outcomes must receive a new dated commit and a rationale; decisions made after outcomes are inspected must be labeled exploratory and cannot replace these analyses silently.

## 1. Question, scope, and checkpoints

The question is whether the released HetNet-Real A2C policy, trained at fixed composition 2P+1A, retains performance with frozen learned parameters at other fixed compositions or after a P agent loses its sensor, and how much performance depends on the policy's message pathways. No SoftRole model is trained in this study. There are no within-episode arrivals or departures and no failure-trained native reference.

Use the approved original architecture and learner, with the documented compatibility, seeding, recording, and gradient-storage deviations. Preserve the released class-wise attention normalization, mixed parameter dtypes, rewards, dynamics, completion rules, 5×5 map, horizon of 80 actions, and training vision setting. [RESEARCH_NOTES.md](RESEARCH_NOTES.md) and [DEVIATIONS.md](DEVIATIONS.md) identify their sources and deviations.

For every trained seed, select **the final epoch-2,000 checkpoint**. Do not select a best checkpoint, best seed, or policy according to training success, final-bank performance, failure resilience, or message-removal results. An absent or invalid final checkpoint is an incomplete run, not permission to substitute an earlier epoch. Log checkpoint-file hashes and full parameter/buffer signatures; strict loading must preserve names, shapes, dtypes, and values. The frozen runner must pass Gate B before final evaluation, construct no optimizer/trainer, and verify immutability. Only the permitted recurrent memory changes during an episode and resets between episodes. This is the frozen-parameter contract of [feasibility-report Eq. 6](research/project_references/SOFTROLE_FEASIBILITY_REPORT.md) and [TASK_SPEC.md §5.1](research/TASK_SPEC.md).

## 2. Policies and complete condition inventory

The fixed training grid contains 21 separately trained runs:

| Composition | Priority | Trained seeds | Role in evaluation |
| --- | --- | --- | --- |
| 2P+1A | P0 | 0, 1, 2, 3, 4 | Source policies; their F-src evaluations also supply the source-native reference |
| 3P+3A | P1 | 0, 1, 2, 3, 4 | Destination-native reference |
| 4P+6A | P1 | 0, 1, 2, 3, 4 | Destination-native reference |
| 3P+1A | P2 | 0, 1, 2 | Destination-native reference |
| 2P+2A | P2 | 0, 1, 2 | Destination-native reference |

Each of the five source checkpoints receives exactly **12 required conditions**, each with 500 episodes:

| ID | Target composition | Intervention | Conditions per source checkpoint |
| --- | --- | --- | ---: |
| F-src | 2P+1A | Intact observations and messages | 1 |
| F-comp | 3P+1A, 2P+2A, 3P+3A, 4P+6A | Intact observations and messages | 4 |
| F-fail | 2P+1A | Each victim \(j\in\{0,1\}\) crossed with each \(t_f\in\{0,10,20\}\) | 6 |
| F-nomsg | 2P+1A | All agent-to-agent relation edge sets empty | 1 |

Evaluate each destination-native checkpoint intact at its own composition (N-comp). At 2P+1A, F-src is already the native family: reuse the same observations in tables rather than fabricating another independent sample or a source native-minus-frozen test.

The architecture-matched U-floor uses seeded, randomly initialized, untrained policies at every composition, with seeds 0–4 for P0/P1 and 0–2 for P2. Reusing the grid's initialization-seed counts is our declared choice. These policies use sampled categorical actions from their untrained logits; do not replace them with uniformly random actions. Their seeds represent initialization replicates, not trained policies.

The scripted O-ref uses privileged prey coordinates, moves greedily toward the prey, and directs each A to capture at the prey, preserving the environment's action, sink, and stacking rules. Evaluate it on all five intact banks and all six source failure conditions. Its rule and deterministic tie-breaking must be fixed and versioned before final evaluation. It is a physical-feasibility reference, not an optimality certificate or a learned-policy upper bound. [TASK_SPEC.md §§5.2–5.5](research/TASK_SPEC.md).

The required inventory is 30,000 source episodes, 8,000 destination-native episodes, 10,500 U-floor episodes, and 5,500 O-ref episodes: **54,000 unique required episode records**. This count reuses F-src as the source-native reference and excludes smoke tests, technical retries, and optional panels. F-comp uses all five source seeds even where N-comp has three native seeds.

## 3. Fixed evaluation inputs and pairing

Create 500 ordered initial conditions for each composition on the 5×5 grid, with bank seed 0. Within every entry, all agent and prey cells are distinct; order bodies by P agents, A agents, then prey. Version the files, their generator revision, and SHA-256 hashes before final evaluation. The bank is never used to train, tune settings, select checkpoints, or debug policy performance. Development gates use separately identified smoke banks and checkpoints. The correct source initialization samples distinct flat cell indices before conversion to coordinates; see [GATE_DESIGN_AUDIT.md §6](research/GATE_DESIGN_AUDIT.md).

For a given composition, every policy and condition uses the same ordered bank entries and the same per-episode action-RNG seed schedule. Declare evaluation-action seed 0. With composition ordinals \(q=0,1,2,3,4\) for \((2P1A,3P3A,4P6A,3P1A,2P2A)\), and zero-based bank index \(e\in\{0,\ldots,499\}\), use action seed \(1000q+e\). This seed derivation is our protocol choice. It contains neither policy identity, training seed, nor intervention identity. Record it in the evaluation manifest. Use a dedicated action generator, separate from environment/bank RNGs, and an unchanged body/draw ordering within a composition. Do not reset the action generator at the event.

Pairing concerns random inputs, not identical actions or trajectories after interventions: different action distributions can yield different actions from the same random stream. Bank indices do not establish physical-state pairing between different compositions. Oracle runs use the same bank and event manifest; the action schedule is recorded even though the deterministic oracle does not use its draws. Environment RNG settings and their separation from action sampling must also be recorded and identical within each composition. Gate B must validate bank reset, manifest identity, and deterministic repeatability.

## 4. Event timing and episode outcomes

Let \(s_0\) be the reset state. After \(t\) actions, the environment is in \(s_t\); observation \(o_t\) is consumed by action \(a_t\), which leads to \(s_{t+1}\). The sensor event applies to observations with \(t\ge t_f\), **before action \(a_{t_f}\)**. Thus \(t_f=0\) affects the first action. No recurrent state is reset at failure.

After the original observation is formed, copy its stacked array and change only the selected P row's semantic channels to the existing blind representation (−1 from the environment's `BASE` boundary onward). Preserve position channels and every other agent's observation at that same physical state. The victim retains its P class, movement, messages, and obligation to reach the prey. This intervention preserves baseline observation construction while preventing the new masking operation from aliasing other rows. Gate B's nonvictim comparison must use the **same physical state**, since trajectories may diverge afterward. [GATE_DESIGN_AUDIT.md §4](research/GATE_DESIGN_AUDIT.md).

For message removal, empty P→P, P→A, A→P, and A→A relations while preserving the released self transforms and agent→state relations. The default graph has no explicit agent self edges; do not add them. Verify zero relation aggregates and finite outputs without changing the attention equations. [GATE_DESIGN_AUDIT.md §5](research/GATE_DESIGN_AUDIT.md); feasibility Eqs. 7–8.

Record first reach and capture times as **state indices**:

\[
\tau_i^{\rm reach}=\inf\{k\ge0:i\text{ is at the prey in }s_k\},\qquad
\tau_i^{\rm cap}=\inf\{k\ge0:i\text{ has captured in }s_k\}\quad(i\in A).
\tag{A1}
\]

A capture caused by \(a_t\) is therefore recorded at \(t+1\). Use null plus an explicit censoring flag if a reach/capture never occurs by termination; do not encode an unobserved event as a successful event at step 80.

For a failure episode with terminal state index \(T\), record separately:

- `moot_victim`: the victim had reached the prey just before the first affected decision, namely \(\tau_j^{\rm reach}\le t_f\).
- `terminated_before_event`: \(T\le t_f\), so no affected action is taken, including termination exactly at state \(s_{t_f}\).
- `event_applied`: \(T>t_f\).

The task's `moot_victim` label is an operational diagnostic for the victim's arrival obligation, not a theorem of zero effect on team return: an already-arrived P can still communicate, and masking its observation can alter messages to teammates. By contrast, `terminated_before_event` establishes that no intervention was applied to an executed decision. For an episode already terminated, evaluate the moot predicate from its recorded first-reach time; do not invent later observations. Keep all moot and terminated-before-event episodes in unconditional results. Report both flag fractions per seed and condition. In a valid paired run, intact and failure trajectories agree up to the first affected observation; the pre-mask status is therefore common to that pair. No subgroup is selected using post-event success.

The **primary endpoint** is \(Y=1\) exactly when original PCP completion occurs by the end of at most 80 actions, and 0 otherwise. Original completion requires every agent to reach the prey and every A to capture it. A success on action 80 counts as success. Define the secondary capped completion-step endpoint

\[
C=\begin{cases}T,&Y=1,\\80,&Y=0.\end{cases}
\tag{A2}
\]

Always average \(C\) over all episodes, never successes alone. Log actual rollout length separately where it differs from this score. Also retain undiscounted per-agent returns, their sum, and per-class mean returns. Raw event times and returns are descriptive secondary outcomes; they do not replace success as the primary endpoint.

## 5. Estimands and signs of contrasts

For family \(a\), composition \(c\), policy seed \(s\), and bank entry \(e\), first form

\[
\bar Y_{a,c,s}=\frac1{500}\sum_{e=0}^{499}Y_{a,c,s,e},\qquad
\bar C_{a,c,s}=\frac1{500}\sum_{e=0}^{499}C_{a,c,s,e}.
\tag{A3}
\]

Every trained seed has equal weight in its family mean, irrespective of episode duration. Report every seed and the arithmetic mean of seed means. For U-floor use the analogous initialization-seed means. For O-ref report its single fixed-bank mean, numerator/denominator, and episode outcomes; there is no population of oracle training seeds and no fabricated seed-level CI.

The declared primary family comprises **11 success contrasts**, all reported:

1. **Four composition-transfer gaps**, each at the same target composition:

   \[
   \widehat\Delta_c^{\rm transfer}
   =\frac1{S_{N,c}}\sum_s\bar Y_{N,c,s}
   -\frac15\sum_s\bar Y_{F,c,s},\qquad
   S_{N,c}\in\{3,5\}.
   \tag{A4}
   \]

   Positive values mean the native family succeeds more often. Do not pair training runs merely because both are labeled seed 0, 1, or 2, and do not discard two source runs to balance the P2 cells.

2. **Six sensor-failure penalties**, one per prespecified victim/time pair:

   \[
   d_{s,j,t_f}^{\rm fail}=\frac1{500}\sum_e
   \left(Y_{F\text{-src},s,e}-Y_{F\text{-fail}(j,t_f),s,e}\right),
   \qquad\widehat\Delta_{j,t_f}^{\rm fail}=\frac15\sum_s d_{s,j,t_f}^{\rm fail}.
   \tag{A5}
   \]

   Positive values mean success is lost under the event. Retain the episode pairing within the same source checkpoint. Do not pool victims or times as additional independent seeds, choose the worst event after inspection, or describe this contrast as an estimated recovery time. Report raw intact/event success alongside the penalty, capped steps, moot fractions, and O-ref outcomes. This is the raw-outcome contrast motivated by feasibility Eq. 36 and §8.4.

3. **One source message-removal effect**:

   \[
   d_s^{\rm msg}=\frac1{500}\sum_e
   \left(Y_{F\text{-src},s,e}-Y_{F\text{-nomsg},s,e}\right),
   \qquad\widehat\Delta^{\rm msg}=\frac15\sum_s d_s^{\rm msg}.
   \tag{A6}
   \]

   Positive values mean lower success with severed agent messages. Report both raw conditions.

Report success effects in probability units and percentage points. For secondary completion-step contrasts, use \(\bar C_F-\bar C_N\) for transfer and \(\bar C_{\rm event}-\bar C_{\rm intact}\) for interventions, so a positive value means greater capped completion cost. Label these signs explicitly. No aggregate score pools heterogeneous compositions or selects a winner across them.

## 6. Uncertainty procedure

Use **20,000 seed-level nonparametric bootstrap replicates**, analysis RNG `numpy.random.Generator(PCG64(20260926))`, and the 2.5th and 97.5th percentiles with NumPy's `method="linear"` interpolation. This is our fixed numerical procedure. Save bootstrap seed-index arrays, their hashes, and the analysis environment/version in the analysis manifest.

In each replicate draw five source-seed indices with replacement, and reuse that source resample jointly across all F-src, F-comp, F-fail, and F-nomsg outcomes and their paired differences. Independently draw each destination-native family's \(S_{N,c}\) indices with replacement. Compute Eq. A4 from those independent native and source resamples. This is the declared **unpaired training-seed bootstrap despite paired evaluation episodes**. Matching numeric training-seed labels do not establish the designed native/source pairing used for the within-policy interventions. The independent bootstrap draws are an analysis convention, not proof that the realized runs are statistically independent: shared seeds can induce initialization or other random-stream correlations across compositions, which remains a limitation. For Eqs. A5–A6, resample the already paired source-seed differences using the shared source indices. Intact controls and all interventions stay together within a source replicate.

Specifically, \(\operatorname{Var}(\bar Y_N-\bar Y_F)=\operatorname{Var}(\bar Y_N)+\operatorname{Var}(\bar Y_F)-2\operatorname{Cov}(\bar Y_N,\bar Y_F)\), whereas independent family resampling omits the cross-family covariance. Label the transfer intervals explicitly as a **working-independence approximation**; their coverage for the potentially coupled training design is not established. This limitation is separate from the small number of seeds and is not repaired by evaluation-episode pairing.

For reproducibility, generate the full source index matrix first, then native index matrices in composition order 3P3A, 4P6A, 3P1A, 2P2A, then U-floor matrices in the five-composition order in §3. Reuse the relevant matrices for raw success, capped steps, and their contrasts. Do not independently resample a shared intact control for each panel. Floor indices are independent of trained-family indices.

These are **pointwise nominal 95% percentile intervals**, conditional on the fixed 500-entry banks and fixed action-seed schedule. There is no episode resampling in the primary procedure: 500 episodes improve measurement of a given policy but do not create 500 independently trained policies. There is no family-wise error control, no simultaneous 95% claim for the 11 contrasts, and no significance testing, p-value threshold, or significant/non-significant labeling. Report effect sizes, intervals, and the complete seed distribution. Zero-width bootstrap intervals can occur when all observed seed scores agree; they do not establish zero population uncertainty.

Three or five runs provide limited information about the training-run distribution. Agarwal et al. explicitly discuss the weakness of few-run single-task bootstraps and document undercoverage with three runs in their benchmark study; their multi-task results do not guarantee coverage here. These intervals must be described as nominal, not exact. [Agarwal et al., NeurIPS 2021, §§2, 4.1 and Fig. 6](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

## 7. Validation, exclusions, and technical failures

Before aggregation, validate the entire expected inventory: every required policy/condition has exactly one valid record for each bank index 0–499; composition, bank/action seeds, event manifests, checkpoint/provenance hashes, parameter immutability, and finite numeric fields agree; endpoints and event-time conventions are consistent. Reject duplicate, missing, corrupted, or mismatched records rather than silently dropping them. Verify all 21 final trained checkpoints before Run 2, as required by [TASK_SPEC.md §5.6](research/TASK_SPEC.md).

Never exclude a valid episode or seed because it performs poorly, fails to learn, is moot, terminates before the event, is an outlier, or reverses the expected contrast. Do not silently omit a crashed training or evaluation attempt, add a replacement seed, or impute technical crashes as ordinary task failures. Task failures from a valid episode remain \(Y=0,C=80\).

Preserve each technical attempt in a unique directory, including its logs, code/config hashes, error, and reason for retry. A retry repeats the same seed, configuration, and approved scientific code; it is not a new independent replicate. Select the earliest complete, valid attempt for that planned run, never the highest-performing attempt. If a scientific-code fix is required, document the deviation, rerun the relevant gates, and specify which runs must be repeated under a common revision before combining results. No silent cross-version pooling is allowed.

An unresolved missing seed or incomplete condition prevents that cell and any dependent contrast from being presented as the complete preregistered result. Publish the missing/failed-run table and expected versus observed counts. Any partial summaries must be labeled incomplete and exploratory, with their actual denominators; do not present a smaller complete-case sample as the original plan. A gate failure is reported and handled under the task's stop rules, rather than repaired by changing the architecture, task, intervention, or endpoint to obtain a favorable result.

## 8. Required outputs and reproduction reporting

Generate the following from validated raw logs, with links to run directories, hashes, configs, and the preregistration commit:

1. All **21** training learning curves: success, steps, and per-class reward. Retain individual seed curves and unsmoothed epoch data; report incomplete or failed runs visibly. Compare source 2P1A qualitatively with the published experiment and document all recipe differences. Larger fixed-map Real natives have paper-linked compositions, but the available Fig. 5c evidence does not establish an exact matched map/horizon/variant reproduction. [TRAINING_RESEARCH.md §§1–2](research/TRAINING_RESEARCH.md); [HetNet §§6.2–6.3.4 and Fig. 5c](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf).
2. Per-composition raw success and capped-step tables for U-floor, F-src/F-comp, N-comp, and O-ref, showing every applicable seed and its sample size. Source-native entries explicitly reuse F-src.
3. The four native-minus-frozen success gaps, their pointwise intervals, and secondary capped-step contrasts.
4. All six failure panels: raw intact/event outcomes, paired seed-level penalties, both moot/termination flag fractions, exposure counts, and the scripted reference. Never show only the worst victim/time or only non-moot episodes.
5. The source live-versus-severed panel with raw means and paired seed-level differences.
6. An audit table of all exclusions of invalid records, crashes, retries, missing checkpoints/conditions, and amendments. Valid low-performing runs remain in the main results.

The future analysis command specified by the task is `python -m hetnet_ext.analyze --runs runs/ --out results/`. It must regenerate all required tables and figures from raw logs and emit a manifest. This plan does not claim that command is implemented or that analysis has run.

## 9. Interpretation limits fixed before outcomes

Permutation equivariance under the stated assumptions is a symmetry property, not a theorem of size generalization; strict checkpoint loading likewise establishes compatibility, not successful frozen performance. Feasibility Proposition 1/Eqs. 22–23 and Eq. 28 support this distinction. Larger PCP teams carry additional reach/capture obligations and greater density on the same map, so frozen outcomes require their same-composition native reference and raw feasibility/floor context.

There is also a source-training **relation-support limitation**. The released source graph has no self edges. At 2P1A, A→A is absent and P→P/A→P neighborhoods are singleton, making the corresponding attention coefficients identically one. Under the declared fresh-training optimizer, the absent A→A branch has no learning signal and singleton relations supply zero gradient to their dedicated attention-score vectors. Larger compositions can activate these previously unsupported paths. Specifically, 3P1A introduces P→P attention comparisons without introducing A→A; 2P2A introduces the source-untrained A→A affine message branch and A→P comparisons, while its A→A attention remains singleton. Larger P1 teams activate both mechanisms. A→state support also changes, but state nodes have no direct outgoing path to the frozen actor. [RELATION_SUPPORT_AUDIT.md](research/RELATION_SUPPORT_AUDIT.md) derives these statements from the released graph and the softmax Jacobian and separates structural probe evidence from performance results.

Therefore a future native/frozen gap quantifies transfer under simultaneous changes in task obligations, graph neighborhoods, and parameter support. It cannot by itself identify failed role inference, attention dilution, or the A→A branch as the cause. No new architecture or ablation is introduced to isolate these mechanisms in this study.

A message-removal drop measures this trained policy's dependence on its message pathways under a changed input distribution. It does not establish task-level communication necessity, useful message semantics, or efficient communication. [Lowe et al., AAMAS 2019, §6.2](https://arxiv.org/abs/1903.05168). The privileged oracle establishes attainable behavior under its particular rule, not optimality. A poor oracle result, including below approximately 95% at 4P6A, requires explanation under the existing physics, not a task modification.

The failure panel compares intact and blinded executions of policies trained intact. It has no separately failure-trained native control. A loss in unconditional success is a sensor-intervention effect under this protocol, not proof that a latent role was or was not inferred. Report smooth-representation and diagnosis limits from feasibility Proposition 3/Eq. 31 and Eq. 33 alongside these results. Any case for later SoftRole work must be based on the measured effect sizes with native, oracle, and floor context; no statistical cutoff automatically declares that research direction successful. Akki makes that decision.

## 10. Optional and exploratory work

Sampled-action evaluation and the 12 required source conditions remain primary. A greedy-action panel, attention entropy/max-weight diagnostics, or message removal at additional compositions may be reported as clearly separated **optional secondary** analyses if their implementation and inclusion are fixed before Run 2 outcomes. They do not replace any required condition or enter the 11 primary contrasts. Attention summaries are diagnostics, not causal attribution.

Any outcome-driven selection of diagnostics, event-aligned/non-moot subsets, additional bootstrap schemes, alternative checkpoints, extra failure times, or new comparisons is exploratory and requires an explicit dated amendment. A non-moot subgroup must disclose its denominator and censoring/selection rule and retain the unconditional primary panel. No optional analysis permits new training recipes, additional architecture changes, or use of the final bank for tuning within the authorized study.
