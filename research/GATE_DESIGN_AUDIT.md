# Gate A/B design audit: original HetNet, PCP, frozen evaluation

Date: 2026-09-26. Baseline examined: `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`, local branch `frozen-eval`. This is a research-first **static audit**. No implementation, simulator execution, training, or gate test was performed for this note. The complete supplied revision-2 instruction file was read. This document does not authorize changes outside its six allowed deviation categories.

## 1. Most important possible blocker: shared parameters are not enough

The source does put each model parameter's data in shared memory before constructing workers ([main.py:289–305](../main.py#L289)). `MultiProcessTrainer` creates `nprocesses−1` workers and retains one main-process trainer ([multi_processing.py:47–58](../multi_processing.py#L47)); workers seed Torch/NumPy using `seed+id+1` in `run` ([lines 16–18](../multi_processing.py#L16)). `spawn` is selected in [main.py:26–27](../main.py#L26). These facts support the intended shared-weight design; they do not prove runtime sharing or correct gradient aggregation under the selected library version.

**Static gradient-storage incompatibility to test before Run 1.** Main and worker gradient tensors are cached only once at [multi_processing.py:69–83](../multi_processing.py#L69). Subsequent aggregation writes through these cached tensors ([lines 105–112](../multi_processing.py#L105)). Both outer optimizers call `zero_grad()` each batch ([worker line 29](../multi_processing.py#L29), [main line 96](../multi_processing.py#L96)). In addition, the actual HetGAT per-class loss calls its policy-local Adam's `zero_grad()` before backward ([policy.py:845–859](../hetgat/policy.py#L845)).

In the pinned **Torch 2.2.1** implementation, `Optimizer.zero_grad` defaults to `set_to_none=True` and explicitly replaces `p.grad` with `None` ([official versioned source, lines 710–745](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/optim/optimizer.py#L710)). A later backward consequently obtains a new current gradient tensor; the old tensor retained in the one-time cache is still a live object. Thus the cache can refer to old gradients while RMSprop reads newly produced `parameter.grad`. Merely proving that a worker observes the main process's updated **weights** would not catch this error.

The runtime gate should cover at least three consecutive updates and verify:

1. Parameter signatures are equal in main and workers after an optimizer step, and actually differ from the pre-step values.
2. In each process, named current gradient tensors correspond to the storage used for aggregation; pointer comparisons are meaningful **within** a process, not as raw virtual addresses across processes.
3. The current optimizer gradient equals the fresh sum of all processes' gradients divided by the actual total collected steps, with the same clipping order as upstream.
4. The set and order of gradient-bearing parameter names match across processes and updates; a list of unnamed non-null tensors is insufficient evidence.

The existing per-class function clips each process's model gradient before the main trainer aggregates it ([policy.py:864–865](../hetgat/policy.py#L864)); the gate reference must preserve that order rather than silently switch to clipping a combined gradient. Source inspection identifies the incompatibility; a failed runtime gate and a concrete minimal proposed patch must be shown before seeking approval to change behavior. An explicit `set_to_none=False` restoration would have to cover the **inner policy optimizer as well as the outer clearing sites**. This note does not apply it or assert that a one-line outer change would suffice.

## 2. Preserve the real architecture and learner

The model factory at [main.py:230–269](../main.py#L230) specifies map-dependent input widths, hidden widths of 16 per type, outputs P=5/A=6/state=8, four heads, two state nodes, Real communication and per-class critics. The model has two HetGAT layers ([uavnet.py:102–109](../hetgat/uavnet.py#L102)); the first concatenates heads, while the final layer averages heads and disables its ReLU ([fastreal.py:357–391](../hetgat/graph/fastreal.py#L357)). Do not introduce a final activation, global neighbor softmax, a shared six-action actor, or a new team-reward loss.

The optimizer that steps is RMSprop, with alpha 0.97 and epsilon 1e-6 ([trainer.py:55–57](../trainer.py#L55), [multi_processing.py:112](../multi_processing.py#L112), [trainer.py:596–614](../trainer.py#L596)). The policy constructor also creates Adam and a scheduler ([policy.py:408–413](../hetgat/policy.py#L408)); that does not make Adam the stepping optimizer. The relevant policy-local optimizer step is commented at [policy.py:867](../hetgat/policy.py#L867). Actual per-class loss includes separately normalized class advantages, `50*policy_loss+critic_loss`, local clipping, and episode/horizon padding ([policy.py:778–877](../hetgat/policy.py#L778)). These are reproduction facts, including their unusual normalizations, not invitations to redesign them.

`Trainer.compute_grad` passes the real configured P/A counts to the A2C loss ([trainer.py:475–487](../trainer.py#L475)). The dummy 3-row `value` tensor in collection is bookkeeping on this path ([trainer.py:158–160](../trainer.py#L158)); do not infer from it that the actual per-class advantage arrays always use three agents. Gate A must still exercise every permitted composition. The singleton-P exclusion follows the supplied demonstrated failure and is consistent with unqualified `.squeeze()` on the P LSTM inputs ([uavnet.py:315–326](../hetgat/uavnet.py#L315)); the A call explicitly reshapes to preserve its batch dimension ([line 334](../hetgat/uavnet.py#L334)). No singleton-P repair is authorized.

The parameterized graph operators are shared within physical classes; actor symmetry is a within-class relabeling statement here. A P/A swap without matching their different sensor/action interfaces is not the requested permutation null test. The prior feasibility proof applies only when observations, graph, memory and action semantics are consistently relabeled.

## 3. Construct a genuinely inference-only model

Do **not** instantiate `A2CPolicy` and then delete its optimizer. Its constructor already creates optimizer/scheduler objects and 500 episode buffer lists by default ([policy.py:393–427](../hetgat/policy.py#L393), [486–496](../hetgat/policy.py#L486)). Do not call its action collection method, which appends critic predictions ([policy.py:516–565](../hetgat/policy.py#L516)).

A minimal frozen runner can construct **`UAVNetA2CEasy` directly**, using exactly the resolved arguments from the training factory, and load the checkpoint's `policy_net` state dictionary with `strict=True`. Retain the complete module, including unused critic heads, so its parameter names, shapes and values can match the source exactly. Build the same state-node graph, call the same forward function, and ignore critic outputs. No optimizer, trainer, replay list or training policy wrapper is necessary. The direct model returns logits, P/A critic values and next hidden state ([uavnet.py:367–391](../hetgat/uavnet.py#L367)).

**Numerical contract:** upstream sets the default tensor type to `torch.DoubleTensor` at [main.py:32](../main.py#L32). Model preprocessing creates untyped `torch.zeros`, `torch.Tensor` and state tensors ([uavnet.py:171–223](../hetgat/uavnet.py#L171), [359–365](../hetgat/uavnet.py#L359)); merely applying `.double()` to module parameters may leave newly allocated helper tensors in float32. Reproduce the CPU float64 construction/forward context explicitly, without importing `main.py` and its top-level environment/trainer side effects. Assert dtypes as well as parameter signatures.

The exact P and A distributions are `Categorical(logits=results['P'])` and `Categorical(logits=results['A'])` ([action_utils.py:29–54](../action_utils.py#L29)). A dedicated generator can sample their normalized probabilities without changing the distribution; validate probabilities and action ordering against the original sampler. Sample P and A independently in a fixed order and use an action RNG independent of NumPy's environment RNG. Do not replace primary sampling with argmax.

Reset hidden state using `model.init_hidden` at every episode start ([uavnet.py:155–167](../hetgat/uavnet.py#L155); original reset at [trainer.py:143–150](../trainer.py#L143)). Preserve it through sensor failure. Under `inference_mode`, detach boundaries do not change forward values; do not reset the memory at `t_fail` or add failure flags. Learned tensors and non-memory buffers remain immutable. Hashes before/after the complete evaluation are mandatory evidence, while construction without learner objects and an allowlist of mutable LSTM state establish the intended transition contract.

A useful no-optimizer construction test temporarily makes optimizer construction raise, then constructs and runs the frozen model. It must not need to bypass that guard. A second test varies the state-node input while fixing all actor-visible features; actor logits should be unchanged because `build_hetgraph` has agent→state and state→state edges but **no state→agent edges** ([hetgat/utils.py:133–152](../hetgat/utils.py#L133)). Do not remove state edges in the baseline to make this test easier.

## 4. Sensor failure: match the blind transformation without repairing the baseline

The non-tensor PCP observation vocabulary has `BASE=D²` position channels and four semantic class channels ([predator_capture_env.py:130–155](../envs/ic3net_envs/predator_capture_env.py#L130)). Agent positions contribute to shared P/A class counts; this path has no separate channel for each agent identity ([lines 265–274](../envs/ic3net_envs/predator_capture_env.py#L265)). The unused `state_len` expression is not the active observation width. These are the concrete grounds for testing the P-agent permutation null.

Original blind A observations retain all first `BASE` channels and set **every channel from `BASE` onward to −1**, at every cell of the local observation window ([lines 282–291](../envs/ic3net_envs/predator_capture_env.py#L282)). This is the correct no-vision transformation for the victim P's observation; it is not all zeros, an entirely −1 vector, removal of the agent, or conversion of its body/graph type to A.

There is existing aliasing: P and A observation windows are appended as array views ([lines 277–285](../envs/ic3net_envs/predator_capture_env.py#L277)), then A blinding writes through one view ([line 291](../envs/ic3net_envs/predator_capture_env.py#L291)); `np.stack` occurs only afterward ([line 306](../envs/ic3net_envs/predator_capture_env.py#L306)). An unconditional `.copy()` at every original slice would repair that legacy behavior and change intact observations. That is not needed for the authorized victim-only intervention.

**Recommended minimal hook:** preserve the upstream observation computation; after it returns its stacked observation, copy the result and apply `[..., BASE:] = -1` to the victim row only. Thus legacy A-view effects remain present in both intact and failure conditions, while the new intervention cannot corrupt anyone else's row. Gate B compares two observations from the **same physical state**, one with and one without this postprocessing. It should assert exact equality of all nonvictim rows and of the victim's position channels. Two freely evolving post-failure trajectories will generally visit different states, so their later nonvictim observations need not match.

Define observation time explicitly: \(o_0\) is the reset observation; action \(a_t\) consumes \(o_t\). Mask the victim when \(t\ge t_{\rm fail}\), so failure at zero is active before the first action. In intact and failure conditions, preserve action, sink, reward and completion rules. Record whether the victim had already reached the prey before its first affected decision, and whether the whole task ended before the event. Keep these episodes in the unconditional endpoint.

One additional source caution should be documented, not silently repaired: `get_obs_features` increments its offset by `self.P_s` ([uavnet.py:211–216](../hetgat/uavnet.py#L211)), while each flattened cell has `in_dim['P']` channels. This is not an obvious semantic-channel stride when those dimensions differ. The frozen runner must call the original forward preprocessing rather than replacing it with an apparently cleaner semantic parser. The requested sensor hook is defined in environment observation space, regardless of how the published model consumes it.

## 5. Message removal without modifying attention math

Keep the four canonical relation types, but make P→P, P→A, A→P and A→A edge sets empty. Preserve node counts, agent self transformations, agent→state edges and state self edges. The normal source does not add same-type graph self-loops for this actor ([policy.py:523–525](../hetgat/policy.py#L523)); its actor self term is instead the explicit transformed node feature.

`fastreal.py` already checks relation edge counts before score/softmax/message operations ([lines 202–269](../hetgat/graph/fastreal.py#L202)) and before adding aggregates to self features ([lines 311–333](../hetgat/graph/fastreal.py#L311)). Thus an empty relation contributes zero by **absence of an addend**, not necessarily by a materialized `ft_*` tensor filled with zeros. Gate B should assert finite logits and equality of each agent's preactivation to its original self transform when all agent relations are removed. It should not require nonexistent graph fields or add denominator epsilons. Preserve existing per-relation, per-destination normalization for nonempty relations.

The graph builder currently computes candidate links and associated `dist` arrays together ([hetgat/utils.py:64–92](../hetgat/utils.py#L64), [133–159](../hetgat/utils.py#L133)). An off-by-default graph hook must empty distance data consistently or remove edges after a correctly constructed graph; mismatched edge-feature lengths are an engineering failure. No change inside the learned layer is required by this intervention.

## 6. Banks, RNG and reproducibility

Use 500 ordered unique-cell rosters per composition, generated before final evaluation from a dedicated bank RNG with seed 0. Each row lists P agents, then A agents, then prey. Sample `N+1` distinct **flat cell indices** from `D²`, then unravel to coordinates, matching original `_get_cordinates` ([PCP:240–250](../envs/ic3net_envs/predator_capture_env.py#L240)). Verify bounds, distinct cells within each entry, cardinality, ordering, file hash and round-trip reset.

The supplied helper is a format/serialization reference, not an exact sampler to copy: [evaluation_conditions.py:22–23](../test_config/evaluation_conditions.py#L22) separately samples x and y coordinates without replacement. That excludes valid repeated rows/columns and cannot generate 7 or 11 occupied cells on a 5×5 grid. The requested distinct-cell bank is feasible because `N+1≤25`; using the original environment's flat-cell sampling satisfies the stated requirement.

Original `reset()` accepts no bank argument ([PCP:202](../envs/ic3net_envs/predator_capture_env.py#L202)); its `args.eval` path reads a developer-specific absolute file ([PCP:243–249](../envs/ic3net_envs/predator_capture_env.py#L243)). The bank hook should supply those coordinates through an explicit supported path and avoid consuming unnecessary random initial placements. Keep `args.eval` from accidentally invoking the old hard-coded file. `_set_grid`, reached/captured flags and observation generation must still be reset exactly as upstream.

Seed Python, NumPy and Torch before model/environment construction; original placement is after construction ([main.py:333–334](../main.py#L333)). Preserve the documented worker seed offset. Workers currently seed Torch and NumPy, while Python `random` is not explicitly seeded in their `run`; the active stationary-PCP path imports `random` but its inspected placement sampling uses NumPy. Record this fact and ensure any new helper does not silently use an uncontrolled RNG.

For paired evaluation, keep bank IDs, event manifests and action RNG seeds independent of policy ID and condition. A fixed stream of random uniforms can couple categorical actions, but equal random seeds do **not** mean different policies select identical actions or visit identical later states. Separate environment, action, bank and bootstrap streams; record the mapping. The final JSONL should contain no timestamps or durations if Gate B requires byte equality. Put nondeterministic process provenance in a separate file.

**Gate A contradiction requiring an explicit interpretation:** the instructions require actual wall time in `metrics.jsonl` and also require two runs' complete `metrics.jsonl` files to be identical. Real wall time is not deterministic. Retain truthful wall times and compare a preregistered projection containing epoch, steps, episodes, success, rewards and losses, excluding timing/host/process metadata. Report both full-file hashes and deterministic-projection hashes. Do not zero, invent or round times to manufacture byte equality.

The metrics logger should consume each raw update's `s` statistics and its own totals. [main.py:399–404](../main.py#L399) adds already cumulative epoch statistics to lifetime counters after each update; those printed counters therefore overcount. A separate correct metrics stream is an allowed logging deviation and can preserve original stdout and learner behavior. Original collection finishes whole episodes until at least `batch_size` steps are reached ([trainer.py:510–527](../trainer.py#L510)); actual steps can exceed `batch_size`, so log actual totals.

## 7. Analysis-plan requirements grounded in primary papers

Read directly: [Agarwal et al., NeurIPS 2021, §§2,4.1 and Figure 6](../../marl-comm/softrole/papers/Agarwal_2021_Statistical_Precipice.pdf); [Lowe et al., AAMAS 2019, §§3 and 6.2](../../marl-comm/softrole/papers/Lowe_2019_Pitfalls_Communication.pdf). The latter local PDF is an author preprint; §6.2 appears on PDF page 8. Primary publication links: [Agarwal](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html), [Lowe](https://arxiv.org/abs/1903.05168).

For policy family \(a\), composition \(c\), trained seed \(s\), and bank entry \(e\), let \(Y_{a,c,s,e}\in\{0,1\}\) denote task success by 80 actions. Define seed-level success \(\bar Y_{a,c,s}=500^{-1}\sum_eY_{a,c,s,e}\). Failure completion time is scored as 80, including failures in its mean, and success remains the primary endpoint. This avoids a successful-episodes-only estimator that can favor unreliable policies.

**Transfer:** for each target composition,

\[
\widehat\Delta_c=
\frac1{S_N}\sum_s\bar Y_{N,c,s}
-\frac1{S_F}\sum_s\bar Y_{F,c,s}.
\]

Resample native and source **training runs independently** as instructed; matching numeric seed labels do not define a paired trained-policy analysis. Evaluation-bank pairing reduces environmental variation, but does not create additional independent trained policies. Preserve the source run's correlation across its multiple target compositions when producing a joint contrast/vector of results.

**Sensor failure and message removal:** within one source checkpoint, compute the episode-paired differences first, then average within training seed. Bootstrap the five source seed-level differences as whole units, retaining intact/event and live/severed pairing. Do not treat 2,500 episodes as 2,500 trained policies, pool the two victims as independent replications, or independently resample the same intact control for every panel.

Predeclare whether intervals condition on the fixed evaluation bank. A primary seed-only bootstrap naturally reports variation across trained runs conditional on that bank and fixed action-seed schedule. If a secondary interval additionally resamples bank entries, use a common bank-index resample across the paired policies/conditions and preserve seed clusters; label the broader estimand. Five or three training runs remain the main limitation. Agarwal §4.1 specifically cautions that small-N single-task bootstrap intervals are limited, and Figure 6 notes undercoverage with three runs; its multi-task aggregation results are not a guarantee for this study's individual composition contrasts. Show every seed and call the requested 95% intervals nominal bootstrap intervals, without promising exact coverage.

Define the primary contrasts before any Run 2 outcomes: the four composition gaps, the six specified victim/time failure penalties, and the source live-minus-severed effect are a finite declared family. Either name a smaller primary subset and label the remainder secondary, or preregister how multiplicity is handled. Do not select the most adverse failure time after inspecting data. Report all raw intact/event means, seed differences and moot-event fractions, even when the penalty is near zero. No significance threshold should automatically decide whether SoftRole merits development.

The native policies provide an intact reference at each **same composition**; larger PCP teams have more completion obligations ([PCP:524–540](../envs/ic3net_envs/predator_capture_env.py#L524)). There is no separately failure-trained reference in the authorized grid. At the source composition, “native” is the same family as F-src; do not imply that the failure panel compares against a new policy trained to tolerate failures. The scripted oracle and untrained floor provide the separately specified feasibility/floor context.

The source-based oracle can target the true prey position using the existing movement actions and capture only when on prey. P agents sink on arrival ([PCP:448–450](../envs/ic3net_envs/predator_capture_env.py#L448)); A capture uses action 5 at the prey ([PCP:444–446](../envs/ic3net_envs/predator_capture_env.py#L444)). Do not assume it attains 100% before testing the actual border/movement/sink rules. A poor oracle score is a gate finding to explain, not permission to alter physics.

Lowe §6.2 explicitly warns that severing communication at test time changes the network's input distribution and may degrade performance even without useful message content. Therefore the authorized live-minus-empty-graph result measures sensitivity/reliance of this trained policy on its message pathways. It does not prove task-level communication necessity, efficient content, or superior emergent semantics. Attention max-weight/entropy are secondary diagnostics, not causal explanations by themselves.

Missing/crashed training seeds must appear in the run ledger with causes. Do not replace the final epoch-2,000 checkpoint with a best-success checkpoint. Successful-subset estimates, if unavoidable, must give expected versus observed seed counts and be labeled conditional/incomplete. The final analysis plan's committed hash must precede Run 2 outcomes; later decisions are exploratory.

## 8. Decisions and boundaries for the root workstream

| Issue | Recommended decision | Status/source |
|---|---|---|
| Original math/learner | Preserve two-layer Real actor, per-class A2C loss, clipping order and stepping RMSprop | Verified static source; implementation not changed |
| Cached gradients on Torch 2.2.1 | Add a multi-update storage/aggregation gate; surface concrete failure before patch approval | Strong source-identified compatibility risk |
| Frozen construction | Direct `UAVNetA2CEasy`, full strict model state, CPU float64, no policy wrapper | Verified constructor/forward structure |
| Failure hook | Post-stack victim-only copy/mask; no unconditional legacy alias repair | Minimal authorized observation intervention |
| Message removal | Empty canonical agent relations only; preserve state/self behavior | Existing empty-edge branches support this without changing math |
| Bank generator | Uniform distinct flat cells, 500 entries, seed 0 | User requirement + environment sampler; existing helper cannot generate large fixed-map rosters |
| Determinism with timing | Compare declared deterministic metrics projection; retain real timing separately | Resolves incompatible literal requirements honestly |
| Statistical intervals | Seed-level nominal bootstrap, paired source interventions, unpaired native/source run resampling | User protocol + Agarwal limitations |
| Communication interpretation | Reliance/sensitivity only for severed-message drop | Lowe §6.2 |

Nothing in this document reports Gate A or Gate B as passed. Runtime evidence, authorized deviations, and the separately committed analysis plan remain necessary.
