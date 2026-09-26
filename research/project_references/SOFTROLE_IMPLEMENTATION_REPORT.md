# Implementing and testing SoftRole-HetGAT on the original HetNet codebase

**Implementation design and experiment specification · 25 September 2026**

This report accompanies the [feasibility report](SOFTROLE_FEASIBILITY_REPORT.md). It specifies a future build; no policy was implemented, trained, or evaluated in preparing it. References marked **F** refer to that report's equations and propositions. Source findings come from a static audit of pinned author repositories, with local copies and provenance under [implementation_sources](research/implementation_sources/). Proposed dimensions, schedules and acceptance thresholds below are design choices, not published results.

## 1. Recommended build and the claim it would test

**Build an isolated research fork of the original HetNet, preserve its discrete PP/PCP/FireCommander environments as reproduction references, and implement SoftRole as a new actor plus explicit roster and inference infrastructure.** The current `commstudy` application serves another research direction and should not supply this experiment's actor, action distribution or trainer assumptions.

The first SoftRole version should use real-valued communication, deterministic soft role gates, two synchronous communication rounds, and a fixed bank of role-pair message functions. Start from the class-wise normalization in HetGAT rather than silently changing it to global attention. Retain an A2C-family learner initially. Add binary message compression and alternative learners only after the architecture comparison is interpretable.

The target claim is specific: **one checkpoint can execute with different numbers and combinations of agents, and update its communication and task responsibilities after membership or capability changes using only execution-time evidence and recurrent state.** Parameter sharing and permutation equivariance make this interface possible. Performance on larger teams and timely recovery remain empirical hypotheses: F Eq.(28) explicitly disproves the implication from equivariance to successful size transfer.

There are three separately reported results:

1. Reproduction of released HetNet on the original task definitions.
2. A matched architecture comparison using common variable-roster infrastructure and a common learner.
3. Frozen-checkpoint tests on held-out populations and declared event extensions of those environments.

Changing rewards, discovery rules, observations or the learner creates a new comparison; its results must not be presented as reproduction of published scores. The [HetNet paper][hetnet-paper] already evaluates different team compositions, including PCP teams of 3, 6 and 10 agents. Those results do not establish the proposed single-checkpoint, within-episode recovery claim.

## 2. What the author code actually provides

The audited HetNet revision is `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`. An unmodified copy is stored in [research/implementation_sources/hetnet](research/implementation_sources/hetnet/), with [archive provenance](research/implementation_sources/hetnet_source.json). The important execution path is `main.py → A2CPolicy → UAVNetA2CEasy → MultiHeteroGATLayerReal`, with `Trainer` handling collection, backpropagation and optimizer steps.

| Existing component | Verified behavior | Implication for the build |
|---|---|---|
| [`main.py:230–275`][h-main] | Real-valued A2C branch: two HetGAT layers, four heads; intermediate widths P/A/state = 16; outputs P = 5, A = 6, state = 8; class critics enabled | Discrete categorical actions are native. The sample `hid_size=128` is not this layer's actual hidden width |
| [`uavnet.py:83–167`][h-model] | Separate position/observation preprocessing and LSTM cells; hidden arrays allocated from cached P/A counts | Preserve the sensor meaning; replace class-slot memory allocation with runtime lifetimes |
| [`utils.py:49–173`][h-graph] | Physical P/A node types and four directed agent relations; positions/ranges determine edges | Retain candidate communication links; replace the hard physical relation selector with soft computational roles |
| [`fastreal.py:147–353`][h-layer] | Attention normalizes separately within each relation; relation aggregates and a self term are added before activation | F Eqs.(17)–(18), including its mass gate, are the appropriate hard-compatible extension |
| [`uavnet.py:359–381`][h-state] | State-node input is a small count/map/batch-summary vector; graph aggregation supplies agent information | The original critic is not simply a full simulator-state critic. A new privileged set critic is a documented change |
| [`policy.py:778–870`][h-loss] | Per-class returns and losses use fixed `[batch, P+A, time]` arrays | Masks, variable rosters, loss semantics and empty classes need a deliberate redesign |
| [`trainer.py:55,596–614`][h-trainer] | The optimizer that steps is RMSprop; policy-local Adam steps are commented in the relevant loss path | Preserve the actual optimizer for reproduction; do not infer it from the policy constructor alone |

The graph has agent-to-state edges, but no state-to-agent edge in this path. Preserve that separation when exporting an actor. Privileged training information must not enter the actor through shared features, normalization, graph construction or capability labels.

There is also an architecture/training discrepancy to resolve explicitly: the supplement §2.1 describes **three** multi-head layers and Adam at \(10^{-3}\), whereas this released path constructs **two** layers and steps RMSprop; README examples use \(10^{-4}\). Reproduce the pinned source configuration as one reference, and treat a paper-recipe reconstruction as a separate variant. A reproduction ledger must identify which target each run addresses.

The non-tensor PCP observation has width

\[
d_o=(2v+1)^2(D^2+4),
\tag{I1}
\]

where \(D\) is map width and \(v\) is vision radius. Its channels describe position and entity classes, rather than one separate channel per agent identity. Thus it is already independent of team count on a fixed map. The main obstacles to changing \(n\) are cached counts, row slicing, memory and rollout arrays—not an unavoidable observation-width change. The continuous PCP constraints in the earlier workspace audit do not apply to this original implementation. Changing map width or vision still changes this representation; map generalization requires an additional encoder design. [Environment source audit](research/HETNET_ENVIRONMENT_IMPLEMENTATION_AUDIT.md).

Before numerical reproduction, create an isolated dependency environment and record the versions actually shown to work. The repository does not provide a complete locked Torch/DGL/Gym stack. It also contains old NumPy APIs and machine-specific evaluation paths. Compatibility changes should have their own small patch and smoke-test evidence. Move random seeding before model construction in the controlled comparison: the released `main.py` seeds Torch and NumPy after constructing the model. Begin with one collector process; the original multiprocessing path caches lists of non-null gradients, an assumption that deserves revalidation when empty classes and conditional branches change which parameters receive gradients. These are source-identified risks, not observed runtime failures.

## 3. Actor contract: physical capability and learned role are separate

Represent each active lifetime by local observation \(o_i\), declared body metadata \(c_i^{\rm declared}\), local action/outcome history, memory \(b_i\), and execution-visible received messages. A lifetime identifier is a storage key only. It must never become an agent-ID embedding or an input that permits slot memorization.

Use modality/body-specific adapters where the original sensor interfaces differ, followed by a common-width recurrent representation:

\[
x_i=E_{\mathrm{body}(i)}(o_i,c_i^{\rm declared}),\qquad
b_{i,t}=F_\theta(b_{i,t-1},x_{i,t},a_{i,t-1},y_{i,t-1},q_{i,t-1}).
\tag{I2}
\]

Here \(y\) denotes permitted local execution feedback and \(q\) the previous received communication summary. Adapter selection follows metadata when rows are permuted. Within each interface, parameters are shared across agents. This obeys F Proposition 1 without requiring physically different sensors to have identical raw inputs.

A shared decoder may predict capability or outcome statistics \(\widehat c_i=C_\theta(b_i)\). The role gate is a separate quantity:

\[
p_i=\operatorname{softmax}\!\left(R_\theta(b_i,\widehat c_i,q_{i,t-1})/T\right),
\quad p_i\in\Delta^{K-1}.
\tag{I3}
\]

The capability estimate describes what an agent appears able to do; the gate selects computation useful in its current context. Neither is automatically a calibrated posterior. The proposed first configuration uses \(K=2\), with \(K=3\) as a validation ablation, and a common feature width such as 64 with four attention heads. Those are initial experimental choices. Preserve the original layer dimensions in the literal reproduction arm, and measure parameter count, payload and compute when comparing models with common-width adapters.

Initially train the router through task return. If hidden capability inference is part of the hypothesis, add a separately ablated prediction loss, for example predicting locally observed action success. Simulator capability labels may supervise a predictor during training, but must not be substituted for its output at execution. Give every inference-based baseline the same feedback, history and auxiliary targets. A direct-true-capability actor is a separately labeled privileged reference.

The original bodies have sharply different sensing and actuation. A soft computational role cannot give a P body an extinguisher. On those tasks, useful learned roles may be information source, relay or recipient within the physical constraints. A stronger claim about changing physical task responsibilities needs overlapping or graded capabilities in a named task extension; otherwise the router may only rediscover P versus A.

### A legal two-round schedule

At the start of a step, update memory from the new local observation and previous received context; compute \(p_i\); hold these gates fixed during two synchronous message rounds; then choose actions. Set \(h_i^{(0)}=b_i\); round \(\ell\) sends \((p_i,h_i^{(\ell-1)})\), computes Eqs.(I4)–(I6) at the receiver, and returns \(h_i^{(\ell)}\). The action head uses \(h_i^{(2)}\). Store received context for the next step. The current role is team-relative through prior communication, with an explicit one-step delay for newly received evidence. An entrant starts from its local prior and fresh memory.

A same-step negotiation round before role selection is a useful later variant. Retaining both subsequent task-message rounds would make three rounds; alternatively, allocate one of the two rounds to negotiation and retain one task-message round. Match that choice and payload allowance across baselines. Do not compute gates from an instantaneous global roster/capability pool that actors cannot observe. Message reachability and latency must agree with the graph-hop lower limit discussed after F Eq.(33).

For a first implementation, an edge can carry the sender feature and gate, \((h_j,p_j)\), with the receiver computing the shared relation transforms. Count those role coordinates in the payload budget. Dense tensors may use `[batch, time, slots, feature]`, gates `[batch, time, slots, K]`, and action-expert probabilities `[batch, time, slots, K, 6]`; `slots` is runtime storage with an active mask, never a learned layer dimension. A sparse graph implementation should expose the same mathematical contract.

## 4. The soft relation layer and its exact connection to HetGAT

Implement an agent graph whose edges depend on actual communication opportunities. Soft roles operate on these edges; they do not turn radio links on or off. Let \(A_{ij}=1\) mean that receiver \(i\) can hear sender \(j\). For ordered receiver/sender roles \((r,s)\), define scores \(e_{ij}^{rs}\) and message functions \(M_{rs}\). For each round and attention head:

\[
D_{i,rs}=\sum_j A_{ij}p_{j,s}e^{e_{ij}^{rs}},\qquad
S_{i,s}=\sum_j A_{ij}p_{j,s},
\tag{I4}
\]

\[
u_i^{rs}=\begin{cases}
\displaystyle\sum_j\frac{A_{ij}p_{j,s}e^{e_{ij}^{rs}}}{D_{i,rs}}M_{rs}(h_j),&D_{i,rs}>0,\\
0,&D_{i,rs}=0,
\end{cases}
\tag{I5}
\]

\[
h_i^+=\sum_r p_{i,r}\,
\sigma\!\left(U_rh_i+\sum_s\min(1,S_{i,s})u_i^{rs}\right).
\tag{I6}
\]

These are F Eqs.(17)–(18). The mass gate suppresses a role group whose total membership approaches zero; without it, a tiny nonzero group's normalized message can remain full strength. Receiver mixing occurs **after** each branch's nonlinearity. Implement stable log weights for nonempty groups and an explicit empty-group branch; a denominator epsilon changes exact algebra and must be documented.

With one-hot roles, nonempty \(S_{i,s}\) is an integer at least one and \(u_i^{rs}\) becomes the original class-restricted attention. Copying the original scores, value transforms, self transforms, biases, activation, edge convention and head merging therefore reproduces that **real-valued layer**. It does not reproduce the entire original actor/learner, its different input widths or its trained weights. Test equivalence with controlled adapters or zero-padded embeddings and matched matrices. The original [`edge_softmax(g['p2p'], ...)` and other relation calls][h-layer] implement separate denominators; DGL's [documented default normalizes incoming edges by destination][dgl-softmax]. Be careful that DGL uses source/destination edge order while the equations above are receiver-first.

A minimal counterexample is useful: with equal scores, one sender from each of two roles, scalar values 1 and 3, and zero self term, the hard HetGAT output is \(1+3=4\). Global attention gives 2. A one-hot router does not correct the wrong denominator. The full [verification protocol](research/SOFTROLE_VERIFICATION_PROTOCOL.md) includes this and further fixtures.

Use a homogeneous runtime node collection with an explicit \(K\times K\) relation axis, rather than rebuilding hard graphs from `argmax(p)`. Hard graph assignment would change the model and remove the intended soft gradient paths. A simple implementation evaluates relation scores/messages along existing edges; with dense linear expert maps, precompute transformed node values and gather them by edge. Parameters scale roughly as \(K^2d^2\); attention/message work grows with edges and \(K^2\), and dense communication remains quadratic in team size. Constant parameter shape does not imply constant execution cost. Basis sharing, as in [R-GCN][rgcn], is a later efficiency option, not part of the first equivalence claim.

Keep the feasibility report's globally normalized soft attention as an explicit ablation. It may generalize better or worse; hard compatibility alone does not establish an empirical advantage for class-wise normalization.

## 5. Action distribution, critic and training objective

Use one fixed six-symbol vocabulary for the grid actions: four moves, stay, and capture/extinguish. P bodies mask the last action; A bodies may use it. Every role expert of a given body receives the same physical feasibility mask. There must always be a legal fallback action. For expert categorical distributions \(\pi_{ir}\), the actor samples the actual mixture

\[
\pi_i(a)=\sum_rp_{i,r}\pi_{ir}(a),\qquad
\log\pi_i(a)=\operatorname{logsumexp}_r\left(\log p_{i,r}+\log\pi_{ir}(a)\right).
\tag{I7}
\]

This is F Eq.(19), not averaged logits. Compute entropy from the resulting six-action distribution. Deterministic soft gates need no sampled-role likelihood. Maintain gradients from a sender's encoder and gate through the receiver's action probability; do not serialize learned messages into detached tensors during the differentiable training forward pass. One ablation should use role-conditioned communication with a shared action head, to determine whether action experts are necessary.

A hidden actuation failure must not automatically remove an action from the visible mask: that would announce the failure. In the hidden condition, the action can remain syntactically valid but fail physically, with only allowed outcome feedback revealing it. Similarly, a sensor-failure flag, forced memory reset or revealed capability vector changes the information available to the actor.

For the variable-roster comparison, use a pooled centralized critic such as

\[
V_\psi(s_t)=v_\psi\!\left(g_t,\sum_{i\in I_t}\phi_\psi(s_{i,t},c_{i,t}),|I_t|\right),
\tag{I8}
\]

with valid-agent masks and privileged features available only in the training branch. This follows F Eq.(21) and [Deep Sets][deep-sets]. It is invariant under agent relabeling, but its count dependence is not a guarantee of extrapolation. A fixed-size critic could also train a size-compatible actor in the single-size regime; pooling is principally useful for mixed-size training.

First reproduce the released MAHAC/A2C loss unchanged. For the matched architecture experiment, a common team actor–critic objective is cleaner than indexing rewards by learned roles. Keep the original reward vector in logs and explicitly define the scalar training reward, for example the mean of currently active agents' rewards. This aggregation is a change from the released class-specific loss; apply it to every matched model and report original task metrics as well.

For a declared discounted team-return objective, its joint policy-gradient score contains

\[
\nabla_\theta\log\Pi_\theta(\mathbf a_t\mid\mathcal H_t)
=\sum_{i\in I_t}\nabla_\theta\log\pi_{\theta,i}(a_{i,t}\mid\mathcal H_t).
\tag{I9}
\]

Use the common team advantage with this sum. Dividing each step's actor loss by its active population changes relative weighting when population varies; it is not an innocuous padding normalization. Record any such modification as an objective choice. Cross-agent computation is another reason not to assign each gate only the reward of an arbitrarily selected class.

An explicit full-episode reference, with \(B\) sampled episodes, is

\[
G_{b,t}=\sum_{u=t}^{T_b-1}\gamma^{u-t}r^{\rm team}_{b,u},\qquad
\widehat A_{b,t}=G_{b,t}-V_\psi(s_{b,t}),
\tag{I10}
\]

\[
\mathcal L_{\rm actor}=-\frac1B\sum_b\sum_t\gamma^t
\operatorname{stopgrad}(\widehat A_{b,t})
\sum_{i\in I_{b,t}}\log\pi_{\theta,i}(a_{b,t,i}\mid\mathcal H_{b,t}).
\tag{I11}
\]

This is an on-policy Monte Carlo reference for discounted team return. A2C bootstrapping, advantage estimation, entropy and prediction losses must be specified as additions or estimators. Treat time-limit bootstrapping consistently with the declared finite-horizon or continuing objective.

Use complete graph timesteps and ordered sequences in the rollout representation. For the first small-grid implementation, full-episode unrolls offer a clear correctness reference; compare with a documented truncated-backpropagation configuration if memory requires it. Mask padded timesteps and agents. An agent departure ends its actor/memory chain, but does not terminate the continuing team's value target. Distinguish environment termination, time-limit truncation and per-agent lifetime resets.

A future PPO variant can reuse this actor, but needs recurrent likelihood replay, masks and common minibatch rules across baselines. It is a separate learner comparison. The existing feasibility report's PPO discussion was broader than this original HetNet implementation; PPO is not required to test the architecture.

## 6. Membership changes require an explicit lifecycle

Maintain an external map from `(physical identifier, generation)` to the agent's recurrent state, previous action, previous received context and previous gate. Survivors keep their own state when array rows move. A new lifetime receives the declared initial state, even if it reuses a vacated storage slot. A capability failure keeps the same memory, so the policy can infer the change from history. Returning-agent memory retention should be a separately specified experiment.

Apply exogenous events at a fixed boundary before observation generation. Remove departed bodies and graph edges; initialize arrivals; recompute observations, candidate links and masks; run the actor; execute only active agents' actions. Empty physical classes, isolated nodes and all-masked message groups need valid definitions. A zero-agent team is an environment termination/infeasibility case, not an input on which an action softmax is required to work.

Padding must not change active-agent outputs, role masses, critic pooling or loss. Size changes may allocate more memory and edge tensors, but may not resize learned layers, add trained identity embeddings or refit normalizers. This is the concrete meaning of a single parameterized actor family across sizes.

This lifecycle has a close implementation precedent in [GPL][gpl-paper]: its Wolfpack code gathers surviving hidden states and initializes entrants. It does not establish that GPL's learning objective or partner-policy assumptions match this cooperative all-agents-shared-policy setting. The transferable part is the storage and inference pattern. [Pinned code analysis](research/RELATED_ALGORITHM_CODE_AUDIT.md).

## 7. Same environments first, then named event extensions

The [environment audit](research/HETNET_ENVIRONMENT_IMPLEMENTATION_AUDIT.md) records observation widths, action semantics, rewards, termination and code-line references. The [HetNet supplement](papers/Seraj_2022_HetNet_Supplementary.pdf) supplies additional published experimental settings. Use a configuration ledger that distinguishes paper values, README examples, runtime defaults and proposed extensions.

| Environment | Initial author-code reference | What it can establish | Extension needed for recovery |
|---|---|---|---|
| PP | `predator_capture`, 3P/0A, 5×5, horizon 80 in README | Homogeneous graph/memory baseline; arbitrary row order and size interface | Membership events require new lifetime/termination rules; weak evidence of heterogeneous role discovery |
| PCP | Same environment, 2P/1A, 5×5, horizon 80 | Complementary perception/capture and frozen composition transfer | Active membership must replace cached counts; completion and reward semantics after departures need explicit definition |
| FC | `fire_commander`, 2P/1A, 5×5, vision 1, one initial fire, horizon 300, reward type 3 | Information flow from sensing to actuation under spreading fires | Capability/membership events, repaired observation construction, and declared information-demand protocol |

These are README examples, not a claim that every published result used that exact configuration. The PP/PCP sample commands use 2,000 epochs; FC uses 1,400. Epoch count alone is not a fair compute/sample budget, particularly when process count or team size changes. [Author README][h-readme].

Two substantive source findings affect the experiment:

**PCP changes task difficulty with team composition.** Released completion requires all agents to reach the prey and the A agents to capture, with reached agents becoming sinks. Adding agents adds obligations. This is not simply a fixed capture task with more interchangeable workers. Report the original objective for reproduction. For an open-team variant, explicitly choose the active-roster completion rule and report how joining/departing agents change demand. Never attribute easier completion caused by removing obligations to learned recovery.

**FC's paper and released simulator differ.** The paper describes discovery preceding extinguishing, but the release's action handler deletes a fire at the A agent's location without requiring it to be in `discovered_fire`. An uninformed searcher can therefore extinguish it. The reward-type-3 path also uses a per-agent fire penalty, source extinguishing reward and false-drop penalty; it is not merely a shared completion reward. In addition, PCP and FC observation construction appends array views before later blinding slices for A agents, creating an aliasing risk for overlapping observations. These are static source findings; the audit has not executed the simulator to quantify their effects.

FC also processes actions sequentially in array order: if two A agents extinguish the same cell, the first can receive credit and remove the fire before the second acts. Actor equivariance does not fix environment-level order dependence. Keep this behavior in the literal release reference; specify simultaneous resolution or an exchangeable tie rule before claiming full trajectory symmetry in the research variant. Initial placement samples distinct cells, so enforce the finite map's population/target capacity too.

Keep the literal release available for reproduction. Build a separately named, tested research variant with independent observation copies and a declared choice about the discovery requirement. A discovery-gated variant can enforce the paper's rule, but that correction alone does not force informative communication forever: stale discovery information can still be enough.

For the first event tests, use original fires/prey and report episodes completed before the scheduled event. For mechanistic analysis, a preregistered bank of nonterminal states with remaining unobserved demand can supply comparable interventions. This is a conditional experiment and must not replace unconditional performance without saying so.

For sustained information-role recovery, add a **separate sustained-ignition FC task**: exogenous new ignitions follow a fixed manifest, all agents retain the same movement/fire rules, success over a fixed window depends on detecting and extinguishing fresh fires, and the horizon/reward/termination delta is published. The legacy environment's natural spread supplies new cells only while fire persists and terminates when fire is out; it does not guarantee fresh post-event demand. An overlapping-capability variant is another separate extension, useful for testing whether flexible agents change task responsibilities rather than merely classify physical types. [FireCommander domain paper][fc-paper].

## 8. What to borrow from related algorithms—and what to compare

The [related-code audit](research/RELATED_ALGORITHM_CODE_AUDIT.md) pins seven author repositories spanning six algorithm families (GPL and PO-GPL have separate repositories), records inspected files and hashes, and distinguishes the supplied implementation from paper-level claims. These references guide components and controls; they are not drop-in substitutes for the HetNet experiment.

| Reference | Concrete author-code finding | Proposed use and boundary |
|---|---|---|
| [ROMA][roma-paper] | [Latent agent code][roma-code] computes a per-step Gaussian role embedding and uses a hypernetwork for the final Q layer; controller/learner paths reshape using configured `n_agents` | Dynamic latent-role baseline. Roles are already dynamic in ROMA. A categorical actor with matched communication is **ROMA-style**, not a full ROMA/QMIX reproduction |
| [RODE][rode-paper] | [Controller][rode-code] uses a role interval and separate role recurrence; action-effect embeddings and clustering produce role action spaces | Useful temporal-role/action-representation reference. Its hard action restrictions differ from physical feasibility and soft communication; preserve the reported target-task fitting distinction in F §3 |
| [CASH][cash-paper] | [JAX actor][cash-code] contains an `AgentHyperRNN` decoder; inspected fire observations concatenate population-dependent capability/position features | Strong capability-conditioned policy comparator. Port its decoder idea onto the common observation contract; do not claim that its inspected raw input path directly solves arbitrary-N execution |
| [Capability-aware GNN][cap-paper] | [Shared graph actor][cap-code] has runtime count reshapes; evaluation updates the count. Supplied GNN config lacks recurrence | Closest role-free graph control: add the same history evidence as SoftRole; its shared weights are not inherently count-indexed |
| [GPL][gpl-paper] | [Wolfpack inference code][gpl-code] explicitly preserves survivor states and initializes entrants | Borrow roster-state mechanics; its learning/partner setting is different |
| [ROCO][roco-paper] | [Role/message module][roco-code] emits `n_agents × latent_dim × 2` outputs | Related role-conditioned communication; variable-N deployment needs redesign of this path. Full publisher text remains unavailable; this claim is verified from code |

The central experiment should first compare: (a) fixed physical-role HetGAT on the common infrastructure; (b) recurrent capability-conditioned shared attention; (c) a wider shared-attention control with similar parameter/payload/compute budget; and (d) SoftRole. Add a common expert bank with uniform routing to isolate adaptive routing. This family distinguishes roles from capacity, better input representation and dynamic-roster engineering.

Next add the CASH-style decoder and ROMA-style dynamic latent actor, sharing the same inference evidence, communication rounds, action masks and learner. Published algorithms trained in their original environments are useful external references, but cannot identify the cause of a gain here. A faithful ROMA reproduction would require its complete value-learning stack and benchmark protocol, not just its latent module.

ROMA's inspected evaluation path still samples Gaussian roles. A matched actor must state whether it uses role means or samples. If sampled latents enter a policy-gradient/PPO port, account for their joint likelihood or marginalize them consistently; drawing fresh roles while replaying an action-only likelihood is not an ordinary deterministic-gate update. This is an adaptation issue, not a criticism of ROMA's original value-learning objective.

To test the proposed mechanism on the same frozen checkpoint, fix surviving agents' gates to their pre-event values, replace them with uniform gates, suppress a directed role-pair message branch, or shuffle useful message content under matched delivery opportunities. For arrivals in a fixed-gate intervention, freeze the gate at their first permitted inference; do not assign them an undefined pre-event gate. Report both message-only and action-only routing ablations. These interventions follow the feasibility report and [Lowe et al.'s cautions about measuring communication][comm-measure].

## 9. Frozen-policy experiments: support, events and immutable state

An inference runner should construct no optimizer, replay buffer or learner. Load an actor-only checkpoint with its architecture/configuration and normalization statistics, use evaluation behavior and disabled gradient recording, and compare all parameter and non-memory buffer hashes before and after episodes. Memory, RNG state, local predictions and role gates may change. Learned tensors, normalization statistics, calibration and gate-temperature schedules may not. `eval()` or `no_grad()` alone does not enforce all of this.

The released HetGAT `--eval` path uses `Trainer` and exits before optimizer updates. It is therefore inaccurate to say that existing evaluation necessarily trains weights. A dedicated runner is still preferable: the old path uses training-oriented buffers, fixed-roster assumptions and process exits, and is awkward for paired event conditions. [Verification protocol, §4](research/SOFTROLE_VERIFICATION_PROTOCOL.md#4-frozen-evaluation-is-an-explicit-state-contract).

Predeclare three training regimes and evaluate each with frozen parameters:

| Regime | Training support | Interpretation of held-out evaluation |
|---|---|---|
| T0 | One small composition, stationary capabilities, no churn | Strongest subset-to-larger-population and first-event transfer test |
| T1 | Several compositions and capability values between episodes | Transfer to unseen sizes/combinations and within-episode recombination of known conditions |
| T2 | Membership and capability events within a bounded training distribution | Recovery under held-out sizes, timing, severity, combinations or frequency |

Run T0 even if T2 is more successful. A policy trained on failures is not encountering failures for the first time in evaluation. Keep a support ledger of body interfaces, capability ranges, compositions, event types, observation feedback and communication graphs. New identities with identical bodies are a weaker split than genuinely held-out capability combinations.

For paper-linked PCP size tests, train at **2P/1A** and load the same actor at **3P/3A** and **4P/6A**, reporting both total count and composition. Include imbalanced compositions and smaller feasible teams. Begin on the same 5×5 map so Eq.(I1)'s input contract remains unchanged. A larger map changes the position encoding and geometry; test that only after a separately validated spatial encoder is trained consistently across every matched baseline. Density, task demand and graph degree should appear as separate factors rather than being treated as pure team-size effects.

Use one event per episode initially: redundant-agent departure; fresh-lifetime arrival; equal-size replacement; sensing degradation; actuation degradation; and information-source handover at fixed roster size. Announced and hidden changes are separate conditions. Event manifests fix time, victim, capability, location and exogenous randomness independently of each policy's internal gates. Removing the sole essential actuator can make the task impossible; declare that stratum and use feasible-survivor references rather than scoring impossibility as failed adaptation.

Only then test combined events and repeated churn. Sweep dwell time between events relative to observed recovery time. A policy cannot infer an invisible failure or propagate evidence faster than its communication schedule permits, regardless of how many experts it has (F Eq.(33)). Mild capability extrapolation remains extrapolation; show its distance from training support and distinguish it from interpolation.

## 10. Acceptance tests tied to the feasibility mathematics

The [verification protocol](research/SOFTROLE_VERIFICATION_PROTOCOL.md) supplies exact fixtures and suggested numerical tolerances. The following are required before interpreting learned recovery curves.

| Mathematical basis | Implementation check | Limit of the conclusion |
|---|---|---|
| F Proposition 1, Eqs.(22)–(23) | Permute observations, metadata, memories, masks and both graph axes; inverse-permute gates, action distributions and next memory | Checks complete actor equivariance, not good returns at new sizes |
| F Proposition 2 | Identical histories in the same graph-symmetry orbit yield identical deterministic gates | A unique deterministic winner would reveal an unintended asymmetry; roles do not solve anonymous leader election automatically |
| F Eqs.(17)–(18), (25) | Inject exact one-hot roles and copied original layer parameters; test separate relation denominators and head merging | Exact real-layer reduction, not equality of full trained algorithms |
| F Eq.(18) | Sweep a role's total mass toward zero; test empty groups and no neighbors | Bounded contribution vanishes with mass; numerical epsilon choices remain visible |
| F Eqs.(26)–(27) | Perturb gates with expert functions/distributions fixed; verify the local bounds | Does not bound a changing recurrent policy as a whole |
| F Eq.(29) | Add distractors at fixed useful-score gap | Measures attention dilution; duplication is not universally a desired invariance |
| F Proposition 3, Eq.(31) | Verify a bounded toy normalized-attention layer with known Lipschitz constants | Does not certify the full class-wise mass-gated actor or task return |
| F Eq.(33) | Hide a failure so pre-evidence histories match exactly; check actor-visible tensors for leakage | Diagnosis before informative evidence cannot be credited to the role model |
| F Eqs.(34)–(35) | Treat contraction and joint-action discrepancy as assumptions requiring separate evidence | A recovery curve does not prove GRU contraction or a worst-case return guarantee |

Also test garbage-filled padding, singleton/empty physical classes, row reorder across time, departure followed by slot reuse, checkpoint loading at a larger \(n\), mixture likelihood/entropy, nonzero sender-to-receiver gradients, and replayed action-probability agreement before the first learner update. Use a reference float64 implementation for algebraic fixtures before benchmarking accelerated reductions. Test probabilities rather than independent action samples unless the random draws are permuted too.

No trained parameter shape may depend on deployment population. Record the parameter-name/shape/value signature for each evaluation condition. Role labels themselves are non-identifiable: permuting gate columns and all corresponding experts together should leave the policy unchanged; assigning a semantic name such as “scout” requires separate behavioral evidence.

## 11. Outcomes, uncertainty and a decision rule

Report legacy success, return and completion time on the original tasks. Include failures/timeouts rather than averaging completion time only over successes. For the sustained-demand FC extension, preregister the fraction of scheduled post-event **source ignition events** whose designated source cell is extinguished within a fixed deadline. Count all scheduled events by event ID, not only discovered fires; define how repeated ignitions in an already burning cell create a renewed source obligation. Report propagated burning-cell burden separately, so source-event success does not hide uncontrolled spread. Recovery latency needs explicit informative-observation and successful-response events.

Use paired intact/event conditions and the recovery-loss contrast in F Eq.(36), while showing all four raw outcomes. Compare against a feasible post-change reference separately: lost physical resources can reduce attainable performance even under perfect coordination. A trained privileged reference is not a mathematical optimum.

Log event time, first informative local evidence, changed capability prediction, changed gate, useful received message and successful response. This makes diagnosis delay, communication delay and action execution distinguishable. Role entropy and F Eq.(37)'s differentiation statistic are diagnostics, not task endpoints.

Keep episodes completed before the event in unconditional outcomes and report their frequency. Dropping them selects different states under different policies. State-bank interventions can answer a conditional mechanism question if their sampling rule is fixed in advance. Treat unrecovered episodes as censored or failed at a declared horizon; avoid success-only latency averages.

Independent training seeds are the inferential units, with evaluation episodes nested within each seed. Pair exogenous manifests across models, separate environment and action RNG streams, select checkpoints only on development data, and report seed-level intervals and individual runs. Use a small pilot to estimate variability before allocating confirmatory seeds. The recommendations in [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*][stats-paper] and its [rliable code][rliable] support uncertainty-aware reporting; many evaluation episodes do not compensate for only one or two trained policies.

Proceed with a thesis claim about soft roles only if the frozen model reduces post-event loss relative to the matched capability-conditioned recurrent graph policy, the effect survives capacity/training-support controls, and gate/message interventions affect the predicted recovery mechanism. If the role-free policy matches it, the useful result is that recurrent capability conditioning suffices on these tasks. If only T2 works, report trained robustness to held-out events rather than spontaneous recovery from a static training population.

## 12. Concrete future file plan and implementation order

The following paths describe a **future isolated fork**, not files implemented by this report. If implementation is later requested under the present folder restriction, place that fork inside `softrole/`; do not retrofit the unrelated application. Preserve the unmodified evidence snapshot.

| Proposed module | Responsibility | Upstream reference |
|---|---|---|
| `hetgat/softrole_actor.py` | Physical adapters, shared recurrence, capability predictor, gate and action experts | `uavnet.py` |
| `hetgat/graph/softrole_real.py` | Eqs.(I4)–(I6), head merging, debug diagnostics | `graph/fastreal.py` |
| `hetgat/roster.py` | Active-lifetime memory map, graph/mask assembly | `utils.py`; GPL lifecycle pattern |
| `hetgat/softrole_policy.py` | Exact masked mixture likelihoods and common team objective | `policy.py` |
| `training/sequence_collector.py` | Whole-graph trajectories, episode/lifetime masks and recurrent replay | `trainer.py`, `multi_processing.py` |
| `envs/open_team_events.py` | Immutable event manifests and explicitly versioned environment deltas | Original PP/PCP/FC environment classes |
| `evaluation/frozen_runner.py` | Actor-only loading, immutable learned state, paired conditions and logs | Replace dependence on process-exiting evaluation paths |
| `tests/` and `configs/` | Mathematical fixtures, environment contracts, reproduction/support ledgers | Verification protocol and pinned source settings |

Implement in this order:

1. **Reproduce and characterize.** Resolve dependency compatibility, run tiny deterministic environment checks, verify observation isolation and completion rules, and establish original PP/PCP/FC learning references. Publish every compatibility or semantic delta.
2. **Make rosters explicit.** Build shared observation/action and lifetime contracts. Run the fixed-role and role-free controls through the same infrastructure; verify with tests that permutations, padding and arrivals satisfy the mathematical contracts.
3. **Add the soft layer.** Pass hard-reduction, empty-role, likelihood and gradient fixtures before training. Train the smallest static task and compare learning behavior under identical budgets.
4. **Freeze and increase population.** Export actor-only artifacts, test T0 size/composition transfer, and verify immutable parameter signatures at every size.
5. **Introduce single events.** Compare announced versus hidden capability changes, then arrivals, departures and replacements. Validate evidence availability and surviving physical feasibility.
6. **Test recovery mechanisms.** Add sustained demand and overlapping capabilities only in explicitly named task extensions, then T1/T2, matched related-algorithm controls, causal interventions and repeated churn.

Measure collection speed, memory, optimizer throughput, payload and edge count during the pilot before assigning a compute budget. Equal environment steps and equal agent decisions are different budgets when \(n\) changes; report both, plus wall time. There is no evidence yet for a credible fixed number of GPU-hours or an expected performance gain.

The implementation is technically feasible as a focused HetNet extension. The largest work items are the lifecycle and evaluation contracts, fair reward/training comparisons, and environment semantics that actually require fresh information after a change. The soft relation equations themselves are compact; establishing that they improve frozen-policy recovery is the substantive research task.

## Evidence and supporting material

- [Original feasibility report](SOFTROLE_FEASIBILITY_REPORT.md) and [mathematical audit](research/MATHEMATICAL_AUDIT.md).
- [HetNet environment and reproduction audit](research/HETNET_ENVIRONMENT_IMPLEMENTATION_AUDIT.md).
- [Related algorithms: pinned author-code audit](research/RELATED_ALGORITHM_CODE_AUDIT.md) and [source manifest](research/implementation_sources/related/SOURCES.json).
- [Theorem-linked verification and frozen-policy protocol](research/SOFTROLE_VERIFICATION_PROTOCOL.md).
- [Paper library](papers/README.md), [consolidated provenance](papers/manifest.json), and [work log](agents.md).

[hetnet-paper]: https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf
[h-readme]: https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/README.md
[h-main]: https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/main.py#L230
[h-model]: https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/uavnet.py#L83
[h-graph]: https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/utils.py#L49
[h-layer]: https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/graph/fastreal.py#L147
[h-state]: https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/uavnet.py#L359
[h-loss]: https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/policy.py#L778
[h-trainer]: https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/trainer.py#L596
[dgl-softmax]: https://www.dgl.ai/dgl_docs/generated/dgl.nn.functional.edge_softmax.html
[deep-sets]: https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
[rgcn]: https://arxiv.org/abs/1703.06103
[fc-paper]: https://arxiv.org/abs/2011.00165
[roma-paper]: https://proceedings.mlr.press/v119/wang20f.html
[rode-paper]: https://openreview.net/forum?id=TTUVg6vkNjK
[cash-paper]: https://proceedings.mlr.press/v305/fu25a.html
[cap-paper]: https://proceedings.mlr.press/v229/howell23a.html
[gpl-paper]: https://proceedings.mlr.press/v139/rahman21a.html
[roco-paper]: https://www.sciencedirect.com/science/article/pii/S0957417425030374
[roma-code]: https://github.com/TonghanWang/ROMA/blob/2d6044899fdab0032a71171931245205248ef090/src/modules/agents/latent_ce_dis_rnn_agent.py#L83
[rode-code]: https://github.com/TonghanWang/RODE/blob/e1715eaf34e9b1d99bf95466f3c3c161392ff8e1/src/controllers/rode_controller.py#L40
[cash-code]: https://github.com/GT-STAR-Lab/CASH/blob/20a31170b442cf16b50cd43aa3f5ae547e2046f4/jaxmarl/policies/policies.py#L128
[cap-code]: https://github.com/GT-STAR-Lab/cap-comm/blob/70814520a3ff84ee5e13fe516c933e8e0d53859e/src/modules/agents/gnn_agent.py#L68
[gpl-code]: https://github.com/uoe-agents/GPL/blob/83bf42d9b02a4a520381f37bed3cb662d86df701/Open_Experiments/Wolfpack/GPL-Q/Agent.py#L150
[roco-code]: https://github.com/surebo/ROCO/blob/5aafe9ed4fb8031e3c915888fe67fb7b8cc3147a/src/modules/agents/roco_agent.py#L20
[comm-measure]: https://arxiv.org/abs/1903.05168
[stats-paper]: https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
[rliable]: https://github.com/google-research/rliable
