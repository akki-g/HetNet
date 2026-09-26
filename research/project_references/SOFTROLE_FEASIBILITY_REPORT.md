# SoftRole-HetGAT under changing teams and capabilities

**Feasibility and research-design report · 25 September 2026**

This report reviews both existing SoftRole documents, the supplied HetNet paper, the Team Changes and Information-Role Recovery guide, relevant current code, and primary literature. It proposes an extension; it does **not** report a trained SoftRole model or experimental validation. Mathematical statements below are either attributed results or explicitly identified derivations with assumptions. Source PDFs, access limitations, and bibliographic metadata are tracked in [papers/README.md](papers/README.md) and [papers/manifest.json](papers/manifest.json).

## 1. Assessment

**The direction is feasible and fits the information-role recovery question. Its strongest research formulation is learned reorganization after changing capability and team composition, with a frozen policy.** The implementation should retain HetNet's idea that different relationships merit different message transformations, while learning the current role of each agent from its history and available team context.

Three qualifications determine whether this becomes a convincing contribution:

1. Learned dynamic roles already exist. ROMA can change roles during an episode; it is not a static-role baseline. RODE also evaluates role-policy transfer to larger teams. Capability-aware GNN policies, CASH, GPL, and recent open-team work cover significant parts of the proposed generalization setting.
2. Permutation **equivariance** makes a policy indifferent to how agents are numbered. It does not establish successful transfer to larger teams, novel capability values, or new partners. Those are separate distribution shifts.
3. A useful latent role must depend on current team need as well as physical capability. Estimating “my sensor is weak” and deciding “I should now relay rather than scout” are different computations. The proposed architecture should expose that distinction.

A defensible central question is:

> Does a recurrent, permutation-equivariant policy that uses inferred capabilities and team context to update soft roles recover useful information flow and task performance after membership or capability changes better than equally informed shared graph policies and conditional-policy baselines, without updating its trained weights?

This is a candidate empirical and architectural contribution, not a verified novelty claim. The most promising evidence would be a benefit specifically from coupling inferred roles to directed message transformations, beyond capability estimation, memory, training diversity, and extra parameters.

| Aspect | Assessment | Condition |
|---|---|---|
| Shared soft action/message routing | High feasibility | Small fixed role bank, correct action likelihoods |
| New permutations of a heterogeneous team | Exact architectural property is possible | Every relevant input, recurrent state, mask, and entity representation must transform consistently |
| New compositions and modest unseen team sizes | Plausible, unproven | Size-compatible observations, exposure to diversity, fair task scaling |
| Announced sensor or actuator deterioration | High feasibility as a study | Capability changes are provided equally to all methods |
| Hidden deterioration inferred from experience | Moderate feasibility | Detectable local evidence, memory, sufficient time to diagnose |
| Repeated arrivals and departures during episodes | Moderate engineering feasibility | Complete active-agent, identity-lifetime, and memory contracts |
| Arbitrary new capabilities or unfamiliar policies | Outside the initial claim | Requires additional interfaces or ad hoc teamwork machinery |
| Broad novelty: “learn roles and generalize” | Unsupported | Strong prior art already exists |

## 2. What changes from the earlier notes

The [architecture note](SOFTROLE_HETGAT_ARCHITECTURE.md) contains useful mechanisms: a simplex-valued router, ordered role-pair message experts, a genuine action-distribution mixture, and causal gate interventions. The [critical review](SOFTROLE_HETGAT_CRITICAL_REVIEW.md) correctly warns about role collapse, unobservability, communication cost, latent-label ambiguity, and overclaiming HetNet equivalence.

Several aspects need updating for this request:

| Earlier element | Assessment now |
|---|---|
| Hidden detector/capturer assignment sampled at reset | Useful implementation check; insufficient evidence of within-episode role recovery |
| Dynamic populations deferred until after constrained allocation and budgeted topology | Reorder the project: population change and recovery are now central; transport allocation and learned topology are optional |
| Local feedforward role router | Adequate for announced capabilities; insufficient for a claim about inference from action outcomes when current observation is ambiguous |
| Replace physical class by learned role everywhere | Retain physical observation/action adapters and feasibility rules; learn computational responsibility separately |
| Exact reduction to the complete original HetNet | Only valid with matching encoders, class-wise attention normalization, message codec, aggregation, heads, and policies; the simple global-softmax model has a narrower reduction |
| Variable size from an equivariant communication layer | The complete actor must be size compatible; current observations also contain ordered, variable-width teammate features |
| SCoUT as an execution-time shared soft-role precedent | Its inspected version uses grouping in training and discards the grouping module at execution; see literature audit |
| July implementation paths and five-action categorical policy | Stale for current `src/commstudy`; the managed current PCP workflow uses continuous actions |
| Fixed-N critic prevents larger-team actor deployment | Too strong: a critic is unnecessary at inference. A size-compatible critic is valuable for mixed-size training, but current actor/spec limitations already block direct reuse |

The [Team Changes guide](../../03_Team_Changes_and_Information_Role_Recovery.pdf) remains a good **first experimental fixture**: two potential scouts and a responder, with a surviving backup after sensor failure. Its earlier preference to avoid architecture work was a scope recommendation, not evidence that this broader question is invalid. The current request explicitly asks for the broader architecture assessment.

## 3. Prior art and the real distinction from ROMA

The detailed evidence audit is in [LITERATURE_ROLE_OPEN_TEAMS.md](research/LITERATURE_ROLE_OPEN_TEAMS.md); capability and failure evidence is in [CAPABILITY_AND_RECOVERY_EVIDENCE.md](research/CAPABILITY_AND_RECOVERY_EVIDENCE.md). The following table is a comparison, not a leaderboard.

| Work | Established mechanism or setting | What remains to test here |
|---|---|---|
| [HetNet, AAMAS 2022](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf) | Supplied heterogeneous classes; class-conditioned communication and policies | Replace externally supplied computational routing with evidence-driven role updates |
| [ROMA, ICML 2020](https://proceedings.mlr.press/v119/wang20f.html) | Stochastic continuous role embeddings, role-conditioned Q-functions, identifiability/specialization losses; dynamic behavior under health/death changes | Explicit learned message transformations plus a controlled entry/exit and hidden-capability recovery protocol |
| [RODE, ICLR 2021](https://openreview.net/forum?id=TTUVg6vkNjK) | Action-effect-based roles and a role selector; larger-team transfer with frozen policies | Strictly zero parameter updates and fixed action interface; online evidence-to-message-role recovery |
| [Learning to Transfer Role Assignment Across Team Sizes, AAMAS 2022](https://arxiv.org/abs/2204.12937) | Role assignment and cross-size transfer with further learning | Evaluate frozen deployment separately from fine-tuning |
| [Capability awareness/communication, CoRL 2023](https://proceedings.mlr.press/v229/howell23a.html) | Continuous capability features and shared GCN policies for changed teams | Whether latent role routing adds value beyond a capability-conditioned GNN |
| [CASH, CoRL 2025](https://proceedings.mlr.press/v305/fu25a.html) | Capability-conditioned generated decoder weights; unseen teams and robots, deterioration and larger-team deployment | Inferred uncertain capability plus a learned directed information-role mechanism |
| [GPL, JMLR 2023](https://jmlr.org/papers/v24/22-099.html) | Type inference, action prediction, value modeling in open ad hoc teams; partial-observation extensions | Jointly learned communication for a shared-policy population is a different training/execution setting |
| [ROCO](https://doi.org/10.1016/j.eswa.2025.129421), [GRDC](https://ojs.aaai.org/index.php/AAAI/article/view/40185), [SCoUT](https://arxiv.org/abs/2603.04833) | Roles/groups and communication already intersect | Identify the exact routing and recovery difference; no “first roles plus communication” claim |
| [GIRA](https://doi.org/10.1016/j.neucom.2026.134437) | Publisher record describes role adaptation, dynamic population sizes and unseen-size generalization for homogeneous agents | Strong overlap; full-text access limitations prevent certifying its event protocol |
| [CE-CM, July 2026 preprint](https://arxiv.org/abs/2607.27177) | Hidden partner-capability inference; appendix tests changing capability across tasks | Stepwise recurrent inference, learned communication, and multiple changing agents |
| [HOLA, July 2026 preprint](https://arxiv.org/abs/2607.04972) | Within-episode team changes, unseen partners and scales | Causal information-role recovery under a controlled capability/task contract |

ROCO, GIRA, SORA and [g-MAT](https://doi.org/10.1016/j.knosys.2026.116612) have verified primary records but access limitations in this audit. g-MAT adds role grouping with within-group and cross-group attention to the overlap. Their abstracts establish overlap; absent full text does not establish that they lack the proposed mechanism. Recent SubMAPG and PLATO preprints are also tracked as open-team/allocation overlap in the paper library.

### ROMA's mathematics and the proposed change

ROMA Section 3 uses a stochastic continuous role, schematically

\[
\rho_i^t\sim p_{\theta_\rho}(\rho_i^t\mid o_i^t),\qquad
\theta_i^t=f(\rho_i^t),\qquad Q_i(\tau_i^t,a_i;\theta_i^t).
\tag{1}
\]

It trains role representations with value learning and auxiliary objectives promoting identifiable and differentiated behavior. This is not a fixed, predefined class assignment. The main-paper health/death example already illustrates changing specialization. See the source audit for the distinction between its execution role encoder and training trajectory-inference network.

The proposed gate instead has a fixed expert index set and is recomputed from local history plus legal team context:

\[
p_i^t=q_\phi(h_i^t,\widehat c_i^t,g_i^t)\in\Delta^{K-1},\qquad
M_{ij}^t=\sum_{r,s}p_{i,r}^tp_{j,s}^tM_{r\leftarrow s}(h_j^t).
\tag{2}
\]

Its intended distinction is **what the latent variable controls and how recovery is tested**: a gate influences both behavior and the translation of messages between roles, while an explicit capability estimate can change with observed outcomes. Replacing a continuous latent with a categorical mixture is an architectural choice, not by itself a research contribution. Neither method's latent roles are guaranteed to acquire human semantic labels.

RODE is especially relevant to the user's subset-to-larger-team idea. Its final Section 4.3/Figure 5 transfers role policies from six to as many as eighteen controlled units. For the expanded action set, it collects **50k new-task samples to train an action encoder**, then maps new actions to the representations and role action spaces of similar old actions; the role policies remain fixed. Thus “frozen policy” and “no deployment-time learning anywhere” must be separated. The strongest proposed experiment keeps the action vocabulary fixed and prohibits **all** parameter updates during evaluation.

## 4. Formalizing the intended problem

Let the active team be \(\mathcal I_t\), of size \(n_t\), and use a disjoint-union state space over finite teams. The Markov state is

\[
s_t=(x_t,\mathcal I_t,\{c_{i,t}\}_{i\in\mathcal I_t},d_t,\eta_t),
\tag{3}
\]

where \(x_t\) is physical state, \(c_{i,t}\) is effective capability, \(d_t\) is task demand, and \(\eta_t\) includes event-process state if needed. A kernel evolves physics, arrivals/departures, and capability changes. Observations follow a kernel \(O\), rewards are shared, and initial-state, horizon/termination and discount conventions are specified. This is an open-population cooperative partially observable model; it must not silently use a fixed-agent Dec-POMDP while changing its membership.

A physical capability vector might contain maximum speed, sensor reliability, sensing radius, and payload. These coordinates retain the same meaning during training and deployment. A computational role is a policy responsibility such as searching, relaying, or responding. Changing the latter cannot restore a broken sensor or grant a new actuator.

Let \(\mathcal F_{i,t}\) contain precisely what agent \(i\) can know: its observations, past executed actions, local execution feedback, received packets with age/validity, and any declared membership/capability announcements. The decentralized actor must be measurable with respect to \(\mathcal F_{i,t}\). The privileged training state never enters this information set.

Define an allowed directed graph with **receiver-first indexing**:

\[
A_{ij,t}=1\quad\text{iff sender }j\text{ may deliver to receiver }i.
\tag{4}
\]

If storage uses padded slots with active mask \(\ell_{i,t}\), the effective mask is

\[
\widetilde A_{ij,t}=\ell_{i,t}\ell_{j,t}\,1_{i\ne j}A_{ij,t}.
\tag{5}
\]

An inactive agent has no actions, messages, physical interactions or actor-loss contributions. A sensor-impaired agent can remain active, move and communicate. A radio outage changes packet delivery without changing physical sensing. These are different interventions.

Train shared parameters on a declared distribution of rosters, capabilities and event schedules:

\[
\max_\theta\;\mathbb E_{e\sim\mathcal D_{\rm train}}
\mathbb E_{\pi_\theta,e}\left[\sum_{t=0}^{H-1}\gamma^t R_t\right].
\tag{6}
\]

At evaluation \(\theta\) is fixed; hidden states, capability beliefs, roles and received information may update. This is adaptation by inference. Updating optimizer state, weights or a new-agent embedding table is a different evaluation condition.

Distinguish four transfers: reordered agents; new combinations of known capabilities; new values in a known capability schema; and unfamiliar teammate policies. The first is a symmetry test. The last is ad hoc teamwork and requires partner-diversity evaluation beyond adding agents running the same trained policy.

## 5. Starting from the actual HetGAT operation

In HetNet Section 4.3, attention is normalized separately over neighbors in each sender class. Rewriting the paper's notation using receiver \(i\), sender \(j\), and class \(c_i\):

\[
e_{ij}^{r\leftarrow s}
=\operatorname{LeakyReLU}\!\left((a_{rs})^\top[W_rh_i\Vert V_{rs}h_j]\right),
\quad
\alpha_{ij}^{r\leftarrow s}
=\frac{\exp e_{ij}^{r\leftarrow s}}
{\sum_{k\in\mathcal N_s(i)}\exp e_{ik}^{r\leftarrow s}},
\tag{7}
\]

\[
h_i'=\sigma\!\left(W_{c_i}h_i+
\sum_s\sum_{j\in\mathcal N_s(i)}
\alpha_{ij}^{c_i\leftarrow s}m_{ij}^{c_i s}\right).
\tag{8}
\]

The Binary variant includes sender encoding, Gumbel-Softmax communication and receiver decoding; the Real variant removes that codec. Original class encoders, heads and policy structure also matter. [HetNet, Equations 2–4 and Section 6.2](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf).

Replacing a selected message operator by a role mixture is exact at one-hot gates. Replacing all class-wise denominators with one global neighbor denominator changes Equation (8). For equal attention scores with two neighbors of class A and one of class B, HetNet gives each class total mass one, whereas global normalization gives masses \(2/3\) and \(1/3\). No one-hot gate fixes that difference.

The main report therefore distinguishes a **minimal HetNet-inspired model** from a **faithful relaxation of its typed attention operator**. The detailed reconstruction and proof conditions are in [MATHEMATICAL_AUDIT.md](research/MATHEMATICAL_AUDIT.md).

## 6. Recommended SoftRole architecture

### 6.1 A size-compatible local representation

Encode ego measurements separately from a set of locally visible entities:

\[
u_{i,t}=\left[\operatorname{ego}_{i,t},\;
\sum_{e\in\mathcal E_{i,t}}\varphi(\operatorname{type}_e,\operatorname{rel}_{ie,t},\operatorname{valid}_{e,t}),\;
|\mathcal E_{i,t}|\right].
\tag{9}
\]

A masked attention/set encoder is an alternative. Fixed-width outputs alone are insufficient: do not hide an ordered concatenation of teammate coordinates inside `ego`. Sum pooling carries counts; normalized means/softmax attention should receive relevant count or mass features explicitly. Counts are local unless the communication protocol makes global membership available. Mask padding before every reduction.

Use the same recurrent encoder across agents:

\[
h_{i,t}=\operatorname{GRU}_\theta\!\left(h_{i,t-1},
[u_{i,t},a_{i,t-1},y_{i,t-1},m_{i,t-1}]\right),
\tag{10}
\]

where \(y\) denotes legally observable execution outcomes, not critic advantage or hidden labels. Shared memory updates preserve equivariance when memory is carried with the agent. A joining agent starts from a declared prior; a departed agent's memory is removed. Reused array slots must receive a new lifetime identifier and reset memory. Surviving agents retain theirs.

### 6.2 Separate capability estimation from role choice

Optionally decode a belief or uncertainty-bearing estimate

\[
b_{i,t}(c)=q_\eta(c\mid h_{i,t}),\qquad
(\widehat c_{i,t},u^c_{i,t})=\operatorname{summary}(b_{i,t}).
\tag{11}
\]

For discrete capability hypotheses, the conceptual filtering target is

\[
b_{i,t}(c)\propto p(y_{i,t-1},o_{i,t}\mid c,a_{i,t-1},\mathcal F_{i,t-1})
\sum_{c'}K(c\mid c')b_{i,t-1}(c').
\tag{12}
\]

This equation assumes a specified likelihood and change kernel; a neural recurrent estimator is only an approximation to it unless those quantities are modeled and validated. The dynamics of peers and the environment can confound the likelihood. A zero capability-transition rate incorrectly rules out failures; unlimited resetting can discard useful evidence. Compare learned recurrence against a simple known-likelihood filter where a controlled fixture allows one.

A self-supervised outcome predictor can train \(h\) or \(b\) without role labels:

\[
\mathcal L_{\rm pred}=-\mathbb E\log p_\omega(y_{i,t}\mid h_{i,t},a_{i,t},\text{local task context}).
\tag{13}
\]

This does not identify physical capability automatically: many latent descriptions can predict outcomes. Report prediction, calibration and capability-probe performance separately. If true capability labels train the estimator, describe that as supervised capability estimation rather than unsupervised role discovery.

### 6.3 Team-relative context and a legal communication schedule

A capable scout may become the responder when a better scout joins. Purely local capability classification cannot express that counterfactual without new context. Use one role-agnostic exchange over allowed neighbors:

\[
g_{i,t}=\operatorname{SetAgg}\{(h_{j,t},\widehat c_{j,t},u^c_{j,t},\operatorname{age}_{ij,t}):j\to i\},
\quad
p_{i,t}=\operatorname{softmax}\!\left(f_\phi(h_{i,t},\widehat c_{i,t},u^c_{i,t},g_{i,t})/T\right).
\tag{14}
\]

The next exchange supplies role-conditioned task content. Equivalently, use prior-step received context to compute current roles in a single new round, accepting one-step latency. Do not compute roles from current role-conditioned messages and then claim those same messages used the newly computed roles without another round or a defined iterative solver.

A practical two-round protocol is: send a local context header; infer roles; send \((p_j,h_j)\); receiver computes its role-pair translation. A bounded-degree graph gives local, potentially incomplete team knowledge. Equalize rounds, header access and payload width in controls. Team-wide counts, quotas or capabilities are privileged unless actually announced or propagated.

### 6.4 Minimal global-attention model

For a first architecture test use a small \(K\), shared content attention, full ordered message experts and one gate:

\[
v_{ij}=\sum_{r,s}p_{i,r}p_{j,s}W_{rs}h_j,
\quad e_{ij}=\frac{q(h_i)^\top k(h_j)}{\sqrt d}+p_i^\top Bp_j,
\tag{15}
\]

\[
\alpha_{ij}=\operatorname{softmax}_{j:\widetilde A_{ij}=1}(e_{ij}),\quad
m_i=\sum_j\alpha_{ij}v_{ij},\quad
z_i=\operatorname{Fuse}(h_i,m_i,g_i).
\tag{16}
\]

Empty neighborhoods produce exactly zero aggregate with finite outputs. Cross-role edges remain available. This model is a mixed-membership typed message layer; related primitives occur in [MMSB](https://www.jmlr.org/papers/v9/airoldi08a.html), [R-GCN](https://arxiv.org/abs/1703.06103), [HGT](https://arxiv.org/abs/2003.01332), and [NRI](https://proceedings.mlr.press/v80/kipf18a.html). It is not exact marginalization over hard-role graphs because nonlinear normalization does not commute with expectation.

A new capability vector can continuously change the role weights while reusing the same experts. The model does not need a new class-specific network for each new agent. However, a convex expert bank can be too restrictive; use the same inferred capability estimate in a CASH-style or ordinary conditional decoder to test that limitation.

### 6.5 Faithful class-wise alternative

For a closer HetGAT extension, define separately normalized soft sender-role groups:

\[
D_{i,rs}=\sum_j\widetilde A_{ij}p_{j,s}\exp e_{ij}^{rs},\quad
w_{ij}^{rs}=\frac{\widetilde A_{ij}p_{j,s}\exp e_{ij}^{rs}}{D_{i,rs}},
\quad S_{i,s}=\sum_j\widetilde A_{ij}p_{j,s}.
\tag{17}
\]

Set a group contribution to zero when its denominator is zero. A mass-aware continuous extension is

\[
m_i^r=\sum_s\min(1,S_{i,s})\sum_j w_{ij}^{rs}M_{rs}(h_j),\qquad
z_i=\sum_r p_{i,r}\,\sigma(W_rh_i+m_i^r).
\tag{18}
\]

At one-hot memberships, a nonempty class has integer mass at least one and the gate equals one; an absent class contributes zero. With matching underlying modules this recovers the hard typed operator. Keeping receiver mixing **outside** the nonlinear branch is part of that definition.

The mass gate is a proposed extension, not a component of published HetNet. Its purpose is to prevent a nearly absent role from contributing a full normalized message: without it, multiplying all \(p_{j,s}\) by an arbitrarily small positive number cancels from the numerator and denominator. Compare Equations (15)–(16) against (17)–(18); do not assume the faithful variant is better for size transfer. A fixed numerical denominator epsilon also changes exact reduction and must be documented.

### 6.6 Action experts and physical feasibility

For a discrete task, use an actual mixture distribution:

\[
\pi_i(a\mid\mathcal F_i)=\sum_rp_{i,r}\pi_r(a\mid z_i),\qquad
\log\pi_i(a)=\operatorname{logsumexp}_r(\log p_{i,r}+\log\pi_r(a\mid z_i)).
\tag{19}
\]

Each expert obeys the **same current physical action mask** of agent \(i\). A role cannot permit an action the body cannot execute. Apply feasibility to experts before mixing, with a legal fallback action. If capability is hidden, an environment-provided availability mask may reveal it; failed actuation with local feedback is a different, useful inference task.

The current continuous-action workflow needs an explicit choice. A mixture of bounded distributions is

\[
\pi_i(a)=\sum_rp_{i,r}\,\operatorname{TanhNormal}(a;\mu_{ir},\sigma_{ir}).
\tag{20}
\]

It requires matching sampling, Jacobians, log-sum-exp likelihoods and entropy estimation. Averaging means and using one Gaussian is a valid conditional policy but is **not** Equation (20). For the first information-role fixture, a declared discrete movement/service action set makes Equation (19) and entropy exact and simple. For current continuous PCP, initially share the action head and test role-conditioned communication alone, or implement a fully specified mixture distribution; do not paste the old categorical code into MAPPO.

### 6.7 Training and regularization

Deterministic soft gates do not require a sampled role-action likelihood. Recompute the full router/message/action graph during PPO updates, retaining gradient paths from a sender's gate to receivers' policies. Compare the recomputed and stored rollout likelihoods before the first update. A recurrent implementation requires ordered sequences, start states and resets, plus a declared burn-in/recomputation policy; flattening time as independent observations is insufficient.

Use a critic that pools privileged **entity** state and capabilities with active masks for mixed-size training, for example

\[
V_\psi(s_t)=f_\psi\!\left(x_t^{\rm global},\sum_i\ell_i\varphi_\psi(x_i,c_i),n_t,d_t\right).
\tag{21}
\]

Do not accidentally pool privileged features into the actor. A fixed-size training critic can still accompany a size-compatible deployable actor; Equation (21) primarily supports coherent training across team sizes.

Start with task return, ordinary exploration and a small prediction auxiliary only if it is part of the stated inference hypothesis. Initialize experts with small differences. Add strong role sharpness, uniform occupancy or switching penalties only after diagnosing a specific failure. A fixed balance prior can prohibit a correct response when useful capability disappears. A switching penalty can reward staying in an obsolete role.

If persistence is needed, apply it only across the same surviving agent lifetime and reduce it when available evidence indicates a change. This is a hypothesis to ablate, not a guarantee of timely recovery. Use \(K=2\) for scout/responder fixtures, then assess a small held-out sweep if relay behavior is necessary. Discovering the number of roles is a separate question.

## 7. Mathematical guarantees and their limits

The propositions in this section are **derived for the stated architecture**. They are mathematical characterizations, not claims that SoftRole has been trained successfully or that these elementary results are novel theorems. Longer proofs and primary theorem references appear in the mathematical audit.

### 7.1 Equivariance of the complete recurrent actor

Let \(Q\) permute active-agent rows, and transform node features, capabilities, memory and adjacency together:

\[
(H,C,B,A)\mapsto(QH,QC,QB,QAQ^\top).
\tag{22}
\]

**Proposition 1.** If local encoders, recurrent updates, routers and action experts share parameters across agents; entity encoders treat unordered entities as sets; relation functions do not depend on array index; and masks/tie handling respect relabeling, then

\[
P(QH,QAQ^\top)=QP(H,A),\qquad
\Pi(QH,QAQ^\top)=Q\Pi(H,A).
\tag{23}
\]

**Proof.** Row-wise shared functions commute with \(Q\). Each edge score and message maps to the corresponding relabeled edge. A bijective reindexing preserves each neighbor sum and softmax denominator. Class-wise sums and mass gates in Equation (18) obey the same rule. Shared fusion and action heads commute with the final row permutation. Induct over communication layers and timesteps, starting from consistently permuted memory. This proves the claim. With stochastic draws, equality holds in distribution; pathwise equality additionally requires permuting the random draws. \(\square\)

A scalar pooled critic is invariant; a vector of actions is equivariant. Symmetry of task return also requires the physical environment and its reward rules to respect the same relabeling. Hard nearest-neighbor ties broken by array index, persistent learned ID embeddings, slot-specific parameters, or ordered concatenated neighbor observations can violate the assumptions.

A physical heterogeneous agent can be relabeled only **together with** its features, capability and action semantics. Permuting capabilities while keeping bodies and observations fixed is an intervention, not a symmetry.

### 7.2 Identical agents cannot elect distinct deterministic roles

**Proposition 2.** If \(QH=H\) and \(QAQ^\top=A\), Equation (23) implies \(P=QP\). Agents in the same symmetry orbit receive identical deterministic role gates.

Thus a vertex-transitive anonymous graph and identical histories cannot by themselves select a unique scout. Regularity alone does not imply vertex transitivity; the proposition applies to agents in the same graph-and-feature symmetry orbit. Positions, different observations, outcome histories, or exchangeable random priorities can break the symmetry. Randomness preserves distributional equivariance, but independent role sampling does not guarantee a coordinated allocation. If every agent selects “scout” independently with probability \(p\),

\[
\Pr(\text{exactly one scout})=np(1-p)^{n-1},
\tag{24}
\]

maximized at \(p=1/n\), tending to \(e^{-1}\). This quantifies why independent stochastic gates are not automatically a reliable team partition. A coordinated protocol or a task with meaningful asymmetry is needed.

### 7.3 Hard-role recovery and gate stability

For \(T_{p,q}(x)=\sum_{r,s}p_rq_sM_{rs}(x)\), one-hot gates select the corresponding expert exactly:

\[
T_{e_a,e_b}(x)=M_{ab}(x).
\tag{25}
\]

If \(\|M_{rs}(x)\|\le B\), then

\[
\|T_{p,q}(x)-T_{p',q'}(x)\|
\le B\big(\|p-p'\|_1+\|q-q'\|_1\big).
\tag{26}
\]

**Proof.** Add and subtract \(T_{p',q}\), apply the triangle inequality, and use that each gate sums to one. If inputs also change and all experts are \(L_M\)-Lipschitz, add \(L_M\|x-x'\|\). \(\square\)

This is a local stability statement about a message operator. A whole-policy statement additionally needs bounds on role inference, attention, fusion and recurrence. Small role temperature can increase sensitivity. The analogous action-mixture bound is

\[
\operatorname{TV}\!\left(\sum_rp_r\pi_r,\sum_rp'_r\pi_r\right)
\le\tfrac12\|p-p'\|_1,
\tag{27}
\]

holding the expert distributions fixed. Actual role changes also affect their inputs, so those effects require additional terms.

Equation (18) has a stronger, precise hard-layer result: one-hot roles produce integer \(S_{i,s}\), the mass gate is one for nonempty classes and zero for empty classes, and \(w\) becomes the class-restricted softmax. This recovers Equation (8) with matched functions. It does not establish identity of the complete original algorithms or their learned parameters.

### 7.4 Equivariance does not prove larger-team generalization

Consider an equivariant family \(F_n\) fitted at \(n_0\). For any vector \(u\),

\[
\widetilde F_n(X)=F_n(X)+(n-n_0)\mathbf1_nu^\top
\tag{28}
\]

is also equivariant and identical on every training example of size \(n_0\), yet can differ arbitrarily at another size. A shared sum channel of constant ones supplies \(n\), so the counterexample does not require size-specific parameters. For several training sizes, replace \(n-n_0\) by a function vanishing on those sizes. Symmetry and perfect training fit still do not identify behavior off support.

There is also a converse architectural hazard: normalized attention over identical neighbors cannot distinguish one copy from many. Duplicating all neighbors leaves the result unchanged. If the correct policy needs total sensing capacity or a fixed number of responders, that missing multiplicity is consequential.

Finally, if one useful sender has score \(a\) and \(m\) distractors have score \(b\),

\[
\alpha_*(m)=\frac{e^a}{e^a+me^b}.
\tag{29}
\]

Keeping \(\alpha_*\ge1-\varepsilon\) requires
\(a-b\ge\log[m(1-\varepsilon)/\varepsilon]\). A score gap sufficient at training size can be insufficient after agents join. Per-role normalization can protect a rare role from other roles' distractors, but not from distractors inside the same role.

[Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) and [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) justify useful symmetry-preserving constructions under their stated assumptions. They do not prove this task-transfer claim. [Wagstaff et al.](https://proceedings.mlr.press/v97/wagstaff19a.html) show continuous set representations have latent-capacity requirements; [Yehudai et al.](https://proceedings.mlr.press/v139/yehudai21a.html) directly study failures from size-induced graph-distribution shifts. Graphon transfer results require compatible graph-generating models and regularity, rather than arbitrary changing-team dynamics; see the theorem-by-theorem table in the audit.

### 7.5 A positive conditional statement across different sizes

For fixed receiver context, represent a nonempty neighborhood by its empirical distribution \(\mu\) over feature/role tuples. One normalized attention aggregate is

\[
M(\mu)=\frac{\int e^{f(x)}v(x)\,d\mu(x)}{\int e^{f(x)}\,d\mu(x)}.
\tag{30}
\]

**Proposition 3.** Suppose \(|f|\le S\), \(\|v\|\le B\), and \(f,v\) are Lipschitz on a shared feature space with constants \(L_f,L_v\). For two possibly different-sized neighborhoods,

\[
\|M(\mu)-M(\nu)\|
\le e^{2S}(L_v+2BL_f)W_1(\mu,\nu).
\tag{31}
\]

**Proof.** The numerator integrand has Lipschitz constant at most \(e^S(L_v+BL_f)\), and the denominator integrand at most \(e^SL_f\). Couple the two measures and bound expectation differences by the respective constants times \(W_1\). Both denominators are at least \(e^{-S}\); both normalized outputs have norm at most \(B\). Subtract fractions and combine the bounds. \(\square\)

This captures the plausible benefit of shared set processing: larger teams can have similar local capability/role distributions, so the representation can remain close. It is a one-layer, bounded, Lipschitz statement. Changing receiver memory or team counts requires additional terms. Removing the only useful sensor can cause a substantial distribution change. A large change in physical task demand is not covered. Critically, closeness of representations does not establish task success unless the policy and controlled dynamics are also sufficiently stable.

For removal of neighbors whose old attention mass is \(\beta\), with surviving values/scores fixed, bounded values give

\[
\|m_{\rm old}-m_{\rm surviving}\|\le2B\beta.
\tag{32}
\]

This follows by expressing the old aggregate as a convex combination of removed and surviving aggregates. It explains stability to low-influence removals, not recovery from losing an essential information source. Joining agents analogously contribute their attention mass under the enlarged neighborhood.

### 7.6 Information limits on failure inference

Let \(P_0^k,P_1^k\) be the distributions of all execution-available information through \(k\) steps after two possible capability states, under a fixed protocol and equal priors. The optimal binary diagnosis error is

\[
\inf_{\widehat c}\Pr(\widehat c\ne c)
=\frac{1-\operatorname{TV}(P_0^k,P_1^k)}2.
\tag{33}
\]

**Proof.** If diagnosis chooses state 1 on event \(D\), error equals \(\tfrac12[1-P_1^k(D)+P_0^k(D)]\). Optimizing \(D\) gives total variation. \(\square\)

If failure creates no observable change, error cannot beat one half. No number of experts fixes that. “Nothing sensed” can mean no target, occlusion, radio loss, or a broken sensor; the task must provide distinguishing histories or an opportunity to probe. A validity flag announcing failure makes diagnosis easier but changes the claim.

A second lower limit is propagation delay: evidence \(d\) communication hops away cannot travel to an agent in fewer than \(d\) synchronous rounds when messages cross one hop per round and there is no other observation path. Continuous churn can outpace inference and propagation. That makes event dwell time relative to recovery time a central independent variable.

### 7.7 Conditional recovery is not an automatic GRU theorem

If a post-event belief error satisfies

\[
e_{k+1}\le\rho e_k+b,\qquad0\le\rho<1,
\]

then induction gives

\[
e_k\le\rho^ke_0+\frac{b(1-\rho^k)}{1-\rho}.
\tag{34}
\]

This describes a contraction plus a residual error floor. For \(b=0\) and \(0<\rho<1\), a sufficient number of updates is \(k\ge\max(0,\log(e_0/\varepsilon)/[-\log\rho])\). A generic trained GRU, role persistence loss, or PPO objective does not imply contraction. This is a conditional theorem target for a restricted estimator, and a diagnostic model for experiments, not a promised guarantee.

For rewards in \([0,R_{\max}]\), compare two policies in the **same post-change environment**, starting from the same distribution of environmental state and history, with the same transition, observation and reward kernels. Suppose their induced joint-action laws differ by at most \(\delta\) in total variation at every common complete environmental history, after marginalizing internal communication and policy randomness. If packets directly affect physical transitions or rewards, include them in the joint action instead. For \(0\le\gamma<1\), they satisfy the coupling bound

\[
|J(\pi)-J(\pi^*)|\le\frac{R_{\max}\delta}{(1-\gamma)^2}.
\tag{35}
\]

Couple action draws until the first disagreement, using common environment randomness while actions agree; probability of trajectory disagreement through step \(k\) is at most \((k+1)\delta\). Sum the discounted per-step reward bound. For a finite horizon, \(R_{\max}\delta H(H+1)/2\) is sufficient, capped by \(HR_{\max}\). Per-agent errors can accumulate: if these conditional joint-action laws factorize, \(\delta\le\min(1,\sum_i\delta_i)\). Marginalizing shared communication randomness can destroy that factorization, so bounds on individual action marginals alone are insufficient in general. Alternatively, include controlled stochastic packets in the coupled outputs and bound their joint law as well. Larger teams therefore do not automatically enjoy size-independent performance bounds even when the local architecture is shared.

### 7.8 Role semantics remain non-identifiable

If \(S\) permutes role columns, transform \(P\to PS\), \(B\to S^\top BS\), and reindex both axes of message experts and all action experts. Behavior is unchanged. Swapping only gate columns while keeping experts fixed is a different intervention. Identical experts introduce still more ambiguity.

Call the RL output a **role gate**, not a calibrated Bayesian posterior, unless a latent probabilistic model justifies the latter. A gate correlated with physical type is evidence of correlation; it is not proof of a true role or a causal mechanism. Even accurate capability estimation does not imply the computational roles should match capability labels one-to-one.

## 8. A research protocol that tests the intended claim

These are proposed experiments, not implementation tasks undertaken here. The mathematical model should be frozen before an empirical build.

### 8.1 Use a recovery task where organization can matter

Begin from Guide 3's two potential scouts and one responder. Both scouts can obtain task information; a failed scout retains motion/radio but loses sensing. Fresh service requests prevent stale information from solving the rest of the episode. A surviving-capability controller verifies that the task remains feasible.

For **behavioral role learning**, extend the fixture so at least some agents have overlapping capabilities. If every body can perform only one physical function, learning “roles” may reduce to classifying immutable types. An agent that can both scout and respond, with speed/reliability tradeoffs and changing neighbors, makes responsibility genuinely team-relative. A new highly reliable sensor can justify an existing flexible agent switching to service. This need not involve allocating new physical tools.

Use three separate event mechanisms:

| Event | Changes | What must remain distinguishable |
|---|---|---|
| Capability loss | A live agent's sensing reliability/range, speed or actuation effectiveness | Reduced physical feasibility versus failure to reassign surviving capacity |
| Membership change | Arrival, departure or replacement | Agent removal versus message dropout; new-agent initialization versus survivor memory |
| Information-role handover | Which capable agent obtains the useful cue | Pure relabeling versus changed information availability at fixed geometry |

For announced events, all policies receive the same descriptors. For hidden events, remove those descriptors and provide action-conditioned evidence consistently. Never judge a policy for failing to diagnose a statistically invisible event.

### 8.2 Separate the distribution shifts

An illustrative, adjustable design is:

| Axis | Training | Held-out evaluation |
|---|---|---|
| Agent numbering | Randomized consistently | Exact permutations as a numerical null |
| Team size | Active sizes 3–5 | Size 6 first; 8 as a harder stress test |
| Capabilities | Fixed schema and bounded ranges | New combinations, interpolation values, then mild extrapolation |
| Events | Isolated entry, exit and impairment with adequate dwell time | Different timing/severity, repeated changes, combined events |
| Communication | Declared candidate graph and packet timing | Controlled outages or different local degree as separate panels |
| Partners | Same frozen policy family | Independent partner policies only in a separately labeled ad hoc extension |

Also include **single-size training** if the desired headline is “trained on a subset, generalizes to a larger team.” Multi-size training is sensible, but cannot substitute silently for that stronger test. Training on only static rosters and evaluating dynamic events tests event generalization; training on events tests robustness to held-out event configurations. Report both if feasible.

Vary size independently from task demand. At fixed arena size, more agents increase density and collision risk; at fixed density, distances and communication diameter can increase. More agents can also make capture trivially easier. Report both a fixed-demand slice and a demand/density-controlled slice, with references using the same surviving resources.

Hold out **capability combinations**, not only agent IDs. An ID split can contain identical physical capabilities in both partitions. Report training support, test distance from it, and feasibility filters. A test outside the training range is extrapolation, however small the shift.

### 8.3 Baselines that distinguish the explanation

The primary comparison should be SoftRole against a recurrent shared graph policy with the same capability evidence, set encoder, communication rounds and training distribution. Then use a small staged baseline family:

| Condition | Question |
|---|---|
| Shared recurrent attention, capability estimate included | Are roles better than ordinary conditional computation? |
| Wider shared attention with matched compute/payload | Is the gain explained by added capacity? |
| Uniform gate over the same expert bank | Does adaptive routing matter? |
| Frozen pre-event gate | Does updating roles matter for recovery? |
| Inferred capability + CASH-style decoder | Is expert routing useful beyond adaptive parameters? |
| True-capability shared policy / typed reference | Is diagnosis the limiting factor? |
| ROMA-style dynamic role-conditioned actor with matched communication | Does directed role-pair communication add value? |
| Separate action and message gates | Does sharing one role variable help or overconstrain? |
| No communication / packet-dropout training | Is recovery dependent on useful information and distinct from message-loss robustness? |

“ROMA-style” or “CASH-style” controls adapted to the common task are not exact reproductions; label them accordingly. Reproducing the original algorithms on a different benchmark answers a separate question. Preserve the same failure cues and history length across conditions. A privileged reference may underperform due to optimization and is not a proven upper bound.

Use a matched curriculum comparison to separate architecture from training exposure: static role assignment, randomized episode-level assignment, and within-episode changes. A stronger curriculum may improve every model and eliminate the need for explicit roles. That is an informative result.

### 8.4 Outcomes and causal evidence

Choose one primary endpoint, such as the fraction of fresh post-event requests completed before deadline. Define a bounded recovery horizon \(H_r\). Report failure-aligned success curves, time to first correct response, and non-recovery probability. Treat unrecovered episodes as censored or at the declared horizon; do not average latency only over successful recoveries.

A paired recovery-loss contrast is

\[
\Delta_{\rm recovery}=
\big(Y_{\rm intact}-Y_{\rm event}\big)_{\rm baseline}
-\big(Y_{\rm intact}-Y_{\rm event}\big)_{\rm SoftRole}.
\tag{36}
\]

Positive values indicate a smaller event penalty, but still report all four raw outcomes: a low intact score can make a penalty misleadingly small. Compare each policy to a feasible post-change reference separately. Changing membership changes the task's attainable reward, so comparing only to the intact optimum conflates coordination failure and lost resources.

Log the evidence chain explicitly:

1. When did the changed capability become locally detectable?
2. When did belief/predicted outcome change?
3. When did role-conditioned behavior change?
4. When did a backup obtain fresh information and deliver a useful packet?
5. When did the receiver complete the correct task?

Use matched frozen-checkpoint interventions: keep pre-event roles fixed; replace gates by uniform; shuffle valid gate assignments among agents; remove one ordered message relation; suppress or shuffle useful content. Include negative-control directions and preserve packet opportunities where appropriate. These establish functional dependence under an intervention; they do not by themselves uniquely identify every internal explanation.

Useful diagnostics include role entropy, population entropy and

\[
I(I;R)=H\!\left(\frac1n\sum_ip_i\right)-\frac1n\sum_iH(p_i).
\tag{37}
\]

Here, at a fixed nonempty team snapshot, \(I\) is a uniformly sampled active-agent index and \(R\mid I=i\sim\operatorname{Categorical}(p_i)\). A uniformly soft population has high population entropy but zero differentiation under this diagnostic. None of these quantities substitutes for task recovery. Any semantic role alignment must be fitted on validation trajectories and held fixed for evaluation.

Use independently trained seeds as the inferential unit, with paired evaluation manifests and episodes nested within seed. Use development runs to estimate variability and choose confirmatory allocation; do not treat a convenient three-seed pilot as strong evidence. No compute-hour estimate is justified before measuring the new task/model, especially with recurrence and multiple communication rounds.

## 9. Feasibility in the current workspace

The read-only [repository audit](research/REPOSITORY_FEASIBILITY.md) found a material mismatch between the July notes and the current system. The active code is in `src/commstudy`, current PCP does not supply the assumed detector/capturer split, and its managed actions are continuous. Existing environment construction showed that increasing predators from three to five changes observation width from 17 to 21 and privileged state width from 22 to 30. Thus the present checkpoint interface cannot establish subset-to-larger-team transfer just by changing the communication module.

This does not invalidate the mathematical proposal. It means the future experimental system needs the set/entity representation, activity semantics and memory contract already specified above. No application or training implementation was changed in this review. The code audit is background evidence only; this deliverable focuses on the mathematical basis and study design.

A dependency order that follows the revised question is:

1. Fix the information model: capability schema, detectable outcomes, event process, active-set convention, graph protocol and exact generalization claim.
2. Select the soft operator: global-attention minimal model or faithful class-wise relaxation. State the hard limit and verify the proofs under the chosen interfaces.
3. Establish that announced handover is feasible and that a role-free shared recurrent policy is a strong control.
4. Test frozen capability/composition transfer and isolated within-episode changes.
5. Introduce hidden diagnosis, repeated churn and combined shifts only after isolating their information requirements.
6. Add constrained role allocation or hard topology learning only if a demonstrated failure requires them.

The major scientific risk is that ordinary capability-conditioned recurrence already recovers well. The major mathematical limitation is that partial observability and task-distribution shifts prevent unconditional recovery guarantees. The major architectural risk is that one low-dimensional gate cannot simultaneously express behavioral responsibility and information-routing needs. Each has a direct control above.

## 10. Recommended thesis framing

A suitable working title is **SoftRole-HetGAT: Capability-Informed Role and Communication Recovery in Changing Teams**.

The proposal should claim a testable mechanism:

> We study whether an execution-time role gate, updated from local capability evidence and communicated team context, improves recovery of directed information flow and cooperative behavior under changing heterogeneous team composition. We characterize its permutation symmetry and typed-message reduction, and evaluate frozen-policy transfer beyond capability-conditioned recurrent and learned-role baselines.

The mathematical deliverables are already well defined: full-actor equivariance, exact hard-layer recovery for a specified variant, local routing/attention stability, an observability lower bound, a conditional cross-size message bound, and explicit counterexamples to unconditional transfer. The empirical deliverables would determine whether the additional latent structure is useful.

Your connection to the third direction is therefore substantive: information-role recovery supplies a concrete mechanism and measurable failure sequence for SoftRole. The next intellectual step is to settle the observation and event model and the choice of attention normalization. There is no need to introduce physical tool allocation, Sinkhorn quotas, automatic role counts, or a new topology optimizer merely to motivate this question.

## 11. Reading order, provenance and remaining uncertainty

Start with published HetNet §4.3, ROMA §3 and its dynamic-role example, RODE's final transfer section, Howell et al.'s capability GCN, CASH §5.2/Appendix F.1, and GPL's open-team formulation. Then read the symmetry/size theory and the more recent role-communication/open-team papers. The [paper index](papers/README.md) distinguishes core, supporting and access-limited records; the machine-readable manifest and BibTeX keep them traceable.

The preserved July documents remain historical proposals. The source audits give exact page/section anchors and identify corrections rather than overwriting that history. Literature searches used primary proceedings, author copies, arXiv and publisher records through the review date. The collection is a broad, prioritized relevant-paper set; it is not a claim to have exhausted every adjacent paper or reproduced published results. In particular, inaccessible full texts and recent preprints leave the novelty boundary provisional.

All new deliverables and evidence copies are confined to `marl-comm/softrole/`. [agents.md](agents.md) records scope, progress and validation. No new training results, algorithm implementation, or empirical generalization claim is included.
