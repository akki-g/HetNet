# Mathematical audit: SoftRole-HetGAT under changing teams

Date: 2026-09-25. This is a research specification, not an implemented or experimentally validated method. Statements marked **derived here** are elementary propositions proved below, not results attributed to an external paper. Existing architecture and critical-review documents were preserved.

The mathematically defensible objective is to learn a **permutation-equivariant, capability-conditioned recurrent policy whose latent computational roles adapt after team changes with frozen parameters**. Equivariance removes dependence on arbitrary agent ordering. It does not establish successful transfer to more agents, unseen capabilities, or unobservable failures. Those require additional assumptions and experiments.

## 1. What the original HetNet actually computes

The supplied [HetNet.pdf](../papers/HetNet.pdf) is the published AAMAS 2022 paper, **Learning Efficient Diverse Communication for Cooperative Heterogeneous Teaming**, by Seraj, Wang, Paleja, Martin, Sklar, Patel, and Gombolay. Its title and author list differ from the early [2021 preprint](https://arxiv.org/abs/2108.09568). Use the [published proceedings version](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf) for the baseline.

Let \(i\) be the receiver, \(j\) the sender, \(c_i=r,c_j=s\), and \(\mathcal N_i^s=\{j:A_{ij}=1,c_j=s\}\). Below, \(r\leftarrow s\) always means sender \(s\) to receiver \(r\), avoiding the paper's confusing `i2l` indexing.

Writing its real-valued relation transform as \(V_{r\leftarrow s}\), the typed attention layer has the structural form

\[
e_{ij}^{rs}=\operatorname{LeakyReLU}
\left(a_{rs}^{\top}[U_rh_i\Vert V_{r\leftarrow s}h_j]\right),
\qquad
\alpha_{ij}^{rs}
=\frac{\exp(e_{ij}^{rs})}
{\sum_{k\in\mathcal N_i^s}\exp(e_{ik}^{rs})},
\tag{H1}
\]
\[
h_i'=\sigma\left(U_rh_i+
\sum_{s\in\mathcal C}\sum_{j\in\mathcal N_i^s}
\alpha_{ij}^{rs}m_{ij}^{rs}\right).
\tag{H2}
\]

An empty relation contributes zero. The critical details are **normalization within each edge type** and **one nonlinearity after adding the self term and all relation aggregates**. The published layer is not a single softmax across every neighbor. This is explicit in §4.3, p.1177, Eqs.(2)–(4). The real-valued baseline removes binarization and the encoder/decoder (§6.2, p.1178).

For binary communication, Eq.(3) instead uses

\[
m_{ij}^{rs}=D_r\big(\operatorname{GumbelSoftmax}(E_sh_j)\big).
\tag{H3}
\]

The architecture also includes class-specific input preprocessors and LSTMs (§4.2, pp.1175–1176), multi-head concatenation with final-head averaging, class-sized action outputs followed by softmax (§4.4, p.1177), and the training-only state embedding node / MAHAC critic (§§4.2,5). Replacing a class label by a simplex vector in one value transform does not recreate all these components.

The public implementation independently corroborates H1–H2: `fastreal.py` applies `edge_softmax` to each `p2p`, `p2a`, `a2p`, and `a2a` subgraph (e.g. lines 205–269), sums relation outputs with self features (lines 311–333), and applies ReLU afterward (lines 349–353). Reference copies and pinned commit provenance are in [source_code_audit/](source_code_audit/PROVENANCE.json), commit `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`. These files are evidence copies, not edits to the project's implementation.

**Why the previous one-hot claim is too strong.** Suppose a receiver has one sender in each of two roles, equal scores, self input zero, positive scalar messages 1 and 3, and ReLU. H2 gives \(1+3=4\), whereas a global softmax gives \((1+3)/2=2\). Both endpoint gates are perfectly one-hot. The previous architecture's §8 therefore does not recover original HetGAT merely by hardening the gates. Its later critical review correctly narrows the claim; the normalization difference is an additional concrete reason.

**Prior size evidence.** HetNet §6.3.4 and Fig.5c, pp.1179–1180, compare training convergence for PCP compositions (2P,1C), (3P,3C), (4P,6C). They support trainability at several compositions. They do not by themselves establish the requested experiment: train one checkpoint at smaller size, freeze it, then add/remove agents or impair capabilities during execution.

## 2. Formalize an open, heterogeneous team

At time \(t\), let \(V_t\) be the active agents, \(c_{t,i}\) physical capability, \(s_t\) the environment state including membership and capabilities, and \(A_t\) the allowed directed communication graph. A transition kernel can jointly evolve the world, membership, and capability:

\[
(s_{t+1},V_{t+1},c_{t+1})\sim
P(\cdot\mid s_t,V_t,c_t,\mathbf a_t).
\tag{P1}
\]

An agent's information is its observation/action/outcome history and messages actually received. Let \(\eta_{t,i}\) denote that history, including sensor-validity flags or action outcome signals only when the experiment makes them observable. A recurrent shared encoder maintains

\[
b_{t,i}=F_\theta(b_{t-1,i},o_{t,i},a_{t-1,i},y_{t-1,i},m_{t-1,i}),
\qquad \widehat c_{t,i}=C_\theta(b_{t,i}).
\tag{P2}
\]

Here \(y\) can contain locally observable success/failure evidence. It should not silently contain privileged true capabilities or a globally attributed performance score. A learned capability representation and a learned computational role are different objects: the same sensing-capable agent may scout or relay depending on its teammates and current demand.

Heterogeneous dimensions require a common interface: modality tokens/adapters and a declared action vocabulary with executable-action masks. A shared MLP cannot consume arbitrary unseen sensor layouts. Masks themselves reveal feasibility and can reveal capability, so separate *descriptor/mask-visible* and *capability-hidden* experiments.

For locally negotiated roles, one allowed role-agnostic communication round produces

\[
d_i=\sum_j A_{ij},\quad
u_i=\operatorname{AGG}\{\psi(b_j,\widehat c_j):A_{ij}=1\},
\]
\[
p_{t,i}=\operatorname{softmax}
\left(R_\phi(b_{t,i},\widehat c_{t,i},u_i,\log(1+d_i),p_{t-1,i})/\tau\right).
\tag{P3}
\]

`AGG` should expose both normalized composition and available mass/count where relevant. It may combine means, sums, and degree; global team counts are valid actor inputs only if communicated or explicitly public. One local round cannot reveal arbitrary remote membership changes. New agents initialize recurrent state with a shared rule; dead agents are removed/masked and their old state must not be assigned to replacements.

Call \(p_i\) a **role gate** unless there is a specified latent generative model and an inference objective that makes it a calibrated posterior. An RL-trained softmax is not automatically Bayesian inference. Frozen-policy adaptation means \(\theta,\phi\) stay fixed while \(b_i,p_i\) change.

## 3. Two coherent extensions, with different claims

### 3.1 Minimal global-attention extension

The existing proposal is a valid conditional-computation model:

\[
x_i=b_i+\sum_r p_{ir}E_r,\quad
v_{ij}=\sum_{r,s}p_{ir}p_{js}M_{r\leftarrow s}(x_j),
\]
\[
e_{ij}=\frac{(W_Qx_i)^\top W_Kx_j}{\sqrt d}+p_i^\top Gp_j,
\quad
\alpha_{ij}=\frac{A_{ij}\exp e_{ij}}{\sum_k A_{ik}\exp e_{ik}},
\quad
h_i'=F(x_i,\sum_j\alpha_{ij}v_{ij}).
\tag{S1}
\]

The endpoint weight \(p_{ir}p_{js}\) is a factorized routing choice; it is not an independently learned joint distribution over endpoint roles. Nonlinear attention means this network is a differentiable relaxation, not exact Bayesian averaging over hard role assignments.

Use zero communication for an empty neighborhood; do not evaluate an all-masked softmax. Attention masks must be applied before normalization. This model exactly recovers its **own hard typed global-attention architecture** at one-hot gates. It is inspired by HetGAT, not an exact extension of H1–H3. This is a reasonable minimal experiment if named honestly.

### 3.2 Relation-normalized extension that recovers the real HetGAT layer

If faithful inheritance is central to the thesis, preserve source-role normalization. For each possible receiver role \(r\) and sender role \(s\), define

\[
c_i^s=\sum_j A_{ij}p_{js},\quad
Z_i^{rs}=\sum_j A_{ij}p_{js}\exp(e_{ij}^{rs}),
\]
\[
\mu_i^{rs}=
\begin{cases}
\dfrac{\sum_j A_{ij}p_{js}\exp(e_{ij}^{rs})M_{r\leftarrow s}(h_j)}{Z_i^{rs}},&Z_i^{rs}>0,\\
0,&Z_i^{rs}=0.
\end{cases}
\tag{S2}
\]

Simply using \(\mu_i^{rs}\) is problematic: if all neighbors have \(p_{js}=\varepsilon\), the small role mass cancels out of numerator and denominator. The role contributes an order-one message even as its mass approaches zero. This is especially undesirable when the last member of a role fails or departs.

One simple proposed remedy (**derived here; not a HetNet equation**) is

\[
g(c)=\min(1,c),\qquad
h_i'=
\sum_r p_{ir}\sigma\left(U_rh_i+
\sum_s g(c_i^s)\mu_i^{rs}\right).
\tag{S3}
\]

This gate is continuous and piecewise differentiable. If message norms are bounded by \(B\), \(\|g(c)\mu\|\le B\min(1,c)\), so an absent relation vanishes continuously. It is one design choice to ablate, not a principled optimum or a claim of new allocation theory.

**Proposition S: hard layer recovery, derived here.** If \(p_i=e_{c_i}\) for every active agent, each \(c_i^s\) is a nonnegative integer. Nonempty relations have \(g(c_i^s)=1\); empty ones give zero. S2 normalizes exactly over \(\mathcal N_i^s\), and the outer mixture selects \(r=c_i\). Thus S3 equals H2 when score functions, relation operators, self transforms, and nonlinearities match. The statement concerns this real-valued layer with common hidden interfaces; full original HetNet additionally requires matching adapters, heads, action outputs, and training components. A binary variant would require explicitly preserving H3.

The location of the receiver mixture matters:

\[
\sum_rp_{ir}\sigma(z_{ir})\ne
\sigma\left(\sum_rp_{ir}z_{ir}\right)
\]

in general. Either is a possible relaxation and both select the hard branch at a vertex, but they express different soft models. State which is implemented.

For action selection, use a probability-space mixture with each expert normalized over executable actions:

\[
\pi_i(a\mid\eta_i)=\sum_r p_{ir}
\frac{\mathbf1[a\in\mathcal A_i^{\rm valid}]\exp\ell_{ir}(a)}
{\sum_{a'\in\mathcal A_i^{\rm valid}}\exp\ell_{ir}(a')}.
\tag{S4}
\]

A safe valid action must exist. Physical feasibility is owned by the environment, not inferred role membership. Log-probabilities for policy optimization must be those of the actual mixture; sampling a role creates a different stochastic policy whose probability must also be accounted for.

In both designs, sender \(j\) must transmit its role token or a sufficient representation. If receiver-conditioned messages are constructed at the sender, the sender also needs the receiver role, requiring metadata exchange or cached roles. Receiver-side decoding of a common sender payload avoids that hidden reverse dependency. Count the negotiation round and role metadata in communication comparisons.

## 4. Exact symmetry statements

### 4.1 Agent permutation equivariance

Let \(Q\) permute active-agent rows. Features, recurrent states, capability descriptions, masks, and edge attributes must all be relabeled consistently:

\[
(H,B,C,A)\mapsto(QH,QB,QC,QAQ^\top).
\]

**Proposition E, derived here.** Shared row-wise encoders and routers, relation transforms independent of agent index, masked neighbor sums/softmax, and shared actors imply

\[
P(QH,QAQ^\top)=QP(H,A),
\qquad
\Pi(QH,QAQ^\top)=Q\Pi(H,A).
\tag{E1}
\]

**Proof.** Shared local maps commute with row permutation. Under \(i\mapsto q(i),j\mapsto q(j)\), every edge score/value is the corresponding relabeled score/value. A bijective reindexing leaves each sum and denominator unchanged; edge tensors transform by \(Q(\cdot)Q^\top\). For S3, \(c_i^s,Z_i^{rs},\mu_i^{rs}\) likewise follow the receiver permutation. Shared fusion and action maps finish the argument. Induction extends it across layers and time if the initial recurrent states are permuted consistently. Independent stochastic action/role draws preserve equivariance **in distribution**; pathwise equality requires correspondingly permuting random draws. □

This requires more than weight sharing: flattened fixed-order global features, slot-specific biases, unpermuted capability masks, hidden ID-based access, or tie breaking by row index can break it. For heterogeneous action spaces, compare actions through their declared semantic adapters.

The decentralized actor is **equivariant**; a scalar team critic or team return is **invariant** under a consistent relabeling of the entire environment. Symmetry of the network alone does not make the environment reward permutation invariant.

### 4.2 Role-label reparameterization is a separate symmetry

If \(S\) permutes role columns, transform \(P'=PS\), \(E'=S^\top E\), \(G'=S^\top GS\), and reindex both axes of every relation expert and all action experts. The policy is unchanged. This is a **parameter/gate reparameterization**, not invariance to permuting gate columns while holding the learned expert parameters fixed. Latent role labels are not uniquely semantic; identical or redundant experts permit even broader nonidentifiability. Role plots alone do not establish useful roles.

### 4.3 Deterministic symmetry barrier

If \(QH=H\) and \(QAQ^\top=A\), E1 gives \(P=QP\). Agents in the same automorphism orbit have identical deterministic role gates. Exactly symmetric anonymous agents cannot deterministically elect a unique scout using these inputs alone. Distinct observations/history or randomness can break symmetry. Independent role samples do not guarantee coordinated counts: if \(N\) identical agents independently choose the scarce role with probability \(p\), the chance of exactly one choosing it is \(Np(1-p)^{N-1}\), maximized at \(p=1/N\), tending to \(e^{-1}\). Reliable allocation requires a coordination mechanism or informative asymmetry.

## 5. Why size-generalization needs its own evidence

**Four different claims:** (i) insensitive to ordering; (ii) callable at different \(N\); (iii) accurate on a new distribution at a different \(N\); (iv) maintains/rebuilds coordination after membership changes within an episode. E1 establishes only (i). N-independent parameter shapes plus valid masks enable (ii). The user's desired result is (iii) and (iv).

**Counterexample 1, derived here: perfect fixed-size fit, arbitrary wrong size behavior.** For any equivariant family \(F_N\), fixed training size \(N_0\), and shared vector \(u\),

\[
\widetilde F_N(X)=F_N(X)+(N-N_0)\mathbf1_Nu^\top
\tag{C1}
\]

is also equivariant, identical on all training inputs of size \(N_0\), and arbitrarily different at another size. A constant-one sum channel represents \(N\) with N-independent parameters. Parameter sharing and zero training error cannot determine behavior away from the training sizes.

**Counterexample 2, derived here: count blindness.** With identical neighbor inputs \(x\), softmax scores and values are identical; the aggregate is \(v(x)\) for one neighbor, ten neighbors, or a hundred. More generally, duplicating every neighbor \(k\) times leaves a normalized attention aggregate unchanged. A policy whose only team input is such aggregates cannot implement tasks whose correct role occupancy depends on absolute available capacity. A task requiring one scout has a different staffing problem at \(N=2\) and \(N=20\), even if capability proportions match. Include counts or sums where task demand requires them; this supplies information, not a generalization guarantee.

**Counterexample 3, derived here: attention dilution.** If one important sender has score \(a\), and \(m\) distractors each have score \(b\), its weight is

\[
\alpha_*(m)=\frac{e^a}{e^a+me^b}.
\tag{C2}
\]

Keeping \(\alpha_*\ge1-\varepsilon\) requires \(a-b\ge\log(m(1-\varepsilon)/\varepsilon)\). A fixed score margin learned in small teams may be inadequate in large ones. Relation-wise normalization protects rare classes from competition across other classes but not dilution within a class. Top-k/radius restrictions, count features, and varied distractor counts are possible mitigations to test, with changed graph reachability and bandwidth consequences.

### Evidence from established mathematics

| Primary source and exact location | What it supports; what it does not |
|---|---|
| [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), §2.2, PDF p.2, Theorem 2 and Lemma 3 | Sum-decomposable invariant functions under stated domain assumptions; tied equivariant layers. The paper explicitly distinguishes countable-universe characterization from its fixed-cardinality uncountable proof. It supplies architecture structure, not learned cross-size policy performance. |
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), PDF p.4, Property 1 / Propositions 1–2; supplement pp.1–2, Lemma 3 | Attention encoder equivariance and invariant pooling. Its universality construction permits an attention activation implementing **sum pooling**. It does not justify universal variable-cardinality counting for an ordinary softmax-only implementation. |
| [Wagstaff et al.](https://proceedings.mlr.press/v97/wagstaff19a.html), PDF p.4, Theorem 4.1; p.5, Theorems 4.3–4.4 | Some continuous invariant functions on M scalar inputs cannot be continuously sum-decomposed through fewer than M latent dimensions. Fixed/bounded size universality has capacity conditions. This does not say every useful task needs M latent dimensions. |
| [Xu et al.](https://arxiv.org/abs/1810.00826), PDF p.4, Lemma 2/Theorem 3; p.5, Lemma 5; §5 | Message-passing expressivity is limited by multiset discrimination; injective aggregation yields WL-level discrimination under assumptions. Normalized means can lose multiplicity. This is a representation result, not a MARL convergence theorem. |
| [Yehudai et al.](https://proceedings.mlr.press/v139/yehudai21a.html), PDF p.5, Theorems 5.1–5.2 | Under finite-support/local-pattern assumptions, there exist solutions fitting small-graph tasks while making large errors on shifted graph distributions. Relevant to size-induced neighborhood shifts; it does not prove all GNNs fail. |
| [Ruiz et al.](https://papers.nips.cc/paper/2020/hash/12bcd658ef0a540cabc36cdf2b1046fd-Abstract.html), §4, Theorems 1–2 | Positive transferability bounds for graph convolutional models built from a shared graphon, with regularity, filter, spectral, and activation conditions. Arbitrary learned attention and arbitrary arrivals/departures are outside this theorem's automatic scope. |
| [Levin et al.](https://proceedings.neurips.cc/paper_files/paper/2025/hash/60b393761eea35bf491687c46e4f1a5a-Abstract-Conference.html), §§2–4, Definition 2.4 / Definition 3.1 / Proposition 3.2 / Proposition 4.2, PDF pp.3–5 | Transferability uses compatible maps between sizes plus continuity in an appropriate common space. Their generalization statement assumes a common sampling model and regularity. It is supervised-learning theory, not a theorem that arbitrary changing-team MARL succeeds. |

## 6. Useful guarantees that can actually be proved

All results in this section are **derived here**, conditional, and local to the stated mathematical model. They are not evidence that training finds the desired representation.

### 6.1 Sensitivity of endpoint-role mixture

For \(T_{p,q}(x)=\sum_{r,s}p_rq_sM_{rs}(x)\) with \(\max_{r,s}\|M_{rs}(x)\|\le B\),

\[
\|T_{p,q}(x)-T_{p',q'}(x)\|
\le B(\|p-p'\|_1+\|q-q'\|_1).
\tag{B1}
\]

Proof: add/subtract \(T_{p',q}\), use the triangle inequality and the simplex sums. If the experts are uniformly \(L_M\)-Lipschitz in \(x\), add \(L_M\|x-x'\|\) when the input changes. The bilinear bias satisfies the analogous bound with \(B=\max_{rs}|G_{rs}|\).

For a fixed set of senders, \(\|\operatorname{softmax}(z)-\operatorname{softmax}(z')\|_1\le2\|z-z'\|_\infty\), since \(J(z)u=p\odot(u-\mathbb E_pu)\). Thus a normalized aggregate with values bounded by \(B\) obeys

\[
\|m-m'\|\le\max_j\|v_j-v_j'\|+2B\|e-e'\|_\infty.
\tag{B2}
\]

A score-temperature \(T\) replaces 2 by \(2/T\). These bounds are not whole-network constants unless fusion, recurrent updates, role inference, and action heads are also controlled. Small temperature can amplify perturbations.

### 6.2 Immediate effect of adding or deleting agents

For one normalized aggregate, hold surviving scores and values fixed. If deleted neighbors had total old attention mass \(\beta\) and at least one sender survives, write

\[
m=(1-\beta)m_{\rm surviving}+\beta m_{\rm deleted}.
\]

Renormalization yields \(m'=m_{\rm surviving}\), so

\[
\|m-m'\|\le2B\beta.
\tag{B3}
\]

For additions use their total attention mass in the enlarged neighborhood. This states that a low-attention change has small **immediate representation** impact. It does not establish graceful recovery after losing the only sensor or actuator. Router recomputation and changed observations need additional terms. For S3 apply the argument within each nonempty relation, and separately account for occupancy-gate changes.

### 6.3 A conditional bridge from different-sized neighborhoods

For S1 with a **fixed receiver context**, let \(\mu\) and \(\nu\) be empirical probability measures of its neighbor feature/role tuples. Define

\[
M(\mu)=\frac{\int e^{f(x)}v(x)\,d\mu(x)}{\int e^{f(x)}\,d\mu(x)}.
\]

Assume \(|f(x)|\le S\), \(\|v(x)\|\le B\), and \(f,v\) are Lipschitz with constants \(L_f,L_v\) on a common metric feature space. Then

\[
\|M(\mu)-M(\nu)\|
\le e^{2S}(L_v+2BL_f)W_1(\mu,\nu).
\tag{B4}
\]

**Proof.** \(e^f\) is \(e^SL_f\)-Lipschitz; \(e^fv\) is \(e^S(L_v+BL_f)\)-Lipschitz. Coupling \(\mu,\nu\) bounds the numerator and denominator differences by those constants times \(W_1\). Both denominators are at least \(e^{-S}\), and each normalized output has norm at most \(B\). Subtract the two fractions and combine. □

This is a precise restricted positive result: size can change while a normalized message stays similar if the neighbor distributions remain similar. It does not require equal sample counts. Conversely, removing the only capable scout, changing connectivity, or making receiver beliefs change can produce a large distribution/context change. If role allocation depends on count, explicitly add a Lipschitz count/context-difference term; empirical probability measures alone intentionally forget multiplicity. B4 bounds one layer, not task return or arbitrary S3 role-normalized networks.

### 6.4 Observability lower bound

Let two capability states induce distributions \(P_0^k,P_1^k\) over all execution-available information through \(k\) steps after a failure, under a specified sensing/action/communication protocol. With equal priors, any capability classifier has error at least

\[
\inf_{\widehat c}\Pr(\widehat c\ne c)
=\frac{1-\operatorname{TV}(P_0^k,P_1^k)}{2}.
\tag{B5}
\]

Proof: if a classifier selects state 1 on event \(D\), its error is \(\frac12(1-P_1^k(D)+P_0^k(D))\); optimize over \(D\), using the definition of total variation. Thus indistinguishable capability states force error \(1/2\). An unannounced broken sensor with no distinguishable observation or action consequences cannot be diagnosed merely by a more expressive role network. Active probing can change these distributions and should be a separate mechanism/condition.

Similarly, when the only evidence of a teammate change is \(d\) graph hops away and each synchronous communication round crosses at most one hop, fewer than \(d\) rounds cannot transmit that evidence. World observations might provide another path; state that possibility when measuring recovery latency.

### 6.5 Conditional recovery and return bounds

Suppose an appropriate belief/representation error to a desired post-change reference satisfies

\[
e_{k+1}\le\rho e_k+b,\quad0\le\rho<1.
\]

Then iterating gives

\[
e_k\le\rho^ke_0+\frac{b(1-\rho^k)}{1-\rho}.
\tag{B6}
\]

With \(b=0\) and \(0<\rho<1\), \(k\ge\max(0,\log(e_0/\varepsilon)/(-\log\rho))\) suffices for error below \(\varepsilon\). This is useful only after establishing observability, reachability, a suitable reference, and the contraction inequality. If \(\rho=0\), the error reaches the residual floor in one update. An ordinary trained GRU, PPO loss, or temporal KL penalty does **not** imply these assumptions. Temporal smoothing may slow precisely the adaptation under study.

For interpretation, suppose two complete post-change joint policies start from the same distribution of environmental state/history, use the same transition, observation and reward kernels, and have rewards in \([0,R_{\max}]\). Assume their induced joint-action laws, after marginalizing internal communication and policy randomness, differ by at most \(\delta\) at every common complete environmental history. Include packets in the controlled action if they directly affect physical transitions or rewards. A sequential coupling gives trajectory divergence probability at most \((k+1)\delta\) through decision \(k\), so for \(0\le\gamma<1\),

\[
|J(\pi)-J(\pi^*)|
\le\frac{R_{\max}\delta}{(1-\gamma)^2}.
\tag{B7}
\]

The trivial cap \(R_{\max}/(1-\gamma)\) also applies. If these conditional joint-action distributions factorize, \(\delta\le\min(1,\sum_i\delta_i)\). Marginalizing shared communication randomness may destroy that factorization; individual action-marginal errors alone cannot bound the discrepancy between arbitrarily correlated joint policies. Therefore a constant per-agent policy error can lead to a weaker bound as teams grow. B7 compares policies in the **same post-change task**; it cannot erase physical performance losses from removal of essential capabilities. A post-change retrained/oracle-capability benchmark provides an appropriate empirical reference, with the usual caveat that an imperfectly trained oracle is not a formal optimum.

## 7. Implications for a feasible study

1. Preserve a clear causal chain: capability evidence → recurrent belief → team-context role gate → directed messages and action mixture → post-change team performance. Instrument each stage.
2. Choose S1 for the simplest conditional-computation study or S2–S3 for a paper-faithful HetGAT layer extension. Do not change the normalization rule silently across baselines.
3. Use a shared recurrent capability-aware policy without roles, a parameter/round-matched shared GAT, fixed/oracle typed HetGAT, and learned roles as separate controls. Otherwise a benefit of memory or extra capacity can masquerade as role learning.
4. Evaluate order permutation, held-out team sizes, held-out capability combinations, and within-episode changes separately. Freeze the checkpoint, report uncertainty across independently trained seeds, and stratify by failure visibility and whether a feasible post-change solution still exists.
5. Test count blindness, rare-capability dropout, attention dilution, identical-agent symmetry, isolated arrivals, and delayed failure evidence as deliberate stress cases.
6. Measure time to recover task reward/success and message usefulness, not just time for role labels to change. Include frozen-gate and matched role/message/expert interventions to show that the adaptive gate is causally useful.
7. Claim mathematical guarantees only for equivariance, exact hard-layer reduction under matched components, and the conditional bounds above. Claim cross-size success and role recovery only after the corresponding experiments.

This framing supports a feasible project: **evidence-conditioned role and communication recovery in open heterogeneous teams**. New physically incompatible agents, unseen sensor/action semantics, arbitrary graph shifts, and unobservable capability loss remain outside an unconditional generalization claim.

## Local primary-source archive

The supplied HetNet PDF is preserved. Downloaded primary-source PDFs, extracted text, source URLs, byte counts, retrieval dates, and SHA256 hashes are tracked in [manifest_math.json](../papers/manifest_math.json). This audit used the full papers for the specific equations/theorems identified above; inclusion in the archive is not a claim that every appendix of every paper was exhaustively assessed.
