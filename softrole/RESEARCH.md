# Capability-conditioned HetNet: implemented research plan

29 September 2026 · branch `softrole` · environment `corrected-observation-v1`

## 1. Research question and scope

Can one shared, capability-conditioned actor retain useful HetNet-style binary
communication when team composition changes or one agent loses its sensor,
without a supplied class label or class-specific actor/communication parameters?

This is a controlled reformulation and empirical question. It does not require
claiming a new theory of roles. A careful negative result against a strong shared
baseline is also informative. The code supports the released PP, PCP and FC
domains; the primary generalization/failure study is PCP. Engineering tests
establish mathematical identities, implementation behavior and reproducibility;
they do not establish learned robustness or reproduce the paper's scores.

The implementation follows the revised deterministic proposal. The supplied
`softrole.patch`, the PDFs and the older local `PLAN.md` are reference inputs,
not code to apply verbatim. Gaussian roles, KL/persistence penalties, diversity
rewards, and within-episode agent arrivals/departures are deferred. Team size can
change between episodes without changing the network. During an episode, sensing
capability can fail while the agent remains present.

## 2. What the critique changes

| Earlier idea or problem | Decision and reason |
|---|---|
| Remove a class embedding but retain P/A occupancy channels | Merge the occupancy **counts**. Otherwise the observation still directly reveals neighbors' physical class. |
| Treat label independence as having no knowledge of heterogeneity | Supply only own current sensing/actuation capability. This assumes local capability diagnostics; it does not solve unknown fault detection. Initially capabilities correlate with classes, so the honest claim is removal of explicit class labels and class-indexed weights, not statistical independence from class. |
| A stochastic latent role is necessary | Start with a deterministic soft gate. This isolates whether conditional parameter sharing helps, without adding a second stochastic control problem. |
| Latent expert indices have semantic roles | No such guarantee. Expert permutations yield the same function; entropy and load are diagnostics, not proof of specialization. |
| Each actor can use only its own return | Use the summed team reward for every action score. An agent can create reward exclusively for a teammate. |
| Binary straight-through training is an exact policy gradient | State its bias and test it against an enumerated score-function reference. Do not turn a surrogate derivative into a theorem. |
| More pooled agents proves transfer | Equivariance and size-independent tensor dimensions permit transfer; neither guarantees performance on a new team. Freeze actual weights and measure it. |
| HetNet necessarily cannot transfer to a new count | Its class-shared weights can transfer once count-dependent graph/bookkeeping and singleton shapes are handled. Include a frozen contextual reference. |
| Masking FC observations simulates sensor failure | FC sensing also updates discovery state. Reject FC failure evaluation until those physical semantics are explicitly changed and tested. |
| Average worker gradients, clip locally, then divide by steps | Sum episode gradients, divide once by global episode count, clip once, update once. Otherwise worker episode counts and episode lengths change the intended weighting. |
| A final task success is automatically post-failure recovery | Record whether the scheduled event was reached. Separate pre-event completions and censored failures. |

The original environment appends NumPy views of a shared grid and then blinds
actuation-agent windows in place. Overlapping sensing-agent windows can therefore
be changed inadvertently. The new adapter opts into independent window copies in
PCP and FC. Original entrypoints retain their old default. Corrected and original
observation results must not be silently pooled.

Capability-aware shared policies and adaptive teaming already have direct
precedent in [Howell et al.](https://proceedings.mlr.press/v229/howell23a.html).
[CASH](https://arxiv.org/html/2501.06058v4) already studies capability-conditioned
parameter generation, changing teams and online capability changes. Thus “first
class-free heterogeneous policy” or “first sensor-failure adaptation” would be
unsupported novelty claims. The useful contribution here is a simpler bounded
mixture in the HetNet communication setting, a clear information contract, and a
reproducible comparison. [ROMA](https://proceedings.mlr.press/v119/wang20f.html)
is relevant role-learning context, not evidence that this deterministic gate
discovers interpretable roles.

## 3. Exact information and physical contract

For agent j, let kappa_j=(s_j,a_j) in {0,1}^2 denote a working environmental
sensor and available capture/extinguish action. The original simulator still
uses P/A names for physical bookkeeping, rewards and evaluation composition.
Those names are never embedded or used to select parameters in the new actor.

Each native local cell has `[position one-hot, A count, outside, target, P count]`.
For B=map_width² and S=(2*vision+1)², the adapter constructs

```
x_j = concat(flatten(position),
             flatten(s_j * [outside, target, P_count + A_count]),
             kappa_j)                         dimension S*(B+3)+2.
```

For the 5×5 PCP map and vision 2 this is 702 entries; FC vision 1 gives 254.
Counts preserve co-location information. The position encoding and map geometry
are retained as the existing proprioceptive/static coordinate information.
Sensor loss therefore means loss of environmental sensory channels, not loss of
localization. The failed agent keeps its recurrent memory and communication;
past observations are not erased. Its own s_j becomes zero before the scheduled
action, and it may communicate this change through its learned payload.

There are six action logits. Action 5 is masked to minus infinity when a_j=0;
the remaining five actions are always available. The environment independently
checks this constraint. Physical actuation failure or adding a sensor to a blind
agent is outside this implementation.

Graph connectivity is determined by actual positions and a common Euclidean
communication range, independent of class. Default range -1 permits every
other agent. Range 0 can still connect co-located agents. No self-edge is used;
self-processing is explicit. Adjacency is indexed `[receiver,sender]`.

## 4. Architecture and claims that can actually be proved

### 4.1 Local recurrence and deterministic conditioning

```
e_jt = ReLU(Linear_702→128(x_jt))
(H_jt,C_jt) = LSTMCell([e_jt, ubar_j,t-1, onehot(action_j,t-1)],
                       [H_j,t-1,C_j,t-1])
g_jt = softmax(Linear_32→4(ReLU(Linear_66→32([H_jt,kappa_jt])))).
```

H and C have width 64; previous communicated features ubar have width 16. Initial
states are zero and the initial previous action is stay (index 4). Use the same
gate in both communication rounds. A gate's inputs include previous received
information through recurrence, but never current global critic features.

Every affine bank uses

\[
 B(x,g)=\sum_{c=1}^{K}g_c(W_cx+b_c),\quad
 g_c\geq0,\quad\sum_cg_c=1. \tag{1}
\]

This equals `(sum_c g_c W_c)x + sum_c g_c b_c` exactly by distributivity.
For any matrix norm, `||sum_c g_c W_c|| <= max_c ||W_c||` by the triangle
inequality. Thus a **fixed-gate** linear map is in the experts' convex hull.
This is not a bound on the full input-dependent Jacobian: differentiating g(x)
adds gate-derivative terms. It is also not a universal approximation or training
stability guarantee. Equation (1) is an architectural restriction chosen to
keep conditional parameter sharing small and auditable.

### 4.2 Two rounds of binary communication

In each round, with input h_j and one gate g_j:

\[
 z_j^q=B_{self}^q(h_j,g_j),\quad
 u_j=B_{enc}(h_j,g_j),\quad
 b_j=1[u_j+\ell_j>0],\qquad \ell_j\sim Logistic(0,1). \tag{2}
\]

There are four heads q, each width 16. The encoder emits **one 16-bit payload**
per agent per round, reused across every receiver and head. Since the logistic
CDF is sigmoid, `Pr(b=1)=sigmoid(u)` coordinatewise. Fresh noise draws are
independent across bits, agents, steps and rounds, and each draw is independent
of its own preceding history and pre-sampling logit. Later logits may depend on
earlier sampled messages.

The implementation's forward value is the hard bit; its backward derivative is
the derivative of `sigmoid(u+ell)` at temperature 1:

```
b_ST = hard_bit + relaxed_bit - stop_gradient(relaxed_bit).
```

This is a biased surrogate. An exact discrete gradient would include the binary
sampling score times downstream return, as formalized by
[stochastic computation graphs](https://proceedings.neurips.cc/paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html).
For a scalar Bernoulli bit with reward b, the exact derivative is p(1-p);
averaging the surrogate derivative need not equal this value. Tests enumerate
the exact reference and explicitly demonstrate the bias.

Receiver j decodes sender k using **j's gate**:

\[
 m_{jk}^q=B_{dec}^q(b_k,g_j),\quad
 e_{jk}^q=a^q(g_j)^T\operatorname{LeakyReLU}
       (W^q(g_j)[z_j^q\Vert m_{jk}^q]+d^q(g_j)). \tag{3}
\]

The score vector has no extra scalar affine bias. GATv2-style attention is
motivated by its ability to avoid the query-independent ranking restriction of
standard GAT; that is an expressivity motivation, not a prediction of improved
MARL performance. See [Brody et al.](https://arxiv.org/abs/2105.14491).

Add a null candidate with score `nu_jq=sum_c g_jc nu_cq` and message exactly zero:

\[
 \alpha_{jk}^q=\frac{\exp e_{jk}^q}
 {\exp\nu_j^q+\sum_{v\in\mathcal N(j)}\exp e_{jv}^q},\quad
 y_j^q=z_j^q+\sum_{k\in\mathcal N(j)}\alpha_{jk}^q m_{jk}^q. \tag{4}
\]

Missing edges are masked before softmax. When the neighborhood is empty,
alpha_null=1 and the received aggregate is exactly zero, even if decoders have
biases. The first round concatenates `ReLU(y_jq)` across heads, producing width64.
The second averages heads without a final ReLU, producing ubar of width16.
A shared 16→6 affine map and the physical action mask produce the action policy.

Dense masked PyTorch operations implement this single-relation graph directly.
They are O(N²), appropriate for these small teams; this is not a claim of sparse
large-team scalability. Generated payload is 32 bits/agent/step. This counts two
logical broadcasts, not radio headers, receiver-side deliveries, graph discovery
or simulator traffic. `comm_off` zeros received aggregates; diagnostic sampled
payload bits can still be generated, so its metric is not measured airtime.

### 4.3 Symmetry and centralized value

Permute agents, local states, capabilities, adjacency rows/columns and assigned
noise consistently. Shared local maps permute their outputs; neighborhood sums
and softmax denominators are unchanged up to index permutation. Induction across
both rounds and recurrent steps proves actor permutation equivariance. Under
independent identically distributed noise, the uncoupled stochastic policy is
equivariant **in distribution**. Tests use coupled noise for exact comparisons.

The critic is

\[
 V=\rho\left[\frac1N\sum_j\phi([H_j,\kappa_j]),
       N,\sum_js_j,\sum_ja_j,\frac{T-t}{T}\right], \tag{5}
\]

where phi is 66→64 with ReLU and rho is 68→64→1 with one hidden ReLU.
The mean is invariant to agent permutation. Counts retain team size/capacity
information that a mean alone can lose. Critic inputs and outputs never enter
the actor forward pass. Value loss does train the shared recurrent features;
this is centralized training with shared representation, not disjoint actor and
critic parameters. A centralized baseline is valid for action-score subtraction
when it is independent of the current sampled action conditional on history.

No weight shape depends on N. This proves execution at other roster sizes, not
successful coordination there. It also does not prove robustness to a new
capability combination, sensor loss, or a changed communication graph.

## 5. Team objective, optimization and approximations

Define the finite-horizon objective

\[
 J(\theta)=\mathbb E\left[\sum_{t=0}^{\tau-1}r_t^{team}\right],
 \qquad r_t^{team}=\sum_{j=1}^N r_{jt},\quad\tau\leq T. \tag{6}
\]

Summing physical rewards intentionally values team reward, not mean-agent reward.
For a deterministic communication reference, the joint action log probability
is the sum of agent log probabilities. Its score multiplies a **common team**
return/advantage. Counterexample to own-agent credit: let agent1 choose
`a~Bernoulli(sigmoid(theta))`, with r1=0 and r2=a. The team gradient is p(1-p)>0,
but weighting agent1's score by its own reward gives zero. This is checked by
enumeration. In the implemented stochastic channel, the same action-score term
is retained and message gradients use the declared straight-through surrogate.

With gamma=1, lambda=.95, and V_tau=0 at actual termination or the task deadline:

\[
 \delta_t=r_t^{team}+V_{t+1}-V_t,\quad
 \hat A_t=\delta_t+\lambda\hat A_{t+1},\quad
 \hat R_t=\operatorname{stopgrad}(\hat A_t+V_t). \tag{7}
\]

The episode loss is

\[
 L_e=-50\sum_t\sum_j\log\pi_\theta(a_{jt}|h_{jt})
                       \operatorname{stopgrad}(\hat A_t)
          +\frac1{\tau_e}\sum_t(V_t-\hat R_t)^2. \tag{8}
\]

Actor time terms are summed; value error is averaged over episode time. There is
no advantage normalization and no extra division by agents or steps. The factor
50, widths, number of experts and lambda are specified experimental choices,
not mathematically optimal constants. [GAE](https://arxiv.org/abs/1506.02438)
provides the estimator construction and bias/variance motivation. An approximate
critic with lambda<1 generally introduces bias.

Collectors receive the same frozen model snapshot, collect complete episodes
until each has at least `batch_steps` environment steps, and return unnormalized
episode-gradient sums. The master performs

\[
 G=\frac{\sum_w\sum_{e\in w}\nabla L_e}{\sum_w |E_w|},
 \qquad \widetilde G=G\min(1,0.75/(\|G\|_2+\epsilon)), \tag{9}
\]

then one RMSprop step (lr=1e-4, alpha=.97, eps=1e-6). Framework clipping uses its
small numerical epsilon. Averaging per-worker means would over-weight workers
that completed fewer episodes; averaging by steps would define a different
weighting. Equation (9) specifies the collected empirical objective. Complete
episode collection can overshoot a step budget; exact counts are saved, and
stopping by a random episode count is not advertised as an unbiased estimator
of every population objective.

All carried recurrent tensors detach together every five steps. Straight-through
messages, GAE, and truncated BPTT make the production update approximate. The
tests and equations support a transparent implementation, not a claim that each
line of software has a convergence proof.

The horizon is the defined task deadline, so the value target at that deadline
is zero. The critic receives remaining time. A future continuing-task experiment
with an administrative rollout cut would need a bootstrap and a different task
specification; see [Pardo et al.](https://proceedings.mlr.press/v80/pardo18a.html).

If stochastic Gaussian roles or KL penalties are added later, specify their
joint distribution and full objective first. For a parameter-dependent future
cost K, earlier action scores must include its downstream cost as well as the
direct derivative of K. A simple counterexample is zero task reward and future
cost `K1=c*a0`: omitting that cost from earlier returns misses its action-choice
gradient. None of those extra losses are present here.

## 6. Baselines and the experiment matrix

All new models share observations, capability/action handling, recurrence,
binary width, two communication rounds, critic, learner and scenarios.

| Variant | Conditioning | PCP trainable parameters |
|---|---|---:|
| `shared` | One actual expert; no gate network | 171,087 |
| `banked` | Four experts; g(H,kappa) | 224,171 |
| `capability` | Four experts; g(kappa) | 222,123 |
| `constant` | Four experts; one learned global soft gate | 221,899 |
| `--no-feedback` | Remove previous communicated feature from LSTM input | Depends on variant |

The primary shared baseline is smaller. Report this capacity difference; a win
does not by itself attribute the gain to adaptive gating. Capability-only and
constant gates provide closer bank-capacity controls. Constant affine mixtures
have the function class of a single affine map at each bank, but a different
overparameterized optimization path, so retain both controls.

The [HetNet paper](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf)
motivates the domains and the class-indexed communication architecture. Our
launcher preserves the repository's domain recipes below while changing the
actor, information and team learner. This is a domain-matched reformulation,
not an exact numerical reproduction of the original method.

| Fixed recipe | Physical team | Map / vision | Horizon | Epochs × updates |
|---|---|---|---:|---:|
| PP | 3P,0A in the PCP simulator | 5×5 / 2 | 80 | 2000×10 |
| PCP | 2P,1A | 5×5 / 2 | 80 | 2000×10 |
| FC | 2P,1A; one fire; reward type3 | 5×5 / 1 | 300 | 1400×10 |

Default collection is at least 500 steps per collector/update, four collectors.
Those settings target at least 40M steps for PP/PCP and 28M for FC before episode
overshoot. Use actual step counts for learning curves and fair compute accounting.
The existing original launchers remain available separately.

For the primary PCP study, keep map5, vision2, horizon80:

| Study | Training compositions | Held-out evaluation |
|---|---|---|
| Composition (`--study composition`) | (1,1),(1,2),(2,1),(2,2),(3,1) | (3,2); separately (4,1),(4,2) extrapolation |
| Failure (`--study failure`) | (2,1),(2,2),(3,1) | Same composition strata, with and without failure; separately (3,2) and extrapolation |

Training samples compositions uniformly per episode. Failure training has
probability .5 of no event; otherwise a uniformly selected sensing agent loses
sensing at an integer step uniformly sampled from 10 through30 inclusive.
At least one other working sensor remains in these study compositions. Event
time is zero-based and failure is applied before that step's action. A task can
finish before its scheduled event; this is retained and explicitly reported.

Use seeds0,1,2 only for an initial screen. Lock architecture, hyperparameters,
checkpoint rule and evaluation distributions before using held-out results.
For a final comparison, plan at least ten independent training seeds if the
compute budget permits, and 500 common evaluation scenarios per composition per
seed. Ten is a practical target, not a guarantee of statistical power; report
uncertainty and all individual seed outcomes. Use a fixed final checkpoint or a
checkpoint selected only on a prespecified training-distribution validation set.

First compare shared and banked under the same training regimes. Then use the
capability-only, constant and no-feedback controls to test the proposed mechanism.
Do not tune on (3,2), (4,1) or (4,2). A frozen gate intervention does not freeze
the entire policy: recurrence and messages still evolve, so it tests the value of
gate changes within that learned model, not the causal necessity of all adaptation.

## 7. Frozen evaluation and uncertainty

Model parameters/buffers are hashed before and after evaluation. Team composition
changes cannot silently rebuild/train a different set of weights. Environment,
action and message randomness have separate namespaces. Action/message streams
are keyed by step; consuming extra diagnostic randomness cannot shift later draws.
These are common random numbers, not a guarantee of identical state trajectories
after different policies act.

Evaluate each failure scenario under normal gating, `freeze_affected`, `freeze_all`
and `comm_off`. Freezes pin the gate from the last pre-intervention step. The
intervention starts at the event, preserving the trajectory prefix. `--sham`
uses the same scheduled event, victim and streams but suppresses physical failure,
allowing the same interventions on matched no-failure controls. It requires
scheduled-event scenarios. Separate no-event evaluation requires no sham flag.

Save success, team return, per-capability-group return, steps, gate entropy/load,
null-attention mass, generated payload and scenario identities. Report event
exposure, pre-event completion and horizon-censored post-event completion. Raw
recovery time is only defined for exposed successful episodes; its mean alone
would selectively drop failures. The summary instead provides unconditional
success/return and horizon-capped completion time, with all episode records
retained for exposure-specific analyses. Conditional comparisons can select
different subsets when policies finish before the event, so primary comparisons
should retain the full assigned scenario panel.

`summarize` averages within each training seed and then weights seeds equally.
It bootstraps whole training seeds within each model/configuration/composition/
event/intervention stratum. A single seed has no between-seed interval. It does
not pool horizons, training-source hashes or checkpoint epoch/update counts, and rejects duplicate
scenarios and conflicting same-seed checkpoints. Intervals are conditional on
the supplied evaluation panels; few-seed percentile intervals remain unstable.
Exact collected steps/episodes remain available but are not equality conditions
for grouping. SoftRole evaluation schema version 2 records the checkpoint's
training-source hash separately from the evaluator source manifest/hash and
runtime inventory. Summary strata also separate evaluator hashes and schema
versions (absent in historical reports). The pilot below archives evaluator source;
standalone `evaluate` records its identity, so retain that checkout when archiving.
Event strata distinguish scheduled event presence,
sham status and intervention offset, rather than infer the event-time/victim
distribution. Callers must hold those evaluation distributions consistent within
a stratum. The emphasis on independent runs and uncertainty follows
[Agarwal et al.](https://arxiv.org/abs/2108.13264). A formal paired difference
analysis or hierarchical episode-and-seed bootstrap is not implemented. The
native pilot calculates descriptive paired means without inferential intervals.

The `evaluate-hetnet` path strictly loads existing `policy_net` tensors, creates
the requested count-specific graph using actual positions, and handles the
singleton LSTM squeeze through an isolated input-shape hook. It does not change
original model code or weights. Real and Binary variants are supported. It uses
corrected **typed** observations and is therefore a richer-information contextual
reference. It refuses sensor-failure scenarios because the original policy has
no corresponding current-capability contract. Evaluating a checkpoint trained
with the original observation bug under corrected observations is also a
distribution shift; label that explicitly when reporting it.

## 8. Running the implementation

Use the existing pinned environment from the repository's setup instructions.
No runtime dependencies were added. Run from the repository root.

```bash
# Inspect effective recipes without training or creating output.
bash scripts/softrole.sh pp shared 0 --dry-run
bash scripts/softrole.sh fc banked 0 --dry-run

# Tiny execution check; give each run a fresh output path.
.venv/bin/python -m softrole train --task pcp --model banked \
  --epochs 1 --updates-per-epoch 2 --batch-steps 4 --max-steps 4 \
  --nprocesses 4 --output runs/softrole_smoke

# Domain-matched fixed training; repeat with shared and independent seeds.
bash scripts/softrole.sh pcp banked 0

# Primary transfer and failure studies use separate roots.
SOFTROLE_RUN_ROOT=runs/softrole_composition \
  bash scripts/softrole.sh pcp banked 0 --study composition
SOFTROLE_RUN_ROOT=runs/softrole_failure \
  bash scripts/softrole.sh pcp banked 0 --study failure

# Held-out frozen composition evaluation: 500 episodes PER composition.
.venv/bin/python -m softrole evaluate \
  --checkpoint runs/softrole_composition/pcp_banked/seed0/checkpoints/epoch2000.pt \
  --compositions 3,2 4,1 4,2 --episodes 500 --seed 700 \
  --output runs/evaluation/composition_seed0.json

# Same checkpoint/scenario seed and failure schedule across interventions.
.venv/bin/python -m softrole evaluate \
  --checkpoint runs/softrole_failure/pcp_banked/seed0/checkpoints/epoch2000.pt \
  --compositions 3,2 --episodes 500 --seed 701 --failure-prob 1 \
  --failure-window 10 30 --intervention freeze_affected --trace \
  --output runs/evaluation/failure_freeze_seed0.json
# Repeat with --intervention none, freeze_all, comm_off; repeat each with
# --sham and a new output filename for matched no-failure controls.

# Resume the remaining budget into a NEW directory, preserving optimizer state.
.venv/bin/python -m softrole train \
  --resume runs/softrole/pcp_banked/seed0/checkpoints/epoch0050.pt \
  --output runs/softrole_resumed/pcp_banked/seed0

# Supply independently trained seed reports; one seed still produces means.
.venv/bin/python -m softrole summarize runs/evaluation/composition_seed0.json \
  --output runs/evaluation/summary.json

# Original typed contextual checkpoint; supply matching domain/model settings.
.venv/bin/python -m softrole evaluate-hetnet --task pcp --variant binary \
  --checkpoint PATH_TO_ORIGINAL_CHECKPOINT --compositions 2,1 3,2 \
  --episodes 500 --seed 700 --output runs/evaluation/hetnet.json

.venv/bin/python -m pytest -q
```

Explicit scenario files are JSON lists of `Scenario` records (see
`scenarios.py`). Evaluation reports preserve those records for replay. By default
evaluation selects a checkpoint's held-out compositions if present, otherwise its
configured fixed team. Always pass `--compositions` when comparing specific panels.

`slurm/softrole.sbatch` maps 18 screening jobs to three domains × two primary
models × three seeds. It is an optional launcher, not an automatic submission.
Its 16GiB/48hour requests are unmeasured starting values. The scripts enforce
fresh output directories. Keep the source checkout and environment unchanged
while worker jobs run; archived source is provenance, not a separately executed
checkout. Use an isolated checkout for concurrent source development.

Every run saves resolved configuration, copied source and SHA manifest, initial
signature, per-episode/update/epoch JSON records, model/optimizer checkpoints and
completion counts. Resume permits only budget/checkpoint-frequency changes and
preserves work-indexed randomness, including a partially completed epoch.

### 8.1 Native PCP failure pilot

`pilot-failure` tests unexpected sensor loss in **nominally trained** weights on
their native 2P1A team, map5, vision2, horizon80. It requires one shared and one
banked checkpoint per supplied training seed, matching training source and
configuration except model, seed and budget/checkpoint-frequency settings. It
does not select checkpoints, change their configuration or evaluate held-out teams.
Record a selection rule before inspecting pilot outcomes; exact checkpoint
steps/episodes and hashes remain in the manifest because equal epochs do not
imply equal samples.

The engineering pilot rule is: seed 0 for both models, first archived checkpoint
reaching 4M environment steps. This selects epoch0200 for both: shared 4,265,196
steps/77,778 episodes and banked 4,279,220/68,993, both 2,000 updates. These are
partial nominal-training checkpoints, not final research checkpoints. The
14,024-step difference is retained; the pilot does not estimate architectural
superiority or support a between-training-seed interval.

```bash
.venv/bin/python -m softrole pilot-failure \
  --checkpoints \
    stokes_runs/runs/softrole_primary/pcp_shared/seed0/checkpoints/epoch0200.pt \
    stokes_runs/runs/softrole_primary/pcp_banked/seed0/checkpoints/epoch0200.pt \
  --checkpoint-rule 'Seed 0; first archived checkpoint reaching 4000000 environment steps per model' \
  --episodes 20 --seed 1700 --output runs/sensor_failure_pilot_reference
```

The fixed reference panel assigns a failure to every scenario, with uniform
integer time 10–30 and uniform sensing victim. Every policy receives the same
panel and separate failure/sham runs with no gate intervention. The seed uses
the same composition namespace as `evaluate --compositions 2,1 --seed 1700`.
`pilot.json`, `scenarios.json` and the evaluator `source/` archive plus manifest
are written before outcomes. Per-condition JSON files include traces and runtime
versions; `summary.json` is written only after all pairs pass checks. An interrupted
pilot is left intact; use a fresh directory when rerunning.

All PCP SoftRole evaluations record these diagnostics, including when `--trace`
is absent. Training collection remains unchanged.

| Episode field | Meaning at the scheduled step, before masking/action |
|---|---|
| `pre_event_victim_reached` | Physical reached flag; under static PCP the victim remains at the target. |
| `pre_event_victim_target_visible` | Target appears in the cached pre-loss view; this view is masked before the failure policy can consume it. |
| `pre_event_victim_target_seen` | Target appeared in a directly delivered sensing view at a strictly earlier step. This excludes the event step and is false for an event at step zero. |

Fields are null when the episode ends before the schedule or no event is assigned.
Diagnostics read cached observations/state without simulator observation calls or
random draws. They never enter actor/critic inputs, rewards or transitions, so
the environment and checkpoint versions remain unchanged. Direct sight does not
measure what the agent remembers or learned through messages. Current visibility,
previous sight and reached status are overlapping diagnostics, not disjoint groups.

The summary verifies full scenario records, checkpoint/evaluator identities,
pre-event traces and diagnostics. It reports absolute success, return and
horizon-capped completion for both conditions, full-panel failure-minus-sham
means and discordant success counts for each policy. Exposure, pre-event success,
failed/censored completion and diagnostic counts retain explicit denominators.
The completion-time difference has the opposite desirability direction to success
and return: a positive value means slower completion under failure. No hypothesis
test, cross-model pooled estimate or paired inferential interval is computed.

To replay a condition or inspect mechanism controls, reuse the saved panel:

```bash
.venv/bin/python -m softrole evaluate \
  --checkpoint stokes_runs/runs/softrole_primary/pcp_banked/seed0/checkpoints/epoch0200.pt \
  --scenarios runs/sensor_failure_pilot_reference/scenarios.json \
  --intervention freeze_affected --trace \
  --output runs/sensor_failure_pilot_controls/banked_failure_freeze.json
# Repeat with --sham and a fresh filename; none/freeze_all/comm_off are also available.
```

Keep the evaluator checkout fixed during evaluation. The reference pilot only
uses `none`; the example freeze is a separate prospective control, not part of
its completed-panel summary. A full assigned-panel mechanism contrast remains
separate analysis. Earlier events or harder conditions require a declared
supplemental protocol before held-out evaluation.

### 8.2 PCP-only failure training launch

`scripts/softrole_failure.sh` uses the existing learner and declared failure
preset: compositions (2,1)/(2,2)/(3,1), failure probability .5, window10–30.
Indices 0/1/2 select shared seeds0/1/2; 3/4/5 select banked seeds0/1/2. Outputs
default to `runs/softrole_failure/pcp_MODEL/seedSEED`; set a new
`SOFTROLE_FAILURE_RUN_ROOT` for each study. The route accepts budget/collector/
checkpoint-frequency flags and `--dry-run`, and rejects protocol/model/seed/
output/resume overrides. It never repurposes nominal `--resume` as failure training.

```bash
bash scripts/softrole_failure.sh 0 --dry-run
# Tiny real training check with the reference horizon/window; not a research run.
SOFTROLE_FAILURE_RUN_ROOT=runs/sensor_failure_training_check \
  bash scripts/softrole_failure.sh 0 --epochs 1 --updates-per-epoch 1 \
  --batch-steps 1 --nprocesses 1 --save-every 1
# Optional research submission, after locking the budget and inspecting the pilot:
# mkdir -p logs_sr
# SOFTROLE_FAILURE_RUN_ROOT=runs/declared_failure_study \
#   sbatch slurm/softrole_failure.sbatch --total-steps DECLARED_BUDGET
```

The dedicated Slurm array is 0–5 with the original unmeasured resource requests;
it submits no jobs automatically. This first screen is distinct from the final
multi-seed comparison, mechanism controls and frozen held-out study in section6.

## 9. Evidence required before making research claims

The implementation is ready for experiments, not a positive research result.
Before claiming generalization, require frozen held-out evaluations across
independent training seeds, unchanged information contracts and fair budgets.
Before attributing a benefit to gating, require the shared/capability/constant
controls and event-timed freezes with shams. Report parameter counts, generated
bit budget, success/return and censored completion together. Retain unsuccessful
seeds. If the shared model matches the banked one, conclude that the extra gate
was unnecessary in that tested setting; do not infer semantic roles from plots.

For a strong graduate-school research portfolio, the valuable deliverable is a
clear question, corrected baseline assumptions, auditable mathematics, careful
experiments and appropriately limited conclusions. Novelty claims should follow
that evidence rather than substitute for it.
