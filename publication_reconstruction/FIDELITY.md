# Paper-v1: evidence, declared choices and validation contract

The approved October 4 reconstruction first reconciles paper/code differences,
then compares DGL and PyTorch execution of **that same new model and learner**.
It supersedes the earlier exact-backend-only performance proposal for new runs.
Historical/public-code/supplement-v1 sources and checkpoints retain their identity.
This is a paper-aligned reconstruction, not a recovered publication checkout.

## Primary evidence and hierarchy

- [HetNet paper](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf), §§4–5,
  equations 2–6 and Algorithm 1; local extraction [local paper text](../research/papers/HetNet.txt).
- [Pinned authors' supplement](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/AAMAS_22___HetNet_Supplementary.pdf),
  §§1–2; local extraction [local supplement text](../research/papers/HetNet_Supplementary.txt).
- Versioned public-source identities in `ORIGINS.json`; pre-change executed bytes
  in `analysis/paper_v1_2026-10-04/baseline`, with `baseline_manifest.json`.

Explicit published requirements take priority. Unspecified details retain declared
public-source conventions. Ambiguities are not represented as recovered facts.
The user selected 64 bits **per head**: four independent heads therefore transmit
256 payload bits/sender/round, 768 across three rounds, excluding critic traffic
and edge/broadcast accounting. Do not label this the publication's 64-total-bit
experiment or reuse its bandwidth-reduction claims.

## Equation-to-code map

| Requirement or decision | Evidence | Implementation/check |
|---|---|---|
| Class-specific preprocessing and recurrence | Paper Fig. 1, §4.2; old `uavnet.py` shared P/A status weights | [`PaperNet`](runtime/hetgat/paper.py); distinct class parameter and recurrent tests |
| Three layers, 4 heads, 16 hidden features/head, concatenation then final averaging | Supplement §2.1 | PaperNet structural and head-merge tests |
| Independent Binary head channels | Paper §4.3 independent sublayers; older `graph/hetgata2c.py` independent ModuleList heads | [`PaperLayer`](runtime/hetgat/graph/paper.py); independent payload/RNG checks;64 bits/head is explicit user interpretation |
| Per-relation incoming attention and self feature addition | Paper equations 2–4 and receiver-phase description | DGL [`PaperLayer`](runtime/hetgat/graph/paper.py), Torch [`aggregate`](runtime/hetgat/graph/torch_backend.py); independent tiny-neighborhood reference |
| Per-class critics with actor isolation | Paper §§4.2,5.2; supplement§2.1 | SEN isolation tests; two-SEN class routing/width 8 retained from public source |
| SEN metadata `[N_P,N_A,world_dim,t]` | Paper §4.2 lists current time among metadata | Zero-based episode step; no metadata edge to actor |
| FC 4/5 physical action widths, discovery-before-extinction | Supplement §1.3 | Paper-only simulator branch; negative/invalid-action and discovery tests |
| FC additive time/new-fire/extinction/false-drop rewards | Supplement §1.3 | Unique ignition/extinction events; hand-computed transition rewards |
| Per-agent policy credit, class-average rewards-to-go critic targets | Paper equation6,§5.2,Algorithm 1 | [`paper_learning.py`](runtime/hetnet_ext/paper_learning.py); unequal-reward/unequal-class tests |
| Global completed-episode mean and total-team-N weighting | Paper defines N as total agents (§3.1); finite episode average is declared batching choice | Gradient partition and global-clipping tests |
| Adam 1e-3 | Supplement §2.1 | Active trainer optimizer and checkpoint validation |

## Precisely declared reconstruction choices

PP/PCP keep 5×5, 3P0A/2P1A, H=80, stationary prey,source-derived sensing radius 2,
five P actions/six A actions,distinct reset placements,reaching-based completion,
absorbing completed agents and zero penalty on the completing transition.

FC keeps 5×5, 2P1A, one initial fire, H=300, radius 1, 29-entry typed sensory cells,
blind A sensory channels and the corrected FARSITE propagation implementation.
Actions are 0..3 movement; native A action 4 extinguishes. Only a previously observed,
currently active fire can be extinguished. Actions/suppression precede spread,
which precedes the next observation. An unsuccessful drop (including an
undiscovered fire) changes no fire state and incurs the A-only false-drop penalty.
Every transition, including successful termination, receives the temporal penalty:

`r_i = -.1 - .1 * unique_new_ignitions + 10 * unique_extinctions - .1 * false_drop_i`.

New ignition counts compare burning grid cells immediately before and after
propagation; repeated/subcell front outputs do not count as new cells. Existing
corrected boundary/front/pruning conventions remain; arbitrary random-neighbor
spread is not substituted for the paper's author-derived FARSITE model.

For each agent, compute GAE from individual rewards and its class value. Separately
compute Monte-Carlo discounted rewards-to-go G. For episode e with total N agents:

`L_actor = -sum_{i,j,t}(log(pi_ijt) * stopgrad(A_ijt)) / N`

`L_value = sum_{i,t}(N_i * (V_it - mean_j(G_ijt))^2) / N`.

Average `L_actor + L_value` once over all completed episodes from all collectors.
Apply one global norm clip at .75 and one active Adam step. No padded standardization,
actor multiplier 50, critic time mean, local clipping or final step denominator is
used in this new learner. Those remain preserved public-code heuristics in old
versions. Gamma 1, lambda .95, detach gap 5, zero entropy,global clip at .75,Adam betas (.9, .999),
epsilon 1e-8 and zero weight decay are declared choices where not specified.
Natural termination and finite task horizon bootstrap to zero.

Four collectors and 500-step floors determine a variable number of complete
episodes per update. Retained 40M PP/PCP and 28M FC budgets are study choices, not
recovered publication budgets. Shared-backbone actor/value losses are combined 1:1.

## Backend and experimental validity

`--reconstruction-spec paper-v1` locks architecture/environment/learner together.
`--message-backend dgl|torch-v1` selects execution; DGL is the default. Torch
initially supports only CPU, fixed-within-episode teams, unlimited non-lossy links.
Topology caches contain no learned tensors. Both backends use the same parameters,
independent-head Gumbel sampling and mixed float64/float32 attention dtypes.

Cross-backend tolerance is `atol=1e-10, rtol=1e-8` for float64 and
`atol=1e-6, rtol=1e-5` for float32. Shapes, dtypes, initial state, RNG consumption and
gradient-presence masks are exact. Same-backend resume must replay exactly.
Long-run stochastic trajectories need not match after roundoff changes. Numerical
agreement with DGL alone is not a substitute for equation and environment tests.

SoftRole paper-FC uses the very same simulator bytes, with its existing six-logit
head: stay is masked for every physical class and internal extinguish action 5 maps to
native action 4. Its own observation transform and learner remain unchanged. Fresh shared
and banked seeds 0–2 are prepared at 28M steps with the same nominal FC panel as
HetNet. Prior FC results are an unmatched environment stratum and are never pooled.

The backend benchmark has 24 sequential runs (4 workloads × 3 pairs × 2 backends), 20
updates each, seed 991, 4 collectors, floor 500 and production horizons. Pair orders are
DGL/Torch, Torch/DGL, DGL/Torch. Report updates 2–20 throughput, all 20, startup, full
process and resource intervals; profile diagnostics are separate. Divergent
trajectories require fixed-rollout compute replay before a pure compute-speed
claim. The official 100-update readiness preflight remains separate.

Earlier Mac estimates concern a different model/learner and do not predict this
version or Stokes. New layers/head channels may increase computation. All measured
results and remaining limitations are recorded in `AGENTS.md` and the fresh audit
folder. Passing finite tests does not prove convergence or reproduce published scores.

## Persistent checks and operational evidence

- [Architecture and backend tests](../tests/test_publication_paper_model.py):
  typed attention, head independence, recurrent outputs and gradients, RNG, empty
  relations, topology isolation, observation layout and critic isolation.
- [Learner equations](../tests/test_publication_paper_learning.py),
  [complete updates and recovery](../tests/test_publication_paper_training.py),
  [fixed-rollout replay](../tests/test_publication_paper_replay.py), and
  [shared simulator/SoftRole checks](../tests/test_softrole_paper.py).
- [Launcher tests](../tests/test_publication_paper_launcher.py) and
  [benchmark integrity tests](../tests/test_publication_paper_benchmark.py):
  fixed budgets, separate diagnostics, episode reconciliation and pairing gates.
- [Archived CLI validation script](../analysis/paper_v1_2026-10-04/validate_integration.py)
  and [implementation record](../AGENTS.md): actual commands, executed source
  identities, completed checks and explicit limits of each measurement.

The throughput gate requires improvements in all three post-startup update pairs
and a higher median segment throughput. Normal episode logs contain returns and
counts, not complete action/observation trajectory digests; the harness therefore
requires fixed-rollout replay for all current workloads. Replay measures serial
model/learner computation, including RNG restoration and Python iteration; it
excludes the environment and interprocess communication. It is not end-to-end
training throughput. Resource review and the official preflight remain separate.
