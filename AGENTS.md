# SoftRole implementation record

## Authority and scope

This branch implements the **latest** agreed Capability-Conditioned HetNet plan
(29 September 2026), not the older `softrole.patch` or the older ignored `PLAN.md`.
The user authorized branch `softrole`, simple implementation, mathematical
traceability, and tests using the released HetNet domains. Original reproduction
entrypoints and their default numerical behavior must remain unchanged.

The main model uses deterministic soft gates. Gaussian roles, KL/persistence
regularization, and within-episode arrivals/departures are explicitly deferred.
Each agent knows its own sensing/actuation capabilities. No P/A label, typed
occupancy channel, agent identity embedding, or typed communication relation may
enter the new actor. Legacy simulator class names remain physical bookkeeping.

## Mathematical contract

1. Observation: concatenate the existing position encoding, three masked sensory
   channels (outside, target, total agent count), and own `kappa=(sense,actuate)`.
   Merge P/A **counts**, not boolean occupancy. Use independent observation copies.
   This removes information relative to the release and is documented as such.
2. Shared recurrence: `LSTM([ReLU(Pre(x)), previous communicated features,
   one_hot(previous_action)])`; one deterministic gate `softmax(MLP([H,kappa]))`
   per agent-step, reused in both rounds. No network dimension depends on team size.
3. Every bank is an affine convex mixture: `W(g)x+b(g)=sum_k g_k(W_k x+b_k)`.
   The shared baseline has one effective transform, not unused expert parameters.
4. Each round broadcasts independent bits `1[u+Logistic(0,1)>0]`, giving
   `Bernoulli(sigmoid(u))`. Heads share this payload. Production backpropagation
   uses the explicitly biased straight-through sigmoid derivative.
5. Receiver-gated decoding/GATv2 attention has a zero-message null candidate.
   Softmax includes all permitted senders plus null. Empty neighborhoods produce
   exactly zero received aggregate. Dense masked tensor operations are equivalent
   to the one-relation graph and keep this small-team implementation simple.
6. A shared six-action head masks physically impossible actuation to probability
   zero. The centralized permutation-invariant critic never feeds the actor.
7. Team reward is the sum of physical-agent rewards. All actor scores use common
   team GAE advantages. Actor coefficient 50, value coefficient 1, gamma 1,
   GAE lambda .95, detach gap 5, RMSprop lr 1e-4/alpha .97/epsilon 1e-6.
   Collectors return unnormalized episode gradient sums. Average once by the
   global number of episodes; clip once at .75; perform one optimizer step.
8. GAE with an approximate critic, straight-through bits, and truncated recurrence
   are approximations. No claim of an unbiased production estimator or proved
   generalization is permitted. Enumeration/score-function tests are references.
9. Evaluation freezes weights, uses separate reproducible random streams, records
   event exposure and censoring, and never tunes on held-out compositions.

## Engineering boundaries

- Add a standalone `softrole` package and launchers; do not apply the old patch.
- Support the released PP (PCP with 3P0A), PCP (2P1A), and FC (2P1A) training
  recipes. The primary composition/sensor-failure study is PCP. Unsupported event
  semantics in another domain must raise a clear error, not silently change it.
- The corrected observation path is versioned and opt-in. Preserve legacy defaults.
- CPU float64 and one Torch thread per collector match the current environment.
- Checkpoints must contain model configuration, optimizer state, counts, and source
  identity. Outputs never overwrite an existing run. Do not launch full research
  sweeps or cluster jobs as part of engineering validation.
- Keep changes small, direct, and auditable. Tests must check behavior or mathematical
  contracts, not merely repeat implementation expressions.

## Work log

### 2026-09-29 — start

- Inspected the clean tracked tree at `7d93a74`; the supplied PDFs and patch remain
  untracked inputs. Created branch `softrole` from `main`.
- Found that local `PLAN.md` contains the earlier stochastic-first proposal. The
  later user-approved deterministic, label-independent plan takes precedence.
- Identified the legacy per-collector clipping/step-division pipeline; the new
  learner will use its own simple aggregate-once update rather than inherit it.

### 2026-09-29 — implementation

- Added `softrole/config.py` with explicit released-domain recipes, deterministic
  variants, training/held-out composition splits and failure constraints. Added
  early validation for malformed programmatic integer/window inputs. Defaults
  are experimental choices rather than theoretically optimal hyperparameters.
- Added `softrole/env.py` and `scenarios.py`. The observation transform merges
  typed counts, retains position encoding, masks environmental sensing and adds
  own capabilities. Reused original simulators and rewards. Added an opt-in
  `independent_observations` copy path in PCP/FC to correct overlapping views;
  its default is false. Added isolated simulator RNG and work-indexed independent
  action/message streams. No original learner or model files were changed.
- Added `softrole/model.py`: shared encoder/LSTM; gate 66→32→4; affine convex
  banks; sender-gated encoders; receiver-gated decoders/GATv2/null logits; exactly
  two shared-head binary broadcasts; four 16-wide attention heads; physical action
  mask; invariant team critic. Score vectors have no extraneous affine bias.
  `shared` uses one actual expert, `capability` uses only own capabilities for its
  gate, `constant` has a learned global gate, and `--no-feedback` removes previous
  communication from the LSTM input. The implementation uses a dense single
  relation indexed receiver/sender instead of a DGL graph for the small teams.
- Added `softrole/learning.py`, `rollout.py`, `train.py`: complete-episode team
  GAE/loss, recurrent detachment, synchronous immutable worker snapshots, summed
  gradients, one global episode average/clip/RMSprop step. No shared gradient
  pointers, local clipping, step denominator or advantage normalization.
- Protected inverse-CDF sampling against cumulative roundoff assigning a masked
  final action. Added explicit finite-gradient checks and global count metrics.
- Added source/configuration snapshots, signatures, model/optimizer checkpoints,
  episode/update/epoch JSON records, fresh-directory protection, and resume.
  Resume validates environment/configuration, preserves parent source identity
  and continues remaining updates in a partial epoch rather than skipping them.
  Budgets may overshoot at a complete batch; exact steps/episodes are recorded.
- Added `softrole/evaluate.py`: frozen weights, explicit replayable scenarios,
  failure exposure/censoring, event-timed affected/all gate freezes and received
  communication removal. Added matched `--sham` evaluation, preserving event
  schedule/victim/noise while suppressing physical sensor loss. Actual and
  scheduled exposure/completion are separate fields. Unsupported FC/PP failure
  scenarios and malformed intervention timings are rejected.
- Added `softrole/hetnet.py`: strict original `policy_net` checkpoint loading,
  actual-position graphs and varying team counts with unchanged tensors; an
  isolated singleton LSTM shape hook preserves original-team outputs exactly.
  Supports Real/Binary PP/PCP/FC as a typed, richer-information contextual
  reference. Rejects original-model sensor failures and does not fabricate
  missing training-seed/source metadata. Corrected-observation evaluation of
  originally trained weights is explicitly a distribution shift.
- Added `softrole/report.py`: composition/configuration/intervention/sham strata,
  equal-weight independent training-seed means and whole-seed bootstrap intervals.
  One seed yields no between-seed interval. Duplicate scenarios and conflicting
  checkpoints are rejected; unsuccessful completion times use the horizon cap.
- Added CLI commands `train`, `evaluate`, `evaluate-hetnet`, `summarize`, a shell
  launcher and optional 18-job Slurm screening array. Extended package discovery
  in `pyproject.toml`; no dependency changes. Added README usage and the full
  revised mathematical plan in [softrole/RESEARCH.md](softrole/RESEARCH.md).
- Kept the supplied PDFs, old patch and ignored historical plan intact. Added
  ignore exceptions for this file and the research document so they can be
  checked in. No full training sweep, cluster submission, commit or push was run.

## Equation-to-code and evidence map

Equations refer to [the revised research document](softrole/RESEARCH.md). Their
scope is explicit: plumbing is justified by invariants/reproducibility, not by a
claim that every implementation line has a theorem or that optimization converges.

| Contract | Implementation | Independent behavioral/mathematical check |
|---|---|---|
| Label-independent input and physical capabilities | `env.py`; opt-in copies in the two simulator files | `test_softrole_env.py`: count preservation, view aliasing, sensory masking, domain success, pure observation and isolated RNG |
| Convex affine identity (1) | `model.py:AffineBank`, `HeadBank` | Explicit mixed-matrix comparison in `test_softrole_model.py` |
| Bernoulli bits (2), declared biased surrogate | `model.py:straight_through_bits` | Marginal sampling, hard-forward/derivative tests; enumerated score reference in `test_softrole_learning.py` |
| Receiver decoding and null attention (3–4) | `model.py:CommunicationRound` | Sender/receiver distinction, directed graph, empty-neighborhood and decoder-bias tests |
| Shared recurrence, equivariance and critic (5) | `model.py:SoftRoleNet` | Coupled-noise permutation test, actor independence from critic inputs/parameters, recurrent feedback and full detach checks |
| Team objective/GAE/loss (6–8) | `learning.py`; `rollout.py` | Enumerated cross-agent reward counterexample, exact finite-horizon GAE, episode weighting, masked-action roundoff regression |
| Global aggregation (9) | `train.py:collect_batch`, `apply_gradients` | Unequal collector episode counts, one clipping call, spawned-versus-serial gradient identity |
| Frozen transfer and interventions | `evaluate.py`, `hetnet.py` | Replay, matched prefixes/shams, immutable parameters, strict loading, singleton and original-domain checks |
| Independent-seed uncertainty | `report.py` | Unequal evaluation counts, no one-seed CI, duplicate/conflict rejection, separate compositions/shams |
| Checkpoint continuation | `train.py:train` | Exact model **and** RMSprop equality after partial-epoch resume versus uninterrupted training |

## Validation record

All checks below completed locally on 29 September 2026 using the existing
CPU float64 environment (Torch 2.2.1, DGL 2.1.0, NumPy 1.26.4). Each collector uses
one Torch thread. These are implementation checks, not estimates of task quality.

- `.venv/bin/python -m pytest -q`: **129 passed in 27.89 seconds**, comprising 100
  SoftRole tests and all 29 existing regression tests. No original regression
  failures were observed.
- `uv lock --check`: succeeded, 59 packages resolved; no lock/dependency update.
  `bash -n scripts/softrole.sh slurm/softrole.sbatch`: succeeded.
  `git diff --check`: succeeded.
- Twelve actual training runs: PP/PCP/FC × shared/banked × one/four collectors,
  two updates each, horizon 4 and batch floor 4. All produced finite nonzero
  gradients, saved checkpoints and changed signatures. One-collector runs each
  collected 8 steps/2 episodes; four-collector runs each collected 32 steps/8 episodes.
  Records: `runs/softrole_validation/final_matrix/matrix.json` and adjacent logs.
- Two actual four-collector PCP study runs: composition and failure presets,
  two updates, horizon 6, batch floor 7 (failure window 1..3 for this smoke only).
  All five composition-training teams and all three failure-training teams were
  exercised; eight failure-training episodes reached their sensor event.
  Records: `runs/softrole_validation/final_studies/`.
- Frozen held-out 3P2A evaluation exercised normal, affected-gate freeze,
  all-gate freeze and communication-off, each with actual failure and matched
  sham (three scenarios per condition). All eight succeeded as executions;
  actual/scheduled exposure fields and eight separate summary strata matched
  expectations. Single-seed intervals remained null. These were tiny untrained
  policy checks, not positive generalization results.
- Original Binary checkpoint CLI evaluation and summary ran on 1P2A and 3P2A with
  immutable weights; Real/Binary PP/PCP/FC and transfer are also covered by tests.
  Final independent mathematical review found no substantive blocker; clarified
  fresh-noise conditioning, terminal V_tau and report grouping limitations.

## Remaining research work and boundaries

Run the locked research protocol and assess actual learning across independent
seeds. No learned robustness, improved return, semantic roles or superiority to
HetNet has been established. Own sensor state is known locally; fault detection
is not learned. Only PCP sensor loss and episode-level roster changes are
implemented. Gaussian roles/KL, mid-episode membership changes, physical FC
sensor-failure semantics and paired inferential comparison are deferred.

Summary confidence intervals condition on the supplied evaluation panels and
require consistent event-time/victim distributions. They stratify training-source
hash and checkpoint epoch/update, not exact step overshoot. Evaluations currently
record the checkpoint training-source hash; preserve the evaluator checkout
separately. Keep source and environment fixed while spawned-worker jobs run.
The optional Slurm memory/time requests are unmeasured starting settings.
