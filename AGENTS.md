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

### 2026-09-29 — architecture comparison document

- Re-read the three supplied PDFs, the original patch, the historical plan,
  current implementation and original HetNet source for the requested comparison.
  Distinguished the 37-page and 39-page implementation guides and credited
  retained proposal mechanisms rather than presenting them as new corrections.
- Added [ARCHITECTURE_COMPARISON.md](softrole/ARCHITECTURE_COMPARISON.md), covering
  a three-way architecture comparison, exact tensor dimensions, deterministic
  gate promotion, observation changes, team credit/critic/aggregation, proofs
  and counterexamples, primary research evidence, and defensible contributions.
- Verified cited primary research on HetNet, capability-aware teaming, CASH,
  ROMA, GATv2, stochastic computation graphs, GAE, Deep Sets and RL evaluation.
  Explicitly separated research precedents from evidence about this implementation.
- Independent reviews checked proposal attribution and original HetNet behavior.
  Corrected the draft to acknowledge that held-out (3,2) combines composition
  and size transfer, and that single-sensor loss has different fractional
  severity across compositions. No experimental result or runtime code changed.
- Added a README link and ignore exception for the document; clarified that the
  original reproduction tranche and the separate SoftRole failure study have
  different scopes. Checked local link targets, math delimiters and whitespace.
  No training or test suite was rerun for these documentation-only changes.

### 2026-09-30 — training-log snapshot analysis

- Analyzed the copied `logs_1/` and `logs_sr/` inputs: 27 runs and 9,708 completed
  epochs. Added the [minor report](analysis/training_2026-09-30/report.md), four
  plot pages in PNG/PDF, per-epoch and per-run CSVs, JSON summaries, a reproducible
  analysis script and input/code hashes in `analysis/training_2026-09-30/`.
- Traced metric definitions to the local training/environment code. Proved that
  reproduction stdout overcounts cumulative steps/episodes by repeatedly adding
  within-epoch cumulative totals; used epochs for those curves. SoftRole counts
  reconcile exactly. Its task/model/seed mapping remains conditional on the
  current Slurm mapping because copied stdout lacks configuration/source records.
- Compared equal-epoch means at common update windows, then weighted seeds
  equally. Observed seed ranges are descriptive, not confidence intervals.
  Documented PP/PCP progress, the current PCP shared-versus-banked gap and FC
  diagnostics without claiming convergence, robustness or final superiority.
- Included explicit counter, entropy, attention and loss-scaling reasoning, with
  primary research references for evaluation uncertainty and GAE. Independent
  reviewers recalculated reproduction summaries and 870 SoftRole summary values;
  all agreed. Clarified sufficient versus necessary conditions for counter
  correction, empty-neighborhood attention and unavailable value diagnostics.
- Reconciled counts/returns, checked finite metrics and contiguous epochs,
  inspected all four figures and verified input hashes, local links and whitespace.
  Plotting used isolated `uv run` dependencies (NumPy 1.26.4, Matplotlib 3.10.7).
  No training dependencies, model code, jobs or checkpoints were changed; no
  training or test-suite rerun was needed for this log-analysis work.

### 2026-09-30 — full run artifact amendment and PDF

- Audited all 37 folders in `stokes_runs/runs/`. Kept 27 primary runs (11,375
  completed epochs, 252,821,848 steps, 10,820,962 episodes) separate from ten
  ancillary launches that exactly repeat primary numerical/model prefixes.
  Streamed all 8,475,482 SoftRole episode records, including partial trailing
  updates; reconciled every completed epoch and all 63,214 available updates.
- Verified recorded configurations, all 73 archived source files per SoftRole
  run, and checkpoint identities. All 120 SoftRole and 97 primary reproduction
  checkpoints passed actual tensor/finite optimizer/count checks. The additional
  speed-check checkpoint passed sidecar/file checks. Documented mixed legacy
  tensor dtypes, overlapping legacy collector seed values and missing host details.
- Amended [the report](analysis/training_2026-09-30/report.md) using true recorded
  sample counts, episode-weighted windows near common task budgets and equal-run
  means. Preserved the earlier numerical snapshot in `stdout_snapshot.md`, with
  the explicit correction that static-prey PP/PCP natural termination implies
  success because reached agents cannot leave the target.
- Added six figures, raw/summary CSVs, JSON audits, source/input/checkpoint hashes
  and reproducible analysis scripts under `analysis/training_2026-09-30/full_runs/`.
  Derived gate variability as expected KL and nominal event exposure from episode
  lengths. Explained ubiquitous clipping with RMSprop, incompatible original
  versus SoftRole loss normalization, incomplete budgets and causal limitations.
- Published `analysis/training_2026-09-30/training_report.pdf` with LaTeX-rendered
  equations and vector figures using isolated Pandoc 3.6.1/Tectonic tooling.
  Independent reviews checked numerical summaries, provenance and mathematical
  interpretation. Checked rendered figures/PDF, local links and whitespace.
- No training, policy evaluation, model edits, input/checkpoint edits, dependency
  changes, cluster actions, commit or push were performed. The analyses do not
  establish capability adaptation: all supplied primary runs have fixed teams
  and zero configured or exposed failures. Different copied ledger endpoints
  are recorded explicitly and are not treated as training corruption.

### 2026-09-30 — sensor failure agent handoff

- Added [SENSOR_FAILURE_HANDOFF.md](SENSOR_FAILURE_HANDOFF.md), a self-contained
  briefing and copyable continuation prompt for the next agent. It distinguishes
  existing PCP failure mechanics from remaining pilot diagnostics, reporting and
  launch preparation, and records the agreed scope and interpretation limits.
- Explicitly carried forward the user's requirements: simple, interpretable,
  reproducible code following repository conventions; record exact work and
  every touched file in this existing uppercase `AGENTS.md`.
- Files touched: `SENSOR_FAILURE_HANDOFF.md` (new handoff), `README.md` (discovery
  link), `.gitignore` (narrow Markdown exception), and `AGENTS.md` (this record).
- Checked current CLI/source and checkpoint availability. Confirmed the failure
  preset with `python -m softrole train --task pcp --study failure --model shared
  --seed 0 --output runs/sensor_failure_dry_run --dry-run` using `.venv/bin/python`;
  no run directory was created. Checked links, shell example syntax and whitespace.
  An independent read-only review checked operational pitfalls and scope.
- No implementation, training, policy evaluation, submission, dependency change,
  checkpoint/input mutation, commit or push was performed. Numerical findings
  and prior test results are attributed to their original snapshot/check dates.

### 2026-09-30 — reproducible native PCP sensor-failure pilot

- Change: audited and reused the existing sensor-loss mechanism, shams and
  interventions. Added optional cached diagnostics for the victim's reached
  flag, current pre-loss target visibility and strictly earlier direct target
  observation. PCP evaluation enables them; default training collection does not.
  Added evaluator source/runtime identity and mutation checks, evaluation schema
  version 2 and corresponding summary strata. Environment/checkpoint versions
  and learner semantics remain unchanged.
- Change: added `pilot-failure`, a native nominal shared/banked comparison with
  an explicit common panel, evaluator source archive and a pre-outcome checkpoint
  selection manifest. It verifies scenario/episode identities, paired prefixes
  and diagnostics, then reports absolute performance and descriptive full-panel
  failure-minus-sham outcomes. Added a six-index PCP-only failure-training route
  and optional Slurm array, reusing the existing launcher and declared preset.
- Files touched (repository files, with purpose):
  - `softrole/env.py`: pure cached PCP victim-status reader.
  - `softrole/rollout.py`: opt-in pre-event history and bool/null diagnostics.
  - `softrole/evaluate.py`: diagnostic activation, source/runtime identity,
    exact loaded-checkpoint byte identity and source/checkpoint mutation checks.
  - `softrole/__init__.py`: separate `EVALUATION_VERSION = 2` constant.
  - `softrole/train.py`: read-only source-identity option; archive and hash the
    same bytes from a single read. No sampling, loss or update changes.
  - `softrole/report.py`: separate evaluator-hash/schema strata.
  - `softrole/pilot.py` (new): native nominal-transfer panel, checkpoint/source/
    configuration eligibility, source archive and paired descriptive summary.
  - `softrole/__main__.py`: pilot CLI and concise evaluator console output.
  - `scripts/softrole_failure.sh` (new): fixed PCP protocol, explicit six-index
    mapping, isolated output root and rejection of protocol/resume overrides.
  - `slurm/softrole_failure.sbatch` (new): optional PCP-only 0–5 array.
  - `tests/test_softrole_evaluation.py`: timing/history/censoring,
    noninterference, provenance fields and mutation-rejection tests.
  - `tests/test_softrole_report.py`: evaluator/schema separation tests.
  - `tests/test_softrole_pilot.py` (new): full-panel arithmetic, invalid pairing,
    eligibility, archive hashes, exact replay and overwrite/immutability checks.
  - `tests/test_softrole_failure_launcher.py` (new): mapping, output isolation,
    default invocation, actual dry run and override-rejection checks.
  - `README.md`: pilot discovery and PCP-only launch instructions.
  - `softrole/RESEARCH.md`: diagnostics, provenance, checkpoint rule,
    pilot/replay/launch commands and inference limits.
  - `AGENTS.md`: this additive record and updated provenance boundary below.
- Files touched (generated artifacts): only fresh
  `runs/sensor_failure_validation_20260930/`. Its `artifact_manifest.json` lists
  the exact paths and SHA256 hashes of all 272 other generated files, including
  copied source trees; it excludes itself. Principal records are
  `pilot/{pilot,scenarios,source_manifest,summary}.json`, four condition reports
  with traces, `training/pcp_{shared,banked}/seed0/` run records/source/checkpoints,
  `{shared,banked}_trained_evaluation.json`, `stratified_summary.json`,
  `validation.json`, and six command logs. Existing inputs, copied checkpoints
  and earlier validation outputs were preserved.
- Mathematical/experimental contract: the event snapshot precedes masking and
  action at zero-indexed step t; previously-seen history uses only steps less
  than t. The current pre-loss view is masked before the failure actor consumes
  it. Unexposed/no-event diagnostics are null. No diagnostic enters actor/critic,
  reward or transitions, and no extra observations/random draws are generated.
  Full assigned panels remain primary; exposed-only counts retain denominators.
  Matching training configurations/source do not imply exact sample equality.
  Paired means are descriptive, with no inferential CI or semantic-role claim.
- Validation: `.venv/bin/python -m pytest -q` completed with **178 passed in
  50.27 seconds**, including all 29 original regressions. Focused checks also
  passed: environment/evaluation 44 tests, launcher 21, pilot 15, and final
  provenance evaluation/report 41. `uv lock --check` resolved 59 packages without
  changes. `bash -n scripts/softrole.sh scripts/softrole_failure.sh
  slurm/softrole.sbatch slurm/softrole_failure.sbatch` and `git diff --check`
  passed. Six direct and six mocked-Slurm dry-run mappings passed without
  training/submission. Documentation shell blocks and local links were checked.
- Validation: two actual one-update training commands used INDEX 0 and 3:

  ```bash
  SOFTROLE_FAILURE_RUN_ROOT=runs/sensor_failure_validation_20260930/training \
    bash scripts/softrole_failure.sh INDEX --epochs 1 --updates-per-epoch 1 \
    --batch-steps 1 --nprocesses 2 --save-every 1
  ```

  Shared collected 160 steps/2 episodes, banked 138/2; each exposed one failure
  and sampled (2,2)/(3,1). Model/optimizer tensors were finite, signatures changed,
  and pre-clip gradient norms were 1561.6020/1136.9471. These tiny executions do
  not establish learning or cover every failure-training composition.
- Validation: for MODEL shared and banked, each fresh checkpoint then ran:

  ```bash
  .venv/bin/python -m softrole evaluate \
    --checkpoint runs/sensor_failure_validation_20260930/training/pcp_MODEL/seed0/checkpoints/epoch0001.pt \
    --compositions 2,1 --episodes 2 --seed 1701 --failure-prob 1 \
    --failure-window 10 30 --trace \
    --output runs/sensor_failure_validation_20260930/MODEL_trained_evaluation.json
  ```

  All four episodes reached the event and checkpoint hashes were unchanged.
  Explicit compositions avoided the failure preset's default held-out panel.
- Validation: the archived nominal pilot ran this exact command:

  ```bash
  .venv/bin/python -m softrole pilot-failure \
    --checkpoints \
      stokes_runs/runs/softrole_primary/pcp_shared/seed0/checkpoints/epoch0200.pt \
      stokes_runs/runs/softrole_primary/pcp_banked/seed0/checkpoints/epoch0200.pt \
    --checkpoint-rule 'Seed 0; first archived checkpoint reaching 4000000 environment steps per model' \
    --episodes 20 --seed 1700 --output runs/sensor_failure_validation_20260930/pilot
  ```

  The predeclared rule selects 4,265,196 shared and 4,279,220 banked steps, both
  2,000 updates; the 14,024-step difference remains recorded. The 20 common
  scenarios use window 10–30, probability 1 and uniform sensing victim. Four
  reports contain 80 outcomes. All paired prefixes/diagnostics and immutable
  checkpoint/model checks passed. Source stayed fixed during jobs at evaluator
  snapshot `288f650c37ca38542a3917157ba1c3bc2c3a9d975db57eaa972c1e9efa6c0351`;
  this log and a README wording clarification were added after completion.
- Validation: `.venv/bin/python -m softrole summarize` on the four pilot condition
  reports, with `--output runs/sensor_failure_validation_20260930/stratified_summary.json`,
  produced four strata and null single-seed intervals. `validation.json` records
  source-archive, tensor/count/signature and checkpoint checks. An independent
  read-only audit recalculated every pilot count/mean/delta/discordance, verified
  all 80 records, panel identities, prefixes, diagnostics, exposure/censoring,
  checkpoint bytes/tensors/progress, all 78 evaluator source files and manifest
  hashes. No implementation/test failure remained.
- Research implications: on this one-seed panel, shared failure/sham success
  was 14/20 versus 15/20 and banked 16/20 versus 15/20. Shared mean returns were
  -3.4575/-3.2300 and horizon-capped completion 47.25/44.30; banked returns were
  -3.6350/-3.6575 and completion 45.40/47.65. Exposure was 15/20 shared and 18/20
  banked; early completions were 5 and 2. Of exposed victims, shared 8/15 and
  banked 11/18 had reached the target, 13/15 and 15/18 had current pre-loss
  visibility, and 14/15 and 18/18 had previously directly seen it. These
  overlapping counts show that this panel mostly removes sensing after target
  discovery. Prior sight is not retained knowledge; reached victims can still
  affect team communication. Both discordant-success pairs involved reached
  victims. Each +/-5-point success change is one scenario, not evidence of
  significance, robustness, model superiority or beneficial sensor loss.
  The reference timing was not adjusted after viewing outcomes.
- Remaining work: prerecord any supplemental earlier/harder condition using
  training compositions, run the locked multi-seed failure-training comparison
  and matched mechanism controls, then perform final frozen held-out evaluation.
  Formal paired seed-level inference and gate difference-in-differences remain
  unimplemented. No full sweep, held-out tuning, cluster action, dependency
  change, legacy-model edit, commit or push occurred. Work stayed on inspected
  `main` at `6994d87`; all pre-existing documentation, ignore-file, handoff,
  analysis/log/archive changes were preserved.

### 2026-09-30 — advisor presentation and speaking guide

- Reworked the supplied 22-page `Capability-Conditioned HetNet Deck Review and
  Revised Outline.pdf` into an 18-slide PDF (15 main slides, three backups) and
  an 18-page speaking guide in `analysis/advisor_sync_2026-09-30/`. Both share
  `content.tex`, with rendered equations, a vector architecture diagram, fresh
  training plots, evidence notes and primary-paper links. The supplied PDF was
  preserved. Added editable LaTeX sources, build instructions, `FACT_CHECK.md`,
  `verified_facts.json` and a PDF/build hash record.
- Rechecked all 27 primary metric ledgers: 11,375 completed epochs,
  252,821,848 steps and 10,820,962 episodes. Independently recomputed the
  common-budget and 4M episode-weighted windows and equal-seed means; verified
  fixed/no-failure configurations, all 63,214 available update norms, and six
  archived source paths against all 18 SoftRole manifests. Distinguished
  current evaluator edits from archived training code. Prior full checkpoint
  and streaming episode audits are cited as prior work, not claimed as rerun.
- Included the separate sensor pilot and 178-test validation record that became
  available during presentation preparation. Read-only checks verified its
  four raw reports, common panel, paired pre-event traces, diagnostic counts,
  success discordances and two checkpoint byte hashes. The one-seed, 20-scenario
  panel is described as a diagnostic pilot; neither the +/-5-point changes nor
  the engineering checks are presented as robustness or architecture rankings.
- Corrected stale Real/Binary comparisons, observation-version claims, nominal
  class-information wording, budget statements and overstrong convex-bank or
  gradient claims. Credited Howell/CASH and separated mathematical identities,
  implementation checks, partial training evidence and remaining experiments.
- Verified both PDFs have 18 pages, embedded fonts, expected numbers and no TeX
  overflow/underflow or missing-character warnings. Rendered and visually
  reviewed every page; checked local links, input hashes and whitespace.
  `git diff --check` passed. Plotting used an isolated NumPy/Matplotlib environment.
  This presentation task changed no runtime code and launched no training,
  evaluation or test suite. Concurrent work was preserved; no commit or push.

### 2026-09-30 — resynced stdout training update

- Located newer data in `logs_1/` and `logs_sr/`; the structured
  `stokes_runs/runs/` archive is unchanged. Parsed 26,878 complete epochs in
  27 runs, 15,503 beyond that archive. Verified all old stdout byte prefixes,
  exact archived SoftRole metric prefixes, reproduction prefixes within their
  printed rounding, configuration identities and unchanged stderr. SoftRole
  step/episode totals reconcile; its new logs record zero exposed failures.
- Added only `analysis/training_2026-09-30_resync/` and this log entry. New source
  files are `analyze.py` (reuses the earlier parser/audit), `build_pdf.py`,
  `quick_report.md`, `README.md` and a local `.gitignore`. Generated records are
  `summary.json`, `provenance.json`, `validation.json`, `analysis_console.json`,
  `epoch_metrics.csv`, `run_summary.csv`, `quick_report.tex`/`.pdf`, and
  `training_plots.pdf`. Individual PNG/PDF figures are `reproduction_training`,
  `softrole_training`, `softrole_common_samples` and `softrole_diagnostics`;
  ignored previews/build logs support visual review. `validation.json` records
  deliverable hashes; source inputs are hashed in `provenance.json`.
- Kept reproduction on an epoch axis because its stdout counters overcount.
  Cross-method epoch windows weight epochs equally; SoftRole sample comparisons
  pool episodes within a run and weight the three seeds equally. Compared PP at
  23.5M, PCP at 15M and FC at 22.5M steps, with endpoint shortfalls below 0.12%.
  Preserved the earlier full-artifact report and advisor PDFs as dated snapshots.
- Findings: PCP shared/banked success is now near 100%, but shared averages
  8.04 versus 10.97 steps per episode at 15M. FC common-budget success is
  66.87% versus 55.09%, followed by substantial seed variability. Shared FC seed
  2 has a sustained recent regression and partial last-ten-epoch recovery;
  value-loss association is not a causal explanation. No convergence, final
  model ranking, fresh clipping audit or adaptation result is claimed.
- Validation: independently recomputed 432 exported window metrics, checked
  input hashes again, local links and whitespace, and visually reviewed all
  four figures and all three report pages. The report compiled without TeX
  overflow/underflow or missing-character warnings. `git diff --check` passed.
  Primary HetNet and RL-evaluation sources support the context and inference
  limits; estimator and counter identities are derived in the report.
- Used isolated plotting/Pandoc environments. No runtime code, dependencies,
  logs, checkpoints or previous analysis artifacts changed; no training,
  evaluation, test suite, cluster action, commit or push was performed.

### 2026-09-30 — FC environment and seed variability audit

- Audited the current default FC simulator, wildfire propagation, observation,
  reward/termination, learner, latest raw stdout and archived source/configuration.
  Added [the audit report](analysis/fc_audit_2026-09-30/report.md) and reproducible
  diagnostic probes. Preserved all pre-existing changes on `main` at `6994d87`.
- Confirmed an active released-simulator defect: skipped top/left-boundary fire
  fronts leave zero-filled `[0,0,0]` rows that FireCommander treats as origin fires
  and +10-reward sources. Eight of 25 initial cells immediately add a spurious
  origin fire; 945/1000 passive resets reached the placeholder by horizon 300,
  716 by step 10. No corrected physical semantics were silently substituted.
- Confirmed original-only observation aliasing (already prevented by SoftRole),
  wind samples outside the nominal maximum, coordinate rounding/truncation
  mismatch, extinguished-cell reentry/front stalling, and source-discovery
  membership misclassification (zero default discovery rewards). All 200 default
  type/position/direction movement cases passed; checked final-fire termination.
  A privileged source-first controller solved 100/100 resets in mean 11.52 steps
  versus random 36/100. This establishes bounded full-state solvability, not
  native partial-observation learnability or the defect's causal effect.
- Rechecked six FC archived configs (only model/seed differ), all 438 archived
  source-file hashes, exact stdout/archive prefixes, 475 log-audit input hashes
  and 180 prior-summary values. Shared seed 2's preceding/latest 50-epoch success
  is 70.87%/20.67%, with value MSE 465.63/2077.97 and partial latest recovery.
  Same-budget 22.5M seed results and legacy context are separately tabulated.
  New stdout tails remain outside the copied checkpoint/source archive.
- Ran a prespecified frozen epoch 200 panel for shared/banked × seeds0,1,2,
  24 common scenarios each (panel seed 260930). Among 144 outcomes, 138 encountered
  the placeholder, 143 captured a source, 73 succeeded and 70 failed after source
  capture. Retained full state/action traces and all checkpoint/source hashes.
  First-episode instrumentation matched ordinary rollout for every policy;
  model tensors and checkpoint bytes stayed unchanged. These older checkpoints
  cannot establish the cause of the epoch 1000 regression.
- Two selected shared seed 0 trajectories exactly replayed under gradient collection,
  without an optimizer step. Weighted actor/value gradient-norm ratios were
  12.13 and 30.68. The failed trajectory's terminal value −106.509 is outside the
  conservative final-reward bound [−7.6, 10], giving advantage +105.509 for reward −1.
  Recorded this checkpoint-specific critic error and structural small-episode
  batches/no entropy bonus without claiming they caused seed divergence.
- Files touched: only `AGENTS.md` and the new directory
  `analysis/fc_audit_2026-09-30/`. Its authored files are `.gitignore`, `report.md`,
  `log_probe.py`, `simulator_probe.py`, `policy_probe.py` and `learner_probe.py`.
  Generated files are `log_findings.json`, `simulator_probe.json`,
  `learner_findings.json`, `validation.json`, `artifact_manifest.json`,
  `policy_panel/{panel,summary}.json` and, for shared/banked × seed0/1/2,
  `policy_panel/MODEL_seedN.json` and `policy_panel/MODEL_seedN_traces.jsonl`.
  The manifest lists all 24 other audit artifacts with exact hashes, excluding
  itself and transient Python bytecode caches.
- Validation: targeted existing environment/learning/training/reproduction
  checks passed **40 tests in 9.88 seconds**; deterministic simulator assertions,
  replay/immutability, script compilation, local links and whitespace checks
  passed. Independent read-only reviews verified every one of 144 traces, all
  six checkpoint hashes, reward arithmetic, terminal bound, report statistics
  and causal limits. No runtime code, training configuration, dependency, input
  log, existing checkpoint, prior analysis, cluster job, commit or push changed.
  Recommended a separately versioned opt-in environment correction and frozen
  replay of resynced checkpoints before attributing the late collapse.

## Remaining research work and boundaries

Run the locked research protocol and assess actual learning across independent
seeds. No learned robustness, improved return, semantic roles or superiority to
HetNet has been established. Own sensor state is known locally; fault detection
is not learned. Only PCP sensor loss and episode-level roster changes are
implemented. Gaussian roles/KL, mid-episode membership changes, physical FC
sensor-failure semantics and paired inferential comparison are deferred.

Summary confidence intervals condition on the supplied evaluation panels and
require consistent event-time/victim distributions. They stratify training-source
hash and checkpoint epoch/update, not exact step overshoot. SoftRole evaluations
now record separate training and evaluator source identities; the pilot archives
evaluator source, while standalone evaluation requires retaining its identified
checkout. Keep source and environment fixed while spawned-worker jobs run.
The optional Slurm memory/time requests are unmeasured starting settings.
