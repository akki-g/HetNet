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

### 2026-09-30 — project overview PDF with newly resynced logs

- Created the professional ten-page
  [project overview PDF](analysis/project_overview_2026-09-30/project_overview.pdf).
  Its first three pages cover foundations/HetNet/prior work, the architecture
  overview, and individual components in accessible language. Remaining pages
  explain comparison methods, PP/PCP/FC return/success/length curves, learning
  diagnostics, the separate sensor-loss pilot, research status and traceability.
- Incorporated the user's additional `logs_1/` and `logs_sr/` resync during this
  task. The final snapshot has 27 runs and 37,566 completed epochs, 10,688 beyond
  the earlier same-day resync; SoftRole records 631,310,527 steps and 66,164,807
  episodes. Verified old stdout byte prefixes and archived metric/configuration
  identities. Recorded but excluded trailing in-progress reproduction batches;
  no incomplete JSON lines or completed-epoch metric blocks were found.
- Independently checked 114 input hashes, 317,427 raw-to-CSV numerical fields,
  702 per-run window fields and 351 grouped statistics. All 135 plotted endpoint
  values agree with the latest summaries. Rechecked the existing 80-outcome PCP
  pilot and its two checkpoint hashes, plus saved FC audit counts and source
  identities; these were read-only checks, not fresh policy executions.
- Used current common SoftRole budgets of 37M PP, 20M PCP and 33.5M FC steps.
  PP remains near ceiling. PCP shared/banked lengths are 7.001/8.752 (20.0%
  fewer for shared), narrowing the earlier 26.7% gap. FC common-budget success
  is 63.912%/32.967%; the new tail includes substantial banked seed-1/2
  regressions and broad seed variation. Nine runs reach their configured epoch
  targets; this is not convergence. Kept HetNet Real on an epoch axis because
  stdout sample counters overcount. Explicitly retained pipeline differences,
  FC physical-simulator defects, the newer-tail provenance boundary and the
  absence of learned adaptation evidence.
- Files touched: only `AGENTS.md` and the fresh directory
  `analysis/project_overview_2026-09-30/`. Authored files there are `.gitignore`,
  `README.txt`, `project_overview.tex`, `prepare_report.py`, `build_pdf.py`,
  `refresh_results.py`, `review_results.py`, `make_figures.py` and
  `validate_report.py`. Generated files include `project_overview.pdf`, its
  expanded TeX/text, six table fragments, `latest_snapshot/{summary.json,
  epoch_metrics.csv,provenance.json,comparison_audit.json}`, comparison CSV,
  independent review JSON, evidence/input/figure/validation records, and five
  PDF/PNG figure pairs. `artifact_manifest.json` lists every deliverable with
  exact paths and SHA256 hashes; it excludes itself and transient build,
  bytecode and visual-review files.
- Verified primary-paper links and exact titles, obtained independent
  architecture and numerical reviews, and inspected all ten rendered pages.
  Final PDF checks: exactly ten pages in the requested order, 27 embedded
  fonts, no TeX overflow/underflow or missing-character warnings, and all
  extracted words within page bounds. Python sources compile; input hashes
  and whitespace checks pass. Used isolated Matplotlib 3.10.7/NumPy 1.26.4
  for plots and Tectonic for LaTeX. No runtime code, dependency, original input,
  prior analysis or checkpoint changed; no training, policy evaluation, runtime
  test suite, cluster action, commit or push. Inspected HEAD was `0ee9ceb`.

### 2026-09-30 — paper comparison and independent results/code audit

- Audited the user's Figure 3 reproduction concern, FC validity, SoftRole
  correctness and exact settings/progress of the three current architecture
  suites. Added `analysis/results_audit_2026-09-30/AUDIT.txt`, with a fresh
  27-run inventory and deterministic/frozen-policy evidence. Read the published
  AAMAS paper and supplement; distinguished Figure 3 training curves from
  Table 1 frozen evaluation, and the published three-layer Adam specification
  from the active two-layer RMSprop release.
- New confirmed finding: `hetgat/uavnet.py:get_obs_features` advances 25 entries
  through 29-entry observation cells. Tagged-cell probes find 96/100 misplaced
  slots per PP/PCP sensing agent and 32/36 for FC. Target impulses survive only
  in 4/25 and 2/9 relative cells, respectively; the separate position branch
  provides no bypass. Independently verified the active call chain. The current
  upstream file is byte-identical to local. This is a serious inherited
  information-loss confound; its causal share of the learning gap and the code
  revision used for the paper remain unestablished. Preserved the legacy path.
- Reconfirmed the FC placeholder-front defect: eight of 25 initial fire cells
  create a phantom origin fire; dropping water on it can give team reward +9.8
  while the real fire remains. All six archived SoftRole FC runs match current
  physics files. Final-real-fire termination passes. The defect affects all
  architectures but does not establish why particular seeds regress late.
- Quantified observation aliasing on paired reset seeds 0–999: legacy blindness
  erases 688/1144 (60.14%) otherwise visible PCP sensing-agent target views and
  123/510 (24.12%) FC views; PP has no copy-path difference. Matched actions in
  90 episodes/10,992 transitions preserve positions, rewards, done and info.
- Audited SoftRole core inputs, recurrence, gates, bits, masking, actor/critic
  separation, GAE, clipping and global episode weighting. Found no new core
  implementation defect in these paths. A prespecified diagnostic used latest
  locally archived seed-0 shared/banked checkpoints per task plus initialization,
  12 common nominal scenarios each (panel seed 93001). All 144 episodes/15,746
  transitions reconciled through independent action replay, including physical
  success and per-step/cumulative rewards; checkpoint hashes were unchanged.
  These small unequal-progress panels support real learned simulator behavior,
  not architecture superiority or valid FC physics. Checked 144 archived source
  files across 18 runs against manifests and scoped current diagnostic changes.
- Freshly parsed 37,566 completed epochs; SoftRole totals reconcile to
  631,310,527 joint steps and 66,164,807 episodes. Nine runs reach epoch caps.
  All suites use four collectors, 500-step floor per collector, ten updates per
  epoch, PP/PCP 2000 epochs and horizon 80, FC 1400 and horizon 300. There is no
  configured global training-step cap. Retained true archived legacy counts and
  marked newer printed totals invalid; the stdout count bug does not invalidate
  its separately normalized episode-length metric. No live job status inferred.
- Validation: `.venv/bin/python -m pytest -q` passed **178 tests in 49.14
  seconds**. New deterministic probes, action replay, raw-log reconciliation,
  source/checkpoint hash checks, Python syntax and whitespace checks passed.
  Independent reviewers verified stride retention/call-chain behavior, FC
  mechanics and inventory arithmetic. Test passage is not a bug-free certificate.
- Files touched: this existing `AGENTS.md` and only the fresh
  `analysis/results_audit_2026-09-30/` directory. Authored files there are
  `.gitignore`, `AUDIT.txt`, `legacy/probe_features.py`, `fc/confirm_fc.py`,
  `training/inventory.py`, `softrole/probe.py`, `softrole/observation_probe.py`
  and `validate.py`. Downloaded references are `legacy/upstream_uavnet.py` and
  `paper/{hetnet_aamas2022,supplement}.pdf`; extracted/rendered references are
  `paper/{hetnet_aamas2022,supplement}.txt` and `paper/figure3_page.png`.
  Generated evidence is `legacy/feature_findings.json`, `fc/findings.json`,
  `training/{inventory.json,progress.csv}`, `softrole/{probe_results.json,
  probe_console.jsonl,observation_results.json,observation_console.json}`,
  `validation.json` and `artifact_manifest.json`. The manifest lists exact
  hashes of all other audit files, excluding itself and transient Python caches.
- No runtime or test source edits, optimizer updates, research training,
  dependency changes, input/checkpoint mutation, cluster actions, commit or push.
  Preserved pre-existing `AGENTS.md` changes and the untracked project-overview
  deliverables. Recommended separately versioned opt-in fixes and corrected
  reference retraining before causal architecture or paper-reproduction claims.

### 2026-09-30 — upstream historical training and defect audit

- Traced all 26 reachable commits across five advertised upstream branches,
  including 18 main ancestors, in a fresh isolated mirror. Archived commit/file
  chronology, GitHub metadata, six exact source snapshots, arXiv v1/v2 and the
  January 2022 supplement. Checked public tags/releases/issues/fork metadata
  and later artifact references. The June 2026 PPO artifact link returned 404;
  no inference about its unavailable contents was made. No messages were sent.
- Identified `d57da0717d5564027df7e8ba75614feca8006960` (15 February 2022) as
  the last public preconference snapshot, **not a verified training commit**.
  Public source cannot establish the model-producing checkout, command/runtime,
  checkpoints, exact sample budget or evaluation panel. Earlier arXiv results
  predate the public code and differ from the final AAMAS table.
- Dated the current sensory-stride defect to **10 October 2022, `6b09d36`**,
  after AAMAS. Exact-source probes show February target retention 25/25 PP/PCP
  cells and 9/9 FC, versus October/current 4/25 and 2/9. This corrects any
  inference that the present upstream extractor necessarily trained the paper's
  models. Its quantitative contribution to the reproduction gap remains untested.
- Confirmed PCP/FC observation aliasing in the earliest public wrappers. The
  February FC bundle cannot reset: its single-cell hotspot passes equal bounds
  to `randint`; upstream fixes this only in May 2026. Independently initialized
  downstream historical-function probes confirm the old phantom-origin/+9.8
  reward issue and discovery/reward differences. These are explicitly not full
  historical FC episodes. Eight of 25 initial positions create a new phantom.
- Traced active February/October architecture and optimizer control flow:
  two HetGAT layers and trainer RMSprop, despite the pre-existing supplement's
  three-layer/Adam specification. Recorded exact recurrent/attention/critic
  dimensions, loss/GAE/gradient normalization, commands, horizons and defaults.
  June PP/PCP examples imply a 10M sample floor; October's four collectors imply
  40M for the same 2000 epochs. FC's amended recipe implies 28M over 1400 epochs.
  These later commands are not proof of the paper's settings; no global step
  cap is configured. The June PP command also omits its wrapper's `--nagents 3`.
- Validation: 18 exact-source synthetic forward/loss checks under the current
  pinned runtime yielded 14 passes and four documented PP failures (February
  Real/Binary zero-A paths and both October Binary paths). Passing cases had
  finite clipped gradients, unchanged parameters and no policy optimizer step.
  Instrumented trainer routing confirmed its step denominator without updating
  weights. Independent reviews checked learner/budget and environment/evaluation
  findings; corrected PP channel wording (both wrappers have 29-wide cells)
  and distinguished current upstream defects from local seed/buffer fixes.
  Source hashes, Git identities, report arithmetic, Python syntax and whitespace
  were checked. The previous 178-test result was not rerun as historical proof.
- Files touched: this existing `AGENTS.md` and only the new
  `analysis/upstream_history_2026-09-30/` tree. Authored files are `.gitignore`,
  `AUDIT.txt`, `collect_history.py`, `collect_extra_evidence.py`, `validate.py`,
  `bugs/{FINDINGS.txt,probe_history.py}`, `training/{TRAINING_AUDIT.txt,
  export_sources.py,probe_model.py}`, and `env_eval/{findings.txt,
  probe_historical.py}`. Generated evidence includes root history/file/snapshot/
  artifact-search/validation JSONs, `timeline.csv`, `public_api/`, `papers/`,
  `diffs/`, `snapshots/`, exact source copies and probe JSON/manifests in each
  subaudit. Root `artifact_manifest.json` records every deliverable's exact path
  and SHA256, excluding itself, ignored `history.git/` and transient caches.
- No runtime/test source edit, optimizer update, research training, dependency
  change, existing input/checkpoint mutation, cluster action, commit or push.
  Preserved earlier uncommitted work. Existing SoftRole PP/PCP nominal learning
  remains meaningful, but a lower training mean than the paper's evaluation mean
  is not evidence of superiority. Recommended separate versioned historical-
  code and paper-specification reconstructions and matched frozen evaluation.

### 2026-09-30 — isolated publication-era reconstruction and author search

- Implemented the user's requested separate runnable reconstruction in
  `publication_reconstruction/`. The February15 `d57da07` environment family
  remains the closest public reference, not an identified publication training
  commit. Original runtime, existing launchers, SoftRole, checkpoints and logs
  remain unchanged. Added only a discovery link to root `README.md`.
- Copied 12 historical environment/dependency files and 25 current reproduction
  scaffold files into an isolated runtime. `ORIGINS.json` identifies each exact
  original commit/path/hash and modified hash; `runtime_changes.patch` records
  the seven changed files against those sources. The public two-layer/RMSprop
  learner is retained; this does not implement the unresolved three-layer/Adam
  paper specification. Restored full29-entry sensory stride for both versions.
- Added explicit `historical-2022` and `corrected-v1` environment versions.
  Historical mode retains old observation/FC defects, with only `np.int`,
  equal-hotspot initialization, empty-array containment and debug-print repairs
  needed for execution. Corrected mode additionally copies PCP/FC views, carries
  actual stationary boundary fronts instead of zero placeholders, keeps prior
  positions for out-of-map proposals and uses coordinate-row membership.
  FC individual reward formulas and suppression without discovery are preserved
  and documented as differences from the paper; no speculative team reward was
  silently substituted. Corrected boundary handling is an explicit choice.
- Added a separate module launcher with dated June/October recipes, fresh output
  protection, verified source origins, a copied-source execution archive, package
  records, and source/config/count provenance in checkpoints. June PP explicitly
  passes `--nagents 3`. Optional sample stopping occurs after complete updates,
  with recorded overshoot and actual partial-epoch counts. Fixed copied stdout
  counters to add each batch once. Added worker/process-group cleanup so failures
  and interruptions terminate owned processes and record nonzero exit status.
- Searched135 public repository records across four directly corroborated author
  accounts, two qualified candidate accounts and the lab organization; examined
  relevant predecessor history. Archived32 public resources with hashes. The
  October2021 FireCommander wildfire file is byte-identical to the February2022
  HetNet dependency, including the phantom-front defect. Its original wrapper's
  nondegenerate rectangles explain why HetNet's equal-bound adaptation fails.
  Predecessor discovery-before-suppression supports domain intent but does not
  identify the unpublished AAMAS wrapper. An institutional source link redirects
  to login; no protected contents or author messages were accessed/sent.
- Recomputed10M/20M SoftRole PP/PCP windows across all12 training seeds. Equal-seed
  episode lengths were PP shared4.840→4.784, banked4.874→4.797; PCP shared
  10.692→7.001, banked17.594→8.752. Every seed's length improves; PP gains are
  small while PCP continues improving strongly. This is nominal training evidence,
  not frozen generalization. One versus four collectors imply10M versus40M floors
  but the same20,000 updates at2000epochs. The10M floor establishes neither the
  actual paper budget nor overtraining/optimal stopping. Checked31 input hashes
  and independently reaggregated48 windows; later legacy sample counts remain
  unavailable rather than reconstructed from invalid printed counters.
- Validation: `.venv/bin/python -m pytest -q` passed **220 tests in42.17seconds**,
  including all178 existing tests. New checks cover exact sensory extraction and
  influence, allsix domain/message forward-gradient paths, unchanged existing
  graph outputs, physical completion/rewards, both observation modes, all25 FC
  starts, coordinate collisions, bounded random FC rollouts, recipe budgets,
  source archive integrity, overwrite protection and actual child/grandchild
  cleanup on SIGINT/SIGTERM/output errors. `uv lock --check` and whitespace checks
  passed without dependency changes. Independent implementation review found and
  resolved the inherited interruption/worker-cleanup issue in the copied runtime.
- Validation:18 final tiny training commands, all2updates, covered PP/PCP/FC ×
  Real/Binary × both versions with one collector, plus PPBinary/PCPReal/FCReal ×
  both versions with two collectors. Combined190 steps/48episodes/36updates;
  each saved finite changed weights and optimizer state, correct step/update/
  episode counts and source identities. Thresholds5/9 deliberately stopped
  inside configured epochs. One earlier successful PP smoke is preserved beside
  the final matrix; its inspection harness initially lacked the legacy `utils`
  import path, which was corrected before rerunning the matrix in a fresh folder.
  These are engineering executions, not performance or replication evidence.
- Files touched: `AGENTS.md`, root `README.md`, and the new
  `publication_reconstruction/{__init__.py,__main__.py,README.md,ORIGINS.json,
  runtime/}` tree. Within runtime, modified files are `main.py`,
  `multi_processing.py`, `hetgat/uavnet.py`, `WildFire_Simulate_Original.py`,
  and `envs/ic3net_envs/{predator_prey_env,predator_capture_env,
  fire_commander_env}.py`; every copied file is enumerated in `ORIGINS.json`.
  New tests are `tests/test_publication_{envs,model,launcher,lifecycle}.py`.
  New analysis is `analysis/publication_reconstruction_2026-09-30/`: authored
  `record_sources.py`, `validate_runs.py`, `finalize.py`, author-search collection/
  probe scripts and `FINDINGS.txt`, budget analysis script/`FINDINGS.txt`, fetched
  public records, numerical evidence, runtime diff, logs, validation and manifests.
  Its `artifact_manifest.json` enumerates every new deliverable with exact hashes;
  `smoke_artifact_manifest.json` separately enumerates all initial/final run
  artifacts excluding transient caches. Runs are confined to fresh
  `runs/publication_reconstruction_validation{,_final}_20260930/` directories.
- No full research training, frozen performance evaluation, cluster submission,
  job cancellation, existing input/checkpoint mutation, dependency change,
  commit or push occurred. Earlier uncommitted audit/presentation work was preserved.

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
