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

### 2026-10-01 — self-contained architecture guide and LaTeX PDF

- Added `docs/SOFTROLE_ARCHITECTURE_GUIDE.md` and its 19-page LaTeX-rendered
  PDF. The guide explains the current deterministic SoftRole model from first
  principles, with 46 displayed equation blocks, a notation glossary, worked
  examples, default tensor dimensions, resource costs and code/test references.
  Coverage includes observations/capabilities, expanded LSTM memory equations,
  affine expert mixtures, two binary broadcasts, straight-through bias,
  receiver-gated attention and null, actions, centralized value, domain reward,
  team GAE, episode loss, global gradient aggregation and RMSprop.
- Added an editable vector architecture diagram with an overall actor/critic
  view and communication inset. The PDF contains two landscape diagram pages,
  selectable vector text/paths, clickable contents and 26 bookmarks. Included
  the generated LaTeX and its two vector figure dependencies for direct rebuilds.
  Preserved the distinction between task cost, training loss and resource cost;
  documented retained FC physics defects and approximation/generalization limits.
- Independent read-only reviews checked model equations, dimensions, memory,
  conditional randomness, parameter counts, simulator rewards, GAE arithmetic,
  loss normalization and optimizer semantics. Corrected nonexistent simulator
  method references, aligned diagram indices, and clarified boundary-front
  rather than generic stationary-front behavior in the inherited FC defect.
- Files added: `docs/SOFTROLE_ARCHITECTURE_GUIDE.{md,tex,pdf}`,
  `docs/SOFTROLE_GUIDE_BUILD.txt`, `docs/softrole_architecture.{svg,pdf}`,
  `docs/softrole_architecture_overview.pdf`,
  `docs/softrole_architecture_communication.pdf`,
  `docs/build_softrole_architecture.py`, `docs/export_softrole_diagram.py`, and
  `docs/build_softrole_guide.py`. Existing files touched: root `README.md`
  (discovery links), `.gitignore` (only these guide artifacts are included),
  and this existing uppercase `AGENTS.md` (work record).
- Validation: rendered/reviewed all PDF pages and checked the final dimension
  table and diagram pages at full size. Confirmed all 36 local Markdown link
  targets exist, checked five primary-source links, all three build scripts parse,
  46 displayed math blocks
  render, and the guide has zero raster images. Final TeX log has no overfull,
  underfull or missing-character messages; XeTeX emits input-version notices
  for the 1.7 figure PDFs, while the final output explicitly uses PDF 1.7.
  Vector exports reproduce byte-for-byte with pinned tooling. Checked whitespace.
- Build tools are isolated: Pandoc via `pypandoc-binary==1.15`, Tectonic,
  `svglib==2.2.0`, `reportlab==5.0.1`, and `pymupdf==1.28.2`. An initial raster
  proof used isolated resvg; its two superseded PNG panels were removed from
  `docs/`. Build logs, scratch conversions, validation JSON and rendered previews
  are under `.tools/softrole_guide_build/`; its `artifact_manifest.json` lists
  exact scratch paths/hashes. Temporary SVG previews also used
  `/tmp/softrole_architecture.svg.png` and `/tmp/softrole_architecture_full.png`.
- No runtime/test source or training dependency changed. No training, policy
  evaluation, cluster action, checkpoint/input mutation, commit or push was
  performed for this documentation task. Existing test results remain dated
  historical records; the test suite was not rerun for these documentation edits.

### 2026-10-02 — latest training audit and three week experiment priorities

- Parsed all 27 fresh primary stdout files and their stderr: 45,259 completed
  epochs, 7,693 beyond the latest project-overview snapshot. Verified historical
  stdout byte prefixes, unchanged archived configuration/metric hashes and
  archived metric agreement. SoftRole counts reconcile to 688,028,940 joint
  steps and 74,972,901 episodes, with zero recorded failure exposure. Inventoried
  and hashed 356 log/metric-ledger files; archival copies, timing, setup and
  validation logs are not additional independent research seeds.
- Confirmed 23/27 runs reached epoch targets. HetNet PP seeds 0/1/2 stop locally
  at 910/866/1228 of 2000; no scheduler status inferred. SoftRole PCP shared
  seed 0 explicitly timed out at epoch 1655/2000, 33,669,601 steps. All other
  primary PCP/FC runs reached their caps; completion is not convergence.
- At common SoftRole budgets, episode-weighted within seed and equally weighted
  across seeds, PCP 33.5M shared/banked lengths are 5.666/6.657 (14.88% fewer
  for shared), both near-ceiling success. FC 34.5M successes are 65.23%/34.32%,
  with large seed variation and late regressions. Preserved FC physics and
  legacy observation confounds; no gate superiority or adaptation established.
- Audited readiness of publication reconstruction, sensor failure and frozen
  transfer. Reconstruction has smoke runs only and needs an evaluator using its
  restored model/environment, not root evaluate-hetnet. Final local checkpoints
  remain stale. PCP failure mechanics, shams and interventions are implemented;
  full failure research runs are absent. Identified required matched-mixture
  no-failure controls, informative event exposure, paired analysis and explicit
  budget/selection-rule grouping beyond current epoch/update summary strata.
- Added a detailed report and proposed Oct 2–23 priorities. Recommended existing
  frozen PCP transfer, six corrected PCP Real/Binary reference runs and twelve
  matched mixture/failure runs if making a failure-training claim. Twenty-million
  step new-run budgets and 30M existing-policy transfer selection are proposals,
  not adopted or executed protocols. Historical replay and corrected FC are
  separately costed extensions. Recorded multi-day legacy runtime evidence and
  allocation/time-limit dependencies rather than promising quick completion.
- Files touched: only this existing uppercase `AGENTS.md` and new
  `analysis/training_2026-10-02/`. Authored files: `.gitignore`, `analyze.py`,
  `plot.py`, `report.md`. Generated files: `summary.json`, `provenance.json`,
  `log_inventory.json`, `epoch_metrics.csv`, `run_summary.csv`,
  `softrole_training.png`, `hetnet_training.png`, `pcp_fc_diagnostics.png`,
  `training_plots.pdf`, `validation.json`, `artifact_manifest.json`. The manifest
  enumerates all other deliverables in that directory with SHA256 hashes.
- Validation: independent raw checks verified 45,259 CSV rows/759,936 field
  comparisons, 1,836 run-window fields, 297 grouped statistics, 386 input hashes,
  and all 356 log-inventory entries. Corrected shallow checkpoint enumeration
  to include 434 nested checkpoint/sidecar files (217 actual checkpoints); this
  was a filename inventory, not a fresh tensor audit. Independently reviewed
  report arithmetic, protocol scope and reconstruction caveats; all eight local
  report links resolve. Viewed all three plot pages, checked Python syntax,
  input immutability and whitespace. Matplotlib 3.10.7/NumPy 1.26.4 ran isolated
  through uv; no project dependency changed or runtime test suite reran.
- No policy execution, training, job submission/cancellation, model/environment
  change, input/checkpoint mutation, commit or push occurred. Pre-existing
  README, ignore-file, documentation and presentation changes were preserved.

### 2026-10-02 — training time, M5 Max measurements and budget amendment

- Recomputed timing for all 27 primary runs and same-work PCP comparisons.
  Shared seed 0 timed out after 47.964 recorded epoch hours at epoch 1655;
  seeds 1/2 reached that point in 20.220/20.402 hours, with nearly identical
  steps and more episodes. Near the stop, epoch time is 133.92 versus
  45.03/45.46 seconds. Same source/configuration except seed; no learning or
  numerical failure explains the work-count comparison. Documented abrupt
  banked timing transitions and missing scheduler/CPU/phase evidence. Per-episode
  serial open/write/close is a profiling candidate, not a proved timeout cause.
- Independently rechecked budget provenance: June/October public commands imply
  10M/40M sample floors but the same 20,000 updates; neither is a verified paper
  budget. Nominal SoftRole 20M reached in about 12–24 recorded hours. Clarified
  fixed-budget validity versus convergence, proposed 10M/20M reporting stages,
  and nine/fifteen/eighteen new-run scopes conditional on intended claims.
- Benchmarked this actual M5 Max CPU (18 cores, 36 GiB; Torch 2.2.1, NumPy1.26.4,
  Apple Accelerate confirmed from build) with unchanged numerical/runtime code.
  A preliminary frozen-gradient probe and a 120-round sustained probe preserve
  weights; sustained collection did 255,859 steps in 92.69 seconds with four
  collectors/500-step floors and real gradient IPC. CPU float64-to-MPS probe
  explicitly fails; GPU porting/float32 are not drop-in protocol-preserving steps.
- Ran two bounded actual SoftRole resumes from archived PCP epoch200, three
  epochs each, same four collectors/batch/horizon/optimizer. Shared collected
  63,013 steps in 22.50 training-function seconds, banked 63,736 in 25.79.
  All six epochs exactly match historical Stokes steps, episodes, success and
  returns; neural diagnostic/loss differences are at most about 1.06e-6.
  Historical epoch-time ratios are 8.31x/9.02x, not isolated chip-only effects.
  Input checkpoints are unchanged; fresh model/optimizer tensors are finite and
  parameters changed. No cross-platform bitwise or long-horizon equality claim.
- Ran three initial corrected-v1 PCP Real reconstruction epochs with the October
  four-collector recipe: 64,709 steps/924 episodes/30 updates, 121.83 epoch
  seconds and 139.25 launcher seconds. Checked finite tensors, changed signature,
  true counts and archived-source identities. Across all actual timing runs:
  nine epochs, 90 optimizer updates, 191,458 new steps, 4,041 episodes. These are
  engineering timing measurements, not independent research runs or task results.
- Measured byte-identical local logging for 3,500 episode records: 0.0961 seconds
  with 3,500 opens versus 0.0190 with ten grouped opens. This 5.07x logging-only
  ratio saves little locally and does not identify remote filesystem latency.
  No log batching, device migration or training-runtime patch was applied.
- Added the timing report/figure and explicitly labeled short-rate projections:
  20M about 2.0–2.25 hours for SoftRole and 10.5–12 hours for corrected Real on
  this Mac if measured early throughput persists. Mature policies, failure
  mixtures, thermal/background load and concurrent jobs remain unmeasured.
- Files touched: existing `AGENTS.md`; amended
  `analysis/training_2026-10-02/{report.md,validation.json,artifact_manifest.json}`
  for the linked timing/scope correction; and new `analysis/timing_2026-10-02/`.
  Authored files: `.gitignore`, `report.md`, `analyze_timing.py`, `plot_timing.py`,
  `benchmark.py`, `benchmark_io.py`, `benchmark_training.py`,
  `benchmark_reconstruction.py` (including rerunnable audit-only postprocessing)
  and `compare_migration.py`.
  Generated principal records: timing CSV/JSON/console, collector probe JSON/logs,
  training/reconstruction timing JSON/logs, migration comparison, local I/O/build
  records, timing PNG/PDF and validation/manifest JSON. Fresh actual-run trees
  are `local_training/{shared,banked}/` and `local_reconstruction_pcp_real/`;
  their full source/configuration/metric/checkpoint paths and hashes are recorded
  individually in the new artifact manifest (excluding transient bytecode).
- Validation: independent raw-log timing/budget reaggregation, benchmark count/
  throughput/hash checks, matched migration aggregates, reconstruction source
  review and interpretation review passed. Checked scripts, local links,
  input immutability and whitespace; viewed the timing plot. Plotting used
  isolated Matplotlib3.10.7/NumPy1.26.4. No runtime test suite reran because runtime
  sources were unchanged. No dependency change, production input/checkpoint edit,
  full research sweep, cluster action, commit or push occurred.

### 2026-10-02 — reward comparability and bank size analysis

- Added [the reward and bank-size report](analysis/reward_banks_2026-10-02/report.md)
  for the user's questions about lower PP/PCP steps but larger negative reward,
  reward improvement without structural changes, and two/three-expert banks.
  Verified the official paper's Table 1 and archived supplementary reward rules
  against the current simulator, rollout, learner and released evaluator.
- Established explicit team-sum versus per-agent reporting. The paper does not
  recoverably specify its scalar agent reduction; mean-agent reporting is a
  plausible interpretation, not a confirmed correction. Its reward/step pairs
  violate the current team-sum bound R <= -0.05(T-1), so direct team-return
  comparison is invalid. Distinguished frozen paper trials from training windows.
- Reaggregated all twelve PP/PCP SoftRole raw logs at the prior common sample
  budgets and reproduced the earlier audited means. Preserved class returns:
  89.89% of the matched PCP banked/shared reward gap comes from the A agent.
  Derived successful-episode sum-completion versus max-completion identities
  and separate navigation/capture-delay accounting without assigning a cause
  unsupported by current aggregate logs.
- Enumerated all 303,600 ordered distinct initial layouts for each domain's
  geometric oracle. Expected optimistic team returns are -0.35 PP/-0.40 PCP;
  expected episode lengths are 4.676759/5.085059. Independent review repeated
  the enumeration and executed 64 short deterministic environment-only oracle
  episodes, plus six separate reward-tradeoff/horizon probes. These are physics
  checks, not learned-policy evaluations or additional research training.
- Counted actual PCP models: shared 171,087 parameters; banked K=2/3/4 has
  190,233/207,202/224,171 including critics. Derived local mixture/gate gradients,
  explained conditional initialization-scale effects and diagnosed the limits
  of entropy evidence. Recommended a matched three-seed K=2 study before optional
  K=3, retaining reconstruction/failure priorities. LR, detach-gap and shared
  initialization alternatives are hypotheses, not implemented improvements.
- Files touched: existing uppercase `AGENTS.md`; new
  `analysis/reward_banks_2026-10-02/{.gitignore,analyze.py,report.md}`.
  Generated files in that directory: `results.json`, `results_repeat.json`,
  `repeat_console.json`, `k2_dry_run.json`, `k3_dry_run.json`, `validation.json`
  and `artifact_manifest.json`. The manifest records every other file and hash.
- Validation: repeated analysis output is identical; all 24 identified input
  hashes match; local links, Python syntax and whitespace checked. Generic CLI
  K=2/K=3 dry runs both preserve the remaining PCP protocol and create no run
  directory. Independent numerical and mathematical reviews passed. No original
  run/checkpoint mutation, production source edit, dependency change, training,
  learned-policy evaluation, full test-suite rerun, cluster action, commit or
  push occurred. All pre-existing changes were preserved.

### 2026-10-02 — HetNet reward reporting correction

- Rechecked the complete released evaluation pipeline for the user's follow-up.
  Found `print_plot_eval.py`, previously missed after reading `eval_trainer.py`.
  It pools per-episode reward vectors and calls `np.mean(rewards)` without an
  axis, explicitly averaging across episodes and agents. The current official
  upstream script has the same operation. SoftRole's epoch summarizer averages
  episode team sums, so its scalar is exactly N times that public reporting
  scalar for identical fixed-N episode data. Exact historical Table 1 source
  provenance remains unverified; the current public aggregation is now confirmed.
- Corrected `analysis/reward_banks_2026-10-02/report.md` to replace the earlier
  missing-reduction statement with the recovered script and equations. Added
  `reporting_correction.json` with code hashes and arithmetic validation;
  amended `validation.json` and regenerated `artifact_manifest.json` in that
  analysis directory. Existing raw inputs, numerical analysis outputs and
  runtime sources are unchanged. This existing `AGENTS.md` records the correction.
- Checked the factor-of-three identity on synthetic episode vectors and the
  previously audited PP/PCP means; checked links and whitespace. No training,
  policy evaluation, dependency change, cluster action, commit or push occurred.

### 2026-10-02 — epoch plots, comparable reward reporting and buffered logs

- Clarified that 20M was a proposed shorter common-budget experiment, not an
  authorized change to current recipes. PP/PCP remain 2,000 epochs, ten updates
  per epoch, four collectors and a 500-step per-collector floor (40M minimum
  joint steps before complete-episode overshoot); FC remains 1,400 epochs/28M.
  No budget, architecture, optimizer, reward or learning objective was changed.
  Supplement section 2.1 describes three layers with four heads, Adam 1e-3;
  the released/reconstruction two-layer RMSprop path remains distinct.
- Added `mean_agent_return` to SoftRole episode/epoch/evaluation records and
  frozen HetNet reports, plus seed-level summaries compatible with archived
  evaluation reports. Each episode contributes its team return divided by its
  own team size, then episodes are averaged within independent training seeds.
  Team-sum reward remains the learning objective. Rechecked the official public
  `print_plot_eval.py` mean-agent reduction; exact historical Table 1 provenance
  is still not established merely by inspecting today's public script.
- Changed SoftRole episode persistence to one buffered file open per optimizer
  update, retaining one complete JSON object per episode in the original order.
  Optional `--episode-log stdout` prints one tagged JSON object per update with
  the full episode array and one explicit flush. This runtime option is outside
  scientific/checkpoint configuration and may change on resume. Default is file;
  epoch/update ledgers and all counters remain available. Buffering avoids a
  whole-run in-memory ledger, but stdout is still I/O and no training speedup
  from this change has been measured. Historical stdout parsers need to filter
  the new record type; their dated source is preserved, and `metrics.jsonl`
  remains the clean epoch source. Neither mode provides atomic crash durability.
- Added single-checkpoint SoftRole evaluation wrappers for direct CPU execution
  and Stokes Slurm so checkpoints can stay on Stokes and result JSON can sync
  first. One CPU/4 GB/four hours are unmeasured resource defaults. This does not
  implement frozen evaluation for publication-reconstruction checkpoints.
- Replotted the audited October 2 snapshot: 27 runs and 45,259 completed epochs.
  PP/PCP/FC panels show success, mean-agent return and episode length; additional
  panels show gate/value/null diagnostics and recorded training time. Every x
  axis is completed epochs. SoftRole windows weight episodes; legacy windows
  average the available rounded epoch means. Thick curves give equal-seed means
  only while all three seeds exist; individual tails remain visible. Captions
  retain sensory/FC defects and training-versus-frozen-evaluation limitations.
  Original sample-axis plots and raw input CSV remain unchanged.
- Files touched (runtime/launchers): `softrole/train.py`, `softrole/rollout.py`,
  `softrole/__main__.py`, `softrole/evaluate.py`, `softrole/hetnet.py`,
  `softrole/report.py`, `scripts/softrole_failure.sh`, new
  `scripts/softrole_evaluate.sh` and `slurm/softrole_evaluate.sbatch`.
- Files touched (tests/docs): `tests/test_softrole_training.py`,
  `tests/test_softrole_evaluation.py`, `tests/test_softrole_hetnet.py`,
  `tests/test_softrole_report.py`, `tests/test_softrole_failure_launcher.py`,
  `README.md`, and this existing uppercase `AGENTS.md`. Added
  `analysis/training_epochs_2026-10-02/{.gitignore,plot.py}`. Generated there:
  `pp_training_epochs`, `pcp_training_epochs`, `fc_training_epochs`,
  `diagnostics_epochs`, `training_time_epochs` (each PNG/SVG),
  `plot_manifest.json`, `validation.json`, and `artifact_manifest.json`.
- Validation: full `.venv/bin/python -m pytest -q` passed **233 tests in 24.17s**.
  Logging tests include actual tiny one/two-collector runs with exact matching
  model/RMSprop/counts between destinations, complete episode reconstruction,
  unchanged update ledgers, one file open per update, and partial-epoch resume
  across logging modes. Variable-roster tests check the intended estimand, and
  evaluator/report tests check new fields and old-file compatibility. Shell
  syntax passed for the six SoftRole scripts/Slurm wrappers; agent checks also
  exercised direct/mocked-Slurm argument forwarding, spaces, working directory,
  runtime variables and missing-checkpoint handling without real evaluation.
- Validation: all five final PNGs visually inspected; corrected title/subtitle
  overlap before delivery. Plot input and all ten figure hashes verified.
  Independent read-only review checked all CSV rows, positive weights, ten
  updates per epoch, return arithmetic, reporting/checkpoint compatibility and
  learner invariance. Python syntax and whitespace passed for files changed by
  this task; unrelated pre-existing `.gitignore` trailing blank lines were
  preserved. Plotting used isolated NumPy 1.26.4/Matplotlib 3.10.7. No dependency
  edit, production checkpoint/input mutation, research training, cluster job,
  commit or push occurred. Actual training in this task was limited to tests.

### 2026-10-02 — traceable Stokes frozen panel and readiness audit

- User confirmed the Stokes primary run root is `runs/softrole_primary`.
  Audited the actual evaluation/training launchers, restored-runtime imports,
  optimizer steps, sensor masking, mixed-roster loss assumptions, FC corrections,
  checkpoint metadata and reporting strata. No live Stokes connection was made.
  Added `analysis/slurm_readiness_2026-10-02/evidence.json`: hashes and line-numbered
  excerpts for 20 sources, independently derived checkpoint expectations from
  the audited CSV, explicit evaluation panels and remaining implementation gaps.
- Added `scripts/prepare_pcp_frozen.py`. Preparation never submits. It selects
  the first available saved checkpoint reaching 30M true steps for each of six
  nominal PCP shared/banked policies, validates contiguous/reconciled ledgers,
  exact checkpoint progress, fixed native configuration, common training source,
  strict model loading, finite tensors and stored/sidecar tensor signatures,
  then rechecks all checkpoint/ledger/sidecar hashes before creating output.
  Existing destinations and incomplete/inconsistent panels are rejected.
- Preparation saves the selection/configuration manifest, explicit scenario
  files, source archive, helper source and artifact hashes. Its separate
  `submit.sh` uses the existing frozen evaluator wrapper for 18 jobs, records
  job IDs and rejects repeated submission-script invocation. On partial sbatch
  failure, recorded IDs identify work already submitted; recovery is manual.
  Paths are shell-quoted. The live Stokes checkout must remain fixed; archiving
  source does not change the execution checkout. `submitted: false` records
  preparation status only, not live scheduler state.
- Proposed submission panel: shared/banked seeds 0/1/2; nominal evaluation seed
  2700 with 500 scenarios per composition `(2,1)`, `(1,2)`, `(2,2)`, `(3,1)`,
  `(3,2)`; native failure/sham seed 2701 with 100 common scenarios, uniform
  event 10–30 and failure probability one. Sensor jobs retain traces for prefix
  checks. Six nominal jobs plus twelve sensor-condition jobs yield 16,200 policy
  episodes and zero training jobs. Four-hour/one-CPU/4-GB requests are unmeasured
  allocation settings. The native sensor panel is an exposure diagnostic.
- The audited logs predict epoch 1500/15,000 updates for all six first-saved
  checkpoints above 30M. Shared steps by seed are 30,552,489 / 30,508,112 /
  30,536,475; banked 30,728,678 / 30,683,393 / 30,635,286. These are expectations
  from copied logs, not a late-checkpoint tensor audit. The helper validates the
  actual Stokes files. Selection is independent of new evaluation outcomes and
  does not change any full training schedule. Actual sample differences remain.
- Readiness findings: original-model health-input absence does not mathematically
  prevent sensor masking; the current explicit rejection is implementation scope.
  A reconstruction evaluator must select the restored model/environment, retain
  nested provenance/counts and isolate colliding import names. HetNet masking
  must preserve physical class, positions, actions, recurrence and communication,
  and share sham schedules/diagnostics. Supplement alignment needs an opt-in
  third layer and the active trainer Adam path, not the unused policy Adam.
  Matching per-episode team mixtures additionally requires roster-aware class
  advantages/loss bookkeeping. Fixed-team failure training is a smaller distinct
  study. Corrected FC still needs SoftRole adapter/version integration and
  cross-method fixed-action replay; existing root FC remains unchanged.
- Readiness findings: current report strata include checkpoint epoch/update but
  lack intended failure-window/victim-protocol identity. Keep timing studies
  separate; add explicit protocol/budget identities and paired seed-level
  analysis before broad pooled claims. The usual SoftRole training shell
  launcher injects `--task`, so continuation must currently use the direct
  `python -m softrole train --resume ...` CLI without `--task`/`--study`.
- Files touched: new `scripts/prepare_pcp_frozen.py` and
  `tests/test_softrole_prepare_frozen.py`; appended `README.md` and this existing
  uppercase `AGENTS.md`; new `analysis/slurm_readiness_2026-10-02/evidence.json`,
  `validation.json`, `artifact_manifest.json`, and `validation_prepare_4m/`.
  The recursive artifact manifest lists every generated file, including all
  archived sources, and excludes itself. All pre-existing changes were preserved.
- Validation: 11 focused tests passed in 2.15 seconds, including preparation
  failure before output, threshold selection, ledger/config/signature rejection,
  explicit common panels, quoted paths, mocked 18-job submission/job-ID recording,
  repeated-invocation refusal and shell syntax. Full suite: **244 passed in
  26.94 seconds**. Independent review verified panel arithmetic and generated
  shell syntax. No real sbatch command was executed.
- Validation: ran actual preparation on all six local archived checkpoints with
  the explicitly validation-only `--min-steps 4000000`; all selected epoch 200
  at 4.255–4.279M steps. Verified 18 input hashes and all generated panel/script
  hashes; no policy execution, result directory or job-ID file was produced.
  Default 30M preparation correctly failed against the early local archive and
  left no output directory. Checked cited source hashes, source/README links,
  Python/shell syntax and scoped whitespace. No production input mutation,
  model/optimizer/budget change, dependency edit, research training, cluster
  submission, commit or push occurred; tiny training was confined to tests.

### 2026-10-02 — proper Slurm array file for the pending frozen panel

- Changed preparation to write `submit.sbatch`, a complete Slurm batch file
  containing all 18 explicit evaluation commands. Its array is `0-17%3`, with
  one CPU/4 GB/four hours per task, the existing account/partition/module setup,
  `srun` execution, and separate `logs_sr/pcp-frozen-%A_%a.out`/`.err` files.
  Each shared/banked seed maps to nominal, failure and sham consecutively.
  The existing checkpoint rule, common scenarios and scientific configuration
  are unchanged. These pending jobs evaluate frozen policies, not training.
- Kept `submit.sh` as an optional convenience wrapper: it now makes one sbatch
  array submission and records the 18 array-task IDs, handling optional cluster
  suffixes. README now gives direct `sbatch .../submit.sbatch` instructions and
  a fresh `_array` output path. Use one route once. Previously prepared plans
  remain untouched; rerun preparation in a fresh directory to obtain the new
  batch file. No already queued cluster job was queried or modified.
- Files touched: `scripts/prepare_pcp_frozen.py`,
  `tests/test_softrole_prepare_frozen.py`, `README.md`, and this existing
  uppercase `AGENTS.md`. New generated evidence is confined to
  `analysis/slurm_array_2026-10-02/`: `validation.json`,
  `artifact_manifest.json`, and `validation_prepare_4m/`, including its source
  archive, manifest, scenario panels, `submit.sh` and `submit.sbatch`. The
  recursive artifact manifest lists every generated file except itself.
- Validation: 12 focused tests passed in 3.39 seconds. Checks cover Slurm
  directives, shell syntax, all 18 task branches via mocked module/srun/Python,
  checkpoint/output/scenario argument quoting, working directory and thread
  environment, failure/sham options, invalid/missing array IDs, a single mocked
  sbatch call, all 18 recorded array-task IDs and repeated-wrapper rejection.
  Actual preparation against six real archived PCP checkpoints at a validation-
  only 4M threshold succeeded; generated scripts passed `bash -n`, and selected
  inputs/generated artifacts were hash-checked. This did not execute policies.
  Scoped whitespace checks passed. No full learner suite was repeated for this
  launcher-only change; the prior 244-test result remains the preceding record.
- No model/optimizer/budget/environment edits, production checkpoint/log
  mutation, dependency change, training, evaluation, real sbatch submission,
  commit or push occurred. All pre-existing changes were preserved.

### 2026-10-02 — limit OpenBLAS threads before frozen-panel preparation

- The supplied Stokes terminal output reports OpenBLAS attempting 64 threads,
  `pthread_create failed`, and `RLIMIT_NPROC 100 current, 105 max`. The helper
  imported Torch before applying its later Torch thread limit, leaving numerical
  library startup unconstrained. The pasted output has no final preparation
  result or exit status, so it does not establish whether preparation completed.
  The separate Gym maintenance warning does not establish a preparation failure.
- Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
  `NUMEXPR_NUM_THREADS` to one before numerical imports. Apply the same forced
  settings in the generated Slurm array, standalone evaluation batch file and
  shell launcher, overriding inherited values such as 64. README includes the
  pre-command exports, which also work with older copies before synchronization.
  No dependency, policy, scenario, checkpoint-selection or budget change.
- Files touched: `scripts/prepare_pcp_frozen.py`,
  `scripts/softrole_evaluate.sh`, `slurm/softrole_evaluate.sbatch`,
  `tests/test_softrole_prepare_frozen.py`, `README.md`, and this existing
  uppercase `AGENTS.md`. Existing prepared plans and archived evidence remain
  untouched; newly prepared plans contain the updated Slurm exports.
- Validation: **16 focused tests passed in 3.82 seconds**. Fresh subprocesses
  checked limits at the first numerical import with absent and inherited-64
  settings; actual CLI help loaded successfully with inherited-64 settings.
  All 18 generated array routes checked limits before mocked srun execution and
  at the mock Python entrypoint. Both standalone launchers also overrode 64.
  Generated/standalone shell syntax and `git diff --check` passed. No Stokes
  process-limit reproduction, runtime speed measurement, full-suite rerun,
  policy execution, training, real Slurm submission, commit or push occurred.

### 2026-10-02 — run frozen-panel preparation through Slurm

- Added `slurm/softrole_prepare_frozen.sbatch` so checkpoint loading, validation
  and panel/source generation can run on a compute node rather than the login
  node. It uses the existing account/partition/Anaconda module, one CPU, 4 GB
  and a provisional 30-minute limit, changes to `SLURM_SUBMIT_DIR`, and forces
  all four numerical-library thread limits to one before `srun` starts Python.
  It forwards arguments directly to the existing helper and preserves its exit
  status. It neither runs policies nor submits the generated evaluation array.
- README now gives two explicit submission steps: create `logs_sr` and submit
  preparation with a fresh output path, then submit the generated `submit.sbatch`
  only after successful preparation with `jobs_prepared: 18`. This avoids Python
  work on the login node and avoids assuming asynchronous preparation finished.
  Checkpoint selection, evaluation panels, budgets and training remain unchanged.
- Files touched for this request: new `slurm/softrole_prepare_frozen.sbatch`,
  `tests/test_softrole_prepare_frozen.py`, `README.md`, and this existing
  uppercase `AGENTS.md`. Earlier thread-limit changes were preserved.
- Validation: **17 focused tests passed in 3.69 seconds**. The added batch route
  was exercised with mocked module/srun/Python, inherited thread counts of 64,
  paths containing spaces/shell metacharacters and explicit CLI overrides.
  Checks verified thread limits before srun/Python, exact forwarded arguments
  and repository working directory; the 18 evaluation-array routes still pass.
  `bash -n` and `git diff --check` passed. No live scheduler validation, real
  submission, training, policy evaluation, dependency edit, commit or push.

## Remaining research work and boundaries

Run the locked research protocol and assess actual learning across independent
seeds. No learned robustness, improved return, semantic roles or superiority to
HetNet has been established. Own sensor state is known locally; fault detection
is not learned. Only PCP sensor loss and episode-level roster changes are
implemented. Gaussian roles/KL, mid-episode membership changes, physical FC
sensor-failure semantics, cross-model paired inference and gate difference-in-differences are deferred. Within-policy failure-minus-sham seed summaries are now implemented (October 3 record below).

Summary confidence intervals condition on the supplied evaluation panels and
require consistent event-time/victim distributions. Legacy reports stratify training-source
hash and checkpoint epoch/update, not exact step overshoot. Explicit validated
evaluation protocols instead group by declared budget/panel and retain actual progress. SoftRole evaluations
now record separate training and evaluator source identities; the pilot archives
evaluator source, while standalone evaluation requires retaining its identified
checkout. Keep source and environment fixed while spawned-worker jobs run.
The optional Slurm memory/time requests are unmeasured starting settings.


### 2026-10-03 — complete reconstruction training and frozen-evaluation workflow

- Implemented the user-approved 12-run nominal first wave: PP Real seeds0–2 at40M,
  PCP Real seeds0–2 at40M, FC Real seeds0–2 at28M, and PCP Binary-16 seeds0–2
  at40M. `STUDY.json` and the resolved study artifact fix corrected-v1, October
  recipes, four collectors,500-step floors,10 updates/epoch, horizons80/300,
  milestones10/20/30/40M or10/20/28M, and epoch caps2000/1400. No research
  array or cluster job was submitted; all actual training here was bounded validation.
- Added explicit public-code-v1 (compatibility default) and supplement-v1 model
  specifications. Supplement mode has three four-head layers, two64-wide hidden
  outputs, final head averaging and the active trainer's Adam1e-3. The inactive
  policy-local optimizer never steps. Public GAE, padding normalization, local
  clipping and global step denominator remain unchanged. The scientific claim is
  supplement-aligned architecture/optimizer with public learner/corrected environment,
  not an exact publication-producing checkout. Public Real parameter dtype
  conventions remain mixed even with float64 inputs/defaults; evaluation preserves them.
- Added SeedSequence initialization/collector/library namespaces for supplement
  mode, complete-update recovery of parent and worker RNG/optimizer/model/counts,
  partial epoch statistics and timing, atomic checkpoints with absolute update
  identities, threshold milestones, and graceful46-hour pause under provisional
 48-hour Slurm allocation. Continuing uses a fresh immutable segment and inherited
  scientific/source configuration. Read-only lineage helpers exclude abandoned
  parent suffixes, including interrupted trailing ledgers. Same-update milestone
  crossings are recorded together before any checkpoint is written.
- Buffered per-episode evidence once per update (file or tagged stdout), while
  retaining epoch/update metrics and per-agent/team rewards. Added resource/timing
  records and a100-update compute-node preflight for each of four workloads.
  The preflight validates optimizer state and executes one native frozen checkpoint
  probe. Update-work and end-to-end segment projections are separate measurements;
  no convergence, Mac/Stokes speed ratio or resource sufficiency is inferred locally.
- Added reconstruction-specific isolated frozen evaluation with strict archived
  actor/environment imports, captured evaluator/helper execution and separately
  published replayable source archives. Covers native PP/PCP/FC, PCP singleton and
  changed compositions, PCP persistent victim-only sensory loss and matched shams,
  both model specifications and Real/Binary tensors. Unsupported events/overrides
  fail explicitly. Reports identify all versions, sources, checkpoint/panel hashes,
  actual progress, pre-event diagnostics and censoring. HetNet receives no health input.
- Preparation locks the existing PCP30M panels without changing their bytes;
  native PP40M and FC28M panels use500 scenarios with seeds2702/2703. It generates
 24 frozen-evaluation jobs after validating all12 policies and first-saved selection.
  Continuation maps are explicit. Reporting adds declared-budget/protocol strata,
  preserves legacy grouping, records actual counts, and computes full-panel paired
  failure-minus-sham within seed before equal-seed aggregation. Small-seed bootstrap
  limitations remain explicit. Corrected FC is separated from old SoftRole FC.
- Runtime provenance refresh preserves September's audit and original per-file
  origins, marks the new recovery helper as a local addition, and writes fresh
  October audit snapshots. Training executes its copied source; frozen evaluator
  archives are independently replayable and reject changed helper bytes.
- Repository files changed (including new files), grouped by purpose:
  - Models: `publication_reconstruction/runtime/hetgat/uavnet.py`,
    `publication_reconstruction/runtime/hetgat/policy.py`.
  - Learner lifecycle/evidence: `publication_reconstruction/runtime/main.py`,
    `publication_reconstruction/runtime/trainer.py`,
    `publication_reconstruction/runtime/multi_processing.py`,
    `publication_reconstruction/runtime/hetnet_ext/recording.py`,
    `publication_reconstruction/runtime/hetnet_ext/recovery.py` (new).
  - CLI/provenance/study: `publication_reconstruction/__init__.py`,
    `publication_reconstruction/__main__.py`, `publication_reconstruction/ORIGINS.json`,
    `publication_reconstruction/artifacts.py` (new), `publication_reconstruction/study.py`
    (new), `publication_reconstruction/STUDY.json` (new), and
    `scripts/record_publication_sources.py` (new).
  - Evaluation/reporting: `publication_reconstruction/evaluation.py` (new),
    `publication_reconstruction/evaluation_worker.py` (new), `softrole/evaluate.py`,
    `softrole/report.py`, `softrole/__main__.py`. SoftRole learner/model were not changed.
  - Operations: five new `slurm/publication_{preflight,train,resume,prepare_evaluation,evaluate}.sbatch` files.
  - Tests: new `tests/test_publication_{spec,artifacts,recovery,evaluation,study,slurm}.py`,
    new immutable `tests/fixtures/publication_uavnet_public_v1.py`, and additions to
    `tests/test_softrole_report.py`. Fixture origin/byte hash are asserted.
  - Documentation/discovery: `publication_reconstruction/README.md`, root `README.md`,
    `.gitignore` (narrow audit exception), and this existing uppercase `AGENTS.md`.
  - Evidence: `analysis/publication_submission_2026-10-03/` contains compatibility
    and smoke validators/results, resolved protocols, source/citation evidence,
    fresh provenance patches/manifests, and final validation records. Its final
    artifact manifest inventories each file and hash without self-hashing.
- Validation already completed before the final integration pass: the full suite
  passed383 tests in124.57s. Additional focused checks exercised all16 array routes,
  quoted Slurm arguments/thread limits/exit codes, atomic and interrupted writes,
  source/checkpoint corruption, exact same-update milestone recovery, archive replay,
  strict native and transfer evaluation, masking/sham prefixes and seed summaries.
  The final integrated result is recorded below and in the validation artifact.
- Independent actual pre-change compatibility audit uses commit
  `9a436e6d9f8d36c864c17744d4d55819f5e5e504`: PCP Real/Binary ×1/4 collectors,
  four updates per old/current run. Models, active RMSprop state, historical logs,
  numerical metrics, true counts, every process RNG and subsequent random draws
  match exactly. Script and hashed results are retained, temporary runtimes/checkpoints
  removed. These bounded checks do not estimate trained task quality.
- Separate archived-launcher smoke: four-collector supplement PCP Binary, horizon3,
  batch floor4, six updates. Pause after update1 (24steps/8episodes), resume into a
  fresh segment, and compare to uninterrupted144steps/48episodes. Weights, optimizer,
  all collector RNG, counts/log and retained epoch metrics match exactly. Two actual
  frozen scenarios per failure/sham condition preserve pre-event traces and weights;
  helper archives verify and a one-seed paired summary is produced. These event-step1
  engineering scenarios do not replace the locked research panel. Exact retained
  artifact hashes are in `archived_smoke.json`; generated runs remain under
  `runs/publication_submission_validation_20261003/`.
- `uv lock --check` passed (59 packages); no dependency update. All five batch
  files passed shell syntax checks. No commit, push, full research sweep, live Stokes
  action, held-out tuning, SoftRole learner change or corrected-FC integration occurred.
  Compute-node preflight remains required before the long research submission.

- Final integration: `.venv/bin/python -m pytest -q` completed with **438 passed
  in153.44 seconds**; complete stdout is `pytest_final.log`. Final historical
  compatibility evidence is `compatibility_final.json` and includes cumulative
  active-time bookkeeping. The final runtime provenance snapshot is
  `provenance_active_time/`; all38 runtime inventory hashes validate. Cumulative
  checkpoint active time excludes startup/downtime and cannot include the later
  serialization of that same checkpoint. Final shell syntax, lock and whitespace
  checks passed. All local software gates passed; live compute-node preflight
  compatibility/resource/timing validation has not been run or claimed.


### 2026-10-03 — stdout-only frozen PCP report and HetNet deployment diagnosis

- Audited all18 `pcp-frozen-902506` stdout/stderr pairs and preparation902504,
  plus four `hetnet-preflight-902621` failures and the legacy reproduction
  root/`prev` copies. The user confirmed full JSONs were not initially synced,
  then explicitly restricted this deliverable to a short two-page stdout report
  while they copied them. No full-report outcomes, composition-specific scores,
  completion times or evaluator-source identities were invented from summaries.
- Observed18 application-completion summaries totaling16200 episode evaluations.
  Equal-seed pooled nominal success: shared99.72%, banked91.1733%; mean-agent
  return: -0.1982969/-0.3935828. Banked seed1 contributes622/662 banked failures.
  Under the preparer's declared five500-episode composition blocks, at least122
  seed1 failures occur in transfer teams; the specific teams remain unidentified.
  The one exposed failure assignment out of600 (six policies, same100 assigned
  scenarios) and zero aggregate failure-minus-sham differences do not establish
  fault tolerance. Distinguished policy-scenario assignments from independent
  scenarios, and equal updates from actual30.508112–30.728678M training steps.
- New HetNet preflights fail before run creation/training at the inventory guard;
  every stdout is empty. Reproduced a concrete Git packaging defect at inspected
  HEAD `e8bca439a19df3f5cca2f57039f40a1990cea0c6`: ORIGINS requires38 runtime
  files but only37 were tracked. The existing `runtime/envs/LICENSE.md` was ignored
  by `*.md`. A clean tracked export fails; copying only that manifest-matching
  license makes validation pass. This explains Git-based deployment, but the old
  remote error lacks filenames, so the exact Stokes inventory remains to confirm.
  Do not regenerate ORIGINS to silently accept an incomplete transfer.
- Added a narrow license ignore exception, precise missing/unexpected/hash-drift
  diagnostics, a read-only JSON source audit and a proper Slurm audit batch file.
  The guard remains strict. Runtime/model/optimizer/environment files and ORIGINS
  bytes are unchanged. Existing license bytes are preserved and made distributable.
- Separate legacy evidence: `logs_1/prev` extends PP seeds0/1/2 to1018/972/1321
  completed epochs and records SIGTERM cancellations, without stating requester
  or cause. PCP/FC copies duplicate their2000/1400-epoch endpoints. These events
  are separate from the new preflight defect; legacy cumulative counters retain
  their previously documented overcount issue.
- Produced exactly two pages at
  `output/pdf/pcp_frozen_and_hetnet_preflight_2026-10-03.pdf`, with all-seed table,
  means/ranges, failure-exposure derivation, bugs/fixes, exact Slurm commands and
  traceable log/code/research citations. Both pages were rendered with Poppler and
  visually checked; text/page-count checks passed. ReportLab4.4.4/pypdf6.1.1 ran
  through isolated `uv run --no-project`; training dependencies were untouched.
- Files changed: `.gitignore` (license/report/evidence exceptions),
  `publication_reconstruction/__main__.py` (inspection/strict diagnostic guard),
  `scripts/record_publication_sources.py` (read-only --check JSON/exit status),
  `slurm/publication_source_audit.sbatch` (new),
  `tests/test_publication_source_audit.py` (new), distribution eligibility of
  `publication_reconstruction/runtime/envs/LICENSE.md` (unchanged existing bytes),
  and this existing uppercase `AGENTS.md`. New analysis files live in
  `analysis/pcp_frozen_2026-10-03/`: analyzer, PDF builder, exact copied summary inputs,
  independent audits, row CSV/summary JSON, PDF build/text records, test log,
  validation record and artifact manifest (full inventory/hashes, excludes itself).
- Validation:28 focused launcher/lifecycle/source-audit tests passed in1.24s;
  full suite **447 passed in150.80s**. Shell syntax and whitespace checks passed.
  New tests cover exact missing/extra/modified inventories, immutable manifests,
  nonzero failure exit, read-only behavior, Git ignore eligibility, and mocked
  Slurm argument forwarding. Independent agents recalculated all18 summaries and
  checked mathematical/statistical interpretation. No logs/checkpoints/old analyses
  were edited, no jobs or research training were launched, and no commit/push
  occurred. Concurrent user synchronization under `stokes_runs/` was preserved
  and excluded from this explicitly stdout-only report.

### 2026-10-03 — synced full PCP frozen-evaluation audit

- The user supplied the full evaluation bundle under
  `stokes_runs/frozen_pcp_30m/frozen_pcp_30m_20261002_slurm/`. Audited all18 JSON
  reports and16200 episode evaluations, preserving the previous stdout-only
  report and every synced input. All stdout aggregates agree with recomputation;
  all30 nominal model/seed/composition cells contain500 assigned scenarios.
- Native2P1A equal-seed success is99.8% shared and100% banked, mean capped steps
  5.9593/6.8947, mean-agent return -0.168722/-0.181311. Across four transfer
  compositions, macro success is99.70%/88.9667%. All662 banked failures occur
  with two A agents. Seed1 contributes622:251 in1P2A,193 in2P2A,178 in3P2A;
  none occur in its native2P1A or3P1A trials. All seeds remain in reporting.
- Derived individual return R_i=-0.05 U_i from the archived reward and absorbing
  completion rules. Every banked seed1 failure has an A return of-4 at H=80:
  both A agents remain unfinished in618 cases and one in4. All P agents reach
  in620/622. This supports an A completion limitation; absent nominal trajectories
  prevent distinguishing navigation failure from failing to execute capture.
  No causal gate/bank-count claim or held-out tuning was performed.
- All600 failure/sham pairs have exactly matching recorded pre-event prefixes;
  599 finish before their own events and have identical entire traces. Only
  shared seed0 scenario2106772258:37 reaches sensor loss at zero-based t=29,
  victim1. The victim had not reached/currently seen/previously directly seen
  the target. Both conditions fail at80 steps with team return-12, but null
  attention differs on all51 post-event steps and actions on44, starting at29.
  Equal outcomes are not unchanged behavior. All banked policies have zero
  exposure, so no comparative robustness/adaptation conclusion is supported.
- Provenance checks passed:122 evaluator source hashes and combined digest,
  exact regeneration of both panels from archived scenario code,18 generated
  Slurm command mappings,16200 episode assignments, and report/manifest identity
  and runtime consistency. Selected epoch1500 weights are absent; local training
  ledgers are older snapshots. Recorded identities match, but selected weight
  bytes, a new replay and first-available>=30M selection cannot be reverified.
  The earlier HetNet packaging diagnosis remains conditional on remote inventory;
  the new evaluation sync does not establish a new Stokes source-audit result.
- Created `analysis/pcp_frozen_full_2026-10-03/` with `analyze.py`,
  `summary.json`, `composition_seed_metrics.csv`, `input_manifest.json`,
  independent `provenance_audit.py`/`.json`, `failure_audit.py`/`.json`,
  `independent_metrics.py`/`.json`, `build_report.py`, `report_text.txt`,
  `pdf_build.json`, `README.txt`, `validation.json`, and `artifact_manifest.json`.
  Created `output/pdf/pcp_frozen_full_results_2026-10-03.pdf` (exactly two pages).
  Changed only `.gitignore` for narrow artifact exceptions and this existing
  uppercase `AGENTS.md`, in addition to the new artifacts/scripts. The prior
  report PDF remains byte-identical. Source/runtime/model files were not edited.
- Validation: all four analysis/audit scripts passed;210 independent nominal
  metric comparisons agree; all147 supplied bundle file hashes remain unchanged.
  Audited1200 failure/sham episodes and7824 recorded trace steps, including3861
  compared pre-event steps. Rendered and visually reviewed both final PDF pages;
  page count, extracted key values and embedded-font layout checks passed.
  The system Python lacks NumPy; the provenance audit passed with the existing
  pinned `.venv/bin/python`, documented in README. PDF dependencies were isolated
  with `uv run --no-project`. No dependency change, policy execution, training,
  cluster submission, commit or push. The earlier447-test runtime validation is
  cited as prior work; no runtime suite rerun was needed for this analysis.

### 2026-10-03 — HetNet restart remedy for tonight

- Rechecked all four902621 failures and independently repeated clean-Git-export
  reproduction: expected38 runtime files versus37 tracked, missing only the
  manifest-approved `runtime/envs/LICENSE.md`. Restoring that exact file makes
  validation pass. Current local38-file source audit passes with zero drift;
  runtime and ORIGINS remain unchanged. The repair is still uncommitted and the
  license/source-audit batch file untracked, so ordinary Git pull would not
  transport it yet. The exact remote inventory still needs the compute-node audit.
- Reviewed launch/preflight/continuation code and official Slurm documentation.
  Prepared a concrete source-sync -> source-audit -> four concurrent100-update
  preflights ->12-job production-array sequence. Command-line `--array=0-3%4`
  changes only independent preflight concurrency; each job retains four collectors,
  500-step floors and production horizons. The training file remains0-11%3 and
  unchanged40M PP/PCP and28M FC budgets, with a fresh output root.
- Documented exact readiness fields, scheduler/resource review, expected package
  inventory/hashes, runtime projections, likely blockers and explicit continuation.
  Clarified that paused_wall_time can exit0 without completing the budget; the
 46h timer starts after setup, SIGTERM exits immediately, and periodic snapshots
  remain the abrupt-stop fallback. Old PP SIGTERM logs and legacy checkpoints
  are separate from these preflight failures and new supplementary runs.
- Created `analysis/hetnet_restart_2026-10-03/RESTART_PLAN.txt`,
  `packaging_audit.json`, `source_audit.json`, `deployment_inventory.json`,
  `validation.json` and `artifact_manifest.json`. Changed `.gitignore` only for
  this evidence directory and appended this existing uppercase `AGENTS.md`.
  No runtime/learner/Slurm file changes were needed during this follow-up.
- Fresh validation:95 focused source/lifecycle/launcher/study/Slurm tests passed
  in14.22s; the four relevant batch files pass bash syntax; all38 source hashes
  and clean-export license-only remedy passed. Earlier447-test full-suite result
  is identified as prior validation. No remote command, submission, training,
  policy evaluation, commit or push occurred. Queue availability and successful
  compute-node preflight remain conditions on starting tonight, not guarantees.

### 2026-10-03 — Stokes preflight903502 readiness check

- Checked all four synced `stokes_runs/hetnet_preflight_903502/` workloads.
  Each completed100 updates/10 epochs, returned0, reconciled step/episode counts,
  saved a finite model/active Adam optimizer checkpoint and passed an immutable
  one-episode frozen checkpoint probe. Strict archived source/checkpoint checks
  passed locally; the required license is present in all source manifests.
  Completed preflight epoch budgets are deliberately ineligible for resume as
  research runs. No new policy execution was performed during this check.
- Four collectors each use one Torch thread and have four assigned CPUs in
  affinity. Recorded per-process peaks show ample headroom relative to16GB, but
  sums of individual peaks are not measured simultaneous whole-job peaks; no
  scheduler accounting was supplied. Torch2.2.1/DGL2.1.0/NumPy1.26.4 match the
  pinned core packages. Source archives and checkpoint hashes agree with probes.
- Full-budget segment projections: PP66.25h, PCP Real114.15h, FC76.67h,
  PCP Binary287.25h. At unchanged early throughput, they need2/3/2/7 nominal
  46h segments respectively. All12 runs at concurrency3 imply about22.68days
  of idealized capacity before queues/continuation delays. Recommended prioritizing
  the three PCP Real runs tonight and addressing Binary throughput before it
  occupies the first study wave. These are projections, not runtime guarantees;
  no scientific budget was reduced and no experimental performance inferred.
- Created only `analysis/hetnet_restart_2026-10-03/preflight_903502_audit.json`,
  updated its existing artifact manifest, and appended this existing uppercase
  `AGENTS.md`. No runtime edits, tests, training, evaluation, remote actions,
  submissions, commit or push. Independent read-only audit confirmed readiness
  and timing values; all synced inputs were preserved.

### 2026-10-03 — local concurrent preflight command and printed episode timing

- The user initially requested running the four Stokes preflights concurrently
  on the Mac, then changed scope before any training started: prepare the command
  and add logging, including terminal printing. No actual preflight or research
  training was launched. Inspected the Apple M5 Max host (18 CPU cores, 36 GiB)
  and pinned Torch2.2.1/DGL2.1.0/NumPy1.26.4 environment. All45 archived Stokes
  source files matched before this logging-only change.
- Added `scripts/publication_preflight_local.py`: the existing locked indices0–3,
  concurrently by default, with four collectors each and unchanged100-update
  recipes. It uses a fresh timestamped directory, supports no-write dry runs,
  separate raw/episode/progress logs, exact command records, per-job status and
  elapsed times. Terminal output identifies each workload and prints individual
  episode steps, success, mean-agent return and rollout duration, followed by
  update summaries. Episode records become available after the update completes.
  Interrupt handling stops launcher groups, including frozen-probe descendants;
  the existing training launcher cleans its separately grouped collectors.
- Added two monotonic clock reads around `get_episode` in training collection.
  `rollout_wall_time_seconds` includes reset, inference, environment and scheduling
  delays, excluding batch gradients/optimization/IPC. Durations overlap across
  collectors and are not summed as elapsed job time. Existing update/epoch loss
  and timing records remain available. No policy, reward, RNG or learner math
  changed; original reproduction entrypoints and old run archives are untouched.
- Refreshed the strict source inventory using the existing provenance tool,
  preserving previous ORIGINS bytes and the timer diff. Only the runtime
  `trainer.py` hash changed. The38-file audit passes. Prior Stokes results keep
  their original identity; new timing runs have a distinct recorded source hash.
- Files touched: `scripts/publication_preflight_local.py` (new local launcher),
  `publication_reconstruction/runtime/trainer.py` (episode timer),
  `publication_reconstruction/ORIGINS.json` (new intentional source identity),
  `publication_reconstruction/README.md` (command/logging/timing definitions),
  `tests/test_publication_episode_timing.py` (new mocked collector/RNG test),
  `tests/test_publication_preflight_local.py` (new mocked local execution tests),
  `tests/test_publication_recovery.py` (ignore only nondeterministic episode time
  in cross-run record comparisons; verify stdout/file preservation), and this
  existing uppercase `AGENTS.md`. New provenance files under
  `analysis/publication_submission_2026-10-03/provenance_episode_timing/` are
  `previous_ORIGINS.json`, `ORIGINS.json`, `runtime_changes.patch` and `audit.json`.
- Validation:97 focused source/launcher/lifecycle/Slurm/study/timing/recording
  checks passed in15.32s. Final local-launcher/timing/recording checks passed8 tests
  in1.11s (six new local tests plus two repeated timer/recorder checks). Actual
  mock subprocess barriers prove four-job overlap; mock failure and cancellation
  checks confirm exit propagation and parent/probe cleanup. Printed per-episode
  values match saved records, and timer reads preserve Python/NumPy/Torch random
  draws. These checks use mocks or dry runs, without training. Dry-run commands
  preserve all four workload mappings and create no run directory. Syntax and
  whitespace checks pass; the full training/recovery integration suite was not run.
  Existing Stokes-readiness edits were preserved; no dependency change, scheduler
  action, commit or push occurred.

### 2026-10-04 — completed Mac concurrent preflight analysis

- Analyzed the user's completed `runs/hetnet_preflight_mac_20261003_234455_727228/`
  against `stokes_runs/hetnet_preflight_903502/`, without running any new policies.
  All four jobs returned 0, completed 100 updates/10 epochs, and passed their
  recorded frozen checkpoint probes. Reconciled all 10,950 local episode records,
  893,647 steps, 400 updates, 40 epoch rows and printed-progress data against both
  stdout copies. Every numeric metric is finite; all episode timers are positive.
- Full local job times were PP 9m18s, PCP Real 16m22s, FC 18m24s, PCP Binary 17m01s.
  Total batch elapsed time was 18m24s. Starts were within 4.5ms; all four launchers
  overlapped for 9m18s, with four one-thread collectors each. Mac/Slurm update
  throughput ratios were 2.33/2.26/2.26/5.48 in that workload order. Episode mean
  rollout times were 0.403/0.749/2.185/0.789s; timers exclude batch optimization.
- Independently loaded all eight local/reference checkpoints and verified source,
  scientific command, counts, finite model/active Adam tensors, optimizer step 100,
  checkpoint/sidecar/probe identities and archived evaluators. Only documentation,
  provenance and the intended episode-timer source addition differ. Cross-platform
  initial models differ in 36 float32 attention tensor hashes per workload; all
  initial float64 hashes and shapes/dtypes agree. This is matching-recipe timing,
  not a controlled hardware-only or bitwise-identical training comparison.
- Recorded declining contention as jobs finished: FC throughput in wholly included
  updates was 197/234/335/411 steps/s across four/three/two/one-launcher phases.
  Four-job execution is viable for this test; optimal concurrency and multi-day
  throughput remain unmeasured. Sum of individual collector RSS peaks is 11.54 GiB,
  not simultaneous system memory; no power/temperature/swap trace was recorded.
  Full-budget linear Mac projections are 28.35/50.24/33.81/52.27h per run, conditional
  on these mixed-concurrency rates. Early training changes remain descriptive
  single-seed diagnostics; no research performance or convergence inference.
- Files touched: only this existing uppercase `AGENTS.md` and the new sibling
  `runs/hetnet_preflight_mac_20261003_234455_727228_analysis/`, containing `analyze.py`,
  `report.txt`, `checkpoint_audit.json`, `timing_audit.json`, `metrics/summary.json`,
  `metrics/input_hashes.json`, and `artifact_manifest.json` (hashes all other
  analysis files, excludes itself). The main analysis checks all 639 non-bytecode
  input file hashes unchanged. Independent numerical and checkpoint/source audits
  agree; whitespace checks pass. No runtime edit, training/evaluation, dependency
  change, scheduler action, commit or push; all pre-existing edits were preserved.

### 2026-10-04 — evidence-backed HetNet performance plan

- Wrote `publication_reconstruction/PERFORMANCE_PLAN.md` at the user's request.
  The ten recommendations link to the authors' paper/supplement, inspected
  executable code, completed Mac/Stokes preflights, existing tests and official
  versioned PyTorch/DGL or Slurm documentation. Observations, unmeasured candidate
  benefits and proposed lab acceptance rules are explicitly distinguished.
- Defined the preservation target as the selected `supplement-v1` /
  `corrected-v1` October reconstruction, retaining its actual learner, mixed
  tensor dtypes, Gumbel draw order, typed graphs, recurrence, four collectors,
  horizons and sample budgets. Documented the existing FC paper/reconstruction
  differences and the unavailable verified publication-producing checkout.
  No exact reproduction or demonstrated acceleration is claimed.
- Proposed profiling PCP Real/Binary on Stokes before choosing an optimization,
  with actual CPU topology/affinity evidence, bounded optional traces, unchanged
  trajectories and profiling-overhead controls. Candidates come from observed
  repeated relation views, feature-copy loops and repeated graph construction;
  smaller allocation/learner/environment edits remain deferred. Every candidate is conditional on measurement
  and complete baseline-versus-candidate equivalence checks.
- Specified independent archived-reference, environment/model/RNG/gradient/Adam,
  multi-update and lifecycle gates; three alternating paired 20-update timings
  per workload; and final official preflights. Benchmark windows and the rule
  requiring improvement in all three pairs are lab choices, not author settings
  or statistical guarantees. Preserved source guards
  and required intentional candidate provenance before any launcher-based check.
- Added `analysis/hetnet_performance_plan_2026-10-04/evidence.json` with 78 file
  hashes, 27 evidence entries, eight primary URLs, recommendation-to-evidence
  mappings and independently recomputed timing/signature/resource metadata.
  Added its standard-library, read-only `verify_evidence.py`; large local run
  archives and paper copies remain separate required inputs. The verifier does
  not import training or claim to repeat the earlier checkpoint tensor audit.
- Validation: verifier passed for all 78 direct file identities, all 639 archived
  run inputs, all 38 runtime provenance entries, all local links, four workload
  pairs and eight checkpoint byte identities. Table rates/projections, sample
  counts, launcher timestamps and initial signature differences reconcile with
  raw artifacts. Three independent read-only reviews checked technical source
  claims, paper attribution and measurement/gate design; their corrections were
  incorporated. Runtime hashes still match the pre-existing source manifest.
- Files touched in this task: `publication_reconstruction/PERFORMANCE_PLAN.md`
  (new), `publication_reconstruction/README.md` (discovery link), `.gitignore`
  (narrow plan/evidence exceptions and cache exclusion), this existing uppercase
  `AGENTS.md`, and `analysis/hetnet_performance_plan_2026-10-04/{evidence.json,
  verify_evidence.py,validation.json,artifact_manifest.json}` (new evidence,
  validation record and final deliverable hashes). Concurrently added
  `check_external_sources.py`, `external_sources.json` and `verification.json`
  were inspected and preserved; corrected their local-PDF availability wording.
  No runtime, optimizer, environment, dependency, checkpoint or input change;
  no profiling, training/evaluation, cluster action, full test-suite rerun,
  commit or push. All earlier working-tree changes were preserved.

### 2026-10-04 — exact-first plan and comparison evidence finalization

- Completed the performance plan around the approved three opt-in changes:
  per-layer relation views, guarded per-collector topology reuse and exact bulk
  feature copies. Defined the proposed `reference|cached-v1` interface, reference
  default, profiling option and fresh-run-only rollout. Environment/wrapper
  bytes, learner formulas, gradient storage and archived continuation remain
  preservation requirements. Other visible redundancies are deferred.
- Added the scientific purpose and comparison contract: twelve locked HetNet
  runs, original/supplement attribution, outcome-independent checkpoint rules,
  common PCP panel bytes, seed-weighted reporting, information/health-input and
  learner differences, and the unmatched existing FC environment boundary.
  This does not certify the unavailable publication-producing checkout.
- Final benchmark specification is three matched pairs of 20 complete updates
  per workload, alternating order, with update 1 and startup reported separately.
  It supersedes the initial draft's 30-update/A-B-B-A and 10%/5% proposal.
  Require exactness independently of speed; report each timing pair and retain
  reference mode when the proposed workload-level rollout test is inconclusive.
- Excluded conversation-only microbenchmark speedup/equality numbers from the
  accepted evidence: their full scripts/raw samples were not retained. Added
  source-level index/topology reasoning and explicit byte/dtype/shape/gradient
  None checks to the required future tests, without claiming they already pass.
- Finalized local evidence inventory and read-only verifier, including a check
  that its 27 source-register entries match the document. Independently verified
  78 direct file hashes, 639 archived inputs, 38 runtime manifest files, eight
  checkpoint byte/probe identities and all four Mac/Stokes workload pairs.
  This is not a new checkpoint tensor audit or policy execution.
- Added `check_external_sources.py` and `external_sources.json`: eight primary
  references fetched successfully; publisher/pinned supplement PDF hashes match
  the prior manifest. Located and rehashed the existing PDFs in `docs/papers/`
  too; the previously inspected `research/papers/` filenames are absent. The
  pinned profiler/timer source hashes also match their installed Torch copies.
  Source text interpretation and prior experimental audits remain distinguished
  from these byte-identity checks.
- Files finalized: `publication_reconstruction/PERFORMANCE_PLAN.md`, its README
  discovery paragraph, this existing uppercase `AGENTS.md`, the narrow plan
  exceptions in `.gitignore`, and
  `analysis/hetnet_performance_plan_2026-10-04/{evidence.json,verify_evidence.py,
  verification.json,validation.json,artifact_manifest.json,check_external_sources.py,
  external_sources.json}`. Final validation/artifact records identify exact
  counts and hashes; the artifact manifest excludes itself.
- No runtime/environment/dependency/checkpoint edit, training, evaluation,
  performance experiment, cluster submission, full test-suite rerun, commit or
  push was performed by this documentation finalization. Pre-existing work,
  including episode timers, launchers, source provenance and reports, was kept.

### 2026-10-04 — shared performance-plan review and validity guards

- Located `publication_reconstruction/PERFORMANCE_PLAN.md` as the current shared
  plan; the restart and older architecture/evaluation plans have different
  scopes. Reviewed it with the three reachable source/learner/operations agents.
  Direct cross-session contact with its other author was not available; the
  user subsequently said that agent was finished. Continued against the shared
  file, without creating a competing acceleration proposal or claiming the
  other author's approval.
- Retained exact-first R5–R7 as candidates conditional on phase measurements and
  complete equivalence. Added a linked review, explicitly separated completed
  guard work from proposed execution-mode/profiling/harness work, and recorded
  benchmark work bounds: 24 runs, 480 updates, 960,000–1,217,280 steps. Documented
  the alternating-order imbalance and the future old-command compatibility
  test required when the execution-mode flag is introduced. No speedup claim.
- Reproduced and fixed outer validity gaps: direct resume and preflight readiness
  now bind checkpoint bytes/length/progress to recorded checkpoint entries;
  existing optimizer state checks include step, moment shape/dtype and
  nonnegative second moments while allowing legitimately unused parameters;
  schema-2 scientific validation checks relevant saved defaults and rejects
  missing fields; selected parent checkpoints are rechecked against the locked
  protocol and actual seed; preflight update ledgers reconcile sequence, epoch
  positions, finite timing and cumulative work with segment/status/checkpoint
  counts. These are malformed-input acceptance defects, not demonstrated damage
  to the archived experiments.
- Added a fail-fast launcher guard for PCP/FC horizon 1 with collector batch
  floor 1. The frozen per-class learner's singleton A advantage has undefined
  sample standard deviation; eight synthetic PCP/FC × Real/Binary × model-spec
  forward/backward probes reproduced nonfinite losses and gradients. Preserved
  the learner formula; production horizons/floors remain unchanged.
- Changed code/tests: `publication_reconstruction/__main__.py` (tiny-config
  guard), `publication_reconstruction/artifacts.py` (configuration/optimizer/
  recorded-byte guards), `publication_reconstruction/study.py` (ledger/readiness
  and selected-parent guards), `tests/test_publication_artifacts.py` (realistic
  schema-2 fixture and ledger), `tests/test_publication_launcher.py` (boundary
  cases), `tests/test_publication_study.py` (malformed ledger/byte/continuation
  cases), and new `tests/test_publication_recovery_integrity.py` and
  `tests/test_publication_validity.py` (corrupt-state/default/label regressions).
- Changed documentation/evidence: `publication_reconstruction/PERFORMANCE_PLAN.md`,
  this existing uppercase `AGENTS.md`, and
  `analysis/hetnet_performance_plan_2026-10-04/{evidence.json,verify_evidence.py,
  verification.json,validation.json,artifact_manifest.json}`. New
  `validity_review/` contains `README.md`, `check_archives.py`,
  `archive_checks.json`, `probe_singleton.py`, `singleton_probe.json`,
  `pytest_full.txt`, `pytest_guards.txt`, `validation.json`,
  `baseline_manifest.json`, `prior_evidence.json`, `prior_performance_plan.md`,
  `prior_validation.json`, `prior_artifact_manifest.json`, `source_revision.json`,
  `guard_changes.patch`, and `baseline/` with the 54 exact paths/hashes listed in
  its manifest. The outer artifact manifest enumerates all final deliverables.
- Before guard edits, froze all 38 runtime files and 16 additional source/test
  files against the existing evidence inventory. Retained prior plan/inventory
  and explicit before/after receipts rather than silently replacing hashes.
  The updated verifier checks 83 direct inputs, four reviewed source revisions,
  all 54 frozen copies, 639 archived run inputs, 38 runtime provenance entries,
  eight checkpoint byte identities and unchanged four-workload timing results.
  Deliberately corrupted transition metadata was rejected.
- Validation: `.venv/bin/python -m pytest -q` passed **524 tests in 150.64s**.
  After the final preflight checkpoint-ledger binding call and its new test,
  the affected five-file guard suite passed **171 tests in 1.45s**. These are
  separate scopes, not a claim of a second full-suite run. Read-only actual
  artifact checks accepted 33 existing checkpoints (Adam/RMSprop, mixed dtypes,
  unused parameters), eight genuine Mac/Stokes ledgers with unchanged rates,
  and two paused-checkpoint resume resolutions. No archive bytes changed.
  `uv lock --check` resolved 59 packages unchanged; syntax for all six
  publication Slurm files, Python parsing, document links, source audit and
  `git diff --check` passed. Tests used bounded training/evaluation fixtures;
  no research/performance run, cluster submission, commit or push.
- Deferred a separate report-parser hardening issue: synthetic inconsistent
  composition/sham fields and string success can be accepted by
  `softrole/report.py`. All four supplied Stokes probe reports passed the
  corresponding consistency checks; no result corruption was observed.
  Full reproduction fidelity and the documented corrected-FC comparison limit
  remain as stated in the plan. Binary's Stokes slowdown is still unexplained.
- Training runtime, environment/wrapper sources, learning formulas, parameter
  dtypes, RNG order, collector partitioning, gradient storage, optimizer settings
  and budgets were unchanged by this review. Existing concurrent timer,
  launcher, provenance and documentation changes were preserved. Runtime
  `ORIGINS.json` remained valid without refresh; new outer-helper/evaluator
  snapshots receive their own hashes through the existing archive mechanism.

### 2026-10-04 — DGL cost, Binary slowdown, communication radius and dense/CUDA implications

- Scope: answered the user's questions on communication radius, the Stokes Binary
  slowdown, replacing DGL with plain PyTorch, and CUDA on Newton. No runtime, test,
  protocol, Slurm or dependency file changed; no cluster action. The requested
  Stokes benchmark is described below but deliberately not written. Measurements
  first ran in a temporary scratchpad; scripts and raw outputs were copied unchanged
  into `analysis/dgl_profile_2026-10-04/`. Its `scripts/summarize.py` recomputes
  every number in this entry into `summary.json`, hashing all 132 inputs. A 6.4x
  microbenchmark ratio quoted in conversation came from an unsaved first run and is
  superseded by the saved rerun (6.20x).
- Current experiments enforce no communication radius. Defaults of -1 mean all
  pairs: `softrole/config.py:30`, `softrole/rollout.py:44-50`, reconstruction
  `runtime/main.py:124-127` (`:222` also disables lossy communication) and
  `runtime/hetgat/utils.py:71,86`; the reconstruction frozen evaluator rejects other
  ranges (`publication_reconstruction/evaluation_worker.py:79`). Recorded values:
  all 18 SoftRole primary configs -1.0; all 8 Mac/Stokes reconstruction preflights
  -1/-1 with lossy false; all 18 frozen PCP 30M result reports -1.0; all 144
  `logs_1` argument records -1. Edge distances are stored but read only by lossy
  layers (`fastreal.py:747`, `fastbinary.py:513`), which are inactive.
- Authors' radius. The paper motivates limited-range communication and adds edges
  "within communication range" (`research/papers/HetNet.txt:231,265`), but Table 1
  (`:514`) states no radius. The only range experiment is Ablation #2, section 6.3.3 /
  Fig. 5a, PCP only: Full, Half ("i.e., limited range") and No communication
  (`:547-556`, `:577`). No radius value or definition of Half is given; the
  supplement text has zero matches for "communication range", "radius" or "half".
  The nine training command lines in public upstream READMEs (snapshots 6b09d36,
  bff9f7f, f54ca5a) set no range, and the parser default is -1. Inference: Table 1
  most likely used full communication, as we do. This is unconfirmed because the
  authors' training commands are unavailable (prior author-search audit).
  Reproducing Fig. 5a would require choosing an undocumented radius.
- Binary versus Real. Binarization arithmetic is small. Mac, same code, one
  collector, uninstrumented: Real 122.4 versus Binary 113.4 steps/s (Binary 7.95%
  slower). `gumbel_softmax` is 0.58% of cProfile time. The archived Mac four-collector
  preflight shows 1.036x collector CPU-seconds per step for Binary. Stokes 903502
  shows 2.518x lower throughput (97.72 versus 38.81 steps/s) and 2.573x CPU-seconds
  per step. Collectors were 86-95% CPU-busy in both jobs (CPU-bound, not waiting),
  and Binary was already slow in updates 1-10 (85.8 versus 44.0 steps/s). The jobs
  had different logical CPU sets (Real 33/35/37/39, Binary 3/5/7/9). No
  non-source file in either preflight archive records a host or node. Unresolved,
  untested candidates: node/core placement or co-tenancy, or an x86-specific slow
  path in a Binary-only operation (for example subnormal arithmetic). The evidence
  rules out the binarization networks' extra arithmetic as the main cause.
- DGL share. Mac, PCP Real, one collector, four updates, seed 991, supplement-v1,
  corrected-v1: DGL takes 69.4% of update time (Binary 66.8%):
  - forward message passing and graph data: 39.2%;
  - DGL autograd backward: 22.4%;
  - per-step graph construction: 7.8%.

  The environment step is 0.7%. By method: `update_all` 12.6%, `apply_edges` 10.2%,
  relation slicing 7.2% (141 calls/step), node-data writes 4.3%, `edge_softmax`
  3.0%. Timers add 5.2%/0.7% overhead and leave update records and parameter
  signatures identical to the uninstrumented runs in all five comparisons.
  cProfile inflates Python overhead and is used only for named-function shares.
- Dense microbenchmark of the replaced portion. Same 2P/1A/two-state topology as
  `build_hetgraph`, three layers x five non-empty relations, 4x16 heads, float64
  features with float32 attention vectors, 80 steps then backward, six alternating
  repetitions. Result: DGL 5.64 versus dense PyTorch 0.91 ms/step (6.20x). This is
  not a numerical-equivalence test.
- Estimate, not measured end to end: replacing the measured in-situ DGL time
  (5.98 of 8.62 ms/step) with 0.91 ms gives 3.55 ms/step, about 2.43x on this Mac.
  The ceiling is 3.27x if DGL cost were zero. The plan's R5+R7 address only relation
  slicing plus graph construction (15.1%), so at most 1.18x. Stokes, multi-collector
  aggregation and later-training behavior are unmeasured. Conditional arithmetic
  only: if the 2.4306x ratio held on Stokes for every workload (unmeasured), the
  903502 projections would become PP 27.3, PCP Real 47.0, FC 31.5 and PCP Binary
  118.2 h. That is 1/2/1/3 nominal 46-hour segments instead of 2/3/2/7, and the
  idealized 12-run wave at concurrency 3 would drop from 22.68 to 9.33 days.
- Effects of DGL -> dense PyTorch (analysis only; not adopted):
  - Unaffected. SoftRole's model/learner imports no DGL and already uses dense masked
    attention (contract item 5): its 18 primary runs, the frozen PCP 30M panel and the
    sensor pilot stay valid. Also unaffected: legacy root reproduction runs (root
    `hetgat/` DGL code stays unchanged) and the root-DGL `softrole evaluate-hetnet`
    path.
  - Reconstruction. No research-run artifacts exist locally and this record shows no
    launch. Adoption would supersede the 903502/Mac throughput projections
    (66.25/114.15/76.67/287.25 h) and require fresh preflights for the new source
    identity; DGL stays as the reference oracle. All 12 first-wave runs must use one
    implementation, and DGL checkpoints are not continued under different source
    (existing guard). If DGL research runs were launched after 903502, either finish
    all 12 on DGL or restart all on dense. Follow-up: the user confirmed on
    4 Oct that no Stokes runs were launched after the preflight, so no DGL
    reconstruction research runs exist to finish or discard.
  - Numerics. Same equations and parameters but a different reduction order, so
    results are not bitwise identical and per-seed trajectories diverge; comparison
    is at the seed-distribution level. Precedent: Mac and Stokes initial models
    already differ in 36 float32 tensors per workload
    (`runs/hetnet_preflight_mac_20261003_234455_727228_analysis/checkpoint_audit.json`),
    and PyTorch gives no cross-platform bitwise guarantee (plan W2).
  - Required before use:
    - fixed-input forward/hidden/value/gradient agreement within declared tolerances
      for PP/PCP/FC x Real/Binary x both model specifications, including empty
      relations and zero-A cases;
    - unchanged Gumbel RNG consumption order, with any flipped hard bits or actions
      counted;
    - identical state_dict names, shapes and dtypes so checkpoints cross-load;
    - DGL-trained checkpoints reproduce their action distributions within tolerance
      under dense evaluation.
  - Plan conflict. `PERFORMANCE_PLAN.md` requires exact equality for its strict route
    (R9, lines 152-156) and lists dense replacement of typed DGL message passing as
    outside the plan (line 196). Adoption needs an explicit amendment adding a
    tolerance-gated opt-in execution mode. Root reproduction defaults stay unchanged.
  - Contribution. SoftRole's contribution is the capability-conditioned,
    label-independent architecture and its composition and sensor-failure
    evaluation. The baseline's graph library is an implementation detail to report
    in the experimental setup, together with the equivalence evidence. It would
    matter only for an exact-reproduction claim, which the protocol `claim` field
    already disclaims. Neither version supports compute-efficiency comparisons with
    SoftRole, since the learners differ.
- CUDA on Newton. The ARCC page (retrieved 4 Oct 2026) lists 10 nodes with 2x V100
  16 GB, 11 with 2x V100 32 GB, 29 with 2x H100 80 GB and 4 with 8x H100 80 GB.
  Removing DGL is not what enables CUDA:
  - the code already moves graphs to `self.device` (`runtime/hetgat/policy.py:528`);
  - the Stokes Torch 2.2.1 install includes CUDA 12.1 libraries (preflight
    `environment.json`);
  - CUDA support in the installed DGL wheel is unverified;
  - the current blocker is our guard at `runtime/main.py:219`.

  The expected benefit is low and unmeasured. Inference is batch-1 per step on tiny
  tensors, and sampled actions are copied to CPU every step
  (`runtime/action_utils.py:40,51`) for the NumPy environment
  (`runtime/trainer.py:192-194`); PyTorch documents that CPU-GPU copies synchronize.
  A material GPU benefit would plausibly require batched or vectorized environments,
  a collection redesign that needs its own protocol and RNG review.
- CUDA follow-up (the user asked why a PyTorch port could not use CUDA). A port
  could run on CUDA. The evidence predicts no speedup, but this is unmeasured.
  - Operation counts. `scripts/count_ops.py` profiled one PCP Real update (557
    steps); its non-timing records equal base_real's first update. Even excluding
    DGL's own kernels, each environment step issues about 2,795 forward and 1,543
    backward top-level PyTorch ops. Excluding listed view/allocation ops, that is
    about 1,138 and 757 compute-type leaf ops, an overestimate of GPU kernels.
  - Operation sizes. The median largest input is 64 elements, and about 80% of
    leaf ops touch at most 1,024 elements. The dense message-passing part alone
    issues about 1,429 forward and 754 backward top-level ops per step, with
    median 8 and maximum 128 elements.
  - Mechanism. PyTorch documents that GPU operations are enqueued asynchronously
    and that CPU-GPU copies synchronize. The CPU-side issue cost of every op
    therefore remains on GPU, and every step's sampled action is copied to CPU
    (`runtime/action_utils.py:40,51`). The arithmetic a GPU accelerates is
    negligible at these sizes. Whether net time rises or falls on Newton is
    untested.
  - Port requirements beyond dropping DGL:
    - remove the CPU-only guard (`runtime/main.py:219-220`) and the CPU double
      default tensor type (`runtime/main.py:55`);
    - capture CUDA RNG for pause/resume, since `recovery.py:50-52` saves only
      Python, NumPy and CPU Torch states;
    - adapt gradient sharing across spawned collectors: raw `p._grad.data`
      pointers (`multi_processing.py:45,123`) with spawn (`main.py:50`). The
      PyTorch 2.2 multiprocessing notes require senders to keep shared CUDA
      tensors alive;
    - accept that seeds become non-comparable. PyTorch 2.2's randomness notes
      say CPU and GPU results may differ even with identical seeds, and CUDA
      sampling uses a separate generator. All 12 runs would need one device.
  - Option. A CUDA arm on Newton can measure this after Stokes Phase A/B.
    Batched multi-environment collection is the plausible route to GPU benefit,
    but it is a separate design and protocol decision.
- Proposed Stokes benchmark (not written; one task, four CPUs, no research runs):
  - Phase A, needs no model code:
    - record node identity: hostname, `lscpu`, `lscpu -e`, `scontrol show job`,
      affinity, Torch configuration, package and source hashes;
    - run `dense_vs_dgl.py` for at least three alternating repetitions;
    - run instrumented and uninstrumented one-collector PCP Real and Binary
      back-to-back in one allocation, with the identity check. This measures the
      x86 DGL share and tests whether Binary's gap persists on the same CPUs.
  - Phase B, only after a dense implementation passes equivalence: the plan's R10
    paired harness (three pairs x 20 updates, four collectors, alternating order),
    then the official preflights.
  - An optional Newton CUDA arm comes only after A and B.
- Files touched:
  - `AGENTS.md` (this entry);
  - `.gitignore` (narrow folder exception excluding caches, the runtime copy and
    rerun outputs);
  - new `analysis/dgl_profile_2026-10-04/`:
    - `README.txt`, `summary.json`, `runtime_identity.json`, `artifact_manifest.json`;
    - `scripts/{instrument.py, instrument_detail.py, run.sh, run_detail.sh,
      analyze.py, dense_vs_dgl.py, summarize.py, count_ops.py, manifest.json}`;
    - `raw/{base_real, base_binary, ins_real, ins_binary, det_real, cprofile_real,
      cprofile_binary}/` (run records, logs, timers, `.prof`),
      `raw/dense_vs_dgl.txt`, `raw/op_counts_{insitu_real,dense}.json`,
      `raw/op_counts_dense_stdout.log` and `raw/op_counts_insitu_run/`.

  The manifest hashes all 96 other files after the CUDA follow-up.
- Validation:
  - `summarize.py` reran end to end; the executed runtime copy matched all 38
    working-tree runtime files.
  - Scripts compile; shell syntax and `git diff --check` pass.
  - Fetched the external ARCC Newton page and PyTorch 2.2 CUDA-semantics page on
    4 Oct 2026; neither page is archived.
  - No tests reran because no runtime source changed.
  - All earlier working-tree changes were preserved; no commit or push.

### 2026-10-04 — paper-v1 reconciliation and PyTorch implementation (in progress)

- User approved a new paper-aligned baseline before acceleration, small numerical
  differences between equivalent backends, matched FC rerun preparation, and a
  target of completing core implementation in this session. This supersedes the
  earlier environment-fixed/exact-backend performance proposal for NEW runs only.
- Explicit user choices: correct paper conflicts before acceleration; retain
  documented public-source defaults for unspecified details; Binary has 64 bits
  per independent head (four heads = 256 payload bits per sender/round). This is
  a declared interpretation, not verified recovery of the paper's 64-bit result.
- Frozen the 38 pre-change runtime files and launcher/validator/evaluator/study
  sources in analysis/paper_v1_2026-10-04/baseline; baseline_manifest.json records
  SHA256 identities. All runtime bytes matched the existing ORIGINS inventory.
  initial_status.txt preserves the pre-existing dirty-tree inventory.
- Evidence: main paper §§4–5 and Algorithm 1 (research/papers/HetNet.txt),
  supplement §§1–2 (research/papers/HetNet_Supplementary.txt), active public
  runtime and older independent-head implementation. New architecture, learner,
  FC semantics and implementation-backend changes will be validated separately.
- Work split: independent class preprocessing and head channels plus Torch
  backend; equation-aligned learner; shared paper-FC environment/SoftRole adapter;
  launcher/checkpoint/frozen-evaluation/profiling/benchmark/provenance integration.
  Existing archives and continuations retain their original source. No cluster
  submission or full research sweep is part of the implementation validation.

- Core implementation progress: added the coherent paper-v1 launcher preset,
  backend/learner/checkpoint metadata and source-bound evaluation dispatch. The
  new benchmark keeps24 sequential20-update runs distinct from100-update
  preflights; phase diagnostics are optional and separate. Added Stokes batch
  files for the benchmark, paper preflights, fresh HetNet wave and matched
  SoftRole FC. No script has been submitted.
- Added publication_reconstruction/FIDELITY.md with primary-paper/supplement
  links, equation-to-code/test mapping, all declared defaults,256-bit aggregate
  Binary accounting and the boundary between backend equivalence and scientific
  fidelity. Amended PERFORMANCE_PLAN.md and README.md without deleting earlier
  evidence. Corrected README attribution: per-class critics are supplement-backed;
  the8-wide state output is the public-source choice.
- First integration checks:47 launcher/benchmark/legacy-launcher tests passed
  (0.29s; the archive-hash test was deferred until intentional provenance refresh).
  Benchmark and study dry runs resolve without creating training outputs. New
  model, learner and environment tests are being independently reviewed; final
  consolidated results will be recorded below, not inferred from these checks.

### 2026-10-04 — paper-v1 implementation completed and launch checks

- Implemented the approved reconciliation as an opt-in, coherent
  `--reconstruction-spec paper-v1` with `--message-backend dgl|torch-v1` (DGL
  default). Legacy defaults and archived continuation sources remain preserved.
  This is a fresh baseline, not a migration of prior trained weights. The
  [fidelity contract](publication_reconstruction/FIDELITY.md) maps each change to
  primary evidence, declared choices and independent tests. Local primary text
  and available PDF identities are in
  `analysis/paper_v1_2026-10-04/primary_evidence.json`.
- Architecture: separate P/A status preprocessing and recurrence; three layers;
  four independent heads; 16-wide hidden heads concatenated, final heads averaged;
  independent Binary encoders/binarizers/decoders; class-routed SENs and critics.
  SEN time now uses actual zero-based episode time. No SEN information reaches
  the actor. Paper Fig. 1/§4 and supplement §2.1 support these structural choices;
  the two SENs, width eight and unspecified defaults are explicitly attributed to
  public source. The selected 64 bits/head means 256 payload bits/sender/round,
  not a verified reconstruction of the publication's reported 64-bit setting.
- Backend: shared parameters and sender computations, with DGL reference relation
  aggregation versus a new receiver-by-sender masked Torch implementation.
  Each typed relation keeps its own incoming softmax; relation results are added
  to self features in order. Empty neighborhoods contribute zero without NaNs.
  Only topology is cached; no feature, gradient or learned tensor is cached.
  CPU, unlimited non-lossy communication and fixed within-episode teams are
  enforced for this initial backend. Old fastreal/fastbinary/uavnet files were
  not changed. Mixed parameter dtypes and independent-head random draw order
  are preserved between the two new backends.
- Learner: individual GAE actor advantages, separate Monte-Carlo rewards-to-go
  for class-average critic targets, total-team-N weighting, global completed-
  episode average, one global .75 clip and one Adam 1e-3 step. No padded
  advantage standardization, actor multiplier 50, critic time averaging, local
  clipping or final environment-step division enters this new learner. These
  choices follow paper §5.2/Algorithm 1 where specified; the remaining defaults
  and the combined shared-backbone objective are declared in FIDELITY.md.
- FC: paper-only branch implements four movement actions for P, movement plus
  extinguish for A, prior discovery before extinction, and additive temporal,
  unique ignition/extinction and false-drop rewards from supplement §1.3.
  Corrected FARSITE propagation/boundary handling is retained. PP/PCP retain
  corrected-v1 physical behavior. SoftRole's opt-in paper-FC adapter loads the
  same verified archived simulator bytes, masks stay and maps internal extinguish
  action 5 to native action 4. Its own actor/critic/learner remain its existing
  method; older FC results must remain a separate environment stratum.
- Checkpoint/evaluation: protocol and checkpoint identities include model,
  environment, learner and backend. Continuation inherits archived bytes and
  backend without an override. Frozen evaluation instantiates the archived
  model/backend and verifies checkpoint immutability. Phase profiling defaults
  off and is separated from unprofiled throughput runs; summed collector wall
  intervals overlap, while parent wait/aggregation/optimizer intervals are separate.
- Benchmark: 24 sequential runs, four workloads, three alternating backend pairs,
  20 updates each, four collectors, 500-step floors, production horizons and
  seed 991. It records startup, updates, process/segment rates, CPU intervals,
  memory units, topology/affinity, package and source/dependency/launcher hashes.
  Optional eight three-update phase runs are separate. The speed gate requires
  all three post-first-update improvements and higher median segment throughput.
  Normal episode logs lack full trajectory hashes, so all current workloads
  require fixed-rollout compute replay. Replay excludes environment/IPC, includes
  RNG restoration/Python iteration, and checks model/hidden/gradient/loss/Adam/RNG
  agreement before timing. Diagnostic cloning and per-update model copying were
  removed from its timed path; non-finite comparisons fail explicitly.
- Review fixes: incomplete/empty episode ledgers now fail rather than vacuously
  certifying identical trajectories. Benchmark updates, episode order, collector
  floors and final checkpoint bytes/counts are reconciled. Segment gating uses
  the ratio of median throughputs, not the median of pair ratios. Process timing
  begins before process creation. These integrity checks prevent unsupported
  performance claims; they do not imply a measured speed improvement.

Files touched by this implementation (pre-existing changes were preserved):

- Reconstruction model: new `publication_reconstruction/runtime/hetgat/paper.py`,
  `runtime/hetgat/graph/paper.py`, `runtime/hetgat/graph/torch_backend.py`.
- Reconstruction learner/runtime, relative to `publication_reconstruction/`:
  new `runtime/hetnet_ext/paper_learning.py` and `paper_replay.py`;
  changed `runtime/hetgat/policy.py`, `runtime/trainer.py`,
  `runtime/multi_processing.py`, `runtime/main.py`,
  `runtime/hetnet_ext/{recording,recovery,signatures}.py`.
- Versioned simulator branches, same prefix:
  `runtime/envs/ic3net_envs/{fire_commander_env,predator_capture_env}.py`,
  `runtime/WildFire_Simulate_Original.py`.
- Reconstruction integration: `__main__.py`, `artifacts.py`,
  `evaluation_worker.py`, `study.py`; new `benchmark.py`, `paper_study.py`;
  `ORIGINS.json` refreshed through the existing audit command.
- SoftRole: `softrole/{__main__,config,env,evaluate,model,train}.py` and new
  `softrole/publication_env.py` for versioned simulator selection/source binding.
- New batch scripts: `slurm/publication_backend_benchmark.sbatch`,
  `publication_paper_preflight.sbatch`, `publication_paper_train.sbatch`,
  `softrole_paper_fc.sbatch`. No submission command was executed.
- New tests: `tests/test_publication_paper_{model,learning,training,replay,
  launcher,benchmark,study}.py` and `tests/test_softrole_paper.py`.
- Documentation/provenance: `.gitignore`, `AGENTS.md`, reconstruction `README.md`,
  amendment to existing `PERFORMANCE_PLAN.md`, new `FIDELITY.md`; fresh
  `analysis/paper_v1_2026-10-04/` (reference snapshot, audit records, reproducible
  validation scripts, exact commands, reports and hashes). Its final artifact
  manifest inventories these files individually. Prior performance reports,
  local preflight launcher, episode timers and existing regression edits remain.

Validation and measured results:

- Full integrated suite: **610 passed in 278.26 s**, recorded in
  `pytest_full_01.txt`. After final replay/benchmark hardening: **173 passed in
  10.72 s** (`pytest_final_focused.txt`), plus three study-provenance tests in
  0.76 s. These suites overlap and their counts are not additive. Focused replay
  tests exercise actual fixed-rollout recording and final DGL model/Adam identity.
- Legacy environments: 30 fixed-action traces across PP/PCP/FC and both old
  environment versions; 600 steps in each source matched exactly, including
  observations, rewards, physical state and final NumPy RNG. Evidence:
  `legacy_environment_replay.json`, using the immutable pre-change snapshot.
- Archived training/frozen loading matrix: 24 fresh runs across all three tasks,
  both message variants, one/four collectors and both backends; 72 complete
  updates, horizon six/floor seven, and 24 frozen one-episode evaluations.
  All model/active-Adam comparisons passed fixed dtype tolerances; initial
  model/optimizer/RNG, final RNG and counts matched across all 12 pairs. Largest
  model absolute difference was 4.659640728821302e-13 (float64); float32 model
  parameters were exact. Frozen checkpoints remained unchanged. Scripts and
  results: `validate_integration.py`, `compare_integration.py`,
  `integration_validation.json`, `integration_numerical_recheck.json`.
- Actual outer-CLI continuation: DGL and Torch PCP Binary, four collectors,
  resumed a first-update checkpoint within a three-update epoch. Model, active
  Adam, logs, recorder/milestone state, counts, retained episode order and all
  collector Python/NumPy/Torch RNG states matched uninterrupted execution
  exactly. Backend/source inheritance and selected checkpoint immutability
  passed. Four segments executed ten new updates, 480 steps and 80 episodes;
  do not double-count retained prefixes. `continuation_validation.json` and its
  298-file artifact manifest bind the evidence.
- Production-shaped functional checks: eight updates (four workloads × two
  backends), four collectors/floor 500/H80 or H300, **18,766 steps / 206 episodes**.
  Paired counts, episode summaries, initial identities, source manifests and final
  RNG matched; model/Adam comparisons passed. Maximum model difference was
  4.44e-15. Complete action/observation traces were not recorded, so equal episode
  summaries are not called identical trajectories. `production_shape_02.json`
  and `production_shape_02_numerical_comparison.json` contain exact commands.
- Both SoftRole paper-FC variants additionally trained eight steps/two episodes/
  one update each and completed one frozen evaluation each. Model and RMSprop
  states were finite and changed; archived simulator bytes matched HetNet;
  evaluation checkpoint bytes were unchanged. The first production attempt
  completed an extra PP DGL update (2,194 steps/32 episodes) before a harness
  error requested resume eligibility for an exhausted epoch budget. That error
  and its completed valid checkpoint were preserved and independently rechecked;
  no runtime fix was needed. Including the extra run and SoftRole checks, this
  validation tranche executed 11 updates/20,976 steps/242 episodes plus two frozen
  evaluations. Its artifact manifest hashes 1,901 files.
- Twelve real four-collector ledgers passed the new benchmark validator and
  correctly required replay. **39 actual batch-script/CLI dry runs** with mocked
  module/srun commands passed without training outputs or submissions. Shell
  syntax, `git diff --check` and `uv lock --check` passed; no dependencies changed.
- Runtime provenance was refreshed twice, preserving both reviews. Final
  ORIGINS SHA256 is
  `d79894723efbf40812232e53e5cf684eacecbdf8916ad4012fb1cfe20410589a`.
  The final prepared study and benchmark plans carry source/dependency identities
  and the common 500-scenario FC panel; they do not authorize or submit jobs.

Remaining scientific/operational gates: run the paired compute-node benchmark,
review resource evidence and fixed-rollout comparisons, then run the separate
official 100-update preflights before choosing a backend for fresh research runs.
No new Stokes speedup or learned-performance result has been measured. Earlier
Mac timings concern a different model/learner. No full sweep, cluster job, commit
or push occurred. Automatic paper-specific final checkpoint selection/manifests
remain unimplemented; the README gives the predeclared first-saved-at-28M rule
and working manual frozen-evaluation commands using the shared panel. Do not use
the legacy `prepare-evaluation` route for paper-v1 or pool unmatched old FC runs.

### 2026-10-05 — Stokes backend benchmark analysis and rollout recommendation

- Analyzed the newly copied `stokes_runs/hetnet_backend_benchmark_904547/` and
  matching stdout/stderr without modifying inputs. Worktree started clean at
  `a178989`; that user commit adds `sapphirerapids` only to the benchmark script.
  The benchmark completed 24 unprofiled runs (480 updates, 1,074,342 steps,
  12,330 episodes), eight separate diagnostic runs (24 updates, 52,100 steps,
  610 episodes), and four replay fixtures (26,050 steps, 305 episodes).
- Added [the evidence-backed report and next steps](analysis/stokes_backend_2026-10-05/report.md).
  Root analysis recomputes update/segment/process rates from raw ledgers and
  checks recorded results, fixed protocols, pairing and both benchmark gates.
  Independent reviews cover actual checkpoint tensors and provenance, replay
  fixtures/assertion coverage, and raw hardware/resource/phase observations.
- Measured median paired post-first-update speedups: PP Real **1.377×**, PCP Real
  **1.351×**, FC Real **1.401×**, PCP Binary **1.282×**. All twelve individual
  pairs improve. Median DGL→Torch rates are 366.3→506.0, 203.8→275.3,
  212.7→298.3 and 155.8→199.6 steps/s. Corresponding update-time reductions
  are 27.4%, 26.0%, 28.6% and 22.0%. Rates and paired ratios are separately
  defined; repetitions use the same seed and are not independent learning seeds
  or a confidence interval. Full-segment median ratios also pass all gates.
- All 68 recorded source identities match the reviewed checkout; ORIGINS validates
  43 runtime files and all twelve direct dependency pins match the Stokes runtime.
  All 32 runs share the same 53-file archived-source manifest
  `17143ed20265d29e321f96609d4999144a84e748dc0b29ad1bde059c61b49599`.
  Checkpoint bytes/signatures/sidecars, counts/configuration, model and active
  Adam finiteness/shape/dtype/step inventories pass. Three repetitions finish
  bitwise identically within each backend; paired initialization and final RNG
  match. The largest cross-backend model difference after 20 updates is
  1.665e-13 (float64), with exact float32 model parameters. All episode-summary
  ledgers match, but full action/observation trajectories were not recorded.
  Sequential order is supported by matching stdout and hash-verified blocking
  subprocess control flow; independent absolute process timestamps were not saved.
- Fixed-rollout median speedups: PP Real 1.491×, PCP Real 1.422×, FC Real 1.479×,
  PCP Binary 1.334×. All four correctness gates and all twelve timing pairs pass.
  Independently loaded/hash-checked all fixtures; finite data/parameters/Adam,
  legal actions, four collector partitions, floor 500, production horizons,
  initial identities and three active optimizer steps reconcile. The reported
  Stokes float64 drift is at most 7.28e-12, below atol 1e-10; float32 drift at most
  9.74e-15. Checked archived fail-closed validation source and log receipts;
  did not rerun numerical kernels on the Mac. Replay includes RNG restoration
  and Python iteration, excludes environment/action sampling/IPC, and is not
  interpreted as end-to-end throughput or per-backend memory evidence.
- Hardware/resource review: ec169, Xeon Gold 5418Y, four distinct cores on NUMA 0,
  affinity `[0,2,4,6]`, four CPUs/16 GiB and one Torch thread in all 128 collector
  intervals. In the 24 unprofiled runs aggregate CPU utilization is 3.41–3.66 cores; system CPU
  is under 0.8%. Summed individual peak RSS is 3.46–6.10 GiB DGL / 3.60–6.41 GiB
  Torch; these are neither simultaneous nor cgroup peaks, and shared pages can
  be double-counted. No recorded evidence identifies resource pressure as the
  main limit; no swap/fault/pressure counters were available. Keep 16 GiB.
- Phase evidence: eliminating graph construction and reducing forward computation
  accounts for 87.8–88.6% of the measured collector-phase reduction. Backward is
  now the largest measured Torch collector phase. Collector wall intervals overlap
  and must not be combined with parent phases as elapsed-time components. Parent
  waiting (9.2–25.2% of update wall) includes residual collector imbalance, not
  just IPC. Checkpoint time is 0.024–0.226% of segment wall. First PP DGL startup
  was 146.68 s versus later 10.58–11.41 s; its whole-process ratio is unsuitable
  as the backend gain and the startup cause is unmeasured.
- Recommendation: proceed to all four official paper-v1/Torch 100-update preflights
  with explicit `sbatch --constraint=sapphirerapids`. The preflight and training
  scripts currently lack that benchmark constraint. The early-rate projection is
  about 52 active minutes for the serial preflight array, excluding queueing and
  probes. If they pass, freeze the implementation and start the fresh seed 0–2
  study plus matched paper-FC SoftRole runs. Retain DGL as the reference and keep
  old FC and supplement-v1/Binary results separate; the historical slowdown is
  not causally explained by this different-model/hardware comparison.
- Scheduling: unchanged per-seed budgets project to 22.0 h PP Real, 40.3 h PCP
  Real, 26.1 h FC Real and 55.5 h PCP Binary with Torch at median segment rates.
  Plan Binary continuation after the existing 46-hour checkpointed pause; do not
  cut its 40M budget or restart a seed. Carry node constraint and explicit core
  binding into the continuation (the existing resume script omits explicit
  `--cpu-bind=cores`). The twelve HetNet runs project to 431.4 job-hours versus
  578.2 for DGL, about six idealized days at three continuously running jobs,
  excluding queueing, rate changes, SoftRole training and evaluation. These are
  planning estimates, not completion guarantees or learned-performance claims.
- Remaining orchestration: prepare the paper-specific FC checkpoint selection
  manifest and commands using the first saved checkpoint at or above 28M steps while training proceeds;
  retain the common 500-scenario panel and separate independent seed reporting.
  Existing manual frozen evaluation is available; the legacy automatic selector
  is not suitable for paper-v1. No new runtime bug or validity blocker was found
  in these reviewed artifacts; the official preflight remains required.
- Files touched: `AGENTS.md` (this additive record), `.gitignore` (narrow exception),
  and fresh `analysis/stokes_backend_2026-10-05/`: `analyze.py`, `report.md`,
  `analysis_console.json`, `metrics/{summary.json,inputs.json,throughput.csv}`;
  `provenance_review/{audit.py,README.md,review.json,input_hashes.json}`;
  `hardware_review/{analyze.py,results.json,verification.json,artifact_manifest.json}`;
  `replay_review/{audit.py,review.json}`; final validation/artifact manifests.
  Auditors verified 3,118 provenance/checkpoint inputs and 171 resource inputs;
  replay fixture/source/log hashes are recorded separately. All inputs remained
  unchanged. Analysis scripts and arithmetic checks passed; no runtime test suite
  was rerun because no implementation changed. No training, frozen policy
  evaluation, cluster submission, dependency/launcher modification, commit or push.

### 2026-10-05 — completed Stokes paper-preflight audit and exact launch runbook

- Analyzed both newly copied groups, `hetnet_paper_preflight_905118` and
  `hetnet_paper_preflight_905404` (the latter is the actual directory matching
  the user's abbreviated `90540`). Both use `paper-v1` model/environment,
  `paper-equations-v1`, `torch-v1`, seed 991, four collectors, floor 500 and
  production horizons. All eight complete the official 100-update and strict
  frozen-checkpoint gates: 800 updates, 1,793,520 steps and 20,318 training
  episodes, plus eight separate frozen execution probes. No new generic
  preflight is needed before the fresh, locked research wave.
- Added [the analysis and exact next steps](analysis/stokes_preflight_2026-10-05/report.md).
  Root calculations reconcile raw update/segment times, projections, windows
  and counts. Independent audits inspect actual checkpoints/source/probes,
  ordered episode and learner records, and process CPU/memory evidence.
  Model, active Adam, all four collector RNG streams, every epoch signature,
  all numerical update/episode ledgers and frozen episodes match bitwise
  between groups after excluding only declared clocks. Intermediate gradients
  and full action trajectories were not saved and were not recomputed.
- All eight 53-file source manifests are identical to benchmark 904547 and
  the reviewed reconstruction: `17143ed20265d29e321f96609d4999144a84e748dc0b29ad1bde059c61b49599`.
  Current ORIGINS validates 43 runtime files. Installed package inventories,
  lock files, initial model/Adam/RNG and first-20 numerical update prefixes
  match the benchmark. All model/active optimizer tensors, losses and recorded
  metrics are finite; active Adam steps are 100. Frozen receipts bind actual
  checkpoint bytes, immutable tensors, scenario bytes and evaluator archives.
- Measured update rates, 905118 -> 905404: PP Real 335.2 -> 342.1 steps/s;
  PCP Real 190.0 -> 263.5; FC Real 191.1 -> 292.4; PCP Binary 128.4 -> 197.4.
  These are same-backend runtime differences, not new backend speedups. No
  hostname, CPU model/topology or actual Slurm node/constraint identity was
  saved; CPU affinity numbers cannot establish Sapphire Rapids placement.
  CPU cost per step rises with slower wall times. Aggregate CPU utilization
  is 3.385–3.600 cores; system CPU 0.405–0.686%; checkpointing only
  0.009–0.0385% of segment time. Summed individual lifetime peak RSS is
  3.630–6.477 GiB, not a concurrent/job peak; memory-growth/pressure data are
  absent. Keep four CPUs and 16 GiB.
- Complete-segment rate projections for unchanged per-seed budgets are
  PP 32.7–33.3 h (40M), PCP Real 42.3–58.6 h (40M), FC 26.7–40.8 h (28M),
  Binary 56.5–86.8 h (40M). Twelve HetNet runs imply 474.3–658.7 job-hours,
  or 6.6–9.1 idealized days at three continuously occupied HetNet slots.
  These early-rate scenarios exclude startup, probes, queueing, future rate
  changes, SoftRole training and evaluation; they are not confidence bounds.
  PP's previous 22-hour benchmark projection is not supported by these longer
  preflights. Binary needs checkpointed continuation after the existing
  46-hour pause; PCP Real may also need one. Do not shorten scientific budgets.
- Found no blocking numerical/learner defect. Reporting cautions: legacy
  checkpoint `log` losses remain per-step, while structured update/epoch
  losses correctly use the paper completed-episode mean; conversions reconcile.
  Legacy entropy zeros are placeholders. All 800 updates trigger the declared
  .75 clip. FC's large finite summed-time losses/norms and positive failed-probe
  return are compatible with its reward/objective, not by themselves evidence
  of a bug. Keep scientific settings fixed rather than tuning to seed 991.
- Prepared concrete source-check, study-preparation, two-array launch,
  historical accounting and HetNet continuation commands, without executing
  submissions. Explicitly set Torch (launcher defaults to DGL), request
  `sapphirerapids`, use fresh seeds 0–2 and one prepared root, and retain the
  matched six paper-FC SoftRole runs. Existing seed-991 preflights exhausted
  their engineering budgets and must not become research continuations.
  The two research arrays each allow three jobs (up to six combined).
- Operational recovery distinction: HetNet resumes its exact parent source
  archive and has a 46-hour graceful pause. SoftRole FC has no wall-time/signal
  pause handler and saves every ten epochs / 100 updates; a timeout can leave
  an unsaved suffix or incomplete checkpoint write. Its array launcher rejects
  `--resume`; the separate documented `softrole train --resume` recipe requires
  a fully validated checkpoint/signature and fresh output. SoftRole executes
  current learner/model code, enforcing simulator identity but not automatic
  full learner-source equality, so its source must be checked and frozen.
- Preserve the declared Binary64-per-head/256-total interpretation and old FC
  separation. Prepare paper-specific final checkpoint selection/manifests
  while training proceeds: first saved FC checkpoint at or above28M along the
  retained lineage, actual hashes/counts and a common500-scenario panel, before
  viewing outcomes. Existing manual evaluation works; legacy automatic
  `prepare-evaluation` is not the paper-v1 route. Preflight repeats/probe
  successes are not independent seeds, convergence or exact-publication proof.
- Files touched: `AGENTS.md` (this additive record), `.gitignore` (narrow
  exception), and fresh `analysis/stokes_preflight_2026-10-05/`: `report.md`,
  `analyze.py`, `metrics/{summary.json,input_hashes.json,throughput.csv,update_windows.csv}`;
  `provenance_review/{audit.py,README.md,review.json,input_hashes.json,softrole_recovery.md}`;
  `learner_review/{audit.py,review.json}`;
  `resource_review/{analyze.py,results.json,verification.json,artifact_manifest.json}`;
  plus validation/artifact manifests and resolved study dry-run evidence.
  Preserved all pre-existing benchmark-analysis and documentation changes.
- Validation: root calculations reproduce identical outputs; independent
  reviews checked timing, learner interpretation and source/launch/recovery
  details. Input hashes are rechecked and original artifacts unchanged.
  Shell blocks, local links, source audit, 12+6 resolved dry-run protocols and
  whitespace are checked in the accompanying validation record. No numerical
  training test suite was rerun for this analysis-only change. No training,
  policy evaluation, input/checkpoint mutation, launcher/dependency change,
  Slurm submission, commit or push occurred.
