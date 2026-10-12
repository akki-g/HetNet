# CapCom proposal programme: implementation and evidence record (12 October 2026)

Base: commit `644f8f1`, the source executed by the running S1 and D1–D6 jobs.
Everything below is opt-in; the defaults reproduce that source bitwise, and the
numerical-source guards of the diagnostics are unchanged. No research run was
executed: the S1 rate selection, manifest locking, submission and evaluation are
the next steps on Stokes. Mathematics: [RESEARCH.md §11](../../softrole/RESEARCH.md).
Commands: [README](../../README.md#capcom-proposal-programme-c1-a1-l1bl1a-l1c-o1-12-october-2026).
Labels: (I) identity, (R) result under stated assumptions, (C) code fact,
(E) engineering execution, (H) hypothesis the programme tests.

## 1. Claim, derivation, code, test, manifest, records, conclusion

### C1 channels (`--communication real_noisy | binary_deterministic`)

- Claim (H): Real's advantage over Binary comes from one of message noise,
  quantization, surrogate bias or capacity. The two added channels split the
  factors: `real_noisy` adds the Binary channel's logistic noise to the Real
  payload with an exact gradient; `binary_deterministic` keeps quantization and
  removes noise, changing the surrogate from σ'(u+ε) to σ'(u).
- Derivation (I): `real_noisy` sends p = u + ε, ∂p/∂u = 1; `binary_deterministic`
  sends 1[u>0] with backward σ'(u); `binary` is unchanged (1[u+ε>0], σ'(u+ε));
  `real` unchanged. Only stochastic Binary has a score function, so D5 applies
  to it alone.
- Code (C): `softrole/model.py` `channel_payload`, `deterministic_bits`,
  `logistic_noise_like`, `NOISE_CHANNELS`, `BIT_CHANNELS`; `softrole/rollout.py`
  keyed noise for `binary` and `real_noisy`, logical payload bits;
  `softrole/train.py` `channel_approximations`; `softrole/config.py` `CHANNELS`.
- Tests: `tests/test_softrole_c1_channels.py` (forward values and surrogate
  derivatives against injected noise and analytic σ'(u); exact previous Binary
  formula; logistic marginals; noise rejection by noise-free channels; keyed
  message streams; payload accounting; CLI; exact resume and frozen replay).
- Arms: `c1_real64` (control), `c1_real_noisy64`, `c1_binary64`,
  `c1_binary_deterministic64`, `c1_real16`, `c1_binary16`.
- Records: `run.json` `learner.communication` and `approximations`;
  checkpoint `model_config.communication`; episode `communication`,
  `payload_value_bits`.
- Conclusion: the channel semantics and derivatives are verified (I/E); no
  statement about learning speed exists until the arms run.

### A1 degree-corrected null attention (`--null-degree-correction`)

- Claim (H): a roster-invariant message gain helps composition transfer.
- Derivation (I): with d permitted senders and equal scores s, the baseline null
  share is e^ν/(e^ν + d e^s); adding log(max(d,1)/2) makes it e^ν/(e^ν + 2e^s),
  independent of d. d = 0 keeps the exact zero aggregate. The −log 2 is the
  proposal's corrected-null initialization folded into the offset so that the
  nominal two-sender case is bitwise identical to the baseline; a corrected
  model at degree d equals a baseline whose null parameter is shifted by log(d/2).
- Code (C): `CommunicationRound(degree_correction=...)`; `SoftRoleNet(null_degree_correction=...)`;
  `Config.null_degree_correction`; `model_kwargs` omits the default.
- Tests: `tests/test_softrole_a1_null_degree.py` (degree invariance at three
  null values, directed in-degrees, exact zero at degree zero, bitwise
  identity at degree two, reparameterization equivalence at degrees one and
  three, gradient flow, parameter count, CLI, composition training, held-out
  evaluation, resume guard).
- Arms: `a1_composition_original_null` (control), `a1_composition_corrected_null`.
- Conclusion: the identities hold (I/E); whether invariance helps transfer is
  the arm's question (H).

### L1b decomposed critic (`--critic-mode decomposed`)

- Claim (H): per-agent values with per-agent λ-return targets give a stronger
  value objective and enable credit mixing.
- Derivation (I): v_jt = ψ([H_jt; κ_j; φ̄_t; N; Σκ; (T−t)/T]), V_t = Σ_j v_jt;
  per-agent GAE with zero terminal bootstrap; Σ_j Â_jt = Â_t by linearity;
  L_V = (1/τ) Σ_t Σ_j (v_jt − R̂_jt)². A sum-only value loss leaves the split
  unidentified: values (a, −a) with zero rewards give zero team loss and 2a²
  per-agent loss.
- Code (C): `softrole/model.py` critic branch (`Linear(134,64)→ReLU→Linear(64,1)`,
  `agent_values` output); `softrole/learning.py` `Episode` per-agent fields,
  `agent_gae`, `episode_loss_terms`; `softrole/rollout.py` per-agent
  log-probabilities, rewards and values recorded without extra draws.
- Tests: `tests/test_softrole_l1b_decomposed_critic.py` (hand-computed
  per-agent GAE, linearity, sum and permutation equivariance of agent values,
  critic isolation, rollout records, β = 1 gradient equality with the team GAE
  on a real episode, cancelling-error supervision, time-mean value loss,
  checkpoint layout, training/evaluation/resume guard).
- Arm: `l1b_decomposed_beta1` versus `c1_real64`.
- Conclusion: at β = 1 the actor gradient equals the production team gradient
  (I/E); the value objective differs by design and is the factor under test.

### L1a credit mixing (`--credit-beta β`, decomposed critic only)

- Claim (R): β < 1 trades variance for a bias equal to the dropped cross-agent
  credit (1−β) E[Σ_t Σ_j ∇log π_jt Σ_{i≠j} Â_it]; in PCP that term is nonzero
  during search.
- Derivation (I): in the enumerated two-agent example, where agent 1's action
  rewards only agent 0, the estimator at β equals β times the exact gradient.
- Code (C): `episode_loss_terms` weights β Σ_i Â_it + (1−β) Â_jt;
  `Config.credit_beta` with the decomposed-critic requirement; run record lists
  `credit mixing: beta<1 drops cross-agent credit`.
- Tests: `tests/test_softrole_l1a_credit.py` (enumeration at β ∈ {1, .5, 0},
  hand-computed mixed weights, own-credit gradient at β = 0, validation of β,
  CLI coupling, run record and resume guard).
- Arms: `l1a_decomposed_beta05`, `l1a_decomposed_beta0` versus `l1b_decomposed_beta1`.
- Conclusion: the bias structure is exact (I); its size in PCP is empirical (H).

### L1c mean-agent objective (`--objective mean_agent`)

- Claim (R): dividing each complete episode loss by N_e optimizes
  E_c[J_team(c)/N_c]; it is a declared objective, not a correction, and it does
  not equalize gradient magnitudes across compositions.
- Code (C): `episode_loss_terms` returns `scale = 1/N`; `compute_episode_loss`
  applies it to the whole episode loss; the learner's single global average and
  single clip are unchanged (`softrole/train.py` `apply_gradients`).
- Tests: `tests/test_softrole_l1c_objective.py` (unequal rosters weighted
  1/N with one global sum, roster-size requirement, collector gradient sums
  equal Σ_e (1/N_e) ∇L_e on real mixed-composition episodes, CLI, run record,
  resume guard).
- Arms: `l1c_composition_mean_agent`, `l1c_failure_mean_agent`.
- Conclusion: implemented as declared (I/E).

### O1 physical-only actor input (`--actor-capabilities physical_only`)

- Claim (R/H): κ_sense is a function of the other observation entries, so
  removing κ from the actor removes no information but one shortcut; whether
  the shared recurrence then converges to A-like behaviour after sensor loss is
  a hypothesis.
- Code (C): `SoftRoleNet.actor_input_dim = obs_dim − 2`, actor slice, banked gate
  on H only, mask and critic keep κ, capability-gated model rejected.
- Tests: `tests/test_softrole_o1_actor_capabilities.py` (bitwise equality of
  common-action logits, messages and memory under κ changes while the mask and
  value change; banked gate independence; contradiction rejection;
  permutation equivariance; layout; blind P and A share the actor input;
  failure training, failure/sham evaluation and resume guard).
- Arm: `o1_physical_only` versus `c1_real64`.

### Defaults unchanged

`tests/test_softrole_programme_defaults.py` builds the current `SoftRoleNet` and
a byte copy of `softrole/model.py` at `644f8f1`
(`tests/fixtures/softrole_model_pre_programme.py`, SHA256
`8cb989455cdba6c61f43f3abd8f8c1ba6f37ef784c14aed2a2c957a172e3fa94`) on the
same machine and compares state dictionaries, forward outputs and gradients
bitwise for seven configurations; the team loss builder is compared bitwise
with the previous arithmetic; the default `model_kwargs` layout is unchanged.
Resume rejects any change to the new settings (each proposal's test).

### Diagnostics version 2

- Claim (C): D1–D6 extend to composition- and failure-trained checkpoints and
  the learner variants without changing version 1.
- Code: `softrole/diagnostics.py` (`PROTOCOL_VERSIONS`, `learner_identity`,
  `training_study`, version-aware `load_checkpoint`/`prepare`/`execute`, D1
  from `episode_loss_terms`, per-episode compositions, D6 training-support
  record); `softrole/diagnostic_estimators.py` (`learned_gae_beta_*` variants,
  `agent_critic_episode_diagnostics`, `summarize_agent_critic_diagnostics`,
  production variant and episode scale); `softrole/diagnostic_surrogate.py`
  (D5 only for stochastic Binary); `softrole/__main__.py` and `train.py`
  training-time diagnostics for any PCP study.
- Tests: `tests/test_softrole_diagnostics_v2.py` (historical next-update
  replay reproduces the production gradient for decomposed/β = .5 composition
  training, banked failure training with the corrected null and Binary bits,
  and physical-only/mean-agent/deterministic bits; learned D3/D4 present only
  for decomposed critics; D5 status by channel; metadata tamper rejection;
  version-1 manifest shape unchanged). The existing diagnostics tests pass
  unchanged.
- Caution (C): the strict numerical-source guard means checkpoints archived
  under `644f8f1` must be diagnosed from a checkout at that commit.

### Locked programme and launchers

- Code: `softrole/programme.py` (arm table, `select-lr`, `prepare`, `launch`,
  `window_statistics`), `scripts/softrole_programme.sh`,
  `slurm/softrole_programme.sbatch`.
- Rule (C): the S1 rate with the lower equal-seed mean horizon-capped
  completion over episodes of complete updates ending in (8M, 12M]; ties
  choose the lower rate; partial windows, missing seeds and foreign
  configurations are refused. `c1_real64` at either rate is exactly the S1
  run configuration (test against the S1 launcher's dry run).
- Tests: `tests/test_softrole_programme.py`, `tests/test_softrole_programme_launcher.py`
  (arm counts and declared factor differences, every entry validates, rule
  semantics on synthetic ledgers, manifest locking and tampering, source-hash
  refusal, bounded launches with provenance records, runtime-only overrides,
  shell and mocked-Slurm mappings).

### Evaluation protocol and paired reporting

- Code: `softrole/programme_evaluation.py` (first-saved-at-or-above-20M
  selection verified against the ledger and the preceding checkpoint; common
  panels for (3,2), (4,1), (4,2); conditions nominal, communication off,
  step-1 failure/sham, failure/sham with communication removed at the event,
  and gate freezes for gated arms; task list and generated Slurm array;
  summaries), `softrole/report.py` (`contrast_groups`,
  `difference_in_differences`).
- Definition (I): with Δ_I,s the within-seed full-panel failure-minus-sham mean
  under intervention I, D_I,s = Δ_I,s − Δ_none,s; equal-seed means and
  whole-seed bootstrap intervals; arm-minus-control contrasts on absolute
  strata, Δ and D_I.
- Freeze certification (C): `tests/test_softrole_programme_evaluation.py`
  shows freeze_affected and freeze_all reproduce the unfrozen trajectory
  exactly for shared and constant actors and change the victim's gate for the
  capability actor; freezes are scheduled for banked and capability arms only.
- HetNet E2/E5 admission (C): the panel files are reusable by
  `publication_reconstruction/evaluation.py`; the gaps are labelled in
  `evaluation_manifest.json` (typed inputs, learner, budget and rate, no health
  input, backend tolerance, selection rule). No HetNet report is produced here.
- Tests: `tests/test_softrole_report_contrasts.py`,
  `tests/test_softrole_programme_evaluation.py` (panels, protocol sidecars,
  selection, end-to-end bounded prepare/run/summarize with six runs).

## 2. Corrections to the review document (11 October)

1. The per-coordinate Adam step bound |Δθ| ≤ η(1−β₁)/√(1−β₂) ≈ 3.16η is the
   sparse-gradient special case of Kingma and Ba, not a bound for arbitrary
   gradient histories. With bias corrections, Cauchy–Schwarz gives
   |m̂_t|/√v̂_t ≤ (1−β₁)/√(1−β₂) · (1−β₁²/β₂)^{−1/2} · √(1−β₂ᵗ)/(1−β₁ᵗ), about
   7.3η asymptotically for β₁ = .9, β₂ = .999, and exactly η at t = 1. The
   programme does not rely on either constant.
2. Var(Â_jt + β B_jt) = σ²[1 + β²(N−1)(1+(N−2)ρ) + 2β(N−1)ρ] describes the
   variance of the actor weight under exchangeability assumptions; it is not a
   bound on the variance of the gradient estimator, which also depends on the
   score vectors and their correlations. D4 measures the empirical SNR instead.

## 3. Verification record

All checks ran on this x86-64 Linux container with the pinned `.venv`
(Torch 2.2.1+cu121 CPU execution, NumPy 1.26.4, pytest 8.3.5), one Torch thread.

- Focused tests: the twelve new files in the README section (124 tests) pass.
- Full suite on the frozen tree: **976 passed, 3 failed in 603 s**. The three
  failures predate this work or are environmental: the publication Slurm
  array-string test (recorded on 8 October), the Binary snapshot test whose
  reference values were captured on a Mac (1-ulp logit differences here,
  also with the unchanged `644f8f1` model), and the evaluator package-version
  assertion (`importlib.metadata` reports `2.2.1` for the CUDA wheel whose
  `torch.__version__` is `2.2.1+cu121`; `evaluate.py` is unchanged). A first
  full run while files were still being edited also showed the programme
  launcher refusing a launch because the checkout hash had changed, which is
  the guard behaving as designed; the rerun above used a fixed tree.
- Bounded validation (`validate.py`, `validation.json`, 51 commands, 297 s,
  checkout `644f8f1` plus the uncommitted programme changes): the arm table;
  six one-update S1-launcher runs, which the research rule refuses as partial
  windows; the rule on synthetic ledgers (tie at 6.0 steps resolved to 3e-4);
  a bounded manifest and the refusal to lock a research manifest from it;
  shell-launcher dry runs and one-update, two-collector launches of entries
  0, 3, 9, 21, 30, 33, 36, 48 and 60 (Real, noisy Real, deterministic bits,
  corrected null, decomposed β = 0, physical-only, banked composition, banked
  failure, failure mean-agent), 1,261 joint steps and 18 episodes in total (the six S1-shaped runs add 480),
  with the two failure-trained entries exposing one sensor event each; the
  rejection of a scientific override and of an existing run; diagnostics
  version 2 in checkpoint-probe mode on entries 30 and 48 (learned D3/D4 and
  production variant `learned_gae_beta_0` for the decomposed checkpoint,
  `team_gae` and in-support step-1 loss for the failure checkpoint; version 1
  refuses both); the bounded evaluation preparer (9 tasks, 186 jobs, 372
  episodes over the three reserved compositions, two scenarios each) and its
  refusal of the research mode; every task, an idempotent rerun that skipped
  all existing reports; the summary (62 strata, 22 failure-minus-sham pairs,
  13 D_I rows and 18 contrasts for (3,2)); and shell syntax of the generated
  array. Six commands were expected refusals and returned the expected
  status. `artifact_manifest.json` hashes 437 artifacts under
  `runs/capcom_programme_validation_20261012/`, excluding archived `source/`
  copies that each run's `source_manifest.json` already fixes (the manifest
  rule was changed to this exclusion after the run and regenerated with
  `--manifest-only`). The source hash recorded by the validation is the tree
  before the AGENTS entry and this section were written.

## 4. Remaining steps (not executed here)

1. When the six S1 runs reach 12M steps, run `select-lr` on Stokes and keep the
   record.
2. Lock the manifest on the final fixed checkout; keep the checkout fixed while
   jobs run; submit the 63-job array.
3. Prepare and submit the evaluation after every run finishes; summarize;
   admit HetNet E2/E5 reports with the labelled gaps.
4. Never evaluate the reserved compositions during development; development
   windows come from training ledgers only.
