# Stale-Capability Study — Implementation Design

Oct 26, 2026 · @Akshat Guduru

## Purpose and scope

Implement opt-in support for the stale-capability study on top of commit `3e946b7` of `akki-g/HetNet` ("repo cleanup"): a graded observation without the sensing-level leak, a declared capability that can differ from the physical one, paired frozen evaluation, and locked launchers for a stability gate and the main study.

**Research question.** When a capability-conditioned policy is told a sensing level different from its physical level, does it act on the declaration or on the evidence? Can training with unreliable declarations make it robust, and at what cost?

**Baseline.** All paths below are under `src/softrole/` unless stated. Line numbers refer to `3e946b7`. The numerical core (`config.py`, `env.py`, `model.py`, `rollout.py`, `scenarios.py`, `learning.py`, `evaluate.py`, `report.py`, `optimizers.py`) is byte-identical to `4b9fd61`.

**Terms used in this doc.**

- *Level*: integer sensing level 0–3 at vision 2. Level L sees cells at Chebyshev distance < L (0 = blind, 1 = own cell, 2 = radius 1, 3 = radius 2).
- *Physical capability* κ = (level / (vision + 1), capture). It drives observation masking, environment action legality and failure events.
- *Declared capability*: the two capability entries at the end of the actor's observation (the "tail"). Today they always equal the descriptor of the physical capability.
- *The leak*: in `corrected-observation-v1` the off-grid "outside" flag is masked by visibility while position one-hots are not, so the observation reveals the agent's own level in 24 of 25 positions.
- *Stale cost* S: horizon-capped completion steps with an over-declared tail minus the same quantity with a correct tail, on identical scenarios and the same checkpoint.

This doc covers changes 0 and B–F. The implementer submits no training or evaluation jobs.

## Ground rules for the implementer

Every change is opt-in, and existing behaviour stays bit-identical. These rules restate `AGENTS.md`; where this doc and `AGENTS.md` disagree, this doc wins for the study and Change 0 brings `AGENTS.md` into line.

1. **Defaults unchanged.** Existing binary and graded-pilot configurations must produce identical scenario records, scenario hashes, observations, random streams, episodes and manifests. New fields are omitted from `to_dict` at their defaults, following `entropy_coeff` (`config.py:110–111`).
2. **Do not touch** `model.py`, `learning.py`, the simulator reward, the optimizer, `detach_gap`, or how random streams are derived, except where a section below says so.
3. **Execution contract:** CPU float64, one Torch thread per collector, the pinned Python 3.12 `.venv` from `uv.lock`.
4. **Tests for every change** (see Required tests). Run the full `pytest` suite before declaring done; record the pass count and time.
5. **No cluster jobs.** Validate launchers with `--dry-run` only, including `scripts/submit_stokes.py --dry-run`.
6. **Immutable evidence.** Do not modify archived sources, existing manifests, `old_logs/`, `runs/` or `analysis/`.
7. **Ask, don't invent.** Values marked *proposed* or *TBD* need the user's confirmation. If a design point here proves wrong in the code, stop and report rather than improvise a protocol change.

## Change 0: documentation

Do this first: later agents read `AGENTS.md`, and two of its current rules contradict this design.

**`AGENTS.md`**

- *Active experiment and authority* (the paragraphs naming Amendment A-2 and the G-cap/G-type pilot as the current direction): replace with the stale-capability study, linking this design, the new runbook and Amendment A-3. Keep one sentence recording the pilot as completed.
- Boundary bullet "Physical sensing/capture capabilities drive observation masking, action legality and failure" is inaccurate. Replace with: physical κ drives masking, environment-side legality (`env.py:258–259`) and failure; the model's action mask (`model.py:364`) uses whatever κ the rollout passes, which is physical for `descriptor="capability"` and the class descriptor for `descriptor="type"`.
- Boundary bullet "The descriptor enters both actor and critic" remains true of the pilot only. Add: in the stale-capability study the actor receives the declared capability through the observation tail, while the model's κ argument (critic and action mask) is physical.
- Add a bullet defining `graded-observation-v2` (Change B).

**`README.md`**

- *Current experiment* section: replace with a short description of the study and links; move the pilot to one "completed pilot" sentence.
- *Protocol and documentation index*: add the new runbook.

**New runbook** `src/softrole/STALE_CAPABILITY.md`, modelled on `GRADED_PILOT.md`: arms, budgets, source locking, the exact `submit_stokes.py` commands for the gate, the main study and evaluation, and output layout. Add `!/src/softrole/STALE_CAPABILITY.md` to `.gitignore` beside the existing `GRADED_PILOT.md` entry (line 45), since `*.md` is ignored.

**Amendment A-3** (`docs/`) is the user's scientific document. Do not write hypotheses or claims into it.

## Change B: graded-observation-v2 (close the leak)

Add a new environment version, `graded-observation-v2`, in which the outside flag is never masked; everything else matches `corrected-observation-v1`.

**Why.** `adapt_observation` (`env.py:36–44`) multiplies all three sensory channels (outside, target, occupancy) by the visibility mask, but never masks the position one-hots. Off-grid cells have an all-zero position code, so a masked outside flag on an off-grid cell exposes the agent's own level. The outside flag adds no world information, because the position codes already determine it. Measured worst case (empty window, constant tail): levels 1 vs 3 differ in 24 of 25 positions, 2 vs 3 in 24, 1 vs 2 in 16.

**Implementation**

1. `env.py` near line 12: add `GRADED_ENVIRONMENT_VERSION = "graded-observation-v2"`. Leave `ENVIRONMENT_VERSION` and `__init__.py:3` unchanged.
2. `adapt_observation(raw_obs, kappa, base, visible=None, tail=None, mask_outside=True)`: when `visible` is given and `mask_outside` is false, apply `visible` to the target and occupancy channels only and pass the outside channel through. The binary path (`visible is None`) is unchanged.
3. `EnvironmentAdapter.__init__`:
   - accept the new version alongside `corrected-observation-v1` and `paper-v1` (lines 69–71);
   - require `capabilities == "graded"` and `task == "pcp"` for v2 (extend lines 78–79);
   - build the native simulator for v2 exactly as for v1 (line 86 branch).
4. `observe` (lines 200–210), graded path: pass `mask_outside=(self.environment_version != GRADED_ENVIRONMENT_VERSION)`.
5. `config.py`: allow the new value at line 123; at lines 166–168 let graded capabilities use v1 or v2, and reject v2 with binary capabilities.
6. `__main__.py:31`: add the new version to the `--env-version` choices.
7. Leave the explicit `corrected-observation-v1` rejections in `diagnostics.py:131` and `diagnostic_archive.py:62, 140` as they are: those tools stay binary-only.

**Unchanged by design**

- The observation width and the model's parameter count. The channel is kept, only unmasked, so v1 and v2 arms share one architecture.
- `pcp_sensor_status` (lines 224–247).
- Checkpoints already record `environment_version`, and `evaluate.py:82–83` refuses a mismatched adapter. Add a test that a v2 checkpoint cannot be evaluated under v1.

**Expected result.** Under v2, with an empty window, levels 1, 2 and 3 give identical observations at every position; level 0 still differs through own-cell occupancy. Under v1 the 24/16/24 pattern is unchanged.

## Change C: declared vs physical capability

A scenario may carry a declared level per agent; the actor sees it in its observation tail, while everything else keeps using the physical level.

**Semantics**

- Physical levels stay in `scenario.sensing`, exactly as now.
- `declared` empty: the tail equals the physical descriptor at every step, including after failure events. This is today's behaviour.
- `declared` non-empty: the tail's sensing entry is fixed to `declared / (vision + 1)` for the whole episode and does not follow failure events. The tail's capture entry always equals physical capture.
- A agents' declared level must equal their physical level; this study only over-declares P agents.

**Routing**

| Consumer | Receives | Where |
| --- | --- | --- |
| Observation masking, environment legality, failure events | physical | `env.py` `_visibility`, `step`; `rollout.py:151–156` |
| Actor (last two observation entries) | declared | `env.py` `observe` tail |
| Model κ argument: action mask, critic | physical | `rollout.py:177–184`, unchanged for `descriptor="capability"` |
| Episode metrics | both | `rollout.py:260–261` |

**Allowed combinations.** Reject a non-empty `declared` unless `capabilities == "graded"` and `descriptor == "capability"`. Also reject it when `model` is `banked` or `capability`: those gates read κ (`model.py:299–308`) and would see the physical value. With `actor_capabilities="physical_only"` a declaration is accepted but cannot reach the actor (the tail is stripped); tests rely on that invariance.

**Implementation**

1. `scenarios.py`: add `@dataclass(frozen=True) class DeclaredScenario(GradedScenario)` with `declared: tuple = ()`. A subclass keeps every existing `GradedScenario` and `Scenario` record and hash unchanged, as `GradedScenario` did for binary scenarios. In `scenario_from_record` (lines 67–73), route records containing `"declared"` to `DeclaredScenario`, normalizing with `_sensing_levels`.
2. `make_scenarios` (lines 76–142): add keywords `overstate_prob=0.0` and `overstate_level=None` (meaning `vision + 1`).
   - When `overstate_prob > 0` (graded only), spawn 7 children instead of 6 (line 120) and seed a declaration RNG from `children[6]`.
   - For each P agent independently, set its declared level to `overstate_level` with probability `overstate_prob`, otherwise to its physical level. A agents always declare their physical level.
   - Emit `DeclaredScenario` for every scenario when `overstate_prob > 0`, even when the declaration equals the physical levels.
   - Do not change any draw from children 0–5. `SeedSequence.spawn` keys children by index, so `spawn(7)[:6]` equals `spawn(6)`; a test must confirm this.
3. `env.py` `reset` (lines 164–185): add `declared=None`. Validate shape, integer range `[0, vision + 1]`, A-agent equality and the allowed combinations above; store `self.declared_levels` or `None`.
4. `env.py` `observe`, graded path: after computing `tail = self.descriptor(kappa)`, if `declared_levels` is set, copy it and overwrite column 0 with `declared_levels / (vision + 1)`. Do not apply the declaration in `_validate_kappa`.
5. `rollout.py`:
   - `validate_scenario` (lines 62–100): validate `DeclaredScenario` lengths and ranges against the config.
   - `reset` call (lines 123–126): pass `declared=scenario.declared` for a non-empty `DeclaredScenario`.
   - metrics (lines 260–261): add `episode.metrics["declared"] = list(scenario.declared)` for `DeclaredScenario`.

**No change to `model.py`.** For `descriptor="capability"` the rollout already passes the physical κ to the model, and the shared actor reads capability only through the observation tail.

## Change D: configuration

Add two training-time fields that control over-declaration; both default to off and are invisible in existing configuration dictionaries.

1. `Config` fields (beside the graded fields at `config.py:44–49`): `overstate_prob: float = 0.0` and `overstate_level: int | None = None`.
2. `to_dict` (lines 107–118): delete both keys when at their defaults, like `entropy_coeff` and the graded fields, so all existing configuration hashes and manifests are unchanged.
3. `validate`:
   - `overstate_prob` is a finite real in \[0, 1\] and not a bool;
   - `overstate_prob > 0` requires `capabilities == "graded"`, `descriptor == "capability"` and `model` in (`shared`, `constant`);
   - `overstate_level`, when set, is an integer in \[1, vision + 1\];
   - the environment-version rules from Change B.
4. `train.py` `collect_batch` (lines 125–128): pass `overstate_prob` and `overstate_level` to `make_scenarios` only when `overstate_prob > 0`, keeping the default call unchanged.
5. `model_kwargs` is unaffected; no model layout changes.

Evaluation panels set declarations explicitly (Change E) and do not use these fields.

## Change E: paired evaluation

The study's primary measurement is the stale cost S, computed within each checkpoint on pairs of scenarios that differ only in the declaration. Reuse `evaluate.evaluate_checkpoint` unchanged; add panels, protocol fields and one paired contrast.

**1. Panels** (new functions in the study module, Change F)

- Compositions (1,1) and (2,1); physical P1 level ℓ ∈ {1, 2, 3}; declaration ∈ {correct, overstated}.
- For each composition, build base scenarios with `make_scenarios(composition_seed(panel_seed, c), episodes, [c])`, reusing `programme_evaluation.composition_seed`.
- Physical sensing: P1 (index 0) at ℓ, any other P agent at `vision + 1`, A agents at 0.
- Both conditions use `DeclaredScenario`. Correct: `declared == sensing`. Overstated: P1 declared `vision + 1`, everyone else as physical.
- Scenario IDs `"{panel_seed}:{c}:{ℓ}:{index}"` and all env, action and message seeds are identical across the two declarations, so outcomes pair by ID.
- 500 episodes per (composition, ℓ, declaration), matching `EPISODES_PER_COMPOSITION`.
- Panel seeds (proposed): 3100 for development, 3101 for the final evaluation. Both are unused by earlier studies (2704, 2711, 2700/2701, 2800/2801). Record them in the manifest.
- Write each panel as JSON records readable by `scenario_from_record`, with its SHA-256.

**2. Protocol schema** (`report.py:21–77`)

- Add optional distribution fields `sensing` (per-agent list), `declared` (per-agent list) and `declaration` (`"correct"` or `"overstated"`).
- When present, check every scenario row against them. When absent, behaviour is unchanged.
- The existing event and victim checks are unaffected because these panels have no events.

**3. Summaries** (`report.py`)

- In `summarize_reports` (lines 367–376), add `sensing` and `declaration` to the stratum identity only for episodes that carry them, so existing identities are unchanged.
- Add a contrast `overstated_minus_correct`, a sibling of `_paired_summaries` (lines 149–197) keyed on `declaration` instead of `sham`. Keep its checks: same checkpoint within each seed, identical complete panel IDs, identical `initial_state` digests, per-seed metric deltas, whole-seed bootstrap via `_uncertainty`. Leave `failure_minus_sham` untouched.
- Cross-arm comparisons such as S(A2) − S(A1) come from `contrast_groups` (lines 274–312) without changes.
- Invariance strata: at ℓ = `vision + 1`, and for any `actor_capabilities="physical_only"` checkpoint at every ℓ, the paired delta must be exactly zero. The summarizer reports a non-zero delta there as an error, not as a result.

**4. Task runner**

- A task list and Slurm array modelled on `programme_evaluation.prepare_evaluation` and `run_task`, writing one report per (checkpoint, composition, ℓ, declaration).
- Checkpoint selection: `first_saved_at_or_above` the training budget. Never best-epoch.
- One exploratory condition: `comm_off` from step 0 on the same panels, labelled exploratory in the protocol.

## Change F: locked launchers

One new module, `src/softrole/stale_capability.py`, locks the stability gate, the main study and the evaluation, following the pattern of `graded_pilot.py`.

**Module contents**

1. Constants: `VERSION`, a study name, the arm table and seeds from *Arm and stability-gate configurations*, and a `PROTOCOL` dict (purpose, primary contrasts, budgets, checkpoint rule, panel seeds).
2. `study_config(arm, seed)` and `gate_config(coefficient, seed)`:
   - derive from `programme_config("l1b_decomposed_beta1", seed, 1e-3, profile=CAPCON_PROFILE)` with `dataclasses.replace`;
   - copy the changed-field guard from `graded_pilot.py:42–47`, so any field outside the declared set raises.
3. `prepare`, `validate_manifest` and `launch`, as in `graded_pilot.py:60–116`:
   - one manifest each for the gate and the main study;
   - source lock at prepare and at launch, re-checked after training;
   - refuse existing outputs and protocol overrides;
   - `--dry-run` support.
4. `gate-summary` command: read each gate run's `updates.jsonl` and episode ledger, apply the gate rule, and write a JSON verdict per coefficient. It decides nothing by itself; the user confirms the chosen coefficient.
5. Panel construction and evaluation task-list preparation from Change E.

**Slurm and submission**

- New files: `slurm/softrole_stale_prepare.sbatch`, `slurm/softrole_stale_train.sbatch` (array over manifest entries) and `slurm/softrole_stale_evaluate.sbatch`.
- Mirror `softrole_graded_pilot*.sbatch`:
  - source `scripts/stokes_run.sh`;
  - default outputs under `$HETNET_RUN_DIR` (`manifest.json`, `training/<arm>/seed<N>/`, `evaluation/`);
  - `PYTHONPATH=src:src/envs`;
  - 4 CPUs per training task, 1 per evaluation task;
  - mark resources as unmeasured starting values.
- Register the new script names in `scripts/submit_stokes.py` `ENV_ROUTES` (lines 23–45) and `RUN_ROOT_OUTPUT_JOBS` (lines 53–56), as the pilot is.
- Confirm in a test that `source_snapshot` (`train.py:54`) picks up the new module and sbatch files. It globs `src/softrole` and `slurm/`, so no change should be needed.

## Arm and stability-gate configurations

All runs start from `programme_config("l1b_decomposed_beta1", seed, 1e-3, profile=CAPCON_PROFILE)` and change only the fields below; the guard in Change F enforces this.

**Common changes**

| Field | Value |
| --- | --- |
| `capabilities` | `"graded"` |
| `descriptor` | `"capability"` |
| `compositions` | `((1, 1), (2, 1))` |
| `sensing_p` | `(1, 2, 3)` |
| `sensing_a` | `(0,)` |
| `held_out_sensing_p`, `held_out_sensing_a` | `()` |
| `failure_prob` | `0.0` |
| `entropy_coeff` | from the stability gate (TBD) |
| `total_steps` | `10_000_000` (proposed) |

**Main-study arms** (seeds 0–4, 20 runs)

| Arm | `env_version` | `actor_capabilities` | `overstate_prob` |
| --- | --- | --- | --- |
| A1 reliable | `graded-observation-v2` | `explicit` | 0 |
| A2 unreliable | `graded-observation-v2` | `explicit` | 0.25 (proposed) |
| A3 undeclared | `graded-observation-v2` | `physical_only` | 0 |
| A1-open | `corrected-observation-v1` | `explicit` | 0 |

A3 has 256 fewer actor parameters (the two dropped tail inputs to the 128-unit `pre` layer). Every other arm has the same 239,795-parameter model.

**Stability gate** (6 runs)

- A1 settings with `total_steps=2_000_000`, `entropy_coeff` ∈ {0.003, 0.01} (proposed), seeds 0–2.
- A seed *collapses* if, after 1M steps, any 0.5M-step window has success below 5%, or any update's mean action entropy falls below 0.01 nats per agent-step. Entropy comes from `action_entropy_sum` in `updates.jsonl`, divided by agent-steps (proposed thresholds).
- Choose the smallest coefficient with no collapsed seed. If both coefficients collapse, stop and report; do not try further values without the user.

## Required tests

Each change ships with tests in new files named `tests/test_softrole_stale_*.py`; existing tests must keep passing unmodified.

| # | Test | Covers |
| --- | --- | --- |
| 1 | v1 binary and graded observations, scenarios and episodes are bit-identical to `3e946b7` behaviour | B, C, D defaults |
| 2 | v2, empty 5×5 window: levels 1, 2, 3 give identical tail-stripped observations at all 25 positions; level 0 differs | B |
| 3 | v1 regression: levels 1 vs 3 differ at 24/25 positions, 2 vs 3 at 24/25, 1 vs 2 at 16/25 | B |
| 4 | v2 is rejected with binary capabilities, non-PCP tasks and `paper-v1` combinations, in both `Config` and the adapter; a v2 checkpoint cannot be evaluated under v1 | B |
| 5 | With a declaration, the tail's sensing entry equals `declared / (vision + 1)` while visibility follows physical levels; without one, output is unchanged | C |
| 6 | After a failure event, a declared tail stays fixed while visibility drops | C |
| 7 | With `overstate_prob > 0`, env/action/message seeds, compositions, sensing and events equal the `overstate_prob = 0` draws; `spawn(7)[:6] == spawn(6)` | C, D |
| 8 | Overstatement frequency for P agents matches `overstate_prob` within a binomial bound; A agents never overstate; prefixes are stable as count grows | C |
| 9 | `DeclaredScenario` round-trips through records; `Scenario` and `GradedScenario` records and hashes match fixtures | C |
| 10 | Routing: a spy on `SoftRoleNet.forward` receives physical κ; the actor observation tail carries the declaration | C |
| 11 | A3 invariance: with `physical_only`, declared and undeclared episodes are bit-identical | C |
| 12 | Full-level invariance: over-declared and correct episodes are bit-identical when P1 is physically at `vision + 1` | C, E |
| 13 | Declarations are rejected with `descriptor="type"`, `banked` or `capability` models, and binary capabilities | C, D |
| 14 | Panels: identical IDs and seeds across declarations; counts per stratum; stable hashes | E |
| 15 | Protocol schema validates the new fields; existing protocols still validate | E |
| 16 | `overstated_minus_correct` computes per-seed deltas, refuses mismatched checkpoints or panels, and errors on non-zero deltas in invariance strata; `failure_minus_sham` output on existing fixtures is unchanged | E |
| 17 | Manifest locks: changed-field guard, source lock, refusal of existing outputs and overrides, dry-run, `submit_stokes.py` routing (pattern: `test_softrole_graded_pilot_launcher.py`) | F |
| 18 | `gate-summary` applies the gate rule correctly to synthetic ledgers | F |

**Local smoke check** (not a research run): for each arm and one gate configuration, train 2 updates locally with one collector and a small batch. Confirm finite weights and optimizer state, then record commands and outcomes as the B1 validation did.

## Order of work and acceptance

Work in this order so the stability gate can be launched (by the user) as soon as Change B lands, while the rest is built.

1. Change 0 (documentation).
2. Change B with tests 2–4, then the gate parts of Change F (gate manifest, `gate-summary`, train sbatch) with tests 17–18. **Stop and hand over:** the gate needs only these.
3. Changes C and D with tests 5–13.
4. Change E with tests 14–16.
5. The rest of Change F (main-study manifest, evaluation task list and sbatch).
6. Test 1, the full `pytest` suite, the local smoke check, and `submit_stokes.py --dry-run` for every new sbatch.

**Done means all of these hold:**

- The full suite passes in the pinned environment; pass count and time are recorded.
- Existing binary and graded-pilot configurations reproduce bit-identical scenarios, observations and episodes.
- New behaviour is reachable only through new config fields or the new environment version.
- Gate and main-study manifests validate under dry-run; no jobs were submitted.
- `AGENTS.md`, `README.md` and the new runbook describe the active study.
- A short implementation note lists changed files, new tests and every deviation from this doc.

**Ask the user before** fixing anything marked proposed or TBD: entropy coefficients and gate thresholds, `total_steps`, `overstate_prob`, panel seeds.

## Out of scope

The implementer must not build any of the following; each is a separate decision for the user.

- **Mid-episode partial failures.** The rollout always drops the victim to level 0 (`rollout.py:154`), and three rules require two sensing agents (`rollout.py:94–95`, `scenarios.py:107–116`, `config.py:214–222`).
- **The gated-experts arm** (`model="capability"`): its gate reads κ and would need the declaration routed to it.
- **Other capabilities or environments:** transmit range, speed, capture probability, larger grids, other simulators.
- **Changes to** `model.py`, `learning.py`, the reward, the optimizer or `detach_gap`.
- **Running anything on Stokes**, and analysing results.
- **Writing Amendment A-3's** hypotheses or scientific claims.
