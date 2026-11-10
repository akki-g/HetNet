# Capability-mechanism study: frozen evaluation

This runbook covers the **evaluation** stage of the capability-mechanism comparison:
the 70 checkpoints from the [training stage](MECHANISM_STUDY.md) (2 conditions x 7
arms x 5 seeds) are each evaluated, without further training, on fixed scenario
panels, a sensor-failure panel with a matched no-failure (sham) control, and
message-off interventions. Implementation and tests authorize no cluster jobs; the
commands below are for the user to run after deploying the reviewed source.

The protocol is locked in code (`softrole.mechanism_evaluation`) and hashed into
the evaluation manifest as `capability_mechanism_evaluation_v1`. Panels, cells,
the checkpoint rule and the metric definitions below are declared before any
checkpoint is evaluated.

## Panels

Every panel uses vision 2, `GradedScenario` records and 500 scenarios per
composition. The scenario seed of (panel, composition) is
`SeedSequence([4100, panel_index, num_p, num_a]).generate_state(1)[0]`, with
`panel_index` the zero-based position in the table, passed to `make_scenarios`.
Panels are identical for every arm, seed and condition: scenarios do not depend on
the map size, so one set of panel files is shared by `s5` and `s6`.

| Panel | Compositions (P, A) | P sensing levels | Sensor failure | Scenarios |
|---|---|---|---|---|
| `in_dist` | (1,1) (1,2) (2,1) (2,2) (3,1) | {1, 3} | none | 2500 |
| `heldout_composition` | (3,2) | {1, 3} | none | 500 |
| `larger_teams` | (4,1) (4,2) | {1, 3} | none | 1000 |
| `heldout_level` | (1,1) (1,2) (2,1) (2,2) (3,1) | {2} | none | 2500 |
| `failure` | (2,1) (2,2) (3,1) | {1, 3} | probability 1; event step uniform in 5..15; victim uniform over the initially sensing agents | 1500 |

A agents have sensing level 0 in every panel. The training distribution uses
P sensing levels {1, 3} and the first five compositions; level 2, composition
(3,2) and the 4-P compositions are held out from training.

## Cells

One frozen-evaluation report is written per cell per checkpoint, with the
`evaluate_checkpoint` interventions below.

| Cell | Panel | Intervention | Sham | Arms |
|---|---|---|---|---|
| `in_dist`, `heldout_composition`, `larger_teams`, `heldout_level` | same name | none | no | all |
| `failure` | `failure` | none | no | all |
| `sham` | `failure` | none | yes | all |
| `in_dist_comm_off` | `in_dist` | `comm_off` from step 0 | no | messaging arms |
| `failure_comm_off` | `failure` | `comm_off`, starting at the sensor-failure event | no | messaging arms |
| `sham_comm_off` | `failure` | `comm_off`, starting at the scheduled event step | yes | messaging arms |

A sham keeps the scheduled event step but suppresses the sensor loss, so its
scenarios are those of the `failure` cell. The messaging arms are `capcon`,
`capcon_class_input`, `class_typed` and `ca_cc_gnn`: 9 cells each. `ca_mlp`,
`rnn_exp` and `cash` exchange no messages and skip the three `comm_off` cells: 6
cells each. The manifest records the cells and the skipped cells of every entry.

## Checkpoint rule

For each training run (`training/<condition>/<arm>/seed<k>/`) the evaluated
checkpoint is the **first committed checkpoint whose `total_steps` is at or above
the training target** (10,000,000), found with `ledger.committed_checkpoints`; never
a best epoch. The run must also contain `finished.json` and `study.json`
(`study.json` must carry the training manifest's hash and source hash and this run's index), the
checkpoint's `source_sha256` must equal the training manifest's source hash, and its
saved configuration must equal the training manifest entry. The manifest records the
checkpoint path relative to the training run directory, its sha256, total steps, epoch
and `source_sha256`.

A run that is absent or incomplete is recorded with `status: "missing"` and a
reason, never dropped. `prepare` fails on any missing run unless `--allow-missing` is
given; a missing run's array task does nothing but print the skip. A source-hash or
configuration disagreement is always an error.

## Index map

The array index is the training index, `index = condition_i * 35 + arm_i * 5 +
seed_i` (see the training runbook): 0-34 are `s5`, 35-69 are `s6`.

## Run the evaluation

Deploy the reviewed commit to `~/HetNet` first: the evaluation source hash covers
the batch scripts and `scripts/submit_stokes.py`, so changing them after
preparation makes every launch refuse the manifest. Keep the checkout and
environment fixed from preparation until every array task ends. The evaluation gets
its own fresh run directory; the finished training run directory is only read.

```bash
cd ~/HetNet
squeue -u "$USER"
export SOFTROLE_MECHANISM_TRAINING_RUN="$PWD/runs/<training run directory>"   # must exist
eval_run_dir=$(.venv/bin/python scripts/submit_stokes.py --init-only \
  --title 'Capability mechanism study evaluation' \
  --purpose 'Frozen evaluation of the 70 mechanism-study checkpoints on fixed panels, sham and comm-off cells.')

eval_prep_job=$(.venv/bin/python scripts/submit_stokes.py --run-dir "$eval_run_dir" -- \
  slurm/softrole_mechanism_eval_prepare.sbatch) && \
  .venv/bin/python scripts/submit_stokes.py --run-dir "$eval_run_dir" -- \
    --dependency="afterok:${eval_prep_job%%;*}" --kill-on-invalid-dep=yes \
    slurm/softrole_mechanism_eval.sbatch
```

If some training runs are knowingly absent, add `--allow-missing` after the
preparation script name (`... -- slurm/softrole_mechanism_eval_prepare.sbatch
--allow-missing`). Preview the submission routing without creating a run directory
or submitting:

```bash
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'Capability mechanism study evaluation' \
  --purpose 'Preview preparation routing.' -- slurm/softrole_mechanism_eval_prepare.sbatch
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'Capability mechanism study evaluation' \
  --purpose 'Preview evaluation routing.' -- slurm/softrole_mechanism_eval.sbatch
```

Preparation requests one CPU, 4 GB and two hours. It selects and verifies every
checkpoint, writes the panels and `evaluation/manifest.json`, then dry-runs all 70
entries (each re-checks the evaluator source hash, re-hashes the checkpoint and the
panels, and validates any existing reports). An existing `evaluation/` directory is
validated, never rewritten. The array uses indices 0-69 with at most twenty tasks
running at once (`--array=0-69%20`); each task requests one CPU, 8 GB and 6 hours on
`normal`/`sapphirerapids`, account `cenyioha`. These are initial allocations, not
measured runtimes. Re-run a single interrupted index with `-- --array=<index>
slurm/softrole_mechanism_eval.sbatch` in the same `--run-dir`.

`SOFTROLE_MECHANISM_TRAINING_RUN` must point at an existing directory other than the
evaluation run directory (it holds `manifest.json` and `training/`). It is a
read-only input and is not routed by the submission helper. `SOFTROLE_MECHANISM_EVAL_DIR`
defaults to `$HETNET_RUN_DIR/evaluation`; unset prior exports or point it inside the
container. The scripts refuse an evaluation directory that resolves outside
`$HETNET_RUN_DIR` (including through a symlink) before running any Python, and the
launcher refuses report paths that escape it or are symlinks. Preparation accepts
only an optional `--allow-missing`; the array accepts only an optional `--dry-run`;
panel, episode, cell, seed and checkpoint overrides are rejected.

**Re-runs and existing reports.** A task writes `reports/<condition>/<arm>/seed<k>/<cell>.json`
with `evaluate_checkpoint`, which refuses to overwrite. A report that already
exists is kept only if it validates against the manifest (checkpoint sha256,
scenarios sha256, intervention, sham, episode count, evaluator source hash,
scenario IDs and intervention step); otherwise the task fails without touching it.
An interrupted task can therefore be resubmitted: complete cells are kept and only
absent cells are evaluated. Do not delete reports to make a retry pass; a truncated
report left by a killed task must be moved aside by hand.

## Summarize

`summarize` is pure post-processing: it executes no model and checks no checkpoint
or source hash, but it imports the evaluator module and therefore PyTorch, so do not
run it on the Stokes login node. Run it after every array task has ended, inside a
short allocation, or on a copy of the run directory (preserve its relative name)
on a machine with the pinned environment:

```bash
cd ~/HetNet
module load anaconda/anaconda-2024.10
srun --account=cenyioha --partition=normal --constraint=sapphirerapids \
  --cpus-per-task=1 --mem=8G --time=00:30:00 env PYTHONPATH="$PWD/src:$PWD/src/envs" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m softrole.mechanism_evaluation summarize \
    --manifest "$eval_run_dir/evaluation/manifest.json" --output "$eval_run_dir/summary"
# locally, from a copy:
PYTHONPATH=src:src/envs .venv/bin/python -m softrole.mechanism_evaluation summarize \
  --manifest runs/<evaluation run directory>/evaluation/manifest.json \
  --output runs/<evaluation run directory>/summary
```

`--output` is a fresh directory (an existing one is refused). Every report is
validated against the manifest first. Missing training runs and missing reports are
listed explicitly in `summary.json` (`missing`); `summarize` fails unless
`--allow-missing` is given.

### Tables

`cells.csv` has one row per (condition, arm, seed, cell, composition):
`episodes`, `success_rate`, `capped_completion` (mean steps, failures counted as the
horizon of 80), `mean_team_return`, and for the cells on the `failure` panel
(`failure`, `sham`, `failure_comm_off`, `sham_comm_off`) `event_exposed_episodes`
(episodes in which the sensor loss was applied; zero in sham cells by construction).

`paired.csv` has one row per (condition, arm, seed, comparison, composition) plus
one pooled row per comparison (`pooled = True`, `num_p`/`num_a` empty, all
compositions of the panel). Each row averages, within one seed, the per-scenario
differences (minuend minus subtrahend) of success (0/1) and capped completion over
the scenarios matched by `scenario_id` (`pairs`). The comparisons are:

| Comparison | Minuend | Subtrahend | Arms |
|---|---|---|---|
| `failure_minus_sham` | `failure` | `sham` | all |
| `in_dist_comm_off_minus_in_dist` | `in_dist_comm_off` | `in_dist` | messaging |
| `failure_comm_off_minus_failure` | `failure_comm_off` | `failure` | messaging |
| `sham_comm_off_minus_sham` | `sham_comm_off` | `sham` | messaging |

Both cells of a comparison run the same scenarios, so a difference compares one
condition against another on identical episodes: the sensor loss against its
suppressed schedule, or messaging switched off from the stated step against intact
messaging. A negative capped-completion difference means fewer steps (faster
completion). `summary.json` holds both tables, the missing list and the manifest
hash. No bootstrap, significance test or arm ranking is computed here.

## Output layout

```text
runs/<timestamp>_capability-mechanism-study-evaluation_<id>/
  README.md
  run_info.json
  submissions/
  logs/softrole-mechanism-eval-prepare-<job>_<task>.{out,err}
  logs/softrole-mechanism-eval-<array job>_<index>.{out,err}
  evaluation/
    manifest.json                       # protocol, 70 entries, hashes, evaluator source
    panels/{in_dist,heldout_composition,larger_teams,heldout_level,failure}.json
    reports/s5/capcon/seed0/{in_dist,...,sham_comm_off}.json
    reports/s5/ca_mlp/seed0/...         # six cells for message-free arms
    ...
  summary/                              # after `summarize`
    cells.csv  paired.csv  summary.json
```

Each report records the checkpoint and its sha256, the evaluator identity and
source hash, the complete scenario records and the per-episode outcomes. Preserve
this directory name when copying results locally.

## Scope

This stage evaluates fixed checkpoints and writes tables of rates, completion times
and paired differences. It trains nothing and chooses no checkpoint but the first at
the target. The panels are declared here before any evaluation; they are not to be
tuned on results. The exploratory local pilots that preceded the study are not
evidence for it.
