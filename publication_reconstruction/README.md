# Historical HetNet reconstruction

This separate source tree reconstructs the **public February 2022 environment
family**, with a runnable copy of the existing two-layer HetNet learner. It is
**not a verified publication training checkout**. The actual source and commands
behind the AAMAS results remain unavailable. The paper's three-layer/Adam model
is a different unresolved specification; this implementation keeps the public
two-layer/RMSprop learner so environment and sensory-path effects can be measured.

The existing `main.py`, `envs/`, `softrole/`, launchers, checkpoints and logs remain
unchanged. The new launcher uses `runs/publication_reconstruction/` by default,
refuses existing output directories, and executes a source archive inside each
new run. Repository edits during training cannot change those archived files.

## Two environment versions

| Version | Purpose | Behavior |
|---|---|---|
| `historical-2022` | Closest runnable public-source reference | February environment behavior, including observation interference and FC phantom-front behavior; only the runtime repairs below. |
| `corrected-v1` | Controlled environment corrections | Independent PCP/FC observation copies; FC front/coordinate repairs described below. |

Both versions restore the **29-entry sensory-cell stride**, which was correct
before the October 2022 regression. They retain typed observations, original
agent actions, individual rewards, per-class GAE, two GAT layers and RMSprop.
They do not use SoftRole's merged channels, gates, team critic or gradient rule.
Real and Binary actors support PP, PCP and FC through the existing local zero-A
compatibility guards.

Environment source: `d57da0717d5564027df7e8ba75614feca8006960`. Learner/runtime
scaffold: local `0ee9ceb38b6133866e9c1db40bed2c6caa3b842d`, including seed-before-
construction, explicit preserved gradient storage, CPU float64 and one Torch
thread per collector. `ORIGINS.json` records each file's source and hashes;
the audit's `runtime_changes.patch` records differences against those sources.

Minimal environment repairs in both versions:

- Replace removed `np.int` aliases with `int`.
- Handle equal hotspot bounds as the requested single coordinate, avoiding the
  historical FC `randint(x,x)` initialization failure.
- Guard an empty FC discovered-fire list before NumPy containment, avoiding a
  modern-runtime broadcast error. Historical nonempty containment is preserved
  only in `historical-2022`.
- Remove FC diagnostic printing that does not affect environment state.

Additional `corrected-v1` choices:

- Copy each P/A sensory window before masking the blind agent's view.
- Preserve a boundary fire's actual source coordinate, with the historical
  stationary boundary behavior; never emit an unfilled zero row as a fire.
- If a proposed fire-front position leaves the grid, retain the previous valid
  front position. This bounded behavior is an explicit reconstruction choice.
- Compare complete coordinates for discovered-fire/source membership rather than
  NumPy's elementwise containment. The underlying individual reward formula stays
  the same.

**FC is still a public-code benchmark reconstruction.** Its original individual
reward formula and ability to extinguish undiscovered fires differ from the
paper's description. These have not been silently replaced with speculative
paper semantics. Corrected boundary behavior is not evidence of what the authors
trained. `historical-2022` intentionally retains known defects for comparison;
do not describe it as a corrected physical benchmark.

## Recipes, budgets and running

Run from the repository root using the existing pinned environment:

```bash
.venv/bin/python -m publication_reconstruction train \
  --task pcp --variant real --seed 0 --recipe june-2022 \
  --env-version historical-2022 --dry-run
```

The dry run prints resolved settings and a structured command without creating
files. Removing `--dry-run` starts training. No research training is launched as
part of implementation validation.

| Recipe/task | Collectors | Epochs | Updates/epoch | Batch floor/collector | Episode horizon | Sample floor |
|---|---:|---:|---:|---:|---:|---:|
| June PP/PCP | 1 | 2000 | 10 | 500 | 80 | 10M |
| October PP/PCP | 4 | 2000 | 10 | 500 | 80 | 40M |
| June/October FC | 4 | 1400 | 10 | 500 | 300 | 28M |

These are dated **postconference sample commands**, not certified study settings.
June PP uses `predator_prey`; October PP uses `predator_capture` with 3P0A. Both
are explicitly given three physical agents, repairing the June command's missing
`--nagents 3`. PP/PCP have radius2 sensing, FC radius1, on 5x5 grids. FC has one
initial fire and reward type3. Learning rate is `1e-4`, detach gap5, gamma1,
GAE lambda.95, local norm clip.75, RMSprop alpha.97/epsilon1e-6. The public
learner's episode averaging and subsequent step denominator remain intact.

To compare the corrected environment under the same recipe:

```bash
.venv/bin/python -m publication_reconstruction train \
  --task pcp --variant real --seed 0 --recipe june-2022 \
  --env-version corrected-v1 --dry-run
```

To specify a sample stopping threshold, add `--max-env-steps 10000000`. The run
ends after the first complete update reaching that threshold, or at the epoch
cap, whichever happens first. Whole episodes and collector batches can overshoot;
the exact counts and theoretical maximum overshoot are recorded. No trajectories
are truncated to force an exact budget. A 10M threshold is a comparison choice,
not a recovered publication budget. Changing collectors changes both samples per
update and update count at a fixed sample budget.
The last epoch record may contain only the updates collected before the threshold;
its epoch number is an index, not a claim that every configured update completed.
The checkpoint's actual update and sample counts identify this partial record.

The launcher supports explicit `--epochs`, `--collectors`, `--updates-per-epoch`,
`--batch-steps`, `--horizon`, `--save-every`, and `--output` for bounded validation
or a preregistered protocol. A small executable smoke example is:

```bash
.venv/bin/python -m publication_reconstruction train \
  --task pp --variant binary --seed 0 --env-version corrected-v1 \
  --epochs 1 --updates-per-epoch 1 --batch-steps 1 --horizon 4 \
  --output runs/publication_smoke_example
```

Every run includes `protocol.json`, resolved args, an environment/package record,
source archive and manifest, true epoch metrics, parameter signatures, checkpoints,
stdout and exit status. Checkpoints additionally contain the environment version,
resolved settings, actual step/episode/update/epoch counts and source-manifest
hash. The launcher does not expose the historical evaluator's missing-panel path
or claim a validated resume workflow. Frozen evaluation on matched scenarios is
still required for performance comparisons.

## What longer training currently shows

Existing SoftRole PP/PCP training windows, episode-weighted within each seed and
then equally weighted across three seeds:

| Model/task | Mean episode length near 10M | Near 20M |
|---|---:|---:|
| PP shared | 4.840 | 4.784 |
| PP banked | 4.874 | 4.797 |
| PCP shared | 10.692 | 7.001 |
| PCP banked | 17.594 | 8.752 |

All twelve seed-level lengths improve. PP shows diminishing gains; PCP improves
substantially beyond10M. This does not diagnose generalization or overfitting:
those require frozen validation checkpoints and a predeclared selection rule.
One and four collectors both imply20,000 updates over2000epochs, despite10M
versus40M sample floors. Existing runs were not stopped, retuned or relabeled.

See the [historical audit](../analysis/upstream_history_2026-09-30/AUDIT.txt),
[author search](../analysis/publication_reconstruction_2026-09-30/author_search/FINDINGS.txt)
and [budget analysis](../analysis/publication_reconstruction_2026-09-30/budgets/FINDINGS.txt).

Validation on30September2026: **220 tests passed**, including all178 existing
regressions. Eighteen tiny training executions covered both environment versions,
allthree domains, Real/Binary policies and representative two-collector runs.
They collected190 total joint steps/48episodes/36updates, verified finite changed
weights and checkpoint/source identities, and exercised stopping in a partial
epoch. These executions do not estimate learned performance. See
[validation evidence](../analysis/publication_reconstruction_2026-09-30/validation.json).
