# 6x6 external anchor: original HetNet and shared-body arms on plain PCP

This runbook covers the locked **training** stage of an external anchor: 4 arms x 3
seeds = 12 fixed 10M-step runs on the 6x6 Predator-Capture-Prey map with one fixed
team of 2 perception agents and 1 capture agent and full sensing. The arms are the
original HetNet system (the paper-v1 reconstruction, PCP Real) and the three shared-body
arms of the capability-mechanism study (`capcon`, `capcon_class_input`, `class_typed`).
The runs are compared on their training logs only; this launcher and its tests contain
no evaluation. They run under their own locked launcher, `softrole.anchor_study`, and
their own manifest, `pcp_6x6_anchor_v1`. Implementation and tests authorize no cluster
jobs; the commands below are for the user to run after deploying the reviewed source.

Declared 2026-11-10, after the parent study's training array was launched and before
any frozen evaluation was run. The parent study (`capability_mechanism_study_v2`, see
[MECHANISM_STUDY.md](MECHANISM_STUDY.md)), its manifest and its runs are not changed
by this study.

## The twelve runs

| Index | Arm | Kind | Seeds |
|---|---|---|---|
| 0-2 | `hetnet_original` | `hetnet` | 0, 1, 2 |
| 3-5 | `capcon` | `softrole` | 0, 1, 2 |
| 6-8 | `capcon_class_input` | `softrole` | 0, 1, 2 |
| 9-11 | `class_typed` | `softrole` | 0, 1, 2 |

`index = arm_i * 3 + seed_i`, with `arm_i` the position in (`hetnet_original`, `capcon`,
`capcon_class_input`, `class_typed`) and `seed_i` the seed; index 0 is `hetnet_original`
seed 0, index 4 is `capcon` seed 1 and index 11 is `class_typed` seed 2. The output of
index `i` is `<run root>/<arm>/seed<k>`. Every run targets 10,000,000 joint environment
steps. All twelve may run at once (`--array=0-11%12`).

### Shared-body arms (indices 3-11)

Each configuration is the parent study's configuration for condition `s6` (map 6x6), the
same arm and the same seed (`mechanism_study.study_config`), with only these three
fields replaced:

| Field | Parent `s6` value | Anchor value |
|---|---|---|
| `compositions` | `((1, 1), (1, 2), (2, 1), (2, 2), (3, 1))` | `((2, 1),)` |
| `sensing_p` | `(1, 3)` | `(3,)` |
| `held_out_sensing_p` | `(2,)` | `()` |

A guard rejects any other difference from the parent configuration. Everything else is
the parent study's: the arm's model and descriptor, Adam at 0.001, decomposed critic
with beta 1, real channel, `entropy_coeff` 0.01, null-degree correction, `hidden_dim`
64, four collectors, horizon 80, `graded-observation-v2`, vision 2, `dim` 6, and
CPU float64 with one Torch thread per collector. The parent's `held_out` compositions
stay in the configuration unchanged; training does not use them. Sensing level 3 is
full sensing at vision 2. A run is trained exactly as the parent study's runs are
(`train` with `episode_log="chunked-gzip"` and `checkpoint_commits=True`), stops at the
first complete collector batch reaching the target, and records its actual counts.
`study.json` is written in the run directory only after training finishes and a final
source check passes.

### Original HetNet (indices 0-2)

The arguments are those of `publication_reconstruction.study.train_arguments(3,
run_root, reconstruction_spec="paper-v1", message_backend="torch-v1")`, the PCP Real
paper-v1 workload of the 5x5 first-wave HetNet study, with exactly these changes:

| Item | 5x5 workload | Anchor |
|---|---|---|
| `--seed` | `0` | the index's seed (0, 1 or 2) |
| `--output` | `<run root>/pcp_real/seed<k>` | `<run root>/hetnet_original/seed<k>/training` |
| `--dim` | absent (the trainer's default, 5) | `6` |
| `--max-env-steps` | `40000000` | `10000000` |
| `--milestones` | `10000000 20000000 30000000 40000000` | `10000000` |

Everything else is unchanged, including `--model-spec paper-v1`, `--env-version
paper-v1`, `--recipe october-2022`, `--collectors 4`, `--batch-steps 500`,
`--updates-per-epoch 10`, `--horizon 80`, `--epochs 2000`, `--save-every 10`,
`--episode-log stdout`, `--reconstruction-spec paper-v1`, `--message-backend torch-v1`
and `--wall-seconds 165600`. The map size is the opt-in `--dim` option of the
`train` sub-command of `publication_reconstruction`; 6 is accepted only for
`--task pcp` with `--reconstruction-spec paper-v1`, and the default 5 leaves every
existing resolved protocol unchanged. The run directory is `<arm dir>/training`
(created by `publication_reconstruction` from an archived copy of its source);
`launch.json` and `launcher_exit.json` are written beside it, in
`<run root>/hetnet_original/seed<k>/`, as `softrole.pcp_additional_seeds` writes them.
Episode records go to the task's Slurm log (`--episode-log stdout`); the run directory
holds `metrics.jsonl`, `updates.jsonl`, checkpoints and the rest of that trainer's
records. The manifest records each entry's `arguments` and resolved protocol with the
run root and interpreter replaced by stand-ins, since it is written before any run
root exists; the launch record and `launch.json` hold the real values. The HetNet
launcher's exit code is the task's exit code. The source hash is checked before this
run starts; the trainer runs from its own archived copy of the source.

### Differences between the rows

The `hetnet_original` rows use the original HetNet system's own learner, observation
and critics, not the study's, so they are not a controlled comparison with the
shared-body rows. The shared-body rows differ among themselves only as the parent
study's arms do (how agent ability reaches the actor).

## Locked settings

The protocol (study name, arms, seeds, target steps, map 6, team (2, 1), full sensing,
parent study, the declaration above, the list of differences and the shared-body
training fields of the parent study's protocol), the twelve entries and the source file
inventory are hashed into the manifest. The launcher validates the manifest, then
checks the current source hash against it before any training. A parent-study or
supplement manifest is rejected by this launcher, and an anchor manifest is rejected by
`softrole.mechanism_study` and `softrole.mechanism_supplement`. Execution fails if the
output directory exists, is a symlink or resolves outside the run root.

## Submit through Slurm

Deploy the reviewed commit to a checkout that no running array is using: the new
batch scripts and submission routing are part of the source snapshot, so changing
them after preparation changes the source hash and makes launches refuse the
manifest. Keep that checkout and its environment fixed from preparation until every
array task ends.

```bash
cd ~/HetNet   # only once no array is running from this checkout
squeue -u "$USER"
run_dir=$(.venv/bin/python scripts/submit_stokes.py --init-only \
  --title 'PCP 6x6 anchor training' \
  --purpose 'External anchor: original HetNet and three shared-body arms on plain 6x6 PCP, one fixed team of 2+1, full sensing, seeds 0-2, 10M steps each.')

anchor_prep_job=$(.venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
  slurm/softrole_anchor_prepare.sbatch) && \
  .venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
    --dependency="afterok:${anchor_prep_job%%;*}" --kill-on-invalid-dep=yes \
    slurm/softrole_anchor_train.sbatch
```

Preview the submission routing without creating a run directory or submitting:

```bash
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'PCP 6x6 anchor training' \
  --purpose 'Preview preparation routing.' -- slurm/softrole_anchor_prepare.sbatch
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'PCP 6x6 anchor training' \
  --purpose 'Preview training routing.' -- slurm/softrole_anchor_train.sbatch
```

Preparation requests one CPU, 2 GB and ten minutes. It creates `manifest.json` and
dry-runs all 12 entries without collecting episodes or creating run directories. An
existing manifest is validated, never rewritten; a source mismatch fails preparation
and prevents the dependent array. The training array uses indices 0-11
(`--array=0-11%12`). Each task requests four CPUs, 16 GB and 48 hours on
`normal`/`sapphirerapids`, account `cenyioha`. These are unmeasured starting values,
not completion forecasts. The jobs export `DGLBACKEND=pytorch` and
`PYTHONNOUSERSITE=1`, as the original-HetNet additional-seed jobs do.

`SOFTROLE_ANCHOR_MANIFEST` and `SOFTROLE_ANCHOR_RUN_ROOT` default to
`$HETNET_RUN_DIR/manifest.json` and `$HETNET_RUN_DIR/training`; they are distinct from
the parent study's and the supplement's variables, which these scripts do not read.
Unset any prior exports or point them inside this container. Preparation accepts no
arguments and training accepts only an optional `--dry-run`; budget, seed, model, map,
collector, manifest and resume overrides are rejected. The training script refuses a
manifest or run root that resolves outside `$HETNET_RUN_DIR` (including through a
symlink) before running any Python. Preserve partial outputs from an interrupted task;
this fresh-only launcher does not implement continuation (including `resume` of the
HetNet rows), and outputs must not be deleted to make a retry pass.

## Output layout

```text
runs/<timestamp>_pcp-6x6-anchor-training_<id>/
  README.md
  run_info.json
  submissions/                          # commands, responses and job IDs
  logs/softrole-anchor-prepare-<job>_<task>.{out,err}
  logs/softrole-anchor-train-<array job>_<index>.{out,err}
  manifest.json
  training/hetnet_original/seed0/launch.json
  training/hetnet_original/seed0/launcher_exit.json
  training/hetnet_original/seed0/training/   # source archive, protocol, metrics, checkpoints
  training/hetnet_original/seed1/ ...
  training/capcon/seed0/                     # source, config, metrics, checkpoints, study.json
  training/capcon_class_input/seed0/
  training/class_typed/seed2/
```

Preserve this directory name when copying results locally.

## Scope

This stage produces training runs only. It performs no evaluation and selects no
checkpoint; the anchor is compared on training logs. The frozen-evaluation stage of the
parent study does not list this study, so it refuses an anchor training directory.
