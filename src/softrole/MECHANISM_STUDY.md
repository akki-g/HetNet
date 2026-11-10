# Capability-mechanism study: training launcher

This runbook covers the locked **training** stage of the capability-mechanism
comparison: 2 conditions x 7 arms x 5 seeds = 70 fixed 10M-step runs. Evaluation is
a separate declared stage (see the last section). Implementation and tests
authorize no cluster jobs; the commands below are for the user to run after
deploying the reviewed source.

## Conditions, arms and seeds

| Condition | Map `dim` |
|---|---|
| `s5` | 5 |
| `s6` | 6 |

| Arm | `Config.model` | `Config.descriptor` |
|---|---|---|
| `capcon` | `shared` | `capability` |
| `capcon_class_input` | `shared` | `type` |
| `class_typed` | `class_typed` | `type` |
| `ca_mlp` | `ca_mlp` | `capability` |
| `ca_cc_gnn` | `ca_cc_gnn` | `capability` |
| `rnn_exp` | `rnn_exp` | `capability` |
| `cash` | `cash` | `capability` |

Seeds are 0, 1, 2, 3 and 4. The target is 10,000,000 joint environment steps per
run.

### Study version

This is `capability_mechanism_study_v2` (manifest version 2). It differs from v1 in one
setting: `entropy_coeff` is 0.01 instead of 0.003, for every arm. The v1 array was
stopped after 50 of 70 runs because some runs lost all action entropy early in
training and did not recover. v1 manifests are rejected by this launcher, and v1
runs are not part of the comparison.

### Representation arms

Three arms share the CapCon learner, decomposed critic, environment and attention
communication and differ only in how agent class or capability reaches the model:

- **Capability input** (`capcon`): one shared actor. The policy descriptor is the
  physical capability vector; it is the final two entries of the actor
  observation, and the same vector is the `kappa` that the action mask and the
  critic receive.
- **Class input** (`capcon_class_input`): the same shared actor with
  `descriptor="type"`. The descriptor is the constant class vector (perception
  `[1, 0]`, capture `[0, 1]`) in place of the physical capabilities, in both the
  actor's observation tail and the critic and action-mask `kappa`. Physical
  capabilities still determine the observation masking, environment-side action
  legality and sensor failure.
- **Class-specific parameters** (`class_typed`): `descriptor="type"` as in the
  class-input arm, and every actor parameter is indexed by agent class
  (`c = 1` if `kappa[:, 1] > 0`, else `0`; 0 is perception and 1 is capture). Each
  class has its own observation encoder, LSTM cell and output head, and its own
  copy of the communication parameters (self map, sender encoder, receiver
  decoder, attention, score and null logit); the attention round structure,
  `comm_off` behaviour and null-degree correction are those of the shared actor.
  The critic is the shared `CapabilityCritic` on the LSTM state and is not
  class-specific. At 5x5 with vision 2 and these settings its actor has 453,604
  parameters, twice the 226,802 of the shared actor. `class_typed` is a
  class-typed variant of CapCon, not a HetNet reproduction.

The remaining four arms (`ca_mlp`, `ca_cc_gnn`, `rnn_exp`, `cash`) are the
capability-conditioned baseline actors of `softrole.baselines`; they read the
physical capability vector from the observation tail.

## Array index map

`index = condition_i * 35 + arm_i * 5 + seed_i`, with `condition_i` the position in
(`s5`, `s6`), `arm_i` the position in (`capcon`, `capcon_class_input`,
`class_typed`, `ca_mlp`, `ca_cc_gnn`, `rnn_exp`, `cash`) and `seed_i` the seed.
Indices 0-34 are `s5` and 35-69 are `s6`; within a condition, each block of five
consecutive indices is one arm at seeds 0-4. For example, index 7 is
`s5`/`capcon_class_input`/seed 2, index 12 is `s5`/`class_typed`/seed 2 and index
69 is `s6`/`cash`/seed 4.

## Locked settings

Every configuration starts from `programme_config("l1b_decomposed_beta1", seed,
1e-3, profile=CAPCON_PROFILE)` and replaces exactly these fields; a guard rejects
any other difference from that primary arm.

| Field | Value |
|---|---|
| `model` | the arm's model |
| `descriptor` | the arm's descriptor |
| `dim` | the condition's `dim` |
| `vision` | 2 |
| `capabilities` | `graded` |
| `env_version` | `graded-observation-v2` |
| `sensing_p` / `held_out_sensing_p` | `(1, 3)` / `(2,)` |
| `compositions` | `COMPOSITION_TRAIN` |
| `held_out` | `COMPOSITION_TEST + EXTRAPOLATION_TEST` |
| `entropy_coeff` | 0.01 |
| `null_degree_correction` | true |
| `total_steps` | 10,000,000 |

All other settings come from the primary arm unchanged for every arm: Adam at
0.001, decomposed critic with beta 1, real channel, `hidden_dim` 64, `hyper_dim` 64,
four collectors, horizon 80. Within a condition and seed, the seven configurations
differ only in `model` and `descriptor`. Execution is CPU float64 with one Torch thread per
collector. The budget stops at the first complete collector batch reaching the
target, so actual counts can slightly exceed it and are recorded. The checkpoint
rule is the first committed checkpoint at or above the target, never a best epoch.
Complete episode records use indexed gzip chunks, and checkpoints carry commit
markers. The protocol, all 70 configurations and the source file inventory are
hashed into the manifest.

## Submit through Slurm

Deploy the reviewed commit to `~/HetNet` first: the new batch scripts and
submission routing are part of the source snapshot, so adding them after
preparation changes the source hash and makes launches refuse the manifest. Keep
the checkout and environment fixed from preparation until every array task ends.

```bash
cd ~/HetNet
squeue -u "$USER"
run_dir=$(.venv/bin/python scripts/submit_stokes.py --init-only \
  --title 'Capability mechanism study training' \
  --purpose 'Confirmatory training: s5/s6 maps, seven arms, seeds 0-4, 10M steps each.')

mech_prep_job=$(.venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
  slurm/softrole_mechanism_prepare.sbatch) && \
  .venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
    --dependency="afterok:${mech_prep_job%%;*}" --kill-on-invalid-dep=yes \
    slurm/softrole_mechanism_train.sbatch
```

Preview the submission routing without creating a run directory or submitting:

```bash
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'Capability mechanism study training' \
  --purpose 'Preview preparation routing.' -- slurm/softrole_mechanism_prepare.sbatch
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'Capability mechanism study training' \
  --purpose 'Preview training routing.' -- slurm/softrole_mechanism_train.sbatch
```

Preparation requests one CPU, 2 GB and ten minutes. It creates `manifest.json`
and dry-runs all 70 entries without collecting episodes or creating trainer leaf
directories. An existing manifest is validated, never rewritten; a source mismatch
fails preparation and prevents the dependent array. The training array uses
indices 0-69 with at most ten tasks running concurrently (`--array=0-69%10`). Each
task requests four CPUs, 16 GB and 12 hours on `normal`/`sapphirerapids`, account
`cenyioha`. These are unmeasured starting values, not completion forecasts.

`SOFTROLE_MECHANISM_MANIFEST` and `SOFTROLE_MECHANISM_RUN_ROOT` default to
`$HETNET_RUN_DIR/manifest.json` and `$HETNET_RUN_DIR/training`. Unset any prior
exports or point them inside this container. Preparation accepts no arguments and
training accepts only an optional `--dry-run`; budget, seed, model, condition,
collector, manifest and resume overrides are rejected. The launcher checks the
source hash before and after training and refuses an existing output directory.
The training script refuses a manifest or run root that resolves outside
`$HETNET_RUN_DIR` (including through a symlink) before running any Python, and the
launcher refuses an output directory that resolves outside the run root or is a
symlink. Preserve partial outputs from an interrupted task; this fresh-only launcher does
not implement continuation, and outputs must not be deleted to make a retry pass.

## Output layout

```text
runs/<timestamp>_capability-mechanism-study-training_<id>/
  README.md
  run_info.json
  submissions/                          # commands, responses and job IDs
  logs/softrole-mechanism-prepare-<job>_<task>.{out,err}
  logs/softrole-mechanism-train-<array job>_<index>.{out,err}
  manifest.json
  training/s5/capcon/seed0/             # source, config, metrics, checkpoints
  training/s5/capcon/seed1/
  ...
  training/s6/cash/seed4/
```

Each leaf directory gets `study.json` only after training finishes and the final
source check passes. Preserve this directory name when copying results locally.

## Scope

This stage produces training runs only. Held-out panels and frozen evaluation are
a separate stage, with their own declared panels, comparisons and checkpoint
rules, prepared and run after training; this launcher performs no evaluation and
selects no checkpoint. The exploratory local pilots that preceded this study are
not evidence for it.
