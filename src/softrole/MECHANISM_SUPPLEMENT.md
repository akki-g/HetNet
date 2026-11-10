# Capability-mechanism supplement: `capcon_hyper` training launcher

This runbook covers the locked **training** stage of one supplementary arm,
`capcon_hyper`, of the capability-mechanism comparison: 2 conditions x 1 arm x 5
seeds = 10 fixed 10M-step runs. It is a supplement to the 70-run
`capability_mechanism_study_v2` (see [MECHANISM_STUDY.md](MECHANISM_STUDY.md), the
parent study) and runs under its own locked launcher, `softrole.mechanism_supplement`,
and its own manifest, `capability_mechanism_supplement_v1`. Implementation and tests
authorize no cluster jobs; the commands below are for the user to run after
deploying the reviewed source.

Declared 2026-11-10, after the parent study's training array was launched and before
any frozen evaluation was run. The parent study's configurations, manifest and runs
are not changed by this supplement.

## The arm

`capcon_hyper` represents agent ability the fourth way the literature does, weights
generated from capabilities by a hypernetwork (CASH), but on the same actor body as
the parent study's `capcon`, `capcon_class_input` and `class_typed` arms. The other
arms differ in how ability reaches the body (capability input, class input,
class-specific weights); this one differs in where it enters: the head.

- **Body.** The shared CapCon actor with its capability input removed: the
  observation without its two capability entries goes through `pre`, then the LSTM
  cell on `[ReLU(pre(obs)), u_bar (if feedback), one-hot previous action]`, then the
  same `comm_rounds` rounds of attention communication (one expert, an all-ones
  gate, independent heads, null-degree correction, `comm_off` behaviour) as the
  shared actor. The LSTM state, messages and null attention therefore do not depend
  on the capability entries.
- **Head.** There is no output layer. Two hypernetworks (`HyperNetwork`, the class
  `cash` uses) read the agent's **full** observation, capability entries included,
  and emit the per-agent output weights `W_i` (`head_dim x 6`) and biases `b_i`;
  `logits_i = u_i W_i + b_i` with `u` the final-round feature. `hyper_b` starts at
  zero and `hyper_w` is orthogonally initialised, as in `cash`. Capture and stay
  masking, the decomposed critic on the LSTM state and the action mask's `kappa`
  are as in the other arms.
- **Matched information.** Teammate capabilities are not an input to the
  hypernetworks (the same rule as `cash`): an agent's capability entries reach only
  its own logits.

How it differs from `capcon`: `capcon` feeds the capability entries to `pre` and
has a plain linear output layer, so capabilities shape the recurrent state, the
messages and the logits. `capcon_hyper` removes them from `pre` and replaces the
output layer by the hypernetwork head, so they reach the logits only. With a
constant hypernetwork map equal to a shared output layer it is exactly the
`physical_only` shared actor (a unit test checks this).

How it differs from `cash`: `cash` is a smaller, different body (a GRU on the
physical observation, no communication, hidden size 64 with a 64 x 6 head);
`capcon_hyper` has the LSTM and attention-communication body of the CapCon actor
and a `head_dim x 6` head on the final-round feature. It exchanges messages, so
it has the three `comm_off` evaluation cells (see
[MECHANISM_EVALUATION.md](MECHANISM_EVALUATION.md)).

### Parameters

Actor-only parameter counts with the study settings (`pre_dim` 128, `hidden_dim` 64,
four independent heads, `head_dim` 16, `msg_dim` 64, three communication rounds,
feedback on, `hyper_dim` 64, vision 2); the critic is the shared one and is not
counted.

| Map | `obs_dim` | `capcon_hyper` | body | `hyper_w` | `hyper_b` | `capcon` | `cash` |
|---|---|---|---|---|---|---|---|
| 5x5 (`s5`) | 702 | 340,466 | 226,444 | 59,936 | 54,086 | 226,802 | 202,566 |
| 6x6 (`s6`) | 977 | 410,866 | 261,644 | 77,536 | 71,686 | 262,002 | 255,366 |

## Conditions, arm and seeds

| Condition | Map `dim` |
|---|---|
| `s5` | 5 |
| `s6` | 6 |

| Arm | `Config.model` | `Config.descriptor` |
|---|---|---|
| `capcon_hyper` | `capcon_hyper` | `capability` |

Seeds are 0, 1, 2, 3 and 4. The target is 10,000,000 joint environment steps per
run, as in the parent study.

## Array index map

`index = condition_i * 5 + seed_i`, with `condition_i` the position in (`s5`, `s6`)
and `seed_i` the seed. Indices 0-4 are `s5` seeds 0-4 and 5-9 are `s6` seeds 0-4;
for example index 0 is `s5`/seed 0, index 7 is `s6`/seed 2 and index 9 is
`s6`/seed 4. The output of index `i` is `<condition>/capcon_hyper/seed<k>`.

## Locked settings

Every configuration is the parent study's `capcon` configuration for the same
condition and seed (`mechanism_study.study_config`) with `model` replaced by
`capcon_hyper`; a guard rejects any other difference. Everything else is therefore
the parent study's: Adam at 0.001, decomposed critic with beta 1, real channel,
`entropy_coeff` 0.01, null-degree correction, `hidden_dim` 64, `hyper_dim` 64,
four collectors, horizon 80, graded-observation-v2 with the same training and
held-out sensing levels and compositions. Execution is CPU float64 with one Torch
thread per collector. The stopping rule, checkpoint rule (the first committed
checkpoint at or above the target, never a best epoch) and episode and checkpoint
logging are the parent study's. The protocol (the parent study's protocol fields
with this arm list and its own purpose text, plus `parent_study` and the declaration above), the ten
configurations and the source file inventory are hashed into the manifest. A
parent-study manifest is rejected by this launcher and a supplement manifest is
rejected by `softrole.mechanism_study`.

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
  --title 'Capability mechanism supplement training' \
  --purpose 'Supplementary training: capcon_hyper arm, s5/s6 maps, seeds 0-4, 10M steps each.')

supp_prep_job=$(.venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
  slurm/softrole_mechanism_supp_prepare.sbatch) && \
  .venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
    --dependency="afterok:${supp_prep_job%%;*}" --kill-on-invalid-dep=yes \
    slurm/softrole_mechanism_supp_train.sbatch
```

Preview the submission routing without creating a run directory or submitting:

```bash
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'Capability mechanism supplement training' \
  --purpose 'Preview preparation routing.' -- slurm/softrole_mechanism_supp_prepare.sbatch
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'Capability mechanism supplement training' \
  --purpose 'Preview training routing.' -- slurm/softrole_mechanism_supp_train.sbatch
```

Preparation requests one CPU, 2 GB and ten minutes. It creates `manifest.json` and
dry-runs all 10 entries without collecting episodes or creating trainer leaf
directories. An existing manifest is validated, never rewritten; a source mismatch
fails preparation and prevents the dependent array. The training array uses indices
0-9 (`--array=0-9%10`, all ten tasks may run at once). Each task requests four CPUs,
16 GB and 12 hours on `normal`/`sapphirerapids`, account `cenyioha`, the parent
study's per-task request. These are unmeasured starting values, not completion
forecasts.

`SOFTROLE_MECHANISM_SUPP_MANIFEST` and `SOFTROLE_MECHANISM_SUPP_RUN_ROOT` default to
`$HETNET_RUN_DIR/manifest.json` and `$HETNET_RUN_DIR/training`; they are distinct
from the parent study's `SOFTROLE_MECHANISM_MANIFEST` and
`SOFTROLE_MECHANISM_RUN_ROOT`, which these scripts do not read. Unset any prior
exports or point them inside this container. Preparation accepts no arguments and
training accepts only an optional `--dry-run`; budget, seed, model, condition,
collector, manifest and resume overrides are rejected. The launcher checks the
source hash before and after training and refuses an existing output directory. The
training script refuses a manifest or run root that resolves outside
`$HETNET_RUN_DIR` (including through a symlink) before running any Python, and the
launcher refuses an output directory that resolves outside the run root or is a
symlink. Preserve partial outputs from an interrupted task; this fresh-only launcher
does not implement continuation, and outputs must not be deleted to make a retry
pass.

## Output layout

```text
runs/<timestamp>_capability-mechanism-supplement-training_<id>/
  README.md
  run_info.json
  submissions/                          # commands, responses and job IDs
  logs/softrole-mechanism-supp-prepare-<job>_<task>.{out,err}
  logs/softrole-mechanism-supp-train-<array job>_<index>.{out,err}
  manifest.json
  training/s5/capcon_hyper/seed0/       # source, config, metrics, checkpoints
  training/s5/capcon_hyper/seed1/
  ...
  training/s6/capcon_hyper/seed4/
```

Each leaf directory gets `study.json` only after training finishes and the final
source check passes. Preserve this directory name when copying results locally.

## Scope

This stage produces training runs only. Frozen evaluation of these checkpoints is a
separate stage on the same panels as the parent study (see
[MECHANISM_EVALUATION.md](MECHANISM_EVALUATION.md)); this launcher performs no
evaluation and selects no checkpoint. The exploratory local pilots that preceded the
parent study are not evidence for it.
