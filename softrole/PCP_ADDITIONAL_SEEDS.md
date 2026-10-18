# Additional independent PCP seeds: 3, 4, 5

This separate route extends the existing 40M PCP Real comparison. It does not
change `softrole.programme.SEEDS`, the original manifests, existing results,
model/learner defaults, or the interpretation of the seeds-0..2 results.
No job is submitted by Python or by the batch script.

| Array index | Recipe | Fresh seed |
|---|---|---:|
| 0, 1, 2 | HetNet `paper-v1` / PCP Real / `torch-v1` | 3, 4, 5 respectively |
| 3, 4, 5 | CapCon `l1a_decomposed_beta0` / Real64 | 3, 4, 5 respectively |
| 6, 7, 8 | Optional CapCon `l1a_decomposed_beta05` / Real64 | 3, 4, 5 respectively |

The first six runs are the main comparison. The final three are a separately
selectable stability check. Every run starts with zero steps/episodes/updates
and a fresh model/optimizer. No checkpoint is loaded and no original seed is
remapped inside a collector. Equal seed labels across systems are not paired
training trajectories.

## Direct submission

First deploy the reviewed source files to the existing frozen Stokes checkout
and retain its existing admitted CapCon preparation and pinned `.venv`. Source
deployment/publication is a separate user action; no commit, push or transfer is
performed by this route. Submit from `~/HetNet`. These commands contain no
Python, environment preparation or dry run on the login node:

```bash
cd "$HOME/HetNet"
env -u BASH_ENV -u ENV sbatch slurm/softrole_pcp_additional_seeds.sbatch
```

Optional three-run beta0.5 stability group:

```bash
cd "$HOME/HetNet"
env -u BASH_ENV -u ENV sbatch --array=6-8%3 slurm/softrole_pcp_additional_seeds.sbatch
```

The main array defaults to `0-5%6`; optional stability is `6-8%3`. This same
script accepts no arguments, resume switch or scientific overrides. Scheduler
concurrency may be lowered without changing the recipes, for example
`--array=0-5%3`. Do not run it with `bash` on a login node. The Python launcher
also rejects execution outside a matching four-CPU Slurm array.

The default baseline is the existing
`runs/capcon_l1_40m_20261017/manifest.json` from the admitted 944671 preparation.
The launch checks the full original configuration table and its six pinned S1
receipts using `programme.validate_manifest`; it requires non-bounded research
evidence selecting Adam .001 and diagnostics every 50 updates. It does not
re-tune the learning rate or require another preflight submission.
If that admitted manifest is stored elsewhere, set `PCP_ADDITIONAL_BASE_MANIFEST`
to its location before `sbatch`; all validation still runs on the compute node.
A missing/invalid manifest or activation fails before creating training output.
No selector, old manifest or receipt is overwritten or silently regenerated.

The non-login batch startup reuses `publication_resume.sbatch`'s successful
stale-Conda bookkeeping reset, guarded module initialization, the same
`anaconda/anaconda-2024.10` module and core binding. It verifies an executable
interpreter, Python 3.12, all exact `pyproject.toml` dependency pins and imports
NumPy/Torch/DGL before training. Version matching follows the [PyPA specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-matching), so a compatible local build suffix such as `+cpu` is recorded and accepted by a public `==` pin. It installs nothing. BLAS/Torch collector pools
remain one thread. Use the existing pinned interpreter; `HETNET_PYTHON` may point
to that same environment when its location differs.

## Preserved recipes

- HetNet arguments come directly from `publication_reconstruction.study.train_arguments`
  for PCP Real (`index=3`, `reconstruction_spec=paper-v1`, backend `torch-v1`).
  Only the explicit root seed and new output path change. This matches the
  logged PCP Real reference and its successful retry's Torch backend, rather
  than the older batch wrapper's DGL fallback. Four collectors, 500-step floor
  per collector, ten updates per epoch, 2,000-epoch ceiling, horizon 80,
  checkpoint every ten epochs, 10/20/30/40M milestones and 46-hour recoverable
  pause are unchanged. The active paper loss remains actor + critic (1/1),
  including individual rewards/total-N actor weighting and class-average
  Monte Carlo critic targets. The legacy namespace `value_coeff=.01` does not
  replace that active paper loss.
- CapCon config comes directly from `programme_config` under
  `capcon_l1_40m_v1`, with `l1a_decomposed_beta0` or
  `l1a_decomposed_beta05`, certified LR .001 and the new seed. Fixed 2P1A PCP,
  corrected observations, shared actor, own capabilities, three communication
  rounds, four independent heads and Real64 are unchanged. Critic decomposition,
  per-agent GAE and beta mixing are unchanged. Actor/value coefficients remain
  **50/1**. Compressed episode logging, committed checkpoint receipts and
  diagnostics every 50 updates match the prior programme launch.
- Both retain Adam .001, betas (.9,.999), epsilon 1e-8, zero weight decay,
  global completed-episode gradient averaging and clipping at .75. This does
  not equalize the architectures, observation inputs, precision or loss scales.
- Both stop only after a complete update reaches 40,000,000 **joint environment
  transitions**, with actual overshoot recorded. This is not 40M steps per
  agent and is not an extra 40M continuation. The 2,000 x 10 x 4 x 500 floor
  is 40M; a wall-time stop can still leave the scientific budget unfinished.

Seed derivation is unchanged: HetNet uses `seedsequence-v1`, with independent
root seed, library, initialization/collector purpose and collector-ID streams
in `runtime/hetnet_ext/recovery.py:seed_stream`. CapCon uses
`SeedSequence([config.seed, update, collector, 2026])` in `train.py:collect_batch`
and its existing scenario action/message/environment streams. Seeds 3/4/5
therefore reach the actual worker streams instead of reusing seeds 0/1/2.

The legacy HetNet `seed + worker` branch would force overlap for adjacent roots,
but `MultiProcessWorker.run` selects `seed_stream(root, collector_id)` for
`paper-v1`; the parent selects collector 0. The checked root/collector streams
for seeds 3/4/5 are distinct. No worker seed derivation is changed here.

CapCon retains a finite-key limitation: `generate_state(1)[0]` projects each
batch coordinate to 32 bits before constructing scenarios. Exhaustively checking
all possible roots 0..5 x updates 0..19999 x collectors 0..3 (480,000 coordinates)
finds 28 repeated batch keys: 10 within old roots, 7 within new roots (4 between distinct new roots and 3 at
different coordinates of the same root), and 11 between old and new roots. This is the full update-ceiling bound, not a claim
that every future job will execute every coordinate. For example, new root 3 /
update 13967 / collector 2 and root 4 / update 12642 / collector 0 share key
756865376. Such matches reuse scenario RNG keys at particular batches, not an
entire worker stream or a trained trajectory. The beta0 and beta0.5 arms also
intentionally use the same scenario keys for the same root/update/collector
coordinates; their evolving policies differ. Further 32-bit scenario seeds can
also repeat. These are existing recipe limitations, so this extension preserves
them rather than silently changing the experiment. Treat the additional seeds
as distinct root-seeded replicates without claiming mathematically disjoint RNG
inputs at every batch. The bounded collision table and its reproduction script
are saved in the task workspace `additional_seed_work/`.

The implementation references are `publication_reconstruction/study.py:train_arguments`,
`publication_reconstruction/__main__.py:resolve`,
`publication_reconstruction/runtime/hetgat/policy.py:batch_finish_paper`,
`publication_reconstruction/runtime/trainer.py:Trainer.__init__`,
`softrole/programme.py:COMMON, CAPCON_ARMS, programme_config`,
`softrole/learning.py:episode_loss_terms`, `softrole/optimizers.py:optimizer_spec`,
`softrole/train.py:collect_batch, apply_gradients, train`, and the two existing
Slurm wrappers `publication_resume.sbatch` / `softrole_capcon_40m.sbatch`.
No dependency or algorithm behavior is inferred from a newly installed package.

## Outputs and provenance

Every submission uses its Slurm array job ID, so a new submission gets a fresh
root and does not collide with seeds 0–2 or another submitted array:

```text
runs/pcp_additional_seeds_v1_<ARRAY_JOB_ID>/
  manifest.json
  provenance/source/...
  provenance/source_manifest.json
  provenance/working_tree.patch
  provenance/untracked/...
  hetnet_pcp_real/seed3/{launch.json,launcher_exit.json,training/...}
  l1a_decomposed_beta0/seed3/{launch.json,launcher_exit.json,training/...}
  l1a_decomposed_beta05/seed3/{launch.json,launcher_exit.json,training/...}
  ... seed4, seed5 for selected array members ...
```

The first selected member locks a shared manifest on the compute node. It
records all nine resolved entries, actual seed, zero starting counters, exact
HetNet arguments or complete CapCon config, pinned environment, baseline/S1
identity, source hashes, Git commit/status, binary working-tree diff hash and
untracked source hashes. The source, actual diff and untracked file bytes are
archived. A file lock serializes preparation; later members require the exact
same source, Git state, environment and plan. Each selected task exclusively
creates its own seed directory and records the manifest hash and selected Slurm
IDs/resources. A repeated index fails instead of overwriting even a failed run.
Only selected array members train; an all-nine manifest is not nine submissions.

The original trainers additionally capture their actual runtime evidence:
HetNet `protocol.json`, archived runtime/source manifest, resolved arguments,
initial model/optimizer/RNG identity, training segment and status/checkpoints;
CapCon `config.json`, `run.json`, archived source, initial signature,
updates/metrics, compressed episode chunks and committed checkpoints.
Completion is established from those counters and checkpoint evidence, not
merely from a zero wrapper exit code. HetNet can pause recoverably at 46 hours
and return normally below 40M; CapCon can hit the scheduler limit.

Keep this checkout frozen while any member is pending or running. HetNet runs
an archived runtime; CapCon retains the existing trainer's execution from the
checkout while archiving its launch source. Do not edit/pull beneath active
collectors. A later source edit is detected before another member starts, but
cannot retroactively protect a running CapCon process. Interrupted runs are
preserved. This fresh-seed route deliberately does not auto-resume or resubmit;
resumption needs the existing recovery procedure and the exact saved config.
Its extension manifest is not an input to the original seeds-0..2 evaluator.

Slurm stdout/stderr go directly to `pcp-add-seeds-<ARRAY_JOB_ID>_<INDEX>.out/.err`
in the submission directory, which already exists; no missing `logs/` directory
can prevent startup. Runtime training evidence is under the fresh paths above.
Manifests contain private source/diff evidence locally on Stokes; nothing is
uploaded, shared, committed or pushed by this implementation.

## Resources and local verification

Each element retains one node/task, **4 CPUs, 16 GB, 48 hours**, normal partition,
account `cenyioha`, Sapphire Rapids constraint and `srun --cpu-bind=cores`.
The main six at full concurrency request 24 CPUs and 96 GB; optional three add
12 CPUs and 48 GB if run concurrently. Target collection is 240M main steps plus
120M optional steps. Maximum reserved CPU-hours are 1,152 main and 576 optional;
actual scheduling/charging and placement are site policy. No specific hostname
or exclusive node is requested. The old DGL-era wall-time estimate is not a
completion guarantee for these new seeds.

Local safe checks (no training or submission):

```bash
bash -n slurm/softrole_pcp_additional_seeds.sbatch
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest -q tests/test_softrole_pcp_additional_seeds.py
```

Tests compare all nine entries with the existing recipes, check canonical seed
parsing, zero counts and unique output paths, exercise both existing worker
seed derivations, verify S1 admission forwarding and engineering rejection,
check shared provenance identity/duplicate-run refusal, and mock actual Slurm
dispatch and failures with paths containing spaces/shell metacharacters. They
perform no policy rollouts, full training, remote operations or scheduler calls.
The staging state, Stokes dependency activation, available admitted manifest,
queue eligibility and eventual 40M completion remain unverified until submitted.
