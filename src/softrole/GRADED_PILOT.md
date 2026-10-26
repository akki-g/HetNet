# Graded sensing: two-run Stokes pilot

The 24 October pilot completed both budgets; the
[26 October analysis](../../analysis/graded_sensing_2026-10-26/report.md) records
its training and saturation findings. The route below prepares a fresh exploratory
execution, throughput and early-learning check. It does not implement amendment
A-2's full B4 programme or replace the pending B2/B3 reviews.

| Array index | Arm | Descriptor | Seed | Target joint environment steps |
|---|---|---|---|---|
| 0 | G-cap | Physical capability | 0 | 2,000,000 |
| 1 | G-type | Constant physical class | 0 | 2,000,000 |

Both use the primary `l1b_decomposed_beta1` CapCon configuration: shared MLP,
decomposed critic with beta 1, Adam at 0.001, real messages, three communication
rounds, independent heads, PCP 2P1A, horizon 80, four CPU collectors and float64.
Each collector uses one Torch thread and collects at least 500 environment steps
per update. P sensing levels are independently sampled from `(1, 3)` per episode;
A sensing is always 0. Sensor failure is disabled. The descriptor enters both
actor and critic. The two arms differ only in the descriptor; equal seeds do not
guarantee equal trajectories after their policies diverge.

The budget stops at the first complete collector batch reaching 2M steps, so
actual counts can slightly exceed the target. Checkpoints are saved every ten
epochs and at completion, with optimizer/count/source records and checkpoint
commit markers. Complete episode records use indexed gzip chunks. The 2,000-epoch
ceiling remains from the primary configuration; the 2M-step limit stops this pilot
earlier. There is no evaluation, held-out selection or automatic verdict. The
held-out sensing fields are recorded only as configuration metadata.

The active implementation is now under `src/softrole/`. Module commands are
unchanged, and both Slurm launchers set the `src/` import paths. Use fresh
preparation manifests after this source-layout revision; earlier manifests and
completed runs remain evidence of their original source and must not be rewritten.

## Submit through Slurm

Use the existing `~/HetNet` checkout after jobs using it have stopped. Source,
tests and these instructions are versioned in Git. Manifests, logs, checkpoints
and local validation artifacts stay under ignored `runs/` directories. A small
Slurm preparation job creates the manifest and dry-runs both entries on a compute
node. The training array starts only if that job succeeds. Keep the checkout fixed
from preparation until both pilot jobs finish.

```bash
cd ~/HetNet
# Use the reviewed source and pinned environment; do not change queued/running jobs.
run_dir=$(.venv/bin/python scripts/submit_stokes.py --init-only \
  --title 'PCP graded sensing capability versus type' \
  --purpose 'Compare G-cap and G-type with P sensing levels 1/3, seed 0 and 2M steps.')

graded_prep_job=$(.venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
  slurm/softrole_graded_pilot_prepare.sbatch) && \
  .venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
    --dependency="afterok:${graded_prep_job%%;*}" --kill-on-invalid-dep=yes \
    slurm/softrole_graded_pilot.sbatch
```

The submission helper uses only the Python standard library. It creates the
parent container, description, metadata and log directories before invoking
Slurm; numerical preparation still executes on a compute node. Both jobs load
Anaconda and use the existing pinned Python environment. Source and environment
must remain fixed from preparation through training. A prior exported
`SOFTROLE_GRADED_PILOT_MANIFEST` or `SOFTROLE_GRADED_PILOT_RUN_ROOT` must point
inside this new container, or be unset so the helper selects the scoped defaults.

Preparation requests one CPU, 2 GB and ten minutes. It creates `manifest.json`
and previews both entries without collecting episodes or creating trainer leaf
directories. Existing manifests are validated without rewriting them; a source
mismatch fails preparation and prevents the dependent array. Use a fresh
container and manifest after source or launcher changes.

The array requests two concurrent jobs, each with four CPUs, 16 GB RAM and six
hours on `normal`/`sapphirerapids`, account `cenyioha`. These are allocation limits,
not measured completion forecasts. Core binding and one-thread BLAS/OpenMP
settings are unchanged. Inspect status with `squeue -u "$USER"`.

All outputs share the selected titled directory:

```text
runs/<timestamp>_pcp-graded-sensing-capability-versus-type_<id>/
  README.md
  run_info.json
  submissions/                         # commands, responses and job IDs
  logs/softrole-graded-pilot-prepare-<job>_<task>.{out,err}
  logs/softrole-graded-pilot-<array job>_<index>.{out,err}
  manifest.json
  training/g_cap/seed0/                 # source, config, metrics, checkpoints
  training/g_type/seed0/
```

Use the same `--run-dir` for later related stages and preserve this directory
name when copying evidence locally. Do not submit the batch files directly:
they require the metadata and environment set by the submission helper. Add
`--dry-run` to the helper to inspect its command and paths without creating a
container or submitting a job; this is separate from the numerical manifest
preview performed by the preparation job.

The launcher rejects budget, seed, model, descriptor, collector and resume
overrides. It refuses existing output directories. Preserve partial outputs if
a job is interrupted; this fresh-only launcher does not implement continuation.
Do not delete or overwrite them to make a retry pass. Use a separately audited
continuation if one becomes necessary.

Local tests and dry runs validate launch wiring. They do not establish Stokes
connectivity, queue availability or training quality. No new job was submitted while updating this launch workflow.
