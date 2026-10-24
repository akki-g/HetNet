# Graded sensing: two-run Stokes pilot

Prepared on 24 October 2026; no jobs have been submitted. This is an exploratory
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

## Submit through Slurm

Use the existing `~/HetNet` checkout after jobs using it have stopped. Source,
tests and these instructions are versioned in Git. Manifests, logs, checkpoints
and local validation artifacts stay under ignored `runs/` directories. A small
Slurm preparation job creates the manifest and dry-runs both entries on a compute
node. The training array starts only if that job succeeds. Keep the checkout fixed
from preparation until both pilot jobs finish.

```bash
cd ~/HetNet
git pull --ff-only origin main
mkdir -p logs_sr
export SOFTROLE_GRADED_PILOT_MANIFEST="runs/graded_sensing_pilot_20261024/manifest_$(git rev-parse --short HEAD).json"
graded_prep_job=$(sbatch --parsable slurm/softrole_graded_pilot_prepare.sbatch) && \
  sbatch --dependency="afterok:${graded_prep_job%%;*}" --kill-on-invalid-dep=yes \
    slurm/softrole_graded_pilot.sbatch
```

Use a clean `main` checkout and the existing working Stokes Python environment.
Both jobs load the Anaconda module and call the Python entrypoint directly through
`srun`; no `scripts/*.sh` wrapper or login-node Python invocation is required.
Preparation requests one CPU, 2 GB and ten minutes. It creates only the manifest;
both dry runs check exact source/config identity and fresh output paths without
collecting episodes or creating run directories. An existing manifest is validated
without being overwritten. A mismatch fails preparation and prevents training.
The command names the manifest after the pulled commit, preserving manifests from
earlier source revisions. Both submissions inherit that same path. Existing
training output directories still prevent duplicate launches.

To inspect the submitted jobs:

```bash
squeue -u "$USER"
```

Create `logs_sr` before `sbatch`, because Slurm opens its output files before the
script body runs. The array requests two concurrent jobs, each with four CPUs,
16 GB RAM and six hours on `normal`/`sapphirerapids`, account `cenyioha`. Six hours
is an allocation limit, not a measured completion forecast. The Slurm route
uses core binding and limits BLAS/OpenMP threads to one.

Outputs are separate fresh directories:

```text
runs/graded_sensing_pilot_20261024/training/g_cap/seed0/
runs/graded_sensing_pilot_20261024/training/g_type/seed0/
logs_sr/graded-sensing-pilot-<array job>_<index>.out
logs_sr/graded-sensing-pilot-<array job>_<index>.err
logs_sr/graded-sensing-prepare-<job>.out
logs_sr/graded-sensing-prepare-<job>.err
```

The launcher rejects budget, seed, model, descriptor, collector and resume
overrides. It refuses existing output directories. Preserve partial outputs if
a job is interrupted; this fresh-only launcher does not implement continuation.
Do not delete or overwrite them to make a retry pass. Use a separately audited
continuation if one becomes necessary.

Local tests and dry runs validate launch wiring. They do not establish Stokes
connectivity, queue availability or training quality. No job was submitted during
preparation.
