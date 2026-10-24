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

## Pull, prepare and dry-run on Stokes

Use the existing `~/HetNet` checkout after jobs using it have stopped. Source,
tests and these instructions are versioned in Git. Manifests, logs, checkpoints
and local validation artifacts stay under ignored `runs/` directories. Prepare
the manifest on Stokes after pulling so it records the deployed commit and source
bytes. Keep that checkout fixed until both pilot jobs finish.

```bash
cd ~/HetNet
git pull --ff-only origin main
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="$PWD/.venv/bin/python"
export SOFTROLE_GRADED_PILOT_MANIFEST=runs/graded_sensing_pilot_20261024/manifest.json
"$HETNET_PYTHON" -m softrole.graded_pilot prepare \
  --output "$SOFTROLE_GRADED_PILOT_MANIFEST"
bash scripts/softrole_graded_pilot.sh 0 --dry-run
bash scripts/softrole_graded_pilot.sh 1 --dry-run
```

Each command must succeed before continuing. Use a clean `main` checkout and the
existing working Stokes Python environment. Preparation creates only the
manifest; both dry runs check exact source/config identity and fresh output paths
without collecting episodes or creating run directories. Preparation refuses an
existing manifest. If this checkout already has a pilot manifest, inspect its
origin and source before reusing it; do not overwrite it or regenerate one to
bypass a mismatch.

Once those checks pass, the user submission command is:

```bash
mkdir -p logs_sr
sbatch slurm/softrole_graded_pilot.sbatch
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
```

The launcher rejects budget, seed, model, descriptor, collector and resume
overrides. It refuses existing output directories. Preserve partial outputs if
a job is interrupted; this fresh-only launcher does not implement continuation.
Do not delete or overwrite them to make a retry pass. Use a separately audited
continuation if one becomes necessary.

Local tests and dry runs validate launch wiring. They do not establish Stokes
connectivity, queue availability or training quality. No job was submitted during
preparation.
