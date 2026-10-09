# HetNet reproduction

Implementation of **Learning Efficient Diverse Communication for Cooperative Heterogeneous Teaming** (Seraj et al., AAMAS 2022): [paper](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf), [author repository](https://github.com/CORE-Robotics-Lab/HetNet), and the included [supplement](AAMAS_22___HetNet_Supplementary.pdf).

Our order of work is:

1. Reproduce the original study, including its domains, comparisons and ablations.
2. Freeze reproduced policies and change only the number/composition of agents on the **same task**, keeping map, sensing, physical capabilities, rewards and rules fixed.
3. Use those results to guide SoftRole development.

The original reproduction tranche excludes sensor degradation, capability loss and within-episode membership experiments. The separate [SoftRole reformulation](softrole/RESEARCH.md) adds a PCP sensor-loss study. The old 21-run PCP transfer-preparation sweep is retired. Full research training and evaluation remain separate from the recorded engineering checks.

## First reproduction tranche

The default array runs the authors' three README recipes with **Real** communication and seeds **0, 1, 2**: **nine separate training runs**. These test whether the original models learn their original tasks. This is **not yet the full paper reproduction** and does not test frozen transfer. Binary is available separately for the published PCP message-width experiments; the initial array does not add unreported PP/FC Binary comparisons.

| Array indices | Task | Team | Map / episode cap | Epochs | Communication |
|---|---|---|---|---:|---|
| 0–2 | Predator–Prey (PP) | 3P, 0A | 5×5 / 80 | 2,000 | Real |
| 3–5 | Predator–Capture (PCP) | 2P, 1A | 5×5 / 80 | 2,000 | Real |
| 6–8 | FireCommander (FC) | 2P, 1A | 5×5 / 300 | 1,400 | Real |

P agents perceive; A agents capture prey or extinguish fire. PP requires all predators to reach prey. PCP additionally requires capture by A agents. FC requires extinguishing the spreading fire. Success, completion steps and returns measure whether these tasks are being learned.

Every run uses four collectors, ten updates/epoch, batch target 500 **per collector**, the released two-layer network/per-class critic, and stepping RMSprop at learning rate 0.0001. A joint step advances the whole environment, not one agent. Complete episodes can overshoot the batch target; use logged counts. Save checkpoints every 50 completed epochs and at the final epoch.

**Release versus paper:** the supplement describes three layers and Adam at 0.001. The active author release has two layers and stepping RMSprop; its policy-local Adam does not step. FC's released action/reward conventions also differ from the supplement. These runs reproduce the released code/README recipe, not a silently reconstructed paper recipe. Binary-16 is the release default; the paper also reports other widths, including Binary-64.

The separate [HetNet reconstruction workflow](publication_reconstruction/README.md) provides public-code compatibility and supplement-aligned three-layer/Adam models, corrected environments, complete-update recovery, and isolated frozen evaluation. Its locked 12-run study has proper preflight, training, continuation and evaluation Slurm files. Start with `sbatch slurm/publication_preflight.sbatch` after creating `logs_1`; long training remains gated on compute-node checks. Original entrypoints remain unchanged, and the public history does not identify the exact publication-producing checkout.

## Run

Use Python 3.12. The setup script creates `.venv` using standard `venv`/`pip` and installs the pinned packages in `requirements.txt`, exported from the committed `uv.lock`. It also installs the bundled environments. **No uv installation is needed.** For Stokes, use the compute-node setup job below.

```bash
bash scripts/setup_env.sh
bash scripts/reproduce.sh pcp real 0 --dry-run
bash scripts/reproduce.sh pcp real 0
```

The short launcher calls the original `main.py`. There is no preflight JSON, budget approval file, calibration gate or separate submission service. Extra arguments are ordinary `main.py` options, recorded verbatim. A quick engineering smoke, **not a research run**:

```bash
HETNET_RUN_ROOT=runs/smoke bash scripts/reproduce.sh pcp real 0 \
  --num_epochs 2 --epoch_size 2 --batch_size 4 --max_steps 4
```

Choose `pp`, `pcp` or `fc`, and `real` or `binary`. A separate message-width run is explicit:

```bash
HETNET_RUN_ROOT=runs/binary64 bash scripts/reproduce.sh pcp binary 0 --msg_dim 64
```

Existing run directories are never overwritten. Use a new `HETNET_RUN_ROOT` for retries or changed settings. The wrapper fixes output paths and otherwise passes training options through; inspect `command.txt` and `resolved_args.json` before comparing runs. Overriding seed/task/method flags can make directory labels differ from the effective settings. `HETNET_PYTHON` can select another equivalent locked environment.

## UCF Stokes

From the repository root on the submitting node:

```bash
mkdir -p logs
setup_job=$(sbatch --parsable slurm/setup.sbatch)
sbatch --dependency=afterok:"$setup_job" slurm/reproduce.sbatch
```

The one-time setup job loads `anaconda/anaconda-2024.10`, checks for Python 3.12, and creates `.venv` **on a compute node**, following [ARCC's installation guidance](https://arcc.ist.ucf.edu/docs/software/anaconda/). It uses `pip`, checks dependency consistency, and tests imports and a CPU DGL operation. Training starts only after setup succeeds. Each training job loads the same module and reuses `.venv`; it does not reinstall packages. The setup job requests one hour, two CPUs and 8 GiB; these are installation limits, not training-time estimates.

Setup output is in `logs/setup-JOB_ID.out` and `.err`. If setup fails, the dependent training array cannot start; inspect these logs, fix the cause, and submit again with a new setup job ID. Cancel the old waiting array with `scancel ARRAY_JOB_ID`. Package downloads require network access and disk space; the locked Linux PyTorch distribution includes CUDA libraries even though these runs use CPUs. Create the environment on Stokes; do not copy a Mac `.venv` there.

Once setup has succeeded, later submissions need only:

```bash
mkdir -p logs
sbatch slurm/reproduce.sbatch
```

To use an existing equivalent environment, export `HETNET_PYTHON=/absolute/path/to/bin/python` before submission; the array preserves this choice. `HETNET_BASE_PYTHON` selects Python 3.12 for environment creation if needed. Existing uv users may still run `uv sync --locked --python 3.12` instead of the pip setup. Do not run setup while training jobs use that environment.

This submits nine runs, at most three concurrently, on `normal`, account `cenyioha`, four CPUs/run, no GPU. The **16 GiB / 48-hour** defaults are starting requests, not measured requirements or runtime predictions. Override them with normal Slurm flags:

```bash
# Just PCP Real seed 0; use the table to select other tasks.
sbatch --array=3 --time=2-00:00:00 --mem=16G slurm/reproduce.sbatch
# Full tranche at a different concurrency:
sbatch --array=0-8%9 slurm/reproduce.sbatch
```

Do not submit overlapping arrays to the same output root. Keep the checkout and environment unchanged while jobs are queued or running: spawned workers import that source. Stokes has heterogeneous CPUs and no duration has been measured yet. If all nine jobs consumed the full default request, the ceiling would be 9 × 4 × 48 = 1,728 allocated CPU-hours; actual charged elapsed time may be less. Inspect an initial run's speed/memory before a large batch. Automatic resume is not implemented.

## Logs and progress

Output lives in `runs/reproduction/<task>_<variant>/seed<seed>/`:

| Artifact | Meaning |
|---|---|
| `command.txt`, `environment.txt`, `source.patch`, `uv.lock`, `requirements.txt` | Exact command, Git revision/dirty state, installed package versions, local source diff and dependency pins. |
| `resolved_args.json` | Effective training/environment defaults and selected model. |
| `metrics.jsonl` | Fresh epoch counts, success, episode length, per-agent returns, losses and epoch time. |
| `stdout.log`, `exit_code.txt` | Training output and exit status. Missing status can mean running or interrupted. |
| `checkpoints/`, `checkpoint_records.jsonl` | Saved models, serialization time and sizes. |
| Initial/epoch/checkpoint signatures | Parameter/buffer identity retained for reproducibility and later frozen evaluation. |

```bash
python -m hetnet_ext.progress --runs runs/reproduction
python -m hetnet_ext.progress --runs runs/reproduction --out runs/progress
squeue -u "$USER"
# Replace JOB_ID with the actual job/array ID.
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS
```

The standard-library-only monitor discovers runs, exports `summary.csv` or prints `--json`, and summarizes the last 50 epochs (`--window` changes this). Success, steps and P/A returns are episode-weighted; losses retain the joint-step reporting denominator. PP has no A return. The monitor does not load or certify checkpoints.

Look for improving success, shorter episodes and coherent returns **across seeds**. Finite/decreasing loss alone does not establish learning. Use `metrics.jsonl` for counts: original stdout overcounts cumulative samples. Epoch timing omits checkpoints and some logging overhead. Training curves do not replace final policy evaluation.

### Training speed

Python allocation tracing (`tracemalloc`) is now off by default. The released code
enabled it throughout collection and backpropagation; this is expensive for the
many small Python/DGL operations per step. Add `--profile_memory` to restore that
diagnostic. The flag is recorded in `resolved_args.json`; it does not change the
training recipe. The stdout allocation peak is labeled `disabled` when tracing is
off. When enabled, it measures traced Python allocations, not total process RAM;
use Slurm's `MaxRSS` for process memory.

An initial local CPU check (Apple M4 Pro, locked environment, PCP Real, one
collector, three updates, batch target 160, horizon 80) took 16.53 seconds with
tracing and 4.17 seconds without it. Each update produced identical rewards,
losses and parameter hashes. A second check used the actual launcher with four
collectors, batch target 500 and horizon 80, shortened to two epochs of two updates:
75.37 seconds with tracing versus 21.05 without (3.6x). Both epochs had identical
model hashes and all non-timing metrics. These are local measurements, not Stokes
runtime predictions. Benchmark a short run on the target node before choosing a
full-run time limit. For example, from an updated checkout:

```bash
# Original PCP recipe, with only the number of epochs shortened for timing.
HETNET_RUN_ROOT=runs/timing-no-tracing \
  sbatch --array=3 --time=01:00:00 slurm/reproduce.sbatch --num_epochs 3
# Optional comparison with the original tracing behavior, in a separate run.
HETNET_RUN_ROOT=runs/timing-with-tracing \
  sbatch --array=3 --time=01:00:00 slurm/reproduce.sbatch --num_epochs 3 --profile_memory
```

Compare epoch times after startup on the same CPU model. Each epoch still collects
at least 20,000 joint steps (4 collectors × 500 steps × 10 updates), so PP/PCP
require at least 40 million steps per seed. Increasing `--cpus-per-task` alone
does not add collectors, and increasing `--nprocesses` with the same batch target
changes the data per update. Reducing epochs or batch size changes the reproduction
budget. A GPU is not a validated shortcut for the current per-step DGL and shared
CPU parameter path.

Already running jobs retain their loaded code. Use a separate checkout for timing
while existing jobs use the old source, and preserve their output directories.
The 48-hour Slurm default is not sufficient if measured epoch time projects beyond
it; automatic resume is still not implemented.

## Full original-study coverage still required

| Paper result | Required experiment | Remaining work |
|---|---|---|
| Table 1; Fig. 3 | PP/PCP/FC quality and PP/PCP learning versus CommNet, IC3Net, TarMAC and MAGIC | Matched baseline recipes and final evaluation. CommNet/IC3Net paths exist; TarMAC/MAGIC implementations are absent from this release despite its original README. |
| Fig. 4; Fig. 5b | PCP communication cost; Real, Binary 4/8/16/32/64, no communication | Explicit width runs and bits/round versus bits/step accounting. Zero-width messages are not necessarily no communication. |
| Fig. 5a | Full/half/no communication range | Resolve numerical “half” setting and no-message construction. Range zero still permits co-located senders. |
| Fig. 5c | Binary training at 2P1A, 3P3A and 4P6A | Establish matching map/horizon. This is separately trained scalability, not frozen transfer. |
| Fig. 6 | Real centralized/per-class/per-agent critics | Expose and validate existing alternate branches; release selects per-class. |
| Supplement Fig. 1 | STE versus Gumbel at 8/16 bits | Wire estimator selector and resolve unspecified task/composition. |

The paper reports seeds 0/1/2 and 50 evaluation trials. The released HetNet evaluator instead hardcodes 100 episodes and has author-specific paths/reset assumptions. It is not a validated Table 1 evaluator. Resolve these issues before final evaluation; do not invent missing settings or claim absent baselines were reproduced. A paper-recipe reconstruction must be labeled separately from the released-code runs above.

The [IC3Net source](https://github.com/IC3Net/IC3Net) supplies the original code lineage; the [FireCommander project](https://github.com/EsiSeraj/FireCommander2020) provides domain background.

## Code and checks

- `main.py`, `trainer.py`, `multi_processing.py`, `hetgat/`, `envs/`: original learner, model and environments.
- `scripts/reproduce.sh`: domain commands and output capture.
- `scripts/setup_env.sh`, `requirements.txt`: Python 3.12 venv/pip setup with pinned runtime and build packages, including the bundled environments.
- `slurm/setup.sbatch`: one-time Stokes compute-node installation and import check.
- `slurm/reproduce.sbatch`: plain array mapping; no JSON prerequisites.
- `hetnet_ext/seeding.py`, `recording.py`, `signatures.py`: early seeding, accurate counts and saved-state identity.
- `hetnet_ext/progress.py`: read-only summaries.
- `tests/`: accounting, launch/progress and gradient regressions.

Retained fixes cover Python/Gym/CPU compatibility, seeding before construction and in-place gradient clearing at the three previously repaired sites. Additional narrow repairs replace removed NumPy integer aliases in FC and handle Binary PP's empty A class without adding agents/messages. They preserve layer count, optimizer, rewards and physical rules.

Twelve tiny actual-training checks passed: three tasks × two communication variants × one/four processes, four updates each. They establish execution, finite metrics and saved/changed weights—not convergence. The Binary fix also preserved existing nonempty-A outputs/gradients exactly.

```bash
.venv/bin/python -m pip install pytest==8.3.5
.venv/bin/python -m pytest -q
```

The test runner is optional for training. If maintaining dependencies, regenerate the pip export after changing the lock:

```bash
uv export --locked --no-emit-project --no-emit-local --no-dev \
  --no-annotate --no-hashes --output-file requirements.txt
```

The old launch framework, PCP-only grid, Mac wrappers and mandatory preflight/budget machinery are removed. Research/evidence files remain locally but are excluded from new checkouts; their full tracked history is preserved at commit `47b99bf`. Work continues on `main`; the merged `frozen-eval` branch is deleted.

## SoftRole reformulation (`softrole` branch)

The standalone `softrole` package implements the revised deterministic,
capability-conditioned architecture. It removes explicit actor class labels and
typed occupancy channels, uses two 16-bit broadcast rounds, and trains with one
team advantage. See [the mathematical plan and experiment protocol](softrole/RESEARCH.md)
and [the implementation record](AGENTS.md). The [architecture comparison](docs/research/ARCHITECTURE_COMPARISON.md)
explains what was retained from the supplied proposal, what changed, the
mathematical reasons, differences from original HetNet, and defensible contribution claims.

For a self-contained explanation, read the [SoftRole architecture guide](docs/SOFTROLE_ARCHITECTURE_GUIDE.pdf)
([Markdown](docs/SOFTROLE_ARCHITECTURE_GUIDE.md), [LaTeX](docs/SOFTROLE_ARCHITECTURE_GUIDE.tex)).
It develops the observation, memory, gates, binary messages, attention, critic,
rewards and training loss step by step, with equations and an architecture diagram.

For the next PCP sensor-failure experiments, start with the
[agent handoff](docs/plans/SENSOR_FAILURE_HANDOFF.md). It records current findings, existing
support, pilot preparation, verified entrypoints and required implementation logs.

```bash
# Inspect the original-domain recipe; no run directory is created.
bash scripts/softrole.sh pcp banked 0 --dry-run

# Small execution check (use a fresh output path).
.venv/bin/python -m softrole train --task pcp --model banked \
  --epochs 1 --updates-per-epoch 2 --batch-steps 4 --max-steps 4 \
  --nprocesses 4 --output runs/softrole_smoke

# Full domain recipe: substitute pp / pcp / fc and shared / banked.
bash scripts/softrole.sh pcp banked 0

# Primary composition or sensor-failure study, isolated from fixed-team runs.
SOFTROLE_RUN_ROOT=runs/softrole_failure \
  bash scripts/softrole.sh pcp banked 0 --study failure
```

For the native-composition sensor pilot, use `pilot-failure`: it saves a common
20-scenario panel by default, evaluator source/runtime identities and matched failure/sham
traces, then checks paired prefixes and reports descriptive outcomes. Supply one
nominal shared/banked checkpoint pair per seed and a predeclared checkpoint rule.
The [pilot protocol and commands](softrole/RESEARCH.md#81-native-pcp-failure-pilot)
explain the pre-event diagnostics and interpretation limits.

The PCP-only failure-training route is `bash scripts/softrole_failure.sh 0 --dry-run`.
Indices 0–2 select shared seeds 0–2; 3–5 select banked seeds 0–2. Set
`SOFTROLE_FAILURE_RUN_ROOT` to a fresh study root. Its optional
`slurm/softrole_failure.sbatch` has six PCP jobs; passing the failure preset to
the original 18-job all-domain array is unsupported. No submission is automatic.

The package also provides `evaluate`, `evaluate-hetnet` and `summarize` commands.
Frozen evaluation supports held-out compositions, event-timed gate/communication
interventions, matched no-failure `--sham` controls and independent random streams.
Summaries use independent training seeds as the uncertainty unit. Detailed commands,
splits, ablations and interpretation limits are in the research document.

`slurm/softrole.sbatch` is an optional 18-job screening array (three domains × two
models × three seeds). No jobs are submitted by training setup. Runs archive source,
configuration, metrics and optimizer checkpoints; resume requires a fresh output
directory. PP/PCP/FC defaults match the repository's domain recipes, but the new
actor, learner and corrected observations constitute a separate experiment.
Original reproduction commands retain their default behavior.

### PCP CapCom Binary/Real communication experiment

`scripts/softrole_channels_pcp.sh` prepares six fresh **shared CapCom** runs,
continuing the model and Adam learner used in the latest PCP trials. Indices
0–2 select Binary seeds 0–2; indices 3–5 select Real seeds 0–2. Each seed therefore
has one run of each channel. Both use three rounds, four independently encoded
64-dimensional payloads per round, four 16-wide attention heads, the same
encoder/decoder architecture, and the same global team critic. Binary samples
hard Bernoulli bits with the existing straight-through training derivative;
Real sends the continuous encoder outputs. The channel is the only configured
difference within each seed pair. Matching seeds does not make subsequent
trajectories identical once the channel changes.

Binary broadcasts **768 logical bits per agent-step** (3 × 4 × 64). Real
broadcasts **768 real-valued scalars**; CPU float64 represents those scalars with
49,152 bits, excluding transport overhead. This is a matched-dimension channel
comparison, not an equal-bandwidth comparison. It does not reproduce HetNet-Real's
removal of its encoder/decoder, and it does not isolate the benefit of capability
banks because this six-run study uses the shared variant.

The fixed protocol uses corrected native PCP observations, 2P1A on a 5×5 grid,
vision 2, horizon 80, no sensor failures, Adam at 1e-4, four collectors, and the
unchanged team objective (actor coefficient 50, value coefficient 1). Each run
targets 40M environment steps with a 2,000-epoch cap, records actual counts and
complete-update overshoot, and saves every 50 epochs plus its terminal
checkpoint. Reduced runtime budgets below are execution checks, not the
research budget.

```bash
# Read the complete fixed configuration without creating output directories.
bash scripts/softrole_channels_pcp.sh 0 --dry-run
bash scripts/softrole_channels_pcp.sh 3 --dry-run

# One small local execution; use a fresh root each time.
SOFTROLE_CHANNELS_PCP_RUN_ROOT=runs/channels_pcp_smoke \
  bash scripts/softrole_channels_pcp.sh 0 --epochs 1 --updates-per-epoch 1 \
    --batch-steps 1 --nprocesses 1 --total-steps 1 --save-every 1

# Optional six-job Slurm array; this command submits the full experiment.
mkdir -p logs_sr
sbatch slurm/softrole_channels_pcp.sbatch
```

Outputs are isolated under
`runs/softrole_channels_pcp_JOBID/pcp_shared/{binary,real}/seed{0,1,2}` on Slurm
and `runs/softrole_channels_pcp_manual/` locally. Set
`SOFTROLE_CHANNELS_PCP_RUN_ROOT` to choose a different fresh study root. Existing
run directories, resume and scientific-setting overrides are rejected. Only
the displayed runtime/budget settings, `--episode-log file|stdout` and `--dry-run`
may be overridden. The array permits three concurrent jobs and retains core
binding and one thread per numerical library; its four-CPU, 16 GB, 48-hour
requests are starting settings, not a measured runtime for this larger model.
No submission is automatic.

SoftRole logs both `team_return` and `mean_agent_return`. The latter is the
mean of each episode's agent returns, matching the released HetNet reporting
scale. For fixed three-agent teams it is `team_return / 3`; for varying teams,
divide within each episode before averaging episodes. Team rewards and the
learning objective are unchanged. Frozen evaluation and seed-level summaries
also include this metric, including summaries of older evaluation files.

Episode JSONL logging now opens the file once per optimizer update and writes
the complete batch of episode records. Records are buffered only for that
update. To put them in Stokes stdout instead, pass `--episode-log stdout` to
the SoftRole training launcher (also supported by the PCP failure launcher).
It prints one flushed JSON object per update with
`record_type: "softrole_episode_batch"`, `update`, and an `episodes` array.
Filter on that record type when extracting episodes from a mixed `.out` file;
`metrics.jsonl` continues to contain only epoch records and `updates.jsonl`
continues to contain update records. The default `--episode-log file` retains
the existing episode JSONL format. Logging mode can change on resume without
changing the scientific configuration. Stdout still incurs I/O and has the
same total episode data; buffering is not a measured training speedup.

The local October 2 snapshot is replotted against completed epochs:
[PP](analysis/training_epochs_2026-10-02/pp_training_epochs.png),
[PCP](analysis/training_epochs_2026-10-02/pcp_training_epochs.png),
[FC](analysis/training_epochs_2026-10-02/fc_training_epochs.png),
[diagnostics](analysis/training_epochs_2026-10-02/diagnostics_epochs.png), and
[recorded training time](analysis/training_epochs_2026-10-02/training_time_epochs.png).
Editable SVGs, input/output hashes and the reproducible `plot.py` are alongside
the figures. All reward curves use mean-agent returns. Epochs correspond to ten
updates in these runs; sample counts remain necessary for budget comparisons.
These historical training curves retain the documented simulator defects and
are not frozen-policy baseline results. Local analysis artifacts are excluded
from new checkouts by the repository's existing ignore rules.

Frozen SoftRole evaluation can run where the checkpoints already reside on
Stokes. From the repository root, create `logs_sr` and submit one checkpoint
with explicit composition, episode count and evaluation seed. Replace the
checkpoint placeholder and use a fresh result path:

```bash
mkdir -p logs_sr
sbatch slurm/softrole_evaluate.sbatch \
  /actual/stokes/selected/checkpoint.pt \
  runs/frozen_pcp/shared_seed0_native.json \
  --compositions 2,1 --episodes 500 --seed 2700
```

The same command without Slurm is
`bash scripts/softrole_evaluate.sh CHECKPOINT OUTPUT [evaluate options...]`.
Use the same evaluator checkout, environment, checkpoint-selection rule and
evaluation panel across models/seeds. Copy result JSON files back first;
`--trace` is optional and can greatly enlarge them. The one-CPU, 4 GB, four-hour
Slurm requests are unmeasured starting settings. This launcher handles SoftRole
checkpoints; it does not add evaluation support for publication reconstruction
checkpoints. It does not submit training or change budgets/architectures.

For the six existing primary PCP policies, prepare the matched frozen panel
on Stokes after syncing this evaluator code. Submit preparation itself as a
Slurm job so checkpoint loading and validation run on a compute node:

```bash
mkdir -p logs_sr
sbatch slurm/softrole_prepare_frozen.sbatch \
  --run-root runs/softrole_primary \
  --output runs/frozen_pcp_30m_20261002_slurm
```

Run this from the repository root. The preparation job requests one CPU, 4 GB
and 30 minutes; these are provisional allocation limits. It uses the existing
Anaconda module and repository virtual environment, limits numerical libraries
to one thread before Python starts, and forwards the preparation CLI arguments.
Logs are `logs_sr/pcp-prepare-JOBID.out` and `.err`. Preparation writes the
18-task `submit.sbatch` array but does not execute or submit evaluations.

After the preparation job finishes successfully and its stdout contains
`"jobs_prepared": 18`, submit the evaluation array:

```bash
sbatch runs/frozen_pcp_30m_20261002_slurm/submit.sbatch
```

Use a fresh output directory for each preparation attempt. If direct execution
is allowed, the same arguments can be passed to
`.venv/bin/python scripts/prepare_pcp_frozen.py` instead of the batch launcher.
The Python helper also enforces thread limits before importing Torch/NumPy,
overriding inherited values such as 64.

The array uses indices 0–17, with at most three tasks running concurrently.
Each task executes one frozen evaluation through `srun`, with one CPU, 4 GB,
a four-hour limit and separate `logs_sr/pcp-frozen-%A_%a.out`/`.err` files.
Within each shared/banked seed, the three indices select nominal, failure and
sham in that order. `manifest.json` records the exact job-to-checkpoint mapping.
The optional generated `submit.sh` submits this same array and records task
IDs in `job_ids.tsv`; use either submission route once, not both. If a plan
was prepared before the Slurm-array update, prepare a new output directory.

The checkpoint rule is the first available saved checkpoint reaching 30M true
environment steps, separately for shared/banked seeds 0/1/2. This is an
intermediate evaluation selection, not a new training stopping budget. The
October 2 audited logs predict epoch 1500 for all six (30.51–30.73M actual steps,
15,000 updates); preparation must validate the actual Stokes checkpoints and
structured epoch ledgers. Missing eligible checkpoints stop preparation instead
of silently substituting older policies. Keep the selected checkpoint files and
evaluator checkout fixed through completion of all jobs; the source archive is
provenance, not a separate execution checkout.

The prepared panel contains 18 one-CPU evaluation jobs: six nominal/transfer jobs
with 500 scenarios each for `(2,1)`, `(1,2)`, `(2,2)`, `(3,1)` and `(3,2)`, plus
six native `(2,1)` failure jobs and six matching shams with 100 scenarios each.
Evaluation seeds are 2700 for nominal/transfer and 2701 for failure/sham. The
sensor pilot retains uniform event times 10–30 and failure probability one.
Explicit panel files preserve the same scenarios across all six policies and
retain the scheduled events in shams. Only the small sensor panel saves step
traces, enabling pre-event prefix checks. Total policy episodes: 16,200.

The four-hour requests are allocation limits, not measured completion forecasts.
Copy result JSON, panel and selection records back before bulky training logs.
Do not change gates, checkpoints or hyperparameters using held-out outcomes.
This sensor panel diagnoses exposure; it does not guarantee an informative
failure-training distribution. Preserve it separately from any preregistered
earlier-event supplement. Existing summaries stratify by checkpoint epoch/update
and do not encode event-window identity, so keep distinct timing protocols in
separate summary invocations. Do not submit the full training arrays as a
substitute for this frozen evaluation panel.

## Citation and license

```bibtex
@inproceedings{seraj2022learning,
  title={Learning efficient diverse communication for cooperative heterogeneous teaming},
  author={Seraj, Esmaeil and Wang, Zheyuan and Paleja, Rohan and Martin, Daniel and Sklar, Matthew and Patel, Anirudh and Gombolay, Matthew},
  booktitle={Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems},
  pages={1173--1182},
  year={2022}
}
```

The tracked [LICENSE](LICENSE) is GPLv3; the original README's MIT claim was inconsistent with that file.
