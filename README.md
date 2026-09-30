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
and [the implementation record](AGENTS.md). The [architecture comparison](softrole/ARCHITECTURE_COMPARISON.md)
explains what was retained from the supplied proposal, what changed, the
mathematical reasons, differences from original HetNet, and defensible contribution claims.

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
