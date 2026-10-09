# Shared PCP Adam experiment against archived RMSprop history

Prepared at base commit `ddcd97286e71f5d97105858569caea0ff00d1e35`, on isolated
branch `codex/capcon-adam-history`. Preparation does not submit or train anything.

The primary array contains shared PCP seeds 0, 1 and 2. It changes only the
optimizer recipe to Adam at the historical learning rate **1e-4**. No banked
jobs or concurrent RMSprop jobs are included. The generic CLI also accepts
`--optimizer adam`; its default remains `rmsprop` for every existing launcher.

| Setting | Fixed value |
|---|---|
| Task / environment / team | PCP / corrected-observation-v1 / 2P1A, fixed |
| Grid / vision / horizon | 5 x 5 / 2 / 80 |
| Architecture | Shared recurrence; two distinct rounds; four 16-wide heads; 16-bit payload shared across heads in each round; feedback enabled |
| Actor / critic | Team-sum GAE; gamma 1, lambda .95; coefficients 50 and 1 |
| Update | Four collectors, 500-step floor each, completed episodes; global episode average, then one .75 gradient clip |
| Recurrence | Hidden width 64, preprocessing 128, detach gap 5 |
| Training budget | 2,000 epochs x 10 updates = 20,000 updates; no step cap |
| Checkpoints / episode logs | Every 50 epochs and at completion / file |
| Adam | LR 1e-4; betas (.9,.999); epsilon 1e-8; zero weight decay; AMSGrad, foreach, fused, capturable and differentiable disabled |
| Historical RMSprop | LR 1e-4; alpha .97; epsilon 1e-6; zero momentum/weight decay; not centered |

Adam parameters follow the [official PyTorch API](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).
The existing pinned local Torch 2.2.1 API was also checked directly. This is an
Adam-recipe comparison at a fixed LR, not a search over optimizer hyperparameters.
Each fresh seed creates its original seeded model and an empty optimizer state.
The full recipe is saved in `config.json`, `run.json` and each checkpoint, alongside
the existing initial signature, source manifest, counts and actual optimizer state.

## Historical reference and comparison

The reference is `stokes_runs/softrole_primary/pcp_shared/seed{0,1,2}/`, including
`config.json`, `source_manifest.json`, `source/`, `metrics.jsonl`, `updates.jsonl`
and checkpoints. Its archived commit is
`6994d87b0094e17afa85a1a1ed28043d1c812b46`; source inventory SHA256 is
`35e7fc3a53d6ad4ba1cc5342a977f9ad4568b79de24a9cb53027c20aced07163`.

All three resolved new configurations were compared with their actual archived
configurations: the sole changed recipe field is the optimizer, after filling the
new explicit environment/default fields. Archived `learning.py`, `scenarios.py`
and the PCP simulator are byte-identical to this checkout. Changes already present
since the archive add paper-FC opt-in support, opt-in event diagnostics and return
reporting; they do not change fixed corrected-observation PCP behavior. Initial
shared model tensors are checked against the archived model for all three seeds.

Compare historical and Adam **at matched recorded environment-step budgets**, with
updates and episodes reported as well. Retain the historical 30M frozen selection
rule where using that panel; do not compare a best Adam checkpoint against an RMSprop
endpoint selected differently. For training curves, completion time and per-agent
return are useful once success saturates. Existing counts are authoritative;
legacy HetNet stdout counters and old/current HetNet learners are separate references.

This is a historical-control experiment, as requested. Different execution dates,
nodes and source/provenance plumbing preclude the strongest contemporaneous causal
claim. It does not test banked gates, Gaussian roles, three-layer communication,
64-bit messages, or paper-FC changes. A future banked counterpart would add three
jobs with the same historical banked recipe and a separate output root; it is not
silently included here.

## Submission and resources

Copy the changed files into the intended Stokes checkout, keeping its existing
`.venv` and dependencies. Do not submit an old checkout containing only the new
Slurm file: it also needs the optimizer/config/train/CLI changes and launcher.
From that checkout root, the submission command is:

```bash
mkdir -p logs_sr && sbatch slurm/softrole_adam_pcp.sbatch
```

The array uses the existing working `cenyioha` account, `normal` partition,
`anaconda/anaconda-2024.10` module, one task, four CPUs, 16 GiB and 48 hours per seed.
There is no GPU request. At most three jobs occupy 12 CPUs / 48 GiB. The first
allocation can consume at most 144 job-hours / 576 allocated CPU-hours; this is a
resource cap, not a runtime prediction. BLAS/OpenMP/Torch collector threading is
limited to one thread. `srun --cpu-bind=cores` binds the allocation.

Outputs are isolated under `runs/softrole_adam_pcp_ARRAYJOBID/pcp_shared/seedN`;
logs use `logs_sr/softrole-adam-pcp-ARRAYJOBID_ARRAYINDEX.{out,err}`. An explicit
`SOFTROLE_ADAM_PCP_RUN_ROOT` may relocate outputs; existing run folders are refused.

The full historical update budget is retained: at least 40M environment steps,
up to 46.32M under complete-episode batch overshoot. This is not a shortened
screen. A 48-hour allocation may end before completion: historical shared seed 0
timed out around 33.57M at its valid epoch-1650 checkpoint. Current learner code
has no graceful wall-time pause; an unsaved suffix can be lost. Never accept an
incomplete checkpoint after timeout. Use a validated saved checkpoint and a fresh
output directory to continue, with the same source, optimizer and scientific
configuration. `--resume` retains task/study and permits only budget/save-cadence
changes; optimizer identity, recorded recipe, saved groups and moment keys are
validated before any state is loaded or collection begins. For example, run this
inside a new four-CPU allocation if continuation is needed:

```bash
.venv/bin/python -u -m softrole train --resume PATH_TO_VALID_ADAM_CHECKPOINT --output NEW_CONTINUATION_DIRECTORY
```

The continuation command must not name an RMSprop checkpoint. It keeps the saved
Adam configuration automatically. Do not shorten the 2,000-epoch budget to fit
the wall-time limit or count an abandoned unsaved suffix as retained training.

## Local validation without training

`bash scripts/softrole_adam_pcp.sh 0 --dry-run` resolves the full configuration
without creating a run. Set `HETNET_PYTHON` to an existing interpreter if validating
from a worktree without its own virtual environment. Unit tests use synthetic
gradients, never environment rollouts or research training. They verify the
unchanged RMSprop update, fresh Adam state, exact optimizer-state continuation,
legacy checkpoint compatibility, mismatch rejection, all three launcher seeds
and the Slurm command via harmless module/srun substitutes. Shell syntax and
whitespace are checked separately. Cluster queue access and execution are not
tested during preparation.

Preparation validation passed 95 tests: 14 new launcher tests, 56 optimizer/model/
learner/environment tests and 25 existing failure-launcher tests. On this Mac the
launcher tests were run in a separate process from the Torch numerical tests:
combining them triggered a sandbox OpenMP shared-memory error during subprocess
forking. Both focused groups then passed, as did `bash -n` and `git diff --check`.
All three archived configurations and initial model states matched, and an actual
archived epoch-50 RMSprop checkpoint passed the legacy resume guard. No training,
policy evaluation, cluster command, dependency install or existing-artifact edit
was performed.
