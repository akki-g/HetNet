# Matched SoftRole FC launch and recovery review

These are prepared commands, not submitted jobs. Run from the reviewed Stokes
checkout with the same installed environment. Keep that checkout unchanged
throughout the run, including while jobs are queued.

## What the current code supports

`slurm/softrole_paper_fc.sbatch` launches six fresh runs: shared seeds 0–2 and
banked seeds 0–2. Each uses four collectors, floor 500, ten updates per epoch,
1,400 epochs and a 28M environment-step cap, with paper-v1 FC semantics. It
requests 48 hours, four CPUs and 16 GiB. It saves every ten epochs (100 updates),
and also when reaching the step or epoch budget.

**SoftRole has no 46-hour graceful pause and no `--wall-seconds` flag.** Its
launcher accepts only an optional `--dry-run`; it cannot be reused by appending
`--resume`. SoftRole's direct `train --resume CHECKPOINT --output FRESH` command
restores configuration, model, RMSprop state and completed work counters. Its
random streams are indexed by seed/update/collector, and it resumes any remaining
updates in a partial epoch. Do not pass `--task` or `--study` on resume; both are
inherited. Keep the 28M budget and all method settings unchanged.

Unlike HetNet's continuation, **SoftRole resumes the current checkout's learner
and model code**. It records the parent source identity and checks that paper
simulator bytes are unchanged, but does not automatically enforce equality of
every learner file or execute the parent's entire code archive. The explicit
source checks below supply the missing operational guard. A full source-manifest
comparison also notices documentation/test edits; investigate a mismatch rather
than rewriting the parent record or treating it as automatically harmless.

Sources: [`softrole/__main__.py`](../../../softrole/__main__.py),
[`softrole/train.py`](../../../softrole/train.py),
[`softrole/publication_env.py`](../../../softrole/publication_env.py),
[`softrole_paper_fc.sbatch`](../../../slurm/softrole_paper_fc.sbatch).

## Consistent fresh launch roots

`paper-study-plan` prepares twelve HetNet runs and six matched SoftRole FC runs,
plus a common nominal 500-scenario FC panel. For a fresh chosen `STUDY_ROOT`, use
`PUBLICATION_RUN_ROOT="$STUDY_ROOT/hetnet"` and
`SOFTROLE_PAPER_RUN_ROOT="$STUDY_ROOT/softrole_fc"`. The plan must explicitly use
`--message-backend torch-v1`; its default and the HetNet launcher's default are
DGL. All initial and continuation output directories must be fresh. Neither
launcher's default job-ID-derived root matches a manually prepared study root
unless those environment variables are supplied.

The main report's exact source check binds the reconstruction to the preflight
archive. The preparation additionally records SoftRole, dependency and launcher
identities; preserve them and verify again before executing queued work. These
HetNet preflights do not estimate SoftRole training time.

## Validate a saved SoftRole checkpoint before recovery

If Slurm terminates a run, retain its entire archive. Work after the last valid
checkpoint is an abandoned suffix and must not be counted twice after replay.
Normally at most 99 completed updates lie beyond the last 100-update save. A
timeout during checkpoint writing can leave an incomplete `.pt` or missing
signature sidecar, because SoftRole's save is not atomic. In that case use the
previous fully valid checkpoint; the discarded suffix can include the save
boundary itself. Select the latest valid checkpoint by its saved update count,
not simply the lexically last filename.

The following concrete example names shared seed 0 and a candidate checkpoint.
Replace `epoch0100.pt` with the actual last valid saved candidate. The candidate
must have fewer than 28M steps; a completed run needs no continuation. If the
candidate fails load/signature checks, inspect the previous candidate. Source
or configuration failures require review, not silent fallback or acceptance.

```bash
export SOFTROLE_CHECKPOINT="$STUDY_ROOT/softrole_fc/shared/seed0/checkpoints/epoch0100.pt"
export SOFTROLE_SEGMENT="$STUDY_ROOT/continuations/softrole_fc/shared/seed0/segment02"
export SOFTROLE_RECOVERY_RECORD="$PREPARATION/softrole_shared_seed0_segment02.json"
test ! -e "$SOFTROLE_SEGMENT"
test ! -e "$SOFTROLE_RECOVERY_RECORD"
.venv/bin/python - <<'PY'
import hashlib, json, os
from pathlib import Path
import torch
from hetnet_ext.signatures import model_signature
from softrole import CHECKPOINT_VERSION
from softrole.config import Config
from softrole.model import SoftRoleNet
from softrole.publication_env import checkpoint_binding, simulator_identity
from softrole.train import source_snapshot

checkpoint = Path(os.environ['SOFTROLE_CHECKPOINT']).resolve()
output = Path(os.environ['SOFTROLE_SEGMENT']).resolve()
record_path = Path(os.environ['SOFTROLE_RECOVERY_RECORD'])
assert not output.exists()
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert saved['format_version'] == CHECKPOINT_VERSION
config = Config(**saved['config'])
config.validate()
assert (config.task, config.env_version, config.model, config.seed) == ('fc', 'paper-v1', 'shared', 0)
assert (config.epochs, config.total_steps, config.nprocesses, config.batch_steps,
        config.updates_per_epoch, config.save_every) == (1400, 28000000, 4, 500, 10, 10)
assert saved['environment_version'] == 'paper-v1'
assert saved['model_config'] == config.model_kwargs()
assert 0 < saved['total_steps'] < 28000000
assert 0 < saved['updates'] < 14000
assert saved['updates'] == saved['completed_epochs'] * 10 + saved['updates_in_partial_epoch']
assert 0 <= saved['updates_in_partial_epoch'] < 10

manifest = json.loads((checkpoint.parent.parent / 'source_manifest.json').read_text())
assert saved['source_sha256'] == manifest['sha256']
assert manifest['sha256'] == hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
for name, expected in manifest['files'].items():
    assert hashlib.sha256((checkpoint.parent.parent / 'source' / name).read_bytes()).hexdigest() == expected, name
current = source_snapshot()
assert current['files'] == manifest['files'], 'Current source differs from the parent; review before resuming'
checkpoint_binding(checkpoint, saved)
assert simulator_identity() == saved['simulator_source']

torch.set_num_threads(1)
model = SoftRoleNet(**config.model_kwargs()).double()
model.load_state_dict(saved['model_state'], strict=True)
for name, value in model.state_dict().items():
    original = saved['model_state'][name]
    assert torch.isfinite(original).all()
    assert value.dtype == original.dtype and torch.equal(value, original), name
actual_signature = model_signature(model)
assert actual_signature == saved['signature']
assert actual_signature == json.loads(Path(str(checkpoint) + '.signature.json').read_text())
optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr, alpha=.97, eps=1e-6)
assert saved['optimizer_state']['param_groups'] == optimizer.state_dict()['param_groups']
optimizer.load_state_dict(saved['optimizer_state'])
assert optimizer.state
for parameter, state in optimizer.state.items():
    assert state['step'].item() == saved['updates']
    assert state['square_avg'].shape == parameter.shape
    assert state['square_avg'].dtype == parameter.dtype
    assert torch.isfinite(state['square_avg']).all() and (state['square_avg'] >= 0).all()
    assert all(not torch.is_tensor(value) or torch.isfinite(value).all() for value in state.values())

record = {'checkpoint': str(checkpoint), 'output': str(output),
          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'source_sha256': manifest['sha256'], 'source_files': manifest['files'],
          'simulator_source': saved['simulator_source'], 'config': saved['config'],
          'updates': saved['updates'], 'total_steps': saved['total_steps'],
          'total_episodes': saved['total_episodes'], 'signature': actual_signature,
          'selection': 'last fully valid completed-update checkpoint retained before timeout'}
with record_path.open('x') as stream:
    json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({key: record[key] for key in ('checkpoint', 'output', 'updates', 'total_steps')}))
PY
```

For banked or another seed, change the candidate/output/record paths **and** the
explicit `(model, seed)` assertion together. The recorded SHA256 binds the file
selected now; SoftRole did not originally publish a whole-checkpoint byte hash,
so it is not claimed to be a historical byte-hash comparison. Validation checks
the existing model signature and all active optimizer state before recording it.

## Submit a separate recovery job only after validation

This heredoc submits a new Slurm script through standard input without changing
the canonical launcher. It checks the recorded checkpoint and source again on
the compute node before executing. `SOFTROLE_*` variables above must remain
exported. Use this only when actually ready to submit; it is not a dry run.

```bash
sbatch --export=ALL --constraint=sapphirerapids \
  --job-name=softrole-paper-fc-resume --account=cenyioha --partition=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=48:00:00 \
  --output='logs_sr/softrole-paper-fc-resume-%j.out' \
  --error='logs_sr/softrole-paper-fc-resume-%j.err' <<'SBATCH'
#!/bin/bash -l
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the reviewed repository root}"
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
"$HETNET_PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
from softrole.train import source_snapshot
record = json.loads(Path(os.environ['SOFTROLE_RECOVERY_RECORD']).read_text())
checkpoint = Path(os.environ['SOFTROLE_CHECKPOINT']).resolve()
output = Path(os.environ['SOFTROLE_SEGMENT']).resolve()
assert str(checkpoint) == record['checkpoint'] and str(output) == record['output']
assert not output.exists()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record['checkpoint_sha256']
assert source_snapshot()['files'] == record['source_files'], 'Source changed while queued'
PY
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole train \
  --resume "${SOFTROLE_CHECKPOINT:?}" --output "${SOFTROLE_SEGMENT:?}"
SBATCH
```

Save the resulting job ID beside the recovery record. If another timeout occurs,
use a fresh next segment and the last valid checkpoint in that segment. Keep the
28M target, original model/seed/configuration, and retained checkpoint lineage.
Do not concatenate an abandoned parent's post-checkpoint ledger suffix with the
resumed work. SoftRole's current summarizer does not automatically assemble
training continuation lineages; use checkpoint update/count boundaries when
building the final study records.

The 46-hour graceful stop remains a **HetNet-only** guarantee in the current
launchers. Even for HetNet it is checked after a complete update and excludes
startup; the 48-hour allocation provides headroom. Neither method's early
preflight speed guarantees completion within one allocation.

Validation on 5 October: both shell blocks pass `bash -n`; both embedded Python
blocks parse; the actual SoftRole parser accepts the resume invocation; the
paper-FC recipe satisfies the explicit configuration assertions. The guard APIs
were checked against current source. These prepared guards have not been run
against a future research checkpoint, and no job or training was launched.
