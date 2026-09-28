# Mac fallback for initial HetNet experiments

**Historical fallback:** Stokes was restored on 28 September 2026. The current target is the full Stokes sweep in [STOKES_RUNBOOK.md](STOKES_RUNBOOK.md); these Mac pilot instructions are retained for reference.

Status, 26 September 2026: Akki selected **short pilots first, followed by measured-cost review**, on a separate Mac mini available all day. The local setup/calibration tools are provided below. The combined tooling/regression validation passed 53 tests on the current M2, including bounded gradient probes; no calibration or study training was launched. The 21-run grid and preregistered final analysis are unchanged, and the existing Stokes launcher has not been relaxed. Target Gate A and calibration results remain pending.

## Target handoff and commands

The delivery bundle is a portable copy of this existing fork's committed branch, including its history and baseline tag. Copy `runs/mac_handoff/frozen-eval-mac-pilots.bundle` and its manifest from the development machine to the Mac mini. The bundle excludes ignored virtual environments and raw runs. For a new dedicated working copy on the target, replace the bundle path below with its actual location:

```bash
git clone --branch frozen-eval /absolute/path/to/frozen-eval-mac-pilots.bundle HetNet-mac-pilots
cd HetNet-mac-pilots
git remote set-url origin https://github.com/akki-g/HetNet.git
git remote add upstream https://github.com/CORE-Robotics-Lab/HetNet.git
git branch main origin/main
```

Keep this checkout fixed throughout setup, Gate A and both calibrations. Do not copy `.venv` from the M2. If uv 0.12.5 is not already installed, install it into this checkout using the versioned official installer; the commands preserve the installer and avoid changing shell startup files. [uv installation](https://docs.astral.sh/uv/getting-started/installation/), [installer options](https://docs.astral.sh/uv/reference/installer/).

```bash
mkdir -p .tools runs/mac_bootstrap
curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.12.5/install.sh \
  -o runs/mac_bootstrap/uv-install.sh
UV_UNMANAGED_INSTALL="$PWD/.tools" sh runs/mac_bootstrap/uv-install.sh
.tools/uv --version
```

Create a fresh locked environment and preserve hardware/import evidence. If uv 0.12.5 is already available elsewhere, replace `.tools/uv` with that executable. Setup may download managed Python 3.12 into the checkout's `.tools/` directory; it does not replace system Python.

```bash
python3 -m hetnet_ext.local_setup \
  --venv .venv-mac-pilot --output runs/mac_setup/attempt01 --uv .tools/uv
```

Stop on any error and retain that attempt. Existing output/environment paths are deliberately refused; subsequent attempts require new names and corresponding path changes below. Inspect `runs/mac_setup/attempt01/evidence.json` to resolve the M4/M4 Pro and memory uncertainty.

From the completed environment, run the local launcher/setup contracts and fresh full Gate A. These commands perform engineering tests, not the full study:

```bash
.venv-mac-pilot/bin/python -m pytest -q tests/test_local_setup.py tests/test_local_job.py
DGLBACKEND=pytorch PYTHONUNBUFFERED=1 .venv-mac-pilot/bin/python -m hetnet_ext.gate_a \
  --output runs/gate_a/mac_attempt01 --run-smokes \
  --clean-environment-evidence runs/mac_setup/attempt01/evidence.json \
  --allowed-upstream-commit 2fadecf --allowed-upstream-commit 7b334c1 \
  --allowed-upstream-commit 0cfcea5 --allowed-upstream-commit 50d0c37
```

The archived M2 pass remains historical evidence; adding the local launcher changes the source inventory, so it cannot authorize a new-revision pilot. Use the fresh passing target report. If Gate A fails, preserve the failure and follow the original gate rules before any learner/task change.

After those checks pass, run the two authorized timing pilots sequentially. This block uses a **two-hour child-runtime limit per calibration** as an initial operational cap, not an estimate of runtime; it stops before the second job if the first fails or times out. Each job still requests exactly 20 epochs with the original per-epoch recipe. An incomplete attempt stays incomplete; review it before repeating with a larger cap. Polling, resource sampling, cleanup and final validation can add time beyond the child-runtime limit. The cap is not a change to successful-run training settings.

```bash
bash -e <<'SH'
for index in 0 1; do
  caffeinate -i .venv-mac-pilot/bin/python -m hetnet_ext.local_job \
    --index "$index" --gate-a runs/gate_a/mac_attempt01/gate_a_report.json \
    --output-root runs/mac_calibration/attempt01 --max-wall-seconds 7200
done
SH
```

Keep that terminal/session open. `caffeinate` prevents idle sleep while its command runs; it is not a restart or terminal-detachment mechanism. No full 21-run launch is included. For command inspection before execution, append `--dry-run` to one `local_job` invocation.

After completion—or a failure—preserve the setup/Gate A directories and both calibration directories. Return `provenance.json`, `config.json`, `resolved_args.json`, `metrics.jsonl`, `checkpoint_records.jsonl`, resource observations and logs for measured-cost review; retain checkpoints and signatures locally for validation. Do not start the proposed 300-epoch learning pilots from this runbook automatically. Frozen testing will follow the original immutability/intervention gates once its runner is implemented.

## Feasibility and hardware evidence

The existing CPU path is a credible starting point: [Gate A](GATE_A_REPORT.md) passed on macOS ARM64 with the locked dependencies, all five compositions, and both one/four total processes. A subsequent [hardware readout](evidence/local_mac/current_host_20260926.json) identifies this workspace host as an **Apple M2, 8 GiB RAM, four performance and four efficiency cores**. It is not the proposed target. Therefore the archived smoke times must not be described as M4 measurements. Akki reports a base M4 Pro with 16GB; Apple's listed base M4 Pro has 12 CPU cores (eight performance/four efficiency) and 24GB, while the base M4 has ten cores (four performance/six efficiency) and 16GB. This discrepancy is unresolved; read the actual chip/RAM through the setup helper rather than silently choosing a specification. The sequential CPU plan applies to either target, subject to measured memory and runtime. [Apple Mac mini specifications](https://support.apple.com/en-us/121555).

Use **CPU execution**, with the same Python 3.12/Torch 2.2.1/DGL 2.1.0 lock. The current selected path chooses CPU unless `--use_cuda` is supplied (`main.py:261–262`), and sets float64 defaults (`main.py:37`) while retaining explicitly float32 attention parameters. PyTorch's exact v2.2.1 MPS implementation rejects float64 allocation. Moving this experiment to MPS is therefore not a device-only change; casting the model would change the tested numerical setup. [PyTorch v2.2.1 `EmptyTensor.cpp`, lines15–16,40,93](https://github.com/pytorch/pytorch/blob/v2.2.1/aten/src/ATen/mps/EmptyTensor.cpp). DGL 2.1 documents macOS and CPU/CUDA builds; it does not establish an MPS path for this model. The local gate supplies the concrete compatibility evidence for the selected lock. [DGL 2.1 installation documentation](https://www.dgl.ai/dgl_docs/en/2.1.x/install/index.html).

## Recommended sequence

| Stage | Work | What it establishes |
|---|---|---|
| Target verification | Record M4 CPU, RAM, OS, free disk and availability; synchronize the locked environment; validate source/lock hashes and run Gate A on that host | Compatibility and engineering correctness on the actual target |
| Authorized short calibration pilots | Sequential 20-epoch 2P1A and 4P6A jobs, seed0, full per-epoch recipe | Runtime, memory pressure and storage estimates; not converged-policy results |
| Measured-cost review | Inspect both complete calibrations and the calendar/storage projection before choosing longer work | Implements Akki's requested pause between short pilots and larger computation |
| Initial learning pilot, proposed after review | 2P1A seeds0–2 through epoch300, with the unchanged per-epoch recipe and saved checkpoints | Whether the source learns consistently enough to justify more computation |
| Frozen diagnostic development | During the learning pilots, build the original-model frozen runner and Gate B in a separate checkout; use fixed smoke checkpoints/banks | Immutability, pairing, sensor isolation, message removal and permutation behavior |
| Expansion decision | Use measured costs and source learning curves to choose continued pilots or the full protocol | A concrete local time budget, with any change to final study scope explicitly recorded |

The 300-epoch pilots are a **proposed separate diagnostic scope after cost review**, not yet authorized by all-day machine availability and not a silent reduction of the required 2,000-epoch runs. Do not relabel a pilot as a complete preregistered run. Initial frozen diagnostic outputs use separate smoke banks and clearly identified checkpoints, never the final 500-entry banks for tuning. The task's specifically required fixed-oracle final-bank check remains as preregistered. No frozen runner exists yet; the cross-composition loading test alone does not implement one.

Start with **one training job at a time and four total collectors per job**, one Torch/OMP/MKL thread per collector. More available CPU cores do not imply that increasing `nprocesses` is free: upstream collects a full batch per process, so that would change data per update and the total sample budget. Likewise, replacing four collectors with one would reduce the nominal sample budget fourfold. If two concurrent four-collector jobs are considered later, measure their combined throughput and memory pressure on the target; do not infer a twofold speedup from core counts.

Keep dimension 5, vision 2, horizon 80, batch target 500 per collector, epoch_size 10, RMSprop learning rate 1e-4, detach_gap 5, and the original architecture. The existing small-batch Gate A timings cannot substitute for this calibration. Save every 50 completed epochs plus each declared pilot/calibration endpoint. Preserve failed attempts and all seeds, including flat learners.

## Costs to measure

For four collectors, the nominal minimum is

\[
S_{\mathrm{epoch}}=10\times4\times500=20{,}000
\]

joint environment transitions. Each collector finishes its last episode, so the actual count may exceed this; record the JSONL totals. At horizon 80, the current collection rule bounds the count by 23,160 per epoch. Each 20-epoch calibration therefore has 400,000–463,200 joint transitions (800,000–926,400 for both). A proposed 300-epoch pilot has 6M–6.948M, while a 2,000-epoch run has 40M–46.32M. The full 21-run grid requires 840M–972.72M. These are counts, not runtime predictions. See [the reproduction ledger](REPRODUCTION_LEDGER.md).

Let measured steady epoch time for composition c be `s_c`, checkpoint-save overhead be `q_c`, and one-time startup cost be `b_c`, all in seconds. A preliminary projection is

\[
T_c(E)\approx b_c+E s_c+K(E)q_c,
\quad K(E)=\lfloor E/50\rfloor+\mathbf1[E\bmod50\ne0].
\]

Use the nearest-rank 90th percentile of epochs 2–20 (the eighteenth of nineteen sorted durations), account for first-epoch excess and unlogged overhead without double counting, and add a stated contingency. More precisely, for observed enclosing job duration L, independently measured in-job setup S, epoch durations t1…t20 and the single final save q, let R=L−S−sum(t)−q. A conservative planning convention is S+max(0,t1−p90)+E*(p90+R/20)+K(E)*q. Reject negative residuals. R combines logging/signatures, imports/spawn not included in S, shutdown and other overhead; amortizing it per epoch is an assumption, not a measured decomposition. If setup was not timed separately inside L, leave it unclassified in R and omit S rather than inventing a setup measurement. Environment installation outside L is separate campaign overhead. Timing can change as episode lengths and learning change; revisit the estimate at the pilot checkpoint. Checkpoint histories can grow, so the epoch 20 save cost/size is preliminary. For sequential full training, total projected time is the sum across all 21 runs. Endpoint calibration supplies an initial envelope for intermediate compositions, not measurements of them.

The following is a **hypothetical sensitivity table**, assuming the same seconds/epoch for every composition and excluding startup, saves, downtime and evaluation. None of these entries is an M4 benchmark:

| Assumed seconds/epoch | One 2,000-epoch run | 21 runs, sequential 24h/day |
|---:|---:|---:|
| 30 | 16.7 hours | 14.6 days |
| 60 | 33.3 hours | 29.2 days |
| 120 | 66.7 hours | 58.3 days |
| 300 | 166.7 hours | 145.8 days |

Available hours/day and concurrent-job slowdown must be measured separately. Even a technically compatible workstation can have a substantial full-study calendar cost. Frozen inference needs its own timing measurement once implemented; forward-only execution removes training work but does not eliminate environment/graph/Python overhead. Budget the complete required 54,000 episodes separately, rather than applying an unsupported speedup factor.

Measure total-process memory pressure and swapping during full-batch calibration. The parent process's RSS alone does not establish the four-process peak; summing RSS can double-count shared pages. Record the measurement method and scope. Estimate disk requirements from actual checkpoint sizes at the final cadence: 40 checkpoints/run, 840 across the full grid, plus signatures/logs, failed attempts, evaluation artifacts and backups. Do not assume the 8GiB smoke host proves full-batch memory sufficiency.

## Local execution preparation

On the target, collect these read-only facts from the HetNet checkout:

```bash
sysctl -n machdep.cpu.brand_string
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
sw_vers
uname -m
df -h .
git rev-parse HEAD
git status --short
```

Preserve command output with a timestamp. Create the target's environment from `uv.lock`; do not copy another machine's virtualenv. Use the existing [Gate A reproduction command](STOKES_RUNBOOK.md) with a fresh output directory and target clean-environment proof. No package upgrade or precision conversion is required by this plan.

The existing `hetnet_ext.train_job` deliberately requires Stokes/Slurm. The separate [local launcher](hetnet_ext/local_job.py) reuses the grid command builder and recorder, with no fabricated scheduler variables. It provides only the two authorized 20-epoch calibration entries, unique directories, resolved arguments, source/lock/Gate A validation, target hardware/runtime binding, a per-user lock, and whole-process-group interruption/timeout cleanup. Completion requires all 20 epoch metrics/signatures and a matching final checkpoint with finite tensors and preserved names, shapes, mixed dtypes and values. The model metadata comes from the fresh Gate A cross-composition load artifact and its hash is recorded and rechecked. No full-study or resume mode is included.

`resource_samples.jsonl` records sampled training-process-group RSS, member process counts and system-wide swap/VM observations. The orchestration parent is outside that sampled group and also uses memory for validation. `provenance.json` records `hardware`, `runtime`, `launcher_wall_time_seconds`, `child_wall_time_seconds`, `setup_wall_time_seconds`, `resource_summary` and before/after observations. The RSS sum and system-wide snapshots retain their explicit measurement limitations; they are not an exact whole-job memory peak. In-job setup measures launcher preflight/import work before launching the child; startup within the training child remains in the residual. Environment setup has its own elapsed time in the [setup helper's](hetnet_ext/local_setup.py) evidence. No Stokes allocation fields or monthly-balance assumptions are invented.

The command block above keeps the machine awake while the process runs. Preserve evidence if the terminal disconnects or the run is interrupted. Checkpoint saves are not true resume: the current loader does not restore all optimizer/scheduler/RNG state. A stopped pilot must not be continued as a faithful full run using weights alone. Larger local work requires the requested measured-cost review; cluster allocation thresholds do not describe a personal Mac's cost.

Adding the launcher/tests changes the scientific inventory used by Gate A, even when model math is untouched. Review and rerun the affected gate on the final source revision before training. Keep long-running training in an immutable checkout and develop Phase B in a separate worktree/environment, as already planned for Stokes.

## Interpretation and preregistration

Changing execution hardware alone does not change the intended mathematical policy or estimands, but equal seeds are not evidence of identical trajectories across platforms. Record hardware for every run; do not silently mix partial/restarted runs across devices. If the final grid, epochs, seeds, banks or contrasts are reduced, commit a prospective amendment before final outcomes and report the actual scope.

Short source-only pilots can establish source learning and exercise frozen mechanics. They cannot establish a native-minus-frozen transfer gap without matched destination-native references. If a smaller scientific pilot is chosen, 3P1A and 2P2A provide distinct first diagnostic targets under the [relation-support audit](research/RELATION_SUPPORT_AUDIT.md), but their selection, training budgets and references must be declared before inspecting those outcomes. A weak partially trained source is not evidence that SoftRole is needed. The original [analysis plan](ANALYSIS_PLAN.md) remains the plan for the complete study.
