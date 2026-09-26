# Stokes source audit and deployment design

Research-only audit, 26 September 2026. The complete revision-2 task attachment was read. No environment was installed, no HetNet training or import was executed, no cluster connection was attempted, and no Slurm job was submitted. Work is in Akki's existing `HetNet` fork on `frozen-eval`; `marl-comm/slurm/run_experiment.sbatch` was read without modification. The commands below specify future gates and do not report completed preflight checks.

## 1. Publicly verified facts and live unknowns

| Topic | Verified primary-source statement | What remains unknown |
|---|---|---|
| Scheduler and allocation | ARCC uses Slurm. Stokes' standard faculty allocation is 80,000 core-hours/month; Newton's is 10,000 CPU core-hours plus 2,000 GPU-hours. Unused allocation does not roll over; `myusage` reports balance. [Scheduler introduction, “Slurm Allocations”](https://arcc.ist.ucf.edu/docs/scheduler/) | Akki's PI account, current balance, and competing use by the group |
| Partitions and priority | Both clusters have `normal` and `preemptable`; ordinary use is `normal`. Documentation describes preemption after allocation exhaustion and a 14-day fairshare history. Faculty-account use is shared. [Limitations, “Queue Limits” and “Account Limits”](https://arcc.ist.ucf.edu/docs/scheduler/limitations/) | Actual default partition, `normal` time cap, account requirement, QoS, concurrency and submission limits |
| Storage | Stokes home paths are `/home/<username>` on `/lustre/fs1`; documented user limits are 1 TB and 1,000,000 files. Group files are under `/groups/<groupname>` and charged to the owner. User/group storage is not backed up. [File System Policy, “Stokes”](https://arcc.ist.ucf.edu/docs/data/files/) | Current quota usage, filesystem identity across clusters, and available free space |
| Software catalog | Catalog lists `anaconda-2023.09`, `anaconda-2024.10`, `apptainer-1.3.3-go-1.22.5`, and `jobstats-1.0`. [Available Modules](https://arcc.ist.ucf.edu/docs/software/availableModules/) | Which exact module names are available on Stokes and what Python they expose |
| Installation location | ARCC requires package/environment creation on a compute node. Its Anaconda job example loads `anaconda/anaconda-2024.10`. [Anaconda, “Creating a Basic Environment” and “Using Anaconda Inside a Slurm Script”](https://arcc.ist.ucf.edu/docs/software/anaconda/) | Compute-node connectivity to the package and binary hosts |
| Operating system | ARCC reports the August 5–9, 2024 conversion to Rocky Linux 8.x and specifically 8.7 in its maintenance explanation. [Completed maintenance notice](https://arcc.ist.ucf.edu/index.php/news/summer-2024-downtime-maintenance-completed) | Current OS/kernel/glibc on the allocated compute node |

The CPU count is a documented inconsistency, not a calibrated resource estimate. The [ARCC home page](https://arcc.ist.ucf.edu/) says **over 8,800** Xeon cores; [About Stokes](https://arcc.ist.ucf.edu/index.php/resources/stokes/about-stokes) says **7,500**; the [proposal facilities statement](https://arcc.ist.ucf.edu/docs/support/proposals/) says **over 8,000**. Use the home-page description only as general context, preserve this discrepancy, and use live scheduler output for scheduling. None of these counts imply that 21 tasks can run concurrently.

The official [submission guide](https://arcc.ist.ucf.edu/docs/scheduler/scripts/) documents `sbatch`, `squeue`, and interactive `srun --pty bash` with requested cores, memory and time. Its one-hour examples are examples, not a Stokes partition maximum. The existing Newton wrapper's 48-hour request also does **not** establish a Stokes cap.

## 2. Required live preflight evidence

Akki should run the attachment's read-only commands on **Stokes** and preserve the complete outputs with timestamps:

```bash
scontrol show partition normal
sinfo -p normal -o "%P %l %c %m %D"
sacctmgr show assoc user="$USER" format=Account,Partition,MaxJobs,MaxSubmit,GrpTRES
myusage
quota -s
module avail anaconda
```

If ordinary quota output is uninformative, use `lfs quota -h -u "$USER" /lustre/fs1`. Blank or inaccessible association columns do not establish unlimited use; the relevant account/QoS limits still need clarification. Also record `hostname`, `date -u`, `uname -a`, `/etc/os-release`, `lscpu`, `getconf GNU_LIBC_VERSION`, the resolved working-directory path, and `df -h .`. These are proposed provenance fields, not facts already observed.

After Gate A and live scheduler checks, obtain the attachment's short compute allocation:

```bash
srun --partition=normal --time=1:00:00 --cpus-per-task=4 --mem=4G --pty bash
```

Add `--account` only if the returned association/preflight requires it. The preflight must confirm access from that compute node to PyPI metadata and wheel hosts (`pypi.org`, `files.pythonhosted.org`), GitHub release downloads, and `astral.sh`. If selecting the optional CPU-only Torch index, also check `download.pytorch.org` and the download host to which it redirects. A successful request to a landing page does not prove a large artifact download will succeed.

No automatic Newton or `preemptable` fallback is authorized. The attachment requires Akki's decision if a module lacks Python 3.12, network access is blocked, limits are unresolved, or a cluster switch is considered. For network isolation, a staged wheelhouse/offline install **inside a compute allocation** is a possible design to discuss. Installing on the login node is not the documented ARCC default; the official installation-location rule above must be considered before approving that fallback.

## 3. Reusing the working wrapper precisely

The read-only reference is `../marl-comm/slurm/run_experiment.sbatch`, lines 28–53. It clears Python environment overrides, purges/loads modules, activates Conda base, verifies Python 3.12, obtains uv 0.12.5 under `.tools/`, serializes environment setup with `flock`, synchronizes from a locked dependency file, activates the venv, and starts Python. Its resource request, two-config array, Newton-specific error text and `.venv-newton` are project-specific and must not be copied as Stokes facts.

Future HetNet wrapper design, after the research gate:

1. Require an active allocation and repository-root submission. Preserve an actionable failure message for each setup stage. Initialize with `set -eo pipefail`; enable `set -u` after module/Conda initialization, as the working wrapper does.
2. Clear `VIRTUAL_ENV`, `PYTHONHOME`, `PYTHONPATH`; run `module purge`, `module load anaconda/anaconda-2024.10`, `conda activate base`. Print module state and capture the interpreter returned by a Python-3.12 assertion. Do not silently choose another Python.
3. Set the uv executable directory to `.tools/uv-0.12.5` and `UV_PROJECT_ENVIRONMENT` to the absolute repository path plus `.venv-stokes`. Create/use a distinct `.tools/stokes-setup.lock`. Check `flock` exists before setup.
4. Within an exclusive lock, install the version-specific uv binary if absent, verify its version, then run `uv sync --locked --python <verified-interpreter> --no-python-downloads`. All installing/syncing happens in the allocation. Record the committed `uv.lock` hash; a stale lock must fail rather than be regenerated.
5. Activate `.venv-stokes`, check imported versions and run the agreed lightweight import/smoke gate before launching the selected grid entry. These checks have not yet been performed.
6. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and configure Torch's thread count to one before training work. Request four CPUs for four HetNet processes. No GPU resource request is needed.

The explicit environment-path mechanism and locked-sync semantics are documented by [uv project configuration](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path) and [locking/syncing](https://docs.astral.sh/uv/concepts/projects/sync/). In particular, `--locked` checks consistency with project metadata; it is not equivalent to trusting a possibly outdated lock via `--frozen`. The version-specific uv release exists on [PyPI](https://pypi.org/project/uv/0.12.5/).

Setup serialization prevents simultaneous installs; it does not justify changing the lock while jobs are running against the same environment. Freeze the environment and code revision for the run series, and include setup/lock-wait time in job-budget accounting. Use Slurm output names containing both array/job identifiers, and create the output directory before submission since scheduler output files can be opened before the script body executes.

## 4. Wheel and native-library evidence

The companion [STOKES_PACKAGE_EVIDENCE.json](STOKES_PACKAGE_EVIDENCE.json) records official PyPI URLs, selected filenames, sizes, dependency metadata and SHA-256 values. Two small DGL wheel archives were downloaded into memory for static archive inspection, their hashes verified, and no contents installed or imported. Torch binaries were not downloaded.

| Package | CPython 3.12 Linux x86-64 evidence | CPython 3.12 macOS ARM evidence |
|---|---|---|
| Torch 2.2.1 | `torch-2.2.1-cp312-cp312-manylinux1_x86_64.whl` | `torch-2.2.1-cp312-none-macosx_11_0_arm64.whl` |
| DGL 2.1.0 | `dgl-2.1.0-cp312-cp312-manylinux1_x86_64.whl` | `dgl-2.1.0-cp312-cp312-macosx_12_0_arm64.whl` |
| TorchData 0.7.1 | A `py3-none-any` wheel is available | Same pure-Python fallback wheel |

Sources: official release files for [Torch](https://pypi.org/project/torch/2.2.1/#files), [DGL](https://pypi.org/project/dgl/2.1.0/#files), and [TorchData](https://pypi.org/project/torchdata/0.7.1/#files). Wheel existence is evidence of a compatible distribution tag, not a passed runtime gate. The Mac DGL wheel specifically requires macOS 12 or later. Rocky Linux 8 is consistent with the manylinux tags, but native imports must still be tested on the actual node.

Both inspected DGL archives contain GraphBolt libraries for Torch **2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0 and 2.2.1**. The [DGL v2.1.0 GraphBolt loader](https://github.com/dmlc/dgl/blob/v2.1.0/python/dgl/graphbolt/__init__.py#L29) selects a library using Torch's version after removing a `+...` suffix. Thus the source handles a `2.2.1+cpu` version string by selecting the `2.2.1` library. This establishes filename resolution, not successful loading of that optional CPU distribution. The requested exact Torch 2.2.1 pin remains the experimental choice; the archive does not mean that 2.2.1 is the only version for which any library is present.

The default Linux PyPI Torch 2.2.1 metadata includes NVIDIA CUDA component dependencies even for a CPU workload. The [official CPU index](https://download.pytorch.org/whl/cpu/torch/) lists `torch-2.2.1+cpu-cp312-cp312-linux_x86_64.whl`, providing a separate distribution route. Switching to it should be a recorded dependency decision with a complete import/forward test before committing the lock. Pin TorchData 0.7.1 as requested; a generic latest TorchData is not an equivalent dependency choice for DGL's older `torchdata.datapipes` imports.

## 5. DGL 2.1 normalization: the precise scope

The version-specific docs URL attempted during this audit did not resolve. The current DGL documentation identifies itself as 2.5, so it must not be mislabeled as 2.1. The exact semantics are instead checked against the official [v2.1.0 operator source](https://github.com/dmlc/dgl/blob/v2.1.0/python/dgl/ops/edge_softmax.py#L12) and the matching file inside the verified 2.1.0 wheel.

The operator defaults to `norm_by="dst"`: for sender-to-receiver edge `j → i`, it normalizes over the incoming edges of receiver `i`. HetNet calls it on each relation subgraph separately, for example [upstream `fastreal.py:198–211`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/graph/fastreal.py#L198). Therefore its denominator is the incoming neighborhood **within that relation**. The per-relation property follows from the caller's sliced graph, not merely from using a heterograph API or receiving a dictionary of edge outputs.

For relation `r`, the mathematical interpretation is

\[
\alpha_{ij}^{(r)} = \frac{\exp(e_{ij}^{(r)})}{\sum_{k:(k\to i)\in E_r}\exp(e_{ik}^{(r)})}.
\]

The research notation writes receiver first; DGL graph construction takes source indices first. Keep that mapping explicit. Upstream guards empty relation branches; the later message-removal test must check the unchanged forward path produces zero relation contribution and finite logits. This audit does not report that runtime test as passed.

## 6. Calibration and honest compute accounting

The attachment's sandbox estimate is a user-provided prior, **not Stokes timing**. Its appendix records approximately 22 seconds/epoch for one process. The production recipe uses four processes, and both the data volume and parallel execution differ. Do not multiply or divide the one-process wall time by four without measurement.

Read-only code facts at upstream `bff9f7f`:

- [`main.py:36–49`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/main.py#L36) defines `epoch_size=10`; it is the number of updates **inside** one reported epoch.
- [`main.py:387–409`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/main.py#L387) runs that update loop.
- [`multi_processing.py:47–58,85–112`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/multi_processing.py#L47) creates `nprocesses−1` workers plus the main collector and combines their gradients/step totals.
- [`trainer.py:512–526`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/trainer.py#L512) appends complete episodes until the batch reaches the requested minimum, so actual steps can exceed `batch_size`.

With `T` epochs, `E=epoch_size`, `P` processes, per-process target `B`, and horizon `H`, the nominal minimum environment-transition count is

\[
S_{\min}=T E P B.
\]

If every episode has at most `H` transitions, full-episode collection gives the bound

\[
T E P B \leq S_{\mathrm{actual}} \leq T E P (B+H-1).
\]

For `T=2000, E=10, P=4, B=500, H=80`, this is **40,000,000 to 46,320,000 environment transitions per training task**, with 20,000 optimizer updates. At 21 tasks the corresponding interval is 840,000,000 to 972,720,000 transitions. Agent decisions multiply transition counts by the composition's agent count. Report actual collected steps, not this upper bound as an observed value.

**Logging caution:** `main.py:401–404` merges the latest batch into `stat` and then increments cumulative counters using the already accumulated `stat`. Repeating this inside the epoch can overcount steps/episodes. New metrics should sum the raw batch return `s` once per update or use the completed epoch aggregate once. Correcting external metric accounting is distinct from changing collection or gradient normalization.

### Timing model and slopes

For each calibrated composition `c`, record total allocation time, setup/initialization time `a_c`, measured epoch durations, measured update counts, and actual transition counts. Define a steady training slope `β_c` in seconds/epoch at the **same** `E,P,B,H` as production. Also report seconds/update and seconds/actual-transition as checks. Startup timing should not be hidden inside a 20-epoch slope that is then multiplied by 2,000.

A planning model is

\[
\widehat W_c = a_c + T\,\beta_c + N_{\rm save}\,q_c,
\qquad N_{\rm save}=40
\]

when `β_c` excludes checkpoint writes, with `q_c` a measured save cost and duplicate final writes avoided. Alternatively include checkpoint cost in the measured slope and do not add it twice. A 20-epoch calibration with a 50-epoch checkpoint interval still needs an explicit final save to measure checkpoint size and latency.

The requested calibration covers 2P+1A and 4P+6A. The three intermediate compositions do **not** thereby receive measured slopes. In their budget rows, either label slope extrapolations explicitly, use a declared conservative envelope from the endpoints without claiming a proof of monotonicity, or request a small additional timing probe. The sandbox ratios are unverified priors and must remain labeled that way.

For requested cores `C_c=4` and seed count `n_c`,

\[
\widehat {\mathrm{CH}}_{\rm train}
=\sum_c n_c C_c\widehat W_c/3600.
\]

Use allocation wall time, including setup/waiting inside an allocated job, for the budget estimate. `TotalCPU` measures utilization and is useful diagnostically; it is not interchangeable with allocated core-hours. Add calibration, setup and evaluation costs explicitly. The official [allocation guide](https://arcc.ist.ucf.edu/docs/scheduler/#slurm-allocations-within-the-ucf-arcc) describes core-hour accounting as cores times runtime.

| Composition | Seeds | Slope source | Projected core-hours |
|---|---:|---|---|
| 2P+1A | 5 | Stokes calibration pending | `5 × 4 × W_2P1A / 3600` |
| 3P+3A | 5 | Not directly calibrated by the requested two-task pilot | `5 × 4 × W_3P3A / 3600` |
| 4P+6A | 5 | Stokes calibration pending | `5 × 4 × W_4P6A / 3600` |
| 3P+1A | 3 | Not directly calibrated by the requested two-task pilot | `3 × 4 × W_3P1A / 3600` |
| 2P+2A | 3 | Not directly calibrated by the requested two-task pilot | `3 × 4 × W_2P2A / 3600` |

Before full submission compute `100 × projected_cost / remaining_balance` from a fresh `myusage`. The attachment requires stopping for Akki's approval if the projection exceeds **4,000 core-hours or 10% of remaining balance**, or if per-task wall time exceeds the live cap. Full-array submission follows calibration and explicit budget confirmation. Do not reduce epochs/batch size to fit silently.

Set requested wall time to approximately `1.3 × W_c`, including known setup/checkpoint overhead, only after verifying it fits the partition limit. Memory should be 1.5 times the measured **whole job** peak; summing process RSS may overcount shared pages, while a single process's peak may understate the multiprocess allocation. Record the accounting tool and measure available Slurm/jobstats/cgroup whole-job evidence. The module catalog proves `jobstats` is listed, not which metrics it exposes on the actual job.

## 7. Ordered gates and current status

1. Finish cited research notes and resolve paper/code choices before implementation.
2. Obtain live Stokes preflight; unresolved cluster facts remain blank.
3. Implement only the authorized deviations and dependency lock, then pass local Gate A, including multiprocess shared weights and all required compositions.
4. Submit the short Stokes calibration after the local gate; capture real timing, memory, saves and budget.
5. Obtain Akki's budget confirmation; submit Run 1; preserve every failed task. Inspect the source-policy learning curves around epoch 300 before spending the remainder.
6. Finish Gate B and commit the analysis plan before Run 2 outcomes; validate all expected final checkpoints before chaining evaluation.

At the time of this audit, only the public documentation, wrapper source, package metadata/archive structure and upstream collection loop have been verified. No live-cluster gate, import gate, training gate or projected Stokes budget has passed.
