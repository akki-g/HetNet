# Agent instructions: original HetNet training + frozen-policy evaluation (UCF Stokes)

Owner: Akki (Akshat Guduru). Issued 26 September 2026 (revision 2: Stokes is the target cluster; work in Akki's existing fork; research-grade standard).
Read this entire file before doing anything. Follow the phase gates in order.

---

## 0. Mission

Two cluster runs, back to back, on **UCF's Stokes cluster**:

- **Run 1 (training).** Train the *original, unmodified* HetNet architecture (HetNet-Real, A2C branch) on Predator-Capture-Prey (PCP). Several team compositions, several seeds each.
- **Run 2 (frozen evaluation).** Load the policies trained at the source composition **2P+1A**, freeze them completely, and evaluate them with no parameter updates:
  - at other compositions;
  - under a mid-episode perception-sensor failure;
  - with messages removed.

  Compare against natively trained policies, a scripted feasibility reference, and an untrained floor.

This is the "T0" regime of the implementation report: train on one small, stationary composition, then test transfer frozen. It decides whether SoftRole is worth building.

**Out of scope for this task.**
- No SoftRole, role gates, soft attention, new critic or new learner.
- No within-episode arrivals or departures.
- No BenchMARL/TorchRL port.
- Do not touch `marl-comm` / `commstudy`. This project is separate.
- No composition with **one P agent** (e.g. 1P+1A). Singleton-P rosters crash in HetNet's LSTM hidden-state handling (`RuntimeError: hidden0 has inconsistent hidden_size`). Singleton-A compositions (2P+1A, 3P+1A) work.

---

## 0.1 The goal is a research-grade paper, not a novelty claim

Akki wants a paper that shows he can do **careful, honest research on real, published work**. That means:

- faithfully reproducing a known method;
- asking a well-posed question about how it behaves outside its training conditions;
- answering it with controlled experiments;
- reporting exactly what the evidence supports.

Novelty is explicitly **not** the objective. Do not add conditions, losses or architecture changes to make the work look new. A clean negative or null result, reported rigorously, is a success. A flashy but confounded result is a failure.

**Working research question.** Does a heterogeneous communication policy (HetNet) trained at one team composition remain effective, with frozen parameters, when:
- the team composition changes; or
- a perception agent loses its sensor?

And how much of its performance depends on its learned messages?

Build everything with these research-grade practices. Each one has a concrete deliverable:

1. **Pre-registered analysis.**
   - Before anyone looks at Run 2 outcomes, commit `ANALYSIS_PLAN.md`. It states: primary endpoint, contrasts, statistical method, checkpoint-selection rule, exclusion rules, and what will be called exploratory.
   - Record its git commit hash in `RESULTS.md`.
   - Anything decided after seeing results is labeled *exploratory*.
2. **Controls and references, not just a method.**
   - Every frozen number is read against a native-trained reference at the same composition.
   - Also against a scripted oracle (physical feasibility) and an untrained-policy floor.
   - Without these, the frozen numbers are uninterpretable.
3. **The training seed is the unit of evidence.**
   - Show every seed; never cherry-pick.
   - Use seed-level uncertainty.
   - Failed or crashed runs are reported, not rerun silently and hidden.
4. **One change at a time, fully traceable.**
   - Each deviation from upstream is its own commit and a row in `DEVIATIONS.md`.
   - Each number in `RESULTS.md` traces to a run directory, a commit, a config and a log file.
5. **Reproducibility.**
   - Locked environment.
   - Seeded everything, with seeding verified by a test.
   - Provenance files for every job.
   - One command regenerates every table and figure from raw logs.
6. **Tests verify scientific claims, not just code health.** Examples: frozen weights never change; blinding one agent changes no one else's observation; paired conditions really see identical initial states.
7. **Claims proportional to evidence.** Write limitations alongside results, not after. Use the interpretation guardrails in §6.
8. **Reproduction first.**
   - Show 2P+1A learning curves for all seeds.
   - Compare qualitatively with what the HetNet paper reports.
   - Document discrepancies (e.g. paper vs. code optimizer) instead of hiding or "fixing" them.

**Speed vs. correctness.** Time is short, so prefer the smallest change that works. But this project's standing rule is that *silent confounds are the primary risk*: code that passes smoke tests while producing wrong conclusions. The gates below exist to catch that. Never skip a gate to save time; ask instead.

---

## 1. Research first: produce `RESEARCH_NOTES.md` before writing code

Do significant reading. Write a cited `RESEARCH_NOTES.md` in the repo, with page, section, equation or file:line for every claim. End it with a **decisions table**: setting, value, and source (paper / README / code default / our choice).

### 1.1 Project documents (in `akki-g/marl-comm`, folder `softrole/`)

| Document | What to take from it |
|---|---|
| `SOFTROLE_FEASIBILITY_REPORT.pdf` (Markdown source `.md` alongside) | **The mathematical basis.** §4, Eqs. 3–6: open-team state, receiver-first mask, frozen θ at evaluation = "adaptation by inference". §5, Eqs. 7–8: HetGAT's **class-wise** attention normalization. §7.1, Prop. 1, Eqs. 22–23: equivariance and its assumptions. §7.4: Eq. 28, equivariance does **not** imply size generalization; Eq. 29, attention dilution. §7.5, Prop. 3, Eq. 31. §7.6, Eq. 33: diagnosis limit. §8.2: separate the distribution shifts. §8.4: Eq. 36 recovery-loss contrast, censoring, report raw outcomes. |
| `SOFTROLE_IMPLEMENTATION_REPORT.pdf` (Akki will place it in `softrole/`; ask if missing) | §2: verified author-code facts (optimizer, seeding order, layer widths, paper/code discrepancies). Eq. (I1): observation width. §7: PCP completion semantics and observation-aliasing risk. §9: frozen-runner contract, T0/T1/T2 regimes. §10: acceptance tests. §11: statistics. |
| `research/MATHEMATICAL_AUDIT.md` | Exact HetGAT reconstruction (H1–H3) with line refs into `fastreal.py`; the 1+3=4 vs 2 normalization counterexample. |
| `research/REPOSITORY_FEASIBILITY.md` | Background only. It audits `commstudy`, which we are *not* using. |
| `research/implementation_sources/envs/HETNET_SUPPLEMENT.txt` | Supplement: PCP on a 5×5 grid, 80-step horizon, seeds 0/1/2, Adam 1e-3 (differs from the released code). |

### 1.2 Primary papers (read the cited parts yourself; do not rely on summaries)

- **HetNet.** Seraj, Wang, Paleja, Martin, Sklar, Patel, Gombolay. *Learning Efficient Diverse Communication for Cooperative Heterogeneous Teaming.* AAMAS 2022. https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf
  - Read §4.2–4.4 (architecture; class-wise attention, Eqs. 2–4), §5 (MAHAC critic), §6.1 (PCP), §6.2 (Real vs. Binary), and §6.3.4 with Fig. 5c (compositions 2P/1C, 3P/3C, 4P/6C).
  - **Record the grid size and horizon used in Fig. 5c.** If the paper used larger maps for larger teams, our 5×5 natives are *new* configurations, not reproductions. Say so.
- **Code lineage.** IC3Net: Singh, Jain, Sukhbaatar. *Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks.* ICLR 2019. arXiv:1812.09755. HetNet's trainer, multiprocessing and environment wrappers come from it; understand `batch_size`, `epoch_size` and `nprocesses`.
- **Frozen composition-transfer precedents** (for protocol design, not baselines):
  - Howell et al., *Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities*, CoRL 2023 (PMLR 229).
  - Fu et al., *Capability-Aware Shared Hypernetworks (CASH)*, CoRL 2025 (PMLR 305), §5.2: frozen evaluation under mid-episode capability reduction.
- **Reporting and interpretation.**
  - Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021: seed-level uncertainty, `rliable`.
  - Lowe et al., *On the Pitfalls of Measuring Emergent Communication*, AAMAS 2019: learned reliance ≠ task necessity.
- **Library semantics.** DGL 2.1 docs for `dgl.ops.edge_softmax` and heterograph relations. Confirm normalization is per relation and per destination. DGL is src→dst; the SoftRole math is receiver-first.

### 1.3 UCF ARCC: Stokes is the target cluster

The workload is CPU-only: small graphs, DGL on CPU. Stokes is ARCC's CPU cluster, with roughly 8,500–8,800 Intel Xeon cores; ARCC's own pages differ. Record the source you rely on.

Allocation, per faculty account per month, shared by all users on that PI account, no rollover, checked with `myusage`:

| Cluster | Monthly allocation |
|---|---|
| **Stokes** | **80,000 core-hours** |
| Newton | 10,000 CPU core-hours + 2,000 GPU-hours |

Sources: https://arcc.ist.ucf.edu/docs/scheduler/ and https://arcc.ist.ucf.edu/docs/scheduler/limitations/

Known conventions (verify each on Stokes and record results):

- **Partitions.** Slurm with `normal` (default; use it) and `preemptable`. Preemptable is only for an exhausted allocation; its jobs can be paused. Priority is fairshare over the last 14 days.
- **Storage.** Stokes home directories are on `/lustre/fs1`, limited to 1 TB and 1,000,000 files per user, and **not backed up**. Group space is `/groups/<groupname>`, with quota charged to the file owner.
  Source: https://arcc.ist.ucf.edu/docs/data/files/
  Download results after runs.
- **Modules.** ARCC's module list includes `anaconda-2023.09`, `anaconda-2024.10`, `apptainer-1.3.3` and `jobstats-1.0`. The list is not per-cluster, so run `module avail anaconda` **on Stokes**.
  Source: https://arcc.ist.ucf.edu/docs/software/availableModules/
- **Environment pattern.** Reuse Akki's working Newton wrapper, `marl-comm/slurm/run_experiment.sbatch`:
  - `module purge`; `module load anaconda/anaconda-2024.10`; `conda activate base`; assert Python 3.12;
  - pinned `uv` (0.12.5) under `.tools/`;
  - `flock`-serialized `uv sync --locked` into a venv, done **inside a compute allocation**.

  On Stokes, name the venv `.venv-stokes` and verify compute-node outbound access to PyPI, GitHub and astral.sh. If there is no outbound access, fall back to building the venv on the login node (light I/O only) or to an Apptainer image; ask Akki first.
- **OS.** Rocky Linux 8 (ARCC migrated in Aug 2024). manylinux wheels for torch 2.2.1 and DGL 2.1.0 are compatible.
- **Interactive testing.** Use `srun --time=1:00:00 --cpus-per-task=4 --mem=4G --pty bash` for a compute node. Never run heavy work on the login node.
- **Unknowns you must not guess.** Wall-time cap of `normal` on Stokes; whether `--account=` is required; whether Stokes and Newton share home directories; per-user concurrent-job limits. Have Akki run and report:
  ```
  scontrol show partition normal
  sinfo -p normal -o "%P %l %c %m %D"
  sacctmgr show assoc user=$USER format=Account,Partition,MaxJobs,MaxSubmit,GrpTRES
  myusage
  quota -s        # or: lfs quota -h -u $USER /lustre/fs1
  module avail anaconda
  ```
- **Newton is the fallback only.** The scripts are cluster-agnostic: same partition name, same module pattern, cluster recorded in provenance. Switching to Newton requires Akki's approval because of its much smaller CPU allocation; never request GPUs.

You (the agent) probably have no cluster access. Write everything so Akki can run it with copy-paste commands, and make every script fail loudly with an actionable message.

---

## 2. Ground rules

1. **Architecture and learner stay as released.**
   - Keep the HetNet-Real A2C path: `main.py → A2CPolicy (per_class_critic=True) → UAVNetA2CEasy → MultiHeteroGATLayerReal`.
   - Keep class-wise attention, the optimizer that actually steps (`trainer.py` builds `optim.RMSprop` over `policy_net.parameters()`; verify), and README hyperparameters.
   - The paper supplement's recipe (3 layers + Adam 1e-3) is a *different* recipe. Do not run it now; record the discrepancy in the ledger.
2. **Every deviation from upstream is a separate, labeled commit** and a row in `DEVIATIONS.md`. Allowed deviations:
   - (a) Python 3.12 / gym 0.26 / CPU compatibility patch.
   - (b) Seeding moved before model/environment construction.
   - (c) Metrics logging and checkpoint cadence.
   - (d) Sensor-failure environment hook, off by default.
   - (e) Initial-condition bank loading.
   - (f) Message-removal switch in graph construction, off by default.

   Anything else needs Akki's approval.
3. **Nothing trained is ever resized or refit for evaluation.** Parameter names and shapes must be identical at every composition. Record a signature: name, shape, SHA-256 of values.
4. **No privileged information enters the actor.**
   - The actor sees exactly what upstream gives it.
   - The state node feeds only the critic; the implementation report found no state→agent edge on this path. Verify and preserve that.
5. **Never overwrite results.** Unique output directory per (run, composition, seed). Failed runs are kept and reported.
6. **Smoke tests are engineering evidence, never results.**
7. **Excluded:** any composition with `nfriendly_P == 1`. Assert this in the config loader.

---

## 3. Repository and environment

1. **Use Akki's existing fork. Do not create a new repository.**
   - Remote: `https://github.com/akki-g/HetNet`. Akki already has a local clone; work there.
   - Verified 26 Sep 2026 with `git ls-remote`: `main` == upstream `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`, unmodified.
   - Add the original as a remote: `git remote add upstream https://github.com/CORE-Robotics-Lab/HetNet`.
   - Tag the baseline: `git tag upstream-bff9f7f bff9f7f`.
   - Do all work on a branch (suggested: `frozen-eval`). Keep `main` equal to upstream so `git diff upstream-bff9f7f` always shows exactly our changes.
   - Keep the MIT `LICENSE` and credit the original authors in the README.
   - Put new code in a new package directory (e.g. `hetnet_ext/`), not scattered through upstream files.
2. **Apply the compatibility patch** `hetnet_py312_compat.patch` (Akki has it; it applies cleanly to `bff9f7f`). If absent, recreate these three changes:
   - `from inspect import getargspec` → `from inspect import getfullargspec as getargspec` in `env_wrappers.py`, `trainer.py`, `eval_trainer.py`.
   - Every `gym.make(...)` in `data.py` and `utils.py` gets `disable_env_checker=True`; gym 0.26's passive checker rejects these environments.
   - Guard `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated(...)` (trainer.py, main.py) with `torch.cuda.is_available()`.
3. **Pinned stack.** Verified on Linux, Python 3.12.3, CPU, running PCP A2C training:

   ```
   torch==2.2.1   dgl==2.1.0   torchdata==0.7.1   numpy<2   gym==0.26.2   visdom   bitstring
   pip install -e envs      # ic3net_envs
   ```

   - Torch **must** be 2.2.1. The PyPI DGL 2.1.0 Linux wheel only ships its graphbolt library for torch 2.0.0–2.2.1, and `import dgl` fails otherwise.
   - DGL 2.1 imports `torchdata.datapipes`, hence `torchdata==0.7.1`.
   - macOS arm64 cp312 wheels exist for torch 2.2.1 and DGL 2.1.0, so local tests on Akki's Mac use the same pins.
   - Put the pins in `pyproject.toml` with a committed `uv.lock`, mirroring marl-comm (`uv run --locked`).
   - Optional: use the PyTorch CPU wheel index to avoid ~2.5 GB of CUDA wheels, but only after verifying DGL still loads with a `+cpu` version string.
4. **Seeding fix (separate commit).** Upstream `main.py` calls `torch.manual_seed` / `np.random.seed` *after* building the policy and trainer, so model initialization is unseeded.
   - Seed Python, NumPy and torch at the top of `main.py`, before anything else is built.
   - Workers seed themselves as `seed + id + 1` (`multi_processing.py`). Keep that and record it.
5. **Metrics logging (separate commit).**
   - Append one JSON line per epoch to `metrics.jsonl`: epoch, wall time, steps, episodes, success rate, steps-taken, per-agent reward vector, policy loss, value loss.
   - Keep upstream stdout.
   - Checkpoints every 50 epochs plus the final one: `--save_every 50`, with an explicit per-run `--save_dir` / `--experiment_name`.

---

## 4. Phase A: Run 1 (training on Stokes)

### 4.1 Grid

All runs share `--dim 5`, `--max_steps 80` and the README PCP settings. Only `nfriendly_P`, `nfriendly_A` and `seed` vary. Keep vision at the README default and record it.

The observation width is (2v+1)²(D²+4) (implementation report Eq. I1). A checkpoint only transfers to runs with the same map size D and vision v, hence the fixed 5×5 grid.

| Priority | Composition | Role | Seeds |
|---|---|---|---|
| P0 | 2P+1A | Source policy for all frozen tests; also the paper's head-to-head setting | 0–4 |
| P1 | 3P+3A | Native reference (paper-linked composition, Fig. 5c) | 0–4 |
| P1 | 4P+6A | Native reference (paper-linked composition, Fig. 5c) | 0–4 |
| P2 | 3P+1A | Native reference: more perceivers, same capturers | 0–2 |
| P2 | 2P+2A | Native reference: more capturers | 0–2 |

That is 21 training tasks. The paper used seeds 0, 1, 2; ours extend to 0–4.

Reference command. It is README-exact except seed, composition and output paths; verify against the README:

```
python main.py --env_name predator_capture --nfriendly_P 2 --nfriendly_A 1 --nprocesses 4 \
  --num_epochs 2000 --hid_size 128 --detach_gap 5 --lrate 0.0001 --dim 5 \
  --batch_size 500 --max_steps 80 --hetgat --hetgat_a2c --seed <s> \
  --save_every 50 --save_dir <out>/checkpoints --experiment_name <comp>_s<s>
```

- **Log the fully resolved `args`** that `main.py` prints, including `comm_range_*`, `A_vision`, `use_binary`, and the per-class critic.
- **`--nprocesses 4` means 4 processes total:** the main process plus 3 workers (`multi_processing.py`). Each process collects a full `batch_size` of steps, and gradients are summed and divided by total steps. Data per update, and core-hours, scale with `nprocesses`.
- **Parameter sharing across processes.** Upstream calls `share_memory_()` on `policy_net` (= `policy.model`) and uses the `spawn` start method. Gate A must verify that workers actually see the main process's updated weights after an optimizer step. A silent stale-weights bug here would invalidate every run.
- **Completion semantics.** PCP success requires *every* agent to reach the prey *and every* A agent to capture it (`predator_capture_env._get_reward`, ~lines 482–540).
  - Larger teams therefore have strictly more obligations and higher density on 5×5; 4P+6A occupies 11 of 25 cells.
  - So every frozen result is read against a native reference at the same composition. State this in the notes.

### 4.2 Local test gate A (must pass on Akki's Mac or a Linux container before any cluster submission)

1. **Environment.** A clean `uv sync --locked` works, and `python -c "import torch, dgl, gym"` prints the pinned versions.
2. **Diff baseline.** `git diff upstream-bff9f7f -- <upstream files>` shows only the labeled deviation commits.
3. **Smoke training at every grid composition.**
   - 3 epochs, `--batch_size 40 --max_steps 20`, with `--nprocesses 1` and with `--nprocesses 4`.
   - Finite losses, and `metrics.jsonl` plus a checkpoint written.
   - The multiprocess path caches lists of non-null gradients (implementation report §2 risk). Confirm it works at 2P+1A and 4P+6A.
4. **Shared weights.** With `--nprocesses 4`, after one optimizer step, a worker's parameter hash equals the main process's.
5. **Determinism.**
   - Same seed, `--nprocesses 1`, twice: identical initial and first-epoch parameter hashes and identical `metrics.jsonl`.
   - Different seeds differ.
6. **Cross-composition load.**
   - Load a 2P+1A smoke checkpoint with `strict=True` into policies built for every grid composition, and write parameter-signature files.
   - Sandbox already verified 3P+3A, 4P+6A, 3P+1A; add 2P+2A.
   - The config loader rejects `nfriendly_P == 1`.
7. **Scripts.**
   - `bash -n` on every sbatch script; `shellcheck` if available.
   - A `--dry-run` prints the resolved command for every array index.
   - The index→(composition, seed) mapping is a bijection.
8. **Resume**, only if implemented (see 4.4). Kill-and-resume restores epoch counter, optimizer state, LR-scheduler state and RNG state. Otherwise, jobs are sized to finish in one allocation.

### 4.3 Stokes calibration and budget (short job before the full array)

- Calibration array: 2P+1A and 4P+6A, 20 epochs each, README settings, `--nprocesses 4`, `--cpus-per-task=4`.
- Measure seconds per epoch, peak RSS (use `sacct` / `jobstats`) and checkpoint size.
- Sandbox reference: ≈22 s/epoch at 2P+1A (batch 500, horizon 80), so ≈12 h per 2,000 epochs. Small-batch smoke runs suggested per-step cost of about 1.1× (3P+1A), 1.2× (3P+3A) and 1.35× (4P+6A) relative to 2P+1A. Stokes Xeon cores may be faster or slower.
- Produce a budget table: composition × seeds × measured core-hours; total; percentage of the remaining monthly Stokes balance from `myusage`.
- **Stop and ask Akki if the projection exceeds 4,000 core-hours or 10% of the remaining monthly balance**, or if the measured per-task wall time exceeds the partition cap.
- Never reduce epochs or batch size without approval; that changes the recipe.

### 4.4 Run 1 sbatch design (`slurm/run1_train.sbatch`)

- **One Slurm array**, one task per (composition, seed). The index mapping comes from a committed `configs/run1_grid.json`.
- **Resources per task:**
  - `--nodes=1 --ntasks=1 --cpus-per-task=4 --partition=normal`;
  - memory from calibration + 50%;
  - `--time` = calibrated wall time × 1.3, within the cap Akki reports;
  - `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and torch threads = 1, to avoid oversubscribing 4 processes on 4 cores.
- **Environment block:** the marl-comm wrapper pattern (module purge/load, conda base, assert Python 3.12, pinned uv, `flock`-serialized `uv sync --locked` into `.venv-stokes`). Fail with a clear message at any step.
- **Outputs:** `runs/run1_train/<comp>/seed<k>/` containing `config.json`, `metrics.jsonl`, `checkpoints/`, the Slurm stdout/err path, and `provenance.json`.
  - Provenance records: git commit + dirty flag, `uv.lock` hash, cluster name (`$SLURM_CLUSTER_NAME`), hostname, CPU model (`lscpu`), job/array IDs, start/end timestamps, resolved args, and the final checkpoint's parameter signature.
  - Refuse to start if a final checkpoint already exists, unless `--resume` is given.
- **Checkpoints** every 50 epochs. Implement true resume (optimizer, scheduler, RNG, epoch counter) plus requeue **only if** calibrated wall time exceeds the partition cap.
- **Submission order:** calibration → Akki confirms budget → full array. Give exact commands, plus monitoring: `squeue -u "$USER"`, `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,TotalCPU`, `myusage`.
- **Early learning check.** After about 300 epochs (≈2–3 h in), Akki sends 2P+1A `metrics.jsonl` from 2–3 seeds. If success is flat relative to epoch 0 across seeds, stop and report before the budget burns.

---

## 5. Phase B: frozen runner and Run 2 (build and test while Run 1 trains)

### 5.1 Frozen-runner contract (implementation report §9; feasibility report Eq. 6)

`hetnet_ext/frozen_runner.py`:

- **Construction and loading**
  - Constructs **no optimizer, no trainer, no replay buffer**. Upstream `--eval` routes through `Trainer`; do not use it.
  - Builds the policy for the *target* composition and loads only `policy_net` weights with `strict=True`.
  - Records the parameter signature and asserts it equals the source checkpoint's.
- **Freezing**
  - Uses `model.eval()`, `torch.inference_mode()`, and `requires_grad_(False)`.
  - Computes SHA-256 over all parameters and non-memory buffers before and after the whole evaluation, and fails on any difference.
  - Only LSTM state changes. Reset it at each episode start, using the upstream reset semantics.
  - Clears or bypasses rollout lists the upstream policy appends to (`batch_*_log_probs`, `batch_rewards`, …).
- **Actions**
  - Sampled from the policy's categorical distribution with a dedicated action RNG, seeded per episode from a fixed evaluation seed.
  - Sampled actions give the primary endpoint; greedy is an optional secondary panel.
- **Environment**
  - Same environment code and settings as training, with a separate environment RNG.
  - Initial conditions come only from the evaluation bank (5.3).

### 5.2 Conditions

| ID | Policy | Evaluation composition | Event | Purpose |
|---|---|---|---|---|
| F-src | 2P+1A source, frozen | 2P+1A | none | In-distribution control for every contrast |
| F-comp | 2P+1A source, frozen | 3P+1A, 2P+2A, 3P+3A, 4P+6A | none | Frozen composition transfer |
| N-comp | native policy for that composition | same composition | none | Native reference; **transfer gap = N − F** per composition |
| F-fail | 2P+1A source, frozen | 2P+1A | P-sensor failure of agent *j* ∈ {0, 1} at t_fail ∈ {0, 10, 20} | Recovery penalty after losing a perceiver (Eq. 36 raw outcomes) |
| F-nomsg | 2P+1A source, frozen | 2P+1A (optionally all compositions) | all agent→agent relations empty; self/state edges unchanged | Learned reliance on messages |
| O-ref | scripted oracle | every composition, and 2P+1A with failure | same events | Physical feasibility / attainable success; not an upper bound for learned policies |
| U-floor | untrained policy (seeded random init, same architecture) | every composition | none | Floor: what an untrained HetNet achieves by chance |

**Sensor failure (environment hook, off by default).**
- From `t_fail`, agent *j*'s observation becomes exactly the "no vision" representation the environment already gives blind A agents (the `A_vision` blinding path in `_get_obs`). Record what that representation is.
- The victim still moves, communicates, and must reach the prey for success. So the fixed-class policy must guide a blinded "perceiver" in using teammates' messages. This is the information-role question.
- **Aliasing warning** (implementation report §7): observation construction slices array views before blinding. Copy arrays so blinding one agent cannot alter another's observation.
- If the victim reached the prey before `t_fail`, the event is moot. Keep those episodes in unconditional results and report their fraction.

**Message removal.**
- Build the heterograph with empty P→P, P→A, A→P and A→A relations.
- Under class-wise normalization (feasibility Eqs. 7–8; audit H1–H2), an empty relation contributes zero. Verify finite outputs; never add silent epsilons.

**Scripted oracle.**
- With privileged prey position, each agent moves greedily to the prey. A agents capture once on it; respect sink/stacking rules as implemented.
- Its purpose is feasibility only. Say so in outputs.

### 5.3 Evaluation bank and pairing

- **Bank.** A fixed bank of **500 initial conditions per composition** on 5×5: distinct cells for all agents and the prey, bank seed 0. Follow the pattern in `test_config/evaluation_conditions.py`.
  - Versioned file with SHA-256.
  - Used only for final evaluation, never for tuning or checkpoint selection.
- **Reset from bank.** Implement environment reset from a bank entry, with a test that the entry's positions appear in the first observation.
- **Pairing.** Identical bank entries, action-RNG seeds and event manifests for every policy in a composition, so F vs N, intact vs event, and live vs severed are paired at the episode level.
- **Checkpoint selection is fixed in advance:** the final (epoch 2,000) checkpoint.

### 5.4 Metrics and logs

Per episode, write JSONL:
- condition ID, policy ID, seed, composition, bank index, event spec;
- success, steps taken (80 if unsuccessful; **never average steps over successes only**), per-agent reached-prey step, per-A capture step, total and per-agent return;
- for failure conditions: whether the victim had reached the prey before `t_fail`.

Optional diagnostic, only if it needs no change to forward math:
- per-relation attention max-weight and entropy per receiver versus number of senders;
- this probes the dilution mechanism of feasibility Eq. 29 as teams grow.

### 5.5 Local test gate B

1. **Immutability.** Parameter/buffer hashes unchanged after 20 episodes; no optimizer object exists.
2. **Strict load** into every composition, with identical signatures.
3. **Determinism.** The same condition run twice gives identical episode JSONL.
4. **Pairing.** Bank hash and event manifest identical across policies; oracle, floor and learned policies see identical initial states.
5. **Sensor-failure hook.**
   - Before `t_fail`, all observations equal the no-event run.
   - After it, the victim's observation equals the blind representation, and every other agent's observation equals the no-event observation in the same state. This is the aliasing check.
6. **Message removal.** Finite forward pass; relation aggregates exactly zero.
7. **Permutation null test** (feasibility Prop. 1).
   - Swap the two P agents' initial positions in a bank entry and relabel. Action *probabilities* must permute correspondingly; compare probabilities, not samples.
   - First confirm the observation has no agent-index features. The environment docstring mentions "identity"; check what is actually encoded.
   - A failure is a finding to report, not something to patch.
8. **Oracle.** Oracle success per composition on intact banks is recorded. Below ~95% at 4P+6A → report why (crowding, sinks, horizon).
9. **Sanity.** At 2P+1A on a smoke checkpoint, runner success is in the same range as the training loop's reported success. It won't match exactly.

### 5.6 Run 2 sbatch (`slurm/run2_frozen_eval.sbatch`), chained to Run 1

- **Array.** One array task per trained checkpoint, plus small tasks for oracle and floor. Each evaluates all its conditions into `runs/run2_frozen/<policy_comp>/seed<k>/`.
  - `--cpus-per-task=1` or `2`, sized from a local timing of 100 episodes per condition.
- **Before submitting Run 2:** `ANALYSIS_PLAN.md` must be committed (see §0.1).
- **Chaining:**
  ```
  sbatch --dependency=afterok:<RUN1_ARRAY_JOBID> slurm/run2_frozen_eval.sbatch
  ```
  - One failed training task blocks Run 2. That is intended: report it, rerun that task, then submit Run 2 manually.
  - A Run 2 preflight verifies that every expected final checkpoint exists and matches its provenance signature, and exits non-zero listing anything missing.

---

## 6. Analysis (`hetnet_ext/analyze.py`, run locally on downloaded results)

- **Seed is the unit.** Average episodes within each seed first. Show every seed, plus mean and bootstrap 95% CI over seeds (`rliable` or a seed-level bootstrap). Many episodes do not substitute for seeds.
- **Reproduction.** Learning curves (success, steps, per-class reward) for all 21 training runs. Include a qualitative comparison of 2P+1A with the HetNet paper, with discrepancies noted.
- **Tables and figures:**
  1. Success and steps-taken for U-floor, F-src, F-comp, N-comp and O-ref per composition.
  2. **Transfer gap** (N − F) per composition, with CI. Seeds are unpaired; episodes and bank are paired.
  3. Failure panel: intact vs event success per (victim, t_fail); recovery penalty Y_intact − Y_event per seed; moot-event fraction; oracle alongside.
  4. Message removal: live − severed success (learned reliance).
- **One command regenerates every table and figure** from raw JSONL (`python -m hetnet_ext.analyze --runs runs/ --out results/`).
- **Interpretation guardrails** (copy into `RESULTS.md`):
  - Equivariance does not imply size transfer (Eq. 28).
  - Larger PCP teams carry more obligations, so only the gap versus native is interpretable.
  - A severed-message drop measures this policy's reliance, not task-level necessity.
  - Oracle success is feasibility, not an optimum.
  - "Promise for SoftRole" requires a clear frozen gap under sensor failure and/or composition change **relative to native, oracle and floor references**.
  - Report effect sizes; do not declare the gate passed. Akki decides.

---

## 7. Rough time estimate

These are estimates from sandbox timing; the Stokes calibration job replaces them with measured numbers.

| Stage | Wall-clock | Compute |
|---|---|---|
| Research notes, repo setup, patches, gate A tests, sbatch + dry-run | 1–2 days of agent work, plus Akki's review | Local only |
| Stokes preflight commands + calibration job | 1–2 h, plus queue | < 20 core-hours |
| **Run 1**: 21 tasks, 4 cores each | ≈ 12–21 h per task at sandbox speed (2P+1A shortest, 4P+6A longest). Plausible range **10–30 h** depending on Stokes per-core speed. If all tasks run concurrently, Run 1 finishes in **about 1 day**; longer if per-user limits throttle the array. | ≈ **1,000–2,100 core-hours** (≈ 1,400 at sandbox speed), about 1.5–2.5% of Stokes' 80k monthly allocation |
| Frozen runner + gate B + `ANALYSIS_PLAN.md` | Built **during** Run 1 | Local only |
| **Run 2**: frozen evaluation (≈ 12–16 conditions × 500 episodes per source checkpoint; forward only) | ≈ **3–6 h** if tasks run concurrently | ≈ 100–200 core-hours |
| Analysis, figures, `RESULTS.md` | 1–2 days | Local |

**End to end: about one week if nothing breaks. Plan for 10–14 days** to absorb queue waits, one failed task, and debugging. A first paper draft on top of that is roughly one more week.

---

## 8. Math anchors to cite in code comments and notes

| Where it matters | Anchor |
|---|---|
| Frozen θ; memory/state may change, weights may not | Feasibility §4, Eq. 6; implementation report §9 |
| Class-wise normalization must not be altered; empty relation → zero | Feasibility Eqs. 7–8; audit H1–H2 (`fastreal.py` edge_softmax per relation); 1+3=4 vs 2 counterexample |
| Receiver-first adjacency vs DGL src→dst | Feasibility Eqs. 4–5; implementation report §4 |
| Permutation null test and its assumptions | Feasibility Prop. 1, Eqs. 22–23 |
| Why size transfer needs a native reference | Feasibility Eq. 28; implementation report §7 (PCP obligations) |
| Attention-dilution diagnostic | Feasibility Eq. 29 |
| Smooth representations do not guarantee success | Feasibility Prop. 3, Eq. 31 and following text |
| Sensor failure must leave observable evidence | Feasibility Eq. 33 |
| Report raw intact/event outcomes, not only differences | Feasibility Eq. 36, §8.4 |
| Observation width fixes D and v | Implementation report Eq. I1 |

---

## 9. Stop and ask Akki when

- A gate test fails and the fix would change architecture, learner, rewards, observations or completion rules.
- The projected budget exceeds the §4.3 threshold, or moving to Newton / `preemptable` is being considered.
- Stokes compute nodes lack outbound network access, or the anaconda module does not provide Python 3.12.
- The paper's Fig. 5c settings conflict with the 5×5 grid in a way that changes what "native reference" means.
- The early learning check shows no learning at 2P+1A.
- The shared-weights check fails, the permutation null test fails, or the observation contains agent-index features.
- You need cluster facts you cannot verify (wall-time cap, account flag, quotas, concurrency limits).

---

## 10. Hand back

1. `RESEARCH_NOTES.md`, with citations and the decisions table.
2. `DEVIATIONS.md` and `REPRODUCTION_LEDGER.md`: which settings come from the README, the paper, code defaults, or our choices, and every paper/code discrepancy.
3. `ANALYSIS_PLAN.md`, committed before Run 2, with its hash recorded.
4. `slurm/run1_train.sbatch`, `slurm/run2_frozen_eval.sbatch`, `configs/run1_grid.json`, and the evaluation banks with hashes.
5. Tests for gates A and B, with a one-command runner and its passing output.
6. `STOKES_RUNBOOK.md`: copy-paste commands for preflight, calibration, budget check, Run 1 submission, early learning check, Run 2 chained submission, monitoring, and downloading results. Include a short "switch to Newton" appendix.
7. After Run 2:
   - `RESULTS.md`: all tables and figures from §6, per-seed numbers, failed/missing runs, interpretation guardrails, and a limitations section.
   - `PAPER_SKELETON.md`: section outline (question, related work, setup and reproduction, frozen transfer, sensor failure, message reliance, limitations). Each planned claim is mapped to the specific table or figure that supports it. No prose claims beyond the data.

---

### Appendix A: facts verified in a sandbox (Linux, Python 3.12.3, CPU)

- Upstream HetNet PCP A2C training runs on the pinned stack after the three compatibility fixes.
- A 2P+1A checkpoint loads with `strict=True` into 3P+3A, 4P+6A and 3P+1A policies and runs, so trained parameter shapes do not depend on team size. That test went through the training loop, so it was **not** a frozen evaluation.
- 1P+1A crashes with an LSTM hidden-size error (singleton-P class).
- ≈22 s/epoch at 2P+1A, one process, batch 500, horizon 80.
- `--nprocesses N` runs N processes total (main + N−1 workers). Workers are seeded `seed + id + 1`. Parameters are shared via `share_memory_()` with the `spawn` start method.
- The A2C loss receives the real `num_P`/`num_A` from `args` (`trainer.py` ~475–488). The hard-coded `torch.Tensor(batch_size, 3)` in `compute_grad` is on the non-HetGAT `hetcomm` path. Re-verify both.
- `github.com/akki-g/HetNet` `main` == `bff9f7f` (via `git ls-remote`, 26 Sep 2026).

### Appendix B: sources for cluster facts

- ARCC scheduler intro and allocations: https://arcc.ist.ucf.edu/docs/scheduler/
- Partitions, fairshare, preemptable: https://arcc.ist.ucf.edu/docs/scheduler/limitations/
- Submission-script and `srun` examples: https://arcc.ist.ucf.edu/docs/scheduler/scripts/
- Stokes file-system policy: https://arcc.ist.ucf.edu/docs/data/files/
- Module list: https://arcc.ist.ucf.edu/docs/software/availableModules/
- Akki's working ARCC pattern: `marl-comm/slurm/run_experiment.sbatch` and `marl-comm/README.md`