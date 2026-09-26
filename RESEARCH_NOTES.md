# Research notes: original HetNet training and frozen evaluation

Research gate prepared 26 September 2026. Target: Akki's existing `akki-g/HetNet` fork, branch `frozen-eval`, baseline `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec` tagged `upstream-bff9f7f`. The initial research gate was committed as `be161ab` before implementation. Later additions explicitly identify their engineering evidence; source deductions and smoke checks are not final experimental results. Cluster facts requiring live verification remain pending.

## 1. Question, scope and primary evidence

The experiment asks whether original HetNet-Real, trained on stationary PCP at 2P+1A, transfers with frozen parameters to different fixed compositions, a perception-sensor failure, and removal of learned messages. Native policies, an untrained architecture-matched floor and a privileged scripted feasibility reference make the frozen results interpretable. No SoftRole, new critic, changed learning objective, arrivals/departures or additional training recipe is authorized. These are the user-specified T0 experiment and controls, not conclusions from the literature.

Primary reading used for these choices:

- Seraj et al., [HetNet, AAMAS 2022](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf), pp.1175–1180, §§4.2–4.4, 5, 6.1–6.3.5, Eqs.2–6, Figs.3–6; [author supplement](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/AAMAS_22___HetNet_Supplementary.pdf), §§1.2 and 2.1.
- Singh et al., [IC3Net, ICLR 2019](https://arxiv.org/abs/1812.09755), architecture/training and predator–prey experiments, and the author code lineage. The local `README.md:20` acknowledges IC3Net; current batch/process semantics must be established from this fork rather than imported from IC3Net results.
- Howell et al., [CoRL 2023](https://proceedings.mlr.press/v229/howell23a.html), §§3.2–4, pp.4–5 of the PDF: capabilities, shared graph policies, and unseen team composition/size evaluation. Its training uses multiple teams and PPO, so it supplies protocol context rather than a reproduction baseline for this single-composition A2C study.
- Fu et al., [CASH, CoRL 2025](https://proceedings.mlr.press/v305/fu25a.html), §5.2, PDF pp.7–8: pretrained policies are deployed in failure and battery-drain scenarios. Failure reduces a randomly selected robot's capability by 75% halfway through an episode. This is a direct precedent for post-training capability-change evaluation, but its Robotarium tasks and explicit capability inputs differ from this grid PCP sensor intervention.
- Agarwal et al., [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html), uncertainty and aggregate-performance methodology; Lowe et al., [AAMAS 2019](https://arxiv.org/abs/1903.05168), communication measurement and intervention pitfalls. These motivate seed-level inference and the distinction between a learned policy's reliance and task necessity. The exact preregistered estimator is an experimental choice.

The prior [feasibility report](research/project_references/SOFTROLE_FEASIBILITY_REPORT.md), [implementation report](research/project_references/SOFTROLE_IMPLEMENTATION_REPORT.md) and [mathematical audit](research/project_references/MATHEMATICAL_AUDIT.md) are read-only research inputs. They are not an implementation to port. The older repository audit concerns `commstudy`, which is outside this task.

## 2. What is being reproduced

HetNet §4.2, p.1176, assigns physical classes to node types and directed class pairs to edge types. The state embedding node receives agent messages, but does not send to agents. Section 4.4, p.1177, uses each graph layer as a communication round and produces class-specific categorical actions. Section 6.2, p.1178, defines HetNet-Real by removing the binarizing codec and communicating continuous embeddings. Section 5 distinguishes centralized, per-class and per-agent critics; the requested branch uses the per-class critic. These are architecture statements, not guarantees about learned transfer.

Verified released execution path:

- `main.py:230–275` constructs `A2CPolicy` with real messages (`use_binary=False`), `per_class_critic=True`, `per_agent_critic=False`, four heads, and two state nodes. It sets intermediate P/A/state widths to 16 and output widths to 5/6/8. `--hid_size 128` is retained because it is in the README command, but is not the hard-coded graph width in this branch.
- `hetgat/uavnet.py:83–115` contains class-specific preprocessing/LSTM cells and two graph layers. `hetgat/graph/fastreal.py:147–353` is the real layer; its class-wise denominators and head merging must remain unchanged.
- `hetgat/utils.py:133–152` constructs P→P, P→A, A→P, A→A and agent→state/state→state edges. There is no state→agent relation in this path. `uavnet.py:359–381` constructs the small state-node feature and class critic outputs.
- `trainer.py:55` creates RMSprop with the configured learning rate, alpha 0.97 and epsilon 1e-6. `trainer.py:475–488` passes actual P/A counts into the A2C loss; the hard-coded three-column tensor on another branch is not this loss. `policy.py:778–870` calculates the class-specific loss and backward pass; its policy-local optimizer step is commented out. The outer trainer/multiprocess trainer performs the step.

The supplement §2.1 describes **three** layers and **Adam at 1e-3**. The selected release constructs **two** layers and actually steps **RMSprop**, while README PCP specifies **1e-4**. We will reproduce the released README path. The supplement recipe is recorded, not run. Comparing results to the paper must state this mismatch.

### Fig.5c: what is and is not specified

HetNet §6.3.4 (pp.1179–1180) and Fig.5c explicitly study **HetNet-Binary**, with `(2P,1C)`, `(3P,3C)` and `(4P,6C)`. The plot is learning over training epochs, not a frozen 2P1C-checkpoint transfer experiment. The supplement §1.2 gives a **5×5 map and 80-step cap for the 2P1C head-to-head task**. Neither the Fig.5c discussion/caption nor a per-curve setting table explicitly establishes a separate map size and horizon for every larger-team curve. No larger-map conflict has been found; an axis near 80 is not proof of a horizon.

Consequently our 3P3A/4P6A **5×5, horizon-80 Real-policy natives are declared configurations with paper-linked compositions**, not exact Fig.5c reproductions. The 2P1A learning curves can be compared qualitatively to the original head-to-head setting, with optimizer/layer differences visible. Do not silently substitute map settings from IC3Net.

### License provenance

The upstream README says MIT, but the actual tracked `LICENSE:1–2` is GNU GPL version 3. Preserve the original license file and credit; do not replace or relabel it based on the README or task text. This is a provenance discrepancy, not an architecture change.

## 3. PCP interface, reward and obligations

The actual PCP definitions are in `envs/ic3net_envs/predator_capture_env.py`. Its default observation vocabulary (`:130–156`) gives each local cell `D²+4` channels: absolute-position channels and semantic entity-class channels. The default flattened width is

\[
d_o=(2v+1)^2(D^2+4).
\]

There are no per-agent identity channels in that path, despite the stale docstring. With `D=5` and default `v=2`, width is 725; a runtime assertion will verify the resolved default and shape. Counts affect the row dimension and recurrent state allocation, not trained weight dimensions. Keep D and v fixed at deployment; strict-load signatures will test all five permitted compositions rather than assume transfer from this formula alone. Source: environment `:65–77,130–156`, wrapper `:97–109`, model construction `main.py:230–275`, implementation report Eq.I1.

P has four moves plus stay; A adds capture. Blind A retains position channels and has semantic channels overwritten with −1 (`predator_capture_env.py:282–298`). Reaching prey makes an agent a sink; an A on prey can capture and remains absorbed (`:392–479`). In mixed mode the per-agent step penalty is −0.05; P stops paying when on prey, and A when captured. Success requires **all agents to reach prey and all A agents to capture** (`:482–543`). Thus larger teams have more obligations and greater density, and removing communication changes a policy, not these physical rules. All initial positions plus prey must fit in 25 distinct cells (`:241–242`); 4P6A initially uses 11 cells.

The failure intervention is on observation `o_t`, with zero-based `t_fail∈{0,10,20}`, including reset observation when `t_fail=0`. After obtaining upstream's stacked observation array, copy it and replace only the selected P row's semantic channels by −1. Retain that P agent's class, location channels, motion, communication and completion obligation. This uses the existing blind representation without rebuilding a P as an A body.

**Preserving the intact release:** upstream observation construction has an existing slice-view aliasing issue (`:280–291`). Do not silently fix the entire intact environment. A post-stack copy for the new victim mask preserves the original output when the hook is off and prevents the new intervention from modifying other rows. Gate B compares intact/event observations at the **same physical state** before and after the event. Naturally diverging trajectories are not a valid isolation fixture. A broader observation repair would require approval and a new deviation.

The privileged greedy reference reads prey coordinates and moves each body toward prey, using capture only for A at prey. It respects the release's sink/stacking rules. Its success is physical feasibility, not an optimality certificate. An event after the victim has reached prey is moot; completed-before-event and moot-victim episodes stay in unconditional endpoints and are separately counted. Source: PCP transition/reward lines above; feasibility report §8.4.

## 4. Mathematics translated into the experiment

The feasibility report's Eqs.7–8 and audit H1–H3 reconstruct class-wise attention: normalize each relation's incoming neighbors separately, add the aggregates and self transform, then apply the activation. With two singleton classes emitting scalar values 1 and 3 at equal scores, this gives 4; global softmax gives 2. Do not substitute global normalization. In DGL terminology edges are source→destination; the feasibility notation is receiver-first. The pinned DGL 2.1 implementation uses destination normalization by default; each `edge_softmax(g[relation], ...)` call restricts its domain to that relation. Message removal empties only the four agent-to-agent relations, leaving node self transforms and state edges untouched. Empty-relation aggregates must be zero without silent denominator epsilons. [DGL API](https://www.dgl.ai/dgl_docs/generated/dgl.nn.functional.edge_softmax.html), [v2.1.0 implementation](https://github.com/dmlc/dgl/blob/v2.1.0/python/dgl/ops/edge_softmax.py).

**Prospective transfer interpretation:** the source graph has no A→A edges, and its P→P, A→P and A→state neighborhoods each contain one sender. For a singleton softmax, the coefficient is 1 and its score derivative is 0. Thus the source leaves eight A→A tensors unused and twelve attention tensors without learning gradients. Increasing P or A count activates different previously unsupported computations, in addition to increasing completion obligations and changing attention normalization. The [relation-support audit](research/RELATION_SUPPORT_AUDIT.md) derives this precisely and verifies it in preserved structural gradients and the first source smoke. This is not a performance result, nor a causal explanation established by the future transfer gap; the requested recipe and conditions remain fixed.

Relevant limits, with exact anchors:

- Feasibility Eq.6: deployment parameters are frozen while recurrent memory may update. A no-optimizer runner, strict load and full parameter/buffer signatures operationalize this claim.
- Proposition 1, Eqs.22–23: agent relabeling must carry observations, memories, type metadata, graph and action semantics together. Test probability vectors when swapping the two P agents. Do not compare independent sampled actions or patch a failing null test without review.
- Eq.28: equivariance does not establish success at an unseen population. Native references at the same composition are essential controls.
- Eq.29: fixed useful/distractor score separation can dilute useful attention as neighborhoods grow. Optional passive attention diagnostics can probe this mechanism without changing forward math.
- Proposition 3, Eq.31: a conditional bounded/Lipschitz attention representation result is not a task-return guarantee for arbitrary changes.
- Eq.33: diagnosis requires statistically distinguishable execution-visible evidence. Here a P sees a structured blind observation, so the intervention is not a general claim about diagnosing invisible hardware failure.
- Eq.36 and §8.4: report raw intact/event outcomes alongside recovery penalties; failure-aligned selected subsets are secondary, with censoring and exposure counts disclosed.

## 5. Training amount, reproducibility and compatibility risks

`main.py:45–48` sets `epoch_size=10` updates per epoch and `batch_size=500` minimum collected environment steps per process/update. `multi_processing.py:50–58` runs the main collector plus `nprocesses−1` workers. Thus 2,000 epochs ×10 updates ×500 steps ×4 processes is **at least 40 million environment steps per task**, with completed-episode overshoot. Twenty-one tasks total at least 840 million environment steps. These counts are design arithmetic, not measured timings. Episode count is not fixed by `batch_size`.

RNG seeding currently occurs after environment/model/trainer construction (`main.py:195,230–334`). Move Python, NumPy and Torch seeding before construction, retaining worker offsets `seed+id+1` (`multi_processing.py:17–18`) and adding Python seeding for consistency. Record seeds/streams. The single-process reproducibility gate checks initial/first-epoch tensor hashes and every non-timing metric. Measured wall time cannot be byte-identical across reruns; retain honest time fields and compare a documented deterministic projection of JSONL, rather than fabricate timestamps.

The main loop accumulates `stat` within an epoch and then adds the cumulative `stat` into total steps/episodes (`main.py:401–404`), overcounting those totals. New metrics/provenance must accumulate each raw batch `s` exactly once. Do not change the collector, loss or published printed output merely to make accounting cleaner. Checkpoint/epoch labels must declare whether they are zero-based counters or completed-epoch numbers.

**High-priority source risk:** `multi_processing.py:69–83` caches non-null gradient storage once, while `:29,96` and `policy.py:846` call `zero_grad()` without an argument. [PyTorch v2.2.1 optimizer source, lines 789–827](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/optim/optimizer.py#L789) defaults `set_to_none=True`, which can replace `.grad` storage on the next backward. Shared parameter hashes can still match while aggregation writes stale gradients. Gate A must inspect at least two updates, compare cached/current storage and fresh worker contributions, and report a failure before any full run. An empirical failure requiring changed clearing/aggregation behavior must be presented to Akki before a fix; no such fix is implied by these notes.

The requested compatibility changes are `getfullargspec` aliases, Gym checker disabling and CUDA-memory-call guards. All are separately traceable in deviation category A. The pinned stack is Python 3.12, Torch 2.2.1, DGL 2.1.0, torchdata 0.7.1, NumPy below 2, Gym 0.26.2, Visdom and bitstring. Resolve exact ancillary versions into `uv.lock`; prove clean sync/import on this Mac or Linux before submission. Singleton-P configurations are explicitly rejected, not repaired. Linux DGL GraphBolt wheel compatibility and CPU suffix loading need actual import evidence if a CPU-index variant is selected.

## 6. Frozen runner and analysis contracts

Construct `UAVNetA2CEasy` directly, reproducing the released model arguments and **float64 default** (`main.py:32`); do not instantiate `A2CPolicy`, which constructs an optimizer and rollout buffers. **Precision clarification from the Gate A audit:** `fastreal.py:81–96` explicitly constructs attention parameters with `torch.FloatTensor`, retaining float32 despite the default. Preserve these mixed dtypes; blanket `model.double()` would silently convert them, and `load_state_dict(strict=True)` alone does not reject dtype conversions. Verify dtype/value signatures after loading. Use original graph construction and forwards, strict `policy_net` state loading, `eval()`, `requires_grad_(False)` and inference mode. Store all names, shapes and SHA256 values, including buffers. LSTM state resets at episode starts and can change within an episode; learned state cannot. Sample categorical actions with a dedicated per-episode generator, separate from environment/bank randomness. Upstream `--eval` uses `Trainer` and exits before optimization (`main.py:311–315`, `trainer.py:538–614`); bypass it for purity, not because it has been shown to update evaluation weights.

Every composition has a versioned, bank-seed-0 set of 500 distinct-cell initial states. Use paired bank entries, action seeds and event manifests across frozen/native/oracle/floor policies. Banks are final-test inputs, never development selection data. Choose final epoch 2,000 checkpoints in advance. Local gates use separately labeled smoke banks and checkpoints, and their outcomes are engineering evidence only.

Preregister the primary endpoint (sampled-action episode success), 80-capped completion steps for failures, native-minus-frozen transfer gaps, within-source-seed intact-minus-failure and live-minus-severed contrasts, handling of missing/crashed runs, and all exploratory diagnostics. Average episodes within training seed first. Native/source training seeds are independent even when labels coincide, so use an unpaired seed bootstrap for transfer gaps; paired within-seed contrasts use paired episode inputs. Show all seeds and native/frozen raw cells. A five-seed interval is still uncertain; 500 episodes do not create 500 independently trained policies. Source: Agarwal primary paper; feasibility §8.4; user §6.

## 7. Stokes: documented facts and live gates

Official [ARCC scheduler](https://arcc.ist.ucf.edu/docs/scheduler/) and [limitations](https://arcc.ist.ucf.edu/docs/scheduler/limitations/) describe Stokes' faculty-account monthly 80,000 CPU core-hour allocation, no rollover, normal/preemptable partitions, and fairshare. The account is shared with other PI users; nominal allocation is not the available balance. [File policy](https://arcc.ist.ucf.edu/docs/data/files/) describes `/lustre/fs1` home, 1 TB/one million files and no backup. [Module list](https://arcc.ist.ucf.edu/docs/software/availableModules/) is not a guarantee that a module exists on the target compute node. Public hardware counts differ; they are irrelevant to a job's permitted allocation and must not substitute for live scheduler facts.

The read-only `marl-comm/slurm/run_experiment.sbatch` provides the bootstrap pattern: module purge/load, conda base, Python 3.12 assertion, pinned uv 0.12.5 and flock-serialized locked sync. Use `.venv-stokes`; keep dependency installs on compute allocations in accordance with ARCC's Anaconda guidance. Verify compute outbound access before selecting a bootstrap route. Any login-node install or offline/container fallback needs the requested user review.

Pending user evidence: normal partition time cap, account requirement, job/concurrency limits, current `myusage`, quota, actual modules, shared-home status and access route. Do not guess these values. After Gate A, calibration runs only 2P1A and 4P6A for 20 epochs at full README batch/process settings. Compute each projected task wall time from measured per-epoch cost; request 1.3× wall time within the cap, and memory at least measured peak plus 50%. Report total reserved core-hours and usage as a fraction of the **remaining** balance. Full-array submission always requires Akki's budget confirmation, and exceeding 4,000 core-hours, 10% of remaining balance or the time cap requires explicit review. No shortened epochs/batches, Newton or preemptable fallback is silently permitted.

Additional primary-source audits: [training lineage and figure settings](research/TRAINING_RESEARCH.md), [Stokes documentation and pinned package evidence](research/STOKES_SOURCE_AUDIT.md), and [scientific gate design](research/GATE_DESIGN_AUDIT.md). These include the required primary reading, exact code anchors and remaining uncertainties.

## Decisions table

| Setting | Value / decision | Provenance |
|---|---|---|
| Method | Released HetNet-Real A2C, per-class critic, unchanged forward/loss | User; `main.py:230–275`; HetNet §§4–5 |
| Baseline | `bff9f7f`; existing `akki-g/HetNet`; `frozen-eval` branch | User; current Git inspection |
| Graph layers / heads | Two layers; four heads; intermediate type widths 16 | Released `uavnet.py:106–109`, `main.py:239–250`; differs from supplement |
| Optimizer / LR | Actual RMSprop; 1e-4; preserve released alpha/epsilon | `trainer.py:55`; README PCP command |
| Map / horizon | 5×5 / 80 for every experiment | README/supplement 2P1A; our declared larger-native settings |
| Vision / width | Default v=2; expected width 725; assert at runtime | PCP `init_args`; Eq.I1; no new CLI override |
| Discount / update count | gamma=1; epoch_size=10; batch_size=500/process | Code defaults plus README |
| Processes | Four total; three workers and main; one Torch/BLAS thread each | Released multiprocessing; user resource constraint |
| Epochs / checkpoints | 2,000; every 50 and final completed epoch | README epochs; user cadence/selection |
| Source | 2P1A, seeds 0–4 | User design; extends original seeds 0–2 |
| Native references | 3P3A/4P6A seeds 0–4; 3P1A/2P2A seeds 0–2 | User design; 21 tasks total |
| Excluded roster | P=1; fail config validation | User sandbox evidence; no singleton repair |
| Communication | Real-valued, full-range default −1 for P/A, no lossy channel | Code defaults / README path |
| A sensing | Default blind, A_vision=−1 | PCP defaults |
| Training observations | Preserve upstream; sensor hook off | User original-method constraint |
| Failure | P victim 0 or 1; o_t becomes semantic-blind at t=0/10/20 | User; post-stack copy avoids new cross-row mutation |
| Message ablation | Four agent-to-agent relations empty; self/state unchanged | User; per-relation empty-aggregate contract |
| Frozen action mode | Categorical samples; dedicated episode action RNG | User; original stochastic actor |
| Banks | 500 distinct-cell states/composition; bank seed 0; hashes | User; final-evaluation-only |
| Primary endpoint | Episode success; also 80-capped steps and raw returns | User preregistration; avoids success-only censoring |
| Statistical unit | Training seed; unpaired N−F seed bootstrap; paired within-seed interventions | User; Agarwal; exact procedure in future analysis plan |
| Platform | Stokes CPU normal partition; live limits pending | User; ARCC sources |
| Budget gates | Calibration first; user confirmation; >4000 core-hours or >10% remaining escalated | User §4.3–4.4 |
| Resume | Not implemented unless calibration exceeds normal cap | User conditional requirement |
| License | Preserve actual GPLv3 file; record README/task MIT mismatch | Actual `LICENSE:1–2` |
