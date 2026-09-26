# HetNet training lineage, reproduction settings and sample-budget evidence

Date: 2026-09-26. Research gate only. This note was prepared by reading the complete supplied instruction file, the primary HetNet paper §§4–6 and its supplement, IC3Net §§3–4/Appendix §§6.1–6.3, and pinned source. No implementation, environment installation, training, or cluster operation was performed for this note. All new files are under `HetNet/research/`; `marl-comm` was read only.

HetNet source references below are pinned to `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`. IC3Net lineage references are pinned to `69b7e0ce51a79def593abfef1a976f43e5e13f75`. Source evidence copies and SHA-256 values are in [source_lineage/IC3NET_PROVENANCE.json](source_lineage/IC3NET_PROVENANCE.json). Primary PDFs, URLs, sizes and hashes are in [papers/TRAINING_PAPER_MANIFEST.json](papers/TRAINING_PAPER_MANIFEST.json).

## 1. Fig. 5c does not establish exact larger-team map/horizon settings

**No explicit conflict with the instructed 5×5 grid was found. However, the exact grid size and step horizon used for each larger-team Fig. 5c curve are not stated in the inspected paper/supplement.**

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| HetNet §6.3.4, proceedings pp. 1179–1180 (PDF pp. 7–8) | The scalability ablation uses HetNet-**Binary** on PCP with 2P/1C, 3P/3C, 4P/6C | A frozen single checkpoint; exact per-composition map/horizon |
| Fig. 5c and caption, p. 1180 | Curves show steps taken versus training epoch, up to 2,000; the legend names the three compositions | Separate map labels or an explicit horizon declaration |
| Supplement §1.2, p. 1 | The head-to-head PCP benchmark uses **2P/1C, 5×5, maximum 80 steps** | An explicit statement that all scalability curves use that map/horizon |
| HetNet README line 24 | Released PCP sample command uses 2P/1A, `--dim 5 --max_steps 80 --num_epochs 2000`, Real by omission of `--use_binary` | A dedicated Fig. 5c configuration for 3P/3A or 4P/6A |

The figure was also inspected visually; its initial step values approach 80, which is compatible with an 80-step cap but is not a definitive specification. The page rendering is [HetNet_Fig5_page.png](papers/HetNet_Fig5_page.png). Sources: [HetNet paper](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf), [supplement](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/AAMAS_22___HetNet_Supplementary.pdf), [README:22–31](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/README.md#L22-L31).

Accordingly, call the proposed larger 5×5/80 Real models **native references at paper-linked compositions under our declared fixed-map protocol**. Do not call them an exact Fig. 5c reproduction. The source 2P/1A experiment has explicit paper/supplement support for its map and horizon, but source-versus-paper architecture/optimizer differences below still preclude claiming the entire published recipe is identical.

IC3Net's own PP study varies map sizes. Its Table 1 (PDF p. 8), Fig. 2/§4.1.1 (p. 5) and Appendix §6.3 (pp. 12–13) discuss 5×5, 10×10 and 20×20; §6.3 explicitly gives horizons 20, 40 and 80. Those are **IC3Net PP** settings, not evidence for HetNet PCP Fig. 5c. The IC3Net appendix's sentence about predator counts conflicts with its tables/figure (the prose lists 5/10/20 whereas tables use 3/5/10); do not transfer that ambiguous count specification into this project. [IC3Net primary paper](https://arxiv.org/pdf/1812.09755), [local PDF](papers/Singh_2019_IC3Net.pdf).

## 2. Architectural and learning facts from the primary papers

HetNet §4.2 (p. 1176) builds class-specific encoders and LSTM processing, assigns directed heterogeneous communication edges, and introduces a state embedding node (SEN) for training. SEN receives agent messages but has no edge back to the agents; the paper explicitly permits its removal at execution. §4.3, Eqs. (2)–(4) (p. 1177), normalizes attention within edge types before adding relation contributions and the self transform. §4.4 says one layer is one communication round and the final output dimension is the class's action-space size followed by softmax. §5.1/§5.2 (pp. 1177–1178) describe on-policy class policies and centralized/per-class/per-agent critics; class-critic targets average returns within physical classes. Algorithm 1 is the paper's conceptual training procedure, not a precise CLI budget specification. [HetNet](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf).

§6.1–§6.2 (p. 1178) distinguish homogeneous PP, complementary PCP and FC, and say that Real removes the binary encoder/decoder and Gumbel channel while retaining class-specific edges. Fig. 5c is Binary; Fig. 6's critic ablation is Real (§6.3.5, p. 1180). Comparisons to these figures should name the variant correctly. Table 1 reports final policies over 50 evaluation trials; it is not evidence for the new planned 500-episode bank. [HetNet](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf).

The supplement §2.1 (p. 2) specifies **three** multi-head layers, four heads, two intermediate 16-feature/head concatenations, final head averaging, Adam at 10^-3 and seeds 0/1/2. The released active model instead has **two** layers at [`hetgat/uavnet.py:102–109`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/uavnet.py#L102-L109). The selected active trainer constructs RMSprop with `alpha=0.97`, `eps=1e-6`, learning rate from the CLI at [`trainer.py:55–57`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/trainer.py#L55-L57). The README PCP rate is 10^-4. Preserve this released recipe as instructed; the supplement recipe is a documented discrepancy, not an alternate run authorized by this task.

The per-class HetNet loss also has specific source semantics: zero-padding to `[episodes,agents,horizon]`; within-class advantage normalization over that padded tensor; per-episode policy and critic losses averaged across episodes; total loss `50*policy_loss + critic_loss`; backward and gradient clipping; policy-local optimizer step commented out. The Trainer/multiprocess optimizer subsequently steps. This is not interchangeable with a generic A2C package. [`policy.py:778–877`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/policy.py#L778-L877), [`trainer.py:475–487`](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/trainer.py#L475-L487).

## 3. IC3Net lineage and exact batch semantics

HetNet's README explicitly credits modified IC3Net files (line 20). IC3Net §3 (PDF pp. 3–4) uses shared recurrent policies and gated continuous communication; it trains action and gate policies with REINFORCE. Appendix §6.1 (p. 11) says training uses RMSprop, 16 cores, each collecting episodes until at least 500 steps, with 10 weight updates per epoch. It uses 1,000 epochs for PP/StarCraft and 2,000 for traffic junction. These are IC3Net settings, not automatic substitutes for HetNet's 4-process/2,000-epoch README recipe. [IC3Net paper](https://arxiv.org/pdf/1812.09755).

The source lineage is unusually direct:

| Meaning | Original IC3Net source | Released HetNet source |
|---|---|---|
| `epoch_size=10`, `batch_size=500`, `nprocesses=16` parser defaults | `main.py:23–32` | `main.py:35–50` |
| One epoch repeats `train_batch` `epoch_size` times | `main.py:207–214` | `main.py:387–404` |
| Each collector gathers complete episodes until `len(batch) >= batch_size` | `trainer.py:225–241` | `trainer.py:499–527` |
| `nprocesses` includes parent; workers = processes minus one | `multi_processing.py:41–51` | `multi_processing.py:47–58` |
| Workers and parent collect/backpropagate, then parent aggregates gradients and steps | `multi_processing.py:74–98` | `multi_processing.py:85–112` |

Pinned IC3Net permalinks: [main](https://github.com/IC3Net/IC3Net/blob/69b7e0ce51a79def593abfef1a976f43e5e13f75/main.py#L23-L32), [trainer](https://github.com/IC3Net/IC3Net/blob/69b7e0ce51a79def593abfef1a976f43e5e13f75/trainer.py#L225-L241), [multiprocessing](https://github.com/IC3Net/IC3Net/blob/69b7e0ce51a79def593abfef1a976f43e5e13f75/multi_processing.py#L41-L98). Corresponding HetNet sources: [main](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/main.py#L35-L50), [trainer](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/trainer.py#L499-L527), [multiprocessing](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/multi_processing.py#L47-L112).

`batch_size` means **joint environment transitions per collector per update**, not agent transitions, not episodes, and not a total divided among processes. Complete episodes overshoot the threshold. Let B be the threshold, H the maximum episode length, P the number of collectors, E the updates per epoch, K the epochs, and b[u,p] the actual transition count. Provided each episode has 1..H steps,

\[
B \le b[u,p] \le B+H-1,\qquad
S_{run}=\sum_{u=1}^{KE}\sum_{p=1}^{P}b[u,p].
\]

The bounds follow directly from appending one final complete episode after a pre-append count below B. The code comment `epoch_size * batch_size * nprocesses` is a **nominal lower bound**, not exact realized samples. Every episode resets recurrent state and can finish before H; reducing H or B changes more than runtime. This derivation is source arithmetic, not a measured training result.

For the requested B=500, H=80, P=4, E=10, K=2,000:

| Unit | Minimum transitions | Maximum transitions under bound |
|---|---:|---:|
| One collector/update | 500 | 579 |
| One joint update | 2,000 | 2,316 |
| One epoch | 20,000 | 23,160 |
| One training run (20,000 optimizer-step calls) | 40,000,000 | 46,320,000 |
| All 21 planned runs | 840,000,000 | 972,720,000 |

At fixed team size N, agent action decisions are N times joint transitions: the nominal per-run lower bounds are 120 million (2P1A), 160 million (3P1A/2P2A), 240 million (3P3A), and 400 million (4P6A). This explains why equal environment steps are not equal agent decisions or compute. Core-hours still require Stokes calibration; none of these counts estimates seconds per epoch.

The gradient aggregation *intends* to sum collector gradients and divide by total joint steps at `multi_processing.py:105–112`. However, each HetNet collector has already episode-averaged and clipped its gradients (`policy.py:848–865`). Do not describe the resulting loss as a mathematically identical global minibatch mean without those qualifications.

## 4. Scientific logging and compatibility risks requiring Gate A evidence

**Cumulative step/episode counters.** In released `main.py:401–404`, the code first merges one update's `s` into the epoch-cumulative `stat`, then adds the already cumulative `stat['num_steps']`/`stat['num_episodes']` into running totals. This double-counts earlier updates within an epoch. With E equal-sized update batches, the printed running total accumulates E(E+1)/2 batches instead of E: E=10 gives a factor 5.5. When batch lengths vary, the overcount is a weighted sum, not a fixed factor. Epoch means later derived from `stat` are a different path and should not all be called wrong. Record raw update counts or sum the fresh `s` values for research metrics; keep original stdout. [main.py:387–419](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/main.py#L387-L419).

**Cached gradient storage under Torch 2.2.1.** HetNet caches `p._grad.data` pointers once at `multi_processing.py:69–83`. Both the outer trainer at `multi_processing.py:29,96` and policy-local code at `policy.py:846` call `zero_grad()` without arguments. Official PyTorch **v2.2.1** defines `zero_grad(set_to_none=True)` at `torch/optim/optimizer.py:789`, and sets existing gradients to None at 818–820. A subsequent backward can allocate new gradient tensors while the cached pointers still refer to old storage. This can undermine worker-gradient aggregation/normalization after the first update even while the workers see shared parameter updates. This is a static compatibility risk, not an executed HetNet failure. Gate A should compare current gradient data pointers with cached pointers and verify worker gradient contribution for at least two successive updates, in addition to checking shared weight hashes. No patch is proposed or applied at this research gate; a required behavioral change must follow the user's deviation/approval rules. Sources: [HetNet pointer cache](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/multi_processing.py#L69-L112), [policy-local reset](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/hetgat/policy.py#L845-L865), [Torch v2.2.1 source](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/optim/optimizer.py#L789-L827), [local source provenance](source_lineage/pytorch_2_2_1_optimizer.provenance.json).

**Seeds and sharing.** Upstream sets spawn at `main.py:26–27`, shares parameter storage at 297–299, constructs trainers before setting main Torch/NumPy seeds at 333–334, and seeds workers with `seed+id+1` at `multi_processing.py:16–18`. These prove the intended mechanics and misplaced initialization seeding, not successful shared-weight or reproducibility tests on the pinned stack. Seed Python as well in the authorized seeding deviation, preserve worker offsets, and keep scientific trajectory hashes separate from nondeterministic wall-time fields. [main.py](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/main.py#L297-L334), [worker seed](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/multi_processing.py#L16-L18).

## 5. Decisions table for the parent research notes

| Setting | Selected value/status | Authority |
|---|---|---|
| Architecture variant | HetNet-Real, released two-layer path | User §2.1; README/code; differs from supplement |
| Learner/optimizer | Released per-class A2C/MAHAC loss; stepping RMSprop | User §2.1; trainer:55–57; policy:778–877 |
| Learning rate | 10^-4 | README PCP line 24 |
| Map/horizon | D=5, H=80 for all requested runs | Explicit user §4.1; supplement supports source head-to-head only |
| Fig. 5c larger-team map/horizon | Not explicitly identified | Paper §6.3.4/Fig. 5c + supplementary scope |
| Composition attribution | 3P3A/4P6A are paper-linked; exact fixed-map Real protocol is declared here | User §4.1 and Fig. 5c Binary distinction |
| Training epochs | 2,000 | README PCP; user §4.1 |
| Epoch size | 10 updates | main.py:45–46 default, inherited IC3Net |
| Batch threshold | 500 environment steps per process/update, whole-episode overshoot | main.py:47–48; trainer:512–526 |
| Process count | 4 total, parent plus 3 workers | README PCP; multi_processing.py:50–58 |
| Main seeds | 0–4 for P0/P1; 0–2 for P2 | User choice; supplement used 0/1/2 |
| Worker seeds | Main seed + worker index + 1 | multi_processing.py:16–18; user preserves |
| Vision | 2 (README omitted it; parser default) | predator_capture_env.py:68–69 |
| Other relevant defaults | gamma=1; use_binary=False; A_vision=-1; full communication ranges -1 | main.py:57–64,86–89; predator_capture_env.py:86–87 |
| Nominal training transitions/run | At least 40 million; log actual overshoot | Source-derived budget above |
| Frozen evaluation bank | 500 states/composition; final epoch-2000 checkpoint | User protocol, not a paper setting |
| Outstanding blocker status | No proven Fig. 5c grid conflict; missing precise figure settings must remain disclosed | Evidence absence, not permission to assert inferred settings |
| Gate A addition | Validate gradient-storage freshness and worker contributions after ≥2 updates | Static Torch-2.2.1 compatibility finding |

