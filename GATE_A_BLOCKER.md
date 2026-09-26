# Gate A blocker: stale multiprocess gradients under Torch 2.2.1

Status: **verified original failure; narrow repair approved by the user and applied; full Gate A remains pending**.

The [recorded three-update probe](evidence/gate_a/gradient_contract_intact_20260926_01.json) ran four spawned processes using the unchanged upstream `MultiProcessTrainer` and a deterministic surrogate loss with the selected A2C path's outer RMSprop and inner Adam clearing behavior. This isolates the aggregator/storage contract; it is not a trained HetNet result.

| Update | Workers see updated main weights | Cached gradients point to current storage | Fresh aggregation correct | Max absolute gradient error |
|---|---|---|---|---:|
| 1 | Yes | Yes | Yes | 1.73e-18 |
| 2 | Yes | No | No | 0.579605245 |
| 3 | Yes | No | No | 0.566336786 |

Torch 2.2.1 clears gradients to `None` by default. The next backward creates new tensors, while `multi_processing.py` continues writing its once-cached gradient pointers. Thus after the first update the optimizer can consume the parent's current uncombined gradient while worker aggregation and normalization update stale storage. A finite loss and matching shared-weight hashes do not establish correct training.

The [reviewable patch](evidence/gate_a/gradient_clear_candidate.patch) changes exactly three calls to `zero_grad(set_to_none=False)`: parent and worker calls in `multi_processing.py`, and the selected `A2CPolicy.batch_finish_per_class` call in `hetgat/policy.py`. It preserves the loss, clipping, aggregation formula, RMSprop and architecture. The user approved this exact change on 26 September 2026. It is applied as deviation G; failed evidence is retained rather than overwritten.

The actual A2CPolicy/Trainer/PCP probe also failed on intact code after update 1: maximum gradient errors were 0.180351550 and 0.165943605 at updates 2 and 3, while shared weights still passed. The repaired [2P1A probe](evidence/gate_a/hetnet_gradient_repaired_2P1A_20260926_01.json) and [4P6A probe](evidence/gate_a/hetnet_gradient_repaired_4P6A_20260926_01.json) each pass all three updates with exactly zero aggregation error. Current gradient storage and shared post-step weights also pass throughout. The full smoke matrix remains a separate gate. These are engineering diagnostics, never training results.

The user's task §9 required the now-recorded approval. No cluster calibration or full training may proceed until all of Gate A passes.
