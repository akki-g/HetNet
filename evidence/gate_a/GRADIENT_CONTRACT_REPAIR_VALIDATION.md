# Gradient transport and storage validation

The user-approved three-site change to `zero_grad(set_to_none=False)` passes the
bounded scientific gradient contract on the actual HetNet learner for both 2P1A
and 4P6A. Historical failed artifacts remain intact. This establishes gradient
transport and shared-parameter correctness for these fixtures; it is not the
complete Gate A smoke run or evidence of learning performance.

## What was measured

`tests/helpers/hetnet_gradient_probe.py` instantiates the original
`A2CPolicy`, PCP environment, `Trainer`, and `MultiProcessTrainer` with four
processes (main plus three spawned workers). It uses the main recipe's real
communication, per-class critic, two state nodes, CPU float64 context, 5x5 map,
and vision 2. Each diagnostic performs three optimizer updates with batch size
4 and horizon 4 per process. The observed total is 16 transitions per update.
These smaller batch and horizon settings are explicitly structural fixtures.

The reference preserves the implementation's existing order of operations.
For process r and update k, define

\[
g_{r,k}=\operatorname{clipnorm}_{0.75}\left[
\nabla_\theta(50 L^{\mathrm{policy}}_{r,k}+L^{\mathrm{critic}}_{r,k})
\right],\qquad
g^{\mathrm{expected}}_k=
\frac{\sum_{r=0}^{3}g_{r,k}}{\sum_{r=0}^{3}T_{r,k}}.
\]

The actual losses and their episode reductions come from
`hetgat/policy.py:819–865`; `Trainer.compute_grad` selects that method at
`trainer.py:477`. Clipping happens within each process before aggregation.
The step denominator is the actual merged `num_steps`, as used by
`multi_processing.py:107–114`. This audit does not replace those choices with
global clipping, an alternative loss, or a different normalization.

The read-only `Trainer` subclass copies every named non-null gradient
immediately after the real `compute_grad` returns, before the aggregator runs.
The optimizer's current gradient is compared with the independently computed
fresh sum above for every parameter. The main process's current `.grad`
storage must equal its cached storage. For each worker, the cache content must
exactly equal that worker's fresh gradient, and its current gradient pointers
must remain equal to their first post-transport values. The initial IPC can
move a tensor into shared memory, so the pointer baseline is measured after
that transfer. Post-step model hashes must agree across all four processes,
and model parameters must change on every update.

The probe adds no loss, optimizer, gradient-clear, clipping, aggregation, or
model behavior. It observes the existing post-step memory-report command to
obtain worker hashes and pointers. Source hashes are checked before and after
each run. An alarm, bounded joins, termination, and kill fallback bound worker
lifetime. Exit 0 means PASS, exit 1 means a scientific FAIL, and exit 2 means
an execution ERROR. A failed diagnostic is never treated as an expected pass.

## Evidence

| Artifact | Result | Maximum gradient errors, updates 1–3 |
| --- | --- | --- |
| [Original actual HetNet, 2P1A](hetnet_gradient_intact_20260926_01.json) | FAIL | 0, 0.1803515501, 0.1659436053 |
| [Repaired actual HetNet, 2P1A, direct worker storage](hetnet_gradient_repaired_2P1A_storage_20260926_01.json) | PASS | 0, 0, 0 |
| [Repaired actual HetNet, 4P6A, direct worker storage](hetnet_gradient_repaired_4P6A_storage_20260926_01.json) | PASS | 0, 0, 0 |
| [Repaired deterministic transport surrogate](gradient_contract_repaired_final_pytest_20260926_01.json) | PASS | Within 1e-12 tolerance |
| [Final actual HetNet pytest fixture](hetnet_gradient_repaired_storage_pytest_20260926_01.json) | PASS | 0, 0, 0 |

The source fixture checks 68 named gradient tensors per update; the largest
checks 76. The differing sets reflect the original composition-specific
computation. Each run requires all four processes to have the same names and
order and requires that set to match the current and cached main gradients.
JSON artifacts name and hash compressed NPZ siblings containing each
process's fresh gradient and the expected, actual, and main cached arrays.
The original [failure summary](GRADIENT_CONTRACT_FAILURE.md) and original
candidate patch are historical records from before approval and application.

The deterministic surrogate runs the actual upstream multiprocessing loop
with an analytically simple linear loss and the same norm bound. It reads the
selected policy's clearing keywords from source, so it follows the approved
inner clearing semantics rather than introducing an independent fixture fix.
The actual-HetNet diagnostic independently executes the real policy loss.

Final test command, executed after the direct worker-storage instrumentation:

```sh
DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HETNET_GRADIENT_PROBE_OUTPUT=evidence/gate_a/gradient_contract_repaired_final_pytest_20260926_01.json \
HETNET_ACTUAL_GRADIENT_PROBE_OUTPUT=evidence/gate_a/hetnet_gradient_repaired_storage_pytest_20260926_01.json \
uv run --locked pytest -q tests/test_gradient_contract.py tests/test_hetnet_gradient_contract.py
```

Result: **6 passed in 25.62s**. The original pre-fix surrogate pytest run
reported two failed assertions and one passed assertion, correctly separating
stale-gradient failure from still-working shared parameter transport.

To produce fresh actual evidence, select a new output filename; existing JSON
and snapshot files are protected against overwriting:

```sh
DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --locked python tests/helpers/hetnet_gradient_probe.py \
  --output evidence/gate_a/new_gradient_probe.json \
  --updates 3 --batch-size 4 --horizon 4 --num-p 2 --num-a 1 \
  --timeout-seconds 120
```

Use `--num-p 4 --num-a 6` for the largest composition. Gate A should require
successful fresh-gradient, storage, and shared-weight checks before its full
source/largest training smokes, checkpoint round trips, and reproducibility
checks. Passing this structural diagnostic alone does not satisfy those
remaining requirements.
