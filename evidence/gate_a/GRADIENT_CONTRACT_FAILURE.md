# Scientific Gate A failure: current gradients diverge from the cached aggregate

Observed locally on 2026-09-26 with locked Python 3.12.13, Torch 2.2.1,
three spawned workers plus the main process, and the unchanged upstream
`MultiProcessTrainer`. This is engineering evidence from a deterministic
surrogate trainer, **not a HetNet training result**. The surrogate retains both
outer RMSprop and inner policy-local Adam default `zero_grad()` calls and
per-process gradient clipping; it does not patch the aggregator.

| Update | Workers see main's updated weights | Cached storage is current | Fresh sum / actual total steps reaches optimizer | Maximum absolute gradient error |
|---|---|---|---|---|
| 1 | Yes | Yes | Yes | 1.73472348e-18 |
| 2 | Yes | No | No | 0.579605245 |
| 3 | Yes | No | No | 0.566336786 |

For update 2, the fresh expected gradient is approximately
`[0.0542528745, 0.0456027189, 0.0059346840]`, but the optimizer's actual current
gradient is `[0.2927699647, 0.3659624559, 0.5855399295]`. The cached aggregate
also differs. Its storage address no longer matches `parameter.grad`.
The shared-parameter test passes while the learner aggregation contract fails.

The direct probe exited **1**, with full evidence in
[gradient_contract_intact_20260926_01.json](gradient_contract_intact_20260926_01.json).
An independent pytest invocation also exited **1**: **2 failed, 1 passed in
6.79 seconds**. It passed shared weights, failed fresh gradient aggregation,
and failed current storage; evidence is
[gradient_contract_pytest_20260926_01.json](gradient_contract_pytest_20260926_01.json).
No `xfail`, skip, or patched clearing behavior was used to turn this into a pass.

Commands used:

```sh
DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python tests/helpers/shared_gradient_probe.py --output evidence/gate_a/gradient_contract_intact_20260926_01.json --updates 3 --timeout-seconds 60
DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HETNET_GRADIENT_PROBE_OUTPUT=evidence/gate_a/gradient_contract_pytest_20260926_01.json uv run --locked pytest -q tests/test_gradient_contract.py
```

These evidence paths must not be reused because the probe refuses overwrite.
Use fresh paths for subsequent checks. The
[candidate clearing change](gradient_clear_candidate.patch.txt) is only a
review proposal; **it has not been applied**. Gate A remains failed, and these
observations must not be described as successful multiprocessing training.
