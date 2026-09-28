# Stokes resources and conditional runtime estimate, 2026-09-28

This is a public-source audit, not a Stokes benchmark. No authenticated cluster query, allocation, submission, or training was performed for this note. **There is no current four-process Stokes timing sample.** The approximately 22 seconds/epoch in [the task specification](TASK_SPEC.md) is an earlier sandbox observation for **one process**; it does not measure this sweep's four-process workload. No fixed speedup or slowdown follows from CPU generation, aggregate core count, or interconnect speed.

## Current public evidence

The official [Node Matrix](https://arcc.ist.ucf.edu/status/stokes.html) obtains its data from [stokesNodes.json](https://arcc.ist.ucf.edu/status/stokesNodes.json). The archived response was fetched at **2026-09-28 17:26:32 UTC**, with HTTP `Last-Modified: Mon, 28 Sep 2026 17:15:01 GMT`. The JSON's own time is `2026-09-28T13:15:01.559186`, without a stated timezone. It lists **160 nodes and 7,336 total `cputot` CPU slots**. These are advertised scheduler CPU slots, not a verified count of physical cores, idle capacity available to this account, or the nodes belonging to the `normal` partition.

| Generation feature, as written | `cputot` slots/node | Node count | Speed feature, as written |
|---|---:|---:|---|
| `skylake` | 32 | 26 | `speed210` |
| `cascadelake` | 48 | 36 | `speed240` |
| `icelake` | 48 | 55 | `speed240` |
| `sapphirerapids` | 48 | 40 | `speed240` |
| `cascadelake` | 40 | 1 | `speed250` |
| `saphirerapids` | 48 | 1 | `speed180` |
| `saphirerapids` | 128 | 1 | `speed180` |

The two spelling variants are retained. The final row also carries a `proc48` feature despite `cputot=128`; the data are not silently reconciled. These scheduler feature labels establish a heterogeneous advertised inventory, **not verified processor SKUs, clock frequencies, socket topology, or physical-core/thread ratios**. In particular, this note does not convert `speed240` to a measured sustained clock or infer runtime from it.

The static official pages disagree about aggregate capacity: [RCI's Stokes page](https://rci.research.ucf.edu/resource/stokes/) states 8,544 compute cores; [About Stokes](https://arcc.ist.ucf.edu/index.php/resources/stokes/about-stokes) states 7,500; the [ARCC homepage](https://arcc.ist.ucf.edu/) states more than 8,800. The RCI-linked [June 2025 ARCC description](https://rci.research.ucf.edu/wp-content/uploads/sites/38/2025/06/ARCC-and-Advanced-Research-Network-Updated-Jun11-2025.docx) describes 192 nodes, approximately 8,500 compute cores, and 100/200 Gb/s HDR interconnect. None establishes the hardware or concurrency that this account can obtain now. The public status feed is more recent inventory evidence, but its scope is not proven to be identical to those descriptions.

Raw JSON, the status HTML, the June 2025 document, exact URLs, response headers, byte lengths, SHA-256 digests, and the derived inventory are preserved in [stokes_20260928/](stokes_20260928/). The JSON is 46,693 bytes with SHA-256 `02f079fa4dbd526578ba5e678bf4143b8d1b88b0225a29fb29d4b321d5627444`.

## Allocation and partition limits

The official [scheduler introduction](https://arcc.ist.ucf.edu/docs/scheduler/) states a default **80,000 CPU core-hours per faculty account per month**, shared by its users, without rollover, and directs users to `myusage` for the actual balance. CPU charges scale with allocated CPUs and elapsed execution time. This nominal monthly allocation is not evidence of the account's remaining balance. Pending queue time affects delivery time but is not elapsed execution time for an allocated training job.

The [limitations guide](https://arcc.ist.ucf.edu/docs/scheduler/limitations/) identifies `normal` as the primary queue and `preemptable` as secondary, with priority affected by the previous 14 days of usage. The reviewed public guide provides **no numeric `normal` maximum walltime, concurrent-job cap, or account/QOS cap**. No such cap is inferred here. Read-only live checks must establish partition walltime, association/QOS limits, permitted CPU allocation, and current balance. The actual allocated job should record CPU model/topology, node name, allocated CPU count, and runtime software versions. An idle percentage in this public snapshot is not a concurrency promise.

## User-supplied live Stokes evidence

Later in this conversation on 28 September, Akki supplied the output of `scontrol`, `sinfo`, `sacctmgr` and `myusage`; raw text is preserved in [user_terminal.txt](../evidence/stokes/preflight_20260928/user_terminal.txt). Its collection timestamp was not provided, and the assistant has not authenticated to the cluster. This is stronger evidence for the target partition/account than the public aggregate pages:

- `normal`: **153 nodes, 6,928 allocated-capacity CPU slots**, state UP; `MaxTime=UNLIMITED` / `infinite`, and a ten-minute default if no time is requested. Explicit calibrated finite job time remains required.
- Associated account `cenyioha`, user `ak565492`, default QoS `normal`; the user association reports `MaxJobs=250`.
- September allocation: **0 CPU-hours used of 80,000**, hence 80,000 remaining at that observation.
- Blank MaxSubmit/GrpTRES fields at the shown association alone do not resolve all aggregate/QoS limits or compute prerequisites. Ordinary association listings include inherited Max limits unless WOPLimits is used; ancestor Grp* and QoS account/group limits require separate interpretation. Those remain to be verified before submission. Unlimited partition time does not establish unlimited effective QoS/account wall time.

Subsequent supplied output shows the `cenyioha` account and its users with MaxJobs=250; the requested `normal` QoS limit fields are blank, the user queue is empty, and Anaconda 2024.10 is available. `quota -s` returned no supplied output. These observations do not replace a compute-node Python/network test or a Lustre headroom check; the root association was not included in the account query. Commands will use the known associated account explicitly.

A 21-job, 84-CPU sweep is below the observed user MaxJobs and uses about 1.21% of normal's advertised CPU capacity if it all runs concurrently; this is not an availability guarantee. The three raw CPU-hour scenarios below consume 1.75%, 3.5% and 7% of the observed remaining balance. Applying only the 1.3 time margin gives 1,820, 3,640 and 7,280 CPU-hours before calibration, overhead and reserves. The independent 4,000-hour review threshold still applies. The [draft preflight](../evidence/stokes/preflight_20260928/preflight_draft.json) records verified facts and unresolved fields explicitly, and is deliberately not launchable yet.

## Workload and conditional costs

The checked-in [grid](../configs/run1_grid.json) contains 21 runs: five seeds each for 2P1A, 3P3A, and 4P6A, and three each for 3P1A and 2P2A. Each run uses four total collector processes, 2,000 epochs, 10 optimizer updates/epoch, a 500-transition target per collector/update, and horizon 80. With complete-episode batch overshoot, that is 40.00–46.32 million joint environment transitions/run, or 840.00–972.72 million over 21 runs. It is not a count of individual-agent observations. Forty checkpoints/run (epochs 50, 100, …, 2000, with the final saved once) implies 840 training checkpoints across the sweep; separate calibration artifacts are additional. The two prescribed 20-epoch endpoint calibrations are separate work.

For an **assumed common** epoch duration \(s\) seconds, excluding setup, checkpoint and other residual costs:

\[
H_{\mathrm{run}}=2000s/3600,\qquad
\mathrm{CH}_{21}=21\times4\times H_{\mathrm{run}}.
\]

With \(C\) jobs able to run continuously, equal durations, immediate scheduling, and no contention slowdown, a wave schedule takes \(\lceil21/C\rceil H_{\mathrm{run}}\). These are sensitivity scenarios, **not measured forecasts**:

| Assumed seconds/epoch | Hours/run | Total allocated CPU-hours | Sequential, hours | 4 concurrent jobs / 16 CPUs, hours | 7 concurrent / 28 CPUs, hours | 21 concurrent / 84 CPUs, hours |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 16.7 | 1,400 | 350 | 100 | 50 | 16.7 |
| 60 | 33.3 | 2,800 | 700 | 200 | 100 | 33.3 |
| 120 | 66.7 | 5,600 | 1,400 | 400 | 200 | 66.7 |

Actual compositions, seeds, CPU generations, and contention can differ in speed. Queue delays, per-job startup, first-epoch costs, checkpoint I/O, calibration, failed attempts, and evaluation are absent from the table. No row asserts that its requested walltime is permitted by `normal`, that its concurrency is attainable, or that its CPU-hour cost fits the current account balance. The 120-second case already exceeds the existing 4,000 CPU-hour review threshold before overhead; the 10%-of-remaining-balance threshold also depends on a fresh `myusage` result.

The useful next timing evidence is the prescribed 20-epoch 2P1A and 4P6A calibration **inside an actual four-CPU Stokes allocation with the unchanged full recipe**. Preserve chronological epoch durations, first-epoch excess, checkpoint duration, total allocated elapsed time, and residual time outside epoch/checkpoint timers. Apply the existing budget calculation to those measurements and the fresh account balance; identify interpolation between the endpoint compositions and any safety margin as assumptions. A CPU-generation tag alone cannot replace this measurement, and local Mac measurements remain local measurements.

This note makes no changes to the experimental recipe or execution code.
