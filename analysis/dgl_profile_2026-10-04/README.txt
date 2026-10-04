DGL cost profile, Binary-versus-Real timing and communication-range evidence (4 October 2026)

Purpose: evidence for three questions: (1) does Binary's extra binarization computation explain the
Stokes PCP Binary slowdown, (2) did the HetNet authors train with a communication radius, and (3) how
much of a HetNet reconstruction update is spent inside DGL. Exploratory engineering measurements on one
Mac; not a Stokes measurement, not a research result, and no runtime source was changed.

Recompute every reported number (read-only; writes summary.json with SHA256 of all 132 inputs):
    .venv/bin/python analysis/dgl_profile_2026-10-04/scripts/summarize.py

Machine and software: Apple M5 Max (18 logical CPUs, 36 GiB), macOS 26.6 arm64, Python 3.12.13,
Torch 2.2.1, DGL 2.1.0, NumPy 1.26.4, one Torch/BLAS thread. runtime_identity.json shows the executed
runtime copy matched all 38 files of publication_reconstruction/runtime in the working tree.

Workload for every in-situ run: PCP 2P1A, supplement-v1, corrected-v1, seed 991, horizon 80,
ONE collector (nprocesses 1), 500-step floor, 4 complete updates (epoch_size 4, num_epochs 1),
Real or Binary-16. Production uses four collectors; parent IPC/aggregation is not measured here.

raw/ contents (checkpoint .pt files omitted; checkpoint_records.jsonl paths point to the deleted scratch):
  base_real, base_binary    uninstrumented control runs (scripts/run.sh TAG 0 VARIANT)
  ins_real, ins_binary      coarse timers (scripts/run.sh TAG 1 VARIANT; scripts/instrument.py)
  det_real                  per-DGL-method timers (scripts/run_detail.sh det_real 1 real)
  cprofile_real/_binary     cProfile runs; .prof files are pstats data (scripts/analyze.py prints shares)
  dense_vs_dgl.txt          saved rerun of scripts/dense_vs_dgl.py with machine/version/load header

To rerun the training-based measurements, place a copy of the runtime next to the scripts and run
from scripts/ with bash (outputs go to scripts/out_TAG):
    cp -R publication_reconstruction/runtime analysis/dgl_profile_2026-10-04/scripts/runtime
    cd analysis/dgl_profile_2026-10-04/scripts && bash run.sh base_real 0 real
The cProfile runs used the same arguments as run.sh with INSTRUMENT unset, invoked as
    .venv/bin/python -m cProfile -o VARIANT.prof runtime/main.py <run.sh arguments>
with --use_binary --msg_dim 16 added for Binary. manifest.json is a placeholder ({}): main.py only
hashes --source_manifest, so these runs carry no source-archive provenance of their own.

Limits: one machine, one seed, four early-training updates, single collector, measured sequentially.
Timers add 5.2% (Real) and 0.7% (Binary) overhead and leave updates/parameter signatures identical to
the uninstrumented runs. The dense microbenchmark reproduces the shapes and per-relation operations of
the replaced DGL portion, not the complete model, and is not a numerical-equivalence test. The 2.43x
end-to-end figure is an estimate from these measurements, not a measured dense implementation.

Operation counts (added for the CUDA question; scripts/count_ops.py):
  raw/op_counts_insitu_real.json   one PCP Real update (557 steps) under torch.autograd.profiler with
                                   record_shapes; counts PyTorch (aten) ops per environment step split into
                                   forward and backward. DGL's own C kernels are not aten ops and are not
                                   counted. raw/op_counts_insitu_run/ holds that run's records; its update
                                   matches base_real's first update in every non-timing field.
  raw/op_counts_dense.json         the dense message-passing half of dense_vs_dgl.py, 80 steps + backward.
  'compute_leaf_ops_per_step' excludes the explicitly listed view/allocation ops; it still includes
  dispatcher no-ops (for example aten::to with no dtype change), so it overstates GPU kernel launches.
  Rerun: copy the runtime into scripts/runtime as above, then from scripts/ with one-thread variables and
  COUNT_UPDATES=$PWD/out_ops/updates.jsonl:
    python count_ops.py insitu ../raw/op_counts_insitu_real.json <run.sh arguments with --epoch_size 1>
    python count_ops.py dense ../raw/op_counts_dense.json
