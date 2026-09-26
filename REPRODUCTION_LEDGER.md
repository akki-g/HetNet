# Reproduction ledger

Research baseline: `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`. All source references use that revision unless a deviation commit is stated. Read `RESEARCH_NOTES.md` and `research/TRAINING_RESEARCH.md` for primary page/line citations.

| Dimension | Paper/supplement | Released path / selected protocol | Attribution |
|---|---|---|---|
| Method | HetNet-Real and Binary both studied; Fig.5c Binary | Real only, per-class A2C | User; main.py230–275 |
| Graph depth | Three layers, supplement §2.1 | Two layers, uavnet.py106–109 | Preserve release |
| Optimizer | Adam, 1e-3, supplement §2.1 | RMSprop, alpha .97, eps 1e-6; README LR 1e-4 | Preserve stepping trainer |
| Layer widths | Four 16-dimensional intermediate heads | Same intermediate widths; P5/A6 output; two layers | Released main.py239–250 |
| Tensor precision | Not a paper recipe | Float64 defaults with explicitly float32 attention parameters, preserved on strict load | main.py default; fastreal.py81–96 |
| Source task | PCP 2P1C, 5×5, cap80 | 2P1A, 5×5, cap80 | Supplement §1.2 + README |
| Larger team figure | Binary 3P3C/4P6C; exact per-curve map/horizon unstated | Real 3P3A/4P6A at fixed 5×5/80 | Declared natives, not exact Fig.5c reproduction |
| Additional natives | Not these figure conditions | 3P1A and 2P2A, fixed 5×5/80 | User experimental design |
| Vision | Not specified in README PCP command | Parser default2; width25×29=725 | PCP.py68–69; implementation Eq.I1 |
| Completion | All agents locate prey; A captures | All agents reach prey, every A captures; sinks retained | PCP.py392–543 |
| Seeds | 0/1/2 | P0/P1 0–4; P2 0–2 | User extension |
| Epochs / processes | Supplement and code differ in recipe | README 2000 / 4 total collectors | README PCP command |
| Epoch size / batch | Algorithm pseudocode not exact CLI accounting | 10 updates/epoch, ≥500 joint steps/collector/update | main.py45–48; trainer.py512–526 |
| Sample budget | Not inferred from plot | 40M–46.32M transitions/run; actual raw counts logged | Source arithmetic, not runtime measurement |
| Main RNG | Released seeding after model construction | Authorized early seed deviation B | Explicit reproduction repair |
| Printed cumulative counters | Not a paper recipe | Upstream overcounts; preserve stdout, correct JSONL accounting | Authorized logging deviation C |
| Observation slicing | Paper sensor contract | Intact upstream output retained; event masks applied after copy | Faithful baseline; no blanket repair |
| Gradient clearing | Older IC3Net lineage cached storage | Torch2.2 default replaces storage; verified failure after update1, repaired by explicit in-place clearing at three sites | User-approved deviation G, `0cfcea5`; original and repaired evidence retained |
| Evaluation policy | Published converged final policy; 50 trials | Fixed final epoch2000; 500-state banks; no learner | User protocol |
| Seed uncertainty | Paper SE across evaluation/training contexts | Show every seed, seed bootstrap, independent native/source seeds | User + Agarwal methodology |
| Cluster | No Stokes benchmark in source | Stokes CPU normal; live settings and calibration pending | User + ARCC docs |
| License | README says MIT | Tracked LICENSE is GPLv3, retained | Actual file controls provenance |

## Runtime ledger

Local Gate A passed as recorded below. Gate B, Stokes calibration, full training and frozen evaluation remain pending. Append concrete environment versions, test artifacts, run IDs and discrepancies as evidence is generated. Never replace a crashed/failed attempt with an unmarked rerun.

On 26 September 2026, clean local synchronization/import passed with Python 3.12.13, Torch 2.2.1, DGL 2.1.0, torchdata 0.7.1, NumPy 1.26.4 and Gym 0.26.2. Actual repaired 2P1A/4P6A three-update gradient probes pass with zero aggregation error and current worker/main gradient storage. These are engineering checks. The first full Gate A attempt, `runs/gate_a/full_20260926_01`, was intentionally interrupted when final gate-only edits raced its source fingerprint; the report, logs and interruption reason remain preserved. It is not a passed gate or a failed learning result.

The replacement full Gate A attempt, `runs/gate_a/full_20260926_02`, passed at 04:45:33 UTC on the same date: all twelve smoke runs, 22 selected tests and three direct gradient probes completed successfully. All 36 checkpoints match recorded tensor signatures; source single-process seed0 repeats have identical initialization, all three epoch signatures and non-timing metrics, while seed1 initialization differs. The final source smoke checkpoint strictly loads into all five compositions with identical names, shapes, dtypes and values. Scientific source hash was unchanged from start to finish. See [GATE_A_REPORT.md](GATE_A_REPORT.md) and the [archived machine report](evidence/gate_a/full_20260926_02/gate_a_report.json). This is local macOS engineering evidence, with no inference about Stokes costs, learning or frozen performance.
