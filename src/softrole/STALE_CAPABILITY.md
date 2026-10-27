# Stale capability: implementation and stability gate

The [supplied implementation design](../../docs/stale_capability_implementation_design.md)
defines this study. This first implementation stage covers Change 0, the opt-in
`graded-observation-v2` observation fix, and the stability-gate portion of Change F.
Stop for handoff here before implementing declarations, paired evaluation or the
main-study launchers. Implementation work authorizes no cluster jobs.

## Scientific document and pending decisions

Amendment A-3 is the user's scientific document. Its file/link has not been
provided; no hypotheses or claims are authored here. The implementation design
is preserved verbatim, including proposals; subsequent confirmations belong in
this runbook and the locked protocol.

On 26 October 2026 the user confirmed gate entropy coefficients `{0.003, 0.01}`,
seeds 0–2, 2M steps, success below 5% and mean action entropy below 0.01 nats per
agent-step, with the sliding-window convention below. Amendment A-3 is pending
at the user's request. Main-study budget 10M, A2 overstatement probability 0.25,
and development/final panel seeds 3100/3101 remain proposals. The main-study
entropy coefficient must be selected by the user after the gate.

## Arms and controls

The gate uses A1 settings, seeds 0–2, and a 2M-step budget per run. Configurations
derive from `programme_config("l1b_decomposed_beta1", seed, 1e-3,
profile=CAPCON_PROFILE)`: shared MLP, decomposed critic with beta 1, Adam at
0.001, real messages, three rounds, independent heads, CPU float64, four
collectors and one Torch thread per collector. Compositions are `(1,1)` and
`(2,1)`, P sensing levels `(1,2,3)`, A sensing `(0,)`, no sensor failures and no
held-out sensing lists. Gate A1 uses physical capability, an explicit actor tail
and `graded-observation-v2`.

The later main study has four arms, each at seeds 0–4:

| Arm | Environment | Actor tail | Overstatement |
|---|---|---|---|
| A1 reliable | graded-observation-v2 | Explicit | Off |
| A2 unreliable | graded-observation-v2 | Explicit | 0.25 proposed |
| A3 undeclared | graded-observation-v2 | Removed | Off |
| A1-open | corrected-observation-v1 | Explicit | Off |

In v2, outside flags remain visible because position codes already determine
them. Physical sensing masks target and total-occupancy counts. Observation width
is unchanged. Existing v1 configurations retain their behavior. In the later
declaration stage, the observation tail will carry declared sensing while the
critic and model action mask receive physical κ.

## Gate rule

Order complete episodes exactly as recorded in the ledger, assign cumulative
completion steps, and count episodes ending in `(start, end]`. Inspect every full
500,000-step sliding interval contained in `[1,000,000, final recorded step]`,
including the final batch's budget overshoot. The implementation checks every
distinct membership at both completion and expiration boundaries. This is a
declared ledger convention; parallel episodes do not have a wall-clock ordering.

For every update ending strictly after 1M steps, divide `action_entropy_sum` by
the sum of `episode.steps * episode.num_agents` in that update. A seed collapses
if any window's success fraction is below 0.05 or any such update's mean entropy
is below 0.01 nats per agent-step. Equality passes. Each coefficient needs all
three seeds complete and free of collapse. The smallest passing coefficient
requires user confirmation; if both collapse, stop without trying other values.

`gate-summary` audits the ledgers, archived source, checkpoint commitments and
completion receipts. Missing or invalid evidence is incomplete/error, never a
pass. It writes one fresh JSON file with per-coefficient and per-seed verdicts,
worst success windows, entropy violations and evidence hashes. It records eligible
coefficients, leaves `selected_coefficient` null and makes no automatic selection.

## Gate preparation and source locking

Use the pinned Python 3.12 `.venv` and `scripts/submit_stokes.py`; inspect active
Stokes jobs before changing its checkout. Prepare a fresh manifest after source
changes and keep the checkout and environment fixed through training. Source
checks run at preparation, launch and after training. Existing outputs and
protocol overrides are refused; interrupted runs are preserved, not overwritten.

These are commands for the user to run after deploying the reviewed source.
Implementation validation uses only local tests and dry runs.

```bash
cd ~/HetNet
squeue -u "$USER"
run_dir=$(.venv/bin/python scripts/submit_stokes.py --init-only \
  --title 'PCP stale capability stability gate' \
  --purpose 'Exploratory A1 stability check: entropy 0.003/0.01, seeds 0-2, 2M steps each.')

stale_prep_job=$(.venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
  slurm/softrole_stale_prepare.sbatch) && \
  .venv/bin/python scripts/submit_stokes.py --run-dir "$run_dir" -- \
    --dependency="afterok:${stale_prep_job%%;*}" --kill-on-invalid-dep=yes \
    slurm/softrole_stale_train.sbatch
```

Preparation creates `manifest.json` and dry-runs all six entries. An existing
manifest is validated without rewriting it. The array uses indices 0–2 for
`a1_entropy_0p003` seeds 0–2, and 3–5 for `a1_entropy_0p01` seeds 0–2. At most two
tasks run concurrently. Each training task requests four CPUs, 16 GB and six
hours; preparation requests one CPU, 2 GB and ten minutes. These are unmeasured
starting allocations. Both use account `cenyioha`, partition `normal` and
constraint `sapphirerapids`.

`SOFTROLE_STALE_MANIFEST` and `SOFTROLE_STALE_RUN_ROOT` default to
`$HETNET_RUN_DIR/manifest.json` and `$HETNET_RUN_DIR/training`. Unset stale exports
or point them inside this container; the submission helper enforces containment.
The batch scripts accept only an optional `--dry-run`, never protocol overrides.

Preview submission routing without creating a run directory or submitting:

```bash
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'PCP stale capability stability gate' \
  --purpose 'Preview gate preparation routing.' -- slurm/softrole_stale_prepare.sbatch
.venv/bin/python scripts/submit_stokes.py --dry-run \
  --title 'PCP stale capability stability gate' \
  --purpose 'Preview gate training routing.' -- slurm/softrole_stale_train.sbatch
```

The numerical CLI also supports a no-write preparation preview:

```bash
.venv/bin/python -m softrole.stale_capability prepare --stage gate \
  --output /tmp/stale-capability-preview/manifest.json --dry-run
```

After all gate jobs finish, summarize their records (or the same copied run
directory locally):

```bash
.venv/bin/python -m softrole.stale_capability gate-summary \
  --manifest "$run_dir/manifest.json" --run-root "$run_dir/training" \
  --output "$run_dir/gate_summary.json"
```

Preserve incomplete/error summaries. A later summary needs a fresh filename;
do not overwrite earlier evidence. A successful launcher writes
`stale_capability.json` only after its post-training source check.

Related preparation and training jobs share one fresh titled container:

```text
runs/<UTC timestamp>_pcp-stale-capability-stability-gate_<id>/
  README.md
  run_info.json
  submissions/
  logs/
  manifest.json
  training/a1_entropy_0p003/seed<N>/
    source/
    source_manifest.json
    config.json
    updates.jsonl
    episodes/
    episodes.index.json
    checkpoints/                       # checkpoints and sidecars
    finished.json
    stale_capability.json
  training/a1_entropy_0p01/seed<N>/
  gate_summary.json
```

The helper creates log directories before submission. All environment output
overrides must remain inside the container. Resource requests are unmeasured
starting values, not completion forecasts. Copy the same relative run directory
when transferring results. Local validation uses temporary directories outside
existing evidence, and submission checks use `--dry-run` only.

## Later stages

Main-study and evaluation commands are deferred until after the required gate
handoff. The design requires `first_saved_at_or_above` the declared training
budget, paired correct/overstated panels, and a separately labelled exploratory
`comm_off` condition. Evaluation outputs belong under the same run's `evaluation/`
directory. Do not run a placeholder command or infer approval from this plan.
