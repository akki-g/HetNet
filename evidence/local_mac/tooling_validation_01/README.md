# Local Mac tooling validation

**53 selected tests passed** on the development M2. This is not a target Mac Gate A pass or a calibration/training result. See [MAC_RUNBOOK.md](../../../MAC_RUNBOOK.md).

Commands used the existing locked `.venv` with `DGLBACKEND=pytorch` and inherited `PYTHONHOME`/`PYTHONPATH` removed:

```bash
.venv/bin/python -m pytest -q tests/test_local_setup.py tests/test_local_job.py \
  tests/test_recording.py tests/test_grid_slurm.py tests/test_gate_a_contracts.py \
  -k 'not smoke_artifacts and not determinism and not cross_composition' \
  --junitxml=runs/mac_tooling_validation/local_contracts.xml
.venv/bin/python -m pytest -q tests/test_gradient_contract.py tests/test_hetnet_gradient_contract.py \
  --junitxml=runs/mac_tooling_validation/gradient_regression.xml
```

The first invocation passed 47 tests in 6.31 s, with three full-Gate-A artifact cases deliberately deselected. The second passed six bounded gradient regression cases in 20.86 s. XML copies preserve each selected test case. `validation.json` records the scientific inventory and lock hash. The original architecture, learner, recipe, lock and Stokes launcher had no changes in this fallback revision.

`runtime_probe_02/` contains a real import/interface check using the existing `.venv-gate-a-clean` interpreter on the M2. The setup helper's runtime program and the local launcher's runtime validation agree on interpreter paths/hash, package versions/origins/hashes, hostname, and hardware identity. Its source hashes match the final helper/launcher revision. `runtime_probe_01/` preserves the earlier successful interface probe before final launcher metadata-validation changes. Neither probe created an environment, proved a fresh installation, or exercised training performance.

The target must produce its own clean setup evidence, run the full Gate A on the current source revision, and then complete the two authorized endpoint calibrations before measured-cost review. No target outcome is supplied by these tests.
