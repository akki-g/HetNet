"""Read-only compatibility checks on existing HetNet engineering artifacts."""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from publication_reconstruction.artifacts import load_checkpoint, _validate_optimizer, _validate_recorded_checkpoint
from publication_reconstruction.study import _validate_preflight_ledger
from publication_reconstruction.__main__ import inspect_source_origins, resolve_resume


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(run):
    protocol = read(run / "protocol.json")
    rows = []
    checkpoints = sorted((run / "checkpoints").rglob("*.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoint: {run}")
    for checkpoint in checkpoints:
        before = digest(checkpoint)
        saved = load_checkpoint(run, checkpoint)
        _validate_optimizer(saved, protocol, saved["reconstruction"]["counts"]["updates"])
        if protocol.get("schema_version") == 2:
            _validate_recorded_checkpoint(run, checkpoint, checkpoint.read_bytes(), saved, protocol)
        if digest(checkpoint) != before:
            raise ValueError(f"Checkpoint changed: {checkpoint}")
        rows.append(dict(path=str(checkpoint.relative_to(ROOT)), sha256=before,
            model_spec=protocol.get("model_spec", "public-code-v1"),
            optimizer=protocol["optimizer"]["name"],
            counts=saved["reconstruction"]["counts"]))
    return rows


result = dict(checkpoints=[], preflight_ledgers=[], resume=[], runtime_origins_valid=False)
for parent in ("stokes_runs/hetnet_preflight_903502", "runs/hetnet_preflight_mac_20261003_234455_727228"):
    for workload in ("pp_real", "pcp_real", "fc_real", "pcp_binary"):
        run = ROOT / parent / workload / "seed991"
        rows = check(run)
        result["checkpoints"].extend(rows)
        saved = load_checkpoint(run, ROOT / rows[-1]["path"])
        updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
        seconds, steps = _validate_preflight_ledger(updates, read(run / "run_status.json"),
            read(run / "training_segment.json"), read(run / "protocol.json"), saved)
        original = read(run / "preflight.json")["measured_update_steps_per_second"]
        if steps / seconds != original:
            raise ValueError(f"Reported throughput changed: {run}")
        result["preflight_ledgers"].append(dict(run=str(run.relative_to(ROOT)),
            updates=len(updates), steps=steps, steps_per_second=original))

for path in sorted((ROOT / "runs/publication_reconstruction_validation_final_20260930").glob("*/protocol.json")):
    result["checkpoints"].extend(check(path.parent))
for name in ("paused", "continued", "uninterrupted"):
    run = ROOT / "runs/publication_submission_validation_20261003" / name
    result["checkpoints"].extend(check(run))
    if name == "paused":
        for checkpoint in sorted((run / "checkpoints").rglob("*.pt")):
            destination = ROOT / "runs/hetnet_validity_read_only_resume_probe"
            resolved = resolve_resume(SimpleNamespace(run_dir=run, checkpoint=checkpoint,
                output=destination, episode_log=None, wall_seconds=0))
            if destination.exists():
                raise ValueError("Read-only resume probe unexpectedly created its output")
            result["resume"].append(resolved["continuation"])
result["runtime_origins_valid"] = inspect_source_origins()[2]["valid"]
if not result["runtime_origins_valid"]:
    raise ValueError("Runtime provenance changed")
result["status"] = "passed"
result["scope"] = "actual checkpoint decoding, optimizer/default validation and ledger reconciliation; no policy execution"
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
