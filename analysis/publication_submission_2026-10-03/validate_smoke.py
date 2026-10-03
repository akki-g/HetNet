"""Read-only reconciliation of the bounded archived-launcher validation runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from publication_reconstruction.artifacts import digest, lineage_metrics, load_checkpoint
from publication_reconstruction.evaluation import verify_evaluator_archive
import numpy as np
import torch


def equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def validate(run_root):
    states, statuses = {}, {}
    for name in ("paused", "continued", "uninterrupted"):
        run = run_root / name
        status = json.loads((run / "run_status.json").read_text())
        statuses[name] = status
        states[name] = load_checkpoint(run, status["checkpoint"])
    actual, expected = states["continued"], states["uninterrupted"]
    for field in ("policy_net", "trainer", "log"):
        equal(actual[field], expected[field])
    for field in ("counts", "rng_states", "recorder_state", "milestones_reached"):
        equal(actual["recovery"][field], expected["recovery"][field])
    rows = lambda name: [{k: v for k, v in row.items() if k != "wall_time_seconds"}
                          for row in lineage_metrics(run_root / name)]
    equal(rows("continued"), rows("uninterrupted"))
    assert statuses["paused"]["stop_reason"] == "paused_wall_time"
    assert not statuses["paused"]["scientific_budget_completed"]
    assert actual["reconstruction"]["counts"] == {"env_steps": 144, "episodes": 48, "updates": 6, "epoch": 2}
    reports = [json.loads((run_root / f"{condition}.json").read_text()) for condition in ("failure", "sham")]
    for report in reports:
        assert report["parameters_unchanged"]
        assert report["checkpoint_sha256"] == digest(statuses["continued"]["checkpoint"])
        archive = report["evaluator"]["source_archive"]
        assert verify_evaluator_archive(archive["path"], report["evaluator"]["source"]) == archive["manifest_sha256"]
    equal(reports[0]["scenarios"], reports[1]["scenarios"])
    for failure, sham in zip(reports[0]["per_episode"], reports[1]["per_episode"]):
        equal(failure["trace"][:failure["event_step"]], sham["trace"][:sham["event_step"]])
    summary = json.loads((run_root / "summary.json").read_text())
    assert len(summary["paired_groups"]) == 1 and summary["paired_groups"][0]["training_seeds"] == 1
    return {"passed": True, "purpose": "bounded engineering checks, not task-performance evidence",
            "run_root": str(run_root), "counts": actual["reconstruction"]["counts"],
            "exact_comparisons": ["model", "active_optimizer", "all_collector_rng", "log", "counts", "retained_epoch_metrics"],
            "failure_sham_episodes_per_condition": 2, "parameters_unchanged": True,
            "paired_groups": 1, "artifacts": {str(p.relative_to(run_root)): digest(p)
                for p in sorted(run_root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts},
            "validator_sha256": digest(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/publication_submission_validation_20261003")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.run_root.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"passed": True, "output": str(args.output), "counts": result["counts"]}))
