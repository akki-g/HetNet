"""Validate the completed reconstruction evidence and record deliverable hashes."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PACKAGE = ROOT / "publication_reconstruction"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    origins = json.loads((PACKAGE / "ORIGINS.json").read_text())
    for name, row in origins["files"].items():
        assert digest(PACKAGE / "runtime" / name) == row["current_sha256"], name
    smoke = json.loads((HERE / "smoke_runs.json").read_text())
    assert len(smoke) == 18
    for row in smoke:
        assert row["finite_checkpoint"] and row["source_hashes_verified"] and row["parameter_signature_changed"]
        assert digest(ROOT / row["checkpoint"]) == row["checkpoint_sha256"]
    test_log = (HERE / "test_suite.txt").read_text()
    found = re.search(r"(\d+) passed in ([\d.]+)s", test_log)
    assert found and int(found[1]) >= 216, test_log[-2000:]
    assert "failed" not in test_log.lower()
    assert (HERE / "lock_check.txt").is_file()
    assert len(json.loads((HERE / "budgets/results.json").read_text())) > 0
    authored = [PACKAGE / "__main__.py", PACKAGE / "__init__.py", *HERE.glob("*.py"),
                *ROOT.glob("tests/test_publication*.py")]
    for path in authored:
        ast.parse(path.read_text(), filename=str(path))
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    changed_tracked = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    assert set(changed_tracked) == {"AGENTS.md", "README.md"}, changed_tracked
    result = {"status": "pass", "tests_passed": int(found[1]), "test_seconds": float(found[2]),
              "runtime_files": len(origins["files"]), "modified_from_recorded_sources": [
                  name for name, row in origins["files"].items() if row["modified"]],
              "actual_smoke_runs": len(smoke), "smoke_steps": sum(r["counts"]["env_steps"] for r in smoke),
              "smoke_episodes": sum(r["counts"]["episodes"] for r in smoke),
              "smoke_updates": sum(r["counts"]["updates"] for r in smoke),
              "original_tracked_runtime_changed": False, "research_training_or_cluster_submission": False,
              "historical_training_commit_identified": False, "source_hashes_verified": True,
              "python_syntax_checked": [str(p.relative_to(ROOT)) for p in authored],
              "git_diff_check": "passed"}
    smoke_files = {}
    for run_root in [ROOT / "runs/publication_reconstruction_validation_20260930",
                     ROOT / "runs/publication_reconstruction_validation_final_20260930"]:
        for path in sorted(run_root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                smoke_files[str(path.relative_to(ROOT))] = digest(path)
    (HERE / "smoke_artifact_manifest.json").write_text(json.dumps({
        "scope": "Exact initial/final validation artifacts; Python caches excluded",
        "initial_probe_note": "One successful PP run preceded a checkpoint-inspection import-path error in the harness; corrected harness reran the full matrix in a fresh final directory",
        "files": smoke_files}, indent=2) + "\n")
    (HERE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    files = {}
    paths = list(HERE.rglob("*")) + list(PACKAGE.rglob("*")) + list(ROOT.glob("tests/test_publication*.py"))
    for path in sorted(paths):
        if not path.is_file() or "__pycache__" in path.parts or path == HERE / "artifact_manifest.json":
            continue
        files[str(path.relative_to(ROOT))] = digest(path)
    (HERE / "artifact_manifest.json").write_text(json.dumps({
        "scope": "New analysis, reconstruction package and tests; excludes caches, itself and smoke-run archives",
        "smoke_archives": "Per-run source manifests plus checkpoint hashes in smoke_runs.json",
        "files": files}, indent=2) + "\n")
    print(json.dumps({**result, "manifest_files": len(files)}))


if __name__ == "__main__":
    main()
