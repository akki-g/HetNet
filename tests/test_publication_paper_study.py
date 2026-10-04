"""Prepared study identities are concrete bytes, distinct from launch approval."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from publication_reconstruction import paper_study


ROOT = Path(__file__).resolve().parents[1]


def test_prepared_study_hashes_sources_dependencies_launchers_and_shared_simulator():
    identity = paper_study.prepared_identity()
    for kind in ("source", "dependencies", "launchers", "simulator_source"):
        manifest = identity[kind]
        expected = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
        assert manifest["sha256"] == expected
    assert set(identity["dependencies"]["files"]) == {"pyproject.toml", "uv.lock"}
    assert set(identity["launchers"]["files"]) == set(paper_study.LAUNCHER_FILES)
    for name in ("publication_reconstruction/runtime/hetgat/paper.py",
                 "publication_reconstruction/runtime/hetgat/graph/torch_backend.py",
                 "publication_reconstruction/paper_study.py", "softrole/model.py",
                 "softrole/publication_env.py", *paper_study.DEPENDENCY_FILES,
                 *paper_study.LAUNCHER_FILES):
        assert identity["source"]["files"][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name, digest in identity["simulator_source"]["files"].items():
        assert identity["source"]["files"][name] == digest
    assert set(identity["runtime_provenance"]) == {
        "valid", "origins_sha256", "missing", "unexpected", "changed"}


def test_pending_provenance_is_reported_without_accepting_it(monkeypatch):
    import publication_reconstruction.__main__ as launcher
    original = launcher.inspect_source_origins

    def pending():
        payloads, origins, audit = original()
        audit = {**audit, "valid": False,
                 "changed": {"example.py": {"expected_sha256": "old", "actual_sha256": "new"}}}
        return payloads, origins, audit

    monkeypatch.setattr(launcher, "inspect_source_origins", pending)
    result = paper_study.prepared_identity()
    assert result["runtime_provenance"]["valid"] is False
    assert "example.py" in result["runtime_provenance"]["changed"]


def test_preparation_writes_the_hashed_common_panel_and_identity(tmp_path):
    output = tmp_path / "preparation"
    paper_study.prepare(SimpleNamespace(run_root=tmp_path / "runs", output=output,
                                        message_backend="dgl", dry_run=False))
    saved = json.loads((output / "study.json").read_text())
    assert "prepared_identity" in saved
    assert saved["fc_scenarios_sha256"] == hashlib.sha256((output / "fc_scenarios.json").read_bytes()).hexdigest()
    assert len(saved["fc_scenarios"]) == 500
    assert saved["checkpoint_selection"].startswith("first saved checkpoint at or above28M")
    assert not (tmp_path / "runs").exists()
