"""Diagnose deployment drift without accepting it or invoking the learner."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from publication_reconstruction import __main__ as launcher


ROOT = Path(__file__).resolve().parents[1]


def source_tree(tmp_path):
    package = tmp_path / "publication_reconstruction"
    source = package / "runtime/main.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('approved')\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    (package / "ORIGINS.json").write_text(json.dumps({"files": {"main.py": {"current_sha256": expected}}}))
    return package, source


@pytest.mark.parametrize("drift", ["missing", "unexpected", "changed", "multiple"])
def test_inventory_diagnoses_exact_files_and_preserves_source(tmp_path, monkeypatch, drift):
    package, source = source_tree(tmp_path)
    before_manifest = (package / "ORIGINS.json").read_bytes()
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if drift == "missing":
        source.unlink()
    if drift in ("unexpected", "multiple"):
        (source.parent / ".DS_Store").write_bytes(b"unapproved metadata")
    if drift in ("changed", "multiple"):
        source.write_text("print('unapproved')\n")
    monkeypatch.setattr(launcher, "HERE", package)
    _, _, audit = launcher.inspect_source_origins()
    assert not audit["valid"]
    assert audit["missing"] == (["main.py"] if drift == "missing" else [])
    assert audit["unexpected"] == ([".DS_Store"] if drift in ("unexpected", "multiple") else [])
    assert audit["expected_inventory"] == {"main.py": expected_hash}
    if drift in ("changed", "multiple"):
        assert audit["changed"]["main.py"] == {
            "expected_sha256": expected_hash, "actual_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    else:
        assert not audit["changed"]
    with pytest.raises(RuntimeError) as error:
        launcher.validate_source_origins()
    assert json.dumps({key: audit[key] for key in ("missing", "unexpected", "changed")}, sort_keys=True) in str(error.value)
    assert "do not regenerate" in str(error.value)
    assert (package / "ORIGINS.json").read_bytes() == before_manifest


def test_bytecode_remains_non_source_but_other_metadata_does_not(tmp_path, monkeypatch):
    package, source = source_tree(tmp_path)
    (source.parent / "__pycache__").mkdir()
    (source.parent / "__pycache__/main.cpython-312.pyc").write_bytes(b"cache")
    (source.parent / "main.pyc").write_bytes(b"cache")
    monkeypatch.setattr(launcher, "HERE", package)
    payloads, _ = launcher.validate_source_origins()
    assert set(payloads) == {"main.py"}


@pytest.mark.parametrize("valid", [True, False])
def test_read_only_cli_reports_inventory_and_exit_status(tmp_path, valid):
    package, source = source_tree(tmp_path)
    for name in ("__init__.py", "__main__.py"):
        shutil.copy2(ROOT / "publication_reconstruction" / name, package / name)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/record_publication_sources.py", scripts)
    if not valid:
        source.unlink()
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, "-B", str(scripts / "record_publication_sources.py"), "--check"],
                            text=True, capture_output=True, check=False)
    audit = json.loads(result.stdout)
    assert result.returncode == (0 if valid else 1)
    assert audit["valid"] == valid and audit["execution"]["hostname"]
    assert audit["missing"] == ([] if valid else ["main.py"])
    assert audit["expected_runtime_files"] == 1 and audit["runtime_files"] == int(valid)
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_all_required_runtime_files_are_distributable_by_git():
    origins = json.loads((ROOT / "publication_reconstruction/ORIGINS.json").read_text())["files"]
    paths = [f"publication_reconstruction/runtime/{name}" for name in origins]
    assert all((ROOT / name).is_file() for name in paths)
    # --no-index tests the ignore rule even when a file is already tracked. A
    # future fresh checkout must not depend on a locally ignored runtime file.
    result = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=ROOT,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 1, f"Required runtime files are ignored: {result.stdout} {result.stderr}"


def test_source_audit_batch_is_read_only_and_propagates_guard_failure(tmp_path):
    script = ROOT / "slurm/publication_source_audit.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    content = script.read_text()
    assert "#SBATCH --cpus-per-task=1" in content and "#SBATCH --time=00:05:00" in content
    tools = tmp_path / "tools"
    tools.mkdir()
    for name, body in {
        "module": "exit 0\n",
        "srun": "exec \"$@\"\n",
        "audit interpreter": 'printf "%s\\n" "$@"\nexit 17\n',
    }.items():
        path = tools / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, text=True, capture_output=True,
        env={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
             "SLURM_SUBMIT_DIR": str(tmp_path), "HETNET_PYTHON": str(tools / "audit interpreter")})
    assert result.returncode == 17
    assert result.stdout.splitlines() == ["-u", "scripts/record_publication_sources.py", "--check"]
    assert "--output" not in result.stdout.splitlines() and " train " not in content
