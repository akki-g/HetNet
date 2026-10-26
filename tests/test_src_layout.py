"""The source layout works outside the checkout and preserves archive provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from softrole.publication_env import (
    SIMULATOR_FILES,
    checkpoint_binding,
    load_environment_class,
    simulator_identity,
)
from softrole.train import source_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_installed_experimental_packages_import_outside_checkout(tmp_path):
    # -I excludes the checkout and PYTHONPATH. This exercises the installed
    # editable distributions used by collectors and launchers, not pytest paths.
    names = ("softrole.config", "softrole.model", "publication_reconstruction",
             "publication_reconstruction.artifacts", "hetgat.policy",
             "hetnet_ext.signatures", "ic3net_envs.predator_capture_env",
             "action_utils", "data", "multi_processing", "trainer")
    code = """
import importlib, json, sys
print(json.dumps({name: importlib.import_module(name).__file__ for name in sys.argv[1:]}))
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, *names], cwd=tmp_path,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)
    assert set(paths) == set(names)
    assert all(Path(path).resolve().is_relative_to(ROOT / "src") for path in paths.values())


def test_snapshot_covers_live_source_and_launcher_files_with_repo_relative_paths(tmp_path):
    expected = {str(path.relative_to(ROOT)) for path in (ROOT / "src").rglob("*.py")
                if not path.is_relative_to(ROOT / "src/envs/examples")}
    # The ignored native-env demonstration is not an experiment dependency;
    # the separate publication runtime example remains pinned source evidence.
    expected.update(str(path.relative_to(ROOT)) for directory in ("scripts", "slurm")
                    for path in (ROOT / directory).iterdir()
                    if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"})
    expected.update(("pyproject.toml", "uv.lock", "requirements.txt"))
    identity = source_snapshot(tmp_path)
    assert expected <= identity["files"].keys()
    assert not any(name.startswith("tests/") for name in identity["files"])
    assert not any(name.startswith(("softrole/", "publication_reconstruction/", "hetgat/", "envs/"))
                   for name in identity["files"])
    for name, digest in identity["files"].items():
        original = (ROOT / name).read_bytes()
        assert (tmp_path / "source" / name).read_bytes() == original
        assert hashlib.sha256(original).hexdigest() == digest
    assert identity["sha256"] == hashlib.sha256(
        json.dumps(identity["files"], sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("prefix", ("", "src/"), ids=("legacy-flat", "src-layout"))
def test_paper_checkpoint_binding_accepts_both_preserved_source_layouts(tmp_path, prefix):
    # Historical manifests keep their original logical paths and hashes. A new
    # archive includes src/ without changing the simulator's scientific identity.
    identity = simulator_identity()
    files = {}
    for name in (*SIMULATOR_FILES, "softrole/__init__.py"):
        relative = prefix + name
        target = tmp_path / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "src" / name).read_bytes())
        files[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (tmp_path / "source_manifest.json").write_text(json.dumps({"files": files, "sha256": digest}))
    checkpoint = tmp_path / "checkpoints/epoch0001.pt"
    binding = checkpoint_binding(checkpoint, {"source_sha256": digest, "simulator_source": identity})
    assert binding["sha256"] == identity["sha256"]
    assert Path(binding["root"]) == tmp_path / "source" / prefix
    assert load_environment_class("pcp", binding).__name__ == "PredatorCaptureEnv"
    (tmp_path / "source" / prefix / SIMULATOR_FILES[-1]).write_text("# changed\n")
    with pytest.raises(ValueError, match="source bytes"):
        checkpoint_binding(checkpoint, {"source_sha256": digest, "simulator_source": identity})
