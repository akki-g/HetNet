"""Generated evaluation wrappers submit through the real run helper and fake Slurm."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from publication_reconstruction import study
from scripts import prepare_pcp_frozen
from softrole import programme_evaluation

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("route", ["frozen", "publication", "programme"])
@pytest.mark.parametrize("parent_mode", ["standalone", "inherited", "fresh_terminal"])
def test_generated_submission_keeps_logs_with_prepared_artifacts(tmp_path, monkeypatch, route, parent_mode):
    parent_container = parent_mode != "standalone"
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/bin/python").symlink_to(sys.executable)
    shutil.copyfile(ROOT / "scripts/submit_stokes.py", checkout / "scripts/submit_stokes.py")
    container = checkout / "runs" / "comparison ' $(never-execute)"
    output = container / "evaluation" if parent_container else container
    output.mkdir(parents=True)
    if parent_container:
        (container / "run_info.json").write_text(json.dumps({"schema_version": 1,
            "title": "Original multi-stage comparison", "purpose": "Compare the locked arms"}))
    batch = output / ("evaluate.sbatch" if route == "programme" else "submit.sbatch")
    batch.write_text("#!/bin/bash\nexit 0\n")
    if route == "frozen":
        monkeypatch.setattr(prepare_pcp_frozen, "ROOT", checkout)
        script = prepare_pcp_frozen.submission_script(output, [dict(model="shared", training_seed=0, condition="nominal")])
    elif route == "publication":
        monkeypatch.setattr(study, "ROOT", checkout)
        script = study.submission_script(output)
    else:
        monkeypatch.setattr(programme_evaluation, "__file__", str(checkout / "src/softrole/programme_evaluation.py"))
        script = programme_evaluation.submission_script(output)
    (output / "submit.sh").write_text(script)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    capture = tmp_path / "submission.json"
    sbatch = binaries / "sbatch"
    sbatch.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
                     "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({'args':sys.argv[1:], "
                     "'run':os.environ['HETNET_RUN_DIR'], 'title':os.environ['HETNET_RUN_TITLE']}))\n"
                     "print('12345;stokes')\n")
    sbatch.chmod(0o755)
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "CAPTURE": str(capture), "HETNET_SUBMISSION_PYTHON": sys.executable}
    env.pop("HETNET_RUN_DIR", None)
    env.pop("HETNET_RUN_TITLE", None)
    if parent_mode == "fresh_terminal":
        env.pop("HETNET_SUBMISSION_PYTHON", None)
    if parent_mode == "inherited":
        env.update(HETNET_RUN_DIR=str(container), HETNET_RUN_TITLE="Original multi-stage comparison")
    result = subprocess.run(["bash", str(output / "submit.sh")], env=env,
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["run"] == str(container)
    assert record["args"][-1] == str(batch)
    assert record["args"][record["args"].index("--output") + 1].startswith(str(container / "logs") + "/")
    assert record["args"][record["args"].index("--error") + 1].startswith(str(container / "logs") + "/")
    assert (container / "run_info.json").is_file()
    assert len(list((container / "submissions").glob("*.response.json"))) == 1
    if parent_container:
        assert record["title"] == "Original multi-stage comparison"
        assert not (output / "run_info.json").exists()
    assert not (checkout / "logs_sr").exists()
    assert not (checkout / "logs_1").exists()
