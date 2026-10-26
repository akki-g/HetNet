"""Every static Stokes route fails before work without its prepared run context."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATCH_FILES = sorted((ROOT / "slurm").glob("*.sbatch"))


@pytest.mark.parametrize("script", BATCH_FILES, ids=lambda path: path.stem)
def test_static_batch_requires_prepared_titled_run_before_any_work(tmp_path, script):
    tools = tmp_path / "tools"
    tools.mkdir()
    sentinel = tmp_path / "unexpected-work"
    for name in ("module", "srun", "sbatch"):
        tool = tools / name
        tool.write_text('#!/bin/bash\nprintf "unexpected" > "$WORK_SENTINEL"\nexit 99\n')
        tool.chmod(0o755)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("HETNET_RUN_", "BASH_FUNC_")) and key not in ("BASH_ENV", "ENV")}
    environment.update(SLURM_SUBMIT_DIR=str(ROOT), PATH=f"{tools}:{environment['PATH']}",
                       WORK_SENTINEL=str(sentinel))
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert not sentinel.exists()
    assert "submit_stokes.py" in result.stderr or "HETNET_RUN_DIR" in result.stderr


@pytest.mark.parametrize("script", BATCH_FILES, ids=lambda path: path.stem)
def test_static_batch_scheduler_paths_come_from_submission_helper(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    directives = [line for line in script.read_text().splitlines() if line.startswith("#SBATCH")]
    assert not any(line.startswith(("#SBATCH --output", "#SBATCH --error")) for line in directives)
