"""Create a temporary prepared run for mocked Slurm shell-route tests."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure_stokes_run(env, tmp_path):
    env = dict(env)
    run = Path(tmp_path) / "titled run"
    run.mkdir(exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    title = "launcher-route-check"
    (run / "run_info.json").write_text(json.dumps({
        "schema_version": 1, "title": title, "purpose": "Test launcher routing without submitting jobs",
        "run_dir": str(run.resolve()),
    }) + "\n")
    env.update(HETNET_RUN_DIR=str(run.resolve()), HETNET_RUN_TITLE=title,
               HETNET_SUBMISSION_PYTHON=sys.executable)
    checkout = Path(env.get("SLURM_SUBMIT_DIR", ROOT))
    if checkout.resolve() != ROOT:
        (checkout / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts/stokes_run.sh", checkout / "scripts/stokes_run.sh")
    return env
