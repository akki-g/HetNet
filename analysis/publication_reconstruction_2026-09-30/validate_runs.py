"""Bounded actual training checks of isolation, budgets, gradients and checkpoints."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / "runs/publication_reconstruction_validation_final_20260930"
# Legacy checkpoints pickle utils.LogField; resolve that name from this frozen scaffold.
sys.path.insert(0, str(ROOT / "publication_reconstruction/runtime"))


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


def main():
    RUNS.mkdir(exist_ok=False)
    logdir = RUNS / "launcher_logs"
    logdir.mkdir()
    cases = [(task, mode, version, 1) for task, mode, version in itertools.product(
        ["pp", "pcp", "fc"], ["real", "binary"], ["historical-2022", "corrected-v1"])]
    cases += [(task, mode, version, 2) for (task, mode), version in itertools.product(
        [("pp", "binary"), ("pcp", "real"), ("fc", "real")], ["historical-2022", "corrected-v1"])]
    records = []
    for task, mode, version, collectors in cases:
        name = f"{task}_{mode}_{version}_{collectors}collector"
        run = RUNS / name
        threshold = collectors * 4 + 1
        command = [sys.executable, "-m", "publication_reconstruction", "train", "--task", task,
            "--variant", mode, "--seed", "0", "--env-version", version,
            "--epochs", "3", "--updates-per-epoch", "3", "--batch-steps", "1", "--horizon", "4",
            "--max-env-steps", str(threshold), "--collectors", str(collectors), "--output", str(run)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
        (logdir / f"{name}.txt").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, f"{name}: {result.stderr[-1500:]} {result.stdout[-1500:]}"
        assert (run / "exit_code.txt").read_text().strip() == "0"
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        checkpoints = list((run / "checkpoints").rglob("*.pt"))
        assert len(checkpoints) == 1
        checkpoint = torch.load(checkpoints[0], map_location="cpu")
        counts = checkpoint["reconstruction"]["counts"]
        assert threshold <= counts["env_steps"] <= threshold + collectors * 4 - 1
        assert counts["epoch"] == metrics[-1]["epoch"] < 3
        assert counts["env_steps"] == metrics[-1]["total_steps"] == sum(m["steps"] for m in metrics)
        assert counts["episodes"] == metrics[-1]["total_episodes"]
        assert counts["updates"] >= 2 and finite(checkpoint)
        assert checkpoints[0].name == f"model_ep{counts['epoch']}.pt"
        manifest_bytes = (run / "source_manifest.json").read_bytes()
        assert checkpoint["reconstruction"]["source_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
        assert checkpoint["reconstruction"]["env_version"] == version
        manifest = json.loads(manifest_bytes)
        for name_in_archive, digest in manifest["files"].items():
            assert hashlib.sha256((run / "source" / name_in_archive).read_bytes()).hexdigest() == digest
        initial = json.loads((run / "initial_signature.json").read_text())
        last = json.loads((run / "epoch_signatures.jsonl").read_text().splitlines()[-1])["signature"]
        assert initial["sha256"] != last["sha256"]
        row = dict(task=task, variant=mode, version=version, collectors=collectors,
                   counts=counts, max_env_steps=threshold, finite_checkpoint=True,
                   source_hashes_verified=True, parameter_signature_changed=True,
                   checkpoint=str(checkpoints[0].relative_to(ROOT)),
                   checkpoint_sha256=hashlib.sha256(checkpoints[0].read_bytes()).hexdigest())
        records.append(row)
        (HERE / "smoke_runs.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
