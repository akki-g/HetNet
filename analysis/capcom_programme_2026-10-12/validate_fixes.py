"""Reproducible bounded validation of the 13 October evidence/recovery fixes.

Run only after source/tests/work-log edits are complete. All outputs go into a
fresh --output directory, never the repository's historical evidence files.
No scheduler, push or research-budget run is invoked. Requires the repository's
pinned Python environment and an unchanged baseline checkout at 644f8f1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BOUNDED = ["--epochs", "1", "--updates-per-epoch", "1", "--batch-steps", "1",
           "--nprocesses", "1", "--total-steps", "1", "--save-every", "1"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


def stripped(value):
    if isinstance(value, dict):
        return {k: stripped(v) for k, v in value.items() if k != "wall_time_seconds"}
    if isinstance(value, list):
        return [stripped(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    python = sys.executable
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.baseline).decode().strip().startswith("644f8f1"), "baseline must be unchanged 644f8f1"
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.baseline).strip(), "baseline tracked files must be clean"
    record = {"validator_sha256": digest(__file__), "scope": "bounded engineering checks only; no research outcome", "commands": [],
              "python": sys.version, "root": str(destination), "baseline": str(args.baseline.resolve())}

    def run(name, command, cwd=ROOT, extra=None, expect=0):
        env = dict(os.environ, PYTHONPATH=f"{cwd}:{cwd / 'envs'}", HETNET_PYTHON=python,
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        env.update(extra or {})
        started = time.monotonic()
        with (destination / "logs" / f"{name}.stdout").open("w") as out, (destination / "logs" / f"{name}.stderr").open("w") as err:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=out, stderr=err)
        row = dict(name=name, command=command, cwd=str(cwd), returncode=result.returncode,
                   seconds=time.monotonic() - started)
        record["commands"].append(row)
        (destination / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if result.returncode != expect:
            raise RuntimeError(f"{name} returned {result.returncode}, expected {expect}: see logs")

    # Two updates of the legacy, S1 Real and channels Binary production recipes.
    runtime = ["--epochs", "2", "--updates-per-epoch", "1", "--batch-steps", "1", "--nprocesses", "1", "--save-every", "1"]
    replay = {}
    for label, checkout in (("baseline", args.baseline.resolve()), ("fixed", ROOT)):
        base = destination / "replay" / label
        run(f"{label}_legacy", [python, "-m", "softrole", "train", "--output", str(base / "legacy"), *runtime], cwd=checkout)
        run(f"{label}_s1", ["bash", "scripts/softrole_lr_pcp.sh", "0", *runtime], cwd=checkout,
            extra={"SOFTROLE_LR_PCP_RUN_ROOT": str(base / "s1")})
        run(f"{label}_binary", ["bash", "scripts/softrole_channels_pcp.sh", "0", *runtime], cwd=checkout,
            extra={"SOFTROLE_CHANNELS_PCP_RUN_ROOT": str(base / "binary")})
        replay[label] = dict(legacy=base / "legacy", s1=base / "s1/pcp_shared/real/adam_lr3e-4/seed0",
                             binary=base / "binary/pcp_shared/binary/seed0")
    import torch
    for family in replay["baseline"]:
        left, right = replay["baseline"][family], replay["fixed"][family]
        a = torch.load(left / "checkpoints/epoch0002.pt", weights_only=False)
        b = torch.load(right / "checkpoints/epoch0002.pt", weights_only=False)
        for key in ("model_state", "optimizer_state", "updates", "total_steps", "total_episodes"):
            same(a[key], b[key])
        assert sorted(p.name for p in left.glob("*.jsonl")) == sorted(p.name for p in right.glob("*.jsonl"))
        for path in left.glob("*.jsonl"):
            same(stripped([json.loads(x) for x in path.read_text().splitlines()]),
                 stripped([json.loads(x) for x in (right / path.name).read_text().splitlines()]))
    record["default_replay"] = {family: "bitwise model, optimizer, counts and every ledger except wall time" for family in replay["baseline"]}

    # Real, internally reconciled six-run bounded evidence, not the old impossible fixtures.
    s1root = destination / "s1"
    for index in range(6):
        run(f"s1_{index}", ["bash", "scripts/softrole_lr_pcp.sh", str(index), *runtime],
            extra={"SOFTROLE_LR_PCP_RUN_ROOT": str(s1root)})
    s1runs = sorted(str(p) for p in s1root.glob("pcp_shared/real/*/seed*"))
    selection, manifest = destination / "selection.json", destination / "manifest.json"
    run("select_lr", [python, "-m", "softrole.programme", "select-lr", "--runs", *s1runs,
                       "--window", "0", "80", "--bounded", "--output", str(selection)])
    run("prepare", [python, "-m", "softrole.programme", "prepare", "--lr-selection", str(selection),
                    "--output", str(manifest), "--bounded"])
    env = {"SOFTROLE_PROGRAMME_MANIFEST": str(manifest), "SOFTROLE_PROGRAMME_RUN_ROOT": str(destination / "runs")}
    for index in range(0, 63, 3):
        run(f"arm_{index}", ["bash", "scripts/softrole_programme.sh", str(index), *BOUNDED], extra=env)
    run("stdout_rejected", ["bash", "scripts/softrole_programme.sh", "1", "--episode-log", "stdout"], extra=env, expect=2)
    run("completed_resume", ["bash", "scripts/softrole_programme.sh", "0", "--resume"], extra=env)
    from softrole.ledger import iter_episodes
    from softrole.programme import read_json, entries
    episodes, gzip_bytes, raw_bytes = 0, 0, 0
    sample = []
    for index in range(0, 63, 3):
        run_dir = destination / "runs" / entries(read_json(manifest)["lr"])[index]["relative_output"]
        rows = list(iter_episodes(run_dir))
        episodes += len(rows)
        sample.extend(rows)
        raw_bytes += sum(len((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()) for row in rows)
        gzip_bytes += sum(p.stat().st_size for p in run_dir.rglob("*.gz"))
    record["storage"] = {"sample": "actual 21-arm one-update ledgers, one collector", "episodes": episodes,
                         "raw_jsonl_bytes": raw_bytes, "gzip_bytes": gzip_bytes,
                         "raw_to_gzip_ratio": raw_bytes / gzip_bytes,
                         "mean_raw_bytes_per_episode": raw_bytes / episodes,
                         "mean_gzip_bytes_per_episode": gzip_bytes / episodes,
                         "projection_1_26B_steps": {str(h): {"episodes": 1_260_000_000 / h,
                              "raw_GB": 1.26 * raw_bytes / episodes / h,
                              "gzip_GB": 1.26 * gzip_bytes / episodes / h} for h in (5, 7, 20, 80)},
                         "limitations": "Small terminal chunks; excludes source/checkpoints/index overhead. Episode length and compression change during learning."}
    # Measure identical writer workloads; this does not extrapolate training speed.
    from softrole.ledger import ChunkedEpisodes
    from softrole.train import log_episode_batch
    timings = {"file": [], "chunked-gzip": []}
    sizes = {}
    for repeat in range(3):
        for mode in (("file", "chunked-gzip") if repeat % 2 == 0 else ("chunked-gzip", "file")):
            bench = destination / "logging_benchmark" / f"{mode}_{repeat}"
            bench.mkdir(parents=True)
            writer = ChunkedEpisodes(bench, horizon=80) if mode == "chunked-gzip" else None
            started = time.monotonic()
            for update in range(1, 301):
                batch = [dict(row, update=update, scenario_id=f"{update}_{i}_{row['scenario_id']}")
                         for i, row in enumerate(sample)]
                if writer:
                    writer.append(batch, update)
                else:
                    log_episode_batch(bench / "episodes.jsonl", batch, update, mode)
            if writer:
                writer.seal()
                writer.close()
            timings[mode].append(time.monotonic() - started)
            sizes[mode] = sum(p.stat().st_size for p in bench.rglob("*") if p.is_file())
    record["logging_benchmark"] = {"updates": 300, "episodes": len(sample) * 300,
        "seconds_per_repeat": timings, "bytes_including_index": sizes,
        "workload": "actual bounded episode metrics repeated with distinct update/scenario identities; 100-update chunk seals",
        "limitations": "Writer-only wall time on this filesystem, no durability-equivalent speed claim; does not measure cluster I/O or full training overhead."}
    indices = list(map(str, range(0, 63, 3)))
    record["evaluations"] = {}
    for phase in ("development", "final"):
        output = destination / phase
        run(f"prepare_{phase}", [python, "-m", "softrole.programme_evaluation", "prepare", "--manifest", str(manifest),
             "--run-root", str(destination / "runs"), "--output", str(output), "--phase", phase,
             "--bounded", "--target-steps", "1", "--episodes", "1", "--entries", *indices])
        evaluation = read_json(output / "evaluation_manifest.json")
        for index in range(evaluation["tasks"]):
            run(f"{phase}_task_{index}", [python, "-m", "softrole.programme_evaluation", "run",
                 "--tasks", str(output / "tasks.json"), "--index", str(index)])
        run(f"{phase}_reuse", [python, "-m", "softrole.programme_evaluation", "run",
             "--tasks", str(output / "tasks.json"), "--index", "0"])
        run(f"{phase}_summary", [python, "-m", "softrole.programme_evaluation", "summarize", "--evaluation", str(output),
             "--output", str(destination / (phase + "_summary")), "--bootstrap-samples", "50"])
        record["evaluations"][phase] = {"tasks": evaluation["tasks"], "reports": evaluation["jobs"],
                                         "manifest_sha256": digest(output / "evaluation_manifest.json")}
    record["source_sha256"] = read_json(manifest)["source_sha256"]
    (destination / "validation.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    hashes = {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*"))
              if p.is_file() and "source" not in p.relative_to(destination).parts and "__pycache__" not in p.parts}
    (destination / "artifact_manifest.json").write_text(json.dumps({"sha256": hashes, "excluded": "archived source copies covered by their source_manifest.json"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
