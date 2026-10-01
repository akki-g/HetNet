"""Validate saved audit evidence, unchanged inputs and deliverable identities."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((HERE / path).read_text())


def main():
    inventory = read("training/inventory.json")
    probes = read("softrole/probe_results.json")
    observation = read("softrole/observation_results.json")
    legacy = read("legacy/feature_findings.json")
    fc = read("fc/findings.json")
    for path, record in inventory["inputs"].items():
        assert sha(ROOT / path) == record["sha256"]
        assert (ROOT / path).stat().st_size == record["bytes"]
    conditions = probes["conditions"]
    assert len(inventory["runs"]) == 27
    assert sum(r["completed_epochs"] for r in inventory["runs"]) == 37566
    assert len(conditions) == 12
    assert sum(len(c["episodes"]) for c in conditions) == 144
    checkpoints = set()
    for condition in conditions:
        assert condition["independent_replay_passed"]
        assert condition["successes"] == sum(e["success"] for e in condition["episodes"])
        assert abs(condition["mean_steps"] - sum(e["steps"] for e in condition["episodes"]) / 12) < 1e-12
        if condition["trained"]:
            path = ROOT / condition["checkpoint"]
            assert sha(path) == condition["checkpoint_sha256"]
            checkpoints.add(str(path))
    assert len(checkpoints) == 6
    for path, expected in probes["current_source_sha256"].items():
        assert sha(ROOT / path) == expected
    archive_count = 0
    for run, files in probes["archived_source"].items():
        manifest = json.loads((ROOT / run / "source_manifest.json").read_text())["files"]
        for path, record in files.items():
            archived = ROOT / run / "source" / path
            assert sha(archived) == manifest[path]
            assert record["matches_manifest"]
            assert (sha(archived) == sha(ROOT / path)) == record["matches_current"]
            archive_count += 1
    assert archive_count == 144
    assert sha(ROOT / "hetgat/uavnet.py") == legacy["uavnet_sha256"]
    assert sha(HERE / "legacy/upstream_uavnet.py") == legacy["uavnet_sha256"]
    assert [r["relative_cells_whose_target_channel_reaches_model"] for r in legacy["domains"]] == [[0, 6, 12, 19], [0, 6, 12, 19], [0, 6]]
    assert fc["new_phantom_origin_fires_of_25_initial_cells"] == 8
    assert fc["phantom_source_capture"]["team_reward"] == 9.8
    assert sum(r["identical_physics_reward_transitions"] for r in observation.values()) == 10992
    assert observation["pcp"]["visible_target_erased"] == 688
    assert observation["fc"]["visible_target_erased"] == 123
    assert sha(HERE / "legacy/probe_features.py") == legacy["probe_sha256"]
    assert sha(HERE / "fc/confirm_fc.py") == fc["probe_sha256"]
    assert sha(HERE / "softrole/probe.py") == probes["probe_sha256"]
    for path in HERE.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    for name in ["paper/hetnet_aamas2022.pdf", "paper/supplement.pdf"]:
        assert (HERE / name).read_bytes().startswith(b"%PDF")
    result = {"full_existing_suite": {"command": ".venv/bin/python -m pytest -q",
                                     "passed": 178, "seconds": 49.14},
              "note": "Suite result recorded from this turn; validator rechecks saved evidence and bytes, not policy executions.",
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runs": 27, "completed_epochs": 37566,
              "archived_source_files_rechecked": archive_count,
              "unchanged_checkpoint_files": len(checkpoints),
              "replayed_policy_episodes": 144,
              "policy_transitions": sum(e["steps"] for c in conditions for e in c["episodes"]),
              "paired_observation_transitions": 10992,
              "python_syntax": "passed", "git_diff_check": "passed",
              "input_hashes_rechecked": len(inventory["inputs"]),
              "AGENTS_sha256": sha(ROOT / "AGENTS.md")}
    with (HERE / "validation.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    files = {str(p.relative_to(HERE)): {"bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(HERE.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.name != "artifact_manifest.json"}
    with (HERE / "artifact_manifest.json").open("x") as stream:
        json.dump({"files": files, "excludes": ["artifact_manifest.json", "__pycache__/"]}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"validation": "passed", "deliverables_hashed": len(files),
                      "policy_episodes": 144, "source_files": archive_count,
                      "checkpoints": len(checkpoints)}, indent=2))


if __name__ == "__main__":
    main()
