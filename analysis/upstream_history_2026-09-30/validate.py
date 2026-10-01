"""Validate historical audit provenance/results and write a deliverable manifest.

Run after all report edits. This reads previously executed bounded probes; it
does not rerun training, query live jobs or certify the unknown original runs.
"""
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MIRROR = HERE / "history.git"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def git(*args):
    return subprocess.check_output(["git", f"--git-dir={MIRROR}", *args])


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2) + "\n")


def main():
    history = read("history.json")
    commits = history["commits"]
    assert len(commits) == 26 and sum(c["on_main"] for c in commits) == 18
    assert len(history["refs"]) == 5
    actual = set(git("rev-list", "--all").decode().splitlines())
    assert actual == {c["sha"] for c in commits}
    preconference = [c for c in commits if c["on_main"] and
                     datetime.fromisoformat(c["committer_date"]).date().isoformat() < "2022-05-09"]
    latest = max(preconference, key=lambda c: datetime.fromisoformat(c["committer_date"]))
    assert latest["sha"].startswith("d57da07")
    for name, metadata in history["api_fetches"].items():
        assert sha((HERE / "public_api" / f"{name}.json").read_bytes()) == metadata["sha256"]
        assert metadata["pagination"] is None
    assert read("public_api/tags.json") == []
    assert read("public_api/releases.json") == []

    snapshot_count = 0
    for short, row in read("snapshot_manifest.json").items():
        for name, expected in row["files"].items():
            payload = (HERE / "snapshots" / short / name).read_bytes()
            assert sha(payload) == expected
            assert payload == git("show", f"{row['sha']}:{name}")
            snapshot_count += 1
    training_count = 0
    for row in read("training/source_manifest.json"):
        payload = (HERE / "training" / row["local_path"]).read_bytes()
        assert sha(payload) == row["sha256"]
        assert payload == git("show", f"{row['commit']}:{row['path']}")
        training_count += 1
    for name, expected in read("env_eval/manifest.json").items():
        assert sha((HERE / "env_eval" / name).read_bytes()) == expected, name

    bugs = read("bugs/findings.json")
    assert sha((HERE / "bugs/probe_history.py").read_bytes()) == bugs["script_sha256"]
    for digest, row in bugs["unique_blobs"].items():
        payload = (HERE / "bugs/source_blobs" / f"{digest}.py").read_bytes()
        assert sha(payload) == digest
        assert payload == git("show", f"{row['first_encounter_commit']}:{row['path']}")
    sensors = {}
    for digest, probe in bugs["probes_by_blob"].items():
        if probe["probe"] == "sensory_extraction":
            first = bugs["unique_blobs"][digest]["first_encounter_commit"][:7]
            sensors[first] = {r["task"]: len(r["target_cells_reaching_model"]) for r in probe["rows"]}
    assert sensors == {"3ca462b": {"pp": 25, "pcp": 25, "fc": 9},
                       "6b09d36": {"pp": 4, "pcp": 4, "fc": 2},
                       "f54ca5a": {"pp": 4, "pcp": 4, "fc": 2}}
    pairs = {r["commit"][:7]: r["results"] for r in bugs["fc_pairs"].values()}
    assert pairs["59d5983"]["reset"]["exception"] == "ValueError"
    assert pairs["bff9f7f"]["reset"]["status"] == "ok"
    for commit in ["59d5983", "bff9f7f"]:
        physical = pairs[commit]["manually_initialized_phantom_probe"]
        assert physical["rewards"] == [-.1, -.1, 10.]
        assert physical["residual_fires"] == [[0, 2]] and physical["done"] is False
    env = read("env_eval/historical_probe.json")
    assert env["unmodified_fc_reset"]["type"] == "ValueError"
    assert env["phantom_fire"]["new_origin_fires_of_25_initial_cells"] == 8

    outcomes = Counter()
    failures = []
    for short in ["d57da07", "6b09d36", "f54ca5a"]:
        probe = read(f"training/{short}_model_probe.json")
        cases = probe["models"]
        assert len(cases) == 6
        for case in cases:
            outcomes[case["status"]] += 1
            if case["status"] == "pass":
                assert case["finite_gradients"] and case["weights_unchanged"]
                assert case["policy_optimizer_steps"] == 0
                assert abs(case["gradient_norm_after_policy"] - .75) < 1e-6
            else:
                failures.append({"commit": short, "task": case["task"], "mode": case["mode"],
                                 "exception": case["exception"], "message": case["message"]})
        assert probe["trainer_instrumented_events"][-1]["gradient_at_step"] == .001
    assert outcomes == {"pass": 14, "error": 4}
    assert [(f["commit"], f["task"], f["mode"]) for f in failures] == [
        ("d57da07", "pp", "real"), ("d57da07", "pp", "binary"),
        ("6b09d36", "pp", "binary"), ("f54ca5a", "pp", "binary")]

    supplement = (HERE / "papers/supplement_jan2022.pdf").read_bytes()
    assert supplement == git("show", "8b28c6f:AAMAS_22___HetNet_Supplementary.pdf")
    assert supplement == (ROOT / "analysis/results_audit_2026-09-30/paper/supplement.pdf").read_bytes()
    search = read("artifact_search.json")
    assert len(search["trees"]) == 26
    old_artifacts = [r for r in search["trees"] if r["author_date"][:4] == "2022" and r["candidate_artifacts"]]
    assert not old_artifacts
    budgets = {}
    for task, epochs, horizon in [("pp", 2000, 80), ("pcp", 2000, 80), ("fc", 1400, 300)]:
        budgets[task] = {"october_min_steps": epochs * 10 * 4 * 500,
                         "october_max_steps": epochs * 10 * 4 * (500 + horizon - 1),
                         "epochs": epochs, "episode_horizon": horizon}
    assert budgets["pp"]["october_min_steps"] == 40_000_000
    assert budgets["pp"]["october_max_steps"] == 46_320_000
    assert budgets["fc"]["october_min_steps"] == 28_000_000
    assert budgets["fc"]["october_max_steps"] == 44_744_000

    authored = ["collect_history.py", "collect_extra_evidence.py", "validate.py",
                "bugs/probe_history.py", "training/export_sources.py", "training/probe_model.py",
                "env_eval/probe_historical.py"]
    for name in authored:
        ast.parse((HERE / name).read_text(), filename=name)
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    save("validation.json", {"status": "pass", "scope": "Source identities and recorded bounded probes; no training",
          "reachable_commits": 26, "main_commits": 18, "branch_refs": 5,
          "last_preconference_commit": latest["sha"],
          "root_snapshot_files_verified_against_git": snapshot_count,
          "training_source_files_verified_against_git": training_count,
          "env_eval_manifest_files_verified": len(read("env_eval/manifest.json")),
          "distinct_bug_source_blobs_verified": len(bugs["unique_blobs"]),
          "sensory_target_retention": sensors, "model_probe_outcomes": dict(outcomes),
          "documented_model_failures": failures, "recipe_budget_bounds": budgets,
          "june_pp_pcp_min_steps": 2000 * 10 * 1 * 500,
          "supplement_sha256": sha(supplement), "authored_python_syntax_checked": authored,
          "git_diff_check": "passed", "historical_dependencies_reconstructed": False,
          "training_commit_identified": False, "runtime_suite_rerun": False})
    files = {}
    for path in sorted(HERE.rglob("*")):
        relative = path.relative_to(HERE)
        if not path.is_file() or any(part in {"history.git", "__pycache__"} for part in relative.parts):
            continue
        if relative.as_posix() == "artifact_manifest.json":
            continue
        files[relative.as_posix()] = sha(path.read_bytes())
    save("artifact_manifest.json", {"scope": "All deliverables except this manifest, bare mirror and Python caches", "files": files})
    print(json.dumps({"validation": "pass", "manifest_files": len(files), "model_probes": dict(outcomes),
                      "root_snapshot_files": snapshot_count, "training_snapshot_files": training_count}))


if __name__ == "__main__":
    main()
