"""Bounded engineering validation of the CapCom proposal programme (12 October 2026).

Runs the exact launchers and CLIs a research run would use, with reduced budgets,
and records commands, exit codes, key outputs and SHA256 hashes. These are
engineering executions on this checkout, not research results. Usage:

    .venv/bin/python analysis/capcom_programme_2026-10-12/validate.py [--root DIR]

The output root must not exist. ``validation.json`` and ``artifact_manifest.json``
are written next to this script; ``--manifest-only ROOT`` rewrites only the manifest.
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
HERE = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
BOUNDED = ["--epochs", "1", "--updates-per-epoch", "1", "--batch-steps", "1", "--nprocesses", "2",
           "--total-steps", "1", "--save-every", "1", "--diagnostics-every", "1"]
# One entry per new mode: c1_real64, real_noisy, binary_deterministic, corrected null,
# decomposed beta 0, physical_only, banked composition, banked failure, failure mean_agent.
ENTRIES = (0, 3, 9, 21, 30, 33, 36, 48, 60)
S1_RATES = (3e-4, 1e-3)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(record, name, command, env=None, expect=0, cwd=ROOT):
    environment = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'envs'}", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", HETNET_PYTHON=PYTHON)
    environment.update(env or {})
    started = time.monotonic()
    result = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True)
    entry = {"name": name, "command": command, "returncode": result.returncode,
             "seconds": round(time.monotonic() - started, 3),
             "stdout_head": result.stdout[:2000], "stderr_tail": result.stderr[-1500:]}
    record["commands"].append(entry)
    if result.returncode != expect:
        raise SystemExit(f"{name}: exit {result.returncode}, expected {expect}\n{result.stderr[-3000:]}")
    return result


def synthetic_s1_ledgers(root, completions):
    """S1-shaped ledgers for the rule check (window endpoints every 500k steps)."""
    sys.path.insert(0, str(ROOT))
    from softrole.programme import normalized, programme_config
    runs = []
    for lr, values in zip(S1_RATES, completions):
        for seed, completion in enumerate(values):
            run_dir = root / f"lr{lr:g}" / f"seed{seed}"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps(normalized(programme_config("c1_real64", seed, lr))))
            with (run_dir / "updates.jsonl").open("w") as updates, (run_dir / "episodes.jsonl").open("w") as episodes:
                for update in range(1, 26):
                    updates.write(json.dumps({"update": update, "total_steps": update * 500_000, "episodes": 1,
                                              "steps": 500_000, "gradient_norm_before_clip": 1.}) + "\n")
                    episodes.write(json.dumps({"update": update, "collector": 0, "steps": completion, "success": True,
                                               "team_return": -.15 * completion, "num_agents": 3,
                                               "mean_agent_return": -.05 * completion, "event_exposed": False}) + "\n")
            runs.append(run_dir)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "runs" / "capcom_programme_validation_20261012"))
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    record = {"date": "2026-10-12", "checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
              "python": sys.version, "root": str(root), "commands": [], "scope":
              "Bounded engineering executions with reduced budgets; synthetic S1 ledgers exercise only the "
              "selection rule; no research outcome is produced."}
    logs = root / "logs"
    logs.mkdir()

    run(record, "arms", [PYTHON, "-m", "softrole.programme", "arms"])
    (root / "arms.json").write_text(record["commands"][-1]["stdout_head"] if False else
                                    subprocess.check_output([PYTHON, "-m", "softrole.programme", "arms"], cwd=ROOT, text=True))

    # Six tiny real S1-shaped runs through the S1 launcher: the research rule must refuse them.
    s1_root = root / "s1_bounded"
    for index in range(6):
        run(record, f"s1_bounded_{index}", ["bash", "scripts/softrole_lr_pcp.sh", str(index), "--epochs", "1",
            "--updates-per-epoch", "1", "--batch-steps", "1", "--nprocesses", "1", "--total-steps", "1",
            "--save-every", "1", "--diagnostics-every", "1"], env={"SOFTROLE_LR_PCP_RUN_ROOT": str(s1_root)})
    s1_runs = sorted(str(p) for p in s1_root.glob("pcp_shared/real/*/seed*"))
    run(record, "select_lr_refuses_partial_real_runs", [PYTHON, "-m", "softrole.programme", "select-lr", "--runs",
        *s1_runs, "--output", str(root / "lr_selection_refused.json")], expect=2)
    assert not (root / "lr_selection_refused.json").exists()
    synthetic = synthetic_s1_ledgers(root / "s1_synthetic", ((6, 6, 6), (5, 5, 8)))
    run(record, "select_lr_synthetic_rule", [PYTHON, "-m", "softrole.programme", "select-lr", "--runs",
        *map(str, synthetic), "--output", str(root / "lr_selection_synthetic.json"), "--bounded"])
    selection = json.loads((root / "lr_selection_synthetic.json").read_text())
    assert selection["selected_lr"] == 3e-4 and selection["bounded_check"]
    run(record, "prepare_manifest_bounded", [PYTHON, "-m", "softrole.programme", "prepare", "--lr-selection",
        str(root / "lr_selection_synthetic.json"), "--output", str(root / "manifest.json"), "--bounded"])
    run(record, "prepare_manifest_refuses_research_from_bounded", [PYTHON, "-m", "softrole.programme", "prepare",
        "--lr-selection", str(root / "lr_selection_synthetic.json"), "--output", str(root / "manifest_research.json")], expect=2)
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["entries"]) == 63 and manifest["bounded_check"]

    programme_env = {"SOFTROLE_PROGRAMME_MANIFEST": str(root / "manifest.json"),
                     "SOFTROLE_PROGRAMME_RUN_ROOT": str(root / "programme")}
    for index in ENTRIES:
        run(record, f"launch_dry_run_{index}", ["bash", "scripts/softrole_programme.sh", str(index), "--dry-run"], env=programme_env)
        run(record, f"launch_bounded_{index}", ["bash", "scripts/softrole_programme.sh", str(index), *BOUNDED], env=programme_env)
    run(record, "launch_rejects_scientific_override", ["bash", "scripts/softrole_programme.sh", "0", "--communication", "binary"],
        env=programme_env, expect=2)
    run(record, "launch_rejects_existing_run", ["bash", "scripts/softrole_programme.sh", "0", *BOUNDED], env=programme_env, expect=2)
    runs = []
    for index in ENTRIES:
        entry = manifest["entries"][index]
        run_dir = root / "programme" / entry["relative_output"]
        checkpoint = run_dir / "checkpoints" / "epoch0001.pt"
        finished = json.loads((run_dir / "finished.json").read_text())
        metrics = json.loads((run_dir / "metrics.jsonl").read_text())
        provenance = json.loads((run_dir.parent / (run_dir.name + ".programme.json")).read_text())
        runs.append({"index": index, "arm": entry["arm"], "seed": entry["seed"], "run": str(run_dir),
                     "learner": {k: entry["config"][k] for k in ("communication", "null_degree_correction", "critic_mode",
                                                                  "credit_beta", "objective", "actor_capabilities", "model")},
                     "steps": finished["total_steps"], "episodes": finished["total_episodes"], "updates": finished["updates"],
                     "event_exposed_episodes": metrics["event_exposed_episodes"], "success_rate": metrics["success_rate"],
                     "checkpoint_sha256": sha(checkpoint), "source_sha256": json.loads((run_dir / "run.json").read_text())["source_sha256"],
                     "bounded_check": provenance["bounded_check"], "diagnostics_rows": sum(1 for _ in (run_dir / "diagnostics.jsonl").open())})
    record["programme_runs"] = runs
    assert len({r["source_sha256"] for r in runs}) == 1 and runs[0]["source_sha256"] == manifest["source_sha256"]

    # Diagnostics protocol version 2 on two bounded checkpoints (terminal: checkpoint-probe mode).
    probe_checkpoints = [r["run"] + "/checkpoints/epoch0001.pt" for r in runs if r["index"] in (30, 48)]
    run(record, "diagnostics_prepare_v2", [PYTHON, "-m", "softrole.diagnostics", "prepare", "--checkpoints", *probe_checkpoints,
        "--checkpoint-rule", "bounded validation: terminal epoch 1 checkpoints of entries 30 and 48",
        "--batch-mode", "checkpoint-probe", "--probe-episodes", "1", "--d5-instances", "1", "--d5-bits", "2",
        "--d5-samples", "32", "--protocol-version", "2", "--output", str(root / "diagnostics" / "manifest.json")])
    run(record, "diagnostics_prepare_v1_refuses_programme_learner", [PYTHON, "-m", "softrole.diagnostics", "prepare",
        "--checkpoints", probe_checkpoints[0], "--checkpoint-rule", "x", "--batch-mode", "checkpoint-probe",
        "--output", str(root / "diagnostics" / "manifest_v1.json")], expect=1)
    diagnostics = []
    for index in range(2):
        run(record, f"diagnostics_dry_run_{index}", ["bash", "scripts/softrole_diagnostics.sh", "--manifest",
            str(root / "diagnostics" / "manifest.json"), "--index", str(index), "--output", str(root / "diagnostics" / f"dry_{index}"), "--dry-run"])
        run(record, f"diagnostics_execute_{index}", ["bash", "scripts/softrole_diagnostics.sh", "--manifest",
            str(root / "diagnostics" / "manifest.json"), "--index", str(index), "--output", str(root / "diagnostics" / f"task_{index}")])
        report = json.loads((root / "diagnostics" / f"task_{index}" / "report.json").read_text())
        diagnostics.append({"index": index, "checkpoint": report["plan"]["checkpoint"], "protocol_version": report["protocol_version"],
                            "learner": report["learner"], "training_study": report["training_study"],
                            "batch_mode": report["batch_mode"], "episodes": len(report["episodes"]),
                            "episode_compositions": report["episode_compositions"],
                            "d1_norm": report["D1"]["gradient_norm_before_clip"], "d1_clip_active": report["D1"]["clip_active"],
                            "d3_learned": report["D3_learned"] is not None, "d4_production_variant": report["D4"]["protocol"]["production_variant"],
                            "d4_variants": sorted(report["D4"]["variants"]), "d5_status": report["D5"]["status"],
                            "d6_strata": len(report["D6"]["strata"]), "d6_training_support": report["D6"]["training_support"],
                            "report_sha256": sha(root / "diagnostics" / f"task_{index}" / "report.json")})
    record["diagnostics"] = diagnostics

    run(record, "evaluation_prepare_bounded", [PYTHON, "-m", "softrole.programme_evaluation", "prepare", "--manifest",
        str(root / "manifest.json"), "--run-root", str(root / "programme"), "--output", str(root / "evaluation"),
        "--episodes", "2", "--target-steps", "1", "--bounded", "--entries", *map(str, ENTRIES)])
    run(record, "evaluation_prepare_refuses_research", [PYTHON, "-m", "softrole.programme_evaluation", "prepare", "--manifest",
        str(root / "manifest.json"), "--run-root", str(root / "programme"), "--output", str(root / "evaluation_research")], expect=2)
    evaluation = json.loads((root / "evaluation" / "evaluation_manifest.json").read_text())
    for index in range(evaluation["tasks"]):
        run(record, f"evaluation_task_{index}", [PYTHON, "-m", "softrole.programme_evaluation", "run", "--tasks",
            str(root / "evaluation" / "tasks.json"), "--index", str(index)])
    run(record, "evaluation_task_rerun_skips", [PYTHON, "-m", "softrole.programme_evaluation", "run", "--tasks",
        str(root / "evaluation" / "tasks.json"), "--index", "0"])
    run(record, "evaluation_summarize", [PYTHON, "-m", "softrole.programme_evaluation", "summarize", "--evaluation",
        str(root / "evaluation"), "--output", str(root / "summary"), "--bootstrap-samples", "200"])
    overview = json.loads((root / "summary" / "overview.json").read_text())
    reports = sorted((root / "evaluation" / "reports").rglob("*.json"))
    record["evaluation"] = {"tasks": evaluation["tasks"], "jobs": evaluation["jobs"], "reports": len(reports),
                            "episodes": sum(json.loads(p.read_text())["episodes"] for p in reports),
                            "compositions": sorted(overview["compositions"]),
                            "strata_3_2": len(overview["compositions"]["3_2"]["strata"]),
                            "paired_3_2": len(overview["compositions"]["3_2"]["failure_minus_sham"]),
                            "did_3_2": len(overview["compositions"]["3_2"]["difference_in_differences"]),
                            "contrasts_3_2": {k: len(v) for k, v in overview["compositions"]["3_2"]["contrasts"].items()},
                            "arms_3_2": sorted({row["arm"] for row in overview["compositions"]["3_2"]["strata"]}),
                            "evaluation_manifest_sha256": sha(root / "evaluation" / "evaluation_manifest.json"),
                            "overview_sha256": sha(root / "summary" / "overview.json"),
                            "evaluate_sbatch_sha256": sha(root / "evaluation" / "evaluate.sbatch")}
    run(record, "evaluate_sbatch_syntax", ["bash", "-n", str(root / "evaluation" / "evaluate.sbatch")])
    record["principal_hashes"] = {"manifest.json": sha(root / "manifest.json"),
                                  "lr_selection_synthetic.json": sha(root / "lr_selection_synthetic.json"),
                                  "diagnostics/manifest.json": sha(root / "diagnostics" / "manifest.json")}
    record["totals"] = {"commands": len(record["commands"]),
                        "training_steps": sum(r["steps"] for r in runs) + sum(
                            json.loads((p / "finished.json").read_text())["total_steps"] for p in map(Path, s1_runs)),
                        "programme_training_episodes": sum(r["episodes"] for r in runs),
                        "evaluation_episodes": record["evaluation"]["episodes"]}
    (HERE / "validation.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    manifest_rows = write_manifest(root)
    print(json.dumps({"commands": len(record["commands"]), "programme_runs": len(runs), "diagnostics": len(diagnostics),
                      "evaluation": record["evaluation"]["reports"], "artifact_files": len(manifest_rows)}))


def write_manifest(root):
    """Hash every artifact except archived source copies, which each run's own
    source_manifest.json already fixes, and transient bytecode."""
    root = Path(root)
    rows = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and "source" not in p.parts}
    (HERE / "artifact_manifest.json").write_text(json.dumps(
        {"root": str(root), "files": len(rows),
         "excluded": "archived source/ directories (covered by each run's source_manifest.json) and bytecode",
         "sha256": rows}, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--manifest-only":
        print(json.dumps({"artifact_files": len(write_manifest(sys.argv[2]))}))
    else:
        main()
