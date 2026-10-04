"""Recompute the synced frozen PCP results without importing the learner.

Run from any directory with Python 3. Writes derived evidence beside this script;
does not change source reports, policies, or the earlier stdout-only analysis.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = ROOT / "stokes_runs/frozen_pcp_30m/frozen_pcp_30m_20261002_slurm"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b)


def metrics(episodes):
    return {
        "episodes": len(episodes),
        "successes": sum(e["success"] for e in episodes),
        "failures": sum(not e["success"] for e in episodes),
        "success_rate": mean(e["success"] for e in episodes),
        "mean_capped_steps": mean(e["steps"] for e in episodes),
        "mean_team_return": mean(e["team_return"] for e in episodes),
        "mean_agent_return": mean(e["team_return"] / e["num_agents"] for e in episodes),
    }


def main():
    manifest = json.loads((INPUT / "manifest.json").read_text())
    compositions = [tuple(c) for c in manifest["nominal_panel"]["compositions"]]
    inputs = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
               "sha256": digest(p)} for p in sorted(INPUT.rglob("*"))
              if p.is_file() and p.name != ".DS_Store"]
    stdout = json.loads((ROOT / "analysis/pcp_frozen_2026-10-03/summary.json").read_text())
    old = {(r["model"], r["training_seed"], r["condition"]): r for r in stdout["records"]}
    rows, reports = [], []
    for job in manifest["jobs"]:
        path = INPUT / "results" / Path(job["output"]).name
        report = json.loads(path.read_text())
        es = report["per_episode"]
        assert report["episodes"] == len(es)
        assert len({e["scenario_id"] for e in es}) == len(es)
        assert len(es) == (2500 if job["condition"] == "nominal" else 100)
        assert report["training_seed"] == job["training_seed"]
        assert report["config"]["model"] == job["model"]
        panel_path = INPUT / ("scenarios_nominal.json" if job["condition"] == "nominal" else "scenarios_failure.json")
        panel = json.loads(panel_path.read_text())
        assert report["scenarios"] == panel
        for e, s in zip(es, panel):
            for key, value in s.items():
                assert e[key] == value, (path.name, e["scenario_id"], key)
            assert e["composition"] == [s["num_p"], s["num_a"]]
            assert e["num_agents"] == len(e["agent_returns"]) == sum(e["composition"])
            close(e["team_return"], sum(e["agent_returns"]))
            close(e["mean_agent_return"], e["team_return"] / e["num_agents"])
            assert 1 <= e["steps"] <= report["config"]["max_steps"]
            # In this static PCP path, all unsuccessful trials reach the horizon.
            if not e["success"]:
                assert e["steps"] == 80
            assert all(math.isfinite(x) for x in e["agent_returns"])
        recomputed = metrics(es)
        prior = old[(job["model"], job["training_seed"], job["condition"])]
        for key in ("success_rate", "mean_team_return", "mean_agent_return"):
            close(recomputed[key], report[key])
            close(recomputed[key], prior[key])
        reports.append({"file": str(path.relative_to(ROOT)), "model": job["model"],
                        "seed": job["training_seed"], "condition": job["condition"],
                        **recomputed})
        if job["condition"] == "nominal":
            for comp in compositions:
                group = [e for e in es if tuple(e["composition"]) == comp]
                assert len(group) == 500
                rows.append({"model": job["model"], "seed": job["training_seed"],
                             "num_p": comp[0], "num_a": comp[1], **metrics(group)})
    groups = []
    keys = ("success_rate", "mean_capped_steps", "mean_team_return", "mean_agent_return")
    for model in ("shared", "banked"):
        for comp in compositions:
            seeds = [r for r in rows if (r["model"],r["num_p"],r["num_a"]) == (model,*comp)]
            assert len(seeds) == 3
            groups.append({"model": model, "composition": list(comp), "training_seeds": 3,
                           "episodes_per_seed": 500,
                           "metrics": {k:{"equal_seed_mean":mean(r[k] for r in seeds),
                                          "seed_min":min(r[k] for r in seeds),
                                          "seed_max":max(r[k] for r in seeds)} for k in keys}})
    # With one fixed horizon, raw rollout length equals horizon-capped completion.
    # This deliberately does not discard failures and average successful episodes only.
    summary = {"input_root":str(INPUT.relative_to(ROOT)), "reports":reports,
               "composition_seed_rows":rows, "composition_means":groups,
               "episodes":sum(r["episodes"] for r in reports),
               "limitations":["Three independent training seeds per model, common scenario panels reused across seeds/models.",
                              "Descriptive seed ranges are not confidence intervals.",
                              "No selected epoch1500 checkpoint tensors are present in this sync.",
                              "No native HetNet result is part of this frozen SoftRole tranche."],
               "checks":{"report_count":len(reports), "all_stdout_summaries_match":True,
                         "scenario_records_match_panels":True, "all_returns_reconcile":True,
                         "all_unsuccessful_episodes_reach_horizon_80":True}}
    assert len(reports) == 18 and summary["episodes"] == 16200
    (HERE/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (HERE/"input_manifest.json").write_text(json.dumps(inputs,indent=2)+"\n")
    with (HERE/"composition_seed_metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for record in inputs:
        assert digest(ROOT/record["path"]) == record["sha256"]
    print(json.dumps({"reports":len(reports),"episodes":summary["episodes"],
                      "composition_rows":len(rows),"hashed_inputs":len(inputs),"checks":summary["checks"]},indent=2))


if __name__ == "__main__":
    main()
