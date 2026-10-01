"""Read-only FC log/config/source audit; uses only Python's standard library.

Run: .venv/bin/python analysis/fc_audit_2026-09-30/log_probe.py --output /tmp/fc_log_findings.json
Defaults to log_findings.json beside this script; refuses existing destinations.
No policies are executed.
"""
import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
INPUTS = {}


def read(path):
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = {
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
    }
    return data.decode()


def load(path):
    return json.loads(read(path))


def lines(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def window(rows, weighted=True):
    weights = [r["episodes"] if weighted else 1 for r in rows]
    total = sum(weights)
    result = {"first_epoch": rows[0]["epoch"], "last_epoch": rows[-1]["epoch"],
              "weighting": "episodes" if weighted else "equal epochs"}
    for key in ("success_rate", "team_return", "steps_taken", "return_nocap", "return_cap",
                "value_loss_per_episode", "policy_loss_per_episode", "gate_entropy", "alpha_null",
                "average_enemy_count"):
        if all(key in r for r in rows):
            result[key] = math.fsum(r[key] * w for r, w in zip(rows, weights)) / total
    if weighted:
        result.update(episodes=total,
                      steps_start=rows[0]["total_steps"] - rows[0]["steps"],
                      steps_end=rows[-1]["total_steps"],
                      successes=round(math.fsum(r["success_rate"] * r["episodes"] for r in rows)))
    return result


def config_differences(configs):
    return {key: [c.get(key) for c in configs]
            for key in sorted(set().union(*configs))
            if len({json.dumps(c.get(key), sort_keys=True) for c in configs}) > 1}


def reproduction(path):
    config, rows, pending = None, [], None
    for line in read(path).splitlines():
        if line.startswith("Namespace("):
            node = ast.parse(line, mode="eval").body
            config = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        match = re.match(r"^Epoch (\d+)\s+Reward \[([^]]+)\]", line)
        if match:
            pending = {"epoch": int(match[1]),
                       "team_return": sum(float(x) for x in match[2].split())}
        if pending is not None:
            for prefix, key in (("Success: ", "success_rate"), ("Steps-taken: ", "steps_taken"),
                                ("Average-Enemy-Count: ", "average_enemy_count")):
                if line.startswith(prefix):
                    pending[key] = float(line[len(prefix):])
                    if key == "average_enemy_count":
                        rows.append(pending)
                        pending = None
    assert config["env_name"] == "fire_commander" and config["use_binary"] is False
    assert [r["epoch"] for r in rows] == list(range(1, len(rows) + 1))
    return config, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "log_findings.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    script_path = Path(__file__).resolve()
    script_digest = hashlib.sha256(script_path.read_bytes()).hexdigest()
    report = {"producer": {"script": str(script_path.relative_to(ROOT)), "sha256": script_digest},
              "scope": "Descriptive FC training logs; no frozen-policy evaluation or causal inference.",
              "common_softrole_budget": 22500000, "softrole": [], "reproduction": []}
    all_configs, all_sources = [], []
    for model in ("shared", "banked"):
        for seed in range(3):
            index = 12 + (3 if model == "banked" else 0) + seed
            paths = list((ROOT / "logs_sr").glob(f"*_{index}.out"))
            assert len(paths) == 1
            rows = lines(paths[0])
            assert [r["epoch"] for r in rows] == list(range(1, len(rows) + 1))
            for count in ("episodes", "steps"):
                cumulative = 0
                for r in rows:
                    cumulative += r[count]
                    assert cumulative == r["total_" + count]
            for r in rows:
                assert all(math.isfinite(x) for x in r.values() if isinstance(x, (int, float)))
                assert r["updates"] == 10 * r["epoch"] and not r["partial_epoch"]
                assert r["event_exposed_episodes"] == 0
                assert abs(r["steps_taken"] * r["episodes"] - r["steps"]) < 1e-7
                assert abs(r["team_return"] - 2 * r["return_nocap"] - r["return_cap"]) < 1e-8
                assert abs(r["success_rate"] * r["episodes"] - round(r["success_rate"] * r["episodes"])) < 1e-8
            archive = ROOT / f"stokes_runs/runs/softrole_primary/fc_{model}/seed{seed}"
            config = load(archive / "config.json")
            assert (config["task"], config["model"], config["seed"]) == ("fc", model, seed)
            archived = lines(archive / "metrics.jsonl")
            assert rows[:len(archived)] == archived
            source = load(archive / "source_manifest.json")
            verified = 0
            for name, digest in source["files"].items():
                read(archive / "source" / name)
                assert INPUTS[str((archive / "source" / name).relative_to(ROOT))]["sha256"] == digest
                verified += 1
            run = load(archive / "run.json")
            assert source["sha256"] == run["source_sha256"]
            all_configs.append(config)
            all_sources.append(source)
            available = [r for r in rows if r["total_steps"] <= report["common_softrole_budget"]]
            item = {"model": model, "seed": seed, "source_log": str(paths[0].relative_to(ROOT)),
                    "completed_epochs": len(rows), "total_steps": rows[-1]["total_steps"],
                    "total_episodes": rows[-1]["total_episodes"], "archive_epochs": len(archived),
                    "archived_source_sha256": source["sha256"], "verified_archived_source_files": verified,
                    "config": config, "latest50": window(rows[-50:]),
                    "common_budget50": window(available[-50:]),
                    "last150_ten_epoch_blocks": [window(rows[start:start + 10])
                                                  for start in range(len(rows) - 150, len(rows), 10)]}
            if model == "shared" and seed == 2:
                item["regression_previous50"] = window(rows[-100:-50])
                item["regression_latest50"] = window(rows[-50:])
                item["regression_latest10"] = window(rows[-10:])
                item["regression_onset_epochs_986_1007"] = rows[985:1007]
            report["softrole"].append(item)
    report["softrole_config_differences"] = config_differences(all_configs)
    report["softrole_archived_sources_identical"] = all(s == all_sources[0] for s in all_sources)
    report["softrole_archived_file_hash_maps_identical"] = all(s["files"] == all_sources[0]["files"] for s in all_sources)
    configs, reproductions = [], []
    for seed in range(3):
        path = ROOT / f"logs_1/reproduce-896848_{6 + seed}.out"
        config, rows = reproduction(path)
        assert config["seed"] == seed
        archive = ROOT / f"stokes_runs/runs/reproduction-fast/fc_real/seed{seed}"
        saved = load(archive / "resolved_args.json")
        assert all(config[k] == saved[k] for k in config if k in saved)
        configs.append(config)
        reproductions.append(rows)
        report["reproduction"].append({"seed": seed, "source_log": str(path.relative_to(ROOT)),
                                       "completed_epochs": len(rows), "latest50": window(rows[-50:], False)})
    report["reproduction_config_differences"] = config_differences(configs)
    common_epoch = min(len(rows) for rows in reproductions)
    report["common_reproduction_epoch"] = common_epoch
    for item, rows in zip(report["reproduction"], reproductions):
        item["common_epoch50"] = window(rows[common_epoch - 50:common_epoch], False)
    # This independent probe must reproduce the previously exported tables.
    prior = load(ROOT / "analysis/training_2026-09-30_resync/summary.json")
    cross_checks = 0
    for item in report["softrole"]:
        old = next(r for r in prior["runs"] if r["task"] == "fc"
                   and r["model"] == "SoftRole " + item["model"] and r["seed"] == item["seed"])
        for new_key, old_key in (("latest50", "latest"), ("common_budget50", "common_sample")):
            for key, value in old[old_key].items():
                actual = item[new_key][key]
                assert math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-10) if isinstance(value, (int, float)) else actual == value
                cross_checks += 1
    for item in report["reproduction"]:
        old = next(r for r in prior["runs"] if r["task"] == "fc"
                   and r["model"] == "HetNet Real" and r["seed"] == item["seed"])
        for new_key, old_key in (("latest50", "latest"), ("common_epoch50", "common_epoch")):
            for key, value in old[old_key].items():
                actual = item[new_key][key]
                assert math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-10) if isinstance(value, (int, float)) else actual == value
                cross_checks += 1
    report["independent_summary_field_cross_checks"] = cross_checks
    report["limitations"] = [
        "Source/config/checkpoints are verified for archived prefixes only; appended stdout has no refreshed source archive.",
        "SoftRole windows pool episodes within seed; comparison between seeds does not make episodes independent training runs.",
        "Reproduction windows weight epochs equally because printed cumulative counters overcount true samples.",
        "Gate entropy is mixture-gate entropy, not policy/action entropy; shared has a single gate and zero gate entropy by construction.",
        "Null attention and value MSE changes are temporal associations and cannot identify why the policy regressed.",
        "Latest windows end at unequal environment sample counts; use the common-budget windows for descriptive comparison."
    ]
    # Detect concurrent changes of any bytes used by this audit.
    for name, identity in INPUTS.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == identity["sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == script_digest
    report["inputs"] = INPUTS
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output_label = str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)
    print(json.dumps({"output": output_label,
                      "verified_inputs": len(INPUTS),
                      "softrole_config_differences": report["softrole_config_differences"],
                      "identical_archived_sources": report["softrole_archived_sources_identical"],
                      "softrole_latest_success": [[r["model"], r["seed"], r["latest50"]["success_rate"]] for r in report["softrole"]]}, indent=2))


if __name__ == "__main__":
    main()
