"""Locked frozen evaluation of the capability-mechanism study (and supplement) checkpoints; no training."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from softrole import mechanism_study, mechanism_supplement
from softrole.config import Config
from softrole.evaluate import evaluate_checkpoint
from softrole.ledger import committed_checkpoints, sha
from softrole.programme import normalized
from softrole.scenarios import make_scenarios, scenario_from_record
from softrole.train import source_snapshot, write_json


VERSION = 1
EVALUATION = "capability_mechanism_evaluation_v1"
PANEL_SEED = 4100
EPISODES = 500  # scenarios per composition per panel
FAILURE_WINDOW = (5, 15)
VISION = 2
MESSAGING_ARMS = ("capcon", "capcon_class_input", "class_typed", "ca_cc_gnn", "capcon_hyper")
# One evaluation code path serves every study with a locked launcher: the module is chosen by the study name
# the training manifest (at preparation) or the evaluation manifest's protocol (afterwards) records.
STUDIES = {mechanism_study.STUDY: mechanism_study, mechanism_supplement.STUDY: mechanism_supplement}
# (panel, compositions, sensing levels of P agents, sensor failure on every scenario)
PANELS = (
    ("in_dist", ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1)), (1, 3), False),
    ("heldout_composition", ((3, 2),), (1, 3), False),
    ("larger_teams", ((4, 1), (4, 2)), (1, 3), False),
    ("heldout_level", ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1)), (2,), False),
    ("failure", ((2, 1), (2, 2), (3, 1)), (1, 3), True),
)
# (cell, panel, intervention, intervention_step, sham, messaging arms only)
CELLS = (
    ("in_dist", "in_dist", "none", None, False, False),
    ("heldout_composition", "heldout_composition", "none", None, False, False),
    ("larger_teams", "larger_teams", "none", None, False, False),
    ("heldout_level", "heldout_level", "none", None, False, False),
    ("failure", "failure", "none", None, False, False),
    ("sham", "failure", "none", None, True, False),
    ("in_dist_comm_off", "in_dist", "comm_off", 0, False, True),
    ("failure_comm_off", "failure", "comm_off", None, False, True),
    ("sham_comm_off", "failure", "comm_off", None, True, True),
)
# (name, minuend cell, subtrahend cell): per-scenario difference, matched by scenario_id.
COMPARISONS = (
    ("failure_minus_sham", "failure", "sham"),
    ("in_dist_comm_off_minus_in_dist", "in_dist_comm_off", "in_dist"),
    ("failure_comm_off_minus_failure", "failure_comm_off", "failure"),
    ("sham_comm_off_minus_sham", "sham_comm_off", "sham"),
)
CELL_COLUMNS = ("condition", "arm", "seed", "cell", "num_p", "num_a", "episodes", "success_rate",
                "capped_completion", "mean_team_return", "event_exposed_episodes")
PAIRED_COLUMNS = ("condition", "arm", "seed", "comparison", "num_p", "num_a", "pooled", "pairs",
                  "success_difference", "capped_completion_difference")


def panel_table():
    return [dict(name=name, panel_index=index, compositions=[list(c) for c in compositions],
                 sensing_p=list(levels), sensing_a=[], vision=VISION,
                 failure_probability=int(failure),
                 failure_window=list(FAILURE_WINDOW) if failure else None,
                 victim_rule="uniform over initially sensing agents" if failure else None,
                 episodes_per_composition=EPISODES, scenarios=len(compositions) * EPISODES)
            for index, (name, compositions, levels, failure) in enumerate(PANELS)]


def cell_table():
    return [dict(name=name, panel=panel, intervention=intervention, intervention_step=step,
                 sham=sham, arms="messaging" if messaging else "all")
            for name, panel, intervention, step, sham, messaging in CELLS]


def applicable_cells(arm):
    """Cells evaluated for an arm; message-free arms skip every comm_off cell."""
    return [cell for cell in cell_table() if cell["arms"] == "all" or arm in MESSAGING_ARMS]


def study_named(name):
    """The study module registered under a recorded study name."""
    if not isinstance(name, str) or name not in STUDIES:
        raise ValueError(f"Unknown mechanism study {name!r}; the evaluation supports {sorted(STUDIES)}")
    return STUDIES[name]


def study_of(manifest):
    """The study module an evaluation manifest was prepared for, from its protocol's study name."""
    record = manifest.get("protocol") if isinstance(manifest, dict) else None
    return study_named(record.get("study") if isinstance(record, dict) else None)


def require_natural(index):
    if type(index) is not int or index < 0:
        raise ValueError("evaluation index must be a nonnegative integer")


def require_index(study, index):
    """An index inside the study's own range (the array of a study with fewer runs is submitted shorter)."""
    require_natural(index)
    if index >= study.COUNT:
        raise ValueError(f"evaluation index {index} is out of range for {study.STUDY}: it has {study.COUNT} "
                         f"entries, indices 0 through {study.COUNT - 1}")


def protocol(study=mechanism_study):
    return {
        "purpose": "Frozen evaluation of the capability-mechanism study checkpoints; the training "
                   "stage is complete and no model is changed here",
        "study": study.STUDY,
        "conditions": [{"name": name, "dim": dim} for name, dim in study.CONDITIONS],
        "arms": [{"name": name, "model": model, "descriptor": descriptor}
                 for name, model, descriptor in study.ARMS],
        "messaging_arms": list(MESSAGING_ARMS),
        "message_free_arms_skip_cells": [cell["name"] for cell in cell_table() if cell["arms"] == "messaging"],
        "seeds": list(study.SEEDS),
        "panel_seed": PANEL_SEED,
        "episodes_per_composition_per_panel": EPISODES,
        "scenario_seed_rule": "make_scenarios(SeedSequence([panel_seed, panel_index, num_p, num_a])"
                              ".generate_state(1)[0], episodes, [composition], ...); panel_index is the "
                              "zero-based position in the panel table",
        "scenarios_shared_across_conditions": True,
        "panels": panel_table(),
        "cells": cell_table(),
        "checkpoint_rule": "first committed checkpoint at or above the target; never best epoch",
        "target_joint_environment_steps_per_run": study.TARGET_STEPS,
        "metrics": {
            "success_rate": "mean episode success",
            "capped_completion": "mean over episodes of steps if success else the horizon "
                                 "(config max_steps, 80)",
            "mean_team_return": "mean episode team return",
            "event_exposed_episodes": "failure-panel cells only: episodes in which the sensor loss was "
                                      "applied (zero in sham cells by construction)",
        },
        "paired_differences": {
            "comparisons": [{"name": name, "minuend": a, "subtrahend": b} for name, a, b in COMPARISONS],
            "definition": "per scenario_id, minuend minus subtrahend on success (0/1) and capped "
                          "completion, averaged within seed over the matched scenarios, per "
                          "composition and pooled over the panel; no bootstrap or significance test",
        },
        "device": "cpu",
        "dtype": "float64",
    }


def panel_seed(panel_index, composition):
    return int(np.random.SeedSequence([PANEL_SEED, panel_index, *map(int, composition)]).generate_state(1)[0])


def build_panels():
    """Scenario lists by panel name; independent of arm, condition and checkpoint."""
    result = {}
    for index, (name, compositions, levels, failure) in enumerate(PANELS):
        scenarios = []
        for composition in compositions:
            scenarios += make_scenarios(panel_seed(index, composition), EPISODES, [composition],
                                        failure_prob=int(failure), failure_window=FAILURE_WINDOW,
                                        sensing_p=levels, sensing_a=(), vision=VISION)
        result[name] = scenarios
    return result


def panel_bytes(scenarios):
    return (json.dumps([asdict(s) for s in scenarios], sort_keys=True, indent=1, allow_nan=False) + "\n").encode()


def is_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def relative_path(value):
    return isinstance(value, str) and not Path(value).is_absolute() and ".." not in Path(value).parts


def core(entry):
    return {key: entry[key] for key in ("index", "condition", "arm", "seed", "relative_output")}


def validate_entry(entry, row, training, study):
    keys = set(core(row))
    if not isinstance(entry, dict) or not keys <= set(entry) or core(entry) != core(row):
        raise ValueError("Manifest entries differ from the locked study")
    cells = applicable_cells(row["arm"])
    skipped = [cell["name"] for cell in cell_table() if cell not in cells]
    if entry.get("cells") != [cell["name"] for cell in cells] or entry.get("skipped_cells") != skipped:
        raise ValueError("Manifest cells differ from the locked cell table")
    common = keys | {"cells", "skipped_cells", "status"}
    if entry.get("status") == "ready" and set(entry) == common | {"checkpoint"}:
        checkpoint = entry["checkpoint"]
        if not isinstance(checkpoint, dict):
            raise ValueError("Malformed checkpoint record")
        path = checkpoint.get("path")
        if (set(checkpoint) != {"path", "sha256", "total_steps", "epoch", "source_sha256"}
                or not relative_path(path)
                or Path(path).parent.as_posix() != f"training/{row['relative_output']}/checkpoints"
                or not (Path(path).name.startswith("epoch") and Path(path).suffix == ".pt")
                or not is_hex(checkpoint["sha256"]) or not is_hex(checkpoint["source_sha256"])
                or checkpoint["source_sha256"] != training["source_sha256"]
                or type(checkpoint["total_steps"]) is not int
                or checkpoint["total_steps"] < study.TARGET_STEPS
                or type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 1):
            raise ValueError("Malformed checkpoint record")
    elif entry.get("status") == "missing" and set(entry) == common | {"reason"}:
        if not isinstance(entry["reason"], str) or not entry["reason"] or not training["allow_missing"]:
            raise ValueError("Malformed or undeclared missing-run record")
    else:
        raise ValueError("Malformed manifest entry")


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
            "version", "evaluation", "protocol", "panels", "entries", "training", "source", "manifest_sha256"}:
        raise ValueError("Unsupported mechanism evaluation manifest")
    study = study_of(manifest)
    if (type(manifest["version"]) is not int or manifest["version"] != VERSION
            or manifest["evaluation"] != EVALUATION
            or study.digest(manifest["protocol"]) != study.digest(protocol(study))):
        raise ValueError("Manifest differs from the locked mechanism evaluation")
    panels = manifest["panels"]
    if not isinstance(panels, dict) or sorted(panels) != sorted(row["name"] for row in panel_table()):
        raise ValueError("Manifest panels differ from the locked panel table")
    for row in panel_table():
        panel = panels[row["name"]]
        if (not isinstance(panel, dict) or set(panel) != {"path", "sha256", "scenarios"}
                or panel["path"] != f"panels/{row['name']}.json" or not is_hex(panel["sha256"])
                or type(panel["scenarios"]) is not int or panel["scenarios"] != row["scenarios"]):
            raise ValueError(f"Malformed panel record: {row['name']}")
    training = manifest["training"]
    if (not isinstance(training, dict)
            or set(training) != {"run_dir", "manifest_sha256", "source_sha256", "allow_missing"}
            or not isinstance(training["run_dir"], str) or not training["run_dir"]
            or not is_hex(training["manifest_sha256"]) or not is_hex(training["source_sha256"])
            or type(training["allow_missing"]) is not bool):
        raise ValueError("Malformed training provenance")
    entries = manifest["entries"]
    expected = study.expected_entries()
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("Manifest requires one entry per training run")
    for entry, row in zip(entries, expected):
        validate_entry(entry, row, training, study)
    source = manifest["source"]
    if not isinstance(source, dict) or not isinstance(source.get("files"), dict) or not source["files"]:
        raise ValueError("Manifest requires a source file inventory")
    for name, value in source["files"].items():
        if not isinstance(name, str) or not relative_path(name) or not is_hex(value):
            raise ValueError("Malformed source file inventory")
    if source.get("sha256") != study.digest(source["files"]):
        raise ValueError("Manifest source inventory hash differs")
    if manifest["manifest_sha256"] != study.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Manifest hash differs")
    return manifest


def select_checkpoint(training_run, entry, training_manifest, study):
    """Return the entry's evaluation fields: a ready checkpoint record or a declared missing reason."""
    run = training_run / "training" / entry["relative_output"]
    cells = applicable_cells(entry["arm"])
    base = dict(core(entry), cells=[cell["name"] for cell in cells],
                skipped_cells=[cell["name"] for cell in cell_table() if cell not in cells])

    def missing(reason):
        return dict(base, status="missing", reason=reason)

    if not run.is_dir():
        return missing("training run directory absent")
    for name in ("finished.json", "study.json"):
        if not (run / name).is_file():
            return missing(f"{name} absent; the training run is incomplete")
    record = json.loads((run / "study.json").read_text())
    if (record.get("index") != entry["index"] or record.get("manifest_sha256") != training_manifest["manifest_sha256"]
            or record.get("source_sha256") != training_manifest["source"]["sha256"]):
        raise ValueError(f"study.json does not belong to this training manifest: {run}")
    candidates = committed_checkpoints(run, stop_at_steps=study.TARGET_STEPS)
    chosen = next((row for row in candidates if row["total_steps"] >= study.TARGET_STEPS), None)
    if chosen is None:
        return missing(f"no committed checkpoint reaches {study.TARGET_STEPS} joint steps")
    if chosen["source_sha256"] != training_manifest["source"]["sha256"]:
        raise ValueError(f"Checkpoint source hash differs from the training manifest: {run}")
    path = Path(chosen["checkpoint"])
    if not path.is_relative_to(training_run):
        raise ValueError(f"Checkpoint must stay inside the training run: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if normalized(Config(**saved["config"])) != entry["config"]:
        raise ValueError(f"Checkpoint configuration differs from its manifest entry: {run}")
    return dict(base, status="ready", checkpoint=dict(
        path=path.relative_to(training_run).as_posix(), sha256=chosen["sha256"],
        total_steps=chosen["total_steps"], epoch=chosen["epoch"], source_sha256=chosen["source_sha256"]))


def prepare(training_run_dir, output_dir, allow_missing=False):
    training_run, output = Path(training_run_dir).resolve(), Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    recorded = json.loads((training_run / "manifest.json").read_text())
    study = study_named(recorded.get("study") if isinstance(recorded, dict) else None)
    training_manifest = study.validate_manifest(recorded)
    entries = [select_checkpoint(training_run, entry, training_manifest, study)
               for entry in training_manifest["entries"]]
    absent = [entry for entry in entries if entry["status"] == "missing"]
    if absent and not allow_missing:
        raise ValueError(f"{len(absent)} training runs are missing or incomplete (first: "
                         f"{absent[0]['relative_output']}: {absent[0]['reason']}); "
                         "use --allow-missing to record them")
    files = {name: panel_bytes(scenarios) for name, scenarios in build_panels().items()}
    manifest = dict(
        version=VERSION, evaluation=EVALUATION, protocol=protocol(study),
        panels={name: dict(path=f"panels/{name}.json", sha256=hashlib.sha256(data).hexdigest(),
                           scenarios=len(json.loads(data))) for name, data in files.items()},
        entries=entries,
        training=dict(run_dir=str(training_run), manifest_sha256=training_manifest["manifest_sha256"],
                      source_sha256=training_manifest["source"]["sha256"], allow_missing=bool(allow_missing)),
        source=source_snapshot())
    manifest["manifest_sha256"] = study.digest(manifest)
    validate_manifest(manifest)
    (output / "panels").mkdir(parents=True)
    for name, data in files.items():
        with (output / "panels" / f"{name}.json").open("xb") as stream:
            stream.write(data)
    write_json(output / "manifest.json", manifest)
    return manifest


def load_panel(root, manifest, name):
    panel = manifest["panels"][name]
    path = root / panel["path"]
    if not path.resolve().is_relative_to(root):
        raise ValueError("Panel path must stay inside the evaluation output")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != panel["sha256"]:
        raise ValueError(f"Scenario panel changed since preparation: {name}")
    scenarios = [scenario_from_record(record) for record in json.loads(data)]
    if len(scenarios) != panel["scenarios"]:
        raise ValueError(f"Scenario panel has the wrong number of scenarios: {name}")
    return scenarios


def validate_report(report, entry, cell, panel, scenarios, source_sha256):
    """Bind a report to its manifest entry, cell and panel; used to keep and to summarize reports."""
    if not isinstance(report, dict):
        raise ValueError("Report must be a JSON object")
    expected = {"checkpoint_sha256": entry["checkpoint"]["sha256"], "scenarios_sha256": panel["sha256"],
                "intervention": cell["intervention"], "sham": cell["sham"], "episodes": len(scenarios)}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"Report does not match the manifest: {key}")
    evaluator = report.get("evaluator")
    if (not isinstance(evaluator, dict) or not isinstance(evaluator.get("source"), dict)
            or evaluator["source"].get("sha256") != source_sha256):
        raise ValueError("Report does not match the manifest: evaluator source")
    episodes = report.get("per_episode")
    if not isinstance(episodes, list) or len(episodes) != len(scenarios):
        raise ValueError("Report does not cover its complete scenario panel")
    for episode, scenario in zip(episodes, scenarios):
        step = cell["intervention_step"] if cell["intervention_step"] is not None else scenario.event_step
        if (not isinstance(episode, dict) or episode.get("scenario_id") != str(scenario.scenario_id)
                or type(episode.get("success")) is not bool or type(episode.get("steps")) is not int
                or not isinstance(episode.get("team_return"), (int, float))
                or not math.isfinite(episode["team_return"]) or episode.get("intervention_step") != step):
            raise ValueError("Report episodes differ from the manifest scenarios or cell")
    return report


def contained(path, root, label):
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"{label} must stay inside the evaluation output")


def launch(manifest_path, index, dry_run=False, training_run=None):
    require_natural(index)
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    require_index(study_of(manifest), index)
    source = source_snapshot()
    if source["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Source changed since evaluation preparation; use the frozen checkout")
    entry = manifest["entries"][index]
    record = dict(evaluation=EVALUATION, index=index, condition=entry["condition"], arm=entry["arm"],
                  seed=entry["seed"], dry_run=bool(dry_run), source_sha256=source["sha256"],
                  manifest_sha256=manifest["manifest_sha256"])
    if entry["status"] == "missing":
        return dict(record, status="skipped", reason=entry["reason"], cells=[])
    root = Path(manifest_path).resolve().parent
    training = Path(training_run or manifest["training"]["run_dir"]).resolve()
    if not training.is_dir():
        raise FileNotFoundError(f"Training run directory not found: {training}")
    checkpoint = training / entry["checkpoint"]["path"]
    contained(checkpoint, training, "Checkpoint")
    if sha(checkpoint) != entry["checkpoint"]["sha256"]:
        raise ValueError(f"Checkpoint changed since preparation: {checkpoint}")
    panels = {name: load_panel(root, manifest, name) for name in manifest["panels"]}
    plan = []
    for cell in applicable_cells(entry["arm"]):
        report = root / "reports" / entry["relative_output"] / f"{cell['name']}.json"
        contained(report, root, "Report")
        if report.is_symlink():
            raise FileExistsError(f"Report path is a symlink: {report}")
        action = "would_evaluate" if dry_run else "evaluate"
        if report.exists():
            validate_report(json.loads(report.read_text()), entry, cell, manifest["panels"][cell["panel"]],
                            panels[cell["panel"]], source["sha256"])
            action = "kept"
        plan.append((cell, report, action))
    record["checkpoint"] = str(checkpoint)
    record["cells"] = [dict(cell=cell["name"], report=str(report), action=action) for cell, report, action in plan]
    if dry_run:
        return dict(record, status="dry_run")
    for cell, report, action in plan:
        if action == "evaluate":
            result = evaluate_checkpoint(
                checkpoint, panels[cell["panel"]], report, intervention=cell["intervention"],
                intervention_step=cell["intervention_step"], sham=cell["sham"],
                scenarios_sha256=manifest["panels"][cell["panel"]]["sha256"])
            validate_report(result, entry, cell, manifest["panels"][cell["panel"]],
                            panels[cell["panel"]], source["sha256"])
    if source_snapshot()["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed during evaluation; inspect the reports before using them")
    for row in record["cells"]:
        row["action"] = "kept" if row["action"] == "kept" else "evaluated"
    return dict(record, status="evaluated")


def outcomes(report):
    """scenario_id -> (composition, success as 0/1, capped completion) with failures at the horizon."""
    horizon = report["config"]["max_steps"]
    return {episode["scenario_id"]: (tuple(episode["composition"]), int(episode["success"]),
                                     episode["steps"] if episode["success"] else horizon)
            for episode in report["per_episode"]}


def cell_rows(identity, cell, report):
    groups = {}
    for episode in report["per_episode"]:
        groups.setdefault(tuple(episode["composition"]), []).append(episode)
    horizon = report["config"]["max_steps"]
    rows = []
    for composition in sorted(groups):
        episodes = groups[composition]
        rows.append(dict(
            identity, cell=cell["name"], num_p=composition[0], num_a=composition[1], episodes=len(episodes),
            success_rate=sum(e["success"] for e in episodes) / len(episodes),
            capped_completion=sum(e["steps"] if e["success"] else horizon for e in episodes) / len(episodes),
            mean_team_return=sum(e["team_return"] for e in episodes) / len(episodes),
            event_exposed_episodes=(sum(bool(e["event_exposed"]) for e in episodes)
                                    if cell["panel"] == "failure" else None)))
    return rows


def paired_rows(identity, comparison, minuend, subtrahend):
    """Per-scenario differences averaged within seed, per composition and pooled."""
    first, second = outcomes(minuend), outcomes(subtrahend)
    if first.keys() != second.keys() or any(first[k][0] != second[k][0] for k in first):
        raise ValueError("Paired cells must cover the same scenarios and compositions")
    groups = {}
    for scenario_id, (composition, success, capped) in first.items():
        groups.setdefault(composition, []).append((success - second[scenario_id][1],
                                                   capped - second[scenario_id][2]))

    def row(composition, pairs):
        return dict(identity, comparison=comparison,
                    num_p=composition[0] if composition else None, num_a=composition[1] if composition else None,
                    pooled=composition is None, pairs=len(pairs),
                    success_difference=sum(p[0] for p in pairs) / len(pairs),
                    capped_completion_difference=sum(p[1] for p in pairs) / len(pairs))

    rows = [row(composition, groups[composition]) for composition in sorted(groups)]
    return rows + [row(None, [pair for pairs in groups.values() for pair in pairs])]


def write_table(path, rows, columns):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in columns})


def summarize(manifest_path, output, allow_missing=False):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Summary output already exists: {output}")
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    root = Path(manifest_path).resolve().parent
    panels = {name: load_panel(root, manifest, name) for name in manifest["panels"]}
    source_sha256 = manifest["source"]["sha256"]
    cells, paired, missing = [], [], []
    for entry in manifest["entries"]:
        identity = dict(condition=entry["condition"], arm=entry["arm"], seed=entry["seed"])
        if entry["status"] == "missing":
            missing.append(dict(identity, index=entry["index"], cell=None, reason=entry["reason"]))
            continue
        reports = {}
        for cell in applicable_cells(entry["arm"]):
            path = root / "reports" / entry["relative_output"] / f"{cell['name']}.json"
            contained(path, root, "Report")
            if not path.is_file():
                missing.append(dict(identity, index=entry["index"], cell=cell["name"], reason="report not found"))
                continue
            report = validate_report(json.loads(path.read_text()), entry, cell, manifest["panels"][cell["panel"]],
                                     panels[cell["panel"]], source_sha256)
            reports[cell["name"]] = report
            cells += cell_rows(identity, cell, report)
        for name, first, second in COMPARISONS:
            if first in reports and second in reports:
                paired += paired_rows(identity, name, reports[first], reports[second])
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} missing training runs or reports (first: {missing[0]}); "
                         "use --allow-missing to list them in the summary")
    output.mkdir(parents=True)
    write_table(output / "cells.csv", cells, CELL_COLUMNS)
    write_table(output / "paired.csv", paired, PAIRED_COLUMNS)
    result = dict(version=VERSION, evaluation=EVALUATION, manifest_sha256=manifest["manifest_sha256"],
                  allow_missing=bool(allow_missing), cells=cells, paired=paired, missing=missing)
    with (output / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = cli.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", allow_abbrev=False)
    preparation.add_argument("--training-run", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--allow-missing", action="store_true")
    execution = commands.add_parser("launch", allow_abbrev=False)
    execution.add_argument("--manifest", type=Path, required=True)
    # The study is known only from the manifest; launch() enforces that study's own range.
    execution.add_argument("--index", type=int, choices=range(max(module.COUNT for module in STUDIES.values())),
                           metavar="INDEX", required=True)
    execution.add_argument("--training-run", type=Path)
    execution.add_argument("--dry-run", action="store_true")
    summary = commands.add_parser("summarize", allow_abbrev=False)
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--allow-missing", action="store_true")
    args = cli.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.training_run, args.output, args.allow_missing)
        elif args.command == "launch":
            result = launch(args.manifest, args.index, args.dry_run, args.training_run)
        else:
            summarize(args.manifest, args.output, args.allow_missing)
            result = dict(summary=str(args.output), allow_missing=bool(args.allow_missing))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
