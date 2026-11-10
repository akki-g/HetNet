"""Frozen evaluation of the capability-mechanism study: panels, cells, manifest, launch, summary and launchers."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import time

import numpy as np
import pytest

from scripts import submit_stokes as submit
from softrole import mechanism_evaluation as mech
from softrole import mechanism_study as study
from softrole.config import COMPOSITION_TEST, COMPOSITION_TRAIN, EXTRAPOLATION_TEST
from softrole.ledger import committed_checkpoints
from softrole.rollout import validate_scenario
from softrole.scenarios import make_scenarios
from stokes_test_support import configure_stokes_run

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "slurm/softrole_mechanism_eval.sbatch"
PREPARE = ROOT / "slurm/softrole_mechanism_eval_prepare.sbatch"
RUNBOOK = ROOT / "src/softrole/MECHANISM_EVALUATION.md"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
ARM_NAMES = [name for name, _, _ in study.ARMS]
MESSAGING = ("capcon", "capcon_class_input", "class_typed", "ca_cc_gnn")


# --- 1. Panels ----------------------------------------------------------------

@pytest.fixture(scope="module")
def panels():
    return mech.build_panels()


def test_locked_constants_and_panel_table():
    assert (mech.VERSION, mech.EVALUATION) == (1, "capability_mechanism_evaluation_v1")
    assert (mech.PANEL_SEED, mech.EPISODES, mech.FAILURE_WINDOW) == (4100, 500, (5, 15))
    assert mech.MESSAGING_ARMS == MESSAGING
    table = {row["name"]: row for row in mech.panel_table()}
    assert list(table) == ["in_dist", "heldout_composition", "larger_teams", "heldout_level", "failure"]
    assert table["in_dist"]["compositions"] == [list(c) for c in COMPOSITION_TRAIN]
    assert table["heldout_composition"]["compositions"] == [list(c) for c in COMPOSITION_TEST]
    assert table["larger_teams"]["compositions"] == [list(c) for c in EXTRAPOLATION_TEST]
    assert table["heldout_level"]["compositions"] == [list(c) for c in COMPOSITION_TRAIN]
    assert table["failure"]["compositions"] == [[2, 1], [2, 2], [3, 1]]
    assert [table[n]["sensing_p"] for n in table] == [[1, 3], [1, 3], [1, 3], [2], [1, 3]]
    assert [table[n]["failure_probability"] for n in table] == [0, 0, 0, 0, 1]
    assert all(row["vision"] == 2 and row["sensing_a"] == [] for row in table.values())
    assert table["failure"]["failure_window"] == [5, 15]
    assert [table[n]["scenarios"] for n in table] == [2500, 500, 1000, 2500, 1500]


def test_panel_counts_per_composition_and_unique_ids(panels):
    table = {row["name"]: row for row in mech.panel_table()}
    seen = set()
    for name, scenarios in panels.items():
        counts = Counter((s.num_p, s.num_a) for s in scenarios)
        assert counts == {tuple(c): 500 for c in table[name]["compositions"]}
        ids = [s.scenario_id for s in scenarios]
        assert len(set(ids)) == len(ids) and not seen & set(ids)
        seen |= set(ids)


def test_sensing_levels_events_and_victims(panels):
    for name, scenarios in panels.items():
        levels = set()
        for s in scenarios:
            assert len(s.sensing) == s.num_p + s.num_a
            assert all(level == 0 for level in s.sensing[s.num_p:])
            levels |= set(s.sensing[:s.num_p])
        assert levels == ({2} if name == "heldout_level" else {1, 3})
    for name, scenarios in panels.items():
        if name != "failure":
            assert all(s.event_step == -1 and s.victim == -1 for s in scenarios)
    failure = panels["failure"]
    assert all(5 <= s.event_step <= 15 for s in failure)
    assert {s.event_step for s in failure} == set(range(5, 16))
    assert all(0 <= s.victim < s.num_p and s.sensing[s.victim] >= 1 for s in failure)
    assert {s.victim for s in failure if (s.num_p, s.num_a) == (3, 1)} == {0, 1, 2}


def test_scenarios_follow_the_seed_rule_and_are_independent_of_arm_and_condition(panels):
    assert not inspect.signature(mech.build_panels).parameters
    assert mech.build_panels() == panels
    assert mech.protocol()["scenarios_shared_across_conditions"] is True
    for index, row in enumerate(mech.panel_table()):
        for composition in row["compositions"]:
            seed = int(np.random.SeedSequence([4100, index, *composition]).generate_state(1)[0])
            mine = [s for s in panels[row["name"]] if (s.num_p, s.num_a) == tuple(composition)]
            reference = make_scenarios(
                seed, 500, [tuple(composition)], failure_prob=row["failure_probability"],
                failure_window=(5, 15), sensing_p=tuple(row["sensing_p"]), sensing_a=(), vision=2)
            assert mine == reference and mine[0].scenario_id == f"{seed}:0"
    assert mech.panel_bytes(panels["failure"]) == mech.panel_bytes(mech.build_panels()["failure"])


# --- 2. Cells -----------------------------------------------------------------

SPEC_CELLS = {
    "in_dist": ("in_dist", "none", None, False), "heldout_composition": ("heldout_composition", "none", None, False),
    "larger_teams": ("larger_teams", "none", None, False), "heldout_level": ("heldout_level", "none", None, False),
    "failure": ("failure", "none", None, False), "sham": ("failure", "none", None, True),
    "in_dist_comm_off": ("in_dist", "comm_off", 0, False), "failure_comm_off": ("failure", "comm_off", None, False),
    "sham_comm_off": ("failure", "comm_off", None, True),
}


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_applicable_cells_per_arm(arm):
    cells = mech.applicable_cells(arm)
    names = [c["name"] for c in cells]
    assert len(names) == (9 if arm in MESSAGING else 6)
    assert names == [n for n in SPEC_CELLS if arm in MESSAGING or "comm_off" not in n]
    for cell in cells:
        assert (cell["panel"], cell["intervention"], cell["intervention_step"], cell["sham"]) == SPEC_CELLS[cell["name"]]
    protocol = mech.protocol()
    assert protocol["message_free_arms_skip_cells"] == ["in_dist_comm_off", "failure_comm_off", "sham_comm_off"]
    assert protocol["messaging_arms"] == list(MESSAGING)


@pytest.mark.parametrize("index", [0, 69])
def test_every_cell_scenario_passes_validate_scenario(panels, index):
    config = study.study_config(index)
    for cell in mech.cell_table():
        for scenario in panels[cell["panel"]]:
            config.validate_composition((scenario.num_p, scenario.num_a))
            validate_scenario(config, scenario, cell["intervention"], cell["intervention_step"], cell["sham"])


# --- 3. End to end on tiny data -----------------------------------------------

TINY_TARGET, TINY_EPISODES, TINY_TOTAL = 30, 2, 100
REAL_STUDY_CONFIG = study.study_config
INDICES = (0, 15)  # s5/capcon/seed0 (messaging) and s5/ca_mlp/seed0 (message-free)


def tiny_study_config(index):
    return replace(REAL_STUDY_CONFIG(index), max_steps=20, batch_steps=20, nprocesses=1, epochs=12,
                   updates_per_epoch=1, save_every=1, hidden_dim=16, pre_dim=16, heads=2,
                   head_dim=4, msg_dim=8, total_steps=TINY_TOTAL)


def apply_tiny(patcher):
    patcher.setattr(study, "study_config", tiny_study_config)
    patcher.setattr(study, "TARGET_STEPS", TINY_TARGET)
    patcher.setattr(mech, "EPISODES", TINY_EPISODES)


@pytest.fixture
def tiny(monkeypatch):
    apply_tiny(monkeypatch)


@pytest.fixture(scope="module")
def training_run(tmp_path_factory):
    container = tmp_path_factory.mktemp("training_container")
    with pytest.MonkeyPatch.context() as patcher:
        apply_tiny(patcher)
        manifest = container / "manifest.json"
        study.prepare(manifest)
        for index in INDICES:
            study.launch(manifest, index, container / "training")
    return container


@pytest.fixture(scope="module")
def evaluated(training_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation_container")
    output, timings = root / "evaluation", {}
    with pytest.MonkeyPatch.context() as patcher:
        apply_tiny(patcher)
        manifest = mech.prepare(training_run, output, allow_missing=True)
        records = {}
        for index in INDICES:
            started = time.perf_counter()
            records[index] = mech.launch(output / "manifest.json", index)
            timings[index] = time.perf_counter() - started
        summary = mech.summarize(output / "manifest.json", root / "summary", allow_missing=True)
    return SimpleNamespace(root=root, output=output, manifest=manifest, records=records, summary=summary,
                           summary_dir=root / "summary", timings=timings)


def copy_evaluation(evaluated, tmp_path):
    destination = tmp_path / "evaluation"
    shutil.copytree(evaluated.output, destination)
    return destination


def report_files(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def load_report(evaluated, entry_path, cell):
    return json.loads((evaluated.output / "reports" / entry_path / f"{cell}.json").read_text())


def test_prepare_selects_the_first_committed_checkpoint_at_the_target(tiny, training_run, evaluated):
    manifest = json.loads((evaluated.output / "manifest.json").read_text())
    assert manifest == json.loads(json.dumps(evaluated.manifest))
    assert mech.validate_manifest(manifest) is manifest
    assert manifest["version"] == 1 and manifest["evaluation"] == "capability_mechanism_evaluation_v1"
    assert manifest["protocol"] == json.loads(json.dumps(mech.protocol()))
    assert manifest["training"]["manifest_sha256"] == json.loads(
        (training_run / "manifest.json").read_text())["manifest_sha256"]
    assert manifest["training"]["run_dir"] == str(training_run.resolve())
    assert manifest["source"]["files"] and manifest["manifest_sha256"] == study.digest(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"})
    assert len(manifest["entries"]) == 70
    for index in INDICES:
        entry = manifest["entries"][index]
        run = training_run / "training" / entry["relative_output"]
        candidates = committed_checkpoints(run)
        first = next(c for c in candidates if c["total_steps"] >= TINY_TARGET)
        assert first["epoch"] < candidates[-1]["epoch"]  # never the final or best checkpoint
        assert entry["status"] == "ready"
        assert (training_run / entry["checkpoint"]["path"]).resolve() == Path(first["checkpoint"])
        assert entry["checkpoint"]["path"] == f"training/{entry['relative_output']}/checkpoints/epoch{first['epoch']:04d}.pt"
        assert {k: entry["checkpoint"][k] for k in ("sha256", "total_steps", "epoch", "source_sha256")} == {
            k: first[k] for k in ("sha256", "total_steps", "epoch", "source_sha256")}
        assert entry["checkpoint"]["source_sha256"] == manifest["training"]["source_sha256"]
    for entry in manifest["entries"]:
        if entry["index"] not in INDICES:
            assert entry["status"] == "missing" and entry["reason"] == "training run directory absent"
    assert manifest["entries"][0]["cells"] == [c for c in SPEC_CELLS]
    assert manifest["entries"][0]["skipped_cells"] == []
    assert manifest["entries"][15]["skipped_cells"] == ["in_dist_comm_off", "failure_comm_off", "sham_comm_off"]
    assert sorted(p.name for p in (evaluated.output / "panels").iterdir()) == sorted(
        f"{n}.json" for n in manifest["panels"])
    for name, panel in manifest["panels"].items():
        data = (evaluated.output / panel["path"]).read_bytes()
        assert panel["sha256"] == hashlib.sha256(data).hexdigest()
        assert len(json.loads(data)) == panel["scenarios"] == len(mech.build_panels()[name])


def test_launch_writes_one_report_per_applicable_cell_and_records_identity(tiny, evaluated):
    manifest = evaluated.manifest
    reports = evaluated.output / "reports"
    assert sorted(str(p.relative_to(reports)) for p in reports.rglob("*") if p.is_file()) == sorted(
        [f"s5/capcon/seed0/{cell}.json" for cell in SPEC_CELLS]
        + [f"s5/ca_mlp/seed0/{cell}.json" for cell in SPEC_CELLS if "comm_off" not in cell])
    for index, relative in ((0, "s5/capcon/seed0"), (15, "s5/ca_mlp/seed0")):
        entry = manifest["entries"][index]
        record = evaluated.records[index]
        assert record["status"] == "evaluated" and record["index"] == index and not record["dry_run"]
        assert [row["action"] for row in record["cells"]] == ["evaluated"] * len(entry["cells"])
        for cell in mech.applicable_cells(entry["arm"]):
            report = load_report(evaluated, relative, cell["name"])
            panel = manifest["panels"][cell["panel"]]
            assert report["checkpoint_sha256"] == entry["checkpoint"]["sha256"]
            assert report["scenarios_sha256"] == panel["sha256"]
            assert report["intervention"] == cell["intervention"] and report["sham"] is cell["sham"]
            assert report["episodes"] == panel["scenarios"] == len(report["per_episode"])
            assert report["evaluator"]["source"]["sha256"] == manifest["source"]["sha256"]
            assert report["source_sha256"] == entry["checkpoint"]["source_sha256"]
            assert report["checkpoint_progress"]["total_steps"] == entry["checkpoint"]["total_steps"]
            assert report["checkpoint"].endswith(entry["checkpoint"]["path"])
            assert all(np.isfinite(e["team_return"]) for e in report["per_episode"])
    capcon = {cell: load_report(evaluated, "s5/capcon/seed0", cell) for cell in SPEC_CELLS}
    assert all(e["intervention_exposed"] for e in capcon["in_dist_comm_off"]["per_episode"])
    assert all(e["sham"] and not e["event_exposed"] for e in capcon["sham"]["per_episode"])
    assert any(e["event_exposed"] for e in capcon["failure"]["per_episode"])
    ids = {cell: [e["scenario_id"] for e in report["per_episode"]] for cell, report in capcon.items()}
    assert ids["failure"] == ids["sham"] == ids["failure_comm_off"] == ids["sham_comm_off"]
    assert ids["in_dist"] == ids["in_dist_comm_off"] and ids["in_dist"] != ids["heldout_level"]


def test_the_two_arms_see_identical_panels(tiny, evaluated):
    for cell in mech.applicable_cells("ca_mlp"):
        a = load_report(evaluated, "s5/capcon/seed0", cell["name"])
        b = load_report(evaluated, "s5/ca_mlp/seed0", cell["name"])
        assert a["scenarios_sha256"] == b["scenarios_sha256"] and a["scenarios"] == b["scenarios"]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_tidy_table_matches_independent_recomputation_from_the_reports(tiny, evaluated):
    rows = read_csv(evaluated.summary_dir / "cells.csv")
    assert list(rows[0]) == list(mech.CELL_COLUMNS)
    # capcon: 5+1+2+5+3 + 3 + 5+3+3 compositions; ca_mlp: 5+1+2+5+3 + 3
    assert len(rows) == 30 + 19 and len(evaluated.summary["cells"]) == len(rows)
    assert json.loads((evaluated.summary_dir / "summary.json").read_text())["cells"] == json.loads(
        json.dumps(evaluated.summary["cells"]))
    lookup = {(r["arm"], r["cell"], int(r["num_p"]), int(r["num_a"])): r for r in rows}
    assert len(lookup) == len(rows)
    for arm, relative in (("capcon", "s5/capcon/seed0"), ("ca_mlp", "s5/ca_mlp/seed0")):
        for cell in mech.applicable_cells(arm):
            report = load_report(evaluated, relative, cell["name"])
            horizon = report["config"]["max_steps"]
            for composition in {tuple(e["composition"]) for e in report["per_episode"]}:
                episodes = [e for e in report["per_episode"] if tuple(e["composition"]) == composition]
                row = lookup[(arm, cell["name"], *composition)]
                assert (row["condition"], row["seed"]) == ("s5", "0")
                assert int(row["episodes"]) == len(episodes)
                assert float(row["success_rate"]) == sum(e["success"] for e in episodes) / len(episodes)
                assert float(row["capped_completion"]) == sum(
                    e["steps"] if e["success"] else horizon for e in episodes) / len(episodes)
                assert float(row["mean_team_return"]) == pytest.approx(
                    sum(e["team_return"] for e in episodes) / len(episodes), rel=1e-12, abs=1e-12)
                if cell["panel"] == "failure":
                    assert int(row["event_exposed_episodes"]) == sum(e["event_exposed"] for e in episodes)
                else:
                    assert row["event_exposed_episodes"] == ""


def test_paired_table_matches_the_reports_and_lists_missing_runs(tiny, evaluated):
    rows = read_csv(evaluated.summary_dir / "paired.csv")
    assert list(rows[0]) == list(mech.PAIRED_COLUMNS)
    counts = Counter((r["arm"], r["comparison"]) for r in rows)
    assert counts == {("capcon", "failure_minus_sham"): 4, ("capcon", "in_dist_comm_off_minus_in_dist"): 6,
                      ("capcon", "failure_comm_off_minus_failure"): 4, ("capcon", "sham_comm_off_minus_sham"): 4,
                      ("ca_mlp", "failure_minus_sham"): 4}
    for name, first, second in mech.COMPARISONS:
        a = load_report(evaluated, "s5/capcon/seed0", first)
        b = load_report(evaluated, "s5/capcon/seed0", second)
        horizon = a["config"]["max_steps"]
        pooled = [r for r in rows if r["arm"] == "capcon" and r["comparison"] == name and r["pooled"] == "True"]
        assert len(pooled) == 1 and pooled[0]["num_p"] == "" and int(pooled[0]["pairs"]) == a["episodes"]
        sb = {e["scenario_id"]: e for e in b["per_episode"]}
        success = [int(e["success"]) - int(sb[e["scenario_id"]]["success"]) for e in a["per_episode"]]
        capped = [(e["steps"] if e["success"] else horizon)
                  - (sb[e["scenario_id"]]["steps"] if sb[e["scenario_id"]]["success"] else horizon)
                  for e in a["per_episode"]]
        assert float(pooled[0]["success_difference"]) == sum(success) / len(success)
        assert float(pooled[0]["capped_completion_difference"]) == sum(capped) / len(capped)
    missing = evaluated.summary["missing"]
    assert len(missing) == 68 and all(m["cell"] is None and m["reason"] for m in missing)
    assert {m["index"] for m in missing} == set(range(70)) - set(INDICES)
    assert json.loads((evaluated.summary_dir / "summary.json").read_text())["missing"] == missing


def fake_report(rows, horizon=80):
    return {"config": {"max_steps": horizon}, "per_episode": [
        dict(scenario_id=sid, composition=list(composition), success=success, steps=steps, team_return=1.0,
             event_exposed=False) for sid, composition, success, steps in rows]}


IDENTITY = dict(condition="s5", arm="capcon", seed=0)


def test_cells_compared_with_themselves_have_exactly_zero_paired_difference(tiny, evaluated):
    for cell in SPEC_CELLS:
        report = load_report(evaluated, "s5/capcon/seed0", cell)
        rows = mech.paired_rows(IDENTITY, "self", report, report)
        assert rows and all(r["success_difference"] == 0.0 and r["capped_completion_difference"] == 0.0 for r in rows)
        assert rows[-1]["pooled"] is True and rows[-1]["pairs"] == report["episodes"]


def test_paired_rows_match_by_scenario_id_and_cap_failures_at_the_horizon():
    a = fake_report([("a", (1, 1), True, 10), ("b", (1, 1), False, 80), ("c", (2, 1), True, 20)])
    b = fake_report([("c", (2, 1), False, 80), ("a", (1, 1), True, 14), ("b", (1, 1), False, 80)])
    rows = mech.paired_rows(IDENTITY, "x", a, b)
    assert [(r["num_p"], r["num_a"], r["pooled"], r["pairs"]) for r in rows] == [
        (1, 1, False, 2), (2, 1, False, 1), (None, None, True, 3)]
    assert [r["success_difference"] for r in rows] == [0.0, 1.0, 1 / 3]
    assert [r["capped_completion_difference"] for r in rows] == [-2.0, -60.0, -64.0 / 3]
    assert mech.cell_rows(IDENTITY, {"name": "x", "panel": "failure"}, a)[0]["capped_completion"] == 45.0
    with pytest.raises(ValueError, match="same scenarios"):
        mech.paired_rows(IDENTITY, "x", a, fake_report([("a", (1, 1), True, 10)]))


# --- 4. Safety ---------------------------------------------------------------

def resign(manifest):
    manifest["manifest_sha256"] = study.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def forbid_evaluation(*args, **kwargs):
    pytest.fail("this check must not execute any evaluation")


def test_existing_output_is_refused_and_preserved(tiny, training_run, evaluated, tmp_path, monkeypatch):
    before = report_files(evaluated.output)
    with pytest.raises(FileExistsError):
        mech.prepare(training_run, evaluated.output, allow_missing=True)
    assert report_files(evaluated.output) == before
    link = tmp_path / "linked"
    link.symlink_to(tmp_path)
    with pytest.raises(FileExistsError):
        mech.prepare(training_run, link, allow_missing=True)
    with pytest.raises(FileExistsError):
        mech.summarize(evaluated.output / "manifest.json", evaluated.summary_dir, allow_missing=True)


CORRUPTIONS = [
    "version", "evaluation", "extra_key", "protocol_episodes", "protocol_extra", "panel_name", "panel_path",
    "panel_count", "panel_hash_format", "entry_arm", "entry_dropped", "entry_order", "status",
    "missing_without_reason", "undeclared_missing", "checkpoint_hash_format", "checkpoint_path_absolute",
    "checkpoint_path_parent", "checkpoint_path_other_run", "checkpoint_source", "checkpoint_below_target",
    "cells", "skipped_cells", "training_missing_key", "training_hash_format", "source_hash",
    "source_file_hash", "unsigned_edit",
]


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_tampered_manifests_are_rejected_before_any_output(tiny, evaluated, tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    manifest = json.loads((evaluated.output / "manifest.json").read_text())
    entries, ready = manifest["entries"], manifest["entries"][0]
    if corruption == "version":
        manifest["version"] = 2
    elif corruption == "evaluation":
        manifest["evaluation"] = study.STUDY
    elif corruption == "extra_key":
        manifest["results"] = {"winner": "capcon"}
    elif corruption == "protocol_episodes":
        manifest["protocol"]["episodes_per_composition_per_panel"] = 5
    elif corruption == "protocol_extra":
        manifest["protocol"]["evaluation_completed"] = True
    elif corruption == "panel_name":
        manifest["panels"] = {("x" if k == "in_dist" else k): v for k, v in manifest["panels"].items()}
    elif corruption == "panel_path":
        manifest["panels"]["in_dist"]["path"] = "../in_dist.json"
    elif corruption == "panel_count":
        manifest["panels"]["in_dist"]["scenarios"] = 1
    elif corruption == "panel_hash_format":
        manifest["panels"]["in_dist"]["sha256"] = "not-a-hash"
    elif corruption == "entry_arm":
        entries[7]["arm"] = "capcon"
    elif corruption == "entry_dropped":
        entries.pop()
    elif corruption == "entry_order":
        entries.reverse()
    elif corruption == "status":
        ready["status"] = "pending"
    elif corruption == "missing_without_reason":
        entries[1]["reason"] = ""
    elif corruption == "undeclared_missing":
        manifest["training"]["allow_missing"] = False
    elif corruption == "checkpoint_hash_format":
        ready["checkpoint"]["sha256"] = "0" * 63
    elif corruption == "checkpoint_path_absolute":
        ready["checkpoint"]["path"] = "/" + ready["checkpoint"]["path"]
    elif corruption == "checkpoint_path_parent":
        ready["checkpoint"]["path"] = "../" + ready["checkpoint"]["path"]
    elif corruption == "checkpoint_path_other_run":
        ready["checkpoint"]["path"] = entries[15]["checkpoint"]["path"]
    elif corruption == "checkpoint_source":
        ready["checkpoint"]["source_sha256"] = "0" * 64
    elif corruption == "checkpoint_below_target":
        ready["checkpoint"]["total_steps"] = TINY_TARGET - 1
    elif corruption == "cells":
        ready["cells"] = ready["cells"][:-1]
    elif corruption == "skipped_cells":
        entries[15]["skipped_cells"] = []
    elif corruption == "training_missing_key":
        del manifest["training"]["run_dir"]
    elif corruption == "training_hash_format":
        manifest["training"]["manifest_sha256"] = "x" * 64
    elif corruption == "source_hash":
        manifest["source"]["sha256"] = "0" * 64
    elif corruption == "source_file_hash":
        manifest["source"]["files"][next(iter(manifest["source"]["files"]))] = "not-a-hash"
    if corruption != "unsigned_edit":
        resign(manifest)
    else:
        manifest["source"]["git_status"] = "edited after signing"
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        mech.validate_manifest(manifest)
    for dry_run in (True, False):
        with pytest.raises(ValueError):
            mech.launch(corrupted, 0, dry_run=dry_run)
    with pytest.raises(ValueError):
        mech.summarize(corrupted, tmp_path / "summary", allow_missing=True)
    assert [p.name for p in tmp_path.iterdir()] == ["corrupted.json"]


def test_a_changed_evaluator_source_is_rejected_before_any_output(tiny, evaluated, tmp_path, monkeypatch):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    changed = deepcopy(mech.source_snapshot())
    changed["sha256"] = "f" * 64
    monkeypatch.setattr(mech, "source_snapshot", lambda: changed)
    destination = copy_evaluation(evaluated, tmp_path)
    before = report_files(destination)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="[Ss]ource"):
            mech.launch(destination / "manifest.json", 0, dry_run=dry_run)
    assert report_files(destination) == before


def test_changed_checkpoint_bytes_are_rejected_before_any_evaluation(tiny, training_run, evaluated, tmp_path, monkeypatch):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    training = tmp_path / "training copy"
    shutil.copytree(training_run, training)
    path = training / evaluated.manifest["entries"][0]["checkpoint"]["path"]
    path.write_bytes(path.read_bytes() + b"tampered")
    destination = copy_evaluation(evaluated, tmp_path)
    shutil.rmtree(destination / "reports")
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="Checkpoint changed"):
            mech.launch(destination / "manifest.json", 0, dry_run=dry_run, training_run=training)
    assert not (destination / "reports").exists()
    # The untouched training run still works for the other entry and the manifest-recorded path is the default.
    assert mech.launch(destination / "manifest.json", 15, dry_run=True)["status"] == "dry_run"
    with pytest.raises(FileNotFoundError):
        mech.launch(destination / "manifest.json", 0, dry_run=True, training_run=tmp_path / "absent")


@pytest.mark.parametrize("name", ["in_dist", "failure"])
def test_changed_panel_bytes_are_rejected(tiny, evaluated, tmp_path, monkeypatch, name):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    destination = copy_evaluation(evaluated, tmp_path)
    shutil.rmtree(destination / "reports")
    panel = destination / "panels" / f"{name}.json"
    records = json.loads(panel.read_text())
    records[0]["action_seed"] += 1
    panel.write_text(json.dumps(records, sort_keys=True, indent=1) + "\n")
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="Scenario panel changed"):
            mech.launch(destination / "manifest.json", 0, dry_run=dry_run)
    with pytest.raises(ValueError, match="Scenario panel changed"):
        mech.summarize(destination / "manifest.json", tmp_path / "summary", allow_missing=True)
    assert not (destination / "reports").exists() and not (tmp_path / "summary").exists()


MISMATCHES = ["checkpoint_sha256", "scenarios_sha256", "intervention", "sham", "episodes", "evaluator_source",
              "scenario_order", "intervention_step", "truncated"]


@pytest.mark.parametrize("field", MISMATCHES)
def test_a_mismatching_existing_report_raises_and_is_never_overwritten(tiny, evaluated, tmp_path, monkeypatch, field):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    destination = copy_evaluation(evaluated, tmp_path)
    path = destination / "reports/s5/capcon/seed0/failure_comm_off.json"
    report = json.loads(path.read_text())
    if field == "checkpoint_sha256":
        report["checkpoint_sha256"] = "0" * 64
    elif field == "scenarios_sha256":
        report["scenarios_sha256"] = "0" * 64
    elif field == "intervention":
        report["intervention"] = "none"
    elif field == "sham":
        report["sham"] = True
    elif field == "episodes":
        report["episodes"] += 1
    elif field == "evaluator_source":
        report["evaluator"]["source"]["sha256"] = "0" * 64
    elif field == "scenario_order":
        report["per_episode"].reverse()
    elif field == "intervention_step":
        report["per_episode"][0]["intervention_step"] += 1
    else:
        report["per_episode"].pop()
    path.write_text(json.dumps(report))
    before = report_files(destination)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="[Rr]eport"):
            mech.launch(destination / "manifest.json", 0, dry_run=dry_run)
    with pytest.raises(ValueError, match="[Rr]eport"):
        mech.summarize(destination / "manifest.json", tmp_path / "summary", allow_missing=True)
    assert report_files(destination) == before and not (tmp_path / "summary").exists()


def test_matching_reports_are_kept_and_only_absent_cells_are_evaluated(tiny, evaluated, tmp_path, monkeypatch):
    destination = copy_evaluation(evaluated, tmp_path)
    before = report_files(destination)
    calls = []
    real = mech.evaluate_checkpoint

    def spy(checkpoint, scenarios, output, *args, **kwargs):
        calls.append(Path(output).name)
        return real(checkpoint, scenarios, output, *args, **kwargs)

    monkeypatch.setattr(mech, "evaluate_checkpoint", spy)
    record = mech.launch(destination / "manifest.json", 0)
    assert calls == [] and {row["action"] for row in record["cells"]} == {"kept"}
    assert report_files(destination) == before
    (destination / "reports/s5/capcon/seed0/sham.json").unlink()
    dry = mech.launch(destination / "manifest.json", 0, dry_run=True)
    assert dry["status"] == "dry_run" and calls == []
    assert {row["cell"]: row["action"] for row in dry["cells"]}["sham"] == "would_evaluate"
    assert not (destination / "reports/s5/capcon/seed0/sham.json").exists()
    record = mech.launch(destination / "manifest.json", 0)
    assert calls == ["sham.json"]
    assert {row["cell"]: row["action"] for row in record["cells"]}["sham"] == "evaluated"
    after = report_files(destination)
    assert {k: v for k, v in after.items() if not k.endswith("/sham.json")} == {
        k: v for k, v in before.items() if not k.endswith("/sham.json")}
    shown = json.loads(after["reports/s5/capcon/seed0/sham.json"])
    assert shown["sham"] is True and shown["scenarios_sha256"] == evaluated.manifest["panels"]["failure"]["sha256"]


def test_dry_run_writes_nothing_even_without_reports(tiny, evaluated, tmp_path, monkeypatch):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    destination = copy_evaluation(evaluated, tmp_path)
    shutil.rmtree(destination / "reports")
    before = report_files(destination)
    for index in INDICES:
        record = mech.launch(destination / "manifest.json", index, dry_run=True)
        assert record["status"] == "dry_run" and record["dry_run"] is True
        assert {row["action"] for row in record["cells"]} == {"would_evaluate"}
    assert report_files(destination) == before and not (destination / "reports").exists()


def test_missing_runs_are_recorded_skipped_and_never_dropped(tiny, training_run, evaluated, tmp_path, monkeypatch):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    with pytest.raises(ValueError, match="allow-missing"):
        mech.prepare(training_run, tmp_path / "strict")
    assert not (tmp_path / "strict").exists()
    destination = copy_evaluation(evaluated, tmp_path)
    before = report_files(destination)
    record = mech.launch(destination / "manifest.json", 1)
    assert record["status"] == "skipped" and record["reason"] == "training run directory absent"
    assert record["cells"] == [] and (record["index"], record["arm"]) == (1, "capcon")
    assert mech.launch(destination / "manifest.json", 1, dry_run=True)["status"] == "skipped"
    assert report_files(destination) == before
    with pytest.raises(ValueError, match="allow-missing"):
        mech.summarize(destination / "manifest.json", tmp_path / "summary")
    assert not (tmp_path / "summary").exists()


def test_incomplete_runs_and_unreached_targets_are_missing_not_errors(tiny, training_run, tmp_path, monkeypatch):
    training = tmp_path / "training copy"
    shutil.copytree(training_run, training)
    (training / "training/s5/ca_mlp/seed0/finished.json").unlink()
    manifest = mech.prepare(training, tmp_path / "a", allow_missing=True)
    assert manifest["entries"][0]["status"] == "ready"
    assert manifest["entries"][15]["status"] == "missing" and "finished.json" in manifest["entries"][15]["reason"]
    (training / "training/s5/capcon/seed0/study.json").unlink()
    manifest = mech.prepare(training, tmp_path / "b", allow_missing=True)
    assert "study.json" in manifest["entries"][0]["reason"]
    monkeypatch.setattr(study, "TARGET_STEPS", 10 * TINY_TOTAL)
    manifest = mech.prepare(training_run, tmp_path / "c", allow_missing=True)
    assert [manifest["entries"][i]["status"] for i in INDICES] == ["missing", "missing"]
    assert all("no committed checkpoint reaches" in manifest["entries"][i]["reason"] for i in INDICES)
    with pytest.raises(ValueError, match="allow-missing"):
        mech.prepare(training_run, tmp_path / "d")


def test_a_checkpoint_from_other_source_is_an_error_even_when_missing_runs_are_allowed(tiny, training_run, tmp_path, monkeypatch):
    real = mech.committed_checkpoints
    monkeypatch.setattr(mech, "committed_checkpoints", lambda *a, **k: [
        dict(row, source_sha256="0" * 64) for row in real(*a, **k)])
    with pytest.raises(ValueError, match="source hash"):
        mech.prepare(training_run, tmp_path / "out", allow_missing=True)
    assert not (tmp_path / "out").exists()


def test_a_run_that_does_not_belong_to_the_training_manifest_is_an_error(tiny, training_run, tmp_path, monkeypatch):
    container = tmp_path / "other training"
    shutil.copytree(training_run / "training", container / "training")
    # A second, valid training manifest whose entry configurations differ from what the runs were trained with.
    monkeypatch.setattr(study, "study_config", lambda index: replace(tiny_study_config(index), epochs=13))
    manifest = study.prepare(container / "manifest.json")
    with pytest.raises(ValueError, match="does not belong"):
        mech.prepare(container, tmp_path / "out", allow_missing=True)
    for index in INDICES:
        path = container / "training" / manifest["entries"][index]["relative_output"] / "study.json"
        path.write_text(json.dumps(dict(json.loads(path.read_text()), manifest_sha256=manifest["manifest_sha256"])))
    with pytest.raises(ValueError, match="configuration differs"):
        mech.prepare(container, tmp_path / "out", allow_missing=True)
    assert not (tmp_path / "out").exists()


def test_summarize_lists_absent_reports_and_fails_without_allow_missing(tiny, evaluated, tmp_path):
    destination = copy_evaluation(evaluated, tmp_path)
    (destination / "reports/s5/capcon/seed0/sham.json").unlink()
    with pytest.raises(ValueError, match="allow-missing"):
        mech.summarize(destination / "manifest.json", tmp_path / "strict")
    assert not (tmp_path / "strict").exists()
    result = mech.summarize(destination / "manifest.json", tmp_path / "lenient", allow_missing=True)
    absent = [m for m in result["missing"] if m["cell"] is not None]
    assert absent == [dict(condition="s5", arm="capcon", seed=0, index=0, cell="sham", reason="report not found")]
    comparisons = {r["comparison"] for r in result["paired"] if r["arm"] == "capcon"}
    assert comparisons == {"in_dist_comm_off_minus_in_dist", "failure_comm_off_minus_failure"}
    assert not any(r["cell"] == "sham" and r["arm"] == "capcon" for r in result["cells"])


def test_report_outputs_cannot_escape_the_evaluation_directory(tiny, evaluated, tmp_path, monkeypatch):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    destination, outside = copy_evaluation(evaluated, tmp_path), tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(destination / "reports")
    (destination / "reports").mkdir()
    (destination / "reports/s5").symlink_to(outside, target_is_directory=True)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="inside the evaluation output"):
            mech.launch(destination / "manifest.json", 0, dry_run=dry_run)
    assert list(outside.iterdir()) == []
    (destination / "reports/s5").unlink()
    leaf = destination / "reports/s5/capcon/seed0"
    leaf.mkdir(parents=True)
    (leaf / "in_dist.json").symlink_to(destination / "manifest.json")
    for dry_run in (True, False):
        with pytest.raises(FileExistsError, match="symlink"):
            mech.launch(destination / "manifest.json", 0, dry_run=dry_run)
    (leaf / "in_dist.json").unlink()
    (leaf / "in_dist.json").symlink_to(outside / "elsewhere.json")
    with pytest.raises(ValueError, match="inside the evaluation output"):
        mech.launch(destination / "manifest.json", 0, dry_run=True)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("index", [-1, 70, True, 0., "0", None])
def test_launch_rejects_bad_indices(tiny, evaluated, monkeypatch, index):
    monkeypatch.setattr(mech, "evaluate_checkpoint", forbid_evaluation)
    with pytest.raises(ValueError, match="index"):
        mech.launch(evaluated.output / "manifest.json", index, dry_run=True)


def test_cli_prepare_launch_and_summarize(tiny, training_run, evaluated, tmp_path, capsys):
    mech.main(["launch", "--manifest", str(evaluated.output / "manifest.json"), "--index", "15", "--dry-run",
               "--training-run", str(training_run)])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run" and result["index"] == 15 and len(result["cells"]) == 6
    mech.main(["launch", "--manifest", str(evaluated.output / "manifest.json"), "--index", "3"])
    assert json.loads(capsys.readouterr().out)["status"] == "skipped"
    mech.main(["summarize", "--manifest", str(evaluated.output / "manifest.json"),
               "--output", str(tmp_path / "summary"), "--allow-missing"])
    assert json.loads(capsys.readouterr().out)["summary"] == str(tmp_path / "summary")
    mech.main(["prepare", "--training-run", str(training_run), "--output", str(tmp_path / "new"), "--allow-missing"])
    assert json.loads(capsys.readouterr().out)["evaluation"] == "capability_mechanism_evaluation_v1"
    for argv in (["prepare", "--training-run", str(training_run), "--output", str(tmp_path / "strict")],
                 ["prepare", "--training-run", str(training_run), "--output", str(evaluated.output)],
                 ["summarize", "--manifest", str(tmp_path / "nothing.json"), "--output", str(tmp_path / "x")]):
        with pytest.raises(SystemExit) as error:
            mech.main(argv)
        assert error.value.code == 2
    assert not (tmp_path / "strict").exists() and not (tmp_path / "x").exists()


@pytest.mark.parametrize("extra", [["--episodes", "5"], ["--arm", "cash"], ["--cell", "sham"], ["--seed", "1"],
                                   ["--output", "elsewhere"], ["--train"], ["--allow-missing"]])
def test_launch_cli_has_no_scientific_overrides(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        mech.main(["launch", "--manifest", str(tmp_path / "unused.json"), "--index", "0", "--dry-run", *extra])
    assert error.value.code == 2


@pytest.mark.parametrize("index", ["-1", "70", "1.5", "true", ""])
def test_cli_rejects_bad_indices(tmp_path, index):
    with pytest.raises(SystemExit) as error:
        mech.main(["launch", "--manifest", str(tmp_path / "unused.json"), "--index", index])
    assert error.value.code == 2


def test_the_study_modules_and_their_constants_are_unchanged_by_this_stage():
    assert (study.STUDY, study.TARGET_STEPS, study.COUNT) == ("capability_mechanism_study_v2", 10_000_000, 70)
    assert study.CONDITIONS == (("s5", 5), ("s6", 6)) and study.SEEDS == (0, 1, 2, 3, 4)
    assert study.study_config(0).total_steps == 10_000_000 and not hasattr(study, "EPISODES")


# --- 5. submit_stokes.py routing and mocked-srun shell tests -------------------

@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "slurm").mkdir(parents=True)
    for path in (EVAL, PREPARE):
        (root / "slurm" / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(submit, "ROOT", root)
    monkeypatch.chdir(root)
    for name in set(key for routes in submit.ENV_ROUTES.values() for key in routes):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SOFTROLE_MECHANISM_TRAINING_RUN", raising=False)
    return root


@pytest.mark.parametrize("job,extra", [("softrole_mechanism_eval_prepare.sbatch", []),
                                       ("softrole_mechanism_eval.sbatch", ["--array=0-69%20"])])
def test_submit_stokes_dry_run_accepts_both_batch_scripts(checkout, capsys, job, extra):
    args = ["--dry-run", "--title", "Capability mechanism study evaluation",
            "--purpose", "Preview launcher routing only.", "--", *extra, "slurm/" + job]
    assert submit.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    run_dir = Path(plan["run_dir"])
    assert plan["output_environment"]["SOFTROLE_MECHANISM_EVAL_DIR"] == str(run_dir / "evaluation")
    assert "SOFTROLE_MECHANISM_TRAINING_RUN" not in plan["output_environment"]
    assert plan["command"][-1] == str(checkout / "slurm" / job)
    assert not (checkout / "runs").exists()


@pytest.mark.parametrize("job", ["softrole_mechanism_eval_prepare.sbatch", "softrole_mechanism_eval.sbatch"])
def test_submit_stokes_cannot_route_the_evaluation_outside_the_run(checkout, monkeypatch, job):
    args = ["--title", "Capability mechanism study evaluation", "--purpose", "Containment check only.",
            "--", "slurm/" + job]
    monkeypatch.setenv("SOFTROLE_MECHANISM_EVAL_DIR", "/tmp/outside")
    assert submit.main(args) == 2
    assert not (checkout / "runs").exists()


def capture_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"names = {(*THREADS, 'PYTHONPATH', 'PYTHONUNBUFFERED', 'DGLBACKEND')!r}\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['PYTHON_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': args, 'cwd': os.getcwd(), 'python': sys.argv[0], "
        "'environment': {name: os.environ.get(name) for name in names}}) + '\\n')\n"
        "assert args[:3] == ['-u', '-m', 'softrole.mechanism_evaluation']\n"
        "operation = args[3]\n"
        "key = operation if operation == 'prepare' else 'launch:' + args[args.index('--index') + 1]\n"
        "if os.environ.get('MOCK_FAIL') == key: sys.exit(17)\n"
        "if operation == 'prepare':\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.mkdir(parents=True)\n"
        "    (output / 'manifest.json').write_text('{\"mock_manifest\": true}\\n')\n"
        "else:\n"
        "    assert pathlib.Path(args[args.index('--manifest') + 1]).is_file()\n"
    )
    path.chmod(0o755)
    return path


def setup_environment(tmp_path, *, index="0", prepared=False, defaults=False):
    root = tmp_path / "checkout"
    root.mkdir()
    tools = tmp_path / "mock-tools"
    tools.mkdir()
    for name, code in {
        "module": '#!/bin/bash\nprintf "%s\\n" "$*" > "$MODULE_CAPTURE"\n',
        "srun": '#!/bin/bash\n[[ $1 == --cpu-bind=cores ]] || exit 3\nshift\nexec "$@"\n',
    }.items():
        file = tools / name
        file.write_text(code)
        file.chmod(0o755)
    env = dict(os.environ)
    for name in ("SOFTROLE_MECHANISM_EVAL_DIR", "SOFTROLE_MECHANISM_TRAINING_RUN", "HETNET_PYTHON", "MOCK_FAIL"):
        env.pop(name, None)
    env.update({name: "32" for name in THREADS})
    env.update(PATH=f"{tools}:{env['PATH']}", PYTHONPATH="inherited-path", PYTHONUNBUFFERED="0", DGLBACKEND="other",
               MODULE_CAPTURE=str(tmp_path / "module.txt"), PYTHON_CAPTURE=str(tmp_path / "python.jsonl"),
               SLURM_SUBMIT_DIR=str(root), SLURM_ARRAY_TASK_ID=index, SLURM_ARRAY_JOB_ID="4242")
    env = configure_stokes_run(env, tmp_path)
    training = tmp_path / "training run"
    training.mkdir()
    env["SOFTROLE_MECHANISM_TRAINING_RUN"] = str(training)
    container = Path(env["HETNET_RUN_DIR"])
    if defaults:
        capture_python(root / ".venv/bin/python")
        evaluation = container / "evaluation"
    else:
        env["HETNET_PYTHON"] = str(capture_python(tmp_path / "capture python"))
        evaluation = container / "evaluation dir"
        env["SOFTROLE_MECHANISM_EVAL_DIR"] = str(evaluation)
    if prepared:
        evaluation.mkdir(parents=True)
        (evaluation / "manifest.json").write_text('{"existing": "preserve these bytes"}\n')
    return root, evaluation, training, env


def run_slurm(script, arguments, tmp_path, env):
    return subprocess.run(["bash", str(script), *arguments], cwd=tmp_path, env=env, text=True, capture_output=True)


def calls(env):
    path = Path(env["PYTHON_CAPTURE"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_environment(call, root):
    assert call["cwd"] == str(root)
    assert all(call["environment"][name] == "1" for name in THREADS)
    assert call["environment"]["PYTHONPATH"] == f"{root}/src:{root}/src/envs:inherited-path"
    assert call["environment"]["PYTHONUNBUFFERED"] == "1"
    assert call["environment"]["DGLBACKEND"] == "pytorch"


def launch_args(evaluation, index, training, *, dry_run=False):
    result = ["-u", "-m", "softrole.mechanism_evaluation", "launch", "--manifest", str(evaluation / "manifest.json"),
              "--index", str(index), "--training-run", str(training)]
    return result + (["--dry-run"] if dry_run else [])


def prepare_args(evaluation, training, *extra):
    return ["-u", "-m", "softrole.mechanism_evaluation", "prepare", "--training-run", str(training),
            "--output", str(evaluation), *extra]


@pytest.mark.parametrize("index", [0, 9, 10, 34, 35, 69])
@pytest.mark.parametrize("dry_run", [False, True])
def test_evaluation_slurm_invokes_python_directly_with_the_index(tmp_path, index, dry_run):
    root, evaluation, training, env = setup_environment(tmp_path, index=str(index), prepared=True)
    result = run_slurm(EVAL, ["--dry-run"] if dry_run else [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)
    assert len(observed) == 1
    assert observed[0]["args"] == launch_args(evaluation, index, training, dry_run=dry_run)
    assert_environment(observed[0], root)
    assert Path(env["MODULE_CAPTURE"]).read_text() == "load anaconda/anaconda-2024.10\n"
    assert not (evaluation / "reports").exists()


def test_evaluation_slurm_default_paths_and_interpreter(tmp_path):
    root, evaluation, training, env = setup_environment(tmp_path, prepared=True, defaults=True)
    result = run_slurm(EVAL, ["--dry-run"], tmp_path, env)
    assert result.returncode == 0, result.stderr
    observed = calls(env)[0]
    assert observed["args"] == launch_args(evaluation, 0, training, dry_run=True)
    assert observed["python"] == str(root / ".venv/bin/python")
    assert_environment(observed, root)


@pytest.mark.parametrize("index", ["70", "-1", "00", "01", "x", "6a", "100", "6 9"])
def test_evaluation_slurm_rejects_invalid_array_indices_before_execution(tmp_path, index):
    _, _, _, env = setup_environment(tmp_path, index=index, prepared=True)
    result = run_slurm(EVAL, [], tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


@pytest.mark.parametrize("args", [
    ["--dry"], ["--dry-run=true"], ["--dry-run", "--dry-run"], ["--allow-missing"], ["--manifest", "other.json"],
    ["--training-run", "other"], ["--output", "other"], ["--episodes", "1"], ["--cell", "sham"],
    ["--total-steps", "1"], ["--seed", "1"], ["--model", "cash"],
])
def test_evaluation_slurm_rejects_all_overrides_before_execution(tmp_path, args):
    _, _, _, env = setup_environment(tmp_path, prepared=True)
    result = run_slurm(EVAL, args, tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


def test_evaluation_slurm_requires_a_prepared_manifest(tmp_path):
    _, _, _, env = setup_environment(tmp_path)
    result = run_slurm(EVAL, ["--dry-run"], tmp_path, env)
    assert result.returncode == 2 and "not found" in result.stderr and not calls(env)


@pytest.mark.parametrize("script,arguments", [(EVAL, ["--dry-run"]), (PREPARE, [])])
def test_training_run_variable_must_name_an_existing_directory_other_than_the_run(tmp_path, script, arguments):
    _, _, training, env = setup_environment(tmp_path, prepared=True)
    absent = dict(env)
    del absent["SOFTROLE_MECHANISM_TRAINING_RUN"]
    result = run_slurm(script, arguments, tmp_path, absent)
    assert result.returncode != 0 and "SOFTROLE_MECHANISM_TRAINING_RUN" in result.stderr and not calls(env)
    for value, message in ((str(tmp_path / "no such run"), "not found"), (env["HETNET_RUN_DIR"], "own run directory")):
        changed = dict(env, SOFTROLE_MECHANISM_TRAINING_RUN=value)
        result = run_slurm(script, arguments, tmp_path, changed)
        assert result.returncode == 2 and message in result.stderr and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


@pytest.mark.parametrize("defaults", [False, True])
@pytest.mark.parametrize("extra", [[], ["--allow-missing"]])
def test_preparation_creates_the_evaluation_then_previews_all_seventy_entries(tmp_path, defaults, extra):
    root, evaluation, training, env = setup_environment(tmp_path, defaults=defaults)
    result = run_slurm(PREPARE, extra, tmp_path, env)
    assert result.returncode == 0, result.stderr
    expected = [prepare_args(evaluation, training, *extra)]
    expected += [launch_args(evaluation, index, training, dry_run=True) for index in range(70)]
    observed = calls(env)
    assert [call["args"] for call in observed] == expected
    for call in observed:
        assert_environment(call, root)
    assert (evaluation / "manifest.json").is_file() and not (evaluation / "reports").exists()


def test_an_existing_evaluation_is_validated_by_all_previews_and_never_rewritten(tmp_path):
    _, evaluation, training, env = setup_environment(tmp_path, prepared=True)
    before = (evaluation / "manifest.json").read_bytes()
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert [call["args"] for call in calls(env)] == [launch_args(evaluation, i, training, dry_run=True)
                                                     for i in range(70)]
    assert (evaluation / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("failure, expected_count", [("prepare", 1), ("launch:0", 2), ("launch:69", 71)])
def test_preparation_stops_on_the_first_failed_step(tmp_path, failure, expected_count):
    _, _, _, env = setup_environment(tmp_path)
    env["MOCK_FAIL"] = failure
    result = run_slurm(PREPARE, [], tmp_path, env)
    assert result.returncode == 17 and len(calls(env)) == expected_count


@pytest.mark.parametrize("args", [["--dry-run"], ["--output", "other"], ["--total-steps", "1"],
                                  ["--allow-missing", "--allow-missing"], ["--training-run", "other"]])
def test_preparation_rejects_arguments_before_execution(tmp_path, args):
    _, _, _, env = setup_environment(tmp_path)
    result = run_slurm(PREPARE, args, tmp_path, env)
    assert result.returncode == 2 and not calls(env)
    assert not Path(env["MODULE_CAPTURE"]).exists()


@pytest.mark.parametrize("script,arguments", [(EVAL, ["--dry-run"]), (PREPARE, [])])
@pytest.mark.parametrize("symlink", [False, True])
def test_slurm_refuses_an_evaluation_directory_outside_the_run_container(script, arguments, symlink, tmp_path):
    _, evaluation, _, env = setup_environment(tmp_path, prepared=True)
    outside = tmp_path / "outside run"
    outside.mkdir()
    (outside / "manifest.json").write_text('{"outside": "preserve"}\n')
    route = outside
    if symlink:
        route = Path(env["HETNET_RUN_DIR"]) / "escaped output"
        route.symlink_to(outside, target_is_directory=True)
    env["SOFTROLE_MECHANISM_EVAL_DIR"] = str(route)
    before = (evaluation / "manifest.json").read_bytes()
    result = run_slurm(script, arguments, tmp_path, env)
    assert result.returncode != 0 and "must stay inside" in result.stderr
    assert not calls(env) and not Path(env["MODULE_CAPTURE"]).exists()
    assert (evaluation / "manifest.json").read_bytes() == before
    assert (outside / "manifest.json").read_text() == '{"outside": "preserve"}\n'


def test_slurm_scripts_source_the_containment_helper_before_any_python():
    for path in (EVAL, PREPARE):
        text = path.read_text()
        assert 'stokes_run_output "$eval_dir"' in text
        assert text.index("stokes_run_output") < text.index("module load") < text.index("srun")


def test_slurm_shell_syntax_and_declared_resources():
    result = subprocess.run(["bash", "-n", str(EVAL), str(PREPARE)], capture_output=True)
    assert result.returncode == 0, result.stderr
    for path, extra in ((EVAL, ("--cpus-per-task=1", "--mem=8G", "--time=06:00:00", "--array=0-69%20")),
                        (PREPARE, ("--cpus-per-task=1", "--mem=4G", "--time=02:00:00"))):
        text = path.read_text()
        for setting in ("--account=cenyioha", "--partition=normal", "--constraint=sapphirerapids", *extra):
            assert f"#SBATCH {setting}\n" in text
        assert "softrole.mechanism_evaluation" in text and "SOFTROLE_MECHANISM_" in text
        assert "graded_pilot" not in text.lower() and "mechanism_study " not in text
        # Every interpreter call runs inside the batch step under srun; none runs at submission time.
        assert text.count('"$HETNET_PYTHON"') == text.count('srun --cpu-bind=cores "$HETNET_PYTHON"') >= 1


# --- Runbook ------------------------------------------------------------------

def test_runbook_exists_is_listed_in_gitignore_and_states_the_commands():
    text = RUNBOOK.read_text()
    assert "!/src/softrole/MECHANISM_EVALUATION.md" in (ROOT / ".gitignore").read_text()
    for needle in ("SOFTROLE_MECHANISM_TRAINING_RUN", "softrole_mechanism_eval_prepare.sbatch",
                   "softrole_mechanism_eval.sbatch", "afterok", "mechanism_evaluation summarize", "--init-only",
                   "capability_mechanism_evaluation_v1", "failure_minus_sham", "--allow-missing"):
        assert needle in text
    for name in SPEC_CELLS:
        assert f"`{name}`" in text
