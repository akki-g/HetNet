"""Contracts for reading concurrently-written scientific evidence without changing it."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from hetnet_ext.grid import load_grid
from hetnet_ext.progress import curves_svg, main, read_jsonl, snapshot, window_metrics


def metric(epoch, success=.1, episodes=10):
    return dict(epoch=epoch, steps=100, episodes=episodes, total_steps=100 * epoch,
                total_episodes=episodes * epoch, success_rate=success, steps_taken=75,
                reward_per_agent=[-1, -3, 2], policy_loss=-.2, value_loss=1.4,
                wall_time_seconds=60)


def run(tmp_path, rows, status="starting"):
    directory = tmp_path / "2P1A/seed0"
    directory.mkdir(parents=True)
    (directory / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "config.json").write_text(json.dumps(dict(entry=load_grid()[0].as_dict(), mode="train", num_epochs=2000)))
    (directory / "provenance.json").write_text(json.dumps(dict(status=status)))
    return directory


def test_weighted_window_not_mean_of_epoch_means():
    rows = [metric(1, 0, 1), metric(2, 1, 9)]
    actual = window_metrics(rows, 2)
    assert actual["success_rate"] == .9
    assert actual["reward_P"] == -2
    assert actual["reward_A"] == 2


def test_live_partial_line_ignored_but_committed_corruption_rejected(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(metric(1)) + '\n{"epoch":2')
    rows, warnings = read_jsonl(path)
    assert len(rows) == 1 and warnings
    path.write_text('{"epoch":broken}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_expected_grid_eta_counters_and_read_only(tmp_path):
    directory = run(tmp_path, [metric(1), metric(2)])
    (directory / "checkpoint_records.jsonl").write_text('{"epoch":1,"wall_time_seconds":2}\n')
    before = {p: p.read_bytes() for p in directory.iterdir()}
    report, _ = snapshot(tmp_path)
    assert len(report["runs"]) == 21
    assert sum(r["status"] == "not_seen" for r in report["runs"]) == 20
    row = report["runs"][0]
    assert row["status"] == "incomplete_check_slurm"
    assert row["total_steps"] == 200 and row["total_episodes"] == 20
    assert row["eta_hours"] == pytest.approx((1998 * 60 + 40 * 2) / 3600)
    assert {p: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("change", [dict(epoch=3), dict(success_rate=math.nan), dict(reward_per_agent=[0]), dict(total_steps=900), dict(steps_taken=81)])
def test_corrupt_metrics_are_reported_without_hiding_other_runs(tmp_path, change):
    row = metric(1)
    row.update(change)
    run(tmp_path, [row])
    report, series = snapshot(tmp_path)
    assert report["runs"][0]["status"] == "invalid_evidence"
    assert len(report["runs"]) == 21
    assert series[0][1] == []


def test_failed_and_false_complete_are_not_success(tmp_path):
    directory = run(tmp_path, [metric(1)], status="failed")
    report, _ = snapshot(tmp_path)
    assert report["runs"][0]["eta_hours"] is None
    assert report["runs"][0]["status"] == "failed"
    (directory / "provenance.json").write_text('{"status":"complete"}')
    report, _ = snapshot(tmp_path)
    assert report["runs"][0]["status"] == "invalid_evidence"
    assert report["complete_runs"] == 0


def test_fixed_review_windows_and_export(tmp_path):
    root = tmp_path / "runs"
    run(root, [metric(e, .1 if e <= 50 else .7) for e in range(1, 301)])
    report, series = snapshot(root, window=10)
    review = report["runs"][0]["early_review"]
    assert review["success_change"] == pytest.approx(.6)
    assert not review["no_success_increase"]
    assert report["source_seeds_at_review"] == 1
    assert not report["source_review_due"]
    ET.fromstring(curves_svg(series, 50))
    output = tmp_path / "snapshot"
    assert main(["--runs", str(root), "--out", str(output)]) == 0
    assert {p.name for p in output.iterdir()} == {"report.json", "progress.txt", "summary.csv", "learning_curves.svg"}
    assert len((output / "summary.csv").read_text().splitlines()) == 22
    with pytest.raises(FileExistsError):
        main(["--runs", str(root), "--out", str(output)])


def test_identity_and_stale_metrics(tmp_path):
    directory = run(tmp_path, [metric(1)])
    report, _ = snapshot(tmp_path, now=(directory / "metrics.jsonl").stat().st_mtime + 1000)
    assert any("stale" in w for w in report["runs"][0]["warnings"])
    wrong = json.loads((directory / "config.json").read_text())
    wrong["entry"]["seed"] = 9
    (directory / "config.json").write_text(json.dumps(wrong))
    report, _ = snapshot(tmp_path)
    assert report["runs"][0]["status"] == "invalid_evidence"


@pytest.mark.parametrize("filename", ["metrics.jsonl", "provenance.json", "config.json", "checkpoint_records.jsonl"])
def test_wrong_json_type_is_isolated_to_one_run(tmp_path, filename):
    directory = run(tmp_path, [metric(1)])
    (directory / filename).write_text("null\n")
    report, _ = snapshot(tmp_path)
    assert report["runs"][0]["status"] == "invalid_evidence"
    assert len(report["runs"]) == 21
