"""Progress summaries preserve task-specific horizons and recorder denominators."""
import csv
import json
import math

import pytest

from hetnet_ext.progress import main, read_jsonl, render_table, snapshot, window_metrics


def metric(epoch=1, success=.1, episodes=10, **changes):
    row = dict(epoch=epoch, steps=100, episodes=episodes, total_steps=100 * epoch,
               total_episodes=episodes * epoch, success_rate=success, steps_taken=75,
               reward_per_agent=[-1, -3, 2], policy_loss=-.2, value_loss=1.4,
               wall_time_seconds=60)
    row.update(changes)
    return row


def make_run(root, rows, name="pcp_real/seed0", exit_code=None, **arguments):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = dict(nfriendly_P=2, nfriendly_A=1, num_epochs=2, max_steps=80)
    args.update(arguments)
    (directory / "resolved_args.json").write_text(json.dumps(args))
    if exit_code is not None:
        (directory / "exit_code.txt").write_text(f"{exit_code}\n")
    return directory


def test_unequal_episode_and_step_denominators():
    rows = [metric(success=0, episodes=1, policy_loss=2),
            metric(2, success=1, episodes=9, steps=300, policy_loss=6,
                   reward_per_agent=[3, 5, 10], steps_taken=25)]
    actual = window_metrics(rows, 2)
    assert actual["success_rate"] == .9
    assert actual["steps_taken"] == 30
    assert actual["reward_P"] == pytest.approx(3.4)
    assert actual["reward_A"] == pytest.approx(9.2)
    assert actual["policy_loss"] == 5  # joint-step weighting, not episode weighting


def test_discovery_pp_no_a_and_read_only(tmp_path):
    directory = make_run(tmp_path, [metric(reward_per_agent=[-1, -3])],
                         name="pp_real/seed8", nfriendly_A=0)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    report = snapshot(tmp_path)
    assert len(report["runs"]) == 1
    run = report["runs"][0]
    assert run["run"] == "pp_real/seed8"
    assert run["recent"]["reward_A"] is None
    assert run["recent"]["reward_P"] == -2
    assert run["total_steps"] == 100 and run["total_episodes"] == 10
    assert run["status"] == "running_or_interrupted"
    assert "-" in render_table(report)
    assert {p: p.read_bytes() for p in directory.iterdir()} == before


def test_fc_horizon_and_recent_window(tmp_path):
    make_run(tmp_path, [metric(steps_taken=300), metric(2, steps_taken=200, wall_time_seconds=20)],
             name="fc_binary/seed1", max_steps=300, num_epochs=10000)
    run = snapshot(tmp_path, window=1)["runs"][0]
    assert run["target_epochs"] == 10000
    assert run["recent"]["steps_taken"] == 200
    assert run["seconds_per_epoch"] == 20
    assert run["status"] != "invalid"


def test_unfinished_line_ignored_but_committed_corruption_isolated(tmp_path):
    bad = make_run(tmp_path, [metric()])
    good = make_run(tmp_path, [metric()], name="fc_real/seed0", exit_code=0, num_epochs=1)
    metrics = bad / "metrics.jsonl"
    complete = metrics.read_text()
    metrics.write_text(complete + '{"epoch":2')
    rows, warnings = read_jsonl(metrics)
    assert len(rows) == 1 and warnings
    report = snapshot(tmp_path)
    assert report["complete_runs"] == 1
    assert report["runs"][1]["epoch"] == 1
    metrics.write_text(complete + '{"epoch":broken}\n')
    report = snapshot(tmp_path)
    assert report["complete_runs"] == 1
    assert report["runs"][0]["path"] == str(good)
    assert report["runs"][1]["status"] == "invalid"
    assert main(["--runs", str(tmp_path)]) == 1


@pytest.mark.parametrize("exit_code,epochs,expected", [(0, 2, "complete"), (0, 3, "incomplete"),
                                                       (7, 2, "failed"), (None, 2, "running_or_interrupted")])
def test_completion_requires_exit_zero_and_all_epochs(tmp_path, exit_code, epochs, expected):
    make_run(tmp_path, [metric(), metric(2)], exit_code=exit_code, num_epochs=epochs)
    report = snapshot(tmp_path)
    assert report["runs"][0]["status"] == expected
    assert report["complete_runs"] == int(expected == "complete")


@pytest.mark.parametrize("change", [dict(epoch=3), dict(success_rate=math.nan), dict(reward_per_agent=[0]),
                                    dict(total_steps=900), dict(steps_taken=301)])
def test_invalid_metrics(tmp_path, change):
    make_run(tmp_path, [metric(**change)], max_steps=300)
    assert snapshot(tmp_path)["runs"][0]["status"] == "invalid"


@pytest.mark.parametrize("filename", ["metrics.jsonl", "resolved_args.json"])
def test_wrong_json_type(tmp_path, filename):
    directory = make_run(tmp_path, [metric()])
    (directory / filename).write_text("null\n")
    assert snapshot(tmp_path)["runs"][0]["status"] == "invalid"


def test_csv_and_json_output(tmp_path, capsys):
    root = tmp_path / "runs"
    make_run(root, [metric(reward_per_agent=[-1, -3])], name="pp_binary/seed2",
             nfriendly_A=0, num_epochs=1, exit_code=0)
    out = tmp_path / "report"
    assert main(["--runs", str(root), "--json", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["complete_runs"] == 1
    with (out / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["reward_A"] == ""
    assert rows[0]["status"] == "complete" and rows[0]["total_steps"] == "100"


def test_empty_root_and_invalid_window(tmp_path):
    assert snapshot(tmp_path)["runs"] == []
    with pytest.raises(ValueError, match="positive"):
        snapshot(tmp_path, window=0)


def test_startup_failure_is_visible_without_metrics_or_arguments(tmp_path):
    for name, code in (("pp_real/seed0", 1), ("pp_real/seed1", None), ("pp_real/seed2", 0)):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        (directory / "command.txt").write_text("python main.py ...\n")
        if code is not None:
            (directory / "exit_code.txt").write_text(f"{code}\n")
    report = snapshot(tmp_path)
    assert [r["status"] for r in report["runs"]] == ["failed", "running_or_interrupted", "incomplete"]
    assert report["complete_runs"] == 0
    assert report["runs"][0]["exit_code"] == 1
    assert "stdout.log" in report["runs"][0]["warnings"][0]
    assert "checkpoints are not verified" in render_table(report)
