"""Stability-gate mathematics and independently audited synthetic evidence.

The ledger fixture is fabricated engineering data, never a training result. It
uses the actual locked 2M-step config, all four collectors, 80-step episodes,
indexed gzip records and a real signed float64 checkpoint.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import pytest
import torch

from hetnet_ext.signatures import model_signature
from softrole import CHECKPOINT_VERSION
from softrole import stale_capability as gate
from softrole.config import Config
from softrole.ledger import ChunkedEpisodes, checkpoint_commit, sha
from softrole.model import SoftRoleNet
from softrole.optimizers import make_optimizer, optimizer_spec


def test_sliding_window_checks_expiration_between_episode_completions():
    # The only failed window ends at 12, when the success at 2 expires. A
    # checker looking only at episode completions would incorrectly pass.
    result = gate._gate_windows([2, 5, 8, 11, 14, 17, 20],
                                [True, False, False, False, True, True, True],
                                final_step=20, start=0, width=10, threshold=0.1)
    assert result["windows_checked"] == 8
    assert result["violating_windows"] == 1
    assert result["worst_window"] == {
        "start_exclusive": 2, "end_inclusive": 12, "episodes": 3,
        "successes": 0, "success_rate": 0.,
    }


def test_window_left_boundary_is_excluded_right_included_and_equality_passes():
    result = gate._gate_windows([5, 10, 15, 20], [True, False, True, False],
                                final_step=20, start=10, width=10, threshold=0.5)
    assert result["windows_checked"] == 1 and result["violating_windows"] == 0
    assert result["worst_window"] == {
        "start_exclusive": 10, "end_inclusive": 20, "episodes": 2,
        "successes": 1, "success_rate": 0.5,
    }


def test_window_sweep_matches_exhaustive_integer_windows():
    rng = random.Random(417)
    for _ in range(150):
        completions, successes, step = [], [], 0
        for _ in range(25):
            step += rng.randint(1, 3)
            completions.append(step)
            successes.append(bool(rng.randrange(2)))
        start, width = rng.randint(0, 10), rng.randint(4, 12)
        threshold = rng.choice((0.05, 0.25, 0.5))
        windows = []
        for end in range(start + width, step + 1):
            selected = [success for completion, success in zip(completions, successes)
                        if end - width < completion <= end]
            windows.append(dict(start_exclusive=end - width, end_inclusive=end,
                                episodes=len(selected), successes=sum(selected),
                                success_rate=sum(selected) / len(selected)))
        actual = gate._gate_windows(completions, successes, step, start, width, threshold)
        assert actual["worst_window"] == min(windows, key=lambda row: row["success_rate"])
        assert bool(actual["violating_windows"]) == any(
            row["success_rate"] < threshold for row in windows)


@pytest.fixture
def small_ledgers(monkeypatch):
    # Scale only pure-helper numerical boundaries. The production manifest and
    # integration fixture below retain every confirmed protocol value.
    monkeypatch.setattr(gate, "GATE_BUDGET", 30)
    monkeypatch.setattr(gate, "GATE_WARMUP_STEPS", 10)
    monkeypatch.setattr(gate, "GATE_WINDOW_STEPS", 10)
    episodes = [
        dict(update=1, steps=10, num_agents=2, success=True),
        dict(update=2, steps=4, num_agents=2, success=True),
        dict(update=2, steps=6, num_agents=3, success=True),
        dict(update=3, steps=10, num_agents=3, success=True),
    ]
    updates = [dict(update=1, total_steps=10, action_entropy_sum=0.),
               dict(update=2, total_steps=20, action_entropy_sum=0.26),
               dict(update=3, total_steps=30, action_entropy_sum=0.3)]
    return updates, episodes


def test_entropy_uses_episode_agent_steps_with_mixed_rosters(small_ledgers):
    updates, episodes = small_ledgers
    updates[1]["action_entropy_sum"] = 0.255
    result = gate._gate_statistics(updates, episodes)
    # 4*2 + 6*3 = 26; an unweighted average of roster sizes gives the wrong
    # denominator 25 and would turn this collapsed update into a pass.
    assert result["status"] == "collapsed" and result["collapsed"] is True
    assert result["minimum_entropy"]["agent_steps"] == 26
    assert result["minimum_entropy"]["mean_action_entropy"] == 0.255 / 26
    assert [row["update"] for row in result["entropy_violations"]] == [2]


def test_entropy_at_warmup_endpoint_is_ignored_and_threshold_equality_passes(small_ledgers):
    result = gate._gate_statistics(*small_ledgers)
    assert result["status"] == "pass" and result["collapsed"] is False
    assert result["entropy_updates_checked"] == 2
    assert result["minimum_entropy"]["mean_action_entropy"] == gate.GATE_MIN_ENTROPY
    assert result["entropy_violations"] == []


def test_incomplete_budget_never_yields_a_pass_or_collapse_decision(small_ledgers):
    updates, episodes = small_ledgers
    updates[1]["action_entropy_sum"] = 0.
    result = gate._gate_statistics(updates[:-1], episodes[:-1])
    assert result["status"] == "incomplete" and result["collapsed"] is None
    assert result["entropy_violations"]  # Retain diagnostics without a verdict.


@pytest.mark.parametrize("bad", [None, True, -1., float("nan"), float("inf")])
def test_invalid_or_missing_entropy_is_an_error_even_before_warmup(small_ledgers, bad):
    updates, episodes = small_ledgers
    if bad is None:
        del updates[0]["action_entropy_sum"]
    else:
        updates[0]["action_entropy_sum"] = bad
    with pytest.raises(ValueError, match="finite nonnegative action_entropy_sum"):
        gate._gate_statistics(updates, episodes)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def complete_gate(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-stale-gate")
    source_bytes = b"# Fabricated engineering fixture, not experimental evidence.\n"
    files = {"fixture.py": hashlib.sha256(source_bytes).hexdigest()}
    source = dict(files=files, sha256=gate.digest(files))
    manifest_path = root / "manifest.json"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gate, "source_snapshot", lambda: deepcopy(source))
        manifest = gate.prepare(manifest_path)
        entry = manifest["entries"][0]
        run_root = root / "training"
        receipt = gate.launch(manifest_path, entry["index"], run_root, dry_run=True)
    run = run_root / entry["relative_output"]
    run.mkdir(parents=True)
    config = Config(**entry["config"])
    write_json(run / "config.json", entry["config"])
    write_json(run / "source_manifest.json", source)
    (run / "source").mkdir()
    (run / "source/fixture.py").write_bytes(source_bytes)
    write_json(run / "run.json", dict(
        source_sha256=source["sha256"], environment_version=config.env_version,
        dtype="float64", resume=None, episode_log="chunked-gzip", checkpoint_commits=True,
    ))
    writer = ChunkedEpisodes(run, config.max_steps)
    with (run / "updates.jsonl").open("x") as stream:
        for update in range(1, 1001):
            episodes = []
            for collector in range(4):
                num_p = 1 + collector % 2
                # Each collector completes six horizon-capped failures and one
                # 20-step success: exactly its configured 500-step batch.
                for index, steps in enumerate((80, 80, 80, 80, 80, 80, 20)):
                    episodes.append(dict(
                        update=update, collector=collector,
                        scenario_id=f"synthetic:{update}:{collector}:{index}",
                        num_p=num_p, num_a=1, num_agents=num_p + 1,
                        composition=[num_p, 1], steps=steps, success=index == 6,
                        team_return=-0.1 * steps * (num_p + 1), mean_agent_return=-0.1 * steps,
                        event_exposed=False, sensing=[1] * num_p + [0],
                    ))
            writer.append(episodes, update)
            stream.write(json.dumps(dict(update=update, total_steps=update * 2000,
                                         steps=2000, episodes=28, action_entropy_sum=50.,
                                         gradient_norm_before_clip=1.)) + "\n")
    writer.seal()
    writer.close()
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = SoftRoleNet(**config.model_kwargs()).double()
    signature = model_signature(model)
    checkpoint = run / "checkpoints/epoch0100.pt"
    checkpoint.parent.mkdir()
    payload = dict(
        format_version=CHECKPOINT_VERSION, config=entry["config"], model_config=config.model_kwargs(),
        model_state=model.state_dict(), optimizer_state=make_optimizer(model.parameters(), config).state_dict(),
        optimizer=optimizer_spec(config), environment_version=config.env_version,
        epoch=100, updates=1000, total_steps=2_000_000, total_episodes=28_000,
        completed_epochs=100, updates_in_partial_epoch=0,
        source_sha256=source["sha256"], signature=signature,
    )
    torch.save(payload, checkpoint)
    write_json(str(checkpoint) + ".signature.json", signature)
    checkpoint_commit(checkpoint, run, payload)
    write_json(run / "finished.json", dict(checkpoint=str(checkpoint),
                                          total_steps=2_000_000, total_episodes=28_000, updates=1000))
    receipt.update(dry_run=False, checkpoint=str(checkpoint))
    write_json(run / "stale_capability.json", receipt)
    return SimpleNamespace(manifest=manifest, manifest_path=manifest_path, entry=entry,
                           run=run, run_root=run_root, checkpoint=checkpoint)


@pytest.fixture
def copied_gate(complete_gate, tmp_path):
    run_root = tmp_path / "training"
    run = run_root / complete_gate.entry["relative_output"]
    shutil.copytree(complete_gate.run, run)
    return SimpleNamespace(**{**vars(complete_gate), "run_root": run_root, "run": run,
                              "checkpoint": run / "checkpoints/epoch0100.pt"})


def verdict(case):
    return gate._gate_seed_verdict(case.manifest, case.entry, case.run_root)


def test_real_ledger_audit_and_committed_checkpoint_produce_evidence_not_selection(complete_gate):
    result = verdict(complete_gate)
    assert result["status"] == "pass" and result["collapsed"] is False
    assert (result["total_steps"], result["total_episodes"]) == (2_000_000, 28_000)
    assert result["entropy_updates_checked"] == 500
    assert result["minimum_entropy"]["agent_steps"] == 5000
    assert result["minimum_entropy"]["mean_action_entropy"] == 0.01
    assert result["audit_sha256"] and result["evidence_sha256"]
    assert result["checkpoint"]["sha256"] == sha(complete_gate.checkpoint)
    assert "selected_coefficient" not in result


@pytest.mark.parametrize("target,field,value,match", [
    ("config.json", "seed", 1, "configuration differs"),
    ("source_manifest.json", "sha256", "0" * 64, "source differs"),
    ("finished.json", "total_steps", 1_999_999, "completion counts differ"),
    ("stale_capability.json", "dry_run", True, "launch receipt differs"),
])
def test_summary_rejects_mutated_run_identity_and_completion(copied_gate, target, field, value, match):
    path = copied_gate.run / target
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)
    with pytest.raises(ValueError, match=match):
        verdict(copied_gate)


def test_summary_rejects_changed_archived_source_bytes(copied_gate):
    (copied_gate.run / "source/fixture.py").write_text("changed\n")
    with pytest.raises(ValueError, match="archived source changed"):
        verdict(copied_gate)


def test_summary_rejects_duplicate_episodes_even_with_refreshed_chunk_hash(copied_gate):
    index_path = copied_gate.run / "episodes.index.json"
    index = json.loads(index_path.read_text())
    chunk = index["chunks"][0]
    path = copied_gate.run / chunk["path"]
    rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
    # Identical-length failures keep all cached chunk statistics unchanged.
    rows[1]["scenario_id"] = rows[0]["scenario_id"]
    path.write_bytes(gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode(), mtime=0))
    chunk.update(sha256=sha(path), bytes=path.stat().st_size)
    write_json(index_path, index)
    with pytest.raises(ValueError, match="Duplicate episode identity"):
        verdict(copied_gate)


@pytest.mark.parametrize("target", ["finished.json", "stale_capability.json"])
def test_missing_completion_or_source_check_receipt_is_incomplete(copied_gate, target):
    (copied_gate.run / target).unlink()
    result = verdict(copied_gate)
    assert result["status"] == "incomplete" and result["collapsed"] is None


@pytest.mark.parametrize("field", ["config", "environment_version", "model_config"])
def test_resigned_checkpoint_must_remain_bound_to_locked_configuration(copied_gate, field):
    payload = torch.load(copied_gate.checkpoint, map_location="cpu", weights_only=False)
    if field == "config":
        payload[field]["entropy_coeff"] = 0.01
    elif field == "environment_version":
        payload[field] = "corrected-observation-v1"
    else:
        payload[field]["comm_rounds"] = 2
    torch.save(payload, copied_gate.checkpoint)
    checkpoint_commit(copied_gate.checkpoint, copied_gate.run, payload)
    with pytest.raises(ValueError, match="checkpoint.*(configuration|environment|model)|Checkpoint.*(configuration|environment|model)"):
        verdict(copied_gate)


@pytest.mark.parametrize("statuses,expected_statuses,decision,eligible", [
    (["pass"] * 6, ["pass", "pass"], "awaiting_user_coefficient_confirmation", [0.003, 0.01]),
    (["collapsed", "pass", "pass", "pass", "pass", "pass"], ["collapsed", "pass"],
     "awaiting_user_coefficient_confirmation", [0.01]),
    (["collapsed"] * 6, ["collapsed", "collapsed"], "both_coefficients_collapsed_stop", []),
    (["incomplete", "collapsed", "pass", "error", "pass", "pass"], ["incomplete", "error"],
     "awaiting_complete_valid_evidence", []),
])
def test_coefficient_aggregation_never_selects_automatically(
        complete_gate, tmp_path, monkeypatch, statuses, expected_statuses, decision, eligible):
    def seed_result(manifest, entry, run_root):
        status = statuses[entry["index"]]
        if status == "error":
            raise ValueError("synthetic invalid evidence")
        return dict(seed=entry["seed"], status=status,
                    collapsed=(status == "collapsed") if status != "incomplete" else None)

    monkeypatch.setattr(gate, "_gate_seed_verdict", seed_result)
    output = tmp_path / "summary.json"
    result = gate.summarize_gate(complete_gate.manifest_path, complete_gate.run_root, output)
    assert [row["status"] for row in result["coefficients"]] == expected_statuses
    assert result["decision"] == decision and result["eligible_coefficients"] == eligible
    assert result["selected_coefficient"] is None
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError, match="already exists"):
        gate.summarize_gate(complete_gate.manifest_path, complete_gate.run_root, output)
