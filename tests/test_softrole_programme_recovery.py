"""Real bounded updates prove recovery/logging preserve the learner trajectory."""
import json
from pathlib import Path

import pytest
import torch

from softrole.config import Config
from softrole.ledger import (INDEX, audit_run, committed_checkpoints, iter_episodes,
                             logical_segments, validate_receipt)
from softrole.programme import (accepted_checkpoints, audit_s1_run, launch, prepare,
                                read_json, select_learning_rate, window_statistics)
from softrole.train import train
from test_softrole_programme import synthetic_selection, synthetic_run


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_plain_and_compressed_logging_preserve_weights_optimizer_and_all_episodes(tmp_path):
    cfg = Config(model="shared", epochs=2, updates_per_epoch=2, batch_steps=1, nprocesses=1,
                 max_steps=3, save_every=1)
    plain = train(cfg, tmp_path / "plain")
    compressed = train(cfg, tmp_path / "compressed", episode_log="chunked-gzip")
    for key in ("model_state", "optimizer_state", "signature", "updates", "total_steps", "total_episodes"):
        same(checkpoint(plain)[key], checkpoint(compressed)[key])
    assert list(iter_episodes(tmp_path / "plain")) == list(iter_episodes(tmp_path / "compressed"))
    assert audit_run(tmp_path / "plain")["updates"] == audit_run(tmp_path / "compressed")["updates"]
    assert window_statistics(tmp_path / "plain", [(0, 6), (6, 12)], 3) == window_statistics(tmp_path / "compressed", [(0, 6), (6, 12)], 3)
    assert not (tmp_path / "compressed" / "episodes.jsonl").exists()
    assert len(read_json(tmp_path / "compressed" / INDEX)["chunks"]) == 2
    # Repeated deterministic compression produces identical bytes, including headers.
    repeated = train(cfg, tmp_path / "repeated", episode_log="chunked-gzip")
    chunks = read_json(tmp_path / "compressed" / INDEX)["chunks"]
    assert chunks == read_json(tmp_path / "repeated" / INDEX)["chunks"]


def setup_manifest(tmp_path):
    _, selection = synthetic_selection(tmp_path / "s1")
    manifest = tmp_path / "manifest.json"
    prepare(selection, manifest, diagnostics_every=None, bounded=True)
    overrides = dict(epochs=3, updates_per_epoch=2, batch_steps=1, nprocesses=1, total_steps=10000, save_every=1)
    root = tmp_path / "runs"
    output = root / "C1/c1_real64/seed0"
    return manifest, overrides, root, output


def test_repeated_recovery_discards_abandoned_tails_and_matches_uninterrupted(tmp_path, monkeypatch):
    import softrole.train as training
    manifest, overrides, root, output = setup_manifest(tmp_path)
    original = training.collect_batch
    def stop_at(update):
        def collect(*args, **kwargs):
            if args[3] == update:
                raise RuntimeError("timeout")
            return original(*args, **kwargs)
        return collect
    monkeypatch.setattr(training, "collect_batch", stop_at(3))
    with pytest.raises(RuntimeError, match="timeout"):
        launch(manifest, 0, root, overrides=overrides)
    assert accepted_checkpoints(output)[-1]["updates"] == 2
    original_tail = (output / "updates.jsonl").read_bytes()
    monkeypatch.setattr(training, "collect_batch", stop_at(5))
    with pytest.raises(RuntimeError, match="timeout"):
        launch(manifest, 0, root, resume=True)
    assert accepted_checkpoints(output)[-1]["updates"] == 4
    monkeypatch.setattr(training, "collect_batch", original)
    launch(manifest, 0, root, resume=True)
    rows, receipt = accepted_checkpoints(output, return_audit=True)
    assert [row["updates"] for row in rows] == [2, 4, 6]
    assert len(logical_segments(output)) == 3
    assert (output / "updates.jsonl").read_bytes() == original_tail
    baseline = tmp_path / "baseline"
    cfg = Config(**read_json(output / "config.json"))
    uninterrupted = train(cfg, baseline, episode_log="chunked-gzip")
    for name in ("model_state", "optimizer_state", "signature"):
        same(checkpoint(rows[-1]["checkpoint"])[name], checkpoint(uninterrupted)[name])
    assert receipt["updates"] == audit_run(baseline)["updates"]
    assert [row["update"] for row in receipt["updates"]] == list(range(1, 7))
    # Lost final receipt does not rerun a completed trajectory.
    (output / "attempts/0002/finished.json").unlink()
    launch(manifest, 0, root, resume=True)
    assert not (output / "attempts/0003").exists()
    metadata = output / "attempts/0002/attempt.json"
    value = read_json(metadata)
    value["manifest_sha256"] = "wrong"
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity mismatch"):
        accepted_checkpoints(output)


def test_interrupted_checkpoint_publication_is_not_a_recovery_anchor(tmp_path, monkeypatch):
    import softrole.ledger as ledger
    manifest, overrides, root, output = setup_manifest(tmp_path)
    original = ledger.checkpoint_commit
    def fail_second(path, run, payload):
        if payload["updates"] == 4:
            raise RuntimeError("publication interrupted")
        return original(path, run, payload)
    monkeypatch.setattr(ledger, "checkpoint_commit", fail_second)
    with pytest.raises(RuntimeError, match="publication interrupted"):
        launch(manifest, 0, root, overrides=overrides)
    assert (output / "checkpoints/epoch0002.pt").exists()
    assert [r["updates"] for r in accepted_checkpoints(output)] == [2]
    monkeypatch.setattr(ledger, "checkpoint_commit", original)
    launch(manifest, 0, root, resume=True)
    assert [r["updates"] for r in accepted_checkpoints(output)] == [2, 4, 6]
    first = output / "checkpoints/epoch0001.pt"
    first.unlink()
    with pytest.raises(ValueError, match="Orphan"):
        accepted_checkpoints(output)


def test_resume_after_failure_before_attempt_zero_directory_exists(tmp_path, monkeypatch):
    import softrole.train as training
    manifest, overrides, root, output = setup_manifest(tmp_path)
    original = training.train
    def fail(*args, **kwargs):
        raise RuntimeError("initialization timeout")
    monkeypatch.setattr(training, "train", fail)
    with pytest.raises(RuntimeError):
        launch(manifest, 0, root, overrides=overrides)
    assert not output.exists()
    monkeypatch.setattr(training, "train", original)
    launch(manifest, 0, root, resume=True)
    assert accepted_checkpoints(output)[-1]["updates"] == 6
    assert read_json(output / "attempts/0001/attempt.json")["parent_checkpoint"] is None


def test_chunked_partial_epoch_resume_matches_uninterrupted(tmp_path):
    common = dict(model="shared", epochs=2, updates_per_epoch=3, batch_steps=1, nprocesses=1,
                  max_steps=3, save_every=1)
    partial = train(Config(**common, total_steps=1), tmp_path / "partial", episode_log="chunked-gzip")
    resumed = train(Config(**common), tmp_path / "resumed", resume=partial, episode_log="chunked-gzip")
    full = train(Config(**common), tmp_path / "full", episode_log="chunked-gzip")
    for key in ("model_state", "optimizer_state", "signature", "updates"):
        same(checkpoint(resumed)[key], checkpoint(full)[key])
    assert list(iter_episodes(tmp_path / "partial")) + list(iter_episodes(tmp_path / "resumed")) == list(iter_episodes(tmp_path / "full"))


def test_stdout_is_rejected_before_output_even_for_dry_run(tmp_path):
    with pytest.raises(ValueError, match="stdout"):
        launch(tmp_path / "absent.json", 0, tmp_path / "runs", episode_log="stdout", dry_run=True)
    assert not (tmp_path / "runs").exists()


def test_s1_stable_prefix_excludes_torn_tails_and_rejects_missing_episode(tmp_path):
    run = synthetic_run(tmp_path / "s1", 3e-4, 0, 6)
    with (run / "episodes.jsonl").open("a") as stream:
        stream.write('{"update": 15, "torn":')
    with (run / "updates.jsonl").open("a") as stream:
        stream.write('{"update": 15, "torn":')
    receipt = audit_s1_run(run, tmp_path / "evidence", target_steps=80, bounded=True)
    assert receipt["total_steps"] == 84 and receipt["covered_updates"] == [1, 14]
    validate_receipt(tmp_path / "evidence/audit.json", reaudit=True)
    damaged = synthetic_run(tmp_path / "damaged", 3e-4, 0, 6)
    path = damaged / "episodes.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="counts disagree|Missing"):
        audit_s1_run(damaged, tmp_path / "rejected", target_steps=80, bounded=True)


def test_audit_receipts_pin_selection_and_prepare_rejects_changed_receipt(tmp_path):
    result, selected = synthetic_selection(tmp_path / "s1")
    audits = [row["audit"] for row in result["runs"]]
    duplicate = select_learning_rate(None, tmp_path / "selected.json", window=(0, 80), bounded=True, audits=audits)
    assert duplicate["equal_seed_means_exact"] == result["equal_seed_means_exact"]
    path = Path(audits[0])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt changed"):
        prepare(selected, tmp_path / "manifest.json", bounded=True)


def test_forged_rehashed_chunk_cannot_override_checkpoint_publication(tmp_path):
    import gzip
    from softrole.ledger import add_episode, empty_stats, sha
    manifest, overrides, root, output = setup_manifest(tmp_path)
    launch(manifest, 0, root, overrides={**overrides, "epochs": 1})
    index_path = output / INDEX
    index = read_json(index_path)
    chunk = index["chunks"][0]
    path = output / chunk["path"]
    with gzip.open(path, "rt") as stream:
        episodes = [json.loads(line) for line in stream]
    episodes[0]["team_return"] += 1.
    episodes[0]["mean_agent_return"] += 1. / 3
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as stream:
            stream.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes).encode())
    chunk["sha256"], chunk["bytes"] = sha(path), path.stat().st_size
    statistics = {}
    for row in episodes:
        add_episode(statistics.setdefault(row["update"], empty_stats()), row, 80)
    chunk["updates"] = [dict(update=u, **stats) for u, stats in sorted(statistics.items())]
    index_path.write_text(json.dumps(index))
    # Index internally agrees with forged records, but publication hashes bind the
    # original records and must be checked independently of this attacker index.
    audit_run(output, through_update=2)
    with pytest.raises(ValueError, match="committed checkpoint evidence"):
        accepted_checkpoints(output)


def test_recovery_preserves_truncated_startup_metadata_before_restarting(tmp_path, monkeypatch):
    import softrole.train as training
    manifest, overrides, root, output = setup_manifest(tmp_path)
    original = training.train
    def fail(config, destination, **kwargs):
        destination.mkdir(parents=True)
        (destination / "config.json").write_text('{"task":')
        raise RuntimeError("startup write interrupted")
    monkeypatch.setattr(training, "train", fail)
    with pytest.raises(RuntimeError):
        launch(manifest, 0, root, overrides=overrides)
    monkeypatch.setattr(training, "train", original)
    launch(manifest, 0, root, resume=True)
    assert (output / "config.json.abandoned0").read_text() == '{"task":'
    assert accepted_checkpoints(output)[-1]["updates"] == 6


def test_exact_comparison_does_not_turn_nearby_means_into_a_tie():
    from fractions import Fraction
    from softrole.programme import _select_exact_rate
    # A pure arithmetic test, not a fabricated large training ledger. The old
    # 1e-12 tolerance incorrectly chose the lower rate for this unequal pair.
    assert _select_exact_rate({3e-4: Fraction(6), 1e-3: Fraction(6) - Fraction(1, 10**13)}) == 1e-3
    assert _select_exact_rate({3e-4: Fraction(6), 1e-3: Fraction(6)}) == 3e-4


def test_entry_lock_prevents_concurrent_attempts(tmp_path):
    import fcntl
    manifest, overrides, root, output = setup_manifest(tmp_path)
    output.parent.mkdir(parents=True)
    lockpath = output.parent / (output.name + ".programme.lock")
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active attempt"):
            launch(manifest, 0, root, overrides=overrides)
    assert not output.exists()
    assert not output.with_name(output.name + ".programme.json").exists()


def test_resume_dry_run_uses_original_runtime_and_requires_existing_entry(tmp_path, monkeypatch):
    import softrole.train as training
    manifest, overrides, root, output = setup_manifest(tmp_path)
    with pytest.raises(ValueError, match="No programme run"):
        launch(manifest, 0, root, dry_run=True, resume=True)
    def fail(*args, **kwargs):
        raise RuntimeError("timeout")
    monkeypatch.setattr(training, "train", fail)
    with pytest.raises(RuntimeError):
        launch(manifest, 0, root, overrides=overrides)
    resolved = launch(manifest, 0, root, dry_run=True, resume=True)
    assert all(getattr(resolved, key) == value for key, value in overrides.items())
    assert not output.exists()


def test_interrupted_attempt_metadata_publication_is_ignored_on_next_resume(tmp_path, monkeypatch):
    import softrole.ledger as ledger
    import softrole.train as training
    manifest, overrides, root, output = setup_manifest(tmp_path)
    original_collect = training.collect_batch
    def stop(*args, **kwargs):
        if args[3] == 3:
            raise RuntimeError("timeout")
        return original_collect(*args, **kwargs)
    monkeypatch.setattr(training, "collect_batch", stop)
    with pytest.raises(RuntimeError):
        launch(manifest, 0, root, overrides=overrides)
    monkeypatch.setattr(training, "collect_batch", original_collect)
    original_atomic = ledger.atomic_json
    def interrupt(path, value):
        path = Path(path)
        if path.name == "attempt.json":
            path.with_name("attempt.json.tmp").write_text('{"attempt":')
            raise RuntimeError("metadata publication interrupted")
        return original_atomic(path, value)
    monkeypatch.setattr(ledger, "atomic_json", interrupt)
    with pytest.raises(RuntimeError, match="metadata publication interrupted"):
        launch(manifest, 0, root, resume=True)
    assert not (output / "attempts/0001/attempt.json").exists()
    assert accepted_checkpoints(output)[-1]["updates"] == 2
    monkeypatch.setattr(ledger, "atomic_json", original_atomic)
    launch(manifest, 0, root, resume=True)
    assert accepted_checkpoints(output)[-1]["updates"] == 6
    assert (output / "attempts/0001/attempt.json.tmp").read_text() == '{"attempt":'
