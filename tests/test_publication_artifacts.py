"""Recovery eligibility and immutable evidence checks, without running a learner."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from publication_reconstruction import __main__ as launcher
from publication_reconstruction.artifacts import lineage_metrics, load_checkpoint, verify_run
from publication_reconstruction.study import checkpoint_candidates
from publication_reconstruction.runtime.hetnet_ext.recovery import capture_rng, scientific_arguments


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def archived_run(tmp_path):
    run = tmp_path / "parent"
    source = run / "source/runtime"
    source.mkdir(parents=True)
    (source / "main.py").write_text("# Source fixture; never executed.\n")
    manifest = {"schema_version": 2, "files": {
        "runtime/main.py": hashlib.sha256((source / "main.py").read_bytes()).hexdigest()}}
    write_json(run / "source_manifest.json", manifest)
    args = SimpleNamespace(task="pcp", variant="real", model_spec="supplement-v1", seed=2,
        env_version="corrected-v1", recipe="october-2022", output=run, epochs=4,
        collectors=2, updates_per_epoch=3, batch_steps=5, horizon=8, save_every=1,
        max_env_steps=100, milestones=[50, 100], episode_log="file", wall_seconds=0)
    protocol = launcher.resolve(args)
    write_json(run / "protocol.json", protocol)
    for name in ("uv.lock", "requirements.txt"):
        (run / name).write_text("fixture\n")
    counts = {"env_steps": 12, "episodes": 2, "updates": 1, "epoch": 1}
    resolved = dict(seed=2, nprocesses=2, epoch_size=3, batch_size=5, max_steps=8,
        num_epochs=4, publication_env_version="corrected-v1", use_binary=False,
        model_spec="supplement-v1", max_env_steps=100, lrate=.001, env_name="predator_capture",
        nfriendly_P=2, nfriendly_A=1, nagents=3, dim=5, vision=2, detach_gap=5,
        hid_size=128, hetgat=True, hetgat_a2c=True, milestones=[50, 100],
        rng_scheme="seedsequence-v1", save_every=1, episode_log="file", wall_seconds=0,
        gamma=1.0, tau=1.0, normalize_rewards=False, entr=0., value_coeff=.01,
        comm_range_P=-1, comm_range_A=-1, lossy_comm=False,
        min_comm_loss=0., max_comm_loss=.3, recurrent=False,
        commnet=False, hetcomm=False, ic3net=False, use_cuda=False,
        random=False, eval=False, msg_dim=16, action_scale=1., nactions='1',
        rnn_type='MLP', total_state_action_in_batch=500,
        no_stay=False, mode='mixed', tensor_obs=False, A_vision=-1,
        nenemies=1, moving_prey=False, enemy_comm=False, second_reward_scheme=False)
    parameter = torch.nn.Parameter(torch.tensor([.25], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=.001, foreach=False, fused=False)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()  # Creates actual finite Adam state; no policy or environment.
    checkpoint = run / "checkpoint.pt"
    saved = {"seed": 2, "policy_net": {"weight": parameter.detach().clone()},
        "trainer": optimizer.state_dict(), "log": {}, "reconstruction": {
            "schema_version": 2, "env_version": "corrected-v1", "model_spec": "supplement-v1",
            "rng_scheme": "seedsequence-v1", "counts": counts, "resolved_args": resolved,
            "source_manifest_sha256": hashlib.sha256((run / "source_manifest.json").read_bytes()).hexdigest()},
        "recovery": {"version": 1, "counts": copy.deepcopy(counts), "completed_epochs": 0,
            "updates_in_epoch": 1, "scientific_args": scientific_arguments(SimpleNamespace(**resolved)),
            "rng_states": [capture_rng(), capture_rng()], "stop_reason": "wall_time",
            "epoch_elapsed_seconds": .1, "milestones_reached": [], "epoch_stat": {},
            "recorder_state": {"total_steps": 0, "total_episodes": 0, "last_epoch": 0,
                "steps": 12, "episodes": 2, "success": 0., "steps_taken": 12.,
                "reward": [-1., -1., -1.], "policy_loss": 1., "value_loss": 1.}}}
    torch.save(saved, checkpoint)
    (run / 'checkpoint_records.jsonl').write_text(json.dumps({
        'path': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'bytes': checkpoint.stat().st_size, 'counts': counts, 'update': counts['updates'],
        'epoch': counts['epoch']}) + '\n')
    return SimpleNamespace(run=run, checkpoint=checkpoint, saved=saved, protocol=protocol, manifest=manifest)


def resume_args(archive, output):
    return SimpleNamespace(run_dir=archive.run, checkpoint=archive.checkpoint,
        output=output, episode_log=None, wall_seconds=0, dry_run=False)


def test_valid_recovery_preserves_science_and_changes_only_segment_paths(archived_run, tmp_path):
    archive = archived_run
    loaded = load_checkpoint(archive.run, archive.checkpoint, require_recovery=True)
    assert torch.equal(loaded["policy_net"]["weight"], archive.saved["policy_net"]["weight"])
    destination = tmp_path / "continued"
    arguments = resume_args(archive, destination)
    arguments.episode_log, arguments.wall_seconds = "stdout", 5
    resolved = launcher.resolve_resume(arguments)
    assert not destination.exists()
    for key in ("task", "seed", "model_spec", "optimizer", "collectors", "epochs", "max_env_steps",
                "episode_horizon", "milestones", "env_version", "rng_scheme", "learner_spec"):
        assert resolved[key] == archive.protocol[key]
    assert resolved["continuation"]["checkpoint_sha256"] == hashlib.sha256(archive.checkpoint.read_bytes()).hexdigest()
    assert resolved["continuation"]["counts"] == archive.saved["reconstruction"]["counts"]
    assert resolved["episode_log"] == "stdout" and resolved["wall_seconds"] == 5
    command = resolved["command"]
    for flag in ("--metrics_file", "--save_dir", "--source_manifest"):
        assert Path(command[command.index(flag) + 1]).parent == destination
    assert command[command.index("--resume_checkpoint") + 1] == str(archive.checkpoint)


def test_original_schema_remains_loadable_but_cannot_claim_recovery(archived_run):
    archive = archived_run
    archive.protocol["schema_version"] = 1
    for key in ("model_spec", "rng_scheme", "learner_spec", "milestones", "episode_log", "wall_seconds"):
        archive.protocol.pop(key)
    archive.saved["reconstruction"]["schema_version"] = 1
    for key in ("model_spec", "rng_scheme"):
        archive.saved["reconstruction"].pop(key)
        archive.saved["reconstruction"]["resolved_args"].pop(key)
    archive.saved.pop("recovery")
    write_json(archive.run / "protocol.json", archive.protocol)
    torch.save(archive.saved, archive.checkpoint)
    assert load_checkpoint(archive.run, archive.checkpoint)["reconstruction"]["schema_version"] == 1
    with pytest.raises(ValueError, match="continuation state"):
        load_checkpoint(archive.run, archive.checkpoint, require_recovery=True)


@pytest.mark.parametrize("corrupt_copy", [False, True])
def test_continuation_source_copy_verifies_destination_bytes(archived_run, tmp_path, monkeypatch, corrupt_copy):
    destination = tmp_path / "continued"
    destination.mkdir()
    original_write = Path.write_bytes
    def write_bytes(path, payload):
        if corrupt_copy and path == destination / "source/runtime/main.py":
            payload += b"# Changed during copying.\n"
        return original_write(path, payload)
    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    if corrupt_copy:
        with pytest.raises(ValueError, match="source mismatch"):
            launcher.copy_resume_source(archived_run.run, destination)
    else:
        launcher.copy_resume_source(archived_run.run, destination)
        for relative in ("source/runtime/main.py", "source_manifest.json", "uv.lock", "requirements.txt"):
            assert (destination / relative).read_bytes() == (archived_run.run / relative).read_bytes()


def test_unreadable_checkpoint_fails_before_output(archived_run, tmp_path):
    archived_run.checkpoint.write_bytes(b"incomplete checkpoint")
    destination = tmp_path / "continued"
    with pytest.raises(ValueError, match="cannot be decoded"):
        launcher.resolve_resume(resume_args(archived_run, destination))
    assert not destination.exists()


@pytest.mark.parametrize("corruption", ["source", "manifest_hash", "escape", "missing_inventory", "extra_runtime"])
def test_archive_corruption_is_rejected(archived_run, tmp_path, corruption):
    archive = archived_run
    if corruption == "source":
        (archive.run / "source/runtime/main.py").write_text("# Changed source.\n")
    elif corruption == "manifest_hash":
        archive.manifest["files"]["runtime/main.py"] = "0" * 64
    elif corruption == "escape":
        external = tmp_path / "external.py"
        external.write_text("pass\n")
        archive.manifest["files"][str(external)] = hashlib.sha256(external.read_bytes()).hexdigest()
    elif corruption == "missing_inventory":
        archive.manifest["files"] = {}
    else:
        (archive.run / "source/runtime/unrecorded.py").write_text("pass\n")
    write_json(archive.run / "source_manifest.json", archive.manifest)
    with pytest.raises(ValueError):
        verify_run(archive.run)


@pytest.mark.parametrize("corruption", ["schema", "source_hash", "model_nan", "negative_count", "bool_count",
    "seed", "env", "variant", "model_spec", "budget", "collector", "epoch_count", "recovery_counts",
    "rng_count", "completed_steps", "completed_epochs", "completed_updates", "optimizer_nan", "optimizer_missing"])
def test_corrupt_checkpoint_is_ineligible_before_fresh_segment(archived_run, tmp_path, corruption):
    archive = archived_run
    saved = archive.saved
    meta, recovery = saved["reconstruction"], saved["recovery"]
    if corruption == "schema":
        meta["schema_version"] = 999
    elif corruption == "source_hash":
        meta["source_manifest_sha256"] = "0" * 64
    elif corruption == "model_nan":
        saved["policy_net"]["weight"][0] = float("nan")
    elif corruption in ("negative_count", "bool_count"):
        meta["counts"]["episodes"] = -1 if corruption == "negative_count" else True
    elif corruption == "seed":
        saved["seed"] += 1
    elif corruption == "env":
        meta["env_version"] = "historical-2022"
    elif corruption == "variant":
        meta["resolved_args"]["use_binary"] = True
    elif corruption == "model_spec":
        meta["resolved_args"]["model_spec"] = "public-code-v1"
    elif corruption == "budget":
        meta["resolved_args"]["max_env_steps"] += 1
    elif corruption == "collector":
        meta["resolved_args"]["nprocesses"] += 1
    elif corruption == "epoch_count":
        meta["resolved_args"]["num_epochs"] += 1
    elif corruption == "recovery_counts":
        recovery["counts"]["updates"] += 1
    elif corruption == "rng_count":
        recovery["rng_states"].pop()
    elif corruption == "completed_steps":
        meta["counts"]["env_steps"] = recovery["counts"]["env_steps"] = 100
    elif corruption == "completed_epochs":
        recovery["completed_epochs"] = 4
    elif corruption == "completed_updates":
        recovery["completed_epochs"] = 1  # Does not agree with update count 1.
    elif corruption == "optimizer_nan":
        next(iter(saved["trainer"]["state"].values()))["exp_avg"][0] = float("nan")
    else:
        saved.pop("trainer")
    torch.save(saved, archive.checkpoint)
    destination = tmp_path / "continued"
    with pytest.raises(ValueError):
        launcher.resolve_resume(resume_args(archive, destination))
    assert not destination.exists()


@pytest.mark.parametrize("field,value", [("task", "fc"), ("rng_scheme", "changed"),
    ("optimizer", {"name": "Adam", "lr": .1}), ("model_spec", "public-code-v1")])
def test_edited_scientific_protocol_is_rejected(archived_run, tmp_path, field, value):
    archived_run.protocol[field] = value
    write_json(archived_run.run / "protocol.json", archived_run.protocol)
    with pytest.raises(ValueError):
        launcher.resolve_resume(resume_args(archived_run, tmp_path / "continued"))


@pytest.mark.parametrize("flag,value", [("--lrate", ".1"), ("--nfriendly_P", "4"),
    ("--max_env_steps", "200"), ("--model_spec", "public-code-v1")])
def test_edited_scientific_command_is_rejected(archived_run, tmp_path, flag, value):
    command = archived_run.protocol["command"]
    command[command.index(flag) + 1] = value
    write_json(archived_run.run / "protocol.json", archived_run.protocol)
    with pytest.raises(ValueError):
        launcher.resolve_resume(resume_args(archived_run, tmp_path / "continued"))


@pytest.mark.parametrize("flag,value", [("--task", "fc"), ("--seed", "3"), ("--collectors", "3"),
    ("--max-env-steps", "200"), ("--model-spec", "public-code-v1"), ("--epochs", "9"),
    ("--horizon", "9"), ("--env-version", "historical-2022"), ("--milestones", "75")])
def test_resume_cli_rejects_scientific_overrides_before_output(archived_run, tmp_path, flag, value):
    destination = tmp_path / "continued"
    with pytest.raises(SystemExit) as error:
        launcher.main(["resume", "--run-dir", str(archived_run.run), "--checkpoint", str(archived_run.checkpoint),
                       "--output", str(destination), flag, value])
    assert error.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_resume_never_replaces_existing_destination(archived_run, tmp_path, kind, monkeypatch):
    destination = tmp_path / "continued"
    if kind == "directory":
        destination.mkdir()
        (destination / "evidence").write_text("original")
    elif kind == "file":
        destination.write_text("original")
    else:
        destination.symlink_to(tmp_path / "absent")
    monkeypatch.setattr(launcher, "copy_resume_source", lambda *args: pytest.fail("Must reject before copying"))
    with pytest.raises(FileExistsError):
        launcher.main(["resume", "--run-dir", str(archived_run.run), "--checkpoint", str(archived_run.checkpoint),
                       "--output", str(destination)])
    if kind == "directory":
        assert list(destination.iterdir()) == [destination / "evidence"]
        assert (destination / "evidence").read_text() == "original"
    elif kind == "file":
        assert destination.read_text() == "original"
    else:
        assert destination.is_symlink() and not (tmp_path / "absent").exists()


def make_segment(path, epochs, parent=None, retained=None):
    path.mkdir()
    protocol = {} if parent is None else {"continuation": {"run_dir": str(parent), "completed_epochs": retained}}
    write_json(path / "protocol.json", protocol)
    (path / "metrics.jsonl").write_text("".join(json.dumps({"epoch": epoch, "origin": path.name}) + "\n" for epoch in epochs))
    return path


def test_lineage_keeps_only_selected_parent_prefix_across_multiple_segments(tmp_path):
    parent = make_segment(tmp_path / "original", [1, 2, 3, 4, 5])
    child = make_segment(tmp_path / "child", [3, 4, 5, 6], parent, 2)
    grandchild = make_segment(tmp_path / "grandchild", [5, 6, 7], child, 4)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    rows = lineage_metrics(grandchild)
    assert [row["epoch"] for row in rows] == list(range(1, 8))
    assert [row["origin"] for row in rows] == ["original"] * 2 + ["child"] * 2 + ["grandchild"] * 3
    assert all(p.read_bytes() == data for p, data in before.items())


def test_lineage_ignores_partial_abandoned_suffix_after_selected_checkpoint(tmp_path):
    parent = make_segment(tmp_path / "original", [1, 2, 3])
    with (parent / "metrics.jsonl").open("a") as stream:
        stream.write('{"epoch": 4, "interrupted":')
    child = make_segment(tmp_path / "continued", [3, 4], parent, 2)
    assert [row["origin"] for row in lineage_metrics(child)] == ["original", "original", "continued", "continued"]
    with pytest.raises(ValueError):
        lineage_metrics(parent)  # The same malformed active segment is not hidden.


@pytest.mark.parametrize("case", ["cycle", "gap", "duplicate", "missing_prefix"])
def test_lineage_rejects_inconsistent_retained_history(tmp_path, case):
    parent = make_segment(tmp_path / "parent", [1, 2])
    child = make_segment(tmp_path / "child", [3], parent, 2)
    if case == "cycle":
        write_json(parent / "protocol.json", {"continuation": {"run_dir": str(child), "completed_epochs": 3}})
    elif case == "gap":
        (child / "metrics.jsonl").write_text('{"epoch": 4}\n')
    elif case == "duplicate":
        (child / "metrics.jsonl").write_text('{"epoch": 2}\n')
    else:
        write_json(child / "protocol.json", {"continuation": {"run_dir": str(parent), "completed_epochs": 3}})
        (child / "metrics.jsonl").write_text('{"epoch": 4}\n')
    with pytest.raises(ValueError):
        lineage_metrics(child)


def test_checkpoint_candidates_trim_abandoned_suffixes_and_relocate_paths(archived_run, tmp_path):
    parent = archived_run.run
    parent_protocol = archived_run.protocol
    parent_protocol["output"] = "/original machine/run with spaces"
    write_json(parent / "protocol.json", parent_protocol)
    def index(run, protocol, updates, partial=False):
        lines = [json.dumps({"counts": {"updates": update, "env_steps": 10 * update},
                            "path": str(Path(protocol["output"]) / "checkpoints" / f"u{update}.pt")})
                 for update in updates]
        (run / "checkpoint_records.jsonl").write_text("\n".join(lines) + "\n" + ('{"counts":' if partial else ""))
    index(parent, parent_protocol, [1, 2], partial=True)
    child = tmp_path / "child"
    shutil.copytree(parent, child)
    child_protocol = {**parent_protocol, "output": "/old child/run", "continuation": {
        "run_dir": str(parent), "counts": {"updates": 2}}}
    write_json(child / "protocol.json", child_protocol)
    index(child, child_protocol, [3, 4], partial=True)
    grandchild = tmp_path / "grandchild"
    shutil.copytree(parent, grandchild)
    grandchild_protocol = {**parent_protocol, "output": "/old grandchild/run", "continuation": {
        "run_dir": str(child), "counts": {"updates": 3}}}
    write_json(grandchild / "protocol.json", grandchild_protocol)
    index(grandchild, grandchild_protocol, [4, 5])
    before = {p: p.read_bytes() for p in tmp_path.rglob("checkpoint_records.jsonl")}
    candidates = checkpoint_candidates(grandchild)
    assert [row["counts"]["updates"] for row in candidates] == [1, 2, 3, 4, 5]
    assert [row["run_dir"] for row in candidates] == list(map(str, [parent, parent, child, grandchild, grandchild]))
    for row in candidates:
        assert row["path"] == str(Path(row["run_dir"]) / "checkpoints" / f"u{row['counts']['updates']}.pt")
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(ValueError):
        checkpoint_candidates(parent)  # A corrupt active suffix still fails.


def test_checkpoint_candidates_reject_corrupt_retained_prefix_and_cycles(archived_run, tmp_path):
    parent = archived_run.run
    (parent / "checkpoint_records.jsonl").write_text('{"path": "missing counts"}\n')
    with pytest.raises(ValueError, match="complete-update counts"):
        checkpoint_candidates(parent, update_limit=2)
    archived_run.protocol["continuation"] = {"run_dir": str(parent), "counts": {"updates": 1}}
    write_json(parent / "protocol.json", archived_run.protocol)
    with pytest.raises(ValueError, match="lineage cycle"):
        checkpoint_candidates(parent)
