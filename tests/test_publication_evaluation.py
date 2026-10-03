"""Frozen reconstruction evaluation must use archived code and matched panels."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from publication_reconstruction.evaluation import evaluate, source_identity
from publication_reconstruction.evaluation_worker import sensory_mask
from softrole.scenarios import make_scenarios


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def archived_runs(tmp_path_factory):
    """Build untrained checkpoint fixtures; no collector, optimizer or job runs."""
    directory = tmp_path_factory.mktemp("publication-evaluation")
    runtime = ROOT / "publication_reconstruction/runtime"
    specifications = [(task, spec, "corrected-v1", 2)
                      for task in ("pp", "pcp", "fc")
                      for spec in ("public-code-v1", "supplement-v1")]
    specifications.append(("pcp", "public-code-v1", "historical-2022", 1))
    runs = {}
    for task, spec, version, schema in specifications:
        run = directory / f"{task}-{spec}-{version}"
        files = {}
        for source in runtime.rglob("*.py"):
            relative = Path("runtime") / source.relative_to(runtime)
            target = run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            target.write_bytes(data)
            files[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        (run / "source_manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}))
        runs[(task, spec, version)] = run
    script = r'''
import argparse, hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
runtime = next(root.glob("*/source/runtime"))
sys.path[:0] = [str(runtime), str(runtime / "envs")]
import torch
from hetgat.uavnet import UAVNetA2CEasy
from ic3net_envs.predator_prey_env import PredatorPreyEnv
from ic3net_envs.predator_capture_env import PredatorCaptureEnv
from ic3net_envs.fire_commander_env import FireCommanderEnv
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(41)
for run in sorted(root.iterdir()):
    task, spec = run.name.split("-", 1)
    version = "historical-2022" if "historical" in run.name else "corrected-v1"
    spec = "supplement-v1" if "supplement" in run.name else "public-code-v1"
    schema = 1 if version == "historical-2022" else 2
    p, a, vision = (3, 0, 2) if task == "pp" else (2, 1, 1 if task == "fc" else 2)
    env_name, cls = {"pp": ("predator_prey", PredatorPreyEnv), "pcp": ("predator_capture", PredatorCaptureEnv),
                     "fc": ("fire_commander", FireCommanderEnv)}[task]
    parser = argparse.ArgumentParser()
    cls().init_args(parser)
    args = vars(parser.parse_args([]))
    args.update(env_name=env_name, publication_env_version=version, seed=41, dim=5, vision=vision,
                nfriendly_P=p, nfriendly_A=a, nagents=p+a, max_steps=4, nprocesses=1, epoch_size=1,
                batch_size=1, num_epochs=1, max_env_steps=0, use_binary=spec == "supplement-v1",
                msg_dim=16, model_spec=spec, nfires=1, reward_type=3, A_vision=-1)
    model = UAVNetA2CEasy(dict(vision=vision, P=29, A=25, state=4), dict(P=29, A=25, state=4),
                         dict(P=16, A=16, state=16), dict(P=5, A=6, state=8), num_P=p, num_A=a,
                         num_heads=4, msg_dim=16, use_CNN=False, use_real=not args["use_binary"],
                         per_class_critic=True, per_agent_critic=False, device=torch.device("cpu"),
                         with_two_state=True, obs=(2*vision+1)**2, action_vision=-1, model_spec=spec)
    protocol = dict(task=task, seed=41, collectors=1, updates_per_epoch=1,
                    batch_step_floor_per_collector=1, episode_horizon=4, epochs=1,
                    variant="binary" if args["use_binary"] else "real", env_version=version,
                    model_spec=spec, max_env_steps=None, learner_spec="public-code-v1",
                    rng_scheme="seedsequence-v1" if spec == "supplement-v1" else "legacy-offset-v1")
    metadata = dict(schema_version=schema, resolved_args=args, env_version=version,
                    model_spec=spec, rng_scheme=protocol["rng_scheme"],
                    counts=dict(env_steps=12, episodes=3, updates=1, epoch=1),
                    source_manifest_sha256=hashlib.sha256((run / "source_manifest.json").read_bytes()).hexdigest())
    if schema == 1:
        args.pop("model_spec")
        metadata.pop("model_spec")
        protocol.pop("model_spec")
    (run / "protocol.json").write_text(json.dumps(protocol))
    torch.save(dict(policy_net=model.state_dict(), seed=41, reconstruction=metadata), run / "checkpoint.pt")
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(directory)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return directory, runs


def arguments(directory, run, name, task="pcp", failure=False, sham=False):
    compositions = [(3, 0)] if task == "pp" else [(2, 1)]
    if task == "pcp" and not failure:
        compositions += [(1, 2), (3, 2)]
    scenarios = []
    for index, composition in enumerate(compositions):
        scenarios.extend(make_scenarios(61 + index, 2, [composition], float(failure), (1, 1)))
    path = directory / f"{name}-scenarios.json"
    path.write_text(json.dumps([asdict(s) for s in scenarios]))
    return SimpleNamespace(run_dir=run, checkpoint=run / "checkpoint.pt", scenarios=path,
                           output=directory / f"{name}.json", protocol=None, sham=sham, trace=True)


@pytest.fixture(scope="module")
def evaluations(archived_runs):
    directory, runs = archived_runs
    reports = {}
    for (task, spec, version), run in runs.items():
        reports[(task, spec, version)] = evaluate(arguments(directory, run, run.name, task))
    run = runs[("pcp", "supplement-v1", "corrected-v1")]
    failure_args = arguments(directory, run, "failure", failure=True)
    failure = evaluate(failure_args)
    sham_args = SimpleNamespace(**vars(failure_args))
    sham_args.sham = True
    sham_args.output = directory / "sham.json"
    sham = evaluate(sham_args)
    replay_args = SimpleNamespace(**vars(failure_args))
    replay_args.output = directory / "replay.json"
    replay = evaluate(replay_args)
    return reports, failure, sham, replay


@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
@pytest.mark.parametrize("spec", ["public-code-v1", "supplement-v1"])
def test_archived_frozen_models_cover_native_domains_and_pcp_singleton_transfer(evaluations, task, spec):
    report = evaluations[0][(task, spec, "corrected-v1")]
    assert report["parameters_unchanged"]
    assert report["model_config"] == spec
    assert report["config"]["learner_spec"] == "public-code-v1"
    assert report["config"]["rng_scheme"] == ("seedsequence-v1" if spec == "supplement-v1" else "legacy-offset-v1")
    assert report["environment_version"].startswith("publication-corrected-v1:")
    assert all("/source/runtime/" in path for path in report["evaluator"]["runtime"]["imported_paths"].values())
    expected = {(2, 1), (1, 2), (3, 2)} if task == "pcp" else ({(3, 0)} if task == "pp" else {(2, 1)})
    assert {tuple(e["composition"]) for e in report["per_episode"]} == expected
    assert all(1 <= e["steps"] <= 4 and np.isfinite(e["team_return"]) for e in report["per_episode"])
    assert all(all(action < 5 for action in t["actions"][:e["num_p"]])
               for e in report["per_episode"] for t in e["trace"])


def test_legacy_v1_checkpoint_and_historical_environment_stay_explicit(evaluations):
    report = evaluations[0][("pcp", "public-code-v1", "historical-2022")]
    assert report["model_config"] == "public-code-v1"
    assert report["environment_version"] == "publication-historical-2022:predator_capture"


def test_failure_and_sham_have_identical_prefixes_diagnostics_and_exact_replay(evaluations):
    _, failure, sham, replay = evaluations
    # The replay has its own immutable archive location; every scientific
    # result and content identity must be identical.
    left, right = json.loads(json.dumps(failure)), json.loads(json.dumps(replay))
    left["evaluator"]["source_archive"].pop("path")
    right["evaluator"]["source_archive"].pop("path")
    assert left == right
    assert failure["model_signature"] == sham["model_signature"]
    assert failure["scenarios"] == sham["scenarios"]
    for loss, control in zip(failure["per_episode"], sham["per_episode"]):
        assert loss["trace"][:1] == control["trace"][:1]
        assert loss["initial_state"] == control["initial_state"]
        assert loss["event_exposed"] == loss["scheduled_event_exposed"]
        assert not control["event_exposed"]
        for name in ("scheduled_event_exposed", "pre_event_victim_reached",
                     "pre_event_victim_target_visible", "pre_event_victim_target_seen"):
            assert loss[name] == control[name]


def test_mask_preserves_positions_classes_of_other_agents_and_source_array():
    original = np.arange(3 * 25 * 29).reshape(3, 25, 29)
    before = original.copy()
    masked = sensory_mask(original, 1, 25)
    assert np.array_equal(original, before)
    assert np.array_equal(masked[[0, 2]], original[[0, 2]])
    assert np.array_equal(masked[1, :, :25], original[1, :, :25])
    assert np.count_nonzero(masked[1, :, 25:]) == 0


def test_pcp_physical_initial_states_match_softrole_with_same_scenarios(evaluations):
    from softrole.env import EnvironmentAdapter
    report = evaluations[0][("pcp", "public-code-v1", "corrected-v1")]
    for row, outcome in zip(report["scenarios"], report["per_episode"]):
        env = EnvironmentAdapter(report["config"])
        env.reset(row["env_seed"], row["num_p"], row["num_a"])
        assert outcome["initial_state"] == {"P": env.raw.predator_loc.tolist(),
                                           "A": env.raw.predator_capture_loc.tolist(),
                                           "targets": env.raw.prey_loc.tolist()}


def test_fresh_output_and_full_source_inventory_are_required(archived_runs, tmp_path):
    directory, runs = archived_runs
    run = next(iter(runs.values()))
    args = arguments(tmp_path, run, "existing", task="pp")
    args.output.write_text("preserve")
    with pytest.raises(FileExistsError):
        evaluate(args)
    assert args.output.read_text() == "preserve"
    copied = tmp_path / "run"
    shutil.copytree(run, copied)
    (copied / "source/runtime/injected.py").write_text("raise AssertionError('shadowed')")
    with pytest.raises(ValueError, match="inventory"):
        source_identity(copied)


def test_input_mutation_during_evaluation_never_writes_report(archived_runs, tmp_path, monkeypatch):
    import publication_reconstruction.evaluation as evaluator
    _, runs = archived_runs
    run = runs[("pcp", "public-code-v1", "corrected-v1")]
    args = arguments(tmp_path, run, "mutating")
    real_run = evaluator.subprocess.run
    def mutate_panel(*positional, **kwargs):
        result = real_run(*positional, **kwargs)
        args.scenarios.write_text(args.scenarios.read_text() + " ")
        return result
    monkeypatch.setattr(evaluator.subprocess, "run", mutate_panel)
    with pytest.raises(RuntimeError, match="changed"):
        evaluate(args)
    assert not args.output.exists()


def test_new_reports_feed_paired_summary_without_schema_translation(archived_runs, evaluations):
    from softrole.report import summarize_reports
    directory, _ = archived_runs
    paired = summarize_reports([directory / "failure.json", directory / "sham.json"],
                               bootstrap_samples=50)["paired_groups"]
    assert len(paired) == 1
    assert paired[0]["training_seeds"] == 1 and paired[0]["episodes"] == 2
    assert all(m["ci95"] is None for m in paired[0]["metrics"].values())


def test_explicit_protocol_and_zero_step_diagnostics(archived_runs, tmp_path):
    from softrole.report import protocol_identity
    _, runs = archived_runs
    run = runs[("pcp", "public-code-v1", "corrected-v1")]
    args = arguments(tmp_path, run, "at-zero", failure=True)
    scenarios = [asdict(s) for s in make_scenarios(81, 2, [(2, 1)], 1., (0, 0))]
    args.scenarios.write_text(json.dumps(scenarios))
    selection = {"rule": "first_saved_at_or_above", "target_steps": 10}
    distribution = {"task": "pcp", "compositions": [[2, 1]], "episodes_per_composition": 2,
                    "scenario_seed": 81, "failure_probability": 1., "failure_window": [0, 0],
                    "victim_rule": "uniform_sensing_agent"}
    protocol = {"selection": selection, "distribution": distribution,
                "protocol_id": protocol_identity(selection, distribution),
                "scenario_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest()}
    args.protocol = tmp_path / "protocol.json"
    args.protocol.write_text(json.dumps(protocol))
    report = evaluate(args)
    assert report["evaluation_protocol"] == protocol
    assert all(e["event_exposed"] and e["pre_event_victim_target_seen"] is False
               and e["post_event_steps"] == e["steps"] for e in report["per_episode"])


def test_fc_sensor_events_are_rejected_before_policy_evaluation(archived_runs, tmp_path):
    _, runs = archived_runs
    run = runs[("fc", "public-code-v1", "corrected-v1")]
    args = arguments(tmp_path, run, "invalid-fc", task="fc")
    scenarios = json.loads(args.scenarios.read_text())
    scenarios[0].update(event_step=0, victim=0)
    args.scenarios.write_text(json.dumps(scenarios))
    with pytest.raises(ValueError, match="Sensor failures require a PCP"):
        evaluate(args)
    assert not args.output.exists()


def test_evaluator_archive_preserves_exact_helpers_and_rejects_corruption(evaluations, tmp_path):
    from publication_reconstruction.evaluation import verify_evaluator_archive
    report = evaluations[1]
    source = report["evaluator"]["source"]
    archive = report["evaluator"]["source_archive"]
    original = Path(archive["path"])
    assert verify_evaluator_archive(original, source) == archive["manifest_sha256"]
    required = {"publication_reconstruction/evaluation.py", "publication_reconstruction/evaluation_worker.py",
                "publication_reconstruction/artifacts.py", "publication_reconstruction/__main__.py",
                "publication_reconstruction/runtime/hetnet_ext/recovery.py", "softrole/report.py"}
    assert required.issubset(source["files"])
    copied = tmp_path / "corrupt-archive"
    shutil.copytree(original, copied)
    helper = copied / "softrole/report.py"
    helper.write_bytes(helper.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        verify_evaluator_archive(copied, source)
    # Replay must reject corruption itself, before executing the changed helper.
    failed = subprocess.run([sys.executable, "-I", "-B", str(copied / "publication_reconstruction/evaluation.py"),
                             "--run-dir", report["checkpoint"].rsplit("/", 1)[0],
                             "--checkpoint", report["checkpoint"],
                             "--scenarios", str(tmp_path / "unused.json"),
                             "--output", str(tmp_path / "not-written.json")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert failed.returncode != 0
    assert "archive manifest differs" in failed.stderr
    assert not (tmp_path / "not-written.json").exists()


def test_existing_evaluator_archive_blocks_without_touching_files(archived_runs, tmp_path):
    from publication_reconstruction.evaluation import archive_path
    _, runs = archived_runs
    run = runs[("pcp", "public-code-v1", "corrected-v1")]
    args = arguments(tmp_path, run, "existing-archive")
    archive = archive_path(args.output)
    archive.mkdir()
    (archive / "keep").write_text("preserve")
    with pytest.raises(FileExistsError):
        evaluate(args)
    assert (archive / "keep").read_text() == "preserve"
    assert not args.output.exists()


def test_staged_archive_corruption_prevents_publication(archived_runs, tmp_path, monkeypatch):
    import publication_reconstruction.evaluation as evaluator
    _, runs = archived_runs
    run = runs[("pcp", "public-code-v1", "corrected-v1")]
    args = arguments(tmp_path, run, "corrupt-staged")
    real_run = evaluator.subprocess.run
    def corrupt_helper(command, **kwargs):
        result = real_run(command, **kwargs)
        staged = Path(command[-1]).parent.parent
        helper = staged / "softrole/report.py"
        helper.write_bytes(helper.read_bytes() + b"\n# changed\n")
        return result
    monkeypatch.setattr(evaluator.subprocess, "run", corrupt_helper)
    with pytest.raises(ValueError, match="archive hash mismatch"):
        evaluate(args)
    assert not args.output.exists()
    assert not evaluator.archive_path(args.output).exists()


def test_captured_bytes_must_match_declared_identity_before_execution(tmp_path):
    from publication_reconstruction.evaluation import evaluator_snapshot, write_evaluator_archive
    identity, payloads = evaluator_snapshot()
    payloads["softrole/report.py"] += b"\n# changed\n"
    archive = tmp_path / "mismatched"
    with pytest.raises(ValueError, match="Captured evaluator bytes"):
        write_evaluator_archive(archive, identity, payloads)
    assert not archive.exists()


def test_published_archive_replays_without_current_checkout_imports(archived_runs, evaluations, tmp_path):
    directory, runs = archived_runs
    original = evaluations[1]
    archive = Path(original["evaluator"]["source_archive"]["path"])
    run = runs[("pcp", "supplement-v1", "corrected-v1")]
    output = tmp_path / "archived-replay.json"
    result = subprocess.run([sys.executable, "-I", "-B", str(archive / "publication_reconstruction/evaluation.py"),
                             "--run-dir", str(run), "--checkpoint", str(run / "checkpoint.pt"),
                             "--scenarios", str(directory / "failure-scenarios.json"),
                             "--output", str(output), "--trace"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    replay = json.loads(output.read_text())
    assert replay["per_episode"] == original["per_episode"]
    assert replay["evaluator"]["source"] == original["evaluator"]["source"]
    assert replay["evaluator"]["source_archive"]["manifest_sha256"] == original["evaluator"]["source_archive"]["manifest_sha256"]
