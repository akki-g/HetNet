"""Bounded real paper-learner updates and exact same-backend continuation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "publication_reconstruction/runtime"


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def invoke(base, name, *, task, binary, collectors, backend, resume=None, profiled=False):
    output = base / name
    output.mkdir()
    manifest = base / "source_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"purpose": "bounded paper learner regression", "files": {
            path.relative_to(RUNTIME).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in RUNTIME.rglob("*.py")}}))
    command = [sys.executable, "-u", str(RUNTIME / "main.py"),
        "--reconstruction_spec", "paper-v1", "--model_spec", "paper-v1",
        "--publication_env_version", "paper-v1", "--learner_spec", "paper-equations-v1",
        "--message_backend", backend, "--source_manifest", str(manifest),
        "--env_name", "fire_commander" if task == "fc" else "predator_capture",
        "--nfriendly_P", "3" if task == "pp" else "2", "--nfriendly_A", "0" if task == "pp" else "1",
        "--nagents", "3", "--hetgat", "--hetgat_a2c", "--seed", "719",
        "--nprocesses", str(collectors), "--num_epochs", "1", "--epoch_size", "3",
        "--batch_size", "7", "--max_steps", "6", "--detach_gap", "5", "--dim", "5",
        "--vision", "1" if task == "fc" else "2", "--hid_size", "128", "--lrate", ".001",
        "--msg_dim", "64", "--save_every", "1", "--milestones", "1",
        "--save_dir", str(output / "checkpoints"), "--metrics_file", str(output / "metrics.jsonl"),
        "--experiment_name", "paper_test", "--episode_log", "file"]
    if task == "fc":
        command += ["--nfires", "1", "--reward_type", "3"]
    if binary:
        command += ["--use_binary"]
    if resume:
        command += ["--resume_checkpoint", str(resume)]
    if profiled:
        command += ["--profile_phases"]
    environment = {**os.environ, "DGLBACKEND": "pytorch", "OPENBLAS_NUM_THREADS": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                               text=True, timeout=120)
    (output / "console.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-6000:]
    status = json.loads((output / "run_status.json").read_text())
    checkpoint = torch.load(status["checkpoint"], map_location="cpu")
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    return output, checkpoint, updates


@pytest.fixture(scope="module", params=[
    ("pp", False, 1), ("pcp", False, 4), ("fc", False, 1), ("pcp", True, 4),
])
def paired_updates(request, tmp_path_factory):
    task, binary, collectors = request.param
    base = tmp_path_factory.mktemp("paper-training")
    options = dict(task=task, binary=binary, collectors=collectors)
    result = {}
    for backend in ("dgl", "torch-v1"):
        output, full, updates = invoke(base, backend, backend=backend, **options)
        milestone = next((output / "checkpoints").rglob("*_steps1.pt"))
        original_bytes = milestone.read_bytes()
        _, resumed, continued_updates = invoke(base, backend + "_continued", backend=backend,
                                              resume=milestone, **options)
        assert milestone.read_bytes() == original_bytes
        result[backend] = dict(output=output, full=full, resumed=resumed,
                               updates=updates, continued_updates=continued_updates)
    return result


def test_short_updates_resume_every_tensor_optimizer_and_rng_exactly(paired_updates):
    for records in paired_updates.values():
        full, resumed = records['full'], records['resumed']
        for key in ('policy_net', 'trainer', 'log'):
            same(full[key], resumed[key])
        for key in ('counts', 'rng_states', 'recorder_state', 'milestones_reached'):
            same(full['recovery'][key], resumed['recovery'][key])
        assert full['recovery']['counts']['updates'] == 3
        assert full['trainer']['state']
        assert all(torch.isfinite(value).all() for value in full['policy_net'].values())
        for state in full['trainer']['state'].values():
            for value in state.values():
                assert torch.isfinite(value).all()
        assert records['continued_updates'][0]['update_in_epoch'] == 2


def test_backend_short_update_losses_and_parameters_are_close(paired_updates):
    left, right = (paired_updates[backend] for backend in ('dgl', 'torch-v1'))
    # Actions and resulting episodes may diverge in general; these locked short
    # fixtures stay coupled and give a complete optimizer-level regression.
    for a, b in zip(left['updates'], right['updates']):
        for key in ('steps', 'episodes', 'total_steps', 'total_episodes', 'reward_per_agent'):
            same(a[key], b[key])
        for key in ('policy_loss', 'value_loss'):
            assert a[key] == pytest.approx(b[key], rel=1e-8, abs=1e-10)
    same(left['full']['recovery']['rng_states'], right['full']['recovery']['rng_states'])
    for name, expected in left['full']['policy_net'].items():
        tolerance = dict(rtol=1e-5, atol=1e-6) if expected.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(expected, right['full']['policy_net'][name], **tolerance)


def test_profiling_is_random_and_numerically_neutral(tmp_path):
    options = dict(task='pcp', binary=True, collectors=1, backend='torch-v1')
    _, reference, _ = invoke(tmp_path, 'unprofiled', **options)
    _, profiled, updates = invoke(tmp_path, 'profiled', profiled=True, **options)
    for key in ('policy_net', 'trainer', 'log'):
        same(reference[key], profiled[key])
    same(reference['recovery']['rng_states'], profiled['recovery']['rng_states'])
    phase_names = ('graph_preparation', 'model_inference', 'environment', 'loss_backward',
                   'collector_wait', 'aggregation', 'optimizer')
    for update in updates:
        phases = update['phases']
        for name in phase_names:
            assert phases['phase_' + name + '_seconds'] >= 0
        assert phases['phase_model_inference_seconds'] > 0 and phases['phase_loss_backward_seconds'] > 0
