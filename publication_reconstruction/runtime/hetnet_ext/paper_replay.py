"""Recorded-trajectory compute comparison for paper-v1 DGL and Torch backends.

This is a SERIAL replay of one/four collector partitions, not a multiprocessing
throughput benchmark. Real DGL collection creates the fixed trajectories before
timing. Replay excludes environment/action sampling/IPC; each forward restores
its recorded RNG boundary so Binary messages receive identical random draws.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from types import SimpleNamespace

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME / "envs"))
sys.path.insert(0, str(RUNTIME))

import numpy as np
import torch

from hetgat.policy import A2CPolicy
from hetnet_ext.paper_learning import normalize_and_clip_gradients
from hetnet_ext.recovery import capture_rng, restore_rng, seed_stream, process_resources
from trainer import Trainer


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone(item) for item in value)
    return copy.deepcopy(value)


def compare(left, right, path="root", *, exact=False, maxima=None):
    """Fail closed; tolerances are fixed before measurements, never adjusted."""
    maxima = {} if maxima is None else maxima
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape:
            raise AssertionError(f"Tensor schema differs: {path}")
        if left.is_floating_point() and (
                not torch.isfinite(left).all() or not torch.isfinite(right).all()):
            raise AssertionError(f"Non-finite tensor: {path}")
        if exact or not left.is_floating_point():
            if not torch.equal(left, right):
                raise AssertionError(f"Exact tensor mismatch: {path}")
        else:
            tolerances = dict(atol=1e-6, rtol=1e-5) if left.dtype == torch.float32 else dict(atol=1e-10, rtol=1e-8)
            torch.testing.assert_close(left, right, **tolerances, msg=path)
            if left.numel():
                key = str(left.dtype)
                maxima[key] = max(maxima.get(key, 0.0), float((left - right).abs().max()))
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(f"Array mismatch: {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Dictionary schema differs: {path}")
        for key in left:
            compare(left[key], right[key], f"{path}.{key}", exact=exact, maxima=maxima)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"Sequence length differs: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]", exact=exact, maxima=maxima)
    elif left != right:
        raise AssertionError(f"Value mismatch: {path}")
    return maxima


def configuration(task, variant, collectors, batch_steps, horizon, seed):
    from ic3net_envs.predator_capture_env import PredatorCaptureEnv
    from ic3net_envs.fire_commander_env import FireCommanderEnv
    parser = argparse.ArgumentParser(add_help=False)
    (FireCommanderEnv if task == 'fc' else PredatorCaptureEnv)().init_args(parser)
    args = parser.parse_args([])
    vars(args).update(model_spec='paper-v1', reconstruction_spec='paper-v1',
        publication_env_version='paper-v1', learner_spec='paper-equations-v1',
        message_backend='dgl', nprocesses=collectors, batch_size=batch_steps,
        max_steps=horizon, seed=seed, use_binary=variant == 'binary',
        nfriendly_P=3 if task == 'pp' else 2, nfriendly_A=0 if task == 'pp' else 1,
        nfriendly=3, nagents=3, dim=5, vision=1 if task == 'fc' else 2,
        env_name='fire_commander' if task == 'fc' else 'predator_capture',
        hetcomm=False, recurrent=False, hetgat=True, hetgat_a2c=True, hard_attn=False,
        commnet=False, display=False, eval=False, tensor_obs=False, profile_phases=False,
        profile_memory=False, hid_size=128, lrate=.001, detach_gap=5, continuous=False,
        num_actions=[5 if task == 'fc' else 6], dim_actions=1, enemy_comm=False)
    return args


def make_trainer(args, backend, *, environment=False):
    if environment:
        import data
        env = data.init(args.env_name, args, False)
    else:
        env = None
    policy = A2CPolicy(dict(vision=args.vision, P=29, A=25, state=4),
        dict(P=29, A=25, state=4), dict(P=16, A=16, state=16),
        dict(P=4 if args.env_name == 'fire_commander' else 5,
             A=5 if args.env_name == 'fire_commander' else 6, state=8),
        args.nfriendly_P, args.nfriendly_A, num_heads=4, msg_dim=64,
        device=torch.device('cpu'), gamma=1, lr=.001, weight_decay=0,
        use_real=not args.use_binary, use_CNN=False, per_class_critic=True,
        with_two_state=True, obs=(2 * args.vision + 1) ** 2, tensor_obs=False,
        model_spec='paper-v1', message_backend=backend, learner_spec='paper-equations-v1',
        total_state_action_in_batch=max(500, args.batch_size))
    return Trainer(args, policy.model, env, policy)


def share_parameter_storage(trainers):
    """Match live shared weights while retaining independent collector grads.

    Configure storage before any forward/timing. These models have no buffers;
    cached topology belongs to each model and is neither learned nor shared.
    """
    parent = dict(trainers[0].policy_net.named_parameters())
    for trainer in trainers:
        if list(trainer.policy_net.buffers()):
            raise ValueError('Replay sharing needs an explicit policy for model buffers')
    for trainer in trainers[1:]:
        parameters = dict(trainer.policy_net.named_parameters())
        if parent.keys() != parameters.keys():
            raise ValueError('Replay collector parameter schemas differ')
        for name, parameter in parameters.items():
            reference = parent[name]
            if parameter.shape != reference.shape or parameter.dtype != reference.dtype:
                raise ValueError('Replay collector parameter tensor schemas differ')
            parameter.data = reference.detach()


def finish_update(trainers, batches, *, capture=False):
    gradients, losses = [], []
    for trainer, episodes in zip(trainers, batches):
        loss = trainer.policy.batch_finish_per_class(len(episodes), trainer.args.max_steps,
                    trainer.args.nfriendly_P, trainer.args.nfriendly_A)
        if capture:
            gradients.append([None if p.grad is None else p.grad.detach().clone()
                              for p in trainer.params])
            losses.append({key: torch.as_tensor(value).clone() for key, value in loss.items()})
    parent = trainers[0]
    for index, parameter in enumerate(parent.params):
        pieces = [trainer.params[index].grad for trainer in trainers]
        if all(piece is None for piece in pieces):
            parameter.grad = None
        elif any(piece is None for piece in pieces):
            raise AssertionError('Collector gradient presence differs')
        else:
            # Same collector order as the live parent; maintain grad storage.
            for piece in pieces[1:]:
                parameter.grad.add_(piece)
    norm = normalize_and_clip_gradients(parent.params, sum(map(len, batches)))
    clipped = [None if p.grad is None else p.grad.detach().clone()
               for p in parent.params] if capture else None
    parent.optimizer.step()
    if capture:
        return dict(collector_losses=losses, collector_gradients=gradients,
                    clipped_gradients=clipped, gradient_norm=norm.detach().clone(),
                    model=clone(parent.policy_net.state_dict()),
                    optimizer=clone(parent.optimizer.state_dict()))


def record_fixture(args, updates):
    seed_stream(args.seed)
    trainers = [make_trainer(args, 'dgl', environment=True) for _ in range(args.nprocesses)]
    initial = clone(trainers[0].policy_net.state_dict())
    initial_optimizer = clone(trainers[0].optimizer.state_dict())
    streams = []
    for trainer in trainers:
        trainer.policy_net.load_state_dict(initial, strict=True)
        seed_stream(args.seed, len(streams))
        streams.append(capture_rng())
    share_parameter_storage(trainers)
    records = []
    for update in range(updates):
        batches = []
        for collector, trainer in enumerate(trainers):
            restore_rng(streams[collector])
            steps = []
            select, append = trainer.policy.batch_select_action_universal, trainer.policy.append_log_probs_properly

            def recording_select(x, index):
                steps.append(dict(observation=clone(x[0]), rng_before=capture_rng()))
                output = select(x, index)
                steps[-1]['rng_after_forward'] = capture_rng()
                return output

            def recording_append(actual):
                steps[-1].update(action=clone(actual),
                    reward=clone(trainer.policy.batch_rewards[trainer.policy.i_b][-1]))
                return append(actual)

            trainer.policy.batch_select_action_universal = recording_select
            trainer.policy.append_log_probs_properly = recording_append
            try:
                _, stat = trainer.run_batch(update // 10)
            finally:
                trainer.policy.batch_select_action_universal = select
                trainer.policy.append_log_probs_properly = append
            streams[collector] = capture_rng()
            episodes, offset = [], 0
            for row in stat['_episode_records']:
                episodes.append(steps[offset:offset + row['steps']])
                offset += row['steps']
            assert offset == len(steps)
            batches.append(episodes)
        finish_update(trainers, batches)
        records.append(batches)
    return dict(args=vars(args), initial_model=initial,
                initial_optimizer=initial_optimizer, updates=records, final_model=clone(trainers[0].policy_net.state_dict()),
                final_optimizer=clone(trainers[0].optimizer.state_dict()), final_environment_streams=streams)


def replay(fixture, backend, *, capture=False):
    args = SimpleNamespace(**fixture['args'])
    trainers = [make_trainer(args, backend) for _ in range(args.nprocesses)]
    for trainer in trainers:
        trainer.policy_net.load_state_dict(fixture['initial_model'], strict=True)
        trainer.optimizer.load_state_dict(fixture['initial_optimizer'])
        assert not trainer.optimizer.state
    share_parameter_storage(trainers)
    trace, intervals = [], []
    before = process_resources()
    start = time.perf_counter()
    for batches in fixture['updates']:
        update_begin = time.perf_counter()
        forwards = []
        for trainer, episodes in zip(trainers, batches):
            for episode_index, episode in enumerate(episodes):
                hidden = trainer.policy_net.init_hidden(1)
                for step_index, step in enumerate(episode):
                    restore_rng(step['rng_before'])
                    trainer.policy_net.set_episode_step(step_index)
                    logits, value, hidden = trainer.policy.batch_select_action_universal(
                        [step['observation'], hidden], episode_index)
                    if capture:
                        # Compare all random libraries at the forward boundary;
                        # action/environment draws are fixed inputs, outside replay.
                        compare(step['rng_after_forward'], capture_rng(), exact=True)
                        forwards.append(clone((logits, value,
                            trainer.policy.batch_A_critics[episode_index][-1], hidden)))
                    if (step_index + 1) % args.detach_gap == 0:
                        hidden = {key: tuple(value.detach() for value in pair)
                                  for key, pair in hidden.items()}
                    trainer.policy.batch_rewards[episode_index].append(step['reward'])
                    trainer.policy.append_log_probs_properly(step['action'])
        update = finish_update(trainers, batches, capture=capture)
        intervals.append(time.perf_counter() - update_begin)
        if capture:
            update['forwards'] = forwards
            trace.append(update)
    elapsed = time.perf_counter() - start
    after = process_resources()
    steps = sum(len(episode) for batches in fixture['updates'] for episodes in batches for episode in episodes)
    summary = dict(backend=backend, wall_seconds=elapsed, update_seconds=intervals,
        first_update_seconds=intervals[0], steps=steps, steps_per_second=steps / elapsed,
        user_cpu_seconds=after['user_cpu_seconds'] - before['user_cpu_seconds'],
        system_cpu_seconds=after['system_cpu_seconds'] - before['system_cpu_seconds'],
        max_rss=after['max_rss'], max_rss_unit=after['max_rss_unit'])
    return trace, summary


def run_replay(output, task, variant, collectors=4, updates=3, batch_steps=500,
               horizon=None, timing_repeats=3, seed=991):
    if task not in ('pp', 'pcp', 'fc') or variant not in ('real', 'binary'):
        raise ValueError('Unsupported replay workload')
    if collectors not in (1, 4) or updates < 3 or batch_steps < 1 or timing_repeats < 1:
        raise ValueError('Replay requires one/four collectors, at least three updates and positive budgets')
    horizon = (300 if task == 'fc' else 80) if horizon is None else horizon
    if horizon < 1:
        raise ValueError('Replay horizon must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    args = configuration(task, variant, collectors, batch_steps, horizon, seed)
    source = {path.relative_to(RUNTIME).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(RUNTIME.rglob('*.py'))}
    started = time.perf_counter()
    fixture = record_fixture(args, updates)
    fixture_seconds = time.perf_counter() - started
    fixture_path = output / 'fixture.pt'
    torch.save(fixture, fixture_path)
    dgl, _ = replay(fixture, 'dgl', capture=True)
    compare(fixture['final_model'], dgl[-1]['model'], exact=True)
    compare(fixture['final_optimizer'], dgl[-1]['optimizer'], exact=True)
    accelerated, _ = replay(fixture, 'torch-v1', capture=True)
    maxima = compare(dgl, accelerated)
    del dgl, accelerated
    pairs = []
    for repetition in range(timing_repeats):
        order = ('dgl', 'torch-v1') if repetition % 2 == 0 else ('torch-v1', 'dgl')
        rows = {}
        for backend in order:
            _, rows[backend] = replay(fixture, backend)
        pairs.append(dict(repetition=repetition + 1, order=list(order), runs=rows,
                          speedup=rows['dgl']['wall_seconds'] / rows['torch-v1']['wall_seconds']))
    current = {path.relative_to(RUNTIME).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(RUNTIME.rglob('*.py'))}
    if source != current:
        raise RuntimeError('Runtime source changed during replay validation')
    report = dict(schema_version=1, task=task, variant=variant, collectors=collectors,
        updates=updates, batch_step_floor_per_collector=batch_steps, horizon=horizon, seed=seed,
        semantics='serial collector compute replay; fixed observations/actions/rewards; excludes environment, action sampling and IPC',
        timing_overhead='includes per-forward RNG restoration and Python replay iteration; weights share storage before timing; no diagnostic tensor copies or per-update weight broadcasts',
        rng_semantics='all RNG libraries restored before each forward and compared after forward in correctness runs',
        tolerances={'float64': dict(atol=1e-10, rtol=1e-8), 'float32': dict(atol=1e-6, rtol=1e-5)},
        correctness_passed=True, maximum_absolute_differences=maxima, fixture_generation_seconds=fixture_seconds,
        fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(), source_sha256=source,
        python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
        resources=process_resources(), pairs=pairs)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['pp', 'pcp', 'fc'], required=True)
    parser.add_argument('--variant', choices=['real', 'binary'], required=True)
    parser.add_argument('--collectors', type=int, choices=[1, 4], default=4)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--batch-steps', type=int, default=500)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=991)
    report = run_replay(**vars(parser.parse_args()))
    print(json.dumps({'correctness_passed': report['correctness_passed'],
                      'speedups': [pair['speedup'] for pair in report['pairs']]}, sort_keys=True))


if __name__ == '__main__':
    main()
