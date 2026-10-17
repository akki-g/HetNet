"""Offline D1--D6 probes. All optimizer steps affect disposable model copies only.

Run ``python -m softrole.diagnostics prepare --help`` to lock inputs first.
Historical claims require an exact next-update episode-ledger replay. A terminal
checkpoint instead needs an explicitly selected, labelled checkpoint-probe batch.

Protocol version 1 is the nominal fixed-2P1A, team-learner protocol and is
preserved unchanged. Version 2 admits composition- and failure-trained
checkpoints and the programme learner variants: it builds D1 from the shared
production loss-term builder, records each episode's actual composition, adds
learned per-agent D3/D4 records for decomposed critics (the V/N allocation
stays labelled separately) and applies D5 only to stochastic Binary bits.
"""
import argparse
import ast
import copy
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

# The direct preparation CLI also runs on login nodes; cap BLAS before imports.
if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import torch

from hetnet_ext.signatures import model_signature
from softrole import CHECKPOINT_VERSION
from softrole.config import Config
from softrole.env import make_env
from softrole.learning import compute_episode_loss, episode_loss_terms, team_gae
from softrole.lr_diagnostics import clone, policy_statistics
from softrole.model import SoftRoleNet
from softrole.optimizers import make_optimizer, validate_optimizer_resume
from softrole.rollout import run_episode
from softrole.scenarios import make_scenarios
from softrole.train import apply_gradients, source_snapshot, write_json

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_ROOT = ROOT
RUNTIMES = ('current', 'archived-adam')
COMPOSITIONS = ((1, 2), (2, 2), (3, 1))
PROTOCOL_VERSIONS = (1, 2)
LEARNER_FIELDS = ('communication', 'null_degree_correction', 'critic_mode', 'credit_beta',
                  'objective', 'actor_capabilities')
# Strict numerical-source guard; diagnostic additions need not match old archives.
CORE = tuple('softrole/' + name + '.py' for name in
             ('config', 'env', 'model', 'learning', 'optimizers', 'rollout', 'scenarios')) + (
                 'envs/ic3net_envs/predator_capture_env.py',)


@contextmanager
def numerical_runtime(runtime, checkpoint):
    """Select verified numerical code for this serial offline diagnostic task."""
    if runtime not in RUNTIMES:
        raise ValueError('Unknown diagnostic numerical runtime')
    if runtime == 'current':
        yield {'runtime': 'current'}
        return
    from softrole.diagnostic_archive import archived_adam_runtime
    with archived_adam_runtime(checkpoint) as archived:
        names = ('Config', 'SoftRoleNet', 'make_env', 'run_episode',
                 'compute_episode_loss', 'team_gae', 'make_scenarios',
                 'apply_gradients', 'make_optimizer', 'validate_optimizer_resume')
        replacements = {name: archived[name] for name in names}
        replacements['NUMERICAL_ROOT'] = archived['numerical_root']
        original = {name: globals()[name] for name in replacements}
        try:
            globals().update(replacements)
            yield {'runtime': runtime, **archived['provenance']}
        finally:
            globals().update(original)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def learner_identity(cfg):
    return {name: getattr(cfg, name) for name in LEARNER_FIELDS}


def training_study(cfg):
    """Classify scheduled sensor loss first, then fixed versus mixed rosters."""
    if cfg.failure_prob:
        return 'failure'
    return 'fixed' if len(cfg.training_compositions) == 1 else 'composition'


def version_one_eligible(cfg):
    return (not cfg.failure_prob and cfg.training_compositions == ((2, 1),)
            and cfg.critic_mode == 'team'
            and cfg.credit_beta == 1 and cfg.objective == 'team')


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def load_checkpoint(path, protocol_version=1):
    if protocol_version not in PROTOCOL_VERSIONS:
        raise ValueError('Unsupported diagnostic protocol version')
    path = Path(path).resolve()
    data = path.read_bytes()
    saved = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    if saved.get('format_version') != CHECKPOINT_VERSION:
        raise ValueError('Unsupported checkpoint format')
    cfg = Config(**saved['config'])
    if cfg.task != 'pcp' or cfg.env_version != 'corrected-observation-v1':
        raise ValueError('D1--D6 require native PCP checkpoints')
    if protocol_version == 1 and not version_one_eligible(cfg):
        raise ValueError('D1--D6 require nominal fixed 2P1A native PCP checkpoints with the team learner; '
                         'composition/failure training and programme learner variants need --protocol-version 2')
    for composition in COMPOSITIONS:
        cfg.validate_composition(composition)
    if cfg.max_steps < 2:
        raise ValueError('D6 needs a horizon of at least two steps')
    if saved['model_config'] != cfg.model_kwargs() or saved['environment_version'] != cfg.env_version:
        raise ValueError('Checkpoint layout/environment disagrees with config')
    with torch.random.fork_rng():
        model = SoftRoleNet(**saved['model_config']).double()
    model.load_state_dict(saved['model_state'], strict=True)
    if any(p.dtype != torch.float64 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Checkpoint requires finite float64 parameters')
    if model_signature(model) != saved['signature']:
        raise ValueError('Checkpoint tensor signature mismatch')
    if read_json(str(path) + '.signature.json') != saved['signature']:
        raise ValueError('Checkpoint signature sidecar mismatch')
    validate_optimizer_resume(cfg, saved)
    groups = saved['optimizer_state']['param_groups']
    states = saved['optimizer_state']['state']
    parameters = list(model.parameters())
    identifiers = [i for group in groups for i in group['params']]
    if (len(groups) != 1 or len(identifiers) != len(parameters)
            or len(set(identifiers)) != len(identifiers) or set(states) != set(identifiers)):
        raise ValueError('Incomplete or misaligned optimizer state')
    for identifier, parameter in zip(identifiers, parameters):
        for name, tensor in states[identifier].items():
            if name == 'step':
                if ((isinstance(tensor, torch.Tensor) and tensor.numel() != 1)
                        or not math.isfinite(float(tensor)) or float(tensor) != saved['updates']):
                    raise ValueError('Optimizer step disagrees with checkpoint progress')
                continue
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError('Optimizer state must contain finite tensors')
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError('Optimizer moment shape/dtype differs from parameter')
            elif name in ('square_avg', 'exp_avg_sq') and (tensor < 0).any():
                raise ValueError('Optimizer squared moments cannot be negative')
    for name in ('updates', 'total_steps', 'total_episodes'):
        if type(saved[name]) is not int or saved[name] < 1:
            raise ValueError('Invalid checkpoint progress')
    return saved, cfg, model, hashlib.sha256(data).hexdigest()


def audit_archive(checkpoint, saved, cfg):
    run = Path(checkpoint).resolve().parent.parent
    manifest = read_json(run / 'source_manifest.json')
    identity = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    if identity != manifest['sha256'] or identity != saved['source_sha256']:
        raise ValueError('Checkpoint/source manifest identity mismatch')
    for name, expected in manifest['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha(run / 'source' / name) != expected:
            raise ValueError(f'Archived source hash mismatch: {name}')
    for name in CORE:
        if manifest['files'].get(name) != sha(NUMERICAL_ROOT / name):
            raise ValueError(f'Numerical source differs from archive: {name}; audit/reconcile before replay')
    # New read-only diagnostic hooks in train() do not alter these numerical
    # functions. Compare their ASTs so unrelated logging additions are allowed.
    for function in ('collect_batch', 'apply_gradients'):
        def function_ast(path):
            tree = ast.parse(path.read_text())
            return ast.dump(next(node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name == function))
        if function_ast(run / 'source' / 'softrole/train.py') != function_ast(NUMERICAL_ROOT / 'softrole/train.py'):
            raise ValueError(f'Numerical training function differs from archive: {function}')
    if Config(**read_json(run / 'config.json')).to_dict() != cfg.to_dict():
        raise ValueError('Run/checkpoint configuration mismatch')
    record = read_json(run / 'run.json')
    if record['source_sha256'] != identity or record['environment_version'] != cfg.env_version:
        raise ValueError('Run/source identity mismatch')
    if record['torch'] != torch.__version__ or record['numpy'] != np.__version__:
        raise ValueError('Replay requires recorded Torch/NumPy versions')
    return run


def logged_batch(run, update):
    from softrole.ledger import iter_episodes
    episodes = []
    for row in iter_episodes(run, through_update=update):
        if row['update'] == update:
            episodes.append(row)
    updates = []
    with (run / 'updates.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['update'] == update:
                updates.append(row)
    if not episodes or len(updates) != 1:
        raise ValueError('No complete next-update ledger: choose an earlier checkpoint, or explicitly prepare --batch-mode checkpoint-probe')
    row = updates[0]
    if row['episodes'] != len(episodes) or row['steps'] != sum(e['steps'] for e in episodes):
        raise ValueError('Next-update ledger counts disagree')
    return episodes, row


def prepare(checkpoints, output, checkpoint_rule, batch_mode='logged-next-update',
            probe_episodes=100, seed=2711, d5_instances=8, d5_bits=3, d5_samples=256,
            protocol_version=1, runtime='current'):
    torch.set_num_threads(1)
    if protocol_version not in PROTOCOL_VERSIONS:
        raise ValueError('Unsupported diagnostic protocol version')
    if runtime not in RUNTIMES:
        raise ValueError('Unknown diagnostic numerical runtime')
    if runtime == 'archived-adam' and protocol_version != 1:
        raise ValueError('Archived Adam requires diagnostic protocol version 1')
    if not checkpoint_rule.strip():
        raise ValueError('Declare the checkpoint selection rule before outcomes')
    if batch_mode not in ('logged-next-update', 'checkpoint-probe'):
        raise ValueError('Unknown batch mode')
    for value in (probe_episodes, d5_instances, d5_bits, d5_samples):
        if type(value) is not int or value < 1:
            raise ValueError('Diagnostic counts must be positive integers')
    if not 2 <= d5_samples <= 65536:
        raise ValueError('D5 samples must be between 2 and 65536')
    if d5_bits > 4 or type(seed) is not int or seed < 0:
        raise ValueError('Use at most four enumerated bits and a nonnegative panel seed')
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    entries = []
    seen = set()
    for checkpoint in checkpoints:
        checkpoint = Path(checkpoint).resolve()
        with numerical_runtime(runtime, checkpoint) as provenance:
            saved, cfg, _, digest = load_checkpoint(checkpoint, protocol_version)
            run = audit_archive(checkpoint, saved, cfg)
        if digest in seen:
            raise ValueError('Duplicate checkpoint')
        seen.add(digest)
        inputs = {str(checkpoint): digest}
        for name in ('config.json', 'run.json', 'source_manifest.json'):
            inputs[str(run / name)] = sha(run / name)
        inputs[str(checkpoint) + '.signature.json'] = sha(str(checkpoint) + '.signature.json')
        if batch_mode == 'logged-next-update':
            from softrole.ledger import ledger_files
            logged_batch(run, saved['updates'] + 1)
            for path in [*ledger_files(run), run / 'updates.jsonl']:
                inputs[str(path)] = sha(path)
        entry = {'checkpoint': str(checkpoint), 'inputs': inputs,
                 'training_seed': cfg.seed, 'communication': cfg.communication,
                 'progress': {key: saved[key] for key in ('updates', 'total_steps', 'total_episodes')},
                 'model': cfg.model, 'source_sha256': saved['source_sha256']}
        if runtime != 'current':
            entry['numerical_runtime_provenance'] = provenance
        if protocol_version >= 2:
            entry.update(learner=learner_identity(cfg), training_study=training_study(cfg),
                         training_compositions=[list(c) for c in cfg.training_compositions],
                         failure_prob=cfg.failure_prob, failure_window=list(cfg.failure_window))
        entries.append(entry)
    if not entries:
        raise ValueError('Select at least one checkpoint')
    manifest = {'version': protocol_version, 'checkpoint_rule': checkpoint_rule, 'batch_mode': batch_mode,
                'numerical_runtime': runtime,
                'probe_episodes_per_composition': probe_episodes, 'panel_seed': seed,
                'd5_instances': d5_instances, 'd5_bits': d5_bits, 'd5_samples': d5_samples,
                'compositions': [list(c) for c in COMPOSITIONS], 'event_step': 1,
                'entries': entries, 'evaluator_source_sha256': source_snapshot()['sha256']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, manifest)
    return manifest


class CaptureAdapter:
    def __init__(self, adapter):
        self.adapter, self.rewards, self.actions = adapter, [], []

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def step(self, actions, kappa):
        result = self.adapter.step(actions, kappa)
        self.actions.append(torch.as_tensor(actions).clone())
        self.rewards.append(np.array(result[1], copy=True))
        return result


def capture_episode(model, cfg, scenario):
    steps, logits = [], []
    adapter = CaptureAdapter(make_env(cfg))

    def capture(module, args, kwargs, out):
        # Copy before rollout writes a_prev into the returned memory.
        steps.append({'args': clone(args), 'kwargs': clone(kwargs),
                      'logits': clone(out['logits']), 'features': clone(out['memory']['u_bar'])})
        logits.append(out['logits'])

    handle = model.register_forward_hook(capture, with_kwargs=True)
    try:
        episode = run_episode(model, adapter, cfg, scenario, training=True)
    finally:
        handle.remove()
    actions = torch.stack(adapter.actions)
    log_probs = torch.stack([torch.distributions.Categorical(logits=x).log_prob(a)
                             for x, a in zip(logits, actions)])
    for step, action in zip(steps, actions):
        step['actions'] = action
    return episode, log_probs, torch.as_tensor(np.stack(adapter.rewards), dtype=torch.float64), steps


def flat_grad(loss, model, retain_graph=False):
    params = tuple(model.parameters())
    gradients = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten()
                      for p, g in zip(params, gradients)])


def cosine(a, b):
    denom = float(a.norm() * b.norm())
    return float(a @ b) / denom if denom else None


@torch.no_grad()
def policy_replay(model, replays):
    sums = {mode: dict(kl=0., entropy_before=0., entropy_after=0., agent_steps=0)
            for mode in ('recomputed', 'teacher_forced')}
    for replay in replays:
        memory = model.initial_memory(replay[0]['args'][0].shape[0])
        for step in replay:
            obs, kappa, old_memory, adjacency = step['args']
            memory['a_prev'] = old_memory['a_prev']
            out = model(obs, kappa, memory, adjacency, **step['kwargs'])
            memory = out['memory']
            forced = model.output(step['features']).masked_fill(~torch.isfinite(step['logits']), -torch.inf)
            for mode, logits in (('recomputed', out['logits']), ('teacher_forced', forced)):
                for key, value in policy_statistics(step['logits'], logits).items():
                    sums[mode][key] += value
    return {mode: {k: v if k == 'agent_steps' else v / row['agent_steps'] for k, v in row.items()}
            for mode, row in sums.items()}


def batch_diagnostics(saved, cfg, model, manifest, run):
    from softrole.diagnostic_estimators import (actor_episode_gradients, GradientMoments,
        agent_critic_episode_diagnostics, critic_episode_diagnostics,
        summarize_agent_critic_diagnostics, summarize_critic_diagnostics)
    from softrole.diagnostic_surrogate import audit_surrogate
    logged, logged_update = (logged_batch(run, saved['updates'] + 1)
                             if manifest['batch_mode'] == 'logged-next-update' else (None, None))
    total_actor = torch.zeros(sum(p.numel() for p in model.parameters()), dtype=torch.float64)
    total_value = torch.zeros_like(total_actor)
    moments, critic, replays, records, audits = GradientMoments(), [], [], [], []
    learned_critic, checks = [], []
    batches = []
    for collector in range(cfg.nprocesses):
        seed = int(np.random.SeedSequence([cfg.seed, saved['updates'], collector, 2026]).generate_state(1)[0])
        scenarios = make_scenarios(seed, cfg.batch_steps, cfg.training_compositions,
                                   cfg.failure_prob, cfg.failure_window)
        gradients = [torch.zeros_like(p) for p in model.parameters()]
        episodes, num_steps = [], 0
        for scenario in scenarios:
            episode, log_probs, rewards, steps = capture_episode(model, cfg, scenario)
            record = dict(episode.metrics, update=saved['updates'] + 1, collector=collector)
            if logged is not None and (len(records) >= len(logged) or record != logged[len(records)]):
                raise ValueError('Exact archived episode replay failed; no historical D1/D2 claim is valid')
            records.append(record)
            values = torch.stack(episode.values)
            # The production loss-term builder gives the exact actor/value split
            # for every learner variant (team or decomposed critic, credit mixing,
            # mean-agent scale); the total below is the production episode loss.
            terms = episode_loss_terms(episode, cfg)
            scale = terms['scale']
            actor = scale * cfg.actor_coeff * terms['policy']
            value = scale * cfg.value_coeff * terms['value']
            total_actor += flat_grad(actor, model, retain_graph=True)
            total_value += flat_grad(value, model, retain_graph=True)
            per_agent_values = getattr(episode, 'agent_values', None)
            agent_values = torch.stack(per_agent_values).detach() if per_agent_values else None
            estimators = actor_episode_gradients(log_probs, values, rewards, model, cfg, retain_graph=True,
                                                 agent_values=agent_values, scale=scale)
            moments.add(estimators['gradients'])
            checks.append(estimators['checks'])
            composition = list(episode.metrics['composition'])
            critic.append({'composition': composition, 'steps': len(steps),
                           'diagnostics': critic_episode_diagnostics(values, rewards,
                                      gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)})
            if agent_values is not None:
                learned_critic.append({'composition': composition, 'steps': len(steps),
                                       'diagnostics': agent_critic_episode_diagnostics(
                                           agent_values, rewards, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)})
            loss, _ = compute_episode_loss(episode, cfg)
            grad = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
            for accumulated, fresh in zip(gradients, grad):
                if fresh is not None:
                    accumulated.add_(fresh.detach())
            if cfg.communication == 'binary' and len(audits) < manifest['d5_instances']:
                # Predetermined earliest eligible episode/step, not selected on the cosine.
                # Only stochastic Binary bits have a score function; the production
                # per-agent actor weights at step 0 (team advantage, or the mixed
                # decomposed advantage) and the episode scale define the objective.
                step = steps[0]
                obs, kappa, memory, adjacency = step['args']
                weights = terms['actor_weights'][0].detach()
                step_advantages = (weights.expand(len(kappa)) if weights.ndim == 0 else weights) * (scale * cfg.actor_coeff)
                audits.append(audit_surrogate(model, obs=obs, kappa=kappa, memory=memory,
                    adjacency=adjacency, remaining=step['kwargs']['remaining'], noise=step['kwargs']['noise'],
                    actions=step['actions'], advantages=step_advantages,
                    n_bits=manifest['d5_bits'], mc_samples=manifest['d5_samples'],
                    seed=manifest['panel_seed'] + len(audits)))
            replays.append(steps)
            episodes.append(episode.metrics)
            num_steps += episode.metrics['steps']
            if num_steps >= cfg.batch_steps:
                break
        batches.append({'gradients': [g.numpy() for g in gradients], 'num_episodes': len(episodes),
                        'num_steps': num_steps, 'episodes': episodes, 'collector': collector})
    if logged is not None and len(records) != len(logged):
        raise ValueError('Archived batch has unexpected extra episodes')
    count = len(records)
    total_actor /= count
    total_value /= count
    original_signature = model_signature(model)
    candidate = copy.deepcopy(model)
    optimizer = make_optimizer(candidate.parameters(), cfg)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']))
    norm = apply_gradients(candidate, optimizer, batches, cfg.max_grad_norm)
    if not np.isclose(norm, float((total_actor + total_value).norm()), rtol=1e-10, atol=1e-10):
        raise RuntimeError('Actor/value decomposition differs from production gradient')
    if logged_update is not None:
        if (not np.isclose(norm, logged_update['gradient_norm_before_clip'], rtol=1e-10, atol=1e-10)
                or saved['total_steps'] + sum(b['num_steps'] for b in batches) != logged_update['total_steps']):
            raise ValueError('Replayed gradient/progress differs from archived update')
    groups = {}
    for (name, before), after in zip(model.named_parameters(), candidate.parameters()):
        group = name.split('.')[0]
        groups[group] = groups.get(group, 0.) + float((after.detach() - before.detach()).square().sum())
    if model_signature(model) != original_signature:
        raise RuntimeError('Diagnostics changed the input policy')
    d5_status = {'binary': 'conditional_binary_audit', 'real': 'not_applicable_real_channel',
                 'real_noisy': 'not_applicable_real_noisy_channel',
                 'binary_deterministic': 'not_applicable_deterministic_bits_have_no_score_function'}[cfg.communication]
    composition_counts = {}
    for record in records:
        key = ','.join(map(str, record['composition']))
        composition_counts[key] = composition_counts.get(key, 0) + 1
    return {'batch_mode': manifest['batch_mode'], 'update': saved['updates'] + 1,
            'episodes': records, 'episode_compositions': composition_counts,
            'learner': learner_identity(cfg), 'training_study': training_study(cfg),
            'D1': {'gradient_norm_before_clip': norm,
                'actor_coefficient': cfg.actor_coeff, 'value_coefficient': cfg.value_coeff,
                'weighted_actor_gradient_l2': float(total_actor.norm()),
                'weighted_value_gradient_l2': float(total_value.norm()),
                'actor_value_cosine': cosine(total_actor, total_value),
                'clip_active': norm > cfg.max_grad_norm,
                'parameter_update_l2': sum(groups.values()) ** .5,
                'parameter_group_update_l2': {k: v ** .5 for k, v in groups.items()},
                'disposable_post_update_signature': model_signature(candidate)['sha256'],
                'historical_update_verified': logged is not None},
            'D2': policy_replay(candidate, replays),
            'D3': summarize_critic_diagnostics(critic), 'D3_per_episode': critic,
            'D3_learned': summarize_agent_critic_diagnostics(learned_critic) if learned_critic else None,
            'D3_learned_per_episode': learned_critic,
            'D4': {**moments.summary(), 'protocol': estimators['protocol'],
                   'identity_checks_per_episode': checks},
            'D5': {'status': d5_status,
                   'instances': audits, 'requested_instances': manifest['d5_instances'],
                   'selection': 'first step of earliest batch episodes, up to requested count'}}


def d6_panels(seed, episodes):
    panels = []
    for index, composition in enumerate(COMPOSITIONS):
        base = make_scenarios(seed + index, episodes, [composition])
        # Explicit scenarios permit the proposal's 1P2A fully-blind diagnostic.
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 6006]))
        event = [replace(s, event_step=1, victim=int(rng.integers(composition[0]))) for s in base]
        panels.append((composition, base, event))
    return panels


def frozen_probes(model, cfg, manifest, output):
    rows = []
    panels = d6_panels(manifest['panel_seed'], manifest['probe_episodes_per_composition'])
    write_json(output / 'D6_scenarios.json', [dict(composition=list(c), nominal=[asdict(s) for s in b],
                                                    event=[asdict(s) for s in e]) for c, b, e in panels])
    with torch.no_grad():
        for composition, base, event in panels:
            outcomes = {}
            for condition, scenarios, sham in (('nominal', base, False), ('failure', event, False), ('sham', event, True)):
                outcomes[condition] = []
                for scenario in scenarios:
                    episode = run_episode(model, make_env(cfg), cfg, scenario, training=False,
                                          sham=sham, trace=True, event_diagnostics=True)
                    outcomes[condition].append(dict(episode.metrics, trace=episode.traces))
                records = outcomes[condition]
                row = {'composition': list(composition), 'condition': condition,
                       'assigned_episodes': len(records),
                       'success_rate': float(np.mean([r['success'] for r in records])),
                       'mean_team_return': float(np.mean([r['team_return'] for r in records])),
                       'mean_capped_completion': float(np.mean([r['steps'] if r['success'] else cfg.max_steps for r in records])),
                       'alpha_null_episode_mean': float(np.mean([r['alpha_null'] for r in records])),
                       'alpha_null_step_mean': sum(r['alpha_null'] * r['steps'] for r in records) / sum(r['steps'] for r in records),
                       'event_exposed': sum(r['event_exposed'] for r in records),
                       'scheduled_exposed': sum(r['scheduled_event_exposed'] for r in records),
                       'scheduled_censored': sum(r['scheduled_censored'] for r in records),
                       'pre_event_early_success': sum(r['pre_event_success'] for r in records)}
                for name in ('reached', 'target_visible', 'target_seen'):
                    key = 'pre_event_victim_' + name
                    row[key + '_true'] = sum(r[key] is True for r in records)
                    row[key + '_observed'] = sum(r[key] is not None for r in records)
                rows.append(row)
            deltas = []
            for nominal, failure, sham in zip(outcomes['nominal'], outcomes['failure'], outcomes['sham']):
                if nominal['trace'] != sham['trace'] or failure['trace'][:1] != sham['trace'][:1]:
                    raise RuntimeError('D6 sham or pre-event prefix does not match')
                for key in ('pre_event_victim_reached', 'pre_event_victim_target_visible', 'pre_event_victim_target_seen'):
                    if failure[key] != sham[key]:
                        raise RuntimeError('D6 pre-event diagnostics do not match')
                deltas.append({'scenario_id': failure['scenario_id'],
                    'success': int(failure['success']) - int(sham['success']),
                    'team_return': failure['team_return'] - sham['team_return'],
                    'capped_completion': (failure['steps'] if failure['success'] else cfg.max_steps) -
                                         (sham['steps'] if sham['success'] else cfg.max_steps)})
            write_json(output / ('D6_' + '_'.join(map(str, composition)) + '.json'),
                       {'outcomes': outcomes, 'paired_failure_minus_sham': deltas,
                        'paired_full_panel_mean': {k: float(np.mean([d[k] for d in deltas]))
                                                   for k in ('success', 'team_return', 'capped_completion')}})
    trained = [list(c) for c in COMPOSITIONS if c in cfg.training_compositions]
    event_trained = bool(cfg.failure_prob) and cfg.failure_window[0] <= 1 <= cfg.failure_window[1]
    return {'strata': rows,
            'training_support': {'training_study': training_study(cfg), 'compositions_in_training': trained,
                                 'step_one_sensor_loss_in_training': event_trained},
            'limits': ['Descriptive within-checkpoint paired probes; no pooled seed inference.',
            '1P2A loses its only sensor; this differs in fractional severity from 2P2A and 3P1A.',
            ('These teams are in composition-training support, outside nominal 2P1A training.' if not trained
             else f'Probe compositions inside this checkpoint\'s training support: {trained}; step-1 sensor loss '
                  f'{"is" if event_trained else "is not"} inside its training event distribution.'),
            'Reserved 3P2A, 4P1A and 4P2A are never evaluated.']}


def execute(manifest_path, index, output, dry_run=False):
    manifest = read_json(manifest_path)
    if type(index) is not int or not 0 <= index < len(manifest['entries']):
        raise ValueError('Array index outside manifest')
    runtime = manifest.get('numerical_runtime', 'current')
    if runtime == 'archived-adam' and manifest['version'] != 1:
        raise ValueError('Archived Adam requires diagnostic protocol version 1')
    entry = manifest['entries'][index]
    with numerical_runtime(runtime, entry['checkpoint']) as provenance:
        if runtime != 'current' and entry.get('numerical_runtime_provenance') != provenance:
            raise ValueError('Archived numerical runtime differs from prepared manifest')
        return _execute(manifest_path, index, output, dry_run, provenance)


def _execute(manifest_path, index, output, dry_run, provenance):
    torch.set_num_threads(1)
    manifest_digest = sha(manifest_path)
    manifest = read_json(manifest_path)
    if (manifest['version'] not in PROTOCOL_VERSIONS or manifest['compositions'] != [list(c) for c in COMPOSITIONS]
            or manifest['event_step'] != 1 or manifest['batch_mode'] not in ('logged-next-update', 'checkpoint-probe')):
        raise ValueError('Unsupported diagnostic protocol')
    if type(index) is not int or not 0 <= index < len(manifest['entries']):
        raise ValueError('Array index outside manifest')
    for key in ('probe_episodes_per_composition', 'd5_instances', 'd5_bits', 'd5_samples'):
        if type(manifest[key]) is not int or manifest[key] < 1:
            raise ValueError('Invalid diagnostic count')
    if not 2 <= manifest['d5_samples'] <= 65536:
        raise ValueError('D5 samples must be between 2 and 65536')
    if manifest['d5_bits'] > 4 or type(manifest['panel_seed']) is not int or manifest['panel_seed'] < 0:
        raise ValueError('Invalid bit count or seed')
    entry = manifest['entries'][index]
    for path, expected in entry['inputs'].items():
        if sha(path) != expected:
            raise ValueError(f'Locked input hash changed: {path}')
    saved, cfg, model, digest = load_checkpoint(entry['checkpoint'], manifest['version'])
    if digest != entry['inputs'][entry['checkpoint']]:
        raise ValueError('Checkpoint hash missing or changed')
    run = audit_archive(entry['checkpoint'], saved, cfg)
    if (entry['training_seed'] != cfg.seed or entry['communication'] != cfg.communication
            or entry['model'] != cfg.model or entry['source_sha256'] != saved['source_sha256']
            or entry['progress'] != {k: saved[k] for k in ('updates', 'total_steps', 'total_episodes')}):
        raise ValueError('Manifest entry metadata differs from checkpoint')
    if manifest['version'] >= 2 and (
            entry.get('learner') != learner_identity(cfg) or entry.get('training_study') != training_study(cfg)
            or entry.get('training_compositions') != [list(c) for c in cfg.training_compositions]
            or entry.get('failure_prob') != cfg.failure_prob or entry.get('failure_window') != list(cfg.failure_window)):
        raise ValueError('Manifest learner/training metadata differs from checkpoint')
    if source_snapshot()['sha256'] != manifest['evaluator_source_sha256']:
        raise ValueError('Evaluator checkout changed after manifest preparation; prepare on the final fixed checkout')
    if manifest['batch_mode'] == 'logged-next-update':
        logged_batch(run, saved['updates'] + 1)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    plan = {'index': index, 'checkpoint': entry['checkpoint'], 'output': str(output),
            'numerical_runtime': provenance,
            'batch_mode': manifest['batch_mode'], 'D6_policy_episodes': 9 * manifest['probe_episodes_per_composition'],
            'training_steps_persisted': 0}
    if dry_run:
        return plan
    output.mkdir(parents=True)
    source = source_snapshot(output)
    if source['sha256'] != manifest['evaluator_source_sha256']:
        raise RuntimeError('Evaluator source changed before archive capture')
    write_json(output / 'protocol.json', manifest)
    before = model_signature(model)
    report = batch_diagnostics(saved, cfg, model, manifest, run)
    report['D6'] = frozen_probes(model, cfg, manifest, output)
    if before != model_signature(model):
        raise RuntimeError('Frozen model changed')
    for path, expected in entry['inputs'].items():
        if sha(path) != expected:
            raise RuntimeError(f'Input changed during diagnostics: {path}')
    audit_archive(entry['checkpoint'], saved, cfg)
    if source_snapshot()['sha256'] != source['sha256'] or sha(manifest_path) != manifest_digest:
        raise RuntimeError('Source or manifest changed during diagnostics')
    report.update(plan=plan, protocol_version=manifest['version'], checkpoint_sha256=digest, model_signature=before['sha256'],
                  source_sha256=source['sha256'], manifest_sha256=manifest_digest,
                  config=cfg.to_dict(), training_source_sha256=saved['source_sha256'],
                  runtime={'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
                           'threads': torch.get_num_threads(), 'dtype': 'float64', 'device': 'cpu'},
                  limits=['D1/D2 describe one next update per selected checkpoint, not an every-update time series.',
                          'Checkpoint-probe batches are fresh diagnostic samples, not historical logged updates.',
                          'D3 uses MC residual proxies; D4 allocated baselines are not learned decomposed critics.',
                          'D5 enumerates a conditional bit subset, not the full sequential policy gradient.'])
    write_json(output / 'report.json', report)
    return plan


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'prepare':
        parser = argparse.ArgumentParser(description='Lock verified checkpoint archives and D1--D6 protocol')
        parser.add_argument('--checkpoints', nargs='+', required=True)
        parser.add_argument('--checkpoint-rule', required=True)
        parser.add_argument('--batch-mode', choices=('logged-next-update', 'checkpoint-probe'), default='logged-next-update')
        parser.add_argument('--probe-episodes', type=positive, default=100)
        parser.add_argument('--seed', type=int, default=2711)
        parser.add_argument('--d5-instances', type=positive, default=8)
        parser.add_argument('--d5-bits', type=positive, default=3)
        parser.add_argument('--d5-samples', type=positive, default=256)
        parser.add_argument('--protocol-version', type=int, choices=PROTOCOL_VERSIONS, default=1,
                            help='1: nominal fixed-2P1A team learner (unchanged); 2: composition/failure training and programme learner variants')
        parser.add_argument('--runtime', choices=RUNTIMES, default='current',
                            help='archived-adam: verified historical job 911488 numerical code; protocol 1 only')
        parser.add_argument('--output', required=True)
        args = vars(parser.parse_args(argv[1:]))
        result = prepare(**args)
        print(json.dumps({'entries': len(result['entries']), 'manifest': args['output']}))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--manifest', required=True)
        parser.add_argument('--index', type=int, required=True)
        parser.add_argument('--output', required=True)
        parser.add_argument('--dry-run', action='store_true')
        args = parser.parse_args(argv)
        print(json.dumps(execute(args.manifest, args.index, args.output, args.dry_run), indent=2))


if __name__ == '__main__':
    main()
