"""Compatibility checks for the archived Adam team's original loss protocol."""
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from softrole.diagnostic_estimators import actor_episode_gradients
from softrole.learning import episode_loss_terms


@dataclass
class LegacyEpisode:
    """The five-field archived Episode protocol, without later agent records."""
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    traces: list = field(default_factory=list)


def legacy_config():
    # No critic_mode, credit_beta, objective, communication or other new fields.
    return SimpleNamespace(gamma=1., gae_lambda=.5, actor_coeff=50., value_coeff=1.)


class LegacyScores(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([-.4, .7], dtype=torch.float64))
        self.value = nn.Parameter(torch.tensor([10., 4., 1.], dtype=torch.float64))


def legacy_sample():
    model = LegacyScores()
    actions = torch.tensor([[0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    scores = torch.distributions.Bernoulli(logits=model.logits).log_prob(actions)
    rewards = torch.tensor([[1., 0.], [0., 2.], [1., 2.]], dtype=torch.float64)
    episode = LegacyEpisode(log_probs=list(scores.sum(-1)), values=list(model.value),
                            rewards=list(rewards.sum(-1)), metrics={'num_agents': 2})
    return model, scores, rewards, episode


def flatten_grad(loss, model, retain_graph=False):
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)])


def test_current_loss_split_accepts_original_episode_and_config_protocol():
    model, scores, _, episode = legacy_sample()
    cfg = legacy_config()
    assert not hasattr(episode, 'agent_values') and not hasattr(episode, 'agent_rewards')
    assert not hasattr(cfg, 'critic_mode') and not hasattr(cfg, 'objective')
    terms = episode_loss_terms(episode, cfg)
    # Independent two/three-step lambda-return mixture for r=(1,2,3),
    # V=(10,4,1), lambda=.5: targets=(5,4,3), advantages=(-5,0,2).
    expected_advantages = model.value.new_tensor([-5., 0., 2.])
    expected_actor = -(scores.sum(-1) * expected_advantages).sum()
    expected_value = (model.value - model.value.new_tensor([5., 4., 3.])).square().mean()
    assert terms['scale'] == 1.
    assert torch.equal(terms['advantages'], expected_advantages)
    assert torch.equal(terms['policy'], expected_actor)
    assert torch.equal(terms['value'], expected_value)
    separated = (flatten_grad(cfg.actor_coeff * terms['policy'], model, True)
                 + flatten_grad(cfg.value_coeff * terms['value'], model, True))
    combined = flatten_grad(cfg.actor_coeff * expected_actor + cfg.value_coeff * expected_value, model)
    assert torch.equal(separated, combined)


def test_current_d4_keeps_old_team_objective_and_has_no_learned_critic_variant():
    model, scores, rewards, episode = legacy_sample()
    cfg = legacy_config()
    terms = episode_loss_terms(episode, cfg)
    expected_actor = flatten_grad(cfg.actor_coeff * terms['policy'], model, True)
    result = actor_episode_gradients(scores, model.value, rewards, model, cfg)
    assert torch.equal(result['gradients']['team_gae'], expected_actor)
    assert torch.allclose(result['gradients']['allocated_gae_beta_1'], expected_actor, atol=1e-12)
    assert result['protocol']['production_variant'] == 'team_gae'
    assert result['protocol']['learned'] is None
    assert result['checks']['episode_scale'] == 1.
    assert not any(name.startswith('learned_') for name in result['gradients'])
    assert all(parameter.grad is None for parameter in model.parameters())


ARCHIVE_CHECKPOINT = (Path(__file__).resolve().parents[1] / 'stokes_runs' /
                      'softrole_adam_pcp_911488' / 'pcp_shared' / 'seed0' /
                      'checkpoints' / 'epoch1000.pt')


@pytest.mark.skipif(not ARCHIVE_CHECKPOINT.exists(), reason='Historical Adam archive is an optional local input')
def test_actual_archived_loss_and_collector_match_diagnostic_actor_value_split():
    """A three-step compatibility sample, not a claim of historical-batch replay."""
    import copy
    import hashlib

    import numpy as np

    from softrole.diagnostic_archive import archived_adam_runtime
    from softrole.diagnostics import CaptureAdapter

    before = hashlib.sha256(ARCHIVE_CHECKPOINT.read_bytes()).hexdigest()
    saved = torch.load(ARCHIVE_CHECKPOINT, map_location='cpu', weights_only=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(), archived_adam_runtime(ARCHIVE_CHECKPOINT) as runtime:
            cfg = runtime['Config'](**{**saved['config'], 'max_steps': 3, 'batch_steps': 1, 'nprocesses': 1})
            assert 'critic_mode' not in cfg.to_dict()
            assert 'objective' not in cfg.to_dict()
            model = runtime['SoftRoleNet'](**saved['model_config']).double()
            model.load_state_dict(saved['model_state'], strict=True)
            state = {name: value.clone() for name, value in model.state_dict().items()}
            collector_model = copy.deepcopy(model)
            seed = int(np.random.SeedSequence([cfg.seed, saved['updates'], 0, 2026]).generate_state(1)[0])
            scenario = runtime['make_scenarios'](seed, 1, cfg.training_compositions)[0]
            adapter = CaptureAdapter(runtime['make_env'](cfg))
            logits = []
            handle = model.register_forward_hook(lambda _m, _args, out: logits.append(out['logits']))
            try:
                episode = runtime['run_episode'](model, adapter, cfg, scenario, training=True)
            finally:
                handle.remove()
            assert not hasattr(episode, 'agent_values')
            assert not hasattr(episode, 'agent_log_probs')
            actions = torch.stack(adapter.actions)
            scores = torch.stack([torch.distributions.Categorical(logits=x).log_prob(a)
                                  for x, a in zip(logits, actions)])
            rewards = torch.as_tensor(np.stack(adapter.rewards), dtype=torch.float64)
            terms = episode_loss_terms(episode, cfg)
            old_total, _ = runtime['compute_episode_loss'](episode, cfg)
            split_total = cfg.actor_coeff * terms['policy'] + cfg.value_coeff * terms['value']
            assert torch.equal(old_total, split_total)
            actor = flatten_grad(cfg.actor_coeff * terms['policy'], model, True)
            value = flatten_grad(cfg.value_coeff * terms['value'], model, True)
            d4 = actor_episode_gradients(scores, torch.stack(episode.values), rewards, model, cfg, True)
            old_gradient = flatten_grad(old_total, model)
            assert torch.equal(d4['gradients']['team_gae'], actor)
            assert torch.allclose(actor + value, old_gradient, rtol=1e-12, atol=1e-10)
            batch = runtime['collect_batch'](collector_model, cfg, 0, saved['updates'])
            native_gradient = torch.cat([torch.as_tensor(part).flatten() for part in batch['gradients']])
            assert batch['episodes'] == [episode.metrics]
            assert torch.equal(old_gradient, native_gradient)
            assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
            assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        torch.set_num_threads(threads)
    assert hashlib.sha256(ARCHIVE_CHECKPOINT.read_bytes()).hexdigest() == before


RUNTIME_SYMBOLS = ('Config', 'SoftRoleNet', 'make_env', 'run_episode',
                   'compute_episode_loss', 'team_gae', 'make_scenarios',
                   'apply_gradients', 'make_optimizer', 'validate_optimizer_resume')


def mocked_archived_loader(monkeypatch, tmp_path):
    from contextlib import contextmanager

    import softrole.diagnostic_archive as archive

    symbols = {name: object() for name in RUNTIME_SYMBOLS}
    symbols.update(numerical_root=tmp_path / 'source', provenance={'profile': 'test-archive', 'source_sha256': 'checked'})
    calls = []

    @contextmanager
    def load(checkpoint):
        calls.append(('enter', checkpoint))
        try:
            yield symbols
        finally:
            calls.append(('exit', checkpoint))

    monkeypatch.setattr(archive, 'archived_adam_runtime', load)
    return symbols, calls


@pytest.mark.parametrize('raises', [False, True])
def test_archived_numerical_context_routes_symbols_and_restores_on_exit(tmp_path, monkeypatch, raises):
    import softrole.diagnostics as diagnostics

    symbols, calls = mocked_archived_loader(monkeypatch, tmp_path)
    original = {name: getattr(diagnostics, name) for name in (*RUNTIME_SYMBOLS, 'NUMERICAL_ROOT')}
    loss_builder = diagnostics.episode_loss_terms
    checkpoint = tmp_path / 'checkpoint.pt'

    def use_context():
        with diagnostics.numerical_runtime('archived-adam', checkpoint) as provenance:
            for name in RUNTIME_SYMBOLS:
                assert getattr(diagnostics, name) is symbols[name]
            assert diagnostics.NUMERICAL_ROOT == symbols['numerical_root']
            assert diagnostics.episode_loss_terms is loss_builder
            assert provenance == {'runtime': 'archived-adam', **symbols['provenance']}
            if raises:
                raise RuntimeError('diagnostic failed')

    if raises:
        with pytest.raises(RuntimeError, match='diagnostic failed'):
            use_context()
    else:
        use_context()
    assert calls == [('enter', checkpoint), ('exit', checkpoint)]
    assert all(getattr(diagnostics, name) is value for name, value in original.items())


def test_legacy_manifest_without_runtime_selector_keeps_current_backend(tmp_path, monkeypatch):
    import json

    import softrole.diagnostic_archive as archive
    import softrole.diagnostics as diagnostics

    manifest = tmp_path / 'legacy.json'
    manifest.write_text(json.dumps({'version': 1, 'entries': [{'checkpoint': 'old.pt'}]}))
    original_model = diagnostics.SoftRoleNet

    def forbidden_loader(_checkpoint):
        raise AssertionError('A legacy manifest must not select archived numerical code implicitly')

    def capture(path, index, output, dry_run, provenance):
        assert diagnostics.SoftRoleNet is original_model
        assert (path, index, output, dry_run) == (manifest, 0, tmp_path / 'out', True)
        return provenance

    monkeypatch.setattr(archive, 'archived_adam_runtime', forbidden_loader)
    monkeypatch.setattr(diagnostics, '_execute', capture)
    assert diagnostics.execute(manifest, 0, tmp_path / 'out', dry_run=True) == {'runtime': 'current'}


@pytest.mark.parametrize('matches', [False, True])
def test_execute_uses_explicit_archive_and_checks_prepared_provenance(tmp_path, monkeypatch, matches):
    import json

    import softrole.diagnostics as diagnostics

    symbols, calls = mocked_archived_loader(monkeypatch, tmp_path)
    actual = {'runtime': 'archived-adam', **symbols['provenance']}
    locked = actual if matches else {**actual, 'source_sha256': 'different'}
    manifest = tmp_path / 'archived.json'
    manifest.write_text(json.dumps({'version': 1, 'numerical_runtime': 'archived-adam',
                                   'entries': [{'checkpoint': 'old.pt', 'numerical_runtime_provenance': locked}]}))
    original = diagnostics.Config
    executions = []

    def capture(_path, _index, _output, _dry_run, provenance):
        assert diagnostics.Config is symbols['Config']
        executions.append(provenance)
        return 'executed'

    monkeypatch.setattr(diagnostics, '_execute', capture)
    if matches:
        assert diagnostics.execute(manifest, 0, tmp_path / 'out', dry_run=True) == 'executed'
        assert executions == [actual]
    else:
        with pytest.raises(ValueError, match='differs from prepared manifest'):
            diagnostics.execute(manifest, 0, tmp_path / 'out', dry_run=True)
        assert executions == []
    assert calls == [('enter', 'old.pt'), ('exit', 'old.pt')]
    assert diagnostics.Config is original
