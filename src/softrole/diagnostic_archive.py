"""Isolated numerical runtime for the audited 911488 Adam archive only.

This is an explicit source profile, not a compatibility bypass. Every archived
file must match the known complete manifest before any archived Python executes.
The numerical modules execute the verified bytes in a private namespace. The
returned overrides are installed by the diagnostic caller for the context's
lifetime; this module never replaces the current ``softrole`` modules.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import uuid


ARCHIVED_COMMIT = '692b88d3634920bf7c4b348840344c0ef21b410d'
ARCHIVED_SOURCE_SHA256 = '4c5dfbd6b9d519d99c999c638abb2d9f0679293e90d7ad235a099ea18a9e876e'
# Both the complete manifest identity and these executed numerical files are
# pinned. Source snapshots with merely similar model layouts are not admitted.
NUMERICAL_HASHES = {
    'softrole/__init__.py': '065591ee37498f71ecb7363077ba94b7976a84171f1c6894acd9235a0ebf646b',
    'softrole/config.py': '6d7fded19a2fa13cd4f29d637b7215f71d5ac2f817c514a9524dcd4a9f389aa4',
    'softrole/model.py': '08bb9f2b24d1549861e9b766adb3c1946072f4725c9b26f2a8c438f08a41889d',
    'softrole/learning.py': 'a3779797064d443d6a87fe51fb8d6e2f04b99a4ccdf86743553a7f528a08a280',
    'softrole/rollout.py': 'f5beaa75cf54c079daaf7927c5723c970a606416fad827cb069897ef9952bcba',
    'softrole/scenarios.py': 'c9f68a1cf7e4f9b4d2194478a783c0789fc2bd7741819613bbde6335d82cc503',
    'softrole/env.py': '385c3c6032964b82e01bcb181e34d8fd3e6b4d832d0616b73b127db0ffa65515',
    'softrole/train.py': 'c0523780cfb593969fdfd6e7e587cd9b50faf6d5dcf472afe06fbfa391f6d9ee',
    'softrole/optimizers.py': '00f7049caa7f32cb990ee0abbfd596c9ec82b9083aa00a941f1722393a992629',
    'envs/ic3net_envs/predator_capture_env.py': '935875ac665877965f86a2e12f5ef52669065d3a2a01ba11b0ed065bfac59583',
    'hetnet_ext/signatures.py': '4f92b26e2cf4d4ad7fa6af8990adfed39b4293cc7ef91e0b258376d48cd7cf4a',
}


def _verified_source(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run = checkpoint.parent.parent
    root = run / 'source'
    manifest = json.loads((run / 'source_manifest.json').read_text())
    identity = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    if (identity != ARCHIVED_SOURCE_SHA256 or manifest['sha256'] != identity
            or manifest.get('git_commit') != ARCHIVED_COMMIT):
        raise ValueError('Unknown archived Adam source profile; only audited job 911488 is supported')
    if any(manifest['files'].get(name) != digest for name, digest in NUMERICAL_HASHES.items()):
        raise ValueError('Archived Adam numerical source does not match the pinned profile')
    verified = {}
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid archived source path: {name}')
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Archived source hash mismatch: {name}')
        if name in NUMERICAL_HASHES:
            verified[name] = data
    config = json.loads((run / 'config.json').read_text())
    if (config.get('task') != 'pcp' or config.get('env_version') != 'corrected-observation-v1'
            or config.get('optimizer') != 'adam' or config.get('model') != 'shared'
            or config.get('failure_prob') != 0 or config.get('compositions')):
        raise ValueError('Archived Adam runtime is restricted to the nominal shared PCP Adam study')
    return root, manifest, verified


@contextmanager
def archived_adam_runtime(checkpoint):
    """Yield archived numerical symbols and provenance without changing globals.

    Config/model subclasses add metadata for diagnostic interfaces only. They
    inherit the old constructor, forward, state-dict layout, recurrence and loss
    unchanged. The environment and collector use the archived PCP implementation.
    No training or checkpoint loading occurs here. Use one context per selected
    checkpoint; nested contexts also have distinct module namespaces.
    """
    root, manifest, verified = _verified_source(checkpoint)
    namespace = '_softrole_adam_archive_' + uuid.uuid4().hex
    registered = []
    loaded = {}

    def package(name, path):
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module
        registered.append(name)
        return module

    def load(suffix, relative):
        name = namespace + '.' + suffix
        path = root / relative
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        registered.append(name)
        # Compile the already checked bytes: never re-read executable code after
        # verification, and never create __pycache__ inside an input archive.
        exec(compile(verified[relative], str(path), 'exec'), module.__dict__)
        loaded[relative] = module
        return module

    try:
        package(namespace, root)
        initial = load('softrole', 'softrole/__init__.py')
        config_module = load('softrole.config', 'softrole/config.py')
        model_module = load('softrole.model', 'softrole/model.py')
        learning = load('softrole.learning', 'softrole/learning.py')
        scenarios = load('softrole.scenarios', 'softrole/scenarios.py')
        optimizers = load('softrole.optimizers', 'softrole/optimizers.py')
        simulator = load('predator_capture_env', 'envs/ic3net_envs/predator_capture_env.py')
        signatures = load('signatures', 'hetnet_ext/signatures.py')
        environment = load('softrole.env', 'softrole/env.py')
        environment.PredatorCaptureEnv = simulator.PredatorCaptureEnv
        rollout = load('softrole.rollout', 'softrole/rollout.py')
        rollout.Episode = learning.Episode
        train = load('softrole.train', 'softrole/train.py')

        class Config(config_module.Config):
            communication = 'binary'
            comm_rounds = 2
            independent_heads = False
            critic_mode = 'team'
            credit_beta = 1.0
            objective = 'team'
            actor_capabilities = 'explicit'
            null_degree_correction = False

        class SoftRoleNet(model_module.SoftRoleNet):
            communication = 'binary'
            comm_rounds = 2
            independent_heads = False
            critic_mode = 'team'
            actor_capabilities = 'explicit'
            null_degree_correction = False

        def make_env(config, environment_source=None):
            if (config.task != 'pcp' or config.env_version != 'corrected-observation-v1'
                    or environment_source is not None):
                raise ValueError('Archived Adam runtime supports native PCP only')
            return environment.make_env(config)

        # These old modules use absolute softrole imports. Bind those numerical
        # dependencies to the corresponding private archived modules before any
        # collector executes, leaving the current modules untouched.
        train.CHECKPOINT_VERSION = initial.CHECKPOINT_VERSION
        train.Config = Config
        train.SoftRoleNet = SoftRoleNet
        train.make_env = make_env
        train.loss_sum = learning.loss_sum
        train.make_optimizer = optimizers.make_optimizer
        train.optimizer_spec = optimizers.optimizer_spec
        train.validate_optimizer_resume = optimizers.validate_optimizer_resume
        train.run_episode = rollout.run_episode
        train.make_scenarios = scenarios.make_scenarios
        train.model_signature = signatures.model_signature
        provenance = {
            'profile': 'adam-pcp-911488',
            'git_commit': ARCHIVED_COMMIT,
            'source_sha256': manifest['sha256'],
            'numerical_root': str(root),
            'module_paths': {name: module.__file__ for name, module in loaded.items()},
            'numerical_file_sha256': dict(NUMERICAL_HASHES),
            'compatibility': 'Config/model metadata attributes only; archived numerical functions unchanged',
        }
        yield {
            'Config': Config, 'SoftRoleNet': SoftRoleNet, 'make_env': make_env,
            'run_episode': rollout.run_episode,
            'compute_episode_loss': learning.compute_episode_loss,
            'team_gae': learning.team_gae, 'make_scenarios': scenarios.make_scenarios,
            'apply_gradients': train.apply_gradients,
            'collect_batch': train.collect_batch,
            'make_optimizer': optimizers.make_optimizer,
            'validate_optimizer_resume': optimizers.validate_optimizer_resume,
            'numerical_root': root, 'provenance': provenance,
        }
    finally:
        for name in reversed(registered):
            sys.modules.pop(name, None)
