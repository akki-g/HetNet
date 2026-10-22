"""Exact archived replay admission. Execute directly in a fresh Python process."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

# Applies to spawned collectors too; archive inputs must never acquire bytecode.
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    path = Path(name)
    resolved = (root / path).resolve()
    if path.is_absolute() or '..' in path.parts or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe archived path: {name}')
    return resolved


def exact(left, right):
    """Compare every optimizer/model value, including tensor dtype and shape."""
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def verify_source(run):
    manifest = read(run / 'source_manifest.json')
    identity = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    if identity != manifest['sha256']:
        raise ValueError('Archived source manifest identity mismatch')
    for name, digest in manifest['files'].items():
        if sha(safe(run / 'source', name)) != digest:
            raise ValueError(f'Archived source hash mismatch: {name}')
    return manifest


def replay(run_dir, output, boundary_update, anchor_before, anchor_after):
    import numpy as np
    import torch
    run, output = Path(run_dir).resolve(), Path(output).resolve()
    before_path, after_path = Path(anchor_before).resolve(), Path(anchor_after).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing output: {output}')
    if output.is_relative_to(run):
        raise ValueError('Replay output must be outside the historical run')
    source = verify_source(run)
    record = read(run / 'run.json')
    if record['torch'] != torch.__version__ or record['numpy'] != np.__version__:
        raise ValueError('Replay requires exact recorded Torch/NumPy versions')
    if record['source_sha256'] != source['sha256']:
        raise ValueError('Run/source identity mismatch')
    root = run / 'source'
    # Absolute imports in the archive are intentional. Never reuse current modules.
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('softrole', 'hetnet_ext', 'envs'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(root):
                raise ValueError('Replay requires an isolated process without current numerical modules')
    sys.path.insert(0, str(root))
    train = importlib.import_module('softrole.train')
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('softrole', 'hetnet_ext', 'envs'):
            path = getattr(module, '__file__', None)
            if path:
                relative = str(Path(path).resolve().relative_to(root))
                if relative not in source['files']:
                    raise ValueError(f'Unverified numerical import: {name}')
    before = torch.load(before_path, map_location='cpu', weights_only=False)
    after = torch.load(after_path, map_location='cpu', weights_only=False)
    cfg = train.Config(**before['config'])
    if (cfg.to_dict() != train.Config(**read(run / 'config.json')).to_dict() or after['config'] != before['config']
            or any(c['source_sha256'] != source['sha256'] for c in (before, after))):
        raise ValueError('Anchor configuration/source identity mismatch')
    if not before['updates'] < boundary_update < after['updates']:
        raise ValueError('Boundary must lie strictly between anchors')
    pinned = {str(p): sha(p) for p in (before_path, after_path, run / 'run.json',
              run / 'config.json', run / 'source_manifest.json', run / 'updates.jsonl')}
    updates = {}
    for line in (run / 'updates.jsonl').open():
        row = json.loads(line)
        if before['updates'] < row['update'] <= after['updates']:
            if row['update'] in updates:
                raise ValueError('Duplicate historical update')
            updates[row['update']] = row
    episodes = {}
    index_path = run / 'episodes.index.json'
    if index_path.exists():
        pinned[str(index_path)] = sha(index_path)
        chunks = read(index_path)['chunks']
        paths = []
        for chunk in chunks:
            if chunk['last_update'] > before['updates'] and chunk['first_update'] <= after['updates']:
                path = safe(run, chunk['path'])
                if sha(path) != chunk['sha256']:
                    raise ValueError('Episode chunk hash mismatch')
                paths.append(path)
    else:
        paths = [run / 'episodes.jsonl']
    for path in paths:
        pinned[str(path)] = sha(path)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if before['updates'] < row['update'] <= after['updates']:
                    episodes.setdefault(row['update'], []).append(row)
    torch.set_num_threads(1)
    model = train.SoftRoleNet(**before['model_config']).double()
    model.load_state_dict(before['model_state'], strict=True)
    optimizer = train.make_optimizer(model.parameters(), cfg)
    optimizer.load_state_dict(before['optimizer_state'])
    steps, count = before['total_steps'], before['total_episodes']
    boundary = None
    executor = ProcessPoolExecutor(max_workers=cfg.nprocesses, mp_context=mp.get_context('spawn')) if cfg.nprocesses > 1 else None
    try:
        for u in range(before['updates'] + 1, after['updates'] + 1):
            if executor is None:
                batches = [train.collect_batch(model, cfg, 0, u - 1)]
            else:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                futures = [executor.submit(train.worker_collect, state, before['model_config'],
                            cfg.to_dict(), rank, u - 1) for rank in range(cfg.nprocesses)]
                batches = [f.result() for f in futures]
            actual = [dict(e, update=u, collector=b['collector']) for b in batches for e in b['episodes']]
            if actual != episodes.get(u):
                raise ValueError(f'Exact archived episode replay failed at update {u}')
            norm = train.apply_gradients(model, optimizer, batches, cfg.max_grad_norm)
            nsteps = sum(b['num_steps'] for b in batches)
            n = sum(b['num_episodes'] for b in batches)
            steps += nsteps
            count += n
            row = dict(update=u, episodes=n, steps=nsteps, total_steps=steps, gradient_norm_before_clip=norm)
            if getattr(cfg, 'objective', 'team') == 'mean_agent':
                row.update({name: sum(b[name] for b in batches) for name in
                            ('objective_policy_loss_sum', 'objective_value_loss_sum', 'objective_total_loss_sum')})
            if row != updates.get(u):
                raise ValueError(f'Exact archived update replay failed at update {u}')
            if u == boundary_update:
                boundary = copy.deepcopy(before)
                boundary.update(model_state=copy.deepcopy(model.state_dict()),
                    optimizer_state=copy.deepcopy(optimizer.state_dict()), updates=u,
                    total_steps=steps, total_episodes=count,
                    epoch=(u + cfg.updates_per_epoch - 1) // cfg.updates_per_epoch,
                    completed_epochs=u // cfg.updates_per_epoch,
                    updates_in_partial_epoch=u % cfg.updates_per_epoch,
                    signature=train.model_signature(model))
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if (steps != after['total_steps'] or count != after['total_episodes']
            or not exact(model.state_dict(), after['model_state'])
            or not exact(optimizer.state_dict(), after['optimizer_state'])):
        raise ValueError('Exact later-anchor model/Adam/count verification failed')
    model.load_state_dict(boundary['model_state'], strict=True)
    if train.model_signature(model) != boundary['signature']:
        raise ValueError('Boundary signature restoration failed')
    verify_source(run)
    if any(sha(path) != digest for path, digest in pinned.items()):
        raise ValueError('Historical input changed during replay')
    output.mkdir(parents=True)
    checkpoint = output / 'boundary.pt'
    torch.save(boundary, checkpoint)
    train.write_json(output / 'boundary.pt.signature.json', boundary['signature'])
    admission = dict(version=1, admitted=True, boundary_update=boundary_update,
        boundary_checkpoint_sha256=sha(checkpoint), signature=boundary['signature'],
        total_steps=boundary['total_steps'], total_episodes=boundary['total_episodes'],
        source_sha256=source['sha256'], runtime={'torch': torch.__version__, 'numpy': np.__version__},
        input_sha256=pinned, anchor_before=str(before_path), anchor_after=str(after_path),
        replayed_updates=after['updates'] - before['updates'],
        verification='Exact episode/update replay and later-anchor model, Adam, counts equality')
    train.write_json(output / 'admission.json', admission)
    return admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run-dir', 'output', 'anchor-before', 'anchor-after'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--boundary-update', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(replay(**vars(args)), sort_keys=True))


if __name__ == '__main__':
    main()
