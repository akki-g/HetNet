"""Read-only provenance and checkpoint integrity checks for copied run artifacts."""
from pathlib import Path
import collections
import hashlib
import json
import math
import re
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / 'stokes_runs/runs'
OUT = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def signature_ok(sig):
    payload = {k: v for k, v in sig.items() if k != 'sha256'}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert expected == sig['sha256']

def checkpoint_epoch(path):
    return int(re.search(r'(?:epoch|ep)(\d+)', path.name).group(1))

report = {'root': str(INPUT.relative_to(ROOT)), 'scope': 'Snapshot integrity, not training quality or live scheduler state', 'runs': []}
for d in sorted(INPUT.glob('*/*/seed*')):
    category = d.relative_to(INPUT).parts[0]
    is_sr = category == 'softrole_primary'
    metric = lines(d / 'metrics.jsonl')
    init = read(d / 'initial_signature.json')
    signature_ok(init)
    init_map = {v['name']: v for v in init['parameters']}
    checkpoints = sorted(d.glob('checkpoints/**/*.pt'), key=checkpoint_epoch)
    r = {'run': d.relative_to(INPUT).as_posix(), 'category': category,
         'epochs': len(metric), 'total_steps_at_last_epoch': metric[-1]['total_steps'],
         'last_epoch': metric[-1]['epoch'], 'checkpoint_count': len(checkpoints),
         'latest_checkpoint_epoch': checkpoint_epoch(checkpoints[-1]) if checkpoints else None,
         'initial_parameter_sha256': init['sha256'],
         'parameter_count': sum(math.prod(v['shape']) for v in init['parameters']),
         'parameter_tensor_dtypes': dict(collections.Counter(v['dtype'] for v in init['parameters'])),
         'exit_code_file': (d / 'exit_code.txt').read_text().strip() if (d / 'exit_code.txt').exists() else None,
         'checkpoint_checks': []}
    assert len(metric) == metric[-1]['epoch']
    if is_sr:
        config = read(d / 'config.json')
        run = read(d / 'run.json')
        manifest = read(d / 'source_manifest.json')
        manifest_id = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
        assert manifest_id == manifest['sha256'] == run['source_sha256']
        r.update(config=config, run_metadata=run, source_sha256=manifest_id,
                 source_git_commit=manifest['git_commit'], source_git_status=manifest['git_status'],
                 archived_source_files=len(manifest['files']), source_mismatches=[], local_source_differences=[])
        for name, expected in manifest['files'].items():
            archived = d / 'source' / name
            assert archived.is_file() and sha(archived) == expected
            local = ROOT / name
            if not local.exists() or sha(local) != expected:
                r['local_source_differences'].append(name)
        u = lines(d / 'updates.jsonl')
        r['last_update'] = u[-1]['update']
        r['updates_minus_last_epoch_updates'] = u[-1]['update'] - metric[-1]['updates']
        assert r['parameter_count'] == run['parameters']
    else:
        config = read(d / 'resolved_args.json')
        env = (d / 'environment.txt').read_text().splitlines()
        r.update(config=config, environment=env, source_git_commit=env[2],
                 source_patch_bytes=(d / 'source.patch').stat().st_size)
        signatures = lines(d / 'epoch_signatures.jsonl')
        signatures_by_epoch = {v['epoch']: v['signature'] for v in signatures}
        r['last_signature_epoch'] = signatures[-1]['epoch']
        for s in signatures:
            signature_ok(s['signature'])
        records = lines(d / 'checkpoint_records.jsonl') if (d / 'checkpoint_records.jsonl').exists() else []
        by_epoch = {v['epoch']: v for v in records}
        assert len(checkpoints) == len(records)
    for p in checkpoints:
        epoch = checkpoint_epoch(p)
        sidecar = read(Path(str(p) + '.signature.json'))
        signature_ok(sidecar)
        ck = {'path': p.relative_to(INPUT).as_posix(), 'epoch': epoch, 'bytes': p.stat().st_size,
              'sha256': sha(p), 'parameter_sha256': sidecar['sha256']}
        if is_sr:
            payload = torch.load(p, map_location='cpu', weights_only=True)
            assert json.loads(json.dumps(payload['config'])) == config
            assert payload['source_sha256'] == manifest_id
            assert payload['environment_version'] == run['environment_version']
            assert payload['signature'] == sidecar
            assert payload['epoch'] == payload['completed_epochs'] == epoch
            assert payload['updates_in_partial_epoch'] == 0
            assert payload['updates'] == epoch * config['updates_per_epoch']
            m = metric[epoch - 1]
            for key in ['total_steps', 'total_episodes', 'updates']:
                assert payload[key] == m[key]
            sidecar_map = {v['name']: v for v in sidecar['parameters'] + sidecar['buffers']}
            state = payload['model_state']
            assert set(state) == set(sidecar_map)
            for name, tensor in state.items():
                entry = sidecar_map[name]
                assert list(tensor.shape) == entry['shape'] and str(tensor.dtype) == entry['dtype']
                raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                assert hashlib.sha256(raw).hexdigest() == entry['sha256']
                assert torch.isfinite(tensor).all()
            optimizer = payload['optimizer_state']
            for state_entry in optimizer['state'].values():
                assert int(state_entry['step']) == payload['updates']
                for value in state_entry.values():
                    if torch.is_tensor(value):
                        assert torch.isfinite(value).all()
            ck['verification'] = 'safe weights-only load; config/source/counts/model tensor hashes/finite optimizer state verified'
        else:
            record = by_epoch[epoch]
            assert record['bytes'] == p.stat().st_size
            assert record['parameter_sha256'] == sidecar['sha256'] == signatures_by_epoch[epoch]['sha256']
            ck['verification'] = 'file bytes match checkpoint record; sidecar equals epoch signature; payload not deserialized'
        r['checkpoint_checks'].append(ck)
    if checkpoints:
        last = read(Path(str(checkpoints[-1]) + '.signature.json'))
        last_map = {v['name']: v for v in last['parameters']}
        r['unchanged_parameter_tensors_initial_to_latest_checkpoint'] = [k for k, v in init_map.items() if v == last_map[k]]
        assert last['sha256'] != init['sha256']
    report['runs'].append(r)

sr = [r for r in report['runs'] if r['category'] == 'softrole_primary']
report['softrole_config_differences_by_task'] = {}
for task in ['pp', 'pcp', 'fc']:
    configs = [r['config'] for r in sr if r['config']['task'] == task]
    report['softrole_config_differences_by_task'][task] = {k: sorted(set(str(c[k]) for c in configs)) for k in configs[0] if len(set(str(c[k]) for c in configs)) > 1}
report['counts'] = dict(collections.Counter(r['category'] for r in report['runs']))
report['checkpoint_counts'] = {category: sum(r['checkpoint_count'] for r in report['runs'] if r['category'] == category) for category in report['counts']}
report['all_assertions_passed'] = True
report['analysis_script_sha256'] = sha(Path(__file__))
(OUT / 'audit.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: report[k] for k in ['counts', 'checkpoint_counts', 'softrole_config_differences_by_task', 'all_assertions_passed']}, indent=2))
