"""Bounded archived-CLI checks, not a throughput or learning-quality experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from publication_reconstruction.artifacts import load_checkpoint
    from publication_reconstruction.runtime.hetnet_ext.signatures import tree_signature
    from publication_reconstruction.study import panel
    rows = []
    for task in ('pp', 'pcp', 'fc'):
        team = [(3, 0)] if task == 'pp' else [(2, 1)]
        scenarios = output / f'{task}_scenarios.json'
        scenarios.write_text(json.dumps(panel(task, team, 1, 4711), indent=2) + '\n')
        for variant in ('real', 'binary'):
            for collectors in (1, 4):
                pair = []
                for backend in ('dgl', 'torch-v1'):
                    name = f'{task}_{variant}_{collectors}_{backend}'
                    run = output / name
                    command = [sys.executable, '-m', 'publication_reconstruction', 'train',
                        '--reconstruction-spec', 'paper-v1', '--task', task, '--variant', variant,
                        '--message-backend', backend, '--seed', '991', '--epochs', '1',
                        '--updates-per-epoch', '3', '--collectors', str(collectors), '--batch-steps', '7',
                        '--horizon', '6', '--output', str(run)]
                    with (output / (name + '.log')).open('x') as log:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'{name}: see log; exit{result.returncode}')
                    status = json.loads((run / 'run_status.json').read_text())
                    saved = load_checkpoint(run, status['checkpoint'])
                    initial = json.loads((run / 'initial_training_identity.json').read_text())
                    assert saved['reconstruction']['message_backend'] == backend
                    assert saved['reconstruction']['learner_spec'] == 'paper-equations-v1'
                    assert status['counts']['updates'] == 3
                    assert initial['model'] != tree_signature(saved['policy_net'])
                    before = hashlib.sha256(Path(status['checkpoint']).read_bytes()).hexdigest()
                    evaluation = output / (name + '_evaluation.json')
                    eval_command = [sys.executable, '-m', 'publication_reconstruction', 'evaluate',
                        '--run-dir', str(run), '--checkpoint', status['checkpoint'], '--scenarios',
                        str(scenarios), '--output', str(evaluation), '--trace']
                    with (output / (name + '_evaluation.log')).open('x') as log:
                        result = subprocess.run(eval_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'{name}: see evaluation log; exit{result.returncode}')
                    report = json.loads(evaluation.read_text())
                    assert report['parameters_unchanged'] and report['episodes'] == 1
                    assert before == hashlib.sha256(Path(status['checkpoint']).read_bytes()).hexdigest()
                    row = dict(task=task, variant=variant, collectors=collectors, backend=backend,
                        run=str(run), command=command, evaluation_command=eval_command,
                        counts=status['counts'], initial_identity={k: initial[k] for k in ('model', 'optimizer', 'rng')},
                        final_model=tree_signature(saved['policy_net']), final_optimizer=tree_signature(saved['trainer']),
                        checkpoint_sha256=before, source=saved['reconstruction']['source_manifest_sha256'],
                        frozen_parameters_unchanged=report['parameters_unchanged'])
                    rows.append(row)
                    pair.append((row, saved))
                    (output / 'progress.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
                    print('passed', name, flush=True)
                assert pair[0][0]['initial_identity'] == pair[1][0]['initial_identity']
                for key, value in pair[0][1]['policy_net'].items():
                    other = pair[1][1]['policy_net'][key]
                    atol, rtol = (1e-6, 1e-5) if value.dtype == torch.float32 else (1e-10, 1e-8)
                    torch.testing.assert_close(value, other, atol=atol, rtol=rtol)
    result = dict(passed=True, runs=len(rows), updates=sum(r['counts']['updates'] for r in rows),
                  frozen_evaluations=len(rows), rows=rows,
                  limitation='horizon6/floor7 engineering checks; no speed or task-quality conclusion')
    (output / 'validation.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
