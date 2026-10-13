"""Historical Adam launcher validates both panels before any mock submission."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case(tmp_path):
    root = tmp_path / 'isolated checkout'
    (root / 'scripts').mkdir(parents=True)
    (root / 'slurm').mkdir()
    for name in ('softrole_adam_diagnostics.sh', 'softrole_diagnostics.sh'):
        shutil.copyfile(ROOT / 'scripts' / name, root / 'scripts' / name)
    shutil.copyfile(ROOT / 'slurm/softrole_diagnostics.sbatch', root / 'slurm/softrole_diagnostics.sbatch')
    shutil.copyfile(ROOT / 'slurm/softrole_adam_diagnostics.sbatch',
                    root / 'slurm/softrole_adam_diagnostics.sbatch')
    runs = root / 'runs/softrole_adam_pcp_911488'
    for seed in range(3):
        directory = runs / f'pcp_shared/seed{seed}/checkpoints'
        directory.mkdir(parents=True)
        for epoch in (1000, 2000):
            (directory / f'epoch{epoch}.pt').write_text('checkpoint fixture')
    capture = tmp_path / 'calls.jsonl'
    interpreter = tmp_path / 'fake interpreter'
    interpreter.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['CAPTURE'], 'a') as stream:
    stream.write(json.dumps({'kind': 'python', 'argv': args, 'cwd': os.getcwd()}) + '\\n')
if 'prepare' in args:
    destination = pathlib.Path(args[args.index('--output') + 1])
    if os.environ.get('FAIL_PREPARE') and destination.stem == os.environ['FAIL_PREPARE']:
        sys.exit(7)
    destination.write_text('{"entries": [{}, {}, {}]}\\n')
else:
    assert '--dry-run' in args
    manifest = pathlib.Path(args[args.index('--manifest') + 1])
    assert manifest.is_file()
    stage = manifest.stem
    index = args[args.index('--index') + 1]
    if os.environ.get('FAIL_DRY_RUN') == stage + ':' + index:
        sys.exit(9)
''')
    interpreter.chmod(0o755)
    sbatch = tmp_path / 'sbatch'
    sbatch.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, sys
with open(os.environ['CAPTURE'], 'a') as stream:
    stream.write(json.dumps({'kind': 'sbatch', 'argv': sys.argv[1:], 'cwd': os.getcwd(),
        'manifest': os.environ['CAPCOM_DIAGNOSTICS_MANIFEST'],
        'output_root': os.environ['CAPCOM_DIAGNOSTICS_OUTPUT_ROOT'],
        'python': os.environ['HETNET_PYTHON'],
        'logs_exists': pathlib.Path('logs_sr').is_dir()}) + '\\n')
print('Submitted batch job 12345')
''')
    sbatch.chmod(0o755)
    env = dict(os.environ, PYTHON=str(interpreter), CAPTURE=str(capture),
               PATH=f'{tmp_path}:{os.environ["PATH"]}')
    # Verify the helper's PYTHON override propagates to the existing launcher.
    env['HETNET_PYTHON'] = str(tmp_path / 'wrong interpreter')
    return root, runs, capture, env


def launch(case, *arguments):
    root, _, _, env = case
    return subprocess.run(['bash', str(root / 'scripts/softrole_adam_diagnostics.sh'), *arguments],
                          cwd=root.parent, env=env, text=True, capture_output=True)


def calls(case):
    capture = case[2]
    return [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []


def argument(row, name):
    return row['argv'][row['argv'].index(name) + 1]


def test_prepares_exact_two_panels_then_six_dry_runs_without_submitting(case):
    root, runs, _, _ = case
    result = launch(case, '--run-root', str(runs), '--output-root', 'runs/my diagnostics')
    assert result.returncode == 0, result.stderr
    rows = calls(case)
    assert len(rows) == 8
    assert all(row['kind'] == 'python' and row['cwd'] == str(root) for row in rows)
    for row, stage, epoch, mode in zip(rows[:2], ('midway', 'final'), (1000, 2000),
                                     ('logged-next-update', 'checkpoint-probe')):
        assert row['argv'][:4] == ['-u', '-m', 'softrole.diagnostics', 'prepare']
        assert argument(row, '--runtime') == 'archived-adam'
        assert argument(row, '--batch-mode') == mode
        start = row['argv'].index('--checkpoints') + 1
        assert row['argv'][start:start+3] == [str(runs / f'pcp_shared/seed{seed}/checkpoints/epoch{epoch}.pt')
                                             for seed in range(3)]
        assert argument(row, '--output') == f'runs/my diagnostics/manifests/{stage}.json'
        assert str(epoch) in argument(row, '--checkpoint-rule')
        assert '--seed' not in row['argv'] and '--probe-episodes' not in row['argv']
    assert 'first archived checkpoint reaching 20000000' in argument(rows[0], '--checkpoint-rule')
    assert 'final archived checkpoint' in argument(rows[1], '--checkpoint-rule')
    for row, (stage, index) in zip(rows[2:], ((s, i) for s in ('midway', 'final') for i in range(3))):
        assert row['argv'] == ['-u', '-m', 'softrole.diagnostics', '--manifest',
            f'runs/my diagnostics/manifests/{stage}.json', '--index', str(index), '--output',
            f'runs/my diagnostics/{stage}/task_{index}', '--dry-run']
    assert not (root / 'logs_sr').exists()
    assert not (root / 'runs/my diagnostics/midway').exists()
    assert 'Results root: runs/my diagnostics' in result.stdout
    assert 'no jobs submitted' in result.stdout


def test_submit_occurs_after_all_validations_and_exports_each_stage(case):
    root, _, _, env = case
    result = launch(case, '--output-root', 'runs/submitted', '--submit')
    assert result.returncode == 0, result.stderr
    rows = calls(case)
    assert [row['kind'] for row in rows] == ['python'] * 8 + ['sbatch'] * 2
    for row, stage in zip(rows[-2:], ('midway', 'final')):
        assert row['argv'] == ['--export=ALL', '--array=0-2%3', 'slurm/softrole_diagnostics.sbatch']
        assert row['manifest'] == f'runs/submitted/manifests/{stage}.json'
        assert row['output_root'] == f'runs/submitted/{stage}'
        assert row['python'] == env['PYTHON']
        assert row['cwd'] == str(root) and row['logs_exists']


def test_default_output_is_fresh_timestamp_and_pid(case):
    result = launch(case)
    assert result.returncode == 0, result.stderr
    root = case[0]
    outputs = list((root / 'runs').glob('capcom_adam_diagnostics_*'))
    assert len(outputs) == 1
    assert len(outputs[0].name.removeprefix('capcom_adam_diagnostics_').split('_')) == 3
    assert (outputs[0] / 'manifests/midway.json').is_file()
    assert (outputs[0] / 'manifests/final.json').is_file()
    assert not any(row['kind'] == 'sbatch' for row in calls(case))


@pytest.mark.parametrize('epoch,seed', [(1000, 0), (1000, 2), (2000, 2)])
def test_missing_checkpoint_refuses_before_preparing_or_submitting(case, epoch, seed):
    root, runs, _, _ = case
    (runs / f'pcp_shared/seed{seed}/checkpoints/epoch{epoch}.pt').unlink()
    result = launch(case, '--output-root', 'runs/missing', '--submit')
    assert result.returncode == 2 and 'Checkpoint not found' in result.stderr
    assert calls(case) == []
    assert not (root / 'runs/missing').exists()


@pytest.mark.parametrize('kind', ['directory', 'file', 'broken_symlink'])
def test_existing_output_refused_before_running_python(case, kind):
    output = case[0] / 'existing'
    if kind == 'directory':
        output.mkdir()
    elif kind == 'file':
        output.write_text('preserved')
    else:
        output.symlink_to('absent')
    result = launch(case, '--output-root', str(output), '--submit')
    assert result.returncode == 2 and 'Output already exists' in result.stderr
    assert calls(case) == []
    if kind == 'file':
        assert output.read_text() == 'preserved'
    elif kind == 'broken_symlink':
        assert output.is_symlink()


@pytest.mark.parametrize('variable,value,count', [('FAIL_PREPARE', 'final', 2),
                                                 ('FAIL_DRY_RUN', 'midway:0', 3),
                                                 ('FAIL_DRY_RUN', 'final:2', 8)])
def test_any_validation_failure_prevents_all_submissions(case, variable, value, count):
    case[3][variable] = value
    result = launch(case, '--output-root', 'runs/fails', '--submit')
    assert result.returncode != 0
    rows = calls(case)
    assert len(rows) == count and all(row['kind'] == 'python' for row in rows)
    assert not (case[0] / 'logs_sr').exists()


@pytest.mark.parametrize('arguments', [['--run-root'], ['--output-root'], ['--unknown'],
                                     ['--submit', '--submit'],
                                     ['--run-root', 'first', '--run-root', 'second'],
                                     ['--output-root', 'first', '--output-root', 'second']])
def test_malformed_options_rejected_before_work(case, arguments):
    result = launch(case, *arguments)
    assert result.returncode == 2 and calls(case) == []


def test_shell_syntax():
    result = subprocess.run(['bash', '-n', str(ROOT / 'scripts/softrole_adam_diagnostics.sh')],
                            capture_output=True)
    assert result.returncode == 0, result.stderr


def slurm_launch(case, *arguments, python_source='HETNET_PYTHON'):
    root, _, capture, original_env = case
    env = dict(original_env, SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID='916000')
    interpreter = original_env['PYTHON']
    env.pop('HETNET_PYTHON', None)
    env.pop('PYTHON', None)
    if python_source:
        env[python_source] = interpreter
    else:
        (root / '.venv/bin').mkdir(parents=True)
        (root / '.venv/bin/python').symlink_to(interpreter)
    for name in ('module', 'srun'):
        command = capture.parent / name
        command.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['CAPTURE'], 'a') as stream:
    stream.write(json.dumps({'kind': name, 'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')
if name == 'srun':
    assert sys.argv[1] == '--cpu-bind=cores'
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
''')
        command.chmod(0o755)
    return subprocess.run(['bash', str(root / 'slurm/softrole_adam_diagnostics.sbatch'), *arguments],
                          cwd=root.parent, env=env, text=True, capture_output=True)


@pytest.mark.parametrize('python_source', ['HETNET_PYTHON', 'PYTHON', None])
def test_slurm_only_entry_prepares_under_srun_then_submits_both_arrays(case, python_source):
    root = case[0]
    result = slurm_launch(case, python_source=python_source)
    assert result.returncode == 0, result.stderr
    rows = calls(case)
    assert [row['kind'] for row in rows] == ['module', 'srun'] + ['python'] * 8 + ['sbatch'] * 2
    assert rows[0]['argv'] == ['load', 'anaconda/anaconda-2024.10']
    assert all(row['cwd'] == str(root) for row in rows)
    assert 'Results root: runs/capcom_adam_diagnostics_916000' in result.stdout
    for row, stage in zip(rows[-2:], ('midway', 'final')):
        assert row['manifest'] == f'runs/capcom_adam_diagnostics_916000/manifests/{stage}.json'
        assert row['output_root'] == f'runs/capcom_adam_diagnostics_916000/{stage}'
        assert row['argv'] == ['--export=ALL', '--array=0-2%3', 'slurm/softrole_diagnostics.sbatch']
        assert row['logs_exists']


def test_slurm_only_entry_honors_explicit_roots_and_stops_on_validation_failure(case):
    case[3]['CAPCOM_ADAM_RUN_ROOT'] = str(case[1])
    case[3]['CAPCOM_ADAM_OUTPUT_ROOT'] = 'runs/custom slurm diagnostics'
    case[3]['FAIL_DRY_RUN'] = 'final:2'
    result = slurm_launch(case)
    assert result.returncode == 9, result.stderr
    rows = calls(case)
    assert [row['kind'] for row in rows] == ['module', 'srun'] + ['python'] * 8
    assert argument(rows[2], '--output') == 'runs/custom slurm diagnostics/manifests/midway.json'
    start = rows[2]['argv'].index('--checkpoints') + 1
    assert rows[2]['argv'][start] == str(case[1] / 'pcp_shared/seed0/checkpoints/epoch1000.pt')


def test_slurm_only_entry_rejects_arguments_before_work(case):
    result = slurm_launch(case, '--submit')
    assert result.returncode != 0
    assert calls(case) == []


def test_slurm_only_entry_placement_logs_and_shell_syntax():
    wrapper = ROOT / 'slurm/softrole_adam_diagnostics.sbatch'
    worker = ROOT / 'slurm/softrole_diagnostics.sbatch'
    for script in (wrapper, worker):
        content = script.read_text()
        for directive in ('--constraint=sapphirerapids', '--account=cenyioha',
                          '--partition=normal', '--cpus-per-task=1'):
            assert '#SBATCH ' + directive in content
    content = wrapper.read_text()
    assert '#SBATCH --array=' not in content
    assert '#SBATCH --output=capcom-adam-prepare-%j.out' in content
    assert '#SBATCH --error=capcom-adam-prepare-%j.err' in content
    result = subprocess.run(['bash', '-n', str(wrapper)], capture_output=True)
    assert result.returncode == 0, result.stderr
