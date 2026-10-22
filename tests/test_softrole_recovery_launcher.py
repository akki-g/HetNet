"""Recovery launch sequencing and scheduler argument checks, without cluster access."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/softrole_pcp_recovery.sh"
WRAPPER = ROOT / "slurm/softrole_pcp_recovery.sbatch"


def mock_executable(path, source):
    path.write_text("#!/usr/bin/env python3\n" + source)
    path.chmod(0o755)
    return path


@pytest.fixture
def shell_env(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "calls.jsonl"
    recorder = "import json, os, pathlib, sys\nwith open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps({'tool': pathlib.Path(sys.argv[0]).name, 'args': sys.argv[1:], 'bash_env': os.environ.get('BASH_ENV'), 'conda': os.environ.get('CONDA_SHLVL')})+'\\n')\n"
    python = mock_executable(binaries / "pinned-python", recorder + "if 'prepare' in sys.argv:\n p=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); p.mkdir(); (p/'manifest.json').write_text('{}')\n")
    mock_executable(binaries / "sbatch", recorder + "print('101;cluster' if 'reconstruct' in sys.argv else '102;cluster')\n")
    mock_executable(binaries / "module", recorder)
    mock_executable(binaries / "srun", recorder)
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'],
               HETNET_PYTHON=str(python), CALL_LOG=str(log),
               SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_JOB_ID='101',
               SLURM_ARRAY_TASK_ID='0', CONDA_SHLVL='99')
    env.pop('BASH_ENV', None)
    env.pop('ENV', None)
    return env, log


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_two_stage_submission_dependency_and_quoted_paths(tmp_path, shell_env):
    env, log = shell_env
    output = tmp_path / 'new output'
    result = subprocess.run(['bash', str(SCRIPT), 'archive with spaces', str(output)],
                            env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records = calls(log)
    assert [r['tool'] for r in records] == ['pinned-python', 'sbatch', 'sbatch']
    assert records[0]['args'] == ['-u', '-m', 'softrole.pcp_recovery', 'prepare',
        '--archive-root', str(tmp_path / 'archive with spaces'), '--output', str(output),
        '--additional-steps', '5000000']
    assert records[1]['args'] == ['--parsable', '--array=0-1%2', str(WRAPPER),
                                  'reconstruct', str(output / 'manifest.json')]
    assert records[2]['args'] == ['--parsable', '--dependency=afterok:101', '--array=0-5%6',
                                  str(WRAPPER), 'run', str(output / 'manifest.json')]
    assert all(r['bash_env'] is None for r in records[1:])


def test_prepare_only_never_submits(tmp_path, shell_env):
    env, log = shell_env
    result = subprocess.run(['bash', str(SCRIPT), str(tmp_path), str(tmp_path / 'out'),
                             '123', '--prepare-only'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert [r['tool'] for r in calls(log)] == ['pinned-python']
    assert calls(log)[0]['args'][-1] == '123'


@pytest.mark.parametrize('phase,index,extra', [('reconstruct', '0', []), ('reconstruct', '1', []),
    ('run', '0', []), ('run', '1', []), ('run', '2', []), ('run', '3', []),
    ('run', '4', []), ('run', '5', []), ('run', '5', ['--resume'])])
def test_wrapper_routes_phase_index_and_resume(tmp_path, shell_env, phase, index, extra):
    env, log = shell_env
    env['SLURM_ARRAY_TASK_ID'] = index
    manifest = tmp_path / 'manifest with spaces.json'
    manifest.write_text('{}')
    result = subprocess.run(['bash', str(WRAPPER), phase, str(manifest), *extra],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records = calls(log)
    assert [r['tool'] for r in records] == ['module', 'srun']
    assert records[1]['args'] == ['--cpu-bind=cores', env['HETNET_PYTHON'], '-u', '-m',
        'softrole.pcp_recovery', phase, '--manifest', str(manifest), '--index', index, *extra]
    assert records[1]['conda'] == '0'
    assert f'phase={phase} index={index}' in result.stderr


@pytest.mark.parametrize('phase,index,extra', [('reconstruct', '2', []), ('run', '6', []),
    ('run', '-1', []), ('reconstruct', '0', ['--resume']), ('run', '0', ['--lr']),
    ('unknown', '0', [])])
def test_wrapper_invalid_protocol_never_runs(tmp_path, shell_env, phase, index, extra):
    env, log = shell_env
    env['SLURM_ARRAY_TASK_ID'] = index
    result = subprocess.run(['bash', str(WRAPPER), phase, str(tmp_path / 'manifest'), *extra],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not log.exists()


def test_reconstruction_submission_failure_never_submits_recovery(tmp_path, shell_env):
    env, log = shell_env
    mock_executable(Path(env['PATH'].split(os.pathsep)[0]) / 'sbatch',
                    "import sys\nprint('scheduler unavailable', file=sys.stderr)\nsys.exit(1)\n")
    result = subprocess.run(['bash', str(SCRIPT), str(tmp_path), str(tmp_path / 'out')],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Recovery job:' not in result.stdout
    assert [r['tool'] for r in calls(log)] == ['pinned-python']


def test_scheduler_resources_and_shell_syntax():
    source = WRAPPER.read_text()
    for setting in ['--account=cenyioha', '--partition=normal', '--constraint=sapphirerapids',
                    '--cpus-per-task=4', '--mem=16G', '--time=48:00:00', '--array=0-5%6']:
        assert '#SBATCH ' + setting in source
    subprocess.run(['bash', '-n', str(SCRIPT), str(WRAPPER)], check=True)
