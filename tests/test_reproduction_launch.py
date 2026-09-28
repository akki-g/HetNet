"""Exercise the real shell entry points without starting expensive training."""
from collections import Counter
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_slurm_array_covers_each_author_readme_recipe_and_seed_once(tmp_path):
    tools = tmp_path / 'bin'
    tools.mkdir()
    srun = tools / 'srun'
    srun.write_text('#!/usr/bin/env bash\nexec "$@"\n')
    srun.chmod(0o755)
    module = tools / 'module'
    module.write_text('#!/usr/bin/env bash\nexit 0\n')
    module.chmod(0o755)
    env = {**os.environ, 'PATH': str(tools) + os.pathsep + os.environ['PATH'],
           'SLURM_SUBMIT_DIR': str(ROOT), 'HETNET_RUN_ROOT': str(tmp_path / 'runs'),
           'HETNET_PYTHON': str(tmp_path / 'custom environment/bin/python')}
    seen = Counter()
    for index in range(9):
        env['SLURM_ARRAY_TASK_ID'] = str(index)
        result = subprocess.run(['bash', 'slurm/reproduce.sbatch', '--dry-run'],
                                cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        command = shlex.split(result.stdout)
        assert command[0] == env['HETNET_PYTHON']
        value = lambda flag: command[command.index(flag) + 1]
        task = 'fc' if value('--env_name') == 'fire_commander' else ('pp' if value('--nfriendly_A') == '0' else 'pcp')
        variant = 'binary' if '--use_binary' in command else 'real'
        seen[task, variant, int(value('--seed'))] += 1
        assert value('--nprocesses') == '4'
        assert value('--num_epochs') == ('1400' if task == 'fc' else '2000')
        assert value('--max_steps') == ('300' if task == 'fc' else '80')
        assert value('--nfriendly_P') == ('3' if task == 'pp' else '2')
    assert seen == Counter({(task, 'real', seed): 1 for task in ('pp', 'pcp', 'fc') for seed in range(3)})
    assert not (tmp_path / 'runs').exists()


def test_failed_training_is_preserved_and_cannot_be_overwritten(tmp_path):
    python = tmp_path / 'fake-python'
    python.write_text('#!/usr/bin/env bash\nif [[ $1 == --version || $1 == -c ]]; then echo test-python; exit 0; fi\necho learner-failed\nexit 7\n')
    python.chmod(0o755)
    env = {**os.environ, 'HETNET_PYTHON': str(python), 'HETNET_RUN_ROOT': str(tmp_path / 'runs')}
    cmd = ['bash', 'scripts/reproduce.sh', 'pcp', 'real', '0']
    first = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    run = tmp_path / 'runs/pcp_real/seed0'
    assert first.returncode == 7
    assert (run / 'exit_code.txt').read_text().strip() == '7'
    assert 'learner-failed' in (run / 'stdout.log').read_text()
    assert (run / 'uv.lock').read_bytes() == (ROOT / 'uv.lock').read_bytes()
    assert (run / 'requirements.txt').read_bytes() == (ROOT / 'requirements.txt').read_bytes()
    saved_command = (run / 'command.txt').read_text()
    second = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    assert second.returncode != 0
    assert (run / 'command.txt').read_text() == saved_command
    assert (run / 'exit_code.txt').read_text().strip() == '7'


def test_missing_environment_explains_setup_without_creating_run(tmp_path):
    env = {**os.environ, 'HETNET_PYTHON': str(tmp_path / 'missing/bin/python'),
           'HETNET_RUN_ROOT': str(tmp_path / 'runs')}
    result = subprocess.run(['bash', 'scripts/reproduce.sh', 'pcp', 'real', '0'],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert env['HETNET_PYTHON'] in result.stderr
    assert 'sbatch slurm/setup.sbatch' in result.stderr
    assert 'bash scripts/setup_env.sh' in result.stderr
    assert not (tmp_path / 'runs').exists()
