"""Submission-time log creation, run identity, containment and failure evidence."""
import json
import os
from pathlib import Path
import sys

import pytest

from scripts import submit_stokes as submit


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    (root / 'slurm').mkdir(parents=True)
    actual = Path(submit.__file__).resolve().parents[1]
    for p in (actual / 'slurm').glob('*.sbatch'):
        (root / 'slurm' / p.name).write_bytes(p.read_bytes())
    monkeypatch.setattr(submit, 'ROOT', root)
    monkeypatch.chdir(root)
    for name in set(k for r in submit.ENV_ROUTES.values() for k in r):
        monkeypatch.delenv(name, raising=False)
    binary = tmp_path / 'bin'
    binary.mkdir()
    sbatch = binary / 'sbatch'
    # Simulate Slurm opening logs immediately on submission. A directory created
    # only by the eventual job would fail this check.
    sbatch.write_text(f'#!{sys.executable}\n' + '''import json, os, pathlib, sys
args=sys.argv[1:]
root=pathlib.Path(os.environ['HETNET_RUN_DIR'])
assert (root/'run_info.json').is_file()
for flag in ('--output','--error'):
    target=pathlib.Path(args[args.index(flag)+1].replace('%A','321').replace('%a','0'))
    assert target.parent.is_dir()
    target.write_text('scheduler stream\\n')
record={'args':args,'env':{k:v for k,v in os.environ.items() if k.endswith('RUN_ROOT') or k in ('HETNET_RUN_DIR','HETNET_RUN_TITLE','SOFTROLE_GRADED_PILOT_MANIFEST')}}
(root/'captured.json').write_text(json.dumps(record))
print(os.environ.get('MOCK_SBATCH_RESPONSE','321;stokes'))
raise SystemExit(int(os.environ.get('MOCK_SBATCH_EXIT','0')))
''')
    sbatch.chmod(0o755)
    monkeypatch.setenv('PATH', str(binary) + os.pathsep + os.environ['PATH'])
    return root


def new_args(job='softrole.sbatch'):
    return ['--title', 'PCP capability versus type', '--purpose', 'Compare two descriptors at the same sensing levels.',
            '--', '--array=0-1%2', 'slurm/' + job]


def only_run(root):
    return next((root / 'runs').iterdir())


def test_submission_creates_logs_before_scheduler_and_keeps_leaf_fresh(checkout, capsys):
    assert submit.main(new_args()) == 0
    assert capsys.readouterr().out.strip() == '321;stokes'
    run = only_run(checkout)
    assert 'pcp-capability-versus-type' in run.name
    assert not (run / 'training').exists()
    captured = json.loads((run / 'captured.json').read_text())
    assert captured['env']['SOFTROLE_RUN_ROOT'] == str(run / 'training')
    assert len(list((run / 'logs').glob('*'))) == 2
    assert 'Compare two descriptors' in (run / 'README.md').read_text()
    response = json.loads(next((run / 'submissions').glob('*.response.json')).read_text())
    assert response['submission_accepted'] is True and response['job_id'] == '321'


def test_prepare_and_training_reuse_one_container(checkout, capsys):
    args = new_args()[:4] + ['--init-only']
    assert submit.main(args) == 0
    run = Path(capsys.readouterr().out.strip())
    for job in ['softrole_graded_pilot_prepare.sbatch', 'softrole_graded_pilot.sbatch']:
        assert submit.main(['--run-dir', str(run), '--', '--dependency=afterok:123', 'slurm/' + job]) == 0
        captured = json.loads((run / 'captured.json').read_text())
        assert captured['env']['SOFTROLE_GRADED_PILOT_MANIFEST'] == str(run / 'manifest.json')
        assert captured['env']['SOFTROLE_GRADED_PILOT_RUN_ROOT'] == str(run / 'training')
        assert '--dependency=afterok:123' in captured['args']
    assert len(list((checkout / 'runs').iterdir())) == 1
    assert len(list((run / 'submissions').glob('*.response.json'))) == 2


def test_dry_run_does_not_create_or_submit(checkout, capsys):
    assert submit.main(['--dry-run', *new_args()]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert 'PCP capability' in plan['title']
    assert not (checkout / 'runs').exists()


@pytest.mark.parametrize('option', ['--output=/tmp/leak.out', '-oleak.out', '--error=x', '-efoo', '--chdir=/tmp', '-Jmisleading', '--export=NONE', '--wrap=echo'])
def test_scheduler_overrides_cannot_split_run(checkout, option):
    args = new_args()
    args.insert(args.index('--') + 1, option)
    assert submit.main(args) == 2
    assert not (checkout / 'runs').exists()


def test_external_checkpoint_is_input_but_output_stays_scoped(checkout):
    args = new_args('softrole_evaluate.sbatch') + ['/external/archive/checkpoint.pt']
    assert submit.main(args) == 0
    run = only_run(checkout)
    command = json.loads((run / 'captured.json').read_text())['args']
    assert command[-2:] == ['/external/archive/checkpoint.pt', str(run / 'evaluation.json')]


def test_cli_output_escape_rejected_before_creating_run(checkout):
    assert submit.main(new_args('publication_evaluate.sbatch') + ['--output', '/tmp/unrelated.json']) == 2
    assert not (checkout / 'runs').exists()


@pytest.mark.parametrize('job,options', [
    ('publication_paper_train.sbatch', ['--run-root', '/tmp/outside']),
    ('publication_paper_preflight.sbatch', ['--run-root=/tmp/outside']),
    ('softrole_graded_pilot.sbatch', ['--run-ro=/tmp/outside']),
    ('softrole_paper_fc.sbatch', ['--out=/tmp/outside']),
    ('publication_evaluate.sbatch', ['--out', '/tmp/outside']),
])
def test_output_root_and_abbreviations_cannot_escape(checkout, job, options):
    assert submit.main(new_args(job) + options) == 2
    assert not (checkout / 'runs').exists()


def test_preparation_can_read_an_external_run_root(checkout):
    assert submit.main(new_args('softrole_prepare_frozen.sbatch') + ['--run-root', '/external/training']) == 0
    run = only_run(checkout)
    command = json.loads((run / 'captured.json').read_text())['args']
    assert command[-4:] == ['--run-root', '/external/training', '--output', str(run / 'evaluation')]


@pytest.mark.parametrize('name', ['logs', 'submissions', 'run_info.json', 'README.md'])
def test_adopted_container_infrastructure_cannot_redirect_outside(checkout, tmp_path, name):
    run = checkout / 'runs' / 'prepared'
    run.mkdir(parents=True)
    outside = tmp_path / 'outside'
    (run / name).symlink_to(outside, target_is_directory=name in {'logs', 'submissions'})
    assert submit.main(['--run-dir', str(run), *new_args()]) == 2
    assert not outside.exists()
    assert sorted(p.name for p in run.iterdir()) == [name]


def test_environment_output_escape_rejected(checkout, monkeypatch):
    monkeypatch.setenv('SOFTROLE_RUN_ROOT', str(checkout / 'runs' / 'different-experiment'))
    assert submit.main(new_args()) == 2
    assert not (checkout / 'runs').exists()


def test_symlink_escape_rejected(checkout, tmp_path):
    (checkout / 'runs').mkdir()
    (checkout / 'runs' / 'alias').symlink_to(tmp_path / 'outside', target_is_directory=True)
    assert submit.main(['--run-dir', 'runs/alias', *new_args()]) == 2
    assert not (tmp_path / 'outside').exists()


def test_failed_submission_retains_diagnostic_receipt(checkout, monkeypatch):
    monkeypatch.setenv('MOCK_SBATCH_EXIT', '17')
    assert submit.main(new_args()) == 17
    run = only_run(checkout)
    response = json.loads(next((run / 'submissions').glob('*.response.json')).read_text())
    assert response['submission_accepted'] is False and response['returncode'] == 17


def test_existing_identity_is_not_relabelled(checkout):
    assert submit.main(new_args()) == 0
    run = only_run(checkout)
    assert submit.main(['--run-dir', str(run), '--title', 'Different test', '--', 'slurm/softrole.sbatch']) == 2
    assert len(list((run / 'submissions').glob('*.response.json'))) == 1


def test_recovery_reuses_enclosing_container_without_moving_manifest(checkout):
    args = new_args()[:4] + ['--run-dir', 'runs/recovery', '--init-only']
    assert submit.main(args) == 0
    run = checkout / 'runs/recovery'
    manifest = run / 'preparation/manifest.json'
    manifest.parent.mkdir()
    manifest.write_text('{"root":"original bytes"}')
    before = manifest.read_bytes()
    assert submit.main(['--run-dir', str(run), '--', 'slurm/softrole_pcp_recovery.sbatch', 'run', str(manifest)]) == 0
    assert manifest.read_bytes() == before


def test_all_static_jobs_have_an_explicit_output_route_or_no_artifact_output(checkout):
    special = {'setup.sbatch', 'publication_source_audit.sbatch', 'softrole_evaluate.sbatch', 'softrole_pcp_recovery.sbatch'}
    assert {p.name for p in (checkout / 'slurm').glob('*.sbatch')} == set(submit.ENV_ROUTES) | set(submit.CLI_ROUTES) | special
