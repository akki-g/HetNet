#!/usr/bin/env python3
"""Submit a Stokes job with its logs and outputs in one titled runs/ directory.

Create a container with --title/--purpose, or reuse --run-dir for another stage.
--init-only prepares the container without submitting; --dry-run has no side effects.
Arguments after -- are sbatch resource options, a batch script, and its arguments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]

# These are output roots, not checkpoint/archive inputs. Existing scientific
# task/model/seed subdirectories remain the responsibility of their launchers.
ENV_ROUTES = {
    'reproduce.sbatch': {'HETNET_RUN_ROOT': 'training'},
    'softrole.sbatch': {'SOFTROLE_RUN_ROOT': 'training'},
    'softrole_failure.sbatch': {'SOFTROLE_FAILURE_RUN_ROOT': 'training'},
    'softrole_adam_pcp.sbatch': {'SOFTROLE_ADAM_PCP_RUN_ROOT': 'training'},
    'softrole_channels_pcp.sbatch': {'SOFTROLE_CHANNELS_PCP_RUN_ROOT': 'training'},
    'softrole_lr_pcp.sbatch': {'SOFTROLE_LR_PCP_RUN_ROOT': 'training'},
    'softrole_paper_fc.sbatch': {'SOFTROLE_PAPER_RUN_ROOT': 'training'},
    'softrole_programme.sbatch': {'SOFTROLE_PROGRAMME_RUN_ROOT': 'training'},
    'softrole_capcon_40m.sbatch': {'SOFTROLE_PROGRAMME_RUN_ROOT': 'training'},
    'softrole_pcp_additional_seeds.sbatch': {'PCP_ADDITIONAL_RUN_ROOT': 'training'},
    'softrole_diagnostics.sbatch': {'CAPCOM_DIAGNOSTICS_OUTPUT_ROOT': 'diagnostics'},
    'softrole_adam_diagnostics.sbatch': {'CAPCOM_ADAM_OUTPUT_ROOT': 'diagnostics'},
    'softrole_graded_pilot.sbatch': {'SOFTROLE_GRADED_PILOT_RUN_ROOT': 'training',
                                     'SOFTROLE_GRADED_PILOT_MANIFEST': 'manifest.json'},
    'softrole_graded_pilot_prepare.sbatch': {'SOFTROLE_GRADED_PILOT_RUN_ROOT': 'training',
                                             'SOFTROLE_GRADED_PILOT_MANIFEST': 'manifest.json'},
    'softrole_mechanism_train.sbatch': {'SOFTROLE_MECHANISM_RUN_ROOT': 'training',
                                        'SOFTROLE_MECHANISM_MANIFEST': 'manifest.json'},
    'softrole_mechanism_prepare.sbatch': {'SOFTROLE_MECHANISM_RUN_ROOT': 'training',
                                          'SOFTROLE_MECHANISM_MANIFEST': 'manifest.json'},
    # The training run container (SOFTROLE_MECHANISM_TRAINING_RUN) is a read-only input, not routed here.
    'softrole_mechanism_eval.sbatch': {'SOFTROLE_MECHANISM_EVAL_DIR': 'evaluation'},
    'softrole_mechanism_eval_prepare.sbatch': {'SOFTROLE_MECHANISM_EVAL_DIR': 'evaluation'},
    'softrole_stale_prepare.sbatch': {'SOFTROLE_STALE_RUN_ROOT': 'training',
                                      'SOFTROLE_STALE_MANIFEST': 'manifest.json'},
    'softrole_stale_train.sbatch': {'SOFTROLE_STALE_RUN_ROOT': 'training',
                                    'SOFTROLE_STALE_MANIFEST': 'manifest.json'},
    'softrole_stale_gate_summary.sbatch': {'SOFTROLE_STALE_RUN_ROOT': 'training',
                                           'SOFTROLE_STALE_MANIFEST': 'manifest.json',
                                           'SOFTROLE_STALE_SUMMARY': 'gate_summary.json'},
    'publication_train.sbatch': {'PUBLICATION_RUN_ROOT': 'training'},
    'publication_paper_train.sbatch': {'PUBLICATION_RUN_ROOT': 'training'},
    'publication_preflight.sbatch': {'PUBLICATION_PREFLIGHT_ROOT': 'preflight'},
    'publication_paper_preflight.sbatch': {'PUBLICATION_PREFLIGHT_ROOT': 'preflight'},
    'publication_backend_benchmark.sbatch': {'PUBLICATION_BENCHMARK_ROOT': 'benchmark'},
}
CLI_ROUTES = {
    'publication_evaluate.sbatch': 'evaluation.json',
    'publication_prepare_evaluation.sbatch': 'evaluation',
    'publication_resume.sbatch': 'continuation',
    'softrole_capcon_readiness.sbatch': 'readiness',
    'softrole_prepare_frozen.sbatch': 'evaluation',
}
RUN_ROOT_OUTPUT_JOBS = {
    'publication_train.sbatch', 'publication_preflight.sbatch',
    'publication_paper_train.sbatch', 'publication_paper_preflight.sbatch',
    'softrole_graded_pilot.sbatch', 'softrole_mechanism_train.sbatch',
    'softrole_pcp_additional_seeds.sbatch',
    'softrole_stale_prepare.sbatch', 'softrole_stale_train.sbatch',
    'softrole_stale_gate_summary.sbatch',
}


def absolute(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def inside(path, root):
    path = absolute(path)
    if not path.is_relative_to(root):
        raise ValueError(f'Output must be inside {root}: {path}')
    return path


def slug(title):
    value = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:80].rstrip('-')
    if not value:
        raise ValueError('Title must contain letters or numbers describing the test')
    return value


def split_command(command):
    command = list(command)
    if command[:1] == ['--']:
        command.pop(0)
    indices = [i for i, value in enumerate(command) if value.endswith('.sbatch') and not value.startswith('-')]
    if not indices:
        raise ValueError('Provide a .sbatch script after --')
    index = indices[0]
    options, script, arguments = command[:index], absolute(command[index]), command[index + 1:]
    if not script.is_file():
        raise ValueError(f'Batch script not found: {script}')
    # The helper owns these options so environment overrides cannot split a run.
    blocked = ('--output', '--error', '--chdir', '--job-name', '--wrap', '--export-file')
    short = ('-o', '-e', '-D', '-J')
    for value in options:
        if value == '--parsable' or value == '--export=ALL':
            continue
        if any(value == flag or value.startswith(flag + '=') for flag in blocked) or any(value.startswith(flag) for flag in short):
            raise ValueError(f'The run helper controls {value}; set --title and --run-dir instead')
        if value.startswith('--export'):
            raise ValueError('The run helper requires --export=ALL')
    return [x for x in options if x not in ('--parsable', '--export=ALL')], script, arguments


def route_outputs(script, arguments, run_dir, environment):
    values = {}
    for variable, relative in ENV_ROUTES.get(script.name, {}).items():
        values[variable] = str(inside(environment.get(variable) or run_dir / relative, run_dir))
    arguments = list(arguments)
    output_found = False
    output_flags = {'--output', '--output-root'}
    if script.name in RUN_ROOT_OUTPUT_JOBS:
        output_flags.add('--run-root')
    for index, value in enumerate(arguments):
        option = value.split('=', 1)[0]
        # Downstream argparse CLIs accept unique abbreviations. Require complete
        # output option names so an abbreviation cannot bypass containment.
        if option.startswith('--') and option not in output_flags and any(
                flag.startswith(option) for flag in output_flags):
            raise ValueError(f'Use the complete output option name instead of {option}')
        if value in output_flags:
            if index + 1 == len(arguments):
                raise ValueError(f'{value} needs a path')
            arguments[index + 1] = str(inside(arguments[index + 1], run_dir))
            output_found |= value == '--output'
        elif option in output_flags and '=' in value:
            arguments[index] = option + '=' + str(inside(value.split('=', 1)[1], run_dir))
            output_found |= option == '--output'
    if script.name in CLI_ROUTES and not output_found:
        arguments += ['--output', str(run_dir / CLI_ROUTES[script.name])]
    if script.name == 'softrole_evaluate.sbatch':
        if not arguments or arguments[0].startswith('-'):
            raise ValueError('SoftRole evaluation requires a checkpoint argument')
        if len(arguments) == 1 or arguments[1].startswith('-'):
            arguments.insert(1, str(run_dir / 'evaluation.json'))
        else:
            arguments[1] = str(inside(arguments[1], run_dir))
    if script.name == 'softrole_pcp_recovery.sbatch':
        if len(arguments) < 2:
            raise ValueError('Recovery requires a phase and prepared manifest')
        arguments[1] = str(inside(arguments[1], run_dir))
    # Generated scripts contain locked literal destinations, so submit them from
    # the same parent container, never under an unrelated newly-created run.
    if script.parent != ROOT / 'slurm':
        inside(script, run_dir)
    return arguments, values


def write_new(path, payload):
    with path.open('x') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--title', help='Clear description of the comparison or check')
    result.add_argument('--purpose', help='Question, contrast or behavior this run tests')
    result.add_argument('--run-dir', type=Path, help='Container under runs/; reuse it for preparation/training/evaluation stages')
    mode = result.add_mutually_exclusive_group()
    mode.add_argument('--init-only', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    result.add_argument('command', nargs=argparse.REMAINDER)
    return result


def execute(args):
    created = datetime.now(timezone.utc)
    existing = None
    if args.run_dir:
        run_dir = inside(args.run_dir, (ROOT / 'runs').resolve())
        if run_dir == (ROOT / 'runs').resolve():
            raise ValueError('Choose a named child directory under runs/')
        # Existing prepared containers may be adopted, but their infrastructure
        # must not redirect logs, receipts or metadata outside this experiment.
        for name in ('logs', 'submissions', 'run_info.json', 'README.md'):
            inside(run_dir / name, run_dir)
        info = run_dir / 'run_info.json'
        if info.is_file():
            existing = json.loads(info.read_text())
            if existing.get('schema_version') != 1:
                raise ValueError('Unsupported run_info.json schema')
            if args.title and args.title.strip() != existing['title']:
                raise ValueError('Existing run title differs; choose another directory')
            if args.purpose and args.purpose.strip() != existing['purpose']:
                raise ValueError('Existing run purpose differs; choose another directory')
    title = existing['title'] if existing else (args.title or '').strip()
    purpose = existing['purpose'] if existing else (args.purpose or '').strip()
    if not title or not purpose:
        raise ValueError('A new run needs both --title and --purpose')
    title_slug = slug(title)
    if not args.run_dir:
        run_dir = ROOT / 'runs' / f"{created:%Y%m%dT%H%M%SZ}_{title_slug}_{uuid.uuid4().hex[:6]}"
    if args.init_only:
        if args.command:
            raise ValueError('--init-only does not accept a job command')
        command, script, exported = None, None, {}
    else:
        options, script, arguments = split_command(args.command)
        arguments, exported = route_outputs(script, arguments, run_dir, os.environ)
        stage = slug(script.stem)
        command = ['sbatch', *options, '--parsable', '--export=ALL', '--chdir', str(ROOT),
                   '--job-name', f'{title_slug}-{stage}', '--open-mode=append',
                   '--output', str(run_dir / 'logs' / f'{stage}-%A_%a.out'),
                   '--error', str(run_dir / 'logs' / f'{stage}-%A_%a.err'), str(script), *arguments]
    exported.update(HETNET_RUN_DIR=str(run_dir), HETNET_RUN_TITLE=title,
                    HETNET_SUBMISSION_PYTHON=sys.executable)
    if args.dry_run:
        print(json.dumps(dict(run_dir=str(run_dir), title=title, purpose=purpose,
                              command=command, output_environment=exported), indent=2))
        return 0
    # This parent is a container. Do not precreate any trainer/evaluator leaf.
    if not args.run_dir and run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))
    (run_dir / 'logs').mkdir(exist_ok=True)
    (run_dir / 'submissions').mkdir(exist_ok=True)
    if existing is None:
        git = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        write_new(run_dir / 'run_info.json', dict(schema_version=1, title=title, purpose=purpose,
                  created_utc=created.isoformat(), run_dir=str(run_dir), checkout=str(ROOT),
                  git_commit=git.stdout.strip() if git.returncode == 0 else None))
        readme = run_dir / 'README.md'
        if not readme.exists():
            readme.write_text(f'# {title}\n\n{purpose}\n\nCreated {created.isoformat()} (UTC).\n\n'
                              'Scheduler output is in `logs/`; submission commands and job IDs are in\n'
                              '`submissions/`. Protocol manifests, source snapshots, metrics, checkpoints\n'
                              'and results stay inside this directory. Submission success is not evidence\n'
                              'that training or evaluation completed; inspect the task result records.\n')
    if args.init_only:
        print(run_dir)
        return 0
    token = f"{created:%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:6]}"
    receipt = dict(schema_version=1, title=title, purpose=purpose, submitted_utc=created.isoformat(),
                   script=str(script), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                   command=command, output_environment=exported)
    write_new(run_dir / 'submissions' / f'{token}.request.json', receipt)
    environment = os.environ.copy()
    environment.update(exported)
    # Shell startup hooks can otherwise change the fixed scientific environment.
    environment.pop('BASH_ENV', None)
    environment.pop('ENV', None)
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        response = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except OSError as error:
        response = dict(returncode=127, stdout='', stderr=str(error))
    identifier = response['stdout'].strip()
    accepted = response['returncode'] == 0 and re.fullmatch(r'[1-9][0-9]*(;[^\s;]+)?', identifier) is not None
    response.update(job_id=identifier.split(';', 1)[0] if accepted else None,
                    submission_accepted=accepted)
    write_new(run_dir / 'submissions' / f'{token}.response.json', response)
    if response['stderr']:
        print(response['stderr'], file=sys.stderr, end='' if response['stderr'].endswith('\n') else '\n')
    if not accepted:
        print(f'Submission failed or returned an invalid job ID; see {run_dir / "submissions"}', file=sys.stderr)
        return response['returncode'] or 2
    print(identifier)  # stdout stays usable in shell dependency captures.
    return 0


def main(argv=None):
    try:
        return execute(parser().parse_args(argv))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f'submit_stokes: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
