"""Execute real batch scripts with mocked module/srun and actual CLI dry runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix='hetnet-slurm-check-') as temporary:
        folder = Path(temporary)
        for name, body in {
            'module': '#!/bin/sh\nexit 0\n',
            'srun': '#!/bin/sh\n[ "$1" = "--cpu-bind=cores" ] || exit 9\nshift\nexec "$@"\n',
        }.items():
            path = folder / name
            path.write_text(body)
            path.chmod(0o755)
        environment = {**os.environ, 'PATH': str(folder) + os.pathsep + os.environ['PATH'],
            'SLURM_SUBMIT_DIR': str(ROOT), 'SLURM_JOB_ID': 'TEST_ONLY',
            'SLURM_ARRAY_JOB_ID': 'TEST_ONLY', 'HETNET_PYTHON': sys.executable,
            'PUBLICATION_RUN_ROOT': str(folder / 'hetnet'),
            'PUBLICATION_PREFLIGHT_ROOT': str(folder / 'preflight'),
            'PUBLICATION_BENCHMARK_ROOT': str(folder / 'benchmark'),
            'SOFTROLE_PAPER_RUN_ROOT': str(folder / 'softrole')}
        for script, count, backends in (
            ('publication_paper_preflight', 4, ('dgl', 'torch-v1')),
            ('publication_paper_train', 12, ('dgl', 'torch-v1')),
            ('softrole_paper_fc', 6, ('dgl',)),
            ('publication_backend_benchmark', 1, ('dgl',)),
        ):
            for backend in backends:
                for index in range(count):
                    command = ['bash', str(ROOT / 'slurm' / (script + '.sbatch')), '--dry-run']
                    completed = subprocess.run(command, cwd=ROOT, env={**environment,
                        'SLURM_ARRAY_TASK_ID': str(index), 'PUBLICATION_MESSAGE_BACKEND': backend},
                        check=True, capture_output=True, text=True)
                    protocol = json.loads(completed.stdout)
                    rows.append(dict(script=script, index=index, backend=backend, command=command,
                                     protocol=protocol))
        for name in ('hetnet', 'preflight', 'benchmark', 'softrole'):
            assert not (folder / name).exists(), 'A dry run created training output'
    with args.output.open('x') as stream:
        json.dump(dict(passed=True, jobs_submitted=False, invocations=len(rows), rows=rows), stream, indent=2)
        stream.write('\n')
    print(f'{len(rows)} real batch-script/CLI dry runs passed; no jobs or training outputs.')


if __name__ == '__main__':
    main()
