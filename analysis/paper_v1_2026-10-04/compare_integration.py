"""Recheck paired final model/Adam/RNG states from validate_integration.py."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def compare(left, right, errors):
    import torch
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        atol, rtol = (1e-6, 1e-5) if left.dtype == torch.float32 else (1e-10, 1e-8)
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        if left.numel():
            key = str(left.dtype)
            errors[key] = max(errors.get(key, 0), float((left - right).abs().max()))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key], errors)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare(a, b, errors)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from publication_reconstruction.artifacts import load_checkpoint
    from publication_reconstruction.runtime.hetnet_ext.signatures import tree_signature
    rows = json.loads((args.runs / 'validation.json').read_text())['rows']
    results = []
    for index in range(0, len(rows), 2):
        left, right = rows[index:index + 2]
        assert (left['backend'], right['backend']) == ('dgl', 'torch-v1')
        assert all(left[key] == right[key] for key in ('task', 'variant', 'collectors'))
        states = []
        for row in (left, right):
            run = Path(row['run'])
            status = json.loads((run / 'run_status.json').read_text())
            states.append(load_checkpoint(run, status['checkpoint']))
        model_errors, optimizer_errors = {}, {}
        compare(states[0]['policy_net'], states[1]['policy_net'], model_errors)
        compare(states[0]['trainer'], states[1]['trainer'], optimizer_errors)
        assert left['counts'] == right['counts']
        assert (tree_signature(states[0]['recovery']['rng_states']) ==
                tree_signature(states[1]['recovery']['rng_states']))
        results.append(dict(task=left['task'], variant=left['variant'], collectors=left['collectors'],
            model_max_absolute_error=model_errors, optimizer_max_absolute_error=optimizer_errors,
            final_rng_equal=True, counts_equal=True))
    with args.output.open('x') as stream:
        json.dump(dict(passed=True, pairs=results), stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
