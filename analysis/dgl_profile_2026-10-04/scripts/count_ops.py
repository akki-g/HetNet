"""Count PyTorch (aten) operations and their tensor sizes per environment step.

Mode 'insitu': runs the reconstruction runtime (copy at scripts/runtime) for ONE update under
torch.autograd.profiler (record_shapes) and attributes ops to forward or backward.
Mode 'dense': profiles the dense half of dense_vs_dgl.py (one 80-step rollout + backward).
DGL's own C kernels are not aten ops, so 'insitu' counts the non-DGL PyTorch work only.
Usage: python count_ops.py insitu OUT.json [runtime args...] | python count_ops.py dense OUT.json
"""
import json, math, os, runpy, statistics, sys, time
import torch
from torch.autograd import profiler

MODE, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
# Metadata/view and allocation ops that do not launch a compute kernel on a GPU; excluded from
# 'compute_leaf_ops_per_step' (the closer proxy for GPU kernel launches). Listed explicitly for review.
NON_COMPUTE = {'aten::as_strided', 'aten::slice', 'aten::select', 'aten::unsqueeze', 'aten::squeeze',
               'aten::view', 'aten::_unsafe_view', 'aten::reshape', 'aten::expand', 'aten::permute',
               'aten::transpose', 'aten::t', 'aten::alias', 'aten::detach', 'aten::lift_fresh',
               'aten::empty', 'aten::empty_strided', 'aten::empty_like', 'aten::resize_',
               'aten::set_', 'aten::split', 'aten::chunk', 'aten::unbind', 'aten::narrow', 'aten::unfold',
               'aten::_reshape_alias', 'aten::numpy_T', 'aten::result_type', 'aten::is_nonzero'}
HERE = os.path.dirname(os.path.abspath(__file__))


def summarize(prof, steps, label):
    events = prof.function_events
    def is_aten(e): return e.name.startswith('aten::')
    def backward(e):
        p = e.cpu_parent
        while p is not None:
            if p.name.startswith('autograd::engine::evaluate_function'):
                return True
            p = p.cpu_parent
        return False
    def has_aten_parent(e):
        p = e.cpu_parent
        while p is not None:
            if is_aten(p):
                return True
            p = p.cpu_parent
        return False
    out = {'label': label, 'environment_steps': steps}
    for part in ['forward', 'backward']:
        evs = [e for e in events if is_aten(e) and backward(e) == (part == 'backward')]
        top = [e for e in evs if not has_aten_parent(e)]
        leaf = [e for e in evs if not any(is_aten(c) for c in e.cpu_children)]
        numels = []
        for e in leaf:
            sizes = [math.prod(s) for s in (e.input_shapes or []) if isinstance(s, list) and s]
            numels.append(max(sizes) if sizes else 0)
        compute = [e for e in leaf if e.name not in NON_COMPUTE]
        names = {}
        for e in compute:
            names[e.name] = names.get(e.name, 0) + 1
        numels.sort()
        q = lambda f: numels[min(len(numels) - 1, int(f * len(numels)))] if numels else None
        out[part] = {'top_level_ops_per_step': len(top) / steps, 'leaf_ops_per_step': len(leaf) / steps,
                     'leaf_largest_input_numel_median': q(0.5), 'leaf_largest_input_numel_p90': q(0.9),
                     'leaf_largest_input_numel_p99': q(0.99), 'leaf_largest_input_numel_max': numels[-1] if numels else None,
                     'leaf_ops_with_largest_input_le_1024_fraction': (sum(n <= 1024 for n in numels) / len(numels)) if numels else None,
                     'compute_leaf_ops_per_step': len(compute) / steps,
                     'top_compute_ops_per_step': {k: v / steps for k, v in sorted(names.items(), key=lambda kv: -kv[1])[:15]}}
    return out


if MODE == 'dense':
    sys.argv = [sys.argv[0]]
    ns = runpy.run_path(os.path.join(HERE, 'dense_vs_dgl.py'), run_name='dense_vs_dgl_import')  # defines functions; runs its own benchmark too
    x_steps = []
    with profiler.profile(record_shapes=True) as prof:
        total = 0
        for t in range(ns['STEPS']):
            total = total + ns['step_dense'](ns['feats']())
        total.backward()
    result = summarize(prof, ns['STEPS'], 'dense message-passing microbenchmark (3 layers x 5 relations)')
else:
    RT = os.path.join(HERE, 'runtime')
    sys.path.insert(0, RT)
    import trainer as trainer_module
    state = {'result': None}
    original = trainer_module.Trainer.train_batch
    def profiled(self, *a, **k):
        if state['result'] is not None:
            return original(self, *a, **k)
        with profiler.profile(record_shapes=True) as prof:
            out = original(self, *a, **k)
        state['result'] = (prof, out)
        return out
    trainer_module.Trainer.train_batch = profiled
    import atexit
    def dump():
        prof, out = state['result']
        steps = json.loads(open(os.environ['COUNT_UPDATES']).readline())['steps']
        json.dump(summarize(prof, steps, 'reconstruction PCP one update, non-DGL aten ops'), open(OUT, 'w'), indent=1)
    atexit.register(dump)
    sys.argv = [os.path.join(RT, 'main.py')] + sys.argv[3:]
    runpy.run_path(sys.argv[0], run_name='__main__')
if MODE == 'dense':
    json.dump(result, open(OUT, 'w'), indent=1)
