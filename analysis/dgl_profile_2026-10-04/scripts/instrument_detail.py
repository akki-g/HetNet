import sys, os, time, runpy, functools, atexit, json, collections
RT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime')
sys.path.insert(0, RT)
MODE = os.environ.get('INSTRUMENT', '1') == '1'
acc = collections.Counter(); calls = collections.Counter(); depth = [0]
def wrap(cat, f):
    @functools.wraps(f)
    def g(*a, **k):
        if depth[0]:
            return f(*a, **k)
        depth[0] += 1; t = time.perf_counter()
        try: return f(*a, **k)
        finally:
            acc[cat] += time.perf_counter() - t; calls[cat] += 1; depth[0] -= 1
    return g
def wrap_free(cat, f):  # timed regardless of depth (for totals)
    @functools.wraps(f)
    def g(*a, **k):
        t = time.perf_counter()
        try: return f(*a, **k)
        finally: acc[cat] += time.perf_counter() - t; calls[cat] += 1
    return g
if MODE:
    import torch, dgl, dgl.view, dgl.heterograph
    import dgl.backend.pytorch.sparse as sp
    import hetgat.utils, hetgat.uavnet, hetgat.graph.fastreal as fr, hetgat.graph.fastbinary as fb, env_wrappers, trainer
    H = dgl.DGLGraph; dview = sys.modules["dgl.view"]
    for m in ['update_all', 'apply_edges', '__getitem__', 'number_of_edges']:
        setattr(H, m, wrap('dgl_fwd:graph.'+m, getattr(H, m)))
    for cls in [dview.HeteroNodeDataView, dview.HeteroEdgeDataView]:
        for m in ['__setitem__', '__getitem__', '__delitem__']:
            setattr(cls, m, wrap('dgl_fwd:'+cls.__name__+'.'+m, getattr(cls, m)))
    for cls in [dview.HeteroNodeView]:
        setattr(cls, '__getitem__', wrap('dgl_fwd:HeteroNodeView.__getitem__', cls.__getitem__))
    for mod in (fr, fb):
        mod.edge_softmax = wrap('dgl_fwd:edge_softmax', mod.edge_softmax)
    for name in ['GSpMM', 'GSDDMM', 'EdgeSoftmax', 'GSpMM_hetero', 'GSDDMM_hetero', 'EdgeSoftmax_hetero']:
        c = getattr(sp, name); c.backward = staticmethod(wrap_free('dgl_backward_ops', c.backward))
    hetgat.utils.build_hetgraph = wrap('graph_build', hetgat.utils.build_hetgraph)
    import hetgat.policy as pol
    if hasattr(pol, 'build_hetgraph'): pol.build_hetgraph = hetgat.utils.build_hetgraph
    if hasattr(hetgat.uavnet, 'build_hetgraph'): hetgat.uavnet.build_hetgraph = hetgat.utils.build_hetgraph
    U = hetgat.uavnet.UAVNetA2CEasy; U.forward = wrap_free('model_forward_total', U.forward)
    torch.autograd.backward = wrap_free('autograd_backward_total', torch.autograd.backward)
    env_wrappers.GymWrapper.step = wrap_free('env_step', env_wrappers.GymWrapper.step) if hasattr(env_wrappers, 'GymWrapper') else None
    def dump():
        json.dump({'seconds': acc, 'calls': calls}, open(os.environ['INSTRUMENT_OUT'], 'w'), indent=1)
    atexit.register(dump)
sys.argv = [os.path.join(RT, 'main.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
