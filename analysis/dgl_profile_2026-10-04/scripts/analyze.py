import pstats, sys, collections
for V in ['real','binary']:
    st = pstats.Stats(f'{V}.prof'); s = st.stats
    total = sum(v[2] for v in s.values())  # total tottime
    bymod = collections.Counter()
    for (fn, ln, name), (cc, nc, tt, ct, callers) in s.items():
        if '/dgl/' in fn: k = 'dgl (python side)'
        elif '/torch/' in fn: k = 'torch python wrappers'
        elif fn == '~':
            k = 'C builtins: ' + ('torch' if 'torch' in name or 'Tensor' in name or '_C.' in name else 'other')
            if 'run_backward' in name: k = 'C: autograd backward'
            if 'dgl' in name.lower() or '_CAPI' in name: k = 'C: dgl capi'
        elif '/numpy/' in fn: k = 'numpy'
        elif '/envs/' in fn or 'env_wrappers' in fn: k = 'environment'
        elif '/runtime/' in fn: k = 'runtime model/learner python'
        else: k = 'other'
        bymod[k] += tt
    print(f'== {V}: total {total:.1f}s')
    for k, v in bymod.most_common(): print(f'   {k:32s} {v:6.2f}s {100*v/total:5.1f}%')
    want = ['get_episode','train_batch','compute_grad','build_hetgraph','heterograph','forward','run_backward','step','edge_softmax','update_all','apply_edges','gumbel_softmax','get_obs_features','remove_excess_action_features_from_all']
    rows=[]
    for (fn, ln, name), (cc, nc, tt, ct, callers) in s.items():
        if name in want and (('runtime' in fn) or ('dgl' in fn) or fn=='~' or 'functional' in fn):
            rows.append((ct, nc, name, fn.split('site-packages/')[-1].split('runtime/')[-1], ln))
    for ct, nc, name, fn, ln in sorted(rows, reverse=True)[:30]:
        print(f'   cum {ct:7.2f}s {100*ct/total:5.1f}%  calls {nc:7d}  {name}  {fn}:{ln}')
