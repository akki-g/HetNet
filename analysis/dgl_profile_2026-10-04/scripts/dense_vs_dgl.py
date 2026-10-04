# Microbenchmark: the DGL-specific part of one HetGAT step (3 layers x 5 non-empty relations,
# 4 heads x 16, 2P/1A/2 state nodes, float64 features, float32 attention vectors) vs a dense
# PyTorch equivalent. Same arithmetic shape; not a bitwise-equivalence check.
import time, torch, dgl, dgl.function as fn
from dgl.ops import edge_softmax
torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(0)
H, D, STEPS, REPS = 4, 16, 80, 6
rels = {('P','p2p','P'): ([0,1],[1,0]), ('P','p2a','A'): ([0,1],[0,0]), ('A','a2p','P'): ([0,0],[0,1]),
        ('A','a2a','A'): ([],[]), ('P','p2s','state'): ([0,1],[0,0]), ('A','a2s','state'): ([0],[1])}
N = {'P': 2, 'A': 1, 'state': 2}
def make_graph():
    return dgl.heterograph({k: (torch.tensor(s, dtype=torch.int64), torch.tensor(d, dtype=torch.int64)) for k, (s, d) in rels.items()}, num_nodes_dict=N)
W = {r: (torch.randn(1,H,D,dtype=torch.float32,requires_grad=True), torch.randn(1,H,D,dtype=torch.float32,requires_grad=True)) for (_,r,_) in rels}
leaky = torch.nn.LeakyReLU(0.2)
def feats():
    return {t: torch.randn(n, H, D, requires_grad=True) for t, n in N.items()}
def step_dgl(g, x):
    out = 0
    for layer in range(3):
        for (s, r, d), (src_idx, _) in rels.items():
            if g[r].number_of_edges() == 0: continue
            ws, wd = W[r]
            a_s = (x[s] * ws).sum(-1).unsqueeze(-1); a_d = (x[d] * wd).sum(-1).unsqueeze(-1)
            g.nodes[s].data['m_'+r] = x[s]
            g[r].srcdata.update({'as_'+r: a_s}); g[r].dstdata.update({'ad_'+r: a_d})
            g[r].apply_edges(fn.u_add_v('as_'+r, 'ad_'+r, 'e_'+r))
            e = leaky(g[r].edata.pop('e_'+r))
            g[r].edata['a_'+r] = edge_softmax(g[r], e)
            g[r].update_all(fn.u_mul_e('m_'+r, 'a_'+r, 'mm_'+r), fn.sum('mm_'+r, 'ft_'+r))
            out = out + g.nodes[d].data['ft_'+r].sum()
    return out
masks = {}
for (s, r, d), (si, di) in rels.items():
    m = torch.zeros(N[d], N[s], dtype=torch.bool)
    for a, b in zip(si, di): m[b, a] = True
    masks[r] = m
def step_dense(x):
    out = 0
    for layer in range(3):
        for (s, r, d) in rels:
            m = masks[r]
            if not m.any(): continue
            ws, wd = W[r]
            a_s = (x[s] * ws).sum(-1); a_d = (x[d] * wd).sum(-1)          # (Ns,H), (Nd,H)
            e = leaky(a_d[:, None, :] + a_s[None, :, :])                    # (Nd,Ns,H)
            e = e.masked_fill(~m[:, :, None], float('-inf'))
            alpha = torch.softmax(e, dim=1).nan_to_num(0.0)                 # empty rows -> 0
            out = out + torch.einsum('dsh,shf->dhf', alpha, x[s]).sum()
    return out
def bench(kind):
    fwd = bwd = build = 0.0
    for rep in range(REPS):
        total = 0
        for t in range(STEPS):
            x = feats()
            if kind == 'dgl':
                t0 = time.perf_counter(); g = make_graph(); build += time.perf_counter() - t0
                t0 = time.perf_counter(); total = total + step_dgl(g, x); fwd += time.perf_counter() - t0
            else:
                t0 = time.perf_counter(); total = total + step_dense(x); fwd += time.perf_counter() - t0
        t0 = time.perf_counter(); total.backward(); bwd += time.perf_counter() - t0
    n = REPS * STEPS
    return 1e3*build/n, 1e3*fwd/n, 1e3*bwd/n
bench('dgl'); bench('dense')  # warmup
for kind in ['dgl', 'dense', 'dgl', 'dense']:
    b, f, k = bench(kind)
    print(f'{kind:6s} per step: graph build {b:.3f} ms, message passing forward {f:.3f} ms, backward {k:.3f} ms, total {b+f+k:.3f} ms')
