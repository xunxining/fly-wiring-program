#!/usr/bin/env python3
# 146_vnc_comm.py — ②b-1d (§53): VNC 闭合的中尺度社区(Louvain)载体检验
import json, time
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + "data_male/"
G = ROOT + "gen_edges/g_p9bm2"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

node_sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
N = len(node_sc)
VNC = {15, 16, 17, 21, 22, 23, 24, 25, 26, 27}
is_vnc = np.isin(node_sc, list(VNC))
vidx = np.where(is_vnc)[0]

# ---- real VNC 诱导子图 (无向, 权重=双向和) ----
pre = np.fromfile(D + "e_pre.i32", dtype=np.int32); post = np.fromfile(D + "e_post.i32", dtype=np.int32)
w = np.fromfile(D + "e_w.i32", dtype=np.int32)
m = is_vnc[pre] & is_vnc[post]
lo = np.minimum(pre[m], post[m]); hi = np.maximum(pre[m], post[m]); ww = w[m].astype(np.float64)
key = lo.astype(np.int64) * N + hi
uk, inv = np.unique(key, return_inverse=True)
uw = np.zeros(len(uk)); np.add.at(uw, inv, ww)
del pre, post, w, m, lo, hi, ww, key, inv
Gnx = nx.Graph()
Gnx.add_nodes_from(range(len(vidx)))
pos = {g: i for i, g in enumerate(vidx)}
eu = np.searchsorted(vidx, uk % N); ev = np.searchsorted(vidx, uk // N)
Gnx.add_edges_from(zip(eu.tolist(), ev.tolist(), ({"weight": float(x)} for x in uw)))
log(f"子图: {Gnx.number_of_nodes():,} 节点 {Gnx.number_of_edges():,} 边")
comm = nx.community.louvain_communities(Gnx, weight="weight", resolution=1.0, seed=13)
COM = np.full(N, -1, dtype=np.int32)
for ci, cset in enumerate(comm):
    for li in cset: COM[vidx[li]] = ci
sizes = sorted((len(c) for c in comm), reverse=True)
Q = nx.community.modularity(Gnx, comm, weight="weight")
log(f"Louvain: {len(comm)} 社区, top sizes {sizes[:6]}, Q={Q:.3f}")
del Gnx

def analyze(pre_f, post_f, nsamp, seed):
    pre = np.fromfile(pre_f, dtype=np.int32); post = np.fromfile(post_f, dtype=np.int32)
    lo = np.minimum(pre, post); hi = np.maximum(pre, post)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    ek = np.unique(lo.astype(np.int64) * N + hi)
    del pre, post, lo, hi, r, c, A
    rng = np.random.default_rng(seed)
    cand = np.where(np.diff(indptr) >= 2)[0]
    U = np.empty(0, np.int64); V = np.empty(0, np.int64); W = np.empty(0, np.int64)
    need = nsamp
    while len(U) < nsamp:
        u = cand[rng.integers(0, len(cand), need)]
        s0, s1 = indptr[u], indptr[u + 1]
        ln = (s1 - s0)
        o1 = rng.integers(0, ln); o2 = rng.integers(0, ln)
        v = indices[s0 + o1]; w2 = indices[s0 + o2]
        ok = v != w2
        U = np.concatenate([U, u[ok]]); V = np.concatenate([V, v[ok]]); W = np.concatenate([W, w2[ok]])
    U, V, W = U[:nsamp], V[:nsamp], W[:nsamp]
    a = np.minimum(V, W).astype(np.int64); b = np.maximum(V, W).astype(np.int64)
    kk = a * N + b
    hit = np.searchsorted(ek, kk)
    closed = (hit < len(ek)) & (ek[np.clip(hit, 0, len(ek) - 1)] == kk)
    del ek, indptr, indices
    return dict(V=V, W=W, closed=closed)

NS = 6_000_000
log("抽样 real ..."); re_ = analyze(D + "e_pre.i32", D + "e_post.i32", NS, 1)
log("抽样 gen  ..."); ge = analyze(G + "_pre.i32", G + "_post.i32", NS, 1)

mB_r = (COM[re_["V"]] >= 0) & (COM[re_["W"]] >= 0)
mB_g = (COM[ge["V"]] >= 0) & (COM[ge["W"]] >= 0)
Vr, Wr, clr = re_["V"][mB_r], re_["W"][mB_r], re_["closed"][mB_r]
Vg, Wg, clg = ge["V"][mB_g], ge["W"][mB_g], ge_["closed" if False else "closed"][mB_g]
same_r = COM[Vr] == COM[Wr]
same_g = COM[Vg] == COM[Wg]
E_real = clr[same_r].mean() / clr[~same_r].mean()
E_gen = clg[same_g].mean() / clg[~same_g].mean()
gap_same = same_r.mean() * (clr[same_r].mean() - clg[same_g].mean())
gap_diff = (1 - same_r.mean()) * (clr[~same_r].mean() - clg[~same_g].mean())
gap_share = gap_same / (gap_same + gap_diff)
log(f"E_real {E_real:.2f} E_gen {E_gen:.2f}, sameC share {same_r.mean():.3f}, gap share {gap_share:.2f}")

rng = np.random.default_rng(7)
cells = np.where(COM >= 0)[0]
nulls = np.empty(200)
for i in range(200):
    perm = rng.permutation(len(cells))
    cp = np.full(N, -1, dtype=np.int32); cp[cells] = COM[cells[perm]]
    same = cp[Vr] == cp[Wr]
    nulls[i] = clr[same].mean() / clr[~same].mean()
p99 = float(np.percentile(nulls, 99))
log(f"null p99 {p99:.3f}")

out = dict(protocol="§53 ②b-1d mesoscale community carrier test", nsamp=NS,
           n_communities=len(comm), modularity=float(Q), top_sizes=sizes[:10],
           n_wedges_real=int(mB_r.sum()), sameC_share=float(same_r.mean()),
           p_real_same=float(clr[same_r].mean()), p_real_diff=float(clr[~same_r].mean()),
           p_gen_same=float(clg[same_g].mean()), p_gen_diff=float(clg[~same_g].mean()),
           E_real=float(E_real), E_gen=float(E_gen), E_ratio=float(E_real / E_gen),
           gap_share_sameC=float(gap_share),
           null_mean=float(nulls.mean()), null_p99=p99,
           gates=dict(G1d=bool(E_real >= 3.0 and E_real > p99 and E_real / E_gen >= 1.5 and gap_share >= 0.50)))
json.dump(out, open(D + "vnc_comm.json", "w"), indent=1)
log("-> data_male/vnc_comm.json")
print(json.dumps(out["gates"], indent=1))
print({k: out[k] for k in ("E_real", "E_gen", "E_ratio", "sameC_share", "gap_share_sameC", "n_communities", "modularity")})
