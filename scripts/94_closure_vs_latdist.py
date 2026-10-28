#!/usr/bin/env python3
"""94_closure_vs_latdist.py — 决定性诊断: 闭合率 vs 晶格距离 (real vs gen)
楔形采样 (node-uniform 中心), 仅保留双端为视叶且有传播坐标的楔形;
按端点晶格距离分箱 (同半球) / 跨半球单列, 统计闭合率。
用法: python3 94_closure_vs_latdist.py <out.json> <gen_prefix>
"""
import numpy as np, json, sys, time
from scipy.sparse import csr_matrix
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
z = np.load(D + "lat2q_f.npz"); PQ = z["pq"]; HEM = z["hem"]
N = len(sc)
om = sc == 5
has_coord = om & (HEM != 0)

BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 1e9)]
NB = len(BINS)

def run(pre, post, nsamp=1_500_000, seed=0):
    lo = np.minimum(pre, post).astype(np.int64); hi = np.maximum(pre, post).astype(np.int64)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N)); A.data[:] = 1
    ip, ix = A.indptr, A.indices
    deg = np.diff(ip); cand = np.where(deg >= 2)[0]
    rng = np.random.default_rng(seed)
    tot = np.zeros(NB + 1, np.int64); cls = np.zeros(NB + 1, np.int64)  # 最后一箱=跨半球
    ctr_sc = np.zeros(NB + 1, np.int64)  # 中心为视叶的计数 (副统计)
    t = 0
    while t < nsamp:
        u = int(cand[rng.integers(0, len(cand))])
        ns_ = ix[ip[u]:ip[u + 1]]
        if len(ns_) < 2: continue
        v, w_ = ns_[rng.integers(0, len(ns_), 2)]
        if v == w_: continue
        if not (has_coord[v] and has_coord[w_]):
            t += 1; continue
        a_, b_ = (v, w_) if v < w_ else (w_, v)
        hit = np.searchsorted(ix[ip[a_]:ip[a_ + 1]], b_)
        cl = int(ip[a_] + hit < ip[a_ + 1] and ix[ip[a_] + hit] == b_)
        if HEM[v] == HEM[w_]:
            dd = float(np.hypot(PQ[v, 0] - PQ[w_, 0], PQ[v, 1] - PQ[w_, 1]))
            bi = NB
            for k, (lo_, hi_) in enumerate(BINS):
                if lo_ <= dd < hi_: bi = k; break
        else:
            bi = NB
        tot[bi] += 1; cls[bi] += cl
        if om[u]: ctr_sc[bi] += 1
        t += 1
    return tot, cls, ctr_sc

tot_r, cls_r, ctr_r = run(ep, eq)
gp = np.fromfile(sys.argv[2] + "_pre.i32", dtype=np.int32)
gq = np.fromfile(sys.argv[2] + "_post.i32", dtype=np.int32)
tot_g, cls_g, ctr_g = run(gp, gq)

out = {"bins": BINS + ["cross_hem"],
       "real": {"tot": tot_r.tolist(), "cls": cls_r.tolist(),
                "rate": (cls_r / np.maximum(tot_r, 1)).round(4).tolist(),
                "optic_center_frac": (ctr_r / np.maximum(tot_r, 1)).round(3).tolist()},
       "gen": {"tot": tot_g.tolist(), "cls": cls_g.tolist(),
               "rate": (cls_g / np.maximum(tot_g, 1)).round(4).tolist()}}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("bin           n_real   real   n_gen    gen   ratio")
for k in range(NB + 1):
    nm = "cross" if k == NB else f"{BINS[k][0]:g}-{BINS[k][1]:g}"
    rr = cls_r[k] / max(tot_r[k], 1); gg = cls_g[k] / max(tot_g[k], 1)
    print(f"{nm:10s} {tot_r[k]:8,} {rr:6.3f} {tot_g[k]:8,} {gg:6.3f}  {rr/max(gg,1e-9):5.2f}x")
print(f"{time.time()-t0:.0f}s")
