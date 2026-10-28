#!/usr/bin/env python3
"""91b_strength_closure_gen.py — 任意生成图: 闭合率 × 楔形两边强度 (与 91 同口径)
用法: python3 91b_strength_closure_gen.py <prefix> <out.json>
"""
import numpy as np, json, sys, time
from scipy.sparse import csr_matrix
D = "/mnt/agents/output/果蝇生成器/data_female/"
prefix, outj = sys.argv[1], sys.argv[2]
t0 = time.time()
ep = np.fromfile(prefix + "_pre.i32", dtype=np.int32)
eq = np.fromfile(prefix + "_post.i32", dtype=np.int32)
ew = np.fromfile(prefix + "_w.i32", dtype=np.int32)
N = 138327
key = ep.astype(np.int64) * N + eq
uk, inv = np.unique(key, return_inverse=True)
wsum = np.bincount(inv, weights=ew).astype(np.int32)
d_pre = (uk // N).astype(np.int32); d_post = (uk % N).astype(np.int32)
del key, inv, ep, eq, ew
lo = np.minimum(d_pre, d_post).astype(np.int64); hi = np.maximum(d_pre, d_post).astype(np.int64)
r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N)); A.data[:] = 1
ip, ix = A.indptr, A.indices
deg = np.diff(ip); cand = np.where(deg >= 2)[0]
dkey = d_pre.astype(np.int64) * N + d_post
def dfind(a, b):
    kk = a.astype(np.int64) * N + b
    i = np.searchsorted(dkey, kk); i = np.clip(i, 0, len(dkey) - 1)
    return np.where(dkey[i] == kk, wsum[i], 0)
rng = np.random.default_rng(0)
sbins = [1, 3, 8, 20, 50, 10**9]
slab = ["1-2", "3-7", "8-19", "20-49", "50+"]
NS = 400_000
tot = np.zeros((5, 5), np.int64); cls = np.zeros((5, 5), np.int64)
mut_c = [0, 0]
t = 0
while t < NS:
    u = int(cand[rng.integers(0, len(cand))])
    ns_ = ix[ip[u]:ip[u + 1]]
    if len(ns_) < 2: continue
    v, w_ = ns_[rng.integers(0, len(ns_), 2)]
    if v == w_: continue
    a_, b_ = (v, w_) if v < w_ else (w_, v)
    hit = np.searchsorted(ix[ip[a_]:ip[a_ + 1]], b_)
    cl = ip[a_] + hit < ip[a_ + 1] and ix[ip[a_] + hit] == b_
    wv = dfind(np.array([u]), np.array([v]))[0] + dfind(np.array([v]), np.array([u]))[0]
    ww = dfind(np.array([u]), np.array([w_]))[0] + dfind(np.array([w_]), np.array([u]))[0]
    bi = min(np.searchsorted(sbins, wv, side="right") - 1, 4)
    bj = min(np.searchsorted(sbins, ww, side="right") - 1, 4)
    tot[bi, bj] += 1; cls[bi, bj] += cl
    if cl:
        vw = dfind(np.array([v]), np.array([w_]))[0]; wv2 = dfind(np.array([w_]), np.array([v]))[0]
        mut_c[int((vw > 0) and (wv2 > 0))] += 1
    t += 1
out = {"prefix": prefix, "strength_bins": slab,
       "closure_by_strength": [[float(cls[i, j] / max(tot[i, j], 1)) if tot[i, j] >= 100 else None
                                for j in range(5)] for i in range(5)],
       "counts": tot.tolist(), "T_check": float(cls.sum() / tot.sum()),
       "closed_edge_mutual_frac": float(mut_c[1] / max(sum(mut_c), 1))}
json.dump(out, open(outj, "w"), indent=1)
print(json.dumps(out["closure_by_strength"]))
print("T_check", out["T_check"], "mutual", out["closed_edge_mutual_frac"], f"{time.time()-t0:.0f}s")
