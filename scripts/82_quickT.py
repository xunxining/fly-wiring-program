#!/usr/bin/env python3
"""82_quickT.py — 快速检验: T + ks_out + 边数 (用于 NPOV 实验迭代)"""
import numpy as np, sys, time, json
from scipy.sparse import csr_matrix
D = "/mnt/agents/output/果蝇生成器/data_female/"
prefix = sys.argv[1]
ep = np.fromfile(D+"e_pre.i32", dtype=np.int32); eq = np.fromfile(D+"e_post.i32", dtype=np.int32)
gp = np.fromfile(prefix+"_pre.i32", dtype=np.int32); gq = np.fromfile(prefix+"_post.i32", dtype=np.int32)
N = 138327
def transitivity(pre, post, nsamp=150_000, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.minimum(pre, post); hi = np.maximum(pre, post)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr); cand = np.where(deg >= 2)[0]
    closed = total = 0
    while total < nsamp:
        u = int(cand[rng.integers(0, len(cand))])
        ns_ = indices[indptr[u]:indptr[u+1]]
        if len(ns_) < 2: continue
        v, w_ = ns_[rng.integers(0, len(ns_), 2)]
        if v == w_: continue
        a_, b_ = (v, w_) if v < w_ else (w_, v)
        hit = np.searchsorted(indices[indptr[a_]:indptr[a_+1]], b_)
        total += 1
        if indptr[a_]+hit < indptr[a_+1] and indices[indptr[a_]+hit] == b_:
            closed += 1
    return closed/total
Tg = transitivity(gp, gq)
od_r = np.bincount(ep, minlength=N); od_g = np.bincount(gp, minlength=N)
mx = max(od_r.max(), od_g.max())
hr = np.bincount(od_r, minlength=mx+1)/N; hg = np.bincount(od_g, minlength=mx+1)/N
ks_out = float(np.abs(np.cumsum(hr)-np.cumsum(hg)).max())
out = {"prefix": prefix, "E_gen": int(len(gp)), "E_real": int(len(ep)),
       "T_gen": float(Tg), "ks_out": ks_out}
print(json.dumps(out)); 
json.dump(out, open(sys.argv[2] if len(sys.argv)>2 else "/tmp/quickT.json","w"))
