#!/usr/bin/env python3
"""Stage 39b: 传递性 (scipy.sparse CSR, 低内存版) —— 补 39 被 OOM 打断的部分"""
import numpy as np
import json
from scipy.sparse import csr_matrix

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"

def log(*a): print(*a, flush=True)

def transitivity(pre, post, N, nsamp=150_000, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.minimum(pre, post)
    hi = np.maximum(pre, post)
    r = np.concatenate([lo, hi])
    c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr)
    cand = np.where(deg >= 2)[0]
    closed = total = 0
    while total < nsamp:
        u = int(cand[rng.integers(0, len(cand))])
        ns = indices[indptr[u]:indptr[u + 1]]
        if len(ns) < 2:
            continue
        v, w_ = ns[rng.integers(0, len(ns), 2)]
        if v == w_:
            continue
        a_, b_ = (v, w_) if v < w_ else (w_, v)
        hit = np.searchsorted(indices[indptr[a_]:indptr[a_ + 1]], b_)
        total += 1
        if indptr[a_] + hit < indptr[a_ + 1] and indices[indptr[a_] + hit] == b_:
            closed += 1
    return closed / total

res = {}
for name, path, N in [("real", OUT + "e", len(np.fromfile(OUT + "node_type.i32", dtype=np.int32))),
                      ("v4", "/tmp/flydata/gen/genv4_S1", None)]:
    pre = np.memmap((path + "_pre.i32") if name == "v4" else (OUT + "e_pre.i32"), dtype=np.int32, mode="r")
    post = np.memmap((path + "_post.i32") if name == "v4" else (OUT + "e_post.i32"), dtype=np.int32, mode="r")
    if N is None:
        N = int(max(pre.max(), post.max())) + 1
    t = transitivity(np.asarray(pre), np.asarray(post), N)
    res[name] = t
    log(f"transitivity {name}: {t:.4f}")
    del pre, post

try:
    v = json.load(open(ROOT + "v4_validation.json"))
except Exception:
    v = {}
v["trans_real"] = res["real"]; v["trans_v4"] = res["v4"]
json.dump(v, open(ROOT + "v4_validation.json", "w"), indent=1)
log("saved:", v)
