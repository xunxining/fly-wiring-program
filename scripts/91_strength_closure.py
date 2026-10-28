#!/usr/bin/env python3
"""91_strength_closure.py — 真实图: 闭合率 × 楔形两边强度 (突触数) + 闭合边的互惠状态
输出 data_female/strength_closure_diag.json
"""
import numpy as np, json, time
from scipy.sparse import csr_matrix
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
ew = np.fromfile(D + "e_w.i32", dtype=np.int32)
N = 138327

# 有向邻接 (合并同向多重边为强度)
key = ep.astype(np.int64) * N + eq
uk, inv = np.unique(key, return_inverse=True)
wsum = np.bincount(inv, weights=ew).astype(np.int32)
d_pre = (uk // N).astype(np.int32); d_post = (uk % N).astype(np.int32)
del key, inv, ep, eq, ew

# 无向 CSR (存在性) + 有向查找 (强度/互惠)
lo = np.minimum(d_pre, d_post).astype(np.int64); hi = np.maximum(d_pre, d_post).astype(np.int64)
r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N)); A.data[:] = 1
ip, ix = A.indptr, A.indices
deg = np.diff(ip); cand = np.where(deg >= 2)[0]

dkey = d_pre.astype(np.int64) * N + d_post          # 已排序 (unique)
def dfind(a, b):
    """有向边 a->b 的强度 (0=无)"""
    kk = a.astype(np.int64) * N + b
    i = np.searchsorted(dkey, kk); i = np.clip(i, 0, len(dkey) - 1)
    hit = dkey[i] == kk
    return np.where(hit, wsum[i], 0)

rng = np.random.default_rng(0)
sbins = [1, 3, 8, 20, 50, 10**9]                     # 强度分箱 (突触数)
slab = ["1-2", "3-7", "8-19", "20-49", "50+"]
NS = 400_000
tot = np.zeros((len(sbins) - 1, len(sbins) - 1), np.int64)
cls = np.zeros_like(tot)
# 闭合边互惠状态
mut_n = [0, 0]; mut_c = [0, 0]   # [闭合边互鉴于否] → (tot, closed)
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
    bi = np.searchsorted(sbins, wv, side="right") - 1
    bj = np.searchsorted(sbins, ww, side="right") - 1
    tot[bi, bj] += 1; cls[bi, bj] += cl
    if cl:
        vw = dfind(np.array([v]), np.array([w_]))[0]; wv2 = dfind(np.array([w_]), np.array([v]))[0]
        m = int((vw > 0) and (wv2 > 0))
        mut_n[m] += 1; mut_c[m] += 1
    t += 1
out = {"strength_bins": slab,
       "closure_by_strength": [[float(cls[i, j] / max(tot[i, j], 1)) if tot[i, j] >= 100 else None
                                for j in range(len(slab))] for i in range(len(slab))],
       "counts": tot.tolist(),
       "T_check": float(cls.sum() / tot.sum()),
       "closed_edge_mutual_frac": float(mut_c[1] / max(sum(mut_c), 1)),
       "n_closed": int(sum(mut_c))}
json.dump(out, open(D + "strength_closure_diag.json", "w"), indent=1)
print(json.dumps(out["closure_by_strength"]))
print("T_check", out["T_check"], "closed-mutual", out["closed_edge_mutual_frac"], f"{time.time()-t0:.0f}s")
