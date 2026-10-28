#!/usr/bin/env python3
"""87_closslope.py — 生成网络的 闭合率×重叠数 曲线 (对比 81 的实测曲线)
楔形采样: 随机中心 u, 取两邻居 v,w; 重叠数 = ov(v,w) [无向: OUT|IN 对称并集掩膜]
分层: 0 / 1 / 2 / 3-5 / 5-8 / 8-15 / 15+
用法: python3 87_closslope.py <prefix> <out.json>
"""
import numpy as np, sys, json
from scipy.sparse import csr_matrix
D = "/mnt/agents/output/果蝇生成器/data_female/"
_z = np.load(D + "np_masks.npz")
OUT_lo, OUT_hi, IN_lo, IN_hi = _z["OUT_lo"], _z["OUT_hi"], _z["IN_lo"], _z["IN_hi"]
A_lo = OUT_lo | IN_lo; A_hi = OUT_hi | IN_hi   # 无向 arbor 掩膜
N = len(A_lo)
prefix = sys.argv[1]
gp = np.fromfile(prefix + "_pre.i32", dtype=np.int32)
gq = np.fromfile(prefix + "_post.i32", dtype=np.int32)
lo_ = np.minimum(gp, gq); hi_ = np.maximum(gp, gq)
r = np.concatenate([lo_, hi_]); c = np.concatenate([hi_, lo_])
A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
A.data[:] = 1
indptr, indices = A.indptr, A.indices
deg = np.diff(indptr)
cand = np.where(deg >= 2)[0]
rng = np.random.default_rng(1)
bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 14), (15, 10**9)]
tot = np.zeros(len(bins), np.int64); cls = np.zeros(len(bins), np.int64)
TRIALS = 300_000
done = 0
while done < TRIALS:
    u = int(cand[rng.integers(0, len(cand))])
    ns_ = indices[indptr[u]:indptr[u + 1]]
    if len(ns_) < 2: continue
    v, w_ = ns_[rng.integers(0, len(ns_), 2)]
    if v == w_: continue
    ov = (int(A_lo[v] & A_lo[w_]).bit_count() + int(A_hi[v] & A_hi[w_]).bit_count())
    bi = next(i for i, (a, b) in enumerate(bins) if a <= ov <= b)
    a_, b_ = (v, w_) if v < w_ else (w_, v)
    hit = np.searchsorted(indices[indptr[a_]:indptr[a_ + 1]], b_)
    closed = indptr[a_] + hit < indptr[a_ + 1] and indices[indptr[a_] + hit] == b_
    tot[bi] += 1; cls[bi] += closed; done += 1
labels = ["0", "1", "2", "3-4", "5-7", "8-14", "15+"]
out = {"prefix": prefix, "trials": TRIALS,
       "by_bin": {labels[i]: {"n": int(tot[i]), "closure": float(cls[i] / max(tot[i], 1))} for i in range(len(bins))},
       "T_overall": float(cls.sum() / max(tot.sum(), 1))}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], "w"))
