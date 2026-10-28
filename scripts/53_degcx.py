#!/usr/bin/env python3
"""Stage 53: 度校正 CX R² —— 检验 v4.2 CX 退化是否评价伪影 (Gram/rank-1 地板)
同协议三图: real / genv4 / genv42
模型: K ~ [1, cos]            -> 原口径 R²
      K ~ [1, r1]             -> 度积地板独占
      K ~ [1, r1, cos]        -> 全模型; ΔR²(cos|r1) = 环在度地板之上的贡献
"""
import numpy as np
import json, gc

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
CX = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]

def cx_kernel(pre, post, ntype):
    tn = [str(x) for x in np.load(OUT + "type_names.npy", allow_pickle=True)]
    cx_idx = [tn.index(x) for x in CX]
    sel = np.isin(ntype[pre], cx_idx) & np.isin(ntype[post], cx_idx)
    cp, cq = pre[sel], post[sel]
    cells = np.unique(np.r_[cp, cq])
    bidx = np.full(len(ntype), -1, np.int64)
    bidx[cells] = np.arange(len(cells))
    ctname = np.array([tn[ntype[b]] for b in cells])
    W = np.zeros((len(cells), len(cells)), np.float32)
    np.add.at(W, (bidx[cq], bidx[cp]), 1)
    epg = np.where(ctname == "EPG")[0]
    K = W[np.ix_(epg, epg)].copy()
    for x in ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]:
        mid = np.where(ctname == x)[0]
        if len(mid):
            K += W[np.ix_(epg, mid)] @ W[mid][:, epg]
    return K

def evaluate(K):
    S_ = K + K.T
    A_ = S_ / (S_.sum(1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    th = np.arctan2(evec[:, -2], evec[:, -3])
    d = th[:, None] - th[None, :]
    y = K.ravel()
    s = K.sum(1)
    r1 = np.outer(s, s).ravel()
    r1 = r1 / (r1.mean() + 1e-9)
    cosf = np.cos(d).ravel()
    one = np.ones_like(y)
    def fit(X):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), coef
    r2_std, c_std = fit(np.stack([one, cosf], -1))
    r2_r1, _ = fit(np.stack([one, r1], -1))
    r2_full, c_full = fit(np.stack([one, r1, cosf], -1))
    return {"r2_std": float(r2_std), "r2_rank1": float(r2_r1),
            "r2_full": float(r2_full), "dR2_cos_given_r1": float(r2_full - r2_r1),
            "b_std": float(c_std[1]), "b_cos_full": float(c_full[2])}

res = {}

# real (typed 子图重标定)
node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
tm = node_type >= 0
remap = np.cumsum(tm, dtype=np.int64) - 1
pre = remap[np.fromfile(OUT + "e_pre.i32", dtype=np.int32)]
post = remap[np.fromfile(OUT + "e_post.i32", dtype=np.int32)]
nt = node_type[tm]
res["real"] = evaluate(cx_kernel(pre, post, nt))
print("real:", res["real"], flush=True)
del pre, post, nt, node_type, remap; gc.collect()

# gen v4.1 / v4.2
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = g['type_count'].astype(np.int64)
gen_type = np.repeat(np.arange(len(tc), dtype=np.int32), tc)
for tag, path in [("v4.1", "/tmp/flydata/gen/genv4_S1"), ("v4.2", "/tmp/flydata/gen/genv42_S1")]:
    pre = np.fromfile(path + "_pre.i32", dtype=np.int32)
    post = np.fromfile(path + "_post.i32", dtype=np.int32)
    res[tag] = evaluate(cx_kernel(pre, post, gen_type))
    print(tag, ":", res[tag], flush=True)
    del pre, post; gc.collect()

json.dump(res, open(ROOT + "degcx.json", "w"), indent=1)
print("SAVED degcx.json")
