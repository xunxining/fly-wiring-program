#!/usr/bin/env python3
"""73_female_diag.py — 雌性真实图四特征诊断 (跨性别泛化的真实侧基准)
与雄性度量定义完全一致:
  (1) 逐型度 CV / Poisson 地板比 (69_cellcv real 侧), EPG 专项
  (2) 传递性 T (39b 楔形抽样, 150k)
  (3) 边距离剖面 (67 real 侧): 全局直方图 + <2um 占比 + 中位数
  (4) CX 角结构预检: EPG/PEN/Delta7/PFL3/PFN/hDelta 细胞数 + 坐标覆盖
输出: data_female/diag_real.json
"""
import numpy as np, pandas as pd, json, time, gc
from scipy.sparse import csr_matrix

D = "/mnt/agents/output/果蝇生成器/data_female/"
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)

node_type = np.fromfile(D + "node_type.i32", dtype=np.int32)
types = np.load(D + "type_names.npy", allow_pickle=True)
T = len(types); N = len(node_type)
tc = np.bincount(node_type, minlength=T)
toff = np.concatenate([[0], np.cumsum(tc)])
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
log(f"N={N} T={T} E={len(ep)}")

# ---------- (1) 度 CV ----------
odr = np.bincount(ep, minlength=N); idr = np.bincount(eq, minlength=N)
rows = []
for t in range(T):
    if tc[t] < 3: continue
    seg = slice(toff[t], toff[t+1])
    for nm, dr in [("out", odr), ("in", idr)]:
        a = dr[seg].astype(float)
        if a.mean() > 5:
            rows.append((t, nm, float(a.mean()), float(a.std()/a.mean())))
cv = np.array([r[3] for r in rows]); mn = np.array([r[2] for r in rows])
poi = 1/np.sqrt(mn)
res = {"N": int(N), "T": int(T), "E": int(len(ep)),
       "cv_real_median": float(np.median(cv)),
       "cv_poisson_median": float(np.median(poi)),
       "ratio_real_median": float(np.median(cv/poi))}
# hot types: top decile mean degree
hot = mn >= np.quantile(mn, 0.9)
res["ratio_hot_median"] = float(np.median((cv/poi)[hot]))
res["ratio_cold_median"] = float(np.median((cv/poi)[~hot]))
# out/in ratio correlation within type
d = {}
for t, nm, m_, c_ in rows: d.setdefault(t, {})[nm] = (m_, c_/ (1/np.sqrt(m_)))
rr = [ (v["out"][1], v["in"][1]) for v in d.values() if "out" in v and "in" in v ]
res["ratio_outin_corr_median"] = float(np.corrcoef([r[0] for r in rr], [r[1] for r in rr])[0,1])
# EPG
tn = {str(x): i for i, x in enumerate(types)}
if "EPG" in tn:
    t = tn["EPG"]; seg = slice(toff[t], toff[t+1])
    a = odr[seg].astype(float)
    res["EPG"] = {"n": int(tc[t]), "out_mean": float(a.mean()),
                  "out_cv": float(a.std()/a.mean()),
                  "poisson_floor": float(1/np.sqrt(a.mean())),
                  "cv_over_floor": float(a.std()/a.mean()*np.sqrt(a.mean()))}
log("(1) CV done", {k: round(v,4) for k,v in res.items() if isinstance(v,float)})

# ---------- (2) 传递性 ----------
def transitivity(pre, post, N, nsamp=150_000, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.minimum(pre, post); hi = np.maximum(pre, post)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr)
    cand = np.where(deg >= 2)[0]
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
res["T_real"] = float(transitivity(ep, eq, N))
log("(2) T_real =", round(res["T_real"], 4))

# ---------- (3) 距离剖面 ----------
xyz = np.fromfile(D + "xyz.f32", dtype=np.float32).reshape(N, 3)
# subsample edges for speed if huge
E = len(ep)
idx = np.arange(E) if E <= 40_000_000 else np.random.default_rng(0).choice(E, 40_000_000, replace=False)
dd = np.sqrt(((xyz[ep[idx]] - xyz[eq[idx]])**2).sum(-1))
bins = np.exp(np.linspace(np.log(0.5), np.log(200), 25))
h, _ = np.histogram(dd, bins=bins)
res["dist"] = {"median": float(np.median(dd)),
               "p_lt2um": float((dd < 2).mean()),
               "hist": h.tolist(),
               "bin_edges": bins.tolist(),
               "nan_frac": float(np.isnan(dd).mean())}
log("(3) dist median =", round(res["dist"]["median"],2), "um, P<2um =", round(res["dist"]["p_lt2um"],4))

# ---------- (4) CX 预检 ----------
cxt = [s for s in tn if any(str(s).startswith(k) for k in
       ["EPG","PEN_","Delta7","PFL","PFN","hDelta"])]
res["cx_types"] = {s: int(tc[tn[s]]) for s in cxt}
res["cx_cells"] = int(sum(tc[tn[s]] for s in cxt))
log("(4) CX types:", len(cxt), "cells:", res["cx_cells"])

json.dump(res, open(D + "diag_real.json", "w"), indent=1)
log("DIAG DONE -> diag_real.json")
