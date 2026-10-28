#!/usr/bin/env python3
"""76_female_validate.py — 雌性 gen 验证 (与雄性 39/67/69 同度量)
用法: python3 76_female_validate.py <gen_prefix> <out_json>
"""
import numpy as np, json, time, sys, gc
from scipy.sparse import csr_matrix

D = "/mnt/agents/output/果蝇生成器/data_female/"
prefix, out_json = sys.argv[1], sys.argv[2]
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)

node_type = np.fromfile(D + "node_type.i32", dtype=np.int32).astype(np.int64)
types = np.load(D + "type_names.npy", allow_pickle=True)
T = len(types); N = len(node_type)
tc = np.bincount(node_type, minlength=T)
toff = np.concatenate([[0], np.cumsum(tc)])
xyz = np.fromfile(D + "xyz.f32", dtype=np.float32).reshape(N, 3)

# real
rep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
req = np.fromfile(D + "e_post.i32", dtype=np.int32)
odr = np.bincount(rep, minlength=N); idr = np.bincount(req, minlength=N)

# gen
gp = np.fromfile(prefix + "_pre.i32", dtype=np.int32)
gq = np.fromfile(prefix + "_post.i32", dtype=np.int32)
odg = np.bincount(gp, minlength=N); idg = np.bincount(gq, minlength=N)
log(f"real E={len(rep):,}  gen E={len(gp):,}")

def transitivity(pre, post, N, nsamp=150_000, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.minimum(pre, post); hi = np.maximum(pre, post)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    cand = np.where(np.diff(indptr) >= 2)[0]
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
    return closed / total

def recip(pre, post):
    kk1 = pre.astype(np.int64) * N + post
    kk1.sort()
    kk2 = post.astype(np.int64) * N + pre
    hit = np.searchsorted(kk1, kk2)
    ok = (hit < len(kk1)) & (kk1[np.clip(hit, 0, len(kk1)-1)] == kk2)
    return float(ok.mean())

def ks(a, b):
    aa = np.sort(a); bb = np.sort(b)
    al = np.concatenate([aa, bb])
    ca = np.searchsorted(aa, al, side="right") / len(aa)
    cb = np.searchsorted(bb, al, side="right") / len(bb)
    return float(np.abs(ca - cb).max())

res = {"E_real": int(len(rep)), "E_gen": int(len(gp))}
res["T_gen"] = transitivity(gp, gq, N)
res["recip_gen"] = recip(gp, gq)
res["recip_real"] = recip(rep, req)
res["ks_out"] = ks(odr, odg); res["ks_in"] = ks(idr, idg)
log("T/recip/KS done", {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})

# 逐型 CV
rows = []
for t in range(T):
    if tc[t] < 3: continue
    seg = slice(toff[t], toff[t+1])
    for nm, dr, dg in [("out", odr, odg), ("in", idr, idg)]:
        a = dr[seg].astype(float); b = dg[seg].astype(float)
        if a.mean() > 5:
            rows.append((t, nm, float(a.mean()), float(a.std()/a.mean()),
                         float(b.std()/max(b.mean(), 1e-9))))
cv_r = np.array([r[3] for r in rows]); cv_g = np.array([r[4] for r in rows])
mn = np.array([r[2] for r in rows]); poi = 1/np.sqrt(mn)
res["cv_corr"] = float(np.corrcoef(cv_r, cv_g)[0, 1])
res["ratio_real_median"] = float(np.median(cv_r/poi))
res["ratio_gen_median"] = float(np.median(cv_g/poi))
tn = {str(x): i for i, x in enumerate(types)}
if "EPG" in tn:
    t = tn["EPG"]; seg = slice(toff[t], toff[t+1])
    ar = odr[seg].astype(float); ag = odg[seg].astype(float)
    res["EPG"] = {"n": int(tc[t]),
                  "out_cv_real": float(ar.std()/ar.mean()),
                  "out_cv_gen": float(ag.std()/max(ag.mean(), 1e-9)),
                  "out_mean_real": float(ar.mean()), "out_mean_gen": float(ag.mean())}
log("CV done", {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})

# 距离 (分块, 低内存)
bins = np.exp(np.linspace(np.log(0.5), np.log(200), 25))
def dprof(p, q):
    n = len(p); h = np.zeros(len(bins)-1, np.int64)
    medsamp = np.empty(min(n, 5_000_000))
    off = 0
    CH = 4_000_000
    for i in range(0, n, CH):
        d = np.sqrt(((xyz[p[i:i+CH]].astype(np.float64) - xyz[q[i:i+CH]].astype(np.float64))**2).sum(-1))
        h += np.histogram(d, bins=bins)[0]
        if i < 5_000_000:
            medsamp[off:off+min(CH, 5_000_000-off)] = d[:min(CH, 5_000_000-off)]
            off += min(CH, 5_000_000-off)
        del d
    return medsamp[:off], h
dr_, hr_ = dprof(rep, req)
dg_, hg_ = dprof(gp, gq)
cr = np.cumsum(hr_/hr_.sum()); cg = np.cumsum(hg_/hg_.sum())
res["dist"] = {"median_real": float(np.median(dr_)), "median_gen": float(np.median(dg_)),
               "ks": float(np.abs(cr-cg).max()),
               "p_lt2_real": float((dr_<2).mean()), "p_lt2_gen": float((dg_<2).mean())}
log("dist done", res["dist"])

json.dump(res, open(out_json, "w"), indent=1)
log("VALIDATE DONE ->", out_json)
