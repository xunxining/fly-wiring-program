#!/usr/bin/env python3
"""Stage 58: v5 (CX 低噪声放置) 全流水线验证
KS / 互惠 / CX R²(谱θ) / 真θ 振幅+残差 / 逐项 / CX 细胞度 CV 对照真实
用法: python3 58_validate_v5.py [gen_prefix] [out.json]
"""
import numpy as np
import json, gc, sys, os

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
GEN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flydata/gen/genv5_S1"
OUTJ = sys.argv[2] if len(sys.argv) > 2 else ROOT + "v5_validation.json"
T = 11751
CX = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]
HOPS = ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]

def log(*a): print(*a, flush=True)

node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
tm = node_type >= 0
remap = np.cumsum(tm, dtype=np.int64) - 1
rpre = remap[np.fromfile(OUT + "e_pre.i32", dtype=np.int32)]
rpost = remap[np.fromfile(OUT + "e_post.i32", dtype=np.int32)]
nt_real = node_type[tm]

gpre = np.fromfile(GEN + "_pre.i32", dtype=np.int32)
gpost = np.fromfile(GEN + "_post.i32", dtype=np.int32)
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = g['type_count'].astype(np.int64)
N_gen = int(tc.sum())
gen_type = np.repeat(np.arange(T, dtype=np.int32), tc)
toff = np.zeros(T, np.int64); toff[1:] = np.cumsum(tc)[:-1]

def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    q = np.concatenate([a, b])
    ca = np.searchsorted(a, q, side="right") / len(a)
    cb = np.searchsorted(b, q, side="right") / len(b)
    return float(np.abs(ca - cb).max())

res = {}
res["ks_out"] = ks(np.bincount(rpre, minlength=len(nt_real)), np.bincount(gpre, minlength=N_gen))
res["ks_in"] = ks(np.bincount(rpost, minlength=len(nt_real)), np.bincount(gpost, minlength=N_gen))
log(f"degree KS: out={res['ks_out']:.4f} in={res['ks_in']:.4f}  [v4.1: 0.0134/0.0088]")

def recip(pre, post, N):
    k = pre.astype(np.int64) * N + post.astype(np.int64)
    k.sort()
    keep = np.ones(len(k), dtype=bool)
    np.not_equal(k[1:], k[:-1], out=keep[1:])
    ku = k[keep]
    rev = (ku % N) * N + ku // N
    i = np.clip(np.searchsorted(ku, rev), 0, len(ku) - 1)
    return float((ku[i] == rev).mean())

res["recip"] = recip(gpre, gpost, N_gen)
log(f"reciprocity: {res['recip']:.4f}  [real 0.2997, v4.1 0.0777]")

# ---- CX ----
tn_str = [str(x) for x in np.load(OUT + "type_names.npy", allow_pickle=True)]
cx_idx = [tn_str.index(x) for x in CX]

def build(pre, post, ntype):
    sel = np.isin(ntype[pre], cx_idx) & np.isin(ntype[post], cx_idx)
    cp, cq = pre[sel], post[sel]
    cells = np.unique(np.r_[cp, cq])
    bidx = np.full(len(ntype), -1, np.int64)
    bidx[cells] = np.arange(len(cells))
    ctname = np.array([tn_str[ntype[b]] for b in cells])
    W = np.zeros((len(cells), len(cells)), np.float32)
    np.add.at(W, (bidx[cq], bidx[cp]), 1)
    epg = np.where(ctname == "EPG")[0]
    K = W[np.ix_(epg, epg)].copy()
    terms = {"direct": W[np.ix_(epg, epg)].copy()}
    for x in HOPS:
        mid = np.where(ctname == x)[0]
        if len(mid):
            T2 = W[np.ix_(epg, mid)] @ W[mid][:, epg]
            K += T2
            terms[x] = T2
    return W, K, terms, cells, epg, ctname

def r2_spectral(K):
    S_ = K + K.T
    A_ = S_ / (S_.sum(1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    th = np.arctan2(evec[:, -2], evec[:, -3])
    d = th[:, None] - th[None, :]
    y = K.ravel()
    X = np.stack([np.ones_like(y), np.cos(d).ravel()], -1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def stats(K, th):
    d = th[:, None] - th[None, :]
    y = K.ravel()
    X = np.stack([np.ones_like(y), np.cos(d).ravel(), np.sin(d).ravel()], -1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    tot = ((y - y.mean()) ** 2).sum(); r_ = ((y - pred) ** 2).sum()
    return {"amp": float(np.hypot(coef[1], coef[2])), "resid_var": float(r_ / len(y)),
            "R2": float(1 - r_ / tot), "Kmax": float(K.max()), "Kmean": float(K.mean())}

Wg, Kg, terms_g, cells_g, epg_g, ctname_g = build(gpre, gpost, gen_type)
res["cx_r2_spectral"] = r2_spectral(Kg)
log(f"CX R2 (spectral θ): {res['cx_r2_spectral']:.3f}  [v4.1: 0.474, real: 0.617]")

# 真 θ
cz = np.load(GEN + "_cxtheta.npz")
th_node = np.full(N_gen, np.nan)
p = 0
for k, ty in enumerate(cz["tidx"]):
    n = int(tc[ty])
    th_node[toff[ty]:toff[ty] + n] = cz["theta"][p:p + n]
    p += n
th_g = th_node[cells_g[epg_g]]
res["cx_true"] = stats(Kg, th_g)
log(f"CX 真θ: R2={res['cx_true']['R2']:.3f} amp={res['cx_true']['amp']:.2f} "
    f"resid={res['cx_true']['resid_var']:.1f} Kmax={res['cx_true']['Kmax']:.0f}")
res["cx_terms"] = {k: stats(v, th_g)["R2"] for k, v in terms_g.items()}
log("逐项 R2:", {k: round(v, 3) for k, v in res["cx_terms"].items()})

# CX 细胞度 CV: EPG 细胞的 CX 总度(入+出), real vs gen
def cx_deg_cv(pre, post, ntype):
    sel = np.isin(ntype[pre], cx_idx) & np.isin(ntype[post], cx_idx)
    epg_nodes = np.where(ntype == tn_str.index("EPG"))[0]
    deg = np.zeros(len(ntype), np.int64)
    np.add.at(deg, pre[sel], 1)
    np.add.at(deg, post[sel], 1)
    d = deg[epg_nodes].astype(float)
    return float(d.std() / d.mean()), float(d.mean())

cv_r = cx_deg_cv(rpre, rpost, nt_real)
cv_g = cx_deg_cv(gpre, gpost, gen_type)
res["epg_cxdeg_cv_real"], res["epg_cxdeg_mean_real"] = cv_r
res["epg_cxdeg_cv_gen"], res["epg_cxdeg_mean_gen"] = cv_g
log(f"EPG CX度 CV: real={cv_r[0]:.3f}(mean {cv_r[1]:.0f}) gen={cv_g[0]:.3f}(mean {cv_g[1]:.0f})")

res["edges"] = int(len(gpre))
json.dump(res, open(OUTJ, "w"), indent=1)
log("SAVED", OUTJ)
