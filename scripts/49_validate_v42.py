#!/usr/bin/env python3
"""Stage 49: v4.2 验证 —— KS / 互惠 / CX R² (triad 由 45 另跑)"""
import numpy as np
import json, time, gc, os

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
GEN = "/tmp/flydata/gen/genv42_S1"
T = 11751

def log(*a): print(*a, flush=True)

node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
rpre = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="r")
rpost = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="r")
N_real = len(node_type)
tm = node_type >= 0

gpre = np.memmap(GEN + "_pre.i32", dtype=np.int32, mode="r")
gpost = np.memmap(GEN + "_post.i32", dtype=np.int32, mode="r")
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = g['type_count'].astype(np.int64)
N_gen = int(tc.sum())
gen_type = np.repeat(np.arange(T, dtype=np.int32), tc)

def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    q = np.concatenate([a, b])
    ca = np.searchsorted(a, q, side="right") / len(a)
    cb = np.searchsorted(b, q, side="right") / len(b)
    return float(np.abs(ca - cb).max())

r_out = np.bincount(np.asarray(rpre), minlength=N_real)[tm]
r_in = np.bincount(np.asarray(rpost), minlength=N_real)[tm]
g_out = np.bincount(np.asarray(gpre), minlength=N_gen)
g_in = np.bincount(np.asarray(gpost), minlength=N_gen)
ks_out = ks(r_out, g_out); ks_in = ks(r_in, g_in)
log(f"degree KS: out={ks_out:.4f} in={ks_in:.4f}  [v4.1: 0.0134/0.0088]")
del r_out, r_in, g_out, g_in; gc.collect()

# ---- 互惠 (对级) ----
def recip(pre, post, N):
    k = pre.astype(np.int64) * N + post.astype(np.int64)
    k.sort()
    keep = np.ones(len(k), dtype=bool)
    np.not_equal(k[1:], k[:-1], out=keep[1:])
    ku = k[keep]
    rev = (ku % N) * N + ku // N
    i = np.searchsorted(ku, rev)
    i = np.clip(i, 0, len(ku) - 1)
    return float((ku[i] == rev).mean())

rec_gen = recip(np.asarray(gpre), np.asarray(gpost), N_gen)
log(f"reciprocity: v4.2={rec_gen:.4f}  [real 0.2997, v4.1 0.0777(pair)/0.0885(edge)]")

# ---- CX 核 ----
tn = [str(x) for x in np.load(OUT + "type_names.npy", allow_pickle=True)]
CX = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]
cx_idx = [tn.index(x) for x in CX]
sel = np.isin(gen_type[np.asarray(gpre)], cx_idx) & np.isin(gen_type[np.asarray(gpost)], cx_idx)
cp = np.asarray(gpre)[sel]; cq = np.asarray(gpost)[sel]
cells = np.unique(np.r_[cp, cq])
bidx = {b: i for i, b in enumerate(cells)}
ctname = np.array([tn[gen_type[b]] for b in cells])
W = np.zeros((len(cells), len(cells)), np.float32)
for p, q in zip(cp, cq):
    W[bidx[q], bidx[p]] += 1
epg = np.where(ctname == "EPG")[0]
K = W[np.ix_(epg, epg)].copy()
for x in ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]:
    mid = np.where(ctname == x)[0]
    if len(mid):
        K += W[np.ix_(epg, mid)] @ W[mid][:, epg]
S_ = np.abs(K) + np.abs(K).T
A_ = S_ / (S_.sum(1, keepdims=True) + 1e-9)
ev, evec = np.linalg.eigh(A_)
th = np.arctan2(evec[:, -2], evec[:, -3])
d = th[:, None] - th[None, :]
X = np.stack([np.ones_like(d), np.cos(d)], axis=-1).reshape(-1, 2)
y = K.ravel()
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
log(f"CX kernel: b={coef[1]:.1f}, R2={r2:.3f}  [v4.1: 0.474-0.508, 同口径真实上限 0.617]")

json.dump({"ks_out": ks_out, "ks_in": ks_in, "recip_v42": rec_gen,
           "cx_b": float(coef[1]), "cx_r2_v42": float(r2),
           "edges": int(len(gpre))},
          open(ROOT + "v42_validation.json", "w"), indent=1)
log("V42 VALIDATION DONE")
