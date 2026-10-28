#!/usr/bin/env python3
"""Stage 39: v4 S=1 验证
1) 度数 KS (typed out/in, vs 真实)
2) 互惠性精确值 (real 0.298 / v3 ~0.010)
3) CX 等效核 cos 拟合 R² (real 0.696 / v3 0.025)
4) 无向聚类系数(采样) (real vs v4)
"""
import numpy as np
import json, time

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
GEN = "/tmp/flydata/gen/genv4_S1"
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
log(f"real edges={len(rpre):,}  gen edges={len(gpre):,}  N_gen={N_gen:,}")

def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    q = np.concatenate([a, b])
    ca = np.searchsorted(a, q, side="right") / len(a)
    cb = np.searchsorted(b, q, side="right") / len(b)
    return float(np.abs(ca - cb).max())

# ---- 1) 度 KS ----
r_out = np.bincount(rpre, minlength=N_real)[tm]
r_in = np.bincount(rpost, minlength=N_real)[tm]
g_out = np.bincount(gpre, minlength=N_gen)
g_in = np.bincount(gpost, minlength=N_gen)
ks_out = ks(r_out, g_out); ks_in = ks(r_in, g_in)
log(f"degree KS: out={ks_out:.4f} in={ks_in:.4f}")
del r_out, r_in, g_out, g_in

# ---- 2) 互惠性 ----
def reciprocity(pre, post, N):
    k = pre.astype(np.int64) * N + post.astype(np.int64)
    ku = np.unique(k)
    rev = post.astype(np.int64) * N + pre.astype(np.int64)
    i = np.searchsorted(ku, rev)
    i = np.clip(i, 0, len(ku) - 1)
    return float((ku[i] == rev).mean())

rec_real = reciprocity(np.asarray(rpre), np.asarray(rpost), N_real)
rec_gen = reciprocity(np.asarray(gpre), np.asarray(gpost), N_gen)
log(f"reciprocity: real={rec_real:.4f} gen_v4={rec_gen:.4f}")

# ---- 3) CX 等效核 ----
tn = [str(x) for x in np.load(OUT + "type_names.npy", allow_pickle=True)]
CX = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]
cx_idx = [tn.index(x) for x in CX]
sel = np.isin(gen_type[gpre], cx_idx) & np.isin(gen_type[gpost], cx_idx)
cp, cq = np.asarray(gpre[sel]), np.asarray(gpost[sel])
ctype = gen_type[np.unique(np.r_[cp, cq])]
cells = np.unique(np.r_[cp, cq])
bidx = {b: i for i, b in enumerate(cells)}
Nc = len(cells)
W = np.zeros((Nc, Nc), np.float32)
for p, q in zip(cp, cq):
    W[bidx[q], bidx[p]] += 1
ct = np.array([tn.index(tn[gen_type[b]]) for b in cells])
ctname = [tn[gen_type[b]] for b in cells]
epg = np.where(np.array(ctname) == "EPG")[0]
log(f"CX cells={Nc}, EPG={len(epg)}, CX edges={sel.sum():,}")

def eff_kernel(Wm):
    K = Wm[np.ix_(epg, epg)].copy()
    for mid in [np.where(np.array(ctname) == x)[0] for x in ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]]:
        if len(mid):
            K += Wm[np.ix_(epg, mid)] @ Wm[mid][:, epg]
    return K

def ring_order(K):
    S_ = np.abs(K) + np.abs(K).T
    A_ = S_ / (S_.sum(axis=1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    return np.arctan2(evec[:, -2], evec[:, -3])

def cos_fit(K, th):
    d = th[:, None] - th[None, :]
    X = np.stack([np.ones_like(d), np.cos(d)], axis=-1).reshape(-1, 2)
    y = K.ravel()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(coef[1]), float(ss)

Kg = eff_kernel(W)
thg = ring_order(Kg)
bg, r2g = cos_fit(Kg, thg)
log(f"CX kernel (gen v4): cos amplitude b={bg:.1f}, R2={r2g:.3f}   [real: b=6696 R2=0.696, v3: b~0.025*real]")

# ---- 4) 传递性(wedge 采样, 无向) ----
def transitivity(pre, post, N, nsamp=120_000, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.minimum(pre, post).astype(np.int64)
    hi = np.maximum(pre, post).astype(np.int64)
    k1 = lo * N + hi
    o1 = np.argsort(k1); k1s = k1[o1]
    e1_lo, e1_hi = lo[o1], hi[o1]
    k2 = hi * N + lo
    o2 = np.argsort(k2); k2s = k2[o2]
    e2_lo, e2_hi = lo[o2], hi[o2]   # 按 hi 排序: u 的邻居在 hi==u 块
    deg = np.bincount(lo, minlength=N) + np.bincount(hi, minlength=N)
    cand = np.where(deg >= 2)[0]
    closed = total = 0
    for _ in range(nsamp):
        u = cand[rng.integers(0, len(cand))]
        b1 = np.searchsorted(k1s, u * N, side="left")
        e1_ = np.searchsorted(k1s, u * N + N, side="left")
        b2 = np.searchsorted(k2s, u * N, side="left")
        e2_ = np.searchsorted(k2s, u * N + N, side="left")
        ns = np.concatenate([e1_hi[b1:e1_], e2_lo[b2:e2_]])
        if len(ns) < 2: continue
        v, w_ = ns[rng.integers(0, len(ns), 2)]
        if v == w_: continue
        a_, b_ = (v, w_) if v < w_ else (w_, v)
        i = np.searchsorted(k1s, a_ * N + b_)
        total += 1
        if i < len(k1s) and k1s[i] == a_ * N + b_:
            closed += 1
    return closed / max(total, 1)

cl_real = transitivity(np.asarray(rpre), np.asarray(rpost), N_real)
cl_gen = transitivity(np.asarray(gpre), np.asarray(gpost), N_gen)
log(f"transitivity (wedge-sampled): real={cl_real:.4f} gen_v4={cl_gen:.4f}")

json.dump({"ks_out": ks_out, "ks_in": ks_in, "recip_real": rec_real, "recip_v4": rec_gen,
           "cx_b_gen": bg, "cx_r2_gen": r2g, "clust_real": cl_real, "clust_v4": cl_gen,
           "edges_gen": int(len(gpre))},
          open(ROOT + "v4_validation.json", "w"), indent=1)
log("VALIDATION DONE")
