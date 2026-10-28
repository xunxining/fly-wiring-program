#!/usr/bin/env python3
"""Stage 27: 环形吸引子的无参数结构检验 + 简化动力学
A) 结构检验: EPG 通过 Delta7/PEG/PEN 中继的等效 EPG->EPG 核是否呈 cos(Δθ) 形式
   (环吸引子的代数本质); real vs 5 种子重连 vs 类型级生成
B) 动力学: 在 46x46 等效核上跑 tanh 速率模型, 行归一化, 测 bump 持续性 (MRA)
"""
import numpy as np
import pandas as pd
import pickle, json

D = "/tmp/flydata/"
def log(*a): print(*a, flush=True)

cells = pd.read_csv(D + "cx_cells.csv")
nt = pickle.load(open(D + "cx_nt.pkl", "rb"))
cells["nt"] = cells["bodyId"].map(nt)
NT_SIGN = {"acetylcholine": 1.0, "glutamate": -1.0, "gaba": -1.0}
N = len(cells)
bidx = {b: i for i, b in enumerate(cells["bodyId"])}
ctype = cells["type"].to_numpy()
csign = cells["nt"].map(NT_SIGN).to_numpy()
epg = np.where(ctype == "EPG")[0]
d7 = np.where(ctype == "Delta7")[0]
peg = np.where(ctype == "PEG")[0]
pen = np.where(np.isin(ctype, ["PEN_a(PEN1)", "PEN_b(PEN2)"]))[0]
N_EPG = len(epg)

e = np.load(D + "cx_edges.npz")
W = np.zeros((N, N), np.float32)
for p, q, w in zip(e["pre"], e["post"], e["w"]):
    if p in bidx and q in bidx:
        W[bidx[q], bidx[p]] += w * csign[bidx[p]]

# ---- 环序: 由 EPG 经由中继的共享输入相似度谱嵌入 ----
def ring_order(K):
    S = np.abs(K) + np.abs(K).T
    if S.sum() == 0:
        return np.arange(N_EPG) / N_EPG * 2 * np.pi
    A = S / (S.sum(axis=1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A)
    return np.arctan2(evec[:, -2], evec[:, -3])

def eff_kernel(Wm):
    K = Wm[np.ix_(epg, epg)].copy()
    for mid in [d7, peg, pen]:
        K += Wm[np.ix_(epg, mid)] @ Wm[np.ix_(mid, epg)]
    return K

def cos_fit(K, th):
    """K[j,i] 按 Δθ=th_j-th_i 分箱, 拟合 a+b*cos(Δθ)"""
    dm = np.angle(np.exp(1j * (th[:, None] - th[None, :])))  # Δθ_ji
    ks = K.ravel(); ds = dm.ravel()
    m = np.abs(ks) > 0
    if m.sum() < 50:
        return 0.0, 0.0
    A = np.stack([np.ones(m.sum()), np.cos(ds[m])], axis=1)
    coef, *_ = np.linalg.lstsq(A, ks[m], rcond=None)
    pred = A @ coef
    r2 = 1 - np.var(ks[m] - pred) / (np.var(ks[m]) + 1e-12)
    return float(coef[1]), float(r2)   # b (cos幅值, 符号带意义), R²

def rewire_sign(Wm, seed):
    r = np.random.default_rng(seed)
    W2 = Wm.copy()
    for sign in [1, -1]:
        posts, pres = np.nonzero(np.sign(W2) == sign)
        vals = W2[posts, pres].copy()
        E = len(posts)
        for _ in range(E * 12):
            i, j = r.integers(0, E, 2)
            pi_, qi_, pj, qj = pres[i], posts[i], pres[j], posts[j]
            if pi_ == pj or qi_ == qj or W2[qi_, pj] != 0 or W2[qj, pi_] != 0:
                continue
            W2[qi_, pi_] = 0; W2[qj, pj] = 0
            W2[qi_, pj] = vals[i]; W2[qj, pi_] = vals[j]
            posts[i], posts[j] = qj, qi_
    return W2

def gen_typed(Wm, seed):
    r = np.random.default_rng(seed)
    W2 = np.zeros_like(Wm)
    for ts in np.unique(ctype):
        for tt in np.unique(ctype):
            sr = np.where(ctype == ts)[0]; tr = np.where(ctype == tt)[0]
            sub = Wm[np.ix_(tr, sr)]
            nz = np.count_nonzero(sub)
            if nz == 0:
                continue
            vals = sub[sub != 0]
            ne = r.poisson(nz)
            for _ in range(ne):
                a = r.integers(0, len(tr)); b = r.integers(0, len(sr))
                W2[tr[a], sr[b]] += vals[r.integers(0, len(vals))]
    return W2

# ---- A: 核结构检验 ----
log("=== A: EPG 等效核的 cos 拟合 ===")
K_real = eff_kernel(W)
th_real = ring_order(K_real)
b_r, r2_r = cos_fit(K_real, th_real)
log(f"real:   cos幅值 b={b_r:8.2f}, R²={r2_r:.3f}")

res = {"real": {"b": b_r, "r2": r2_r}}
bs, r2s = [], []
for s in range(5):
    Kr = eff_kernel(rewire_sign(W, seed=10 + s))
    b, r2 = cos_fit(Kr, ring_order(Kr))
    bs.append(b); r2s.append(r2)
res["rewired"] = {"b": bs, "r2": r2s}
log(f"rewired(5 seeds): b={np.mean(bs):8.2f}±{np.std(bs):.2f}, R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}")

bg, r2g = [], []
for s in range(5):
    Kg = eff_kernel(gen_typed(W, seed=20 + s))
    b, r2 = cos_fit(Kg, ring_order(Kg))
    bg.append(b); r2g.append(r2)
res["gentype"] = {"b": bg, "r2": r2g}
log(f"gentype(5 seeds): b={np.mean(bg):8.2f}±{np.std(bg):.2f}, R²={np.mean(r2g):.3f}±{np.std(r2g):.3f}")

# ---- B: 46x46 核上的 bump 动力学 ----
log("=== B: 等效核 bump 动力学 ===")
def bump_dyn(K, gain, T=150, dt=0.05):
    th = ring_order(K)
    Kn = K / (np.abs(K).sum(axis=1, keepdims=True) + 1e-9)   # 行归一化
    r = np.zeros(N_EPG)
    mras = []
    for st in range(int(T / dt)):
        t = st * dt
        I = np.zeros(N_EPG)
        if t < 20:
            I[np.abs(np.angle(np.exp(1j * th))) < 0.5] = 1.0
        x = gain * (Kn @ r) + I
        r = np.clip(r + dt * (-r + np.tanh(np.clip(x, 0, None))), 0, 1)
        if st % int(1 / dt) == 0:
            s = r.sum()
            mras.append(abs((r * np.exp(1j * th)).sum()) / s if s > 1e-8 else 0.0)
    return float(np.mean(mras[40:])), float(mras[-1])

for name, K in [("real", K_real),
                ("rewired", eff_kernel(rewire_sign(W, seed=99))),
                ("gentype", eff_kernel(gen_typed(W, seed=98)))]:
    row = {}
    for g in [2, 4, 6, 8, 12]:
        ml, mf = bump_dyn(K, g)
        row[g] = (round(ml, 3), round(mf, 3))
        log(f"  {name} gain={g}: MRA_late={ml:.3f}")
    res[name + "_dyn"] = {str(g): row[g] for g in row}

json.dump(res, open(D + "kernel_results.json", "w"), indent=2)
np.save(D + "K_real.npy", K_real)
np.save(D + "th_real.npy", th_real)
log("KERNEL DONE")
