#!/usr/bin/env python3
"""Stage 30: bump 角扩散 —— "脑为什么变大"的严格检验
理论 (Burak & Fiete 2012): 环吸引子位置漂移的扩散系数 D ∝ 1/N
协议: 每 S 3 个网络实现, 调谐增益=1.5, 小噪声 σ=0.1 驱动扩散,
      跟踪 bump 中心角 800τ, 环形 MSD 拟合 D
输出: /mnt/agents/output/果蝇生成器/diffusion_results.json
"""
import numpy as np
import pandas as pd
import pickle, json

M = "/mnt/agents/output/果蝇生成器/"

def log(*a): print(*a, flush=True)

cells = pd.read_csv(M + "data_cx/cx_cells.csv")
nt = pickle.load(open(M + "data_cx/cx_nt.pkl", "rb"))
cells["nt"] = cells["bodyId"].map(nt)
NT_SIGN = {"acetylcholine": 1.0, "glutamate": -1.0, "gaba": -1.0}
ctype = cells["type"].to_numpy()
csign = cells["nt"].map(NT_SIGN).to_numpy()
type_list = sorted(set(ctype))
t2i = {t: i for i, t in enumerate(type_list)}
ci = np.array([t2i[t] for t in ctype])
NT = len(type_list)
epg_t, d7_t, peg_t = t2i["EPG"], t2i["Delta7"], t2i["PEG"]
pen_ts = {t2i["PEN_a(PEN1)"], t2i["PEN_b(PEN2)"]}

th_np = np.load(M + "theta_learned.npy")
pr = pickle.load(open(M + "latent_params.pkl", "rb"))
a_np, phi_np, b_np = pr["a"], pr["phi"], pr["b"]
# latent_params 的 type_list 顺序需一致
assert pr["type_list"] == type_list, (pr["type_list"], type_list)

rng = np.random.default_rng(11)

def spec_phase(Mm):
    S = np.abs(Mm) + np.abs(Mm).T
    A_ = S / (S.sum(axis=1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    return np.arctan2(evec[:, -2], evec[:, -3])

def gen_scaled(S):
    idx = []
    for t in range(NT):
        pool = th_np[ci == t]
        if len(pool) == 0: continue
        pick = rng.integers(0, len(pool), len(pool) * S)
        phases = (pool[pick] + rng.normal(0, 0.1, len(pool) * S)) % (2 * np.pi)
        sg = csign[ci == t][0]
        for ph in phases:
            idx.append((t, ph, sg))
    n = len(idx)
    ti = np.array([x[0] for x in idx]); ph = np.array([x[1] for x in idx]); sg = np.array([x[2] for x in idx])
    d = ph[:, None] - ph[None, :]
    L = np.exp(a_np[np.ix_(ti, ti)] * np.cos(d + phi_np[np.ix_(ti, ti)]) + b_np[np.ix_(ti, ti)]) / S
    np.fill_diagonal(L, 0)
    return rng.poisson(np.clip(L, 0, 1e4)).astype(np.float32) * sg[None, :], ti

def eff_kernel_t(Wm, ti):
    ep = np.where(ti == epg_t)[0]
    K = Wm[np.ix_(ep, ep)].copy()
    for mt in [d7_t, peg_t]:
        mid = np.where(ti == mt)[0]
        K += Wm[np.ix_(ep, mid)] @ Wm[np.ix_(mid, ep)]
    mid = np.where(np.isin(ti, list(pen_ts)))[0]
    K += Wm[np.ix_(ep, mid)] @ Wm[np.ix_(mid, ep)]
    return K, ep

def measure_diffusion(K, gain=1.5, noise=0.1, T=800, dt=0.1, seed=0):
    r_ = np.random.default_rng(seed)
    th = spec_phase(K)
    Kn = K / (np.abs(K).sum(axis=1, keepdims=True) + 1e-9)
    n = K.shape[0]
    r = np.zeros(n)
    centers, mras = [], []
    for st in range(int(T / dt)):
        t = st * dt
        I = np.zeros(n)
        if t < 20:
            I[np.abs(np.angle(np.exp(1j * th))) < 0.5] = 1.0
        x = gain * (Kn @ r) + I + noise * r_.standard_normal(n)
        r = np.clip(r + dt * (-r + np.tanh(np.clip(x, 0, None))), 0, 1)
        if st % int(1 / dt) == 0 and t >= 40:
            z = (r * np.exp(1j * th)).sum()
            centers.append(np.angle(z))
            s = r.sum()
            mras.append(abs(z) / s if s > 1e-8 else 0.0)
    centers = np.unwrap(np.array(centers))
    mras = np.array(mras)
    # 环形 MSD: 多个 lag 的均方位移 (只取 bump 存活段 MRA>0.3)
    alive = mras > 0.3
    lags = np.arange(1, min(300, len(centers) // 2))
    msd = []
    for lag in lags:
        d2 = (centers[lag:] - centers[:-lag]) ** 2
        ok = alive[lag:] & alive[:-lag]
        msd.append(d2[ok].mean() if ok.sum() > 10 else np.nan)
    msd = np.array(msd)
    m = ~np.isnan(msd)
    if m.sum() < 10:
        return np.nan, float(alive.mean())
    D = np.polyfit(lags[m], msd[m], 1)[0] / 2   # MSD = 2D·lag
    return float(D), float(alive.mean())

results = {}
for S in [1, 2, 4, 8, 16]:
    Ds, alives = [], []
    for rep in range(3):
        Wg, ti = gen_scaled(S)
        K, ep = eff_kernel_t(Wg, ti)
        D, al = measure_diffusion(K, seed=rep)
        Ds.append(D); alives.append(al)
        log(f"  S={S} rep{rep}: D={D:.5f} rad²/τ, bump存活={al:.2f}")
    results[S] = {"D": Ds, "D_mean": float(np.nanmean(Ds)), "alive_mean": float(np.mean(alives))}
    log(f"S={S:2d}: D={np.nanmean(Ds):.5f}±{np.nanstd(Ds)/np.sqrt(3):.5f}, 存活={np.mean(alives):.2f}")

json.dump(results, open(M + "diffusion_results.json", "w"), indent=2)
log("DIFFUSION DONE")
