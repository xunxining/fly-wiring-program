#!/usr/bin/env python3
"""Stage 31: 路径积分精度 —— bump 追踪旋转驱动
驱动窗以角速度 ω 旋转 600τ (一圈), 测 bump 中心与驱动中心的环形误差
小环钉扎 -> 阶梯式跳动 -> 误差大; 大环平滑 -> 误差小
输出: /mnt/agents/output/果蝇生成器/tracking_results.json
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
assert pr["type_list"] == type_list

rng = np.random.default_rng(21)

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

def track(K, gain=1.5, noise=0.05, omega=None, T=600, dt=0.1, seed=0):
    r_ = np.random.default_rng(seed)
    th = spec_phase(K)
    n = K.shape[0]
    Kn = K / (np.abs(K).sum(axis=1, keepdims=True) + 1e-9)
    if omega is None:
        omega = 2 * np.pi / T
    r = np.zeros(n)
    errs, mras, centers, drives = [], [], [], []
    for st in range(int(T / dt)):
        t = st * dt
        thd = omega * t
        dd = np.abs(np.angle(np.exp(1j * (th - thd))))
        I = (dd < 0.5).astype(np.float32) * 0.6   # 弱驱动: 引导而非锁死
        x = gain * (Kn @ r) + I + noise * r_.standard_normal(n)
        r = np.clip(r + dt * (-r + np.tanh(np.clip(x, 0, None))), 0, 1)
        if st % int(1 / dt) == 0 and t >= 40:
            z = (r * np.exp(1j * th)).sum()
            cen = np.angle(z)
            s = r.sum()
            mra = abs(z) / s if s > 1e-8 else 0.0
            err = np.angle(np.exp(1j * (cen - thd)))
            errs.append(err); mras.append(mra)
            centers.append(cen); drives.append(thd % (2 * np.pi))
    errs = np.array(errs); mras = np.array(mras)
    ok = mras > 0.25
    rms = float(np.sqrt((errs[ok] ** 2).mean())) if ok.sum() > 20 else np.nan
    lag = float(np.mean(errs[ok])) if ok.sum() > 20 else np.nan
    return rms, lag, float(ok.mean()), np.array(centers), np.array(drives)

results = {}
traj = {}
for S in [1, 2, 4, 8, 16]:
    rr, ll, aa = [], [], []
    for rep in range(3):
        Wg, ti = gen_scaled(S)
        K, ep = eff_kernel_t(Wg, ti)
        rms, lag, alive, cen, drv = track(K, seed=rep)
        rr.append(rms); ll.append(lag); aa.append(alive)
        if rep == 0:
            traj[S] = {"c": cen[::10].tolist(), "d": drv[::10].tolist()}
        log(f"  S={S} rep{rep}: RMS误差={rms:.4f} rad, 滞后={lag:.4f}, 存活={alive:.2f}")
    results[S] = {"rms": rr, "rms_mean": float(np.nanmean(rr)), "rms_sem": float(np.nanstd(rr) / np.sqrt(3)),
                  "lag_mean": float(np.nanmean(ll)), "alive": float(np.mean(aa))}
    log(f"S={S:2d}: RMS={np.nanmean(rr):.4f}±{np.nanstd(rr)/np.sqrt(3):.4f} rad")

json.dump(results, open(M + "tracking_results.json", "w"), indent=2)
json.dump(traj, open(M + "tracking_traj.json", "w"))
log("TRACKING DONE")
