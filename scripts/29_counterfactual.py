#!/usr/bin/env python3
"""Stage 29: 反事实实验 —— "脑为什么变大?"
用学出的潜位置程序生成 S∈{1,2,4,8,16} 倍大的 CX 航向环路:
  实例数 ×S (相位从学出池重采样+抖动), 类型对参数不变, λ/S 度数守恒
测量: 等效 EPG 核 bump 动力学 —— 无噪 MRA + 噪声鲁棒性曲线
输出: /mnt/agents/output/果蝇生成器/counterfactual_results.json
"""
import numpy as np
import pandas as pd
import pickle, json, torch

M = "/mnt/agents/output/果蝇生成器/"
torch.manual_seed(0)

def log(*a): print(*a, flush=True)

# ---------- 数据与重拟合 ----------
cells = pd.read_csv(M + "data_cx/cx_cells.csv")
nt = pickle.load(open(M + "data_cx/cx_nt.pkl", "rb"))
cells["nt"] = cells["bodyId"].map(nt)
NT_SIGN = {"acetylcholine": 1.0, "glutamate": -1.0, "gaba": -1.0}
N = len(cells)
bidx = {b: i for i, b in enumerate(cells["bodyId"])}
ctype = cells["type"].to_numpy()
type_list = sorted(set(ctype))
t2i = {t: i for i, t in enumerate(type_list)}
ci = np.array([t2i[t] for t in ctype])
NT = len(type_list)
csign = cells["nt"].map(NT_SIGN).to_numpy()
epg_t = t2i["EPG"]; d7_t = t2i["Delta7"]; peg_t = t2i["PEG"]
pen_ts = {t2i["PEN_a(PEN1)"], t2i["PEN_b(PEN2)"]}

e = np.load(M + "data_cx/cx_edges.npz")
W = np.zeros((N, N), np.float32)
for p, q, w in zip(e["pre"], e["post"], e["w"]):
    if p in bidx and q in bidx:
        W[bidx[q], bidx[p]] += w * csign[bidx[p]]
Wabs = torch.tensor(np.abs(W))

def spec_phase(Mm):
    S = np.abs(Mm) + np.abs(Mm).T
    A_ = S / (S.sum(axis=1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    return np.arctan2(evec[:, -2], evec[:, -3])

th0 = spec_phase(W)
theta = torch.tensor(th0, dtype=torch.float32, requires_grad=True)
a_par = torch.full((NT, NT), 0.5, requires_grad=True)
phi_par = torch.zeros(NT, NT, requires_grad=True)
b_par = torch.full((NT, NT), 1.0, requires_grad=True)
ci_t = torch.tensor(ci)
mask = torch.ones(N, N, dtype=torch.bool); mask.fill_diagonal_(False)

def lam(pp, qq):
    d = theta[pp] - theta[qq]
    return torch.exp(a_par[ci_t[pp], ci_t[qq]] * torch.cos(d + phi_par[ci_t[pp], ci_t[qq]])
                     + b_par[ci_t[pp], ci_t[qq]])

def nll():
    pp, qq = torch.nonzero(mask, as_tuple=True)
    L = lam(pp, qq); y = Wabs[pp, qq]
    return (L - y * torch.log(L + 1e-9)).mean()

opt = torch.optim.Adam([theta, a_par, phi_par, b_par], lr=0.03)
for ep in range(4000):
    opt.zero_grad(); loss = nll(); loss.backward(); opt.step()
log(f"refit nll={loss.item():.4f}")
# 相位扫描精修 2 轮
# 2026-10-22 修复: 格点吸附 -> 粗扫+连续精修; 且不再覆写共享产物 theta_learned.npy (见下)
from scipy.optimize import minimize_scalar
grid = torch.linspace(0, 2 * np.pi, 97)[:-1]
half = float(np.pi / 96)
pp_m, qq_m = torch.nonzero(mask, as_tuple=True)
for rnd in range(2):
    with torch.no_grad():
        for i in range(N):
            mm = (pp_m == i) | (qq_m == i)
            if not mm.any(): continue
            p_i, q_i = pp_m[mm], qq_m[mm]; y = Wabs[p_i, q_i]
            ci_p, ci_q = ci_t[p_i], ci_t[q_i]
            def cell_loss(gv):
                d = torch.where(p_i == i, gv - theta[q_i], theta[p_i] - gv)
                L = torch.exp(a_par[ci_p, ci_q] * torch.cos(d + phi_par[ci_p, ci_q]) + b_par[ci_p, ci_q])
                return float((L - y * torch.log(L + 1e-9)).sum())
            best_th, best_loss = theta[i].item(), None
            for g in grid:
                l = cell_loss(float(g))
                if best_loss is None or l < best_loss: best_loss, best_th = l, g.item()
            r = minimize_scalar(cell_loss, bounds=(best_th - half, best_th + half),
                                method="bounded", options={"xatol": 1e-7})
            if r.fun < best_loss - 1e-12:
                best_loss, best_th = float(r.fun), float(r.x) % (2 * np.pi)
            theta[i] = best_th
    opt2 = torch.optim.Adam([a_par, phi_par, b_par], lr=0.03)
    for ep in range(800):
        opt2.zero_grad(); loss = nll(); loss.backward(); opt2.step()
log(f"refined nll={loss.item():.4f}")

th_np = theta.detach().numpy() % (2 * np.pi)
a_np, phi_np, b_np = a_par.detach().numpy(), phi_par.detach().numpy(), b_par.detach().numpy()
np.save(M + "theta_learned_cf29.npy", th_np)  # 不覆写 28 的共享产物

# ---------- 缩放生成 ----------
rng = np.random.default_rng(7)
def gen_scaled(S):
    """实例 ×S: 每类型学出相位池重采样+抖动; λ/S 度数守恒"""
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
    sam = rng.poisson(np.clip(L, 0, 1e4)).astype(np.float32)
    Wg = sam * sg[None, :]
    return Wg, ti

def eff_kernel_t(Wm, ti):
    ep = np.where(ti == epg_t)[0]
    K = Wm[np.ix_(ep, ep)].copy()
    for mt in [d7_t, peg_t]:
        mid = np.where(ti == mt)[0]
        K += Wm[np.ix_(ep, mid)] @ Wm[np.ix_(mid, ep)]
    mid = np.where(np.isin(ti, list(pen_ts)))[0]
    K += Wm[np.ix_(ep, mid)] @ Wm[np.ix_(mid, ep)]
    return K, ep

def bump_dyn(K, gain=2.0, noise=0.0, T=150, dt=0.05, seed=0):
    r_ = np.random.default_rng(seed)
    th = spec_phase(K)
    Kn = K / (np.abs(K).sum(axis=1, keepdims=True) + 1e-9)
    n = K.shape[0]
    r = np.zeros(n); mras = []
    for st in range(int(T / dt)):
        t = st * dt
        I = np.zeros(n)
        if t < 20:
            I[np.abs(np.angle(np.exp(1j * th))) < 0.5] = 1.0
        x = gain * (Kn @ r) + I + noise * r_.standard_normal(n)
        r = np.clip(r + dt * (-r + np.tanh(np.clip(x, 0, None))), 0, 1)
        if st % int(1 / dt) == 0:
            s = r.sum()
            mras.append(abs((r * np.exp(1j * th)).sum()) / s if s > 1e-8 else 0.0)
    return float(np.mean(mras[40:]))

# ---------- 主循环: 每 S 多个网络实现, 每实现先调谐增益再测噪声 ----------
results = {}
noises = [0.0, 0.05, 0.1, 0.2, 0.4]
gains = [1.0, 1.5, 2.0, 3.0]
for S in [1, 2, 4, 8, 16]:
    reps = []
    for rep in range(3):
        Wg, ti = gen_scaled(S)
        K, ep = eff_kernel_t(Wg, ti)
        # 调谐: 无噪下选最优增益 (模拟生物稳态调谐)
        best_g, best_m = None, -1
        for g in gains:
            m = bump_dyn(K, gain=g, noise=0.0, seed=100 + rep)
            if m > best_m: best_m, best_g = m, g
        row = {}
        for nz in noises:
            row[nz] = np.mean([bump_dyn(K, gain=best_g, noise=nz, seed=s) for s in range(3)])
        reps.append((best_g, best_m, row))
        log(f"  S={S} rep{rep}: best_gain={best_g} clean_MRA={best_m:.3f}")
    agg = {nz: (float(np.mean([r[2][nz] for r in reps])),
                float(np.std([r[2][nz] for r in reps]) / np.sqrt(len(reps)))) for nz in noises}
    results[S] = {"best_gains": [r[0] for r in reps],
                  "clean": [round(r[1], 3) for r in reps],
                  **{f"nz{nz}": [round(v[0], 3), round(v[1], 3)] for nz, v in agg.items()}}
    log(f"S={S:2d}: " + " ".join(f"σ={nz}:{agg[nz][0]:.3f}±{agg[nz][1]:.3f}" for nz in noises))

json.dump(results, open(M + "counterfactual_results.json", "w"), indent=2)
log("COUNTERFACTUAL DONE")
