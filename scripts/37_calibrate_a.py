#!/usr/bin/env python3
"""Stage 37: 振幅模拟校准 (间接推断)
目标: 生成数据经同一裂半测量管线后, 测得 A 应等于普查实测 A_obs
方法: a 在 bessel_inv(A_obs)*[1,1.3,1.7,2.2,3] 网格上, 用全数据相位池模拟生成 E 条边
      -> 裂半测量 A_sim (2 重复, 大矩阵 1 次), 取 |A_sim - A_obs| 最小者
输出: data_census/poskernel_cal.npz (覆盖 a 字段, 另存 a_lb 下界)
"""
import numpy as np
import time
from scipy.special import i0, i1
from scipy.sparse.linalg import svds, LinearOperator

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
DENSE_CAP = 400_000
BIG = 1200          # m 或 n 超过则减重复
RNG = np.random.default_rng(3)

def log(*a): print(*a, flush=True)

pk = np.load(OUT + "poskernel.npz")
d = np.load(OUT + "census_pairs.npz")
hit = d['hit']
A_obs_map = {(int(s), int(t)): float(a) for s, t, a in zip(d['s'][hit], d['t'][hit], d['A'][hit])}
E_map = {(int(s), int(t)): float(e) for s, t, e in zip(d['s'][hit], d['t'][hit], d['E'][hit])}

agrid = np.linspace(0, 8, 1601)
cgrid = i1(agrid) / i0(agrid)
def bessel_inv(c): return float(np.interp(min(max(c, 1e-4), 0.999), cgrid, agrid))

def top2(B, r, c, E1):
    m, n = B.shape
    if m * n <= DENSE_CAP:
        Bc = B - np.outer(r, c) / E1
        U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
        return U[:, :2], Vt[:2, :]
    def mv(x):  return B @ x - r * (c @ x) / E1
    def rmv(x): return B.T @ x - c * (r @ x) / E1
    U, S, Vt = svds(LinearOperator((m, n), matvec=mv, rmatvec=rmv), k=2)
    o = np.argsort(S)[::-1]
    return U[:, o], Vt[o, :]

def sim_A(th_p, th_q, a, phi, E, seed):
    """用给定相位按核生成 E 条边, 裂半测 A"""
    m, n = len(th_p), len(th_q)
    rng = np.random.default_rng(seed)
    acc = max(0.08, i0(a) / np.exp(a))
    got_i, got_j = [], []
    need = E
    while need > 0:
        M = int(need / acc * 1.3) + 100
        ii = rng.integers(0, m, M); jj = rng.integers(0, n, M)
        w = np.exp(a * (np.cos(th_p[ii] - th_q[jj] + phi) - 1.0))
        keep = rng.random(M) < w
        got_i.append(ii[keep]); got_j.append(jj[keep])
        need -= int(keep.sum())
    gi = np.concatenate(got_i)[:E]; gj = np.concatenate(got_j)[:E]
    h1 = rng.random(E) < 0.5
    if h1.sum() < 10 or (~h1).sum() < 10:
        return np.nan
    B = np.zeros((m, n), np.float32)
    B[gi[h1], gj[h1]] = 1.0
    r, c = B.sum(1), B.sum(0)
    U2, V2 = top2(B, r, c, B.sum())
    tp2 = np.arctan2(U2[:, 0], U2[:, 1]); tq2 = np.arctan2(V2[0, :], V2[1, :])
    return float(np.abs(np.exp(1j * (tp2[gi[~h1]] - tq2[gj[~h1]])).mean()))

MULT = [1.0, 1.3, 1.7, 2.2, 3.0]
s_arr, t_arr = pk['s'], pk['t']
a_cal = np.zeros(len(s_arr), np.float32)
a_lb = np.zeros(len(s_arr), np.float32)
t0 = time.time()
for k in range(len(s_arr)):
    s, t = int(s_arr[k]), int(t_arr[k])
    A_obs = A_obs_map[(s, t)]
    E = int(E_map[(s, t)])
    o0, o1 = pk['th_off'][k], pk['th_off'][k + 1]
    ns_, nt_ = int(pk['ns'][k]), int(pk['nt'][k])
    th = pk['theta'][o0:o1]
    th_p = th[:ns_]
    th_q = th[:ns_] if pk['is_self'][k] else th[ns_:ns_ + nt_]
    phi = float(pk['phi'][k])
    alb = bessel_inv(A_obs)
    a_lb[k] = alb
    reps = 1 if max(ns_, nt_) > BIG else 2
    best_a, best_d = alb, 1e9
    for mult in MULT:
        a_try = min(alb * mult, 8.0)
        vals = [sim_A(th_p, th_q, a_try, phi, E, seed=1000 * k + r) for r in range(reps)]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        dd = abs(np.mean(vals) - A_obs)
        if dd < best_d:
            best_d, best_a = dd, a_try
    a_cal[k] = best_a
    if (k + 1) % 400 == 0:
        log(f"{k+1}/{len(s_arr)}  {time.time()-t0:.0f}s  a_cal med={np.median(a_cal[:k+1]):.2f}")

np.savez(OUT + "poskernel_cal.npz",
         **{key: pk[key] for key in pk.files if key != 'a'},
         a=a_cal.astype(np.float32), a_lb=a_lb)
log(f"DONE a_cal: med={np.median(a_cal):.2f} p90={np.quantile(a_cal,0.9):.2f} "
    f"a_lb med={np.median(a_lb):.2f}  {time.time()-t0:.0f}s")
