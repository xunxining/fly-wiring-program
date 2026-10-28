#!/usr/bin/env python3
"""Stage 26: 环形吸引子动力学实验 —— 实例级全环路速率模型读出 (含三组对照)
问题: EPG/PEN/PEG/Delta7 航向环路的真实布线能否维持活动 bump (无输入时持续) 并支持积分?
四组: real(真实+递质符号) / rewired(度+符号保持重连) / gentype(类型级保留,实例随机) / ideal(手工理想环)
协议: 前 20τ 给 EPG 局部输入驱动 bump, 撤输入后跑到 200τ, 测 EPG 群体向量幅值 MRA
"""
import numpy as np
import pandas as pd
import pickle, json

D = "/tmp/flydata/"
rng = np.random.default_rng(0)

def log(*a): print(*a, flush=True)

cells = pd.read_csv(D + "cx_cells.csv")
nt = pickle.load(open(D + "cx_nt.pkl", "rb"))
cells["nt"] = cells["bodyId"].map(nt)
NT_SIGN = {"acetylcholine": 1.0, "glutamate": -1.0, "gaba": -1.0}
N = len(cells)
bidx = {b: i for i, b in enumerate(cells["bodyId"])}
ctype = cells["type"].to_numpy()
csign = cells["nt"].map(NT_SIGN).to_numpy()
epg_rows = np.where(ctype == "EPG")[0]
N_EPG = len(epg_rows)
log(f"N={N}, EPG={N_EPG}")

e = np.load(D + "cx_edges.npz")
W_real = np.zeros((N, N), np.float32)   # W[post, pre]
for p, q, w in zip(e["pre"], e["post"], e["w"]):
    if p in bidx and q in bidx:
        W_real[bidx[q], bidx[p]] += w * csign[bidx[p]]
log(f"W_real nnz={np.count_nonzero(W_real)}, exc={np.count_nonzero(W_real>0)}, inh={np.count_nonzero(W_real<0)}")

# ---- 度+符号保持重连 ----
def rewire_sign(W, n_swap=300000, seed=1):
    r = np.random.default_rng(seed)
    W2 = W.copy()
    for sign in [1, -1]:
        posts, pres = np.nonzero(np.sign(W2) == sign)
        vals = W2[posts, pres].copy()
        E = len(posts)
        for _ in range(min(n_swap, E * 10)):
            i, j = r.integers(0, E, 2)
            pi_, qi_, pj, qj = pres[i], posts[i], pres[j], posts[j]
            if pi_ == pj or qi_ == qj or W2[qi_, pj] != 0 or W2[qj, pi_] != 0:
                continue
            W2[qi_, pi_] = 0; W2[qj, pj] = 0
            W2[qi_, pj] = vals[i]; W2[qj, pi_] = vals[j]
            posts[i], posts[j] = qj, qi_
    return W2

# ---- 类型级生成 (实例随机, 权重从真实池重采样) ----
def gen_typed(W, seed=2):
    r = np.random.default_rng(seed)
    W2 = np.zeros_like(W)
    tset = np.unique(ctype)
    for ts in tset:
        for tt in tset:
            srows = np.where(ctype == ts)[0]; trows = np.where(ctype == tt)[0]
            sub = W[np.ix_(trows, srows)]
            nz = np.count_nonzero(sub)
            if nz == 0:
                continue
            vals = sub[sub != 0]
            # 保持类型对边数: Poisson(nz), 实例均匀, 权重经验重采样
            ne = r.poisson(nz)
            if ne == 0:
                continue
            ii = r.integers(0, len(trows), ne); jj = r.integers(0, len(srows), ne)
            vv = vals[r.integers(0, len(vals), ne)]
            for a, b, v in zip(ii, jj, vv):
                W2[trows[a], srows[b]] += v
    return W2

# ---- 手工理想环 (经典 Amari 形式: 局域兴奋+全局抑制) ----
# 勘误 2026-10-27 (§148): 原注释"已验证可成 bump"与本臂输出矛盾——谱半径归一化下 DC 模态被压穿,
# 全部测试增益下 MRA=0/act≈0。本臂实为阴性对照, 不可称为已验证 bump。
def ideal_ring(seed=3):
    W2 = np.zeros_like(W_real)
    th = np.arange(N_EPG) / N_EPG * 2 * np.pi
    for a, i in enumerate(epg_rows):
        for b, j in enumerate(epg_rows):
            dd = np.angle(np.exp(1j * (th[a] - th[b])))
            W2[j, i] = -0.8 + 2.5 * np.cos(dd)
    return W2

arms = {"real": W_real,
        "rewired": rewire_sign(W_real),
        "gentype": gen_typed(W_real),
        "ideal": ideal_ring()}
for k, W in arms.items():
    log(f"{k}: nnz={np.count_nonzero(W)}")

# ---- EPG 环序: 由各组自身 EPG 连接相似度谱嵌入 ----
def ring_order(W):
    sub = np.abs(W[np.ix_(epg_rows, epg_rows)])
    S = sub + sub.T
    if S.sum() == 0:
        return np.arange(N_EPG) / N_EPG * 2 * np.pi
    A = S / (S.sum(axis=1, keepdims=True) + 1e-9)
    evals, evecs = np.linalg.eigh(A)
    v1, v2 = evecs[:, -2], evecs[:, -3]
    return np.arctan2(v1, v2)

# ---- 动力学 ----
def run_dyn(W, gain, T=200, dt=0.05, drive_T=20, seed=0):
    rho = max(abs(np.linalg.eigvals(W)).max(), 1e-6)
    Ws = W * (gain / rho)
    th = ring_order(W)
    r = np.zeros(N)
    hist_mra, hist_act = [], []
    steps = int(T / dt)
    for st in range(steps):
        t = st * dt
        I = np.zeros(N)
        if t < drive_T:
            # 局部驱动 theta=0 附近 ±0.5 rad 的 EPG
            dd = np.abs(np.angle(np.exp(1j * (th - 0))))
            I[epg_rows[dd < 0.5]] = 2.0
        x = Ws @ r + I
        r = np.clip(r + dt * (-r + np.tanh(np.clip(x, 0, None))), 0, 1)
        if st % int(1 / dt) == 0:
            re = r[epg_rows]
            s = re.sum()
            mra = abs((re * np.exp(1j * th)).sum()) / s if s > 1e-8 else 0.0
            hist_mra.append(mra); hist_act.append(s)
    return np.array(hist_mra), np.array(hist_act), th

results = {}
gains = [0.9, 1.1, 1.3, 1.6, 2.0]
for name, W in arms.items():
    for g in gains:
        mra, act, th = run_dyn(W, g)
        # 取撤输入后 (t>40) 的平均 MRA 与终值
        mra_late = float(mra[40:].mean())
        act_late = float(act[40:].mean())
        results[f"{name}_g{g}"] = {"mra_late": mra_late, "mra_final": float(mra[-1]),
                                   "act_late": act_late}
        log(f"{name} g={g}: MRA_late={mra_late:.3f}, MRA_final={mra[-1]:.3f}, act_late={act_late:.1f}")

json.dump(results, open(D + "ring_results.json", "w"), indent=2)
np.savez(D + "ring_W.npz", **{k: v for k, v in arms.items()})
log("RING DONE")
