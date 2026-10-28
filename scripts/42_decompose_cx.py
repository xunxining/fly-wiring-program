#!/usr/bin/env python3
"""Stage 42: CX 等效核 R² 损失分解 (0.738 -> 0.503)
梯子设计, 每级只改一个因素, 5 种子取 mean±sd:
  真实参考: signed/count/bin 三种矩阵 × masked(§12式)/full(39式) 两种 cos 拟合
  L0: §12 原样重生成 Poisson(λ)·sign  (复现 0.738)
  L1: Poisson(λ) 无符号计数          (隔离: 符号/权重)
  C0: v4.1 盒原样 (a[pre,post], φ 原样, 供体重采样+jitter0.1)   (复现 ~0.503)
  C1: 修正索引 a[post,pre], φ->-φ     (隔离: 索引 bug)
  C2: C0 + 精确θ (无重采样无jitter)   (隔离: 实例化噪声)
  C3: C1 + 精确θ                      (盒子上限)
  C4: C1 + 仅jitter0.1
  C5: C1 + 仅供体重采样
  U0: 全均匀                           (无坐标地板)
评估: eff_kernel(EPG, 中继 Delta7/PEG/PEN_a/PEN_b), 自环序, masked+full 双口径
"""
import numpy as np, pickle, json, pandas as pd
from scipy.special import i0 as besseli0

ROOT = "/mnt/agents/output/果蝇生成器/"
NS = 5  # 种子数

cells = pd.read_csv(ROOT + "data_cx/cx_cells.csv")
ntmap = pickle.load(open(ROOT + "data_cx/cx_nt.pkl", "rb"))
cells["nt"] = cells["bodyId"].map(ntmap)
NT_SIGN = {"acetylcholine": 1.0, "glutamate": -1.0, "gaba": -1.0}
N = len(cells)
bidx = {b: i for i, b in enumerate(cells["bodyId"])}
ctype = cells["type"].to_numpy()
type_list = list(pickle.load(open(ROOT + "latent_params.pkl", "rb"))["type_list"])
t2i = {t: i for i, t in enumerate(type_list)}
ci = np.array([t2i[t] for t in ctype])
csign = cells["nt"].map(NT_SIGN).to_numpy()
epg = np.where(ctype == "EPG")[0]
mids = [np.where(ctype == x)[0] for x in ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]]

e = np.load(ROOT + "data_cx/cx_edges.npz")
Ws = np.zeros((N, N), np.float32)   # signed
Wc = np.zeros((N, N), np.float32)   # unsigned synapse counts
for p, q, w in zip(e["pre"], e["post"], e["w"]):
    if p in bidx and q in bidx:
        Ws[bidx[q], bidx[p]] += w * csign[bidx[p]]
        Wc[bidx[q], bidx[p]] += w

lp = pickle.load(open(ROOT + "latent_params.pkl", "rb"))
a_par, phi_par, b_par = lp["a"], lp["phi"], lp["b"]
th_exact = np.load(ROOT + "theta_learned.npy")

# v4 盒所需: 基因组边数 + 类型细胞数
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
uk, cnt = g["uk"].astype(np.int64), g["cnt"].astype(np.int64)
tcnt = g["type_count"].astype(np.int64); T = len(tcnt)
tn = [str(x) for x in np.load(ROOT + "data_census/type_names.npy", allow_pickle=True)]
cxi = [tn.index(x) for x in type_list]
cmap = {(int(uk[k] // T), int(uk[k] % T)): int(cnt[k]) for k in range(len(uk))}
npair = np.array([[cmap.get((cxi[s], cxi[t]), 0) for t in range(6)] for s in range(6)])
ncell = np.array([int(tcnt[i]) for i in cxi])
th_pool = {x: th_exact[(ctype == x)] for x in type_list}

# ---------- 评估器 ----------
def eff_kernel(Wm):
    K = Wm[np.ix_(epg, epg)].copy()
    for mid in mids:
        K += Wm[np.ix_(epg, mid)] @ Wm[mid][:, epg]
    return K

def ring_order(K):
    S_ = np.abs(K) + np.abs(K).T
    A_ = S_ / (S_.sum(axis=1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    return np.arctan2(evec[:, -2], evec[:, -3])

def cos_fit_masked(K, th):
    dm = np.angle(np.exp(1j * (th[:, None] - th[None, :])))
    ks, ds = K.ravel(), dm.ravel()
    m = np.abs(ks) > 0
    A_ = np.stack([np.ones(m.sum()), np.cos(ds[m])], axis=1)
    coef, *_ = np.linalg.lstsq(A_, ks[m], rcond=None)
    pred = A_ @ coef
    return float(1 - np.var(ks[m] - pred) / (np.var(ks[m]) + 1e-12))

def cos_fit_full(K, th):
    d = th[:, None] - th[None, :]
    X = np.stack([np.ones_like(d), np.cos(d)], axis=-1).reshape(-1, 2)
    y = K.ravel()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))

def score(Wm):
    K = eff_kernel(Wm)
    th = ring_order(K)
    return cos_fit_masked(K, th), cos_fit_full(K, th)

# ---------- 采样器 ----------
def lam_matrix(th, A, P, B):
    d = th[:, None] - th[None, :]           # d[i,j] = θ_i - θ_j, i=post, j=pre (§12 口径)
    return np.exp(A[ci][:, ci] * np.cos(d + P[ci][:, ci]) + B[ci][:, ci])

def sample_L(seed, signed):
    rng = np.random.default_rng(seed)
    Wg = rng.poisson(np.clip(lam_matrix(th_exact, a_par, phi_par, b_par), 0, 1e4)).astype(np.float32)
    return Wg * csign[None, :] if signed else Wg

def kernel_sample(rng, thp, thq, a, phi, e_, accept_fix=True):
    """accept_fix=True: 任意符号 a 的正确接受率 exp(a·cos - |a|);
       accept_fix=False: 复现 v4.1 的 exp(a(cos-1)) (a<0 时退化为均匀)"""
    amax = abs(a) if accept_fix else a
    acc = max(0.05, float(besseli0(abs(a)) / np.exp(abs(a))))
    thp32 = np.asarray(thp, dtype=np.float32); thq32 = np.asarray(thq, dtype=np.float32)
    gp, gq, need = [], [], int(e_)
    while need > 0:
        M = min(int(need / acc * 1.25) + 64, 3_000_000)
        ii = rng.integers(0, len(thp32), M); jj = rng.integers(0, len(thq32), M)
        wij = np.exp(np.float32(a) * np.cos(thp32[ii] - thq32[jj] + np.float32(phi)) - np.float32(amax))
        keep = rng.random(M, dtype=np.float32) < wij
        gp.append(ii[keep]); gq.append(jj[keep]); need -= int(keep.sum())
    return np.concatenate(gp)[:e_], np.concatenate(gq)[:e_]

def sample_cartridge(seed, fix_index, resample, jitter, accept_fix=True):
    """v4 盒模拟. fix_index: 修正 a/φ 索引; accept_fix: 修正负 a 接受率"""
    rng = np.random.default_rng(1000 + seed)
    ths = {}
    for x in type_list:
        pool = th_pool[x]; n = len(pool)
        if resample:
            dd = rng.integers(0, n, n)
            ths[x] = pool[dd] + rng.normal(0, jitter, n)
        else:
            ths[x] = pool + rng.normal(0, jitter, n)
    W = np.zeros((N, N), np.float32)
    # 新实例按类型连续编号 (与 v4 offsets 一致), 局部索引->全局 152 行序
    off = {}
    o = 0
    for x in type_list:
        off[x] = o; o += len(ths[x])
    for xs in range(6):
        for xt in range(6):
            e_ = rng.poisson(npair[xs, xt])
            if e_ == 0: continue
            sn, tn_ = type_list[xs], type_list[xt]
            if fix_index:
                a = float(a_par[xt, xs]); phi = -float(phi_par[xt, xs])
            else:
                a = float(a_par[xs, xt]); phi = float(phi_par[xs, xt])
            if abs(a) <= 0.05:
                pi = rng.integers(0, ncell[xs], e_); qi = rng.integers(0, ncell[xt], e_)
            else:
                pi, qi = kernel_sample(rng, ths[sn], ths[tn_], a, phi, e_, accept_fix)
            # W[post, pre]: pre 是 s 型, post 是 t 型
            np.add.at(W, (off[tn_] + qi, off[sn] + pi), 1.0)
    # v4 生成的类型顺序是 type_list 序, 而评估用 cx_cells 行序 -> 重排
    order = []
    pos = {x: 0 for x in type_list}
    newrow = []
    for t in ctype:  # cx_cells 行序里每个细胞在新 W 中的行号
        newrow.append(off[t] + pos[t]); pos[t] += 1
    return W[np.ix_(newrow, newrow)]

def sample_uniform(seed):
    rng = np.random.default_rng(2000 + seed)
    W = np.zeros((N, N), np.float32)
    off = {}; o = 0
    for x in type_list: off[x] = o; o += int(ncell[t2i[x]])
    for xs in range(6):
        for xt in range(6):
            e_ = rng.poisson(npair[xs, xt])
            if e_ == 0: continue
            pi = rng.integers(0, ncell[xs], e_); qi = rng.integers(0, ncell[xt], e_)
            np.add.at(W, (off[type_list[xt]] + qi, off[type_list[xs]] + pi), 1.0)
    pos = {x: 0 for x in type_list}; newrow = []
    for t in ctype:
        newrow.append(off[t] + pos[t]); pos[t] += 1
    return W[np.ix_(newrow, newrow)]

# ---------- 跑梯子 ----------
out = {}
def run(name, fn, seeds):
    r_m, r_f = [], []
    for s in range(seeds):
        m_, f_ = score(fn(s))
        r_m.append(m_); r_f.append(f_)
    out[name] = {"masked_mean": float(np.mean(r_m)), "masked_sd": float(np.std(r_m)),
                 "full_mean": float(np.mean(r_f)), "full_sd": float(np.std(r_f))}
    print(f"{name:34s} masked={np.mean(r_m):.3f}±{np.std(r_m):.3f}  full={np.mean(r_f):.3f}±{np.std(r_f):.3f}", flush=True)

print("== 真实参考 ==")
run("real_signed", lambda s: Ws, 1)
run("real_count", lambda s: Wc, 1)
run("real_binary", lambda s: (Wc > 0).astype(np.float32), 1)
print("== 模型级 (精确θ, §12 λ) ==")
run("L0_poisson_signed(§12原样)", lambda s: sample_L(s, True), NS)
run("L1_poisson_count(去符号)", lambda s: sample_L(s, False), NS)
print("== v4 盒梯子 (bug 分解) ==")
run("C0_v4原样(转置+负a均匀)", lambda s: sample_cartridge(s, False, True, 0.1, False), NS)
run("C1_仅修正索引", lambda s: sample_cartridge(s, True, True, 0.1, False), NS)
run("C2_仅修正负a接受率", lambda s: sample_cartridge(s, False, True, 0.1, True), NS)
run("C6_双修正(=v4.2盒)", lambda s: sample_cartridge(s, True, True, 0.1, True), NS)
run("C3_双修正+精确θ(上限)", lambda s: sample_cartridge(s, True, False, 0.0, True), NS)
run("C4_双修正+仅jitter", lambda s: sample_cartridge(s, True, False, 0.1, True), NS)
run("C5_双修正+仅重采样", lambda s: sample_cartridge(s, True, True, 0.0, True), NS)
print("== 地板 ==")
run("U0_全均匀", sample_uniform, NS)

json.dump(out, open(ROOT + "cx_decompose.json", "w"), indent=1)
print("DECOMPOSE DONE -> cx_decompose.json")
