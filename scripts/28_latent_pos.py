#!/usr/bin/env python3
"""Stage 28 v2: 实例级潜位置模型 —— Poisson 权重版 + 谱热启动 + 相位扫描精修
模型: w_ij ~ Poisson(λ_ij),  λ_ij = exp( a_st * cos(θ_i - θ_j + φ_st) + b_st )
两阶段: (1) 谱嵌入给全体实例相位热启动, Adam 联合优化
        (2) 相位扫描坐标精修 (每实例在 96 格网格上取最优) + 重拟合 (a,φ,b)
评估: 留出 10% 细胞对 —— 二值 AUC + 权重相关; vs 类型级基线 λ=exp(b_st)
终检: 重生成 -> EPG 等效核 cos R²; 学出相位 vs 谱环序 Kendall tau
"""
import numpy as np
import pandas as pd
import pickle, json, torch
from scipy.optimize import minimize_scalar

ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + "data_cx/"   # 2026-10-22 修复: 原 /tmp/flydata 非持久; 数据三件套在 data_cx/
torch.manual_seed(0)

def log(*a): print(*a, flush=True)

# ---------- 数据 ----------
cells = pd.read_csv(D + "cx_cells.csv")
nt = pickle.load(open(D + "cx_nt.pkl", "rb"))
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
epg = np.where(ctype == "EPG")[0]
d7 = np.where(ctype == "Delta7")[0]
peg = np.where(ctype == "PEG")[0]
pen = np.where(np.isin(ctype, ["PEN_a(PEN1)", "PEN_b(PEN2)"]))[0]

e = np.load(D + "cx_edges.npz")
W = np.zeros((N, N), np.float32)
for p, q, w in zip(e["pre"], e["post"], e["w"]):
    if p in bidx and q in bidx:
        W[bidx[q], bidx[p]] += w * csign[bidx[p]]
Wt = torch.tensor(W)
log(f"N={N}, nnz={np.count_nonzero(W)}, syn={np.abs(W).sum():.0f}")

# ---------- 留出划分 ----------
rng = np.random.default_rng(0)
mask = torch.ones(N, N, dtype=torch.bool)
mask.fill_diagonal_(False)
perm = rng.permutation(N * N - N)
te_flat = perm[: int(0.1 * (N * N - N))]
te_mask = torch.zeros(N, N, dtype=torch.bool).flatten()
te_mask[te_flat] = True
te_mask = te_mask.reshape(N, N)
tr_mask = mask & ~te_mask

# ---------- 谱热启动相位 ----------
def spec_phase(M):
    S = np.abs(M) + np.abs(M).T
    A_ = S / (S.sum(axis=1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    return np.arctan2(evec[:, -2], evec[:, -3])

th0 = spec_phase(W)
theta = torch.tensor(th0, dtype=torch.float32, requires_grad=True)
a_par = torch.full((NT, NT), 0.5, requires_grad=True)
phi_par = torch.zeros(NT, NT, requires_grad=True)
b_par = torch.full((NT, NT), 1.0, requires_grad=True)
ci_t = torch.tensor(ci)

def lam(pp, qq):
    d = theta[pp] - theta[qq]
    return torch.exp(a_par[ci_t[pp], ci_t[qq]] * torch.cos(d + phi_par[ci_t[pp], ci_t[qq]])
                     + b_par[ci_t[pp], ci_t[qq]])

# 符号约束: 抑制性突触前类型的行, 权重目标为负 -> 对 |W| 建模, 符号由突触前递质固定
Wabs = torch.tensor(np.abs(W))
sign_pre = torch.tensor(csign)   # sign_pre[j] = 突触前 j 的符号

def nll(m):
    pp, qq = torch.nonzero(m, as_tuple=True)
    L = lam(pp, qq)
    y = Wabs[pp, qq]
    return (L - y * torch.log(L + 1e-9)).mean()

opt = torch.optim.Adam([theta, a_par, phi_par, b_par], lr=0.03)
for ep in range(4000):
    opt.zero_grad()
    loss = nll(tr_mask)
    loss.backward()
    opt.step()
    if ep % 800 == 0:
        log(f"  adam ep{ep} nll={loss.item():.4f}")

# ---------- 相位扫描精修 ----------
# 2026-10-22 修复: 原实现把 theta[i] 直接吸附到 96 格点 (linspace(0,2π,97)[:-1]),
# 造成 EPG 46 细胞仅 44 唯一相位、2 对精确并列, 人为制造对称性。
# 改为: 96 格粗扫定位 + 最佳格点 ±半格距内 minimize_scalar 连续精修 (θ 保持连续, 损失单调不降)。
log("phase sweep refinement (grid + continuous)...")
grid = torch.linspace(0, 2 * np.pi, 97)[:-1]
half = float(np.pi / 96)
pp_tr, qq_tr = torch.nonzero(tr_mask, as_tuple=True)
for rnd in range(3):
    improved = 0
    with torch.no_grad():
        for i in range(N):
            m = (pp_tr == i) | (qq_tr == i)
            if not m.any():
                continue
            p_i, q_i = pp_tr[m], qq_tr[m]
            y = Wabs[p_i, q_i]
            ci_p, ci_q = ci_t[p_i], ci_t[q_i]
            def cell_loss(gv):
                d = torch.where(p_i == i, gv - theta[q_i], theta[p_i] - gv)
                L = torch.exp(a_par[ci_p, ci_q] * torch.cos(d + phi_par[ci_p, ci_q])
                              + b_par[ci_p, ci_q])
                return float((L - y * torch.log(L + 1e-9)).sum())
            best_th, best_loss = theta[i].item(), None
            for g in grid:
                l = cell_loss(float(g))
                if best_loss is None or l < best_loss:
                    best_loss, best_th = l, g.item()
            # 连续精修: 最佳格点 ±半格距 (环绕安全: cos 周期, 存值 mod 2π)
            r = minimize_scalar(cell_loss, bounds=(best_th - half, best_th + half),
                                method="bounded", options={"xatol": 1e-7})
            if r.fun < best_loss - 1e-12:
                best_loss, best_th = float(r.fun), float(r.x) % (2 * np.pi)
            if abs(best_th - theta[i].item()) > 1e-6:
                theta[i] = best_th; improved += 1
    # 重拟合 a, phi, b
    opt2 = torch.optim.Adam([a_par, phi_par, b_par], lr=0.03)
    for ep in range(800):
        opt2.zero_grad()
        loss = nll(tr_mask)
        loss.backward(); opt2.step()
    log(f"  round {rnd}: moved={improved}, nll={loss.item():.4f}")

# ---------- 评估 ----------
def auc_rank(y, s):
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

with torch.no_grad():
    pp_te, qq_te = torch.nonzero(te_mask, as_tuple=True)
    lam_te = lam(pp_te, qq_te).numpy()
    y_te = Wabs[pp_te, qq_te].numpy()
    auc_lat = auc_rank((y_te > 0).astype(int), lam_te)
    nz = y_te > 0
    corr_lat = float(np.corrcoef(np.log1p(lam_te[nz]), np.log1p(y_te[nz]))[0, 1])
    # 类型级基线
    lam_base = torch.exp(b_par[ci_t[pp_te], ci_t[qq_te]]).numpy() * 0 + np.exp(
        np.log(Wabs[tr_mask].mean().item()))
    # 正确的基线: 每类型对训练集均值率
    lam_base = np.zeros(len(pp_te))
    for s in range(NT):
        for t in range(NT):
            m_te = (ci[pp_te.numpy()] == s) & (ci[qq_te.numpy()] == t)
            m_tr = (ci[pp_tr.numpy()] == s) & (ci[qq_tr.numpy()] == t)
            if m_te.any():
                lam_base[m_te] = Wabs[pp_tr[m_tr], qq_tr[m_tr]].mean().item() if m_tr.any() else 0.0
    auc_type = auc_rank((y_te > 0).astype(int), lam_base)
    corr_type = float(np.corrcoef(np.log1p(lam_base[nz] + 1e-6), np.log1p(y_te[nz]))[0, 1])
log(f"留出: 潜位置 AUC={auc_lat:.4f}, 权重相关={corr_lat:.4f} | 类型级 AUC={auc_type:.4f}, 相关={corr_type:.4f}")

# ---------- 学出相位 vs 谱环序 ----------
th_np = theta.detach().numpy() % (2 * np.pi)
def eff_kernel(Wm):
    K = Wm[np.ix_(epg, epg)].copy()
    for mid in [d7, peg, pen]:
        K += Wm[np.ix_(epg, mid)] @ Wm[np.ix_(mid, epg)]
    return K
K_real = eff_kernel(W)
# 直接在 EPG 核上求序
th_spec = spec_phase(K_real)
th_learn_epg = th_np[epg]
from scipy.stats import kendalltau
rank_l = np.argsort(np.argsort(th_learn_epg)); rank_s = np.argsort(np.argsort(th_spec))
tau = kendalltau(rank_l, rank_s).statistic
log(f"EPG 相位 vs 谱环序 Kendall tau = {tau:.3f}")

# ---------- 重生成 -> cos 核 ----------
with torch.no_grad():
    pp_all, qq_all = torch.nonzero(mask, as_tuple=True)
    L_all = lam(pp_all, qq_all).numpy()
    P_rate = np.zeros((N, N)); P_rate[pp_all.numpy(), qq_all.numpy()] = L_all
rng2 = np.random.default_rng(1)
Wg = np.zeros((N, N), np.float32)
sam = rng2.poisson(np.clip(P_rate, 0, 1e4))
Wg = sam.astype(np.float32) * csign[None, :]
def cos_fit(K, th):
    dm = np.angle(np.exp(1j * (th[:, None] - th[None, :])))
    ks = K.ravel(); ds = dm.ravel(); m = np.abs(ks) > 0
    A_ = np.stack([np.ones(m.sum()), np.cos(ds[m])], axis=1)
    coef, *_ = np.linalg.lstsq(A_, ks[m], rcond=None)
    pred = A_ @ coef
    return float(coef[1]), float(1 - np.var(ks[m] - pred) / (np.var(ks[m]) + 1e-12))
b_g, r2_g = cos_fit(eff_kernel(Wg), spec_phase(eff_kernel(Wg)))
b_r, r2_r = cos_fit(K_real, th_spec)
log(f"cos核: real R²={r2_r:.3f} (b={b_r:.0f}) | 潜位置重生成 R²={r2_g:.3f} (b={b_g:.0f})")

json.dump({"auc_latent": auc_lat, "corr_latent": corr_lat,
           "auc_type": auc_type, "corr_type": corr_type, "tau": float(tau),
           "r2_real": r2_r, "r2_gen": r2_g, "b_real": b_r, "b_gen": b_g,
           "theta_fix": "continuous-refine 2026-10-22"},
          open(ROOT + "latent_results.json", "w"), indent=2)
np.save(ROOT + "theta_learned.npy", th_np)
pickle.dump({"a": a_par.detach().numpy(), "phi": phi_par.detach().numpy(),
             "b": b_par.detach().numpy(), "type_list": type_list},
            open(ROOT + "latent_params.pkl", "wb"))
log("LATENT DONE")
