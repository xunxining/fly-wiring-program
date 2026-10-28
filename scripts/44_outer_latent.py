#!/usr/bin/env python3
"""Stage 44: 外潜空间基线 —— 守卫 §12 "cos 环编码" 结论
同一训练/留出切分 (§12 原样复现, N=152, 10% 细胞对留出), 同一 Poisson NLL,
对照模型类:
  M0 类型级           λ = exp(b_st)                          (参考, §12 AUC 0.784)
  M1 cos 环 (§12)     λ = exp(a_st cos(θi-θj+φ_st) + b_st)   (参考, §12 AUC 0.909)
  M2 2D 欧氏距离      λ = exp(-a_st·||zi-zj|| + b_st),  a_st≥0
  M3 点积 K=4         λ = exp(zi·zj + b_st)
  M4 1D 线性距离      λ = exp(-a_st·|xi-xj| + b_st),  a_st≥0   (环 vs 线)
  M5 θ 打乱 (M1 参数, 实例几何销毁)                          (几何本身值多少)
判定: M1 留出 AUC ≥ 所有替代品 -> "cos 编码"成立; 任一替代品显著更好 -> 结论需修订
"""
import numpy as np
import pandas as pd
import pickle, json, torch

ROOT = "/mnt/agents/output/果蝇生成器/"
torch.manual_seed(0)
torch.set_num_threads(4)

def log(*a): print(*a, flush=True)

# ---------- 数据 (与 §12 完全一致) ----------
cells = pd.read_csv(ROOT + "data_cx/cx_cells.csv")
ntmap = pickle.load(open(ROOT + "data_cx/cx_nt.pkl", "rb"))
cells["nt"] = cells["bodyId"].map(ntmap)
NT_SIGN = {"acetylcholine": 1.0, "glutamate": -1.0, "gaba": -1.0}
N = len(cells)
bidx = {b: i for i, b in enumerate(cells["bodyId"])}
ctype = cells["type"].to_numpy()
type_list = sorted(set(ctype))
t2i = {t: i for i, t in enumerate(type_list)}
ci = np.array([t2i[t] for t in ctype])
NT = len(type_list)
e = np.load(ROOT + "data_cx/cx_edges.npz")
W = np.zeros((N, N), np.float32)
for p, q, w in zip(e["pre"], e["post"], e["w"]):
    if p in bidx and q in bidx:
        W[bidx[q], bidx[p]] += w
Wabs = torch.tensor(np.abs(W))

# ---------- 留出划分 (§12 原样) ----------
rng = np.random.default_rng(0)
mask = torch.ones(N, N, dtype=torch.bool); mask.fill_diagonal_(False)
perm = rng.permutation(N * N - N)
te_mask = torch.zeros(N * N, dtype=torch.bool)
te_mask[perm[: int(0.1 * (N * N - N))]] = True
te_mask = te_mask.reshape(N, N)
tr_mask = mask & ~te_mask
pp_tr, qq_tr = torch.nonzero(tr_mask, as_tuple=True)
pp_te, qq_te = torch.nonzero(te_mask, as_tuple=True)
ci_t = torch.tensor(ci)

def auc_rank(y, s):
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt_ = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt_)); np.add.at(sums, inv, r)
    r = (sums / cnt_)[inv]
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def evaluate(lam_fn, name):
    with torch.no_grad():
        lam_te = lam_fn(pp_te, qq_te).numpy()
    y = Wabs[pp_te, qq_te].numpy()
    auc = auc_rank((y > 0).astype(int), lam_te)
    nz = y > 0
    corr = float(np.corrcoef(np.log1p(lam_te[nz] + 1e-9), np.log1p(y[nz]))[0, 1])
    log(f"{name:28s} held-out AUC={auc:.4f}  wcorr={corr:.4f}")
    return {"auc": auc, "corr": corr}

def fit(param_lam, params, epochs=2500, lr=0.03, name=""):
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        L = param_lam(pp_tr, qq_tr)
        y = Wabs[pp_tr, qq_tr]
        loss = (L - y * torch.log(L + 1e-9)).mean()
        loss.backward(); opt.step()
    log(f"  fit {name}: nll={loss.item():.4f}")
    return param_lam

out = {}

# ---------- M1: cos 环 (§12 已拟合参数, 直接评估) ----------
lp = pickle.load(open(ROOT + "latent_params.pkl", "rb"))
a_par = torch.tensor(lp["a"]); phi_par = torch.tensor(lp["phi"]); b_par = torch.tensor(lp["b"])
th_le = torch.tensor(np.load(ROOT + "theta_learned.npy"))
lam_M1 = lambda pp, qq: torch.exp(a_par[ci_t[pp], ci_t[qq]] *
        torch.cos(th_le[pp] - th_le[qq] + phi_par[ci_t[pp], ci_t[qq]]) + b_par[ci_t[pp], ci_t[qq]])
out["M1_cos_ring(§12)"] = evaluate(lam_M1, "M1 cos环 (§12)")

# ---------- M0: 类型级 ----------
b0 = torch.full((NT, NT), 1.0, requires_grad=True)
fit(lambda pp, qq: torch.exp(b0[ci_t[pp], ci_t[qq]]), [b0], 1500, name="M0")
out["M0_type_level"] = evaluate(lambda pp, qq: torch.exp(b0[ci_t[pp], ci_t[qq]]), "M0 类型级")

# ---------- M2: 2D 欧氏距离 ----------
# 谱热启动: 用 |W| 的前两个奇异向量
uu, ss, vv = np.linalg.svd(np.abs(W) - np.abs(W).mean(), full_matrices=False)
z2 = torch.tensor(uu[:, :2] * ss[:2] ** 0.5, dtype=torch.float32, requires_grad=True)
a2 = torch.full((NT, NT), 0.5, requires_grad=True)
b2 = torch.full((NT, NT), 1.0, requires_grad=True)
def lam_M2(pp, qq):
    d = (z2[pp] - z2[qq]).norm(dim=1)
    return torch.exp(-torch.nn.functional.softplus(a2)[ci_t[pp], ci_t[qq]] * d + b2[ci_t[pp], ci_t[qq]])
fit(lam_M2, [z2, a2, b2], 2500, name="M2")
out["M2_euclid_2d"] = evaluate(lam_M2, "M2 2D欧氏距离")

# ---------- M3: 点积 K=4 ----------
z4 = torch.tensor(uu[:, :4] * ss[:4] ** 0.5, dtype=torch.float32, requires_grad=True)
b3 = torch.full((NT, NT), 1.0, requires_grad=True)
def lam_M3(pp, qq):
    return torch.exp((z4[pp] * z4[qq]).sum(1) + b3[ci_t[pp], ci_t[qq]])
fit(lam_M3, [z4, b3], 2500, name="M3")
out["M3_dotprod_k4"] = evaluate(lam_M3, "M3 点积 K=4")

# ---------- M4: 1D 线性距离 ----------
x1 = torch.tensor(np.argsort(np.argsort(th_le.numpy())) / N * 2 - 1, dtype=torch.float32, requires_grad=True)
a4 = torch.full((NT, NT), 0.5, requires_grad=True)
b4 = torch.full((NT, NT), 1.0, requires_grad=True)
def lam_M4(pp, qq):
    d = (x1[pp] - x1[qq]).abs()
    return torch.exp(-torch.nn.functional.softplus(a4)[ci_t[pp], ci_t[qq]] * d + b4[ci_t[pp], ci_t[qq]])
fit(lam_M4, [x1, a4, b4], 2500, name="M4")
out["M4_linear_1d"] = evaluate(lam_M4, "M4 1D线性距离")

# ---------- M5: θ 打乱 (用 M1 参数, 销毁实例几何) ----------
rng5 = np.random.default_rng(7)
th_scr = th_le.clone()
for t in range(NT):
    m = ci == t
    th_scr[m] = th_le[m][torch.tensor(rng5.permutation(int(m.sum())))]
out["M5_theta_scrambled"] = evaluate(
    lambda pp, qq: torch.exp(a_par[ci_t[pp], ci_t[qq]] *
        torch.cos(th_scr[pp] - th_scr[qq] + phi_par[ci_t[pp], ci_t[qq]]) + b_par[ci_t[pp], ci_t[qq]]),
    "M5 θ打乱(几何值多少)")

json.dump(out, open(ROOT + "outer_latent.json", "w"), indent=1)
log("OUTER LATENT DONE -> outer_latent.json")
