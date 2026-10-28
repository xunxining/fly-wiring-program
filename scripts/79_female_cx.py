#!/usr/bin/env python3
"""79_female_cx.py — 雌性 CX 潜位置模型 (28_latent_pos 复刻)
CX 类型: EPG, EPGt, PEN_a/PEN1, PEN_b/PEN2, PEG, Delta7 (命名对应雄性)
模型: w_ij ~ Poisson(exp(a_st cos(θ_i-θ_j+φ_st)+b_st)), 符号由突触前递质固定
评估: 10% 留出 AUC/权重相关; 与雄性 (a,φ) 矩阵对比; EPG 角序 vs PB 几何
输出: data_female/cx_latent_f.npz + cx_latent_f.json
"""
import numpy as np, pandas as pd, json, time, pickle
import torch
from scipy.special import i0, i1

D = "/mnt/agents/output/果蝇生成器/data_female/"
ROOT = "/mnt/agents/output/果蝇生成器/"
torch.manual_seed(0)
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)

CX_TYPES = ["EPG", "EPGt", "PEN_a/PEN1", "PEN_b/PEN2", "PEG", "Delta7"]

types = np.load(D + "type_names.npy", allow_pickle=True)
nt_arr = np.fromfile(D + "node_type.i32", dtype=np.int32).astype(np.int64)
node_ids = np.fromfile(D + "node_ids.i64", dtype=np.int64)
T = len(types)
tn = {str(x): i for i, x in enumerate(types)}
cx_tidx = np.array([tn[t] for t in CX_TYPES])
_loc = {int(g): i for i, g in enumerate(cx_tidx)}
cx_mask = np.isin(nt_arr, cx_tidx)
cx_nodes = np.flatnonzero(cx_mask)
N = len(cx_nodes)
g2l = np.full(len(nt_arr), -1, np.int64)
g2l[cx_nodes] = np.arange(N)
ctype = nt_arr[cx_nodes]
ci = np.array([_loc[int(x)] for x in ctype])     # local type index 0..5
log("CX cells:", N, {t: int((ctype == tn[t]).sum()) for t in CX_TYPES})

# NT signs
nr = pd.read_csv(D + "neurons.csv.gz", usecols=["root_id", "nt_type"])
ntmap = dict(zip(nr.root_id, nr.nt_type))
NT_SIGN = {"ACH": 1.0, "GLUT": -1.0, "GABA": -1.0}
csign = np.array([NT_SIGN.get(str(ntmap.get(int(node_ids[g]), "")).upper(), 1.0) for g in cx_nodes])
log("NT sign dist:", dict(zip(*np.unique(csign, return_counts=True))))

# CX internal edges
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
ew = np.fromfile(D + "e_w.i32", dtype=np.int32)
m = cx_mask[ep] & cx_mask[eq]
lp_, lq_ = g2l[ep[m]], g2l[eq[m]]
W = np.zeros((N, N), np.float32)
np.add.at(W, (lq_, lp_), ew[m].astype(np.float32) * csign[lp_])   # W[post, pre]
Wt = torch.tensor(W)
log(f"nnz={np.count_nonzero(W)}, |W|sum={np.abs(W).sum():.0f}")

# holdout
rng = np.random.default_rng(0)
mask = torch.ones(N, N, dtype=torch.bool); mask.fill_diagonal_(False)
perm = rng.permutation(N * N)
te_mask = torch.zeros(N * N, dtype=torch.bool); te_mask[perm[: int(0.1 * (N * N))]] = True
te_mask = te_mask.reshape(N, N) & mask
tr_mask = mask & ~te_mask

def spec_phase(M):
    S = np.abs(M) + np.abs(M).T
    A_ = S / (S.sum(1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    return np.arctan2(evec[:, -2], evec[:, -3])

th0 = spec_phase(W)
theta = torch.tensor(th0, dtype=torch.float32, requires_grad=True)
NT = len(CX_TYPES)
a_par = torch.full((NT, NT), 0.5, requires_grad=True)
phi_par = torch.zeros(NT, NT, requires_grad=True)
b_par = torch.full((NT, NT), 1.0, requires_grad=True)
ci_t = torch.tensor(ci)

def lam(pp, qq):
    d = theta[pp] - theta[qq]
    return torch.exp(a_par[ci_t[pp], ci_t[qq]] * torch.cos(d + phi_par[ci_t[pp], ci_t[qq]])
                     + b_par[ci_t[pp], ci_t[qq]])

Wabs = torch.tensor(np.abs(W))
def nll(m):
    pp, qq = torch.nonzero(m, as_tuple=True)
    L = lam(pp, qq)
    y = Wabs[pp, qq]
    return (L - y * torch.log(L + 1e-9)).mean()

opt = torch.optim.Adam([theta, a_par, phi_par, b_par], lr=0.03)
for epi in range(4000):
    opt.zero_grad(); loss = nll(tr_mask); loss.backward(); opt.step()
    if epi % 1000 == 0: log(f"  adam {epi} nll={loss.item():.4f}")

# phase sweep
# 2026-10-22 修复: 原实现把 theta[i] 吸附到 96 格点, 造成相位精确并列; 改格点粗扫+连续精修
log("phase sweep (grid + continuous)...")
from scipy.optimize import minimize_scalar
grid = torch.linspace(0, 2 * np.pi, 97)[:-1]
half = float(np.pi / 96)
pp_tr, qq_tr = torch.nonzero(tr_mask, as_tuple=True)
for rnd in range(3):
    improved = 0
    with torch.no_grad():
        for i in range(N):
            mm = (pp_tr == i) | (qq_tr == i)
            if not mm.any(): continue
            p_i, q_i = pp_tr[mm], qq_tr[mm]
            y = Wabs[p_i, q_i]
            ci_p, ci_q = ci_t[p_i], ci_t[q_i]
            def cell_loss(gv):
                d = torch.where(p_i == i, gv - theta[q_i], theta[p_i] - gv)
                L = torch.exp(a_par[ci_p, ci_q] * torch.cos(d + phi_par[ci_p, ci_q])
                              + b_par[ci_p, ci_q])
                return float((L - y * torch.log(L + 1e-9)).sum())
            best_th, best_loss = theta[i].item(), None
            for g_ in grid:
                l = cell_loss(float(g_))
                if best_loss is None or l < best_loss:
                    best_loss, best_th = l, g_.item()
            r = minimize_scalar(cell_loss, bounds=(best_th - half, best_th + half),
                                method="bounded", options={"xatol": 1e-7})
            if r.fun < best_loss - 1e-12:
                best_loss, best_th = float(r.fun), float(r.x) % (2 * np.pi)
            if abs(best_th - theta[i].item()) > 1e-6:
                theta[i] = best_th; improved += 1
    opt2 = torch.optim.Adam([a_par, phi_par, b_par], lr=0.03)
    for epi in range(800):
        opt2.zero_grad(); loss = nll(tr_mask); loss.backward(); opt2.step()
    log(f"  sweep {rnd}: improved {improved}, nll={loss.item():.4f}")

# held-out eval
with torch.no_grad():
    pp, qq = torch.nonzero(te_mask, as_tuple=True)
    L = lam(pp, qq).numpy()
    y = Wabs[pp, qq].numpy()
    yb = (y > 0).astype(int)
    from scipy.stats import rankdata
    # AUC via rank
    r = rankdata(L)
    n1 = yb.sum(); n0 = len(yb) - n1
    auc = (r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    wcorr = float(np.corrcoef(np.log1p(L), np.log1p(y))[0, 1])
    # type-level baseline
    mu = np.zeros((NT, NT))
    tr_pp, tr_qq = torch.nonzero(tr_mask, as_tuple=True)
    ytr = Wabs[tr_pp, tr_qq].numpy()
    cit = ci
    for s_ in range(NT):
        for t_ in range(NT):
            mm = (cit[tr_pp.numpy()] == s_) & (cit[tr_qq.numpy()] == t_)
            mu[s_, t_] = ytr[mm].mean() if mm.any() else ytr.mean()
    Lb = mu[cit[pp.numpy()], cit[qq.numpy()]]
    rb = rankdata(Lb)
    auc_b = (rb[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
log(f"heldout AUC={auc:.4f} (type baseline {auc_b:.4f}), wcorr={wcorr:.4f}")

th_np = theta.detach().numpy() % (2 * np.pi)
res = {"N": int(N), "auc_latent": float(auc), "auc_type_baseline": float(auc_b),
       "wcorr": wcorr,
       "a": a_par.detach().numpy().round(3).tolist(),
       "phi": phi_par.detach().numpy().round(3).tolist(),
       "types": CX_TYPES}

# male comparison
try:
    lp = pickle.load(open(ROOT + "latent_params.pkl", "rb"))
    res["male_types"] = list(lp["type_list"])
    res["male_a"] = np.asarray(lp["a"]).round(3).tolist()
    # aligned comparison on shared types
    mtypes = [str(x) for x in lp["type_list"]]
    fmap = {"PEN_a/PEN1": "PEN_a(PEN1)", "PEN_b/PEN2": "PEN_b(PEN2)"}
    shared = [(i, mtypes.index(fmap.get(t, t))) for i, t in enumerate(CX_TYPES)
              if fmap.get(t, t) in mtypes]
    fa = a_par.detach().numpy(); ma = np.asarray(lp["a"])
    fp = phi_par.detach().numpy(); mp = np.asarray(lp["phi"])
    fi = [s[0] for s in shared]; mi = [s[1] for s in shared]
    Fa = fa[np.ix_(fi, fi)].ravel(); Ma = ma[np.ix_(mi, mi)].ravel()
    res["a_cross_sex_corr"] = float(np.corrcoef(Fa, Ma)[0, 1])
    res["shared_types"] = [CX_TYPES[i] for i in fi]
    # EPG ring order: female theta of EPG cells sorted; amplitude of EPG->EPG
    res["EPG_a_female"] = float(fa[0, 0]); res["EPG_a_male"] = float(ma[mi[0], mi[0]]) if 0 in fi else None
    log(f"cross-sex a corr: {res['a_cross_sex_corr']:.3f} over {len(shared)} shared types")
except Exception as e:
    log("male compare failed:", e)

np.savez(D + "cx_latent_f.npz", theta=th_np, ci=ci,
         a=a_par.detach().numpy(), phi=phi_par.detach().numpy(), b=b_par.detach().numpy(),
         node_ids=node_ids[cx_nodes])
json.dump(res, open(D + "cx_latent_f.json", "w"), indent=1)
log("CX LATENT DONE")
