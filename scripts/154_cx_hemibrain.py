#!/usr/bin/env python3
"""154_cx_hemibrain.py — HemiBrain CX 余弦核拟合 + 模型赛（§151 G7a，注册先于计算）
逐行移植 28_latent_pos.py（谱热启动 + Adam + 96 格相位扫描精修）与 44_outer_latent.py（M0–M5 模型赛）。
数据：data_hemibrain/hb_graph.npz 的 CX 六型子网（EPG/EPGt/PEG/PEN_a(PEN1)/PEN_b(PEN2)/Delta7，152 细胞）。
注册假设（§151 冻结）：递质按类型名 exact match 从雄性普查转移（六型在雄性均单值：
Delta7=glutamate，余五型=acetylcholine）。
主读数（规范不变量，§151 注册修正 #1）：g = a·cos(φ)，a 取 a[post=EPG, pre=Delta7]。
参照：male a=−6.1297 φ=0.3461 g=−5.7663；female a=+3.9580 φ=−0.0640 g=+3.9499；r(3×3, M–F)=0.9843。
输出：data_hemibrain/cx_latent_hb.json
"""
import json
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau

ROOT = '/mnt/agents/output/果蝇生成器/'
torch.manual_seed(0)
torch.set_num_threads(4)


def log(*a):
    print(*a, flush=True)


# ---------- 数据：hemibrain CX 子网 ----------
d = np.load(ROOT + 'data_hemibrain/hb_graph.npz', allow_pickle=True)
bodyId, ci_all = d['bodyId'], d['ci']
e_pre, e_post, e_w = d['e_pre'], d['e_post'], d['e_w']
types_all = d['types'].astype(str)
CX6 = ['EPG', 'EPGt', 'PEG', 'PEN_a(PEN1)', 'PEN_b(PEN2)', 'Delta7']
NT_BY_TYPE = {'Delta7': 'glutamate', 'EPG': 'acetylcholine', 'EPGt': 'acetylcholine',
              'PEG': 'acetylcholine', 'PEN_a(PEN1)': 'acetylcholine', 'PEN_b(PEN2)': 'acetylcholine'}
NT_SIGN = {'acetylcholine': 1.0, 'glutamate': -1.0, 'gaba': -1.0}
cx_type_ids = {t: int(np.flatnonzero(types_all == t)[0]) for t in CX6}
is_cx = np.isin(ci_all, list(cx_type_ids.values()))
cells = np.flatnonzero(is_cx)                      # hemibrain 子图内行号
g2l = {int(g): i for i, g in enumerate(cells)}     # 全局行号 -> CX 局部行号
ctype = np.array([types_all[ci_all[g]] for g in cells])
type_list = sorted(set(ctype))                     # 字典序 == 雄性 type_list 序
t2i = {t: i for i, t in enumerate(type_list)}
ci = np.array([t2i[t] for t in ctype])
NT = len(type_list)
N = len(cells)
csign = np.array([NT_SIGN[NT_BY_TYPE[t]] for t in ctype])
epg = np.where(ctype == 'EPG')[0]
d7 = np.where(ctype == 'Delta7')[0]
peg = np.where(ctype == 'PEG')[0]
pen = np.where(np.isin(ctype, ['PEN_a(PEN1)', 'PEN_b(PEN2)']))[0]
log(f'CX 子网: N={N}（EPG {len(epg)} / Delta7 {len(d7)} / PEG {len(peg)} / PEN {len(pen)} / EPGt {int((ctype=="EPGt").sum())}）')

m_cx = is_cx[e_pre] & is_cx[e_post]
W = np.zeros((N, N), np.float32)
for p, q, w in zip(e_pre[m_cx], e_post[m_cx], e_w[m_cx]):
    W[g2l[int(q)], g2l[int(p)]] += w * NT_SIGN[NT_BY_TYPE[types_all[ci_all[int(p)]]]]
Wt = torch.tensor(W)
log(f'nnz={np.count_nonzero(W)}, syn={np.abs(W).sum():.0f}')

# ---------- 留出划分（28 号逐行：seed 0，10% 细胞对） ----------
rng = np.random.default_rng(0)
mask = torch.ones(N, N, dtype=torch.bool)
mask.fill_diagonal_(False)
perm = rng.permutation(N * N - N)
te_flat = perm[: int(0.1 * (N * N - N))]
te_mask = torch.zeros(N * N, dtype=torch.bool)
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
    dd = theta[pp] - theta[qq]
    return torch.exp(a_par[ci_t[pp], ci_t[qq]] * torch.cos(dd + phi_par[ci_t[pp], ci_t[qq]])
                     + b_par[ci_t[pp], ci_t[qq]])


Wabs = torch.tensor(np.abs(W))


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
        log(f'  adam ep{ep} nll={loss.item():.4f}')

# ---------- 相位扫描精修（96 格粗扫 + 连续精修；2026-10-22 修复版逐行） ----------
log('phase sweep refinement (grid + continuous)...')
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
                dd = torch.where(p_i == i, gv - theta[q_i], theta[p_i] - gv)
                L = torch.exp(a_par[ci_p, ci_q] * torch.cos(dd + phi_par[ci_p, ci_q])
                              + b_par[ci_p, ci_q])
                return float((L - y * torch.log(L + 1e-9)).sum())

            best_th, best_loss = theta[i].item(), None
            for g in grid:
                l = cell_loss(float(g))
                if best_loss is None or l < best_loss:
                    best_loss, best_th = l, g.item()
            r = minimize_scalar(cell_loss, bounds=(best_th - half, best_th + half),
                                method='bounded', options={'xatol': 1e-7})
            if r.fun < best_loss - 1e-12:
                best_loss, best_th = float(r.fun), float(r.x) % (2 * np.pi)
            if abs(best_th - theta[i].item()) > 1e-6:
                theta[i] = best_th
                improved += 1
    opt2 = torch.optim.Adam([a_par, phi_par, b_par], lr=0.03)
    for ep in range(800):
        opt2.zero_grad()
        loss = nll(tr_mask)
        loss.backward()
        opt2.step()
    log(f'  round {rnd}: moved={improved}, nll={loss.item():.4f}')

# ---------- 评估（28 号逐行） ----------
def auc_rank(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
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
    lam_base = np.zeros(len(pp_te))
    pp_tr_np, qq_tr_np = pp_tr.numpy(), qq_tr.numpy()
    for s in range(NT):
        for t in range(NT):
            m_te = (ci[pp_te.numpy()] == s) & (ci[qq_te.numpy()] == t)
            m_tr = (ci[pp_tr_np] == s) & (ci[qq_tr_np] == t)
            if m_te.any():
                lam_base[m_te] = Wabs[pp_tr[m_tr], qq_tr[m_tr]].mean().item() if m_tr.any() else 0.0
    auc_type = auc_rank((y_te > 0).astype(int), lam_base)
    corr_type = float(np.corrcoef(np.log1p(lam_base[nz] + 1e-6), np.log1p(y_te[nz]))[0, 1])
log(f'留出: 潜位置 AUC={auc_lat:.4f}, 权重相关={corr_lat:.4f} | 类型级 AUC={auc_type:.4f}, 相关={corr_type:.4f}')

# ---------- EPG 相位 vs 谱环序 ----------
th_np = theta.detach().numpy() % (2 * np.pi)


def eff_kernel(Wm):
    K = Wm[np.ix_(epg, epg)].copy()
    for mid in [d7, peg, pen]:
        K += Wm[np.ix_(epg, mid)] @ Wm[np.ix_(mid, epg)]
    return K


K_real = eff_kernel(W)
th_spec = spec_phase(K_real)
th_learn_epg = th_np[epg]
rank_l = np.argsort(np.argsort(th_learn_epg))
rank_s = np.argsort(np.argsort(th_spec))
tau = kendalltau(rank_l, rank_s).statistic
log(f'EPG 相位 vs 谱环序 Kendall tau = {tau:.3f}')

# ---------- 模型赛（44 号 M0–M5 逐行移植） ----------
a_f = a_par.detach().clone()
phi_f = phi_par.detach().clone()
b_f = b_par.detach().clone()
th_le = theta.detach().clone()
out = {}


def evaluate(lam_fn, name):
    with torch.no_grad():
        lam_te_ = lam_fn(pp_te, qq_te).numpy()
    y = Wabs[pp_te, qq_te].numpy()
    auc = auc_rank((y > 0).astype(int), lam_te_)
    nz_ = y > 0
    corr = float(np.corrcoef(np.log1p(lam_te_[nz_] + 1e-9), np.log1p(y[nz_]))[0, 1])
    log(f'{name:28s} held-out AUC={auc:.4f}  wcorr={corr:.4f}')
    return {'auc': auc, 'corr': corr}


def fit(param_lam, params, epochs=2500, lr=0.03, name=''):
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        L = param_lam(pp_tr, qq_tr)
        y = Wabs[pp_tr, qq_tr]
        loss = (L - y * torch.log(L + 1e-9)).mean()
        loss.backward()
        opt.step()
    log(f'  fit {name}: nll={loss.item():.4f}')
    return param_lam


out['M1_cos_ring'] = evaluate(
    lambda pp, qq: torch.exp(a_f[ci_t[pp], ci_t[qq]] *
                             torch.cos(th_le[pp] - th_le[qq] + phi_f[ci_t[pp], ci_t[qq]])
                             + b_f[ci_t[pp], ci_t[qq]]), 'M1 cos环 (hemibrain fit)')
b0 = torch.full((NT, NT), 1.0, requires_grad=True)
fit(lambda pp, qq: torch.exp(b0[ci_t[pp], ci_t[qq]]), [b0], 1500, name='M0')
out['M0_type_level'] = evaluate(lambda pp, qq: torch.exp(b0[ci_t[pp], ci_t[qq]]), 'M0 类型级')
uu, ss, vv = np.linalg.svd(np.abs(W) - np.abs(W).mean(), full_matrices=False)
z2 = torch.tensor(uu[:, :2] * ss[:2] ** 0.5, dtype=torch.float32, requires_grad=True)
a2 = torch.full((NT, NT), 0.5, requires_grad=True)
b2 = torch.full((NT, NT), 1.0, requires_grad=True)


def lam_M2(pp, qq):
    dd = (z2[pp] - z2[qq]).norm(dim=1)
    return torch.exp(-torch.nn.functional.softplus(a2)[ci_t[pp], ci_t[qq]] * dd + b2[ci_t[pp], ci_t[qq]])


fit(lam_M2, [z2, a2, b2], 2500, name='M2')
out['M2_euclid_2d'] = evaluate(lam_M2, 'M2 2D欧氏距离')
z4 = torch.tensor(uu[:, :4] * ss[:4] ** 0.5, dtype=torch.float32, requires_grad=True)
b3 = torch.full((NT, NT), 1.0, requires_grad=True)
fit(lambda pp, qq: torch.exp((z4[pp] * z4[qq]).sum(1) + b3[ci_t[pp], ci_t[qq]]), [z4, b3], 2500, name='M3')
out['M3_dotprod_k4'] = evaluate(
    lambda pp, qq: torch.exp((z4[pp] * z4[qq]).sum(1) + b3[ci_t[pp], ci_t[qq]]), 'M3 点积 K=4')
x1 = torch.tensor(np.argsort(np.argsort(th_le.numpy())) / N * 2 - 1, dtype=torch.float32, requires_grad=True)
a4 = torch.full((NT, NT), 0.5, requires_grad=True)
b4 = torch.full((NT, NT), 1.0, requires_grad=True)


def lam_M4(pp, qq):
    dd = (x1[pp] - x1[qq]).abs()
    return torch.exp(-torch.nn.functional.softplus(a4)[ci_t[pp], ci_t[qq]] * dd + b4[ci_t[pp], ci_t[qq]])


fit(lam_M4, [x1, a4, b4], 2500, name='M4')
out['M4_linear_1d'] = evaluate(lam_M4, 'M4 1D线性距离')
rng5 = np.random.default_rng(7)
th_scr = th_le.clone()
for t in range(NT):
    mm = ci == t
    th_scr[mm] = th_le[mm][torch.tensor(rng5.permutation(int(mm.sum())))]
out['M5_theta_scrambled'] = evaluate(
    lambda pp, qq: torch.exp(a_f[ci_t[pp], ci_t[qq]] *
                             torch.cos(th_scr[pp] - th_scr[qq] + phi_f[ci_t[pp], ci_t[qq]])
                             + b_f[ci_t[pp], ci_t[qq]]), 'M5 θ打乱')

# ---------- G7a 主读数：规范不变量 g = a·cos(φ)，a[post=EPG, pre=Delta7] ----------
iE, iD = type_list.index('EPG'), type_list.index('Delta7')
a_ED = float(a_f[iE, iD])
phi_ED = float(phi_f[iE, iD])
g_ED = a_ED * float(np.cos(phi_ED))
log(f'G7a 主读数: a[EPG,D7]={a_ED:.4f}  phi={phi_ED:.4f}  g={g_ED:.4f}')
log(f'   参照: male g=−5.7663 / female g=+3.9499；判定: g{"<0 → pipeline 伪影读法" if g_ED < 0 else ">0 → 跨 pipeline 复现"}')

# 振幅矩阵与两性相关（EPG/PEN 3×3 与全 6×6）
lp = __import__('pickle').load(open(ROOT + 'latent_params.pkl', 'rb'))
am = np.array(lp['a'])
fj = json.load(open(ROOT + 'data_female/cx_latent_f.json'))
af = np.array(fj['a'])
ah = a_f.numpy()
blk_h = ['EPG', 'PEN_a(PEN1)', 'PEN_b(PEN2)']
tm = lp['type_list']
tf = fj['types']
blk_f = ['EPG', 'PEN_a/PEN1', 'PEN_b/PEN2']
Bh = ah[np.ix_([type_list.index(t) for t in blk_h], [type_list.index(t) for t in blk_h])]
Bm = am[np.ix_([tm.index(t) for t in blk_h], [tm.index(t) for t in blk_h])]
Bf = af[np.ix_([tf.index(t) for t in blk_f], [tf.index(t) for t in blk_f])]
r3_hm = float(np.corrcoef(Bh.ravel(), Bm.ravel())[0, 1])
r3_hf = float(np.corrcoef(Bh.ravel(), Bf.ravel())[0, 1])
# 全 6×6（雌性类型名映射到 Janelia 命名）
f2j = {'EPG': 'EPG', 'EPGt': 'EPGt', 'PEN_a/PEN1': 'PEN_a(PEN1)', 'PEN_b/PEN2': 'PEN_b(PEN2)',
       'PEG': 'PEG', 'Delta7': 'Delta7'}
Af6 = af[np.ix_(range(6), range(6))]  # female 与 male 类型集合相同、次序不同
order_f = [tf.index(next(k for k, v in f2j.items() if v == t)) for t in type_list]
Af6 = af[np.ix_(order_f, order_f)]
order_m = [tm.index(t) for t in type_list]
Am6 = am[np.ix_(order_m, order_m)]
r6_hm = float(np.corrcoef(ah.ravel(), Am6.ravel())[0, 1])
r6_hf = float(np.corrcoef(ah.ravel(), Af6.ravel())[0, 1])
log(f'振幅相关: EPG/PEN 3×3  H–M r={r3_hm:.4f} / H–F r={r3_hf:.4f} | 全 6×6  H–M r={r6_hm:.4f} / H–F r={r6_hf:.4f}')

res = {
    'N': N, 'types': type_list,
    'a': ah.tolist(), 'phi': phi_f.numpy().tolist(), 'b': b_f.numpy().tolist(),
    'theta': th_np.tolist(),
    'auc_latent': auc_lat, 'wcorr_latent': corr_lat,
    'auc_type_baseline': auc_type, 'wcorr_type': corr_type,
    'kendall_tau_epg': float(tau),
    'a_EPG_D7': a_ED, 'phi_EPG_D7': phi_ED, 'g_EPG_D7': g_ED,
    'r3x3_HM': r3_hm, 'r3x3_HF': r3_hf, 'r6x6_HM': r6_hm, 'r6x6_HF': r6_hf,
    'model_race': out,
}
json.dump(res, open(ROOT + 'data_hemibrain/cx_latent_hb.json', 'w'), indent=1)
log('DONE -> data_hemibrain/cx_latent_hb.json')
