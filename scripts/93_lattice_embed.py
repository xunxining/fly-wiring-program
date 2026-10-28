#!/usr/bin/env python3
"""93_lattice_embed.py — 视叶子图谱嵌入: 每个视叶细胞一个二维潜晶格坐标
泄漏控制: 只用 90% 随机边拟合嵌入; 10% 留边用于泛化检验 (边长分布对比)。
输出: data_female/lat2d_f.npz {lat:(N,2) f32 NaN=非视叶, comp, fit_edge_mask_info}
"""
import numpy as np, json, time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
rng = np.random.default_rng(42)

ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
ew = np.fromfile(D + "e_w.i32", dtype=np.int32)
sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
N = len(sc)
om = sc == 5                       # optic superclass
oi = np.where(om)[0]
no = len(oi)
print(f"optic cells: {no:,}", flush=True)

# 视叶-视叶无向加权边
sel = om[ep] & om[eq]
p, q, w = ep[sel], eq[sel], ew[sel].astype(np.float64)
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)   # oi 有序
keep = p2 != q2
p2, q2, w = p2[keep], q2[keep], w[keep]
fit = rng.random(len(p2)) < 0.9                            # 90% 拟合
E_oo = len(p2)
print(f"optic-optic edges: {E_oo:,} (fit {fit.sum():,}, held {(~fit).sum():,})", flush=True)

def build(p_, q_, w_):
    r = np.concatenate([p_, q_]); c = np.concatenate([q_, p_]); ww = np.concatenate([w_, w_])
    A = csr_matrix((ww, (r, c)), shape=(no, no))
    A.sum_duplicates()
    return A

A = build(p2[fit], q2[fit], w[fit])
# 连通分量 (在未加权结构上)
from scipy.sparse.csgraph import connected_components
nc, lab = connected_components(A, directed=False)
sizes = np.bincount(lab)
print(f"components: {nc}, top sizes: {np.sort(sizes)[::-1][:6]}", flush=True)

lat = np.full((N, 2), np.nan, np.float32)
comp_id = np.full(N, -1, np.int32)
diag = {"n_optic": int(no), "E_oo": int(E_oo), "components": nc,
        "comp_sizes": np.sort(sizes)[::-1][:8].tolist()}

for ci in np.argsort(-sizes)[:4]:            # 处理前四大分量
    if sizes[ci] < 100: break
    mem = np.where(lab == ci)[0]
    As = A[mem][:, mem]
    dvec = np.asarray(As.sum(1)).ravel()
    dvec[dvec <= 0] = 1e-9
    Dm12 = 1.0 / np.sqrt(dvec)
    # 规范: S = D^-1/2 A D^-1/2
    from scipy.sparse import diags as sp_diags
    Dm = sp_diags(Dm12)
    S = (Dm @ As @ Dm).tocsr()
    vals, vecs = eigsh(S, k=3, which="LA", maxiter=3000)
    # 特征值≈1 是平凡分量; 取第 2、3 大作为坐标
    order_ev = np.argsort(-vals)
    u1 = vecs[:, order_ev[1]] / Dm12
    u2 = vecs[:, order_ev[2]] / Dm12
    glo = oi[mem]
    lat[glo, 0] = u1; lat[glo, 1] = u2
    comp_id[glo] = ci
    print(f"comp {ci}: n={len(mem):,} eig={np.round(vals,4)}", flush=True)

# 泛化检验: 留边 vs 拟合边的潜距离分布 (KS)
def latdist(p_, q_):
    a, b = oi[p_], oi[q_]
    ok = ~np.isnan(lat[a, 0]) & ~np.isnan(lat[b, 0])
    d = np.sqrt(((lat[a[ok]] - lat[b[ok]]) ** 2).sum(1))
    return d, ok
dh, okh = latdist(p2[~fit], q2[~fit])
df_, okf = latdist(p2[fit], q2[fit])
df_ = df_[np.random.default_rng(0).choice(len(df_), min(len(df_), 200000), replace=False)]
from scipy.stats import ks_2samp
diag["holdout_latdist_ks"] = float(ks_2samp(dh, df_).statistic)
diag["held_edges_mapped_frac"] = float(okh.mean())
diag["held_latdist_median"] = float(np.median(dh))
diag["randompair_latdist_median"] = float(np.median(np.sqrt(((lat[oi[rng.integers(0,no,100000)],0]-lat[oi[rng.integers(0,no,100000)],0])**2 + (lat[oi[rng.integers(0,no,100000)],1]-lat[oi[rng.integers(0,no,100000)],1])**2))))
print("holdout KS:", diag["holdout_latdist_ks"], flush=True)

np.savez_compressed(D + "lat2d_f.npz", lat=lat, comp=comp_id)
json.dump(diag, open(D + "lat2d_diag.json", "w"), indent=1)
print(f"SAVED lat2d_f.npz  {time.time()-t0:.0f}s")
