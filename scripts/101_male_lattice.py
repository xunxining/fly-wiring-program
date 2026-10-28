#!/usr/bin/env python3
"""101_male_lattice.py — 雄侧视叶晶格嵌入(M1)
雌性 93 式的雄性移植: 视叶掩码 = sc==13 (ol_intrinsic, 89k)。
泄漏控制: 90% 边拟合, 10% 留边做潜距离 KS 泛化检验。
输出: data_census/lat2d_m.npz {lat:(N,2) f32 NaN=非视叶, comp} + lat2d_diag_m.json
"""
import numpy as np, json, time
from scipy.sparse import csr_matrix, diags as sp_diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp
DC = "/mnt/agents/output/果蝇生成器/data_census/"
t0 = time.time()
rng = np.random.default_rng(42)

ep = np.fromfile(DC + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(DC + "e_post.i32", dtype=np.int32)
ew = np.fromfile(DC + "e_w.i32", dtype=np.int32)
sc = np.fromfile(DC + "node_sc.i32", dtype=np.int32)
N = len(sc)
om = sc == 13                      # ol_intrinsic = 雄侧视叶
oi = np.where(om)[0]
no = len(oi)
print(f"male optic (ol_intrinsic) cells: {no:,}", flush=True)

sel = om[ep] & om[eq]
p, q, w = ep[sel], eq[sel], ew[sel].astype(np.float64)
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)
keep = p2 != q2
p2, q2, w = p2[keep], q2[keep], w[keep]
fit = rng.random(len(p2)) < 0.9
E_oo = len(p2)
print(f"optic-optic edges: {E_oo:,} (fit {fit.sum():,}, held {(~fit).sum():,})", flush=True)

r = np.concatenate([p2[fit], q2[fit]]); c = np.concatenate([q2[fit], p2[fit]])
ww = np.concatenate([w[fit], w[fit]])
A = csr_matrix((ww, (r, c)), shape=(no, no)); A.sum_duplicates()

nc, lab = connected_components(A, directed=False)
sizes = np.bincount(lab)
print(f"components: {nc}, top: {np.sort(sizes)[::-1][:6]}", flush=True)

lat = np.full((N, 2), np.nan, np.float32)
comp_id = np.full(N, -1, np.int32)
diag = {"n_optic": int(no), "E_oo": int(E_oo), "components": int(nc),
        "comp_sizes": np.sort(sizes)[::-1][:8].tolist()}

for ci in np.argsort(-sizes)[:4]:
    if sizes[ci] < 100: break
    mem = np.where(lab == ci)[0]
    As = A[mem][:, mem]
    dvec = np.asarray(As.sum(1)).ravel()
    dvec[dvec <= 0] = 1e-9
    Dm12 = 1.0 / np.sqrt(dvec)
    S = (sp_diags(Dm12) @ As @ sp_diags(Dm12)).tocsr()
    vals, vecs = eigsh(S, k=3, which="LA", maxiter=3000)
    order_ev = np.argsort(-vals)
    u1 = vecs[:, order_ev[1]] / Dm12
    u2 = vecs[:, order_ev[2]] / Dm12
    glo = oi[mem]
    lat[glo, 0] = u1; lat[glo, 1] = u2
    comp_id[glo] = ci
    print(f"comp {ci}: n={len(mem):,} eig={np.round(vals,4)}", flush=True)

def latdist(p_, q_):
    a, b = oi[p_], oi[q_]
    ok = ~np.isnan(lat[a, 0]) & ~np.isnan(lat[b, 0])
    return np.sqrt(((lat[a[ok]] - lat[b[ok]]) ** 2).sum(1)), ok
dh, okh = latdist(p2[~fit], q2[~fit])
df_, _ = latdist(p2[fit], q2[fit])
df_ = df_[np.random.default_rng(0).choice(len(df_), min(len(df_), 200000), replace=False)]
diag["holdout_latdist_ks"] = float(ks_2samp(dh, df_).statistic)
diag["held_edges_mapped_frac"] = float(okh.mean())
diag["held_latdist_median"] = float(np.median(dh))
ra = oi[rng.integers(0, no, 100000)]; rb = oi[rng.integers(0, no, 100000)]
diag["randompair_latdist_median"] = float(np.median(np.sqrt((lat[ra,0]-lat[rb,0])**2 + (lat[ra,1]-lat[rb,1])**2)))
print("holdout KS:", diag["holdout_latdist_ks"], flush=True)

np.savez_compressed(DC + "lat2d_m.npz", lat=lat, comp=comp_id)
json.dump(diag, open(DC + "lat2d_diag_m.json", "w"), indent=1)
print(f"SAVED lat2d_m.npz  {time.time()-t0:.0f}s", flush=True)
