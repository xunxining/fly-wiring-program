#!/usr/bin/env python3
"""93b_lattice_propagate.py — 谐扩展传播: 真实六边形列坐标 (p,q) 沿视叶连接图扩散到全部视叶细胞
锚点: column_assignment.csv.gz (43k 视叶细胞, 31 髓质柱状型, ~796 列 × 2 半球)
泄漏控制: 只用 90% 拟合边 (与 93 同一 rng seed=42 划分); 另扣 10% 锚点做恢复精度检验
输出: data_female/lat2q_f.npz {pq:(N,2) f32 NaN=未覆盖, hem:(N,) i8 -1/0/+1}
      data_female/lat2q_diag.json
"""
import numpy as np, json, time
import pandas as pd
from scipy.sparse import csr_matrix
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()

ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
ew = np.fromfile(D + "e_w.i32", dtype=np.int32).astype(np.float64)
sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
N = len(sc)
om = sc == 5
oi = np.where(om)[0]; no = len(oi)

# ---- 与 93 完全一致的拟合/留边划分 (rng 42, 同一随机序列) ----
rng = np.random.default_rng(42)
sel = om[ep] & om[eq]
p, q, w = ep[sel], eq[sel], ew[sel]
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)
keep = p2 != q2
p2, q2, w = p2[keep], q2[keep], w[keep]
fit = rng.random(len(p2)) < 0.9
print(f"optic-optic edges fit {fit.sum():,} held {(~fit).sum():,}", flush=True)

r = np.concatenate([p2[fit], q2[fit]]); c = np.concatenate([q2[fit], p2[fit]])
ww = np.concatenate([w[fit], w[fit]])
A = csr_matrix((ww, (r, c)), shape=(no, no)); A.sum_duplicates()
dvec = np.asarray(A.sum(1)).ravel(); dvec[dvec <= 0] = 1e-9
Dinv = 1.0 / dvec

# ---- 锚点 ----
node_ids = np.fromfile(D + "node_ids.i64", dtype=np.int64)
order = np.argsort(node_ids); sids = node_ids[order]
ca = pd.read_csv(D + "column_assignment.csv.gz")
j = np.searchsorted(sids, ca.root_id.values)
okm = (j < len(sids)) & (sids[np.clip(j, 0, len(sids) - 1)] == ca.root_id.values)
glo = order[j[okm]]                       # 全局节点号
lok = np.searchsorted(oi, glo)            # 视叶内局部号 (oi 有序)
isl = om[glo]
lok, ca2 = lok[isl], ca[okm][isl]
hemv = (ca2.hemisphere.values == "right").astype(np.float64) * 2 - 1   # L=-1 R=+1
labels = np.column_stack([ca2.p.values.astype(np.float64),
                          ca2.q.values.astype(np.float64), hemv])
print(f"anchors: {len(lok):,}", flush=True)

# 扣 10% 锚点做恢复检验
rng2 = np.random.default_rng(7)
hold = rng2.random(len(lok)) < 0.10
anch = np.ones(no, bool); anch[lok[~hold]] = True   # 注意: 仅锚点为 True
anch = np.zeros(no, bool); anch[lok[~hold]] = True
Y = np.zeros((no, 3)); Y[lok[~hold]] = labels[~hold]

# ---- 谐扩展迭代 ----
X = Y.copy()
for it in range(1500):
    Xn = Dinv[:, None] * (A @ X)
    Xn[anch] = Y[anch]
    mx = np.abs(Xn - X).max()
    X = Xn
    if mx < 1e-4:
        break
print(f"converged it={it} maxdelta={mx:.2e}  {time.time()-t0:.0f}s", flush=True)

# ---- 恢复精度 (held anchors) ----
hl = lok[hold]; true = labels[hold]
pred = X[hl]
err = np.sqrt(((pred[:, :2] - true[:, :2]) ** 2).sum(1))
hem_acc = (np.sign(pred[:, 2]) == true[:, 2]).mean()
print(f"heldout anchors: median err {np.median(err):.3f} hex units, "
      f"P(err<=1)={np.mean(err<=1):.3f}, hem acc {hem_acc:.4f}", flush=True)

# ---- 输出坐标: 每半球独立坐标架, hem 分离 ----
pq = X[:, :2].astype(np.float32)
hem = np.sign(X[:, 2]).astype(np.int8)
deg0 = np.asarray((A != 0).sum(1)).ravel() == 0
pq[deg0] = np.nan; hem[deg0] = 0
PQ = np.full((N, 2), np.nan, np.float32); HEM = np.zeros(N, np.int8)
PQ[oi] = pq; HEM[oi] = hem

# ---- 留边潜距离 KS (泛化) ----
def latdist(pp, qq):
    a, b = pq[pp], pq[qq]
    same = (hem[pp] == hem[qq]) & (hem[pp] != 0)
    d = np.sqrt(((a - b) ** 2).sum(1))
    return d[same & ~np.isnan(d)]
dh = latdist(p2[~fit], q2[~fit])
df_ = latdist(p2[fit], q2[fit])
df_ = df_[np.random.default_rng(0).choice(len(df_), min(len(df_), 300000), replace=False)]
from scipy.stats import ks_2samp
ks = float(ks_2samp(dh, df_).statistic)
rand_a = rng2.integers(0, no, 200000); rand_b = rng2.integers(0, no, 200000)
dr = latdist(rand_a, rand_b)
diag = {
    "anchors": int(len(lok)), "anchors_held": int(hold.sum()),
    "held_anchor_median_err_hex": float(np.median(err)),
    "held_anchor_err_le1_frac": float(np.mean(err <= 1)),
    "held_anchor_hem_acc": float(hem_acc),
    "holdout_edge_latdist_ks": ks,
    "held_edge_latdist_median": float(np.median(dh)),
    "random_pair_latdist_median": float(np.median(dr)),
    "coverage_frac_optic": float((~deg0).mean()),
    "n_deg0": int(deg0.sum()), "iters": int(it),
}
np.savez_compressed(D + "lat2q_f.npz", pq=PQ, hem=HEM)
json.dump(diag, open(D + "lat2q_diag.json", "w"), indent=1)
print(json.dumps(diag, indent=1), flush=True)
print(f"SAVED lat2q_f.npz  {time.time()-t0:.0f}s")
