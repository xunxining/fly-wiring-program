#!/usr/bin/env python3
"""93c_lattice_propagate_v2.py — 谐扩展 v2: 图扩到 视叶∪感光∪视觉投射∪视觉离心
锚点不变 (43k 髓质柱状细胞); 感光细胞(R1-8 等)经与 lamina 细胞的边获得列坐标。
泄漏控制: rng42 在扩展边集上划 90/10; 96b 复刻同一划分。
输出: data_female/lat2q_f2.npz {pq:(N,2) f32, hem:(N,) i8}
      data_female/lat2q2_diag.json
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
sc_names = np.load(D + "sc_names.npy", allow_pickle=True)
want = {"optic", "sensory", "visual_projection", "visual_centrifugal"}
vis = np.array([str(sc_names[s]) in want for s in sc])
oi = np.where(vis)[0]; no = len(oi)
print(f"visual-set cells: {no:,}", flush=True)

# ---- 扩展边集上的拟合划分 (rng42) ----
rng42 = np.random.default_rng(42)
sel = vis[ep] & vis[eq]
p, q, w = ep[sel], eq[sel], ew[sel]
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)
keep = p2 != q2
p2, q2, w = p2[keep], q2[keep], w[keep]
fit = rng42.random(len(p2)) < 0.9
print(f"visual-visual edges: {len(p2):,} (fit {fit.sum():,})", flush=True)

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
glo = order[j[okm]]
isl = vis[glo]
lok = np.searchsorted(oi, glo[isl]); ca2 = ca[okm][isl]
hemv = (ca2.hemisphere.values == "right").astype(np.float64) * 2 - 1
labels = np.column_stack([ca2.p.values.astype(np.float64), ca2.q.values.astype(np.float64), hemv])
print(f"anchors: {len(lok):,}", flush=True)

rng2 = np.random.default_rng(7)
hold = rng2.random(len(lok)) < 0.10
anch = np.zeros(no, bool); anch[lok[~hold]] = True
Y = np.zeros((no, 3)); Y[lok[~hold]] = labels[~hold]

X = Y.copy()
for it in range(1500):
    Xn = Dinv[:, None] * (A @ X)
    Xn[anch] = Y[anch]
    mx = np.abs(Xn - X).max()
    X = Xn
    if mx < 1e-4: break
print(f"converged it={it} maxdelta={mx:.2e}  {time.time()-t0:.0f}s", flush=True)

hl = lok[hold]; true = labels[hold]
err = np.sqrt(((X[hl, :2] - true[:, :2]) ** 2).sum(1))
hem_acc = (np.sign(X[hl, 2]) == true[:, 2]).mean()
print(f"heldout anchors: median err {np.median(err):.3f}, P<=1 {np.mean(err<=1):.3f}, hem {hem_acc:.4f}", flush=True)

pq = X[:, :2].astype(np.float32); hem = np.sign(X[:, 2]).astype(np.int8)
deg0 = np.asarray((A != 0).sum(1)).ravel() == 0
pq[deg0] = np.nan; hem[deg0] = 0
PQ = np.full((N, 2), np.nan, np.float32); HEM = np.zeros(N, np.int8)
PQ[oi] = pq; HEM[oi] = hem

# 各 superclass 覆盖
import collections
cov = {}
for s in range(len(sc_names)):
    m = (sc == s)
    if m.sum() == 0: continue
    cov[str(sc_names[s])] = [int(m.sum()), float((HEM[m] != 0).mean())]
diag = {"n_visual": int(no), "anchors": int(len(lok)),
        "held_anchor_median_err_hex": float(np.median(err)),
        "held_anchor_hem_acc": float(hem_acc),
        "coverage_by_sc": cov, "iters": int(it)}
np.savez_compressed(D + "lat2q_f2.npz", pq=PQ, hem=HEM)
json.dump(diag, open(D + "lat2q2_diag.json", "w"), indent=1)
print(json.dumps(cov, indent=1), flush=True)
print(f"SAVED lat2q_f2.npz  {time.time()-t0:.0f}s")
