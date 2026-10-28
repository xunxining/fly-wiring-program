#!/usr/bin/env python3
"""96b_snap_kernel_v2.py — 偏移分布 v2: 坐标 lat2q_f3 (视叶 v1 + 感光/VPN/离心 v2), R=20
实边: 两端都有坐标的边; 拟合划分复刻 93c (rng42 在扩展边集上) 的 fit 掩码映射回原边。
注意: 93c 的 fit 定义在 visual-visual 边集; 这里核估计边集 = 双端有坐标 (与生成器 t_lat 口径一致)。
输出: data_female/lat_snap_f2.npz {big_uk, crbig, cr_glob, R}
"""
import numpy as np, json, time
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
nt = np.fromfile(D + "node_type.i32", dtype=np.int32)
z = np.load(D + "lat2q_f3.npz"); PQ = z["pq"]; HEM = z["hem"]
N = len(sc); T = int(nt.max()) + 1
has = HEM != 0
sc_names = np.load(D + "sc_names.npy", allow_pickle=True)
want = {"optic", "sensory", "visual_projection", "visual_centrifugal"}
vis = np.array([str(sc_names[s]) in want for s in sc])
oi = np.where(vis)[0]

# ---- 复刻 93c 划分 ----
rng42 = np.random.default_rng(42)
sel = vis[ep] & vis[eq]
p, q = ep[sel], eq[sel]
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)
keep = p2 != q2
fit = rng42.random(len(p2)) < 0.9
fit_g = np.zeros(len(ep), bool)
idx_sel = np.where(sel)[0][keep]
fit_g[idx_sel[fit]] = True
# 核估计边: 双端有坐标且为拟合边
ek = fit_g & has[ep] & has[eq]
P, Q = ep[ek], eq[ek]
log(f"核估计拟合边 {len(P):,}")

R = 20; NB1 = 2 * R + 1; NB = NB1 * NB1 + 2
OVER, CROSS = NB1 * NB1, NB1 * NB1 + 1
okp, okq = has[P], has[Q]
same = okp & okq & (HEM[P] == HEM[Q])
dp = np.zeros(len(P), np.int64); dq = np.zeros(len(P), np.int64)
dp[same] = np.rint(PQ[Q[same], 0] - PQ[P[same], 0])
dq[same] = np.rint(PQ[Q[same], 1] - PQ[P[same], 1])
inr = same & (np.abs(dp) <= R) & (np.abs(dq) <= R)
b = np.full(len(P), CROSS, np.int64)
b[same] = OVER
b[inr] = (dp[inr] + R) * NB1 + (dq[inr] + R)

k = nt[P].astype(np.int64) * T + nt[Q]
Hr = np.bincount(b, minlength=NB).astype(np.float64)
g_r = Hr / Hr.sum()

uk, cnts = np.unique(k, return_counts=True)
MIN_E = 200
big_uk = np.sort(uk[cnts >= MIN_E]); nbig = len(big_uk)
li = np.searchsorted(big_uk, k)
hit = (li < nbig) & (big_uk[np.clip(li, 0, nbig - 1)] == k)
c_real = np.bincount(li[hit] * NB + b[hit], minlength=nbig * NB).reshape(nbig, NB).astype(np.float64)
log(f"大对 {nbig:,}, 覆盖 {hit.mean():.2%}")

KAPPA = 25.0
sr = c_real.sum(1, keepdims=True)
prbig = ((c_real + KAPPA * g_r) / (sr + KAPPA)).astype(np.float32)
crbig = np.cumsum(prbig, axis=1).astype(np.float32)
cr_glob = np.cumsum(g_r.astype(np.float32))
pr_glob = g_r.astype(np.float32)
np.savez(D + "lat_snap_f2.npz", big_uk=big_uk, crbig=crbig, cr_glob=cr_glob, R=R,
         pr_glob=pr_glob)
bm = prbig.argmax(1)
diag = {"fit_edges": int(len(P)), "nbig": int(nbig), "big_edge_share": float(hit.mean()),
        "real_cross_frac": float(g_r[CROSS]), "real_over_frac": float(g_r[OVER]),
        "P_center": float(g_r[R * NB1 + R]),
        "over_modal_pairs": int((bm == OVER).sum()), "cross_modal_pairs": int((bm == CROSS).sum()),
        "peakiness_median": float(np.median(prbig.max(1))),
        "peakiness_p90": float(np.percentile(prbig.max(1), 90))}
json.dump(diag, open(D + "lat_snap_diag2.json", "w"), indent=1)
log(json.dumps(diag, indent=1))
log("SNAP KERNEL v2 DONE")
