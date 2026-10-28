#!/usr/bin/env python3
"""130_male_snap2.py — 雄侧 snap 核 m2 (P9a-2, §40): 坐标升级 lat2q_m2 (soma直测); 其余与 104 完全一致
核估计边 = rng42 拟合划分 ∩ 双端有坐标 (与 103 同一划分)。
输出: data_male/lat_snap_m2.npz {big_uk, crbig, cr_glob, pr_glob, R} + lat_snap_diag_m2.json
"""
import numpy as np, json, time, gc
D = "/mnt/agents/output/果蝇生成器/data_male/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

ep = np.fromfile(D+"e_pre.i32", dtype=np.int32)
eq = np.fromfile(D+"e_post.i32", dtype=np.int32)
sc = np.fromfile(D+"node_sc.i32", dtype=np.int32)
nt = np.fromfile(D+"node_type.i32", dtype=np.int32)
z = np.load(D+"lat2q_m2.npz"); PQ = z["pq"]; HEM = z["hem"]
N = len(sc); T = int(nt.max())+1
has = HEM != 0
vis = np.isin(sc, [13, 14, 18, 19])
oi = np.where(vis)[0]

# ---- 复刻 103 划分 ----
rng42 = np.random.default_rng(42)
sel = vis[ep] & vis[eq]
p, q = ep[sel], eq[sel]
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)
keep = p2 != q2
p2, q2 = p2[keep], q2[keep]
fit = rng42.random(len(p2)) < 0.9
fit_g = np.zeros(len(ep), bool)
idx_sel = np.where(sel)[0][keep]
fit_g[idx_sel[fit]] = True
ek = fit_g & has[ep] & has[eq]
P, Q = ep[ek], eq[ek]
del p, q, p2, q2, keep, fit, sel, idx_sel, fit_g; gc.collect()
log(f"核估计拟合边 {len(P):,}")

R = 20; NB1 = 2*R+1; NB = NB1*NB1+2
OVER, CROSS = NB1*NB1, NB1*NB1+1
same = HEM[P] == HEM[Q]
dp = np.zeros(len(P), np.int64); dq = np.zeros(len(P), np.int64)
dp[same] = np.rint(PQ[Q[same],0] - PQ[P[same],0])
dq[same] = np.rint(PQ[Q[same],1] - PQ[P[same],1])
inr = same & (np.abs(dp) <= R) & (np.abs(dq) <= R)
b = np.full(len(P), CROSS, np.int64)
b[same] = OVER
b[inr] = (dp[inr]+R)*NB1 + (dq[inr]+R)
del dp, dq, same, inr; gc.collect()

k = nt[P].astype(np.int64)*T + nt[Q]
Hr = np.bincount(b, minlength=NB).astype(np.float64)
g_r = Hr/Hr.sum()
uk, cnts = np.unique(k, return_counts=True)
MIN_E = 200
big_uk = np.sort(uk[cnts >= MIN_E]); nbig = len(big_uk)
li = np.searchsorted(big_uk, k)
hit = (li < nbig) & (big_uk[np.clip(li, 0, nbig-1)] == k)
c_real = np.bincount(li[hit]*NB + b[hit], minlength=nbig*NB).reshape(nbig, NB).astype(np.float64)
log(f"大对 {nbig:,}, 覆盖 {hit.mean():.2%}")

KAPPA = 25.0
sr = c_real.sum(1, keepdims=True)
prbig = ((c_real + KAPPA*g_r)/(sr+KAPPA)).astype(np.float32)
crbig = np.cumsum(prbig, axis=1).astype(np.float32)
cr_glob = np.cumsum(g_r.astype(np.float32))
pr_glob = g_r.astype(np.float32)
np.savez(D+"lat_snap_m2.npz", big_uk=big_uk, crbig=crbig, cr_glob=cr_glob, R=R, pr_glob=pr_glob)
bm = prbig.argmax(1)
diag = {"fit_edges": int(len(P)), "nbig": int(nbig), "big_edge_share": float(hit.mean()),
        "real_cross_frac": float(g_r[CROSS]), "real_over_frac": float(g_r[OVER]),
        "P_center": float(g_r[R*NB1+R]),
        "over_modal_pairs": int((bm == OVER).sum()), "cross_modal_pairs": int((bm == CROSS).sum()),
        "peakiness_median": float(np.median(prbig.max(1))),
        "peakiness_p90": float(np.percentile(prbig.max(1), 90))}
json.dump(diag, open(D+"lat_snap_diag_m2.json", "w"), indent=1)
log(json.dumps(diag, indent=1))
log("SNAP KERNEL M DONE")
