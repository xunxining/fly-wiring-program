#!/usr/bin/env python3
"""94b_lat_kernel.py — 视叶型对二维晶格偏移核 w_st(Δp,Δq), 与 74 一维距离核同口径
实边: 仅 90% 拟合边 (复刻 93b 的 rng=42 划分); 基线: 型群内随机细胞对偏移分布
分箱: Δp,Δq ∈ [-10,10] 整数网格 21x21=441; 441=溢出(同半球远程); 442=跨半球/无坐标
输出: data_female/lat_kernel_f.npz {big_uk, wbig, wmax_big, w_glob, wmax_glob}
      data_female/lat_kernel_diag.json
"""
import numpy as np, json, time
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
nt = np.fromfile(D + "node_type.i32", dtype=np.int32)
z = np.load(D + "lat2q_f.npz"); PQ = z["pq"]; HEM = z["hem"]
N = len(sc); T = int(nt.max()) + 1
om = sc == 5
has = om & (HEM != 0)

# ---- 复刻 93b 拟合划分 ----
rng42 = np.random.default_rng(42)
oi = np.where(om)[0]
sel = om[ep] & om[eq]
p, q = ep[sel], eq[sel]
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)
keep = p2 != q2
fit = rng42.random(len(p2)) < 0.9
fit_g = np.zeros(len(ep), bool)
idx_sel = np.where(sel)[0][keep]
fit_g[idx_sel[fit]] = True
P = ep[fit_g]; Q = eq[fit_g]
log(f"拟合边 {len(P):,}")

R = 10; NB1 = 2 * R + 1; NB = NB1 * NB1 + 2   # 441 + overflow + cross
OVER, CROSS = NB1 * NB1, NB1 * NB1 + 1
def bin_of(sp_, sq_, hem_s, hem_t, ok_s, ok_t):
    same = ok_s & ok_t & (hem_s == hem_t)
    dp = (sq_[:, 0] - sp_[:, 0]).round().astype(np.int64)
    dq = (sq_[:, 1] - sp_[:, 1]).round().astype(np.int64)
    inr = same & (np.abs(dp) <= R) & (np.abs(dq) <= R)
    b = np.full(len(dp), CROSS, np.int64)
    b[same] = OVER
    b[inr] = (dp[inr] + R) * NB1 + (dq[inr] + R)
    return b

k = nt[P].astype(np.int64) * T + nt[Q]
b_real = bin_of(PQ[P], PQ[Q], HEM[P], HEM[Q], has[P], has[Q])
Hr = np.bincount(b_real, minlength=NB).astype(np.float64)

# ---- 大对 ----
cnt = np.bincount(k, minlength=1)  # 太慢改 unique
uk, cnts = np.unique(k, return_counts=True)
MIN_E = 200
big_uk = np.sort(uk[cnts >= MIN_E]); nbig = len(big_uk)
li = np.searchsorted(big_uk, k)
hit = (li < nbig) & (big_uk[np.clip(li, 0, nbig - 1)] == k)
c_real = np.bincount(li[hit] * NB + b_real[hit], minlength=nbig * NB).reshape(nbig, NB).astype(np.float64)
log(f"大对 {nbig:,} (>= {MIN_E} 拟合边), 覆盖拟合边 {hit.mean():.2%}")

# ---- 细胞坐标按型分组 (toff 结构) ----
order_t = np.argsort(nt, kind="stable")
toff = np.append(np.searchsorted(nt[order_t], np.arange(T)), N).astype(np.int64)
# 全局数组: 型内第 j 个细胞 -> 全局号 gcell[t_off + j]
gcell = order_t.astype(np.int64)

rng = np.random.default_rng(5)
NS = 64
def cell_coords(t_, js):
    g = gcell[toff[t_] + js]
    return PQ[g], HEM[g], has[g]

# 全局基线
M = 2_000_000
si_s = rng.integers(0, E_ := len(P), M)   # 以真实拟合边的 (型,源细胞) 为锚, 目标随机
# 简化: 从全体视叶细胞池抽源、目标
opool = np.where(has)[0]
S_ = opool[rng.integers(0, len(opool), M)]
G_ = opool[rng.integers(0, len(opool), M)]
Hbg = np.bincount(bin_of(PQ[S_], PQ[G_], HEM[S_], HEM[G_], has[S_], has[G_]), minlength=NB).astype(np.float64)

# 逐大对基线
bs = (big_uk // T).astype(np.int64); bt = (big_uk % T).astype(np.int64)
c_base = np.zeros((nbig, NB))
ns_arr = toff[bs + 1] - toff[bs]; nt_arr = toff[bt + 1] - toff[bt]
for c0 in range(0, nbig, 256):
    c1 = min(c0 + 256, nbig)
    for r in range(c0, c1):
        s_, t_ = int(bs[r]), int(bt[r])
        js = rng.integers(0, ns_arr[r], NS); jt = rng.integers(0, nt_arr[r], NS)
        ps_, hs_, oks_ = cell_coords(s_, js)
        pt_, ht_, okt_ = cell_coords(t_, jt)
        PP = np.repeat(ps_, NS, axis=0); PT = np.tile(pt_, (NS, 1))
        HHs = np.repeat(hs_, NS); HHt = np.tile(ht_, NS)
        OOs = np.repeat(oks_, NS); OOt = np.tile(okt_, NS)
        bb = bin_of(PP, PT, HHs, HHt, OOs, OOt)
        c_base[r] = np.bincount(bb, minlength=NB)
    if (c0 // 256) % 4 == 0: log(f"  基线 {c1:,}/{nbig:,}")

KAPPA, WCAP = 25.0, 24.0
g_r = Hr / Hr.sum(); g_b = Hbg / Hbg.sum()
def make_w(cr, cb):
    sr = cr.sum(1, keepdims=True); sb = cb.sum(1, keepdims=True)
    pr = (cr + KAPPA * g_r) / (sr + KAPPA)
    pb = (cb + KAPPA * g_b) / (sb + KAPPA)
    bad = (sr.ravel() < 1) | (sb.ravel() < 1)
    wk = pr / np.maximum(pb, 1e-12)
    wk = wk / np.maximum((wk * pb).sum(1, keepdims=True), 1e-12)
    wk = np.minimum(wk, WCAP)
    wk = wk / np.maximum((wk * pb).sum(1, keepdims=True), 1e-12)
    wg = g_r / np.maximum(g_b, 1e-12); wg = wg / (wg * g_b).sum()
    wk[bad] = np.minimum(wg, WCAP)
    return wk.astype(np.float32), wk.max(1).astype(np.float32), bad

wbig, wmax_big, bad = make_w(c_real, c_base)
wg = g_r / np.maximum(g_b, 1e-12); wg = wg / (wg * g_b).sum()
w_glob = np.minimum(wg, WCAP).astype(np.float32); w_glob /= (w_glob * g_b).sum()
wmax_glob = float(w_glob.max())
np.savez(D + "lat_kernel_f.npz", big_uk=big_uk, wbig=wbig, wmax_big=wmax_big,
         w_glob=w_glob, wmax_glob=wmax_glob, R=R)
diag = {"fit_edges": int(len(P)), "nbig": int(nbig), "big_edge_share": float(hit.mean()),
        "degenerate_pairs": int(bad.sum()),
        "wmax_mean": float(wmax_big.mean()), "wmax_p50": float(np.percentile(wmax_big, 50)),
        "wmax_p90": float(np.percentile(wmax_big, 90)), "wmax_glob": wmax_glob,
        "real_cross_frac": float(g_r[CROSS]), "base_cross_frac": float(g_b[CROSS]),
        "real_over_frac": float(g_r[OVER]),
        "w_glob_cross": float(w_glob[CROSS]), "w_glob_over": float(w_glob[OVER]),
        "w_glob_center_(0,0)": float(w_glob[R * NB1 + R])}
json.dump(diag, open(D + "lat_kernel_diag.json", "w"), indent=1)
log(json.dumps(diag, indent=1))
log("KERNEL DONE")
