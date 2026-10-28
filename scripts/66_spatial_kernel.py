#!/usr/bin/env python3
# 66_spatial_kernel.py — 空间距离核学习 (§22 收获项 1)
# 逐 (s,t) 对学 w_st(d) = 真实边距离直方图 / 配额均匀基线直方图 (24 log bins, um)
# 输出 spatial_kernel.npz: bins, big_uk, wbig (nbig,24), wmax_big, w_glob (24), wmax_glob
# 输出 xyz_gen.f32: 全节点 soma 坐标 (um, float32, NaN=缺失), 顺序 = genome/census 掩码序
import sys, json, time
sys.path.insert(0, "/mnt/agents/output/果蝇生成器/vendor")
import numpy as np, pandas as pd

ROOT = "/mnt/agents/output/果蝇生成器/"
rng = np.random.default_rng(7)
t0 = time.time()

def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

T = 11751
NB = 24
BINS = np.geomspace(0.5, 200.0, NB + 1).astype(np.float64)   # um
KAPPA = 25.0          # 向全局收缩的伪计数
WCAP = 24.0           # 接受率上限的代价换保真
NSAMP = 64            # 逐对基线: 64x64 源-目标采样

# ---- 坐标 (沿用 64 的构建, 转 um) ----
ann = pd.read_feather(ROOT + "data/body-annotations.feather", columns=["bodyId", "somaLocation"])
node_ids = np.fromfile(ROOT + "data_census/node_ids.i64", dtype=np.int64)
nt = np.fromfile(ROOT + "data_census/node_type.i32", dtype=np.int32)
tm = nt >= 0
nid_valid = node_ids[tm] if len(node_ids) == len(nt) else node_ids
N = int(tm.sum())
sl = ann["somaLocation"]
has = sl.notna().to_numpy()
amp = pd.Series(np.arange(len(ann)), index=ann["bodyId"].values)
pos_idx = amp.reindex(nid_valid).to_numpy()
okn = ~np.isnan(pos_idx)
loc_arr = np.full((len(ann), 3), np.nan)
loc_arr[has] = np.vstack(sl[has].to_numpy())
xyz = np.full((N, 3), np.nan, np.float64)
xyz[okn] = loc_arr[pos_idx[okn].astype(int)]
xyz = (xyz / 1000.0).astype(np.float32)   # nm -> um
cov = ~np.isnan(xyz[:, 0])
xyz.tofile(ROOT + "xyz_gen.f32")
del ann, sl, loc_arr, amp, has, pos_idx, okn, nid_valid
import gc; gc.collect()
log(f"节点 {N:,}, soma 覆盖 {cov.mean():.3f}")

# ---- 图与基因组 ----
# 注意: 普查紧凑序不按类型分组! 真正的节点->基因组型标签是 nt_full[tm] (=ntc),
# 而非 repeat(arange, tc)。xyz_gen 需按 ntc 的 argsort 重排, 使基因组位置
# offsets[t]+j 对应第 j 个真实 t 型节点。
remap = np.cumsum(tm, dtype=np.int64) - 1
pre = np.fromfile(ROOT + "data_census/e_pre.i32", dtype=np.int32)
post = np.fromfile(ROOT + "data_census/e_post.i32", dtype=np.int32)
rp = remap[pre].astype(np.int64); rq = remap[post].astype(np.int64)
del pre, post
gd = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = gd["type_count"].astype(np.int64)
ntc = nt[tm].astype(np.int64)          # 紧凑序的真实基因组型标签
assert (np.bincount(ntc, minlength=T) == tc).all(), "type_count 与普查标签不符"
perm = np.argsort(ntc, kind="stable")  # 紧凑序 -> 基因组序 (按型分组)
gtype = np.repeat(np.arange(T), tc)    # 基因组序的型标签 (仅用于校验/池)
toff = np.concatenate([[0], np.cumsum(tc)])
uk = gd["uk"]; cnt = gd["cnt"]
# xyz 重排为基因组序并重写
xyz_compact = xyz
xyz = xyz_compact[perm]
xyz.tofile(ROOT + "xyz_gen.f32")
cov = ~np.isnan(xyz[:, 0])
log(f"真实边 {len(rp):,}, 基因组对 {len(uk):,}")

# ---- §23b 逐细胞活跃度: a_out/a_in = 真实逐细胞度 / 型均值 (基因组序) ----
odr_c = np.bincount(rp, minlength=N); idr_c = np.bincount(rq, minlength=N)
odr = odr_c[perm].astype(np.float64); idr = idr_c[perm].astype(np.float64)
tm_o = np.maximum(np.bincount(gtype, weights=odr, minlength=T) / np.maximum(tc, 1), 1e-9)
tm_i = np.maximum(np.bincount(gtype, weights=idr, minlength=T) / np.maximum(tc, 1), 1e-9)
a_out = np.clip(odr / tm_o[gtype], 0.1, 10.0)
a_in = np.clip(idr / tm_i[gtype], 0.1, 10.0)
a_in.astype(np.float32).tofile(ROOT + "act_in.f32")
log(f"活跃度: a_in 中位 {np.median(a_in):.2f}, p90 {np.percentile(a_in,90):.2f}")

# 类型 -> 有坐标节点池 (基因组序) + 活跃度权重前缀和缓存
cov_idx_g = np.flatnonzero(cov)                       # 基因组序的有坐标节点
order_cov = cov_idx_g[np.argsort(gtype[cov_idx_g], kind="stable")]
cnt_cov = np.bincount(gtype[cov_idx_g], minlength=T)
poff = np.concatenate([[0], np.cumsum(cnt_cov)])
_cw = {}
def pool_draw(t_, m, w):
    """类型 t 池内按权重 w 抽 m 个基因组序节点"""
    if t_ not in _cw:
        pool = order_cov[poff[t_]:poff[t_] + cnt_cov[t_]]
        wt = np.cumsum(w[pool])
        _cw[t_] = (pool, wt)
    pool, wt = _cw[t_]
    if len(pool) == 0:
        return None
    return pool[np.searchsorted(wt, rng.random(m) * wt[-1])]

def bin_of(d):
    return np.clip(np.searchsorted(BINS, d, side="right") - 1, 0, NB - 1)

# ---- 真实边距离: 全局直方图 + 逐大对直方图 ----
# 边端点用紧凑序坐标 xyz_compact 与真实标签 ntc
cov_c = ~np.isnan(xyz_compact[:, 0])
both = cov_c[rp] & cov_c[rq]
k_all = ntc[rp] * np.int64(T) + ntc[rq]
d_all = np.linalg.norm(xyz_compact[rp[both]] - xyz_compact[rq[both]], axis=1)
k_b = k_all[both]
Hr = np.bincount(bin_of(d_all), minlength=NB).astype(np.float64)
log(f"双端有坐标边 {both.sum():,} ({both.mean():.2%}), 中位 d {np.median(d_all):.1f}um")

big_uk = np.sort(uk[cnt >= 20])
nbig = len(big_uk)
li = np.searchsorted(big_uk, k_b)
hit = (li < nbig) & (big_uk[np.clip(li, 0, nbig - 1)] == k_b)
c_real = np.bincount(li[hit] * NB + bin_of(d_all[hit]),
                     minlength=nbig * NB).reshape(nbig, NB).astype(np.float64)
log(f"大对 (cnt>=20): {nbig:,}, 覆盖真实边 {hit.sum():,} ({hit.sum()/both.sum():.2%} 有坐标边)")

# ---- 全局基线: 随机真实边, 目标换成同型有坐标随机节点 ----
M = 2_000_000
sel = rng.integers(0, len(d_all), M)
d0 = np.empty(M, np.float64)
eidx = np.flatnonzero(both)[sel]
tt = ntc[rq[eidx]]
src = xyz_compact[rp[eidx]]
for t_ in np.unique(tt):
    m = tt == t_
    if cnt_cov[t_] == 0:
        d0[m] = np.nan; continue
    jj = pool_draw(t_, int(m.sum()), a_in)
    d0[m] = np.linalg.norm(src[m] - xyz[jj], axis=1)
ok0 = ~np.isnan(d0)
Hbg = np.bincount(bin_of(d0[ok0]), minlength=NB).astype(np.float64)
log(f"全局基线采样 {ok0.sum():,}, 中位 d {np.median(d0[ok0]):.1f}um")

# ---- 逐大对基线: 64x64 采样 (分块向量化) ----
bs = (big_uk // T).astype(np.int64)
bt = (big_uk % T).astype(np.int64)
c_base = np.zeros((nbig, NB), np.float64)
CH = 512
for c0 in range(0, nbig, CH):
    c1 = min(c0 + CH, nbig)
    n = c1 - c0
    S = np.empty((n, NSAMP, 3), np.float32)
    G = np.empty((n, NSAMP, 3), np.float32)
    good = np.ones(n, bool)
    for r in range(n):
        ps, pt = bs[c0 + r], bt[c0 + r]
        if cnt_cov[ps] == 0 or cnt_cov[pt] == 0:
            good[r] = False; S[r] = 0; G[r] = 0; continue
        S[r] = xyz[pool_draw(ps, NSAMP, a_out)]
        G[r] = xyz[pool_draw(pt, NSAMP, a_in)]
    dd = np.sqrt(((S[:, :, None, :] - G[:, None, :, :]) ** 2).sum(-1))  # (n,64,64)
    bb = bin_of(dd.ravel())
    cnts = np.bincount(np.repeat(np.arange(n), NSAMP * NSAMP) * NB + bb,
                       minlength=n * NB).reshape(n, NB)
    c_base[c0:c1][good] = cnts[good]
    if (c0 // CH) % 40 == 0:
        log(f"  基线 {c0:,}/{nbig:,}")
log("逐对基线完成")

# ---- 核: 平滑 -> 比值 -> 归一 -> 截断 ----
g_r = Hr / Hr.sum()
g_b = Hbg / Hbg.sum()

def make_w(cr, cb):
    """cr, cb: (m, NB) 计数 -> 归一化核 w 与 wmax"""
    sr = cr.sum(1, keepdims=True); sb = cb.sum(1, keepdims=True)
    pr = (cr + KAPPA * g_r) / (sr + KAPPA)
    pb = (cb + KAPPA * g_b) / (sb + KAPPA)
    bad = (sr.ravel() < 1) | (sb.ravel() < 1)
    w = pr / np.maximum(pb, 1e-12)
    nrm = (w * pb).sum(1, keepdims=True)
    w = w / np.maximum(nrm, 1e-12)
    w = np.minimum(w, WCAP)
    w = w / np.maximum((w * pb).sum(1, keepdims=True), 1e-12)  # 截断后重归一
    w[bad] = (g_r / np.maximum(g_b, 1e-12) / 1.0)[None, :]     # 退化对 -> 全局核形状
    if bad.any():
        wg = g_r / np.maximum(g_b, 1e-12)
        wg = wg / (wg * g_b).sum()
        w[bad] = np.minimum(wg, WCAP)
    return w.astype(np.float32), w.max(1).astype(np.float32), bad

wbig, wmax_big, bad_big = make_w(c_real, c_base)
wg = g_r / np.maximum(g_b, 1e-12)
wg = wg / (wg * g_b).sum()
w_glob = np.minimum(wg, WCAP).astype(np.float32)
w_glob /= (w_glob * g_b).sum()
wmax_glob = float(w_glob.max())

np.savez(ROOT + "spatial_kernel.npz", bins=BINS, big_uk=big_uk,
         wbig=wbig, wmax_big=wmax_big, w_glob=w_glob, wmax_glob=wmax_glob)

# ---- 诊断 ----
exp_pdf = np.zeros(NB)   # 期望生成剖面 ~ Σ w·pb (按真实边质量加权, 大对) + 全局 (小对)
mass_big = c_real.sum(1)
pb_big = (c_base + KAPPA * g_b) / (c_base.sum(1, keepdims=True) + KAPPA)
exp_big = (wbig * pb_big)
exp_big /= np.maximum(exp_big.sum(1, keepdims=True), 1e-12)
mass_small = float(both.sum() - hit.sum())
exp_pdf = (exp_big * mass_big[:, None]).sum(0) + w_glob * ((w_glob * g_b) / (w_glob * g_b).sum()) * mass_small
exp_pdf /= exp_pdf.sum()
cdf_r = np.cumsum(Hr / Hr.sum()); cdf_e = np.cumsum(exp_pdf)
med = lambda cdf: float(BINS[:-1][np.searchsorted(cdf, 0.5)])
diag = {
    "nodes": int(N), "soma_cov": float(cov.mean()),
    "edges_both_coord": int(both.sum()),
    "real_median_um": float(np.median(d_all)),
    "base_median_um": float(np.median(d0[ok0])),
    "expected_gen_median_um": med(cdf_e),
    "real_median_binned_um": med(cdf_r),
    "nbig": int(nbig), "big_edge_share": float(hit.sum() / both.sum()),
    "degenerate_pairs": int(bad_big.sum()),
    "wmax_mean": float(wmax_big.mean()), "wmax_p90": float(np.percentile(wmax_big, 90)),
    "wmax_glob": wmax_glob, "WCAP": WCAP, "KAPPA": KAPPA,
    "expected_accept_rate": float((1.0 / np.maximum(wmax_big, 1)).mean()),
    "bins": BINS.round(2).tolist(),
    "Hr": Hr.astype(int).tolist(), "Hbg": Hbg.astype(int).tolist(),
}
json.dump(diag, open(ROOT + "spatial_kernel_diag.json", "w"), ensure_ascii=False, indent=1)
log(f"wmax 均值 {diag['wmax_mean']:.2f} (p90 {diag['wmax_p90']:.2f}), 期望接受率 {diag['expected_accept_rate']:.3f}")
log(f"中位距离: 真实 {diag['real_median_um']:.1f} 基线 {diag['base_median_um']:.1f} -> 期望生成 {diag['expected_gen_median_um']:.1f} um")
log("完成")
