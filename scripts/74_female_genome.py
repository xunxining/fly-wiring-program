#!/usr/bin/env python3
"""74_female_genome.py — 雌性基因组 + CELLQ + ACT + 空间核 (与雄性 v6 定义一致)
输入: data_female/{node_type,node_ids,offsets}.i32/i64, e_pre/e_post/e_w.i32, xyz.f32, type_names.npy
      data_female/classification.csv.gz (super_class -> node_sc)
输出:
  data_female/genome_f.npz        uk, cnt, type_count, sc_types
  data_female/cq_cell/cq_t/cq_cnt.i32   逐细胞配额 (细胞=基因组序)
  data_female/act_in.f32
  data_female/spatial_kernel_f.npz      bins, big_uk, wbig, wmax_big, w_glob, wmax_glob
  data_female/kernel_diag_f.json
"""
import numpy as np, pandas as pd, json, time, gc, resource, os, sys

D = "/mnt/agents/output/果蝇生成器/data_female/"
XYZ_PATH = os.environ.get("XYZ", D + "xyz.f32")
SKIP_SPATIAL = os.environ.get("SKIP_SPATIAL", "0") == "1"
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
rng = np.random.default_rng(7)

node_type = np.fromfile(D + "node_type.i32", dtype=np.int32).astype(np.int64)
types = np.load(D + "type_names.npy", allow_pickle=True)
T = len(types); N = len(node_type)
tc = np.bincount(node_type, minlength=T)
toff = np.concatenate([[0], np.cumsum(tc)])
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
w = np.fromfile(D + "e_w.i32", dtype=np.int32)
E = len(ep)
log(f"N={N:,} T={T:,} E={E:,} rss={rss()}MB")

# ---------- super_class (for weight pools) ----------
cls = pd.read_csv(D + "classification.csv.gz", usecols=["root_id", "super_class"])
node_ids = np.fromfile(D + "node_ids.i64", dtype=np.int64)
scs = np.sort(cls.super_class.dropna().unique())
scmap = {s: i for i, s in enumerate(scs)}
cls["sc"] = cls.super_class.map(scmap)
id2row = pd.Series(np.arange(len(cls)), index=cls.root_id.values)
pos = id2row.reindex(node_ids).to_numpy()
node_sc = np.full(N, -1, np.int32)
ok = ~np.isnan(pos)
node_sc[ok] = cls.sc.values[pos[ok].astype(int)]
node_sc.tofile(D + "node_sc.i32")
np.save(D + "sc_names.npy", scs.astype(object))
log("super_class mapped:", ok.mean().round(3))

# ---------- 基因组: 类型对配额 ----------
k = node_type[ep] * T + node_type[eq]
uk, cnt = np.unique(k, return_counts=True)
np.savez(D + "genome_f.npz", uk=uk, cnt=cnt, type_count=tc, sc_types=scs)
log(f"基因组对 {len(uk):,}, 覆盖率 {cnt.sum()/E:.3f}")

# ---------- CELLQ: 逐细胞配额 ----------
kc = ep.astype(np.int64) * T + node_type[eq]
uq, st = np.unique(kc, return_index=True)
st = np.append(st, len(kc))
cc = np.diff(st)   # 每个 (cell, target-type) 的边数 (kc 已排序, unique 保序)
cq_cell = (uq // T).astype(np.int32)
cq_t = (uq % T).astype(np.int32)
cq_cnt = cc.astype(np.int32)
assert cq_cnt.sum() == E
cq_cell.tofile(D + "cq_cell.i32"); cq_t.tofile(D + "cq_t.i32"); cq_cnt.tofile(D + "cq_cnt.i32")
log(f"CELLQ 记录 {len(uq):,}, 平均 {E/len(uq):.2f} 边/记录, rss={rss()}MB")

# ---------- ACT ----------
odr = np.bincount(ep, minlength=N).astype(np.float64)
idr = np.bincount(eq, minlength=N).astype(np.float64)
tm_o = np.maximum(np.bincount(node_type, weights=odr, minlength=T) / np.maximum(tc, 1), 1e-9)
tm_i = np.maximum(np.bincount(node_type, weights=idr, minlength=T) / np.maximum(tc, 1), 1e-9)
a_out = np.clip(odr / tm_o[node_type], 0.1, 10.0)
a_in = np.clip(idr / tm_i[node_type], 0.1, 10.0)
a_in.astype(np.float32).tofile(D + "act_in.f32")
log(f"ACT: a_in 中位 {np.median(a_in):.2f} p90 {np.percentile(a_in,90):.2f}")

# ---------- 空间核 (66 复刻, 全坐标覆盖) ----------
if SKIP_SPATIAL:
    log("SKIP_SPATIAL=1: 空间核部分跳过")
    sys.exit(0)
NB = 24
BINS = np.geomspace(0.5, 200.0, NB + 1)
KAPPA, WCAP, NSAMP = 25.0, 24.0, 64
xyz = np.fromfile(XYZ_PATH, dtype=np.float32).reshape(N, 3).astype(np.float64)
cov = ~np.isnan(xyz[:, 0])
log(f"XYZ 来源: {XYZ_PATH}")
log(f"坐标覆盖 {cov.mean():.3f}")
cov_idx = np.flatnonzero(cov)
cnt_cov = np.bincount(node_type[cov_idx], minlength=T)
order_cov = cov_idx[np.argsort(node_type[cov_idx], kind="stable")]
poff = np.concatenate([[0], np.cumsum(cnt_cov)])
_cw = {}
def pool_draw(t_, m, wt_all):
    if t_ not in _cw:
        pool = order_cov[poff[t_]:poff[t_] + cnt_cov[t_]]
        _cw[t_] = (pool, np.cumsum(wt_all[pool]))
    pool, cwt = _cw[t_]
    if len(pool) == 0: return None
    return pool[np.searchsorted(cwt, rng.random(m) * cwt[-1])]
def bin_of(d): return np.clip(np.searchsorted(BINS, d, side="right") - 1, 0, NB - 1)

d_all = np.linalg.norm(xyz[ep] - xyz[eq], axis=1)
Hr = np.bincount(bin_of(d_all), minlength=NB).astype(np.float64)
log(f"边距离中位 {np.median(d_all):.2f}um")

big_uk = np.sort(uk[cnt >= 20]); nbig = len(big_uk)
li = np.searchsorted(big_uk, k)
hit = (li < nbig) & (big_uk[np.clip(li, 0, nbig - 1)] == k)
c_real = np.bincount(li[hit] * NB + bin_of(d_all[hit]), minlength=nbig * NB).reshape(nbig, NB).astype(np.float64)
log(f"大对 {nbig:,}, 覆盖边 {hit.mean():.2%}, rss={rss()}MB")

# 全局基线
M = 2_000_000
sel = rng.integers(0, E, M)
tt = node_type[eq[sel]]
src = xyz[ep[sel]]
d0 = np.empty(M)
for t_ in np.unique(tt):
    m = tt == t_
    jj = pool_draw(int(t_), int(m.sum()), a_in)
    d0[m] = np.linalg.norm(src[m] - xyz[jj], axis=1)
Hbg = np.bincount(bin_of(d0), minlength=NB).astype(np.float64)
log(f"基线中位 {np.median(d0):.2f}um")

# 逐大对基线
bs = (big_uk // T).astype(np.int64); bt = (big_uk % T).astype(np.int64)
c_base = np.zeros((nbig, NB))
CH = 512
for c0 in range(0, nbig, CH):
    c1 = min(c0 + CH, nbig); n = c1 - c0
    S = np.empty((n, NSAMP, 3)); G = np.empty((n, NSAMP, 3))
    for r in range(n):
        S[r] = xyz[pool_draw(int(bs[c0 + r]), NSAMP, a_out)]
        G[r] = xyz[pool_draw(int(bt[c0 + r]), NSAMP, a_in)]
    dd = np.sqrt(((S[:, :, None, :] - G[:, None, :, :]) ** 2).sum(-1))
    bb = bin_of(dd.ravel())
    c_base[c0:c1] = np.bincount(np.repeat(np.arange(n), NSAMP * NSAMP) * NB + bb,
                                minlength=n * NB).reshape(n, NB)
    if (c0 // CH) % 50 == 0: log(f"  基线 {c0:,}/{nbig:,} rss={rss()}MB")
log("逐对基线完成")

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
np.savez(D + "spatial_kernel_f.npz", bins=BINS, big_uk=big_uk, wbig=wbig,
         wmax_big=wmax_big, w_glob=w_glob, wmax_glob=wmax_glob)
diag = {"edges": int(E), "real_median_um": float(np.median(d_all)),
        "base_median_um": float(np.median(d0)),
        "nbig": int(nbig), "big_edge_share": float(hit.mean()),
        "degenerate_pairs": int(bad.sum()),
        "wmax_mean": float(wmax_big.mean()), "wmax_p90": float(np.percentile(wmax_big, 90)),
        "wmax_glob": wmax_glob,
        "bins": BINS.round(3).tolist(), "Hr": Hr.astype(int).tolist(), "Hbg": Hbg.astype(int).tolist(),
        "w_glob": w_glob.round(4).tolist()}
json.dump(diag, open(D + "kernel_diag_f.json", "w"), indent=1)
log(f"核: wmax 均值 {diag['wmax_mean']:.2f}, 真实 {diag['real_median_um']:.1f} vs 基线 {diag['base_median_um']:.1f} um")
log("GENOME DONE")
