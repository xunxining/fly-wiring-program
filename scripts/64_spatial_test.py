#!/usr/bin/env python3
# 64_spatial_test.py — soma 距离是否解释邻域重叠 (§21 机制 v5 前置判决 2)
import numpy as np, pandas as pd, time

ROOT = "/mnt/agents/output/果蝇生成器/"
rng = np.random.default_rng(5)

ann = pd.read_feather(ROOT + "data/body-annotations.feather", columns=["bodyId", "somaLocation"])
node_ids = np.fromfile(ROOT + "data_census/node_ids.i64", dtype=np.int64)
nt = np.fromfile(ROOT + "data_census/node_type.i32", dtype=np.int32)
tm = nt >= 0
nid_valid = node_ids[tm] if len(node_ids) == len(nt) else node_ids
N = int(tm.sum())
xyz = np.full((N, 3), np.nan, np.float64)
sl = ann["somaLocation"]
has = sl.notna().to_numpy()
amp = pd.Series(np.arange(len(ann)), index=ann["bodyId"].values)
pos_idx = amp.reindex(nid_valid).to_numpy()
okn = ~np.isnan(pos_idx)
loc_arr = np.full((len(ann), 3), np.nan)
loc_arr[has] = np.vstack(sl[has].to_numpy())
xyz[okn] = loc_arr[pos_idx[okn].astype(int)]
xyz = xyz.astype(np.float32)
cov = ~np.isnan(xyz[:, 0])
print(f"soma 覆盖: {cov.mean():.3f} ({cov.sum():,}/{N:,})", flush=True)
del ann, sl, loc_arr, amp, has, pos_idx, okn
import gc; gc.collect()

remap = np.cumsum(tm, dtype=np.int64) - 1
pre = np.fromfile(ROOT + "data_census/e_pre.i32", dtype=np.int32)
post = np.fromfile(ROOT + "data_census/e_post.i32", dtype=np.int32)
rp = remap[pre].astype(np.int32); rq = remap[post].astype(np.int32)
gtype = np.repeat(np.arange(11751), np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)["type_count"].astype(np.int64))

both = cov[rp] & cov[rq]
d = np.linalg.norm(xyz[rp[both]] - xyz[rq[both]], axis=1)
print(f"双端有坐标的边: {both.sum():,} ({both.mean():.2%})")

# 基线: 同型均匀 j'
M = 300_000
idx = np.flatnonzero(both)
sel = idx[rng.integers(0, len(idx), M)]
tt = gtype[rq[sel]]
toff = np.concatenate([[0], np.cumsum(np.bincount(gtype, minlength=11751))])
jrand = np.empty(M, np.int64)
for t_ in np.unique(tt):
    m = tt == t_
    jrand[m] = rng.integers(toff[t_], toff[t_ + 1], m.sum())
okb = cov[jrand]
d0 = np.linalg.norm(xyz[rp[sel][okb]] - xyz[jrand[okb]], axis=1)

rs = [2, 5, 10, 20, 40, 80, 160]  # 单位: 坐标原始单位 (约 nm/体素?)
print("距离富集曲线 (P(d<r) 真实 vs 基线):")
for r in rs:
    pr = (d < r * 1000).mean()  # 坐标看起来是 nm 量级 (37124 等), 按 um 换 r*1000
    pb = (d0 < r * 1000).mean()
    print(f"  <{r}um: 真实 {pr:.4f} 基线 {pb:.4f} 富集 {pr/max(pb,1e-9):5.2f}x")
print(f"中位距离: 真实 {np.median(d)/1000:.1f}um 基线 {np.median(d0)/1000:.1f}um")

# 边上 C 随距离衰减
mn = np.minimum(rp, rq); mx = np.maximum(rp, rq)
keep = mn != mx
key = mn[keep].astype(np.int64) * N + mx[keep]
del mn, mx, keep
ptab = np.unique(key)
del key
Nn = np.int64(N)
a = (ptab // Nn).astype(np.int32); b = (ptab - a.astype(np.int64) * Nn).astype(np.int32)
# 双段 CSR (免 argsort): a 有序 => off_a/nbr_a=b; b 重新计数 => off_b/nbr_b=a (b 也有序, 因 ptab 排序保证同 a 内 b 升序, 但跨 a 的 b 全局无序 — 需按 b 重排)
offa = np.zeros(N + 1, np.int64); np.cumsum(np.bincount(a, minlength=N), out=offa[1:])
nbra = b  # 按 a 升序, 段内 b 升序
obb = np.zeros(N + 1, np.int64); np.cumsum(np.bincount(b, minlength=N), out=obb[1:])
ordb = np.argsort(b, kind="stable")
nbrb = a[ordb]
del ordb, a, b

def Cund(xs, ys):
    out = np.zeros(len(xs), np.int32)
    B = 8000
    for s0 in range(0, len(xs), B):
        x = xs[s0:s0 + B]; y = ys[s0:s0 + B].astype(np.int64)
        d1 = (offa[x + 1] - offa[x]).astype(np.int64)
        d2 = (obb[x + 1] - obb[x]).astype(np.int64)
        dd = d1 + d2
        tot = int(dd.sum())
        base = np.repeat(np.concatenate([[0], np.cumsum(dd)[:-1]]), dd)
        ar = np.arange(tot) - base
        own = np.repeat(np.arange(len(x)), dd)
        # 段内位置: 前 d1 个在 offa 段, 其余在 obb 段
        xr = np.repeat(x, dd)
        d1r = np.repeat(d1, dd)
        ina = ar < d1r
        pos = np.empty(tot, np.int64)
        pos[ina] = offa[xr][ina] + ar[ina]
        pos[~ina] = obb[xr][~ina] + (ar[~ina] - d1r[~ina])
        u = np.empty(tot, np.int64)
        u[ina] = nbra[pos[ina]]
        u[~ina] = nbrb[pos[~ina]]
        yr = y[own]
        kk = np.minimum(u, yr) * Nn + np.maximum(u, yr)
        p = np.searchsorted(ptab, kk); p = np.clip(p, 0, len(ptab) - 1)
        hit = ptab[p] == kk
        out[s0:s0 + B] = np.bincount(own[hit], minlength=len(x))
    return out

ei = idx[rng.integers(0, len(idx), 150_000)]
C = Cund(rp[ei], rq[ei])
de = np.linalg.norm(xyz[rp[ei]] - xyz[rq[ei]], axis=1) / 1000
bins = [0, 5, 10, 20, 40, 80, 160, 1e9]
print("边上 C 随 soma 距离:")
for lo_, hi_ in zip(bins[:-1], bins[1:]):
    m = (de >= lo_) & (de < hi_)
    if m.sum():
        print(f"  {lo_:>3.0f}-{hi_:>4.0f}um: C均值 {C[m].mean():6.1f} (n={m.sum():,})")
