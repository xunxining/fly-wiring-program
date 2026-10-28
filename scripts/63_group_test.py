#!/usr/bin/env python3
# 63_group_test.py — group (13.7k 细空间组) 是否闭三角的正确尺度 (§21 机制 v5 前置判决)
import numpy as np, pandas as pd, json, time

ROOT = "/mnt/agents/output/果蝇生成器/"
rng = np.random.default_rng(3)

ann = pd.read_feather(ROOT + "data/body-annotations.feather", columns=["bodyId", "group", "superclass"])
node_ids = np.fromfile(ROOT + "data_census/node_ids.i64", dtype=np.int64)
nt = np.fromfile(ROOT + "data_census/node_type.i32", dtype=np.int32)
tm = nt >= 0
nid_valid = node_ids[tm] if len(node_ids) == len(nt) else node_ids
N = int(tm.sum())
print(f"node_ids: {len(node_ids):,}, 有效节点 {N:,}")

amap = pd.Series(ann["group"].values, index=ann["bodyId"].values)
grp = amap.reindex(nid_valid).to_numpy()
# 校验映射: superclass 重算 vs node_sc_gen.i32
smap = pd.Series(ann["superclass"].fillna("UNK").values, index=ann["bodyId"].values)
sc_check = smap.reindex(nid_valid).to_numpy()
sc_gen = np.fromfile(ROOT + "node_sc_gen.i32", dtype=np.int32)
scv = pd.Categorical(sc_check).codes.astype(np.int32)
print("superclass 一致率(校验映射): %.4f" % (pd.Series(sc_check).fillna("UNK").to_numpy() ==
      pd.Series(sc_gen).fillna("UNK").to_numpy() if sc_gen.dtype == object else -1))

gcov = pd.notna(grp)
print(f"group 覆盖率: {gcov.mean():.3f}, 唯一组 {pd.Series(grp[gcov]).nunique():,}")
gcat = pd.Categorical(grp)  # NaN -> -1
g = gcat.codes.astype(np.int32)

remap = np.cumsum(tm, dtype=np.int64) - 1
pre = np.fromfile(ROOT + "data_census/e_pre.i32", dtype=np.int32)
post = np.fromfile(ROOT + "data_census/e_post.i32", dtype=np.int32)
rp = remap[pre].astype(np.int32); rq = remap[post].astype(np.int32)
gtype = np.repeat(np.arange(11751), np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)["type_count"].astype(np.int64))

same_g = (g[rp] == g[rq]) & (g[rp] >= 0)
print(f"同group边占比: 真实 {same_g.mean():.4f}")

# 配额保持基线: 每条真实边 (i,j), j' ~ 同型均匀, P(g(j')=g(i))
M = 400_000
idx = rng.integers(0, len(rp), M)
tt = gtype[rq[idx]]
jrand = np.empty(M, np.int32)
toff = np.concatenate([[0], np.cumsum(np.bincount(gtype, minlength=11751))])
for t_ in np.unique(tt):
    m = tt == t_
    jrand[m] = rng.integers(toff[t_], toff[t_ + 1], m.sum())
base_g = (g[rp[idx]] == g[jrand]) & (g[rp[idx]] >= 0)
print(f"同group基线(配额均匀): {base_g.mean():.4f}  → 富集 {same_g.mean()/max(base_g.mean(),1e-9):.2f}x")

# 边上 C(i,j) 按同group分层
mn = np.minimum(rp, rq).astype(np.int64); mx = np.maximum(rp, rq).astype(np.int64)
keep = mn != mx
ptab = np.unique(mn[keep] * N + mx[keep])
Nn = np.int64(N)
a = (ptab // Nn).astype(np.int32); b = (ptab - a.astype(np.int64) * Nn).astype(np.int32)
src = np.concatenate([a, b]); dst = np.concatenate([b, a])
deg = np.bincount(src, minlength=N)
off = np.zeros(N + 1, np.int64); np.cumsum(deg, out=off[1:])
od = np.argsort(src, kind="stable")
nbr = dst[od]
del src, dst, od, a, b, mn, mx, keep

def Cund(xs, ys):
    out = np.zeros(len(xs), np.int32)
    B = 8000
    for s0 in range(0, len(xs), B):
        x = xs[s0:s0 + B]; y = ys[s0:s0 + B].astype(np.int64)
        lo = off[x]; d = (off[x + 1] - lo).astype(np.int64)
        tot = int(d.sum())
        bb = np.repeat(lo - np.concatenate([[0], np.cumsum(d)[:-1]]), d)
        u = nbr[bb + np.arange(tot)].astype(np.int64)
        own = np.repeat(np.arange(len(x)), d)
        yr = y[own]
        kk = np.minimum(u, yr) * Nn + np.maximum(u, yr)
        p = np.searchsorted(ptab, kk); p = np.clip(p, 0, len(ptab) - 1)
        hit = ptab[p] == kk
        out[s0:s0 + B] = np.bincount(own[hit], minlength=len(x))
    return out

t0 = time.time()
ei = rng.integers(0, len(rp), 300_000)
C = Cund(rp[ei], rq[ei])
sg = same_g[ei]
print(f"边上 C: 同group均值 {C[sg].mean():.1f} (n={sg.sum():,}) vs 跨group {C[~sg].mean():.1f} (n={(~sg).sum():,})")

# 楔闭合率按三节点同组性分层
ui = rng.integers(0, N, 400_000)
lo = off[ui]; d = off[ui + 1] - lo
ok = d >= 2
ui, lo, d = ui[ok], lo[ok], d[ok]
r1 = (rng.random(len(ui)) * d).astype(np.int64)
r2 = (rng.random(len(ui)) * (d - 1)).astype(np.int64)
r2 = r2 + (r2 >= r1)
i1 = nbr[lo + r1]; j1 = nbr[lo + r2]
kk = np.minimum(i1, j1).astype(np.int64) * Nn + np.maximum(i1, j1).astype(np.int64)
p = np.searchsorted(ptab, kk); p = np.clip(p, 0, len(ptab) - 1)
closed = ptab[p] == kk
gu, gi, gj = g[ui], g[i1], g[j1]
allsame = (gu >= 0) & (gu == gi) & (gi == gj)
ijsame = (gi >= 0) & (gi == gj)
for lab, m in [("三节点同组", allsame), ("仅i,j同组", ijsame & ~allsame), ("i,j不同组", ~ijsame)]:
    print(f"楔闭合率 | {lab}: {closed[m].mean():.4f} (n={m.sum():,})")
print(f"({time.time()-t0:.0f}s)")
