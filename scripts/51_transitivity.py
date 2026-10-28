#!/usr/bin/env python3
# 全局传递性直接无偏测量: 均匀楔采样 -> 闭口比例
# 用法: python3 51_transitivity.py real|genv4|genv42
import sys, gc, time
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器/"
WHICH = sys.argv[1]
NSAM = 6_000_000

def load(which):
    if which == "real":
        OUT = ROOT + "data_census/"
        node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
        pre = np.fromfile(OUT + "e_pre.i32", dtype=np.int32)
        post = np.fromfile(OUT + "e_post.i32", dtype=np.int32)
        tm = node_type >= 0
        remap = np.cumsum(tm, dtype=np.int64) - 1
        return remap[pre].astype(np.int32), remap[post].astype(np.int32), int(tm.sum())
    elif which == "genv4":
        p = "/tmp/flydata/gen/genv4_S1"
    elif which == "genv42":
        p = "/tmp/flydata/gen/genv42_S1"
    else:
        p = which  # 直接给完整前缀路径
    g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
    N = int(g['type_count'].astype(np.int64).sum())
    return (np.fromfile(p + "_pre.i32", dtype=np.int32),
            np.fromfile(p + "_post.i32", dtype=np.int32), N)

pre, post, N = load(WHICH)
t0 = time.time()
mn = np.minimum(pre, post).astype(np.int64)
mx = np.maximum(pre, post).astype(np.int64)
keep = mn != mx
und = mn[keep] * N + mx[keep]
del mn, mx, keep, pre, post; gc.collect()
und.sort()
n_u = np.unique(und)          # 排序无向对表 (供存在性二分)
del und; gc.collect()
# 无向 CSR
dst = (n_u % N) * N + (n_u // N)
key = np.r_[n_u, dst]
del dst; gc.collect()
key.sort()
nbr = (key % N).astype(np.int32)
bounds = np.searchsorted(key, np.arange(N + 1, dtype=np.int64) * N)
deg = np.diff(bounds)
off = bounds
del key, bounds; gc.collect()
print(f"{WHICH}: CSR built ({time.time()-t0:.0f}s), und_edges={len(n_u):,}", flush=True)

rng = np.random.default_rng(7)
cw = deg * (deg - 1) // 2
Wtot = int(cw.sum())
cdf = np.cumsum(cw)
u = np.searchsorted(cdf, rng.integers(0, Wtot, NSAM))
dg = deg[u]
i1 = rng.integers(0, dg)
i2 = (i1 + 1 + rng.integers(0, dg - 1)) % dg
v = nbr[off[u] + i1].astype(np.int64)
w = nbr[off[u] + i2].astype(np.int64)
mnn = np.minimum(v, w); mxx = np.maximum(v, w)
kk = mnn * N + mxx
i = np.searchsorted(n_u, kk)
i = np.clip(i, 0, len(n_u) - 1)
closed = n_u[i] == kk
T = float(closed.mean())
print(f"{WHICH}: global transitivity = {T:.4f}  (se={np.sqrt(T*(1-T)/NSAM):.5f}, NSAM={NSAM})")
