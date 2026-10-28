#!/usr/bin/env python3
# 60_wedge_ceiling.py — 楔机制天花板诊断: 2-跳可达对的公共邻居数 C(i,j) 分布
# i 均匀(有度节点), u~U(N(i)), j~U(N(u)); C=|N(i)∩N(j)| 用无向对表全局 searchsorted 向量化
# 另测 max-of-8 (同一 i 抽 8 次取最大 C) —— "多抽择优"换边方案的收益上限
import numpy as np, json, time

ROOT = "/mnt/agents/output/果蝇生成器/"
NS = 300_000
K = 8
rng = np.random.default_rng(7)

def load_real():
    nt = np.fromfile(ROOT + "data_census/node_type.i32", dtype=np.int32)
    pre = np.fromfile(ROOT + "data_census/e_pre.i32", dtype=np.int32)
    post = np.fromfile(ROOT + "data_census/e_post.i32", dtype=np.int32)
    tm = nt >= 0
    remap = np.cumsum(tm, dtype=np.int64) - 1
    return remap[pre].astype(np.int32), remap[post].astype(np.int32), int(tm.sum())

def load_gen(p):
    pre = np.fromfile(p + "_pre.i32", dtype=np.int32)
    post = np.fromfile(p + "_post.i32", dtype=np.int32)
    N = int(max(pre.max(), post.max())) + 1
    return pre, post, N

def build_und(pre, post, N):
    """返回: off/nbr 无向 CSR + ptab 排序无向对键表 + deg"""
    mn = np.minimum(pre, post).astype(np.int64)
    mx = np.maximum(pre, post).astype(np.int64)
    keep = mn != mx
    key = np.unique(mn[keep] * N + mx[keep])
    a = (key // N).astype(np.int32); b = (key % N).astype(np.int32)
    src = np.concatenate([a, b]); dst = np.concatenate([b, a])
    deg = np.bincount(src, minlength=N)
    off = np.zeros(N + 1, np.int64); np.cumsum(deg, out=off[1:])
    od = np.argsort(src, kind="stable")
    nbr = dst[od]
    return off, nbr, key, deg

def diag(name, off, nbr, ptab, deg):
    t0 = time.time()
    cand = np.flatnonzero(deg > 0)
    ii = cand[rng.integers(0, len(cand), NS)]
    C1 = np.zeros(NS, np.int32)
    Cmax = np.zeros(NS, np.int32)
    B = 20_000
    for k in range(K):
        for s in range(0, NS, B):
            ib = ii[s:s + B]
            lo = off[ib]; d = off[ib + 1] - lo
            uu = nbr[lo + (rng.random(len(ib)) * d).astype(np.int64)]
            lo2 = off[uu]; d2 = off[uu + 1] - lo2
            jj = nbr[lo2 + (rng.random(len(ib)) * d2).astype(np.int64)]
            # 展开 N(i): own=行号, x=邻居; 键 = (min(j,x), max(j,x))
            cnt = d.astype(np.int64)
            own = np.repeat(np.arange(len(ib)), cnt)
            x = np.concatenate([nbr[off[i]:off[i + 1]] for i in ib])
            jx = np.repeat(jj, cnt)
            a = np.minimum(jx, x); b = np.maximum(jx, x)
            kk = a * (np.int64(len(off) - 1)) + b
            pos = np.searchsorted(ptab, kk)
            pos = np.clip(pos, 0, len(ptab) - 1)
            hit = ptab[pos] == kk
            Ck = np.bincount(own[hit], minlength=len(ib)).astype(np.int32)
            Ck[jj == ib] = 0
            if k == 0:
                C1[s:s + B] = Ck
            Cmax[s:s + B] = np.maximum(Cmax[s:s + B], Ck)
    for lab, C in (("单次抽", C1), ("max-of-8", Cmax)):
        q = np.percentile(C, [50, 90, 99, 99.9])
        print(f"  {name} {lab}: mean={C.mean():.2f} p50={q[0]:.0f} p90={q[1]:.0f} p99={q[2]:.0f} "
              f"p99.9={q[3]:.0f} P(>=5)={np.mean(C >= 5):.4f} P(>=20)={np.mean(C >= 20):.5f} max={C.max()}")
    print(f"  ({time.time() - t0:.0f}s)")
    return {"C1_mean": float(C1.mean()), "Cmax8_mean": float(Cmax.mean()),
            "P5": float(np.mean(C1 >= 5)), "P20": float(np.mean(C1 >= 20))}

out = {}
pre, post, N = load_real()
off, nbr, ptab, deg = build_und(pre, post, N)
print(f"真实: N={N:,} 无向边={len(ptab):,}")
out["real"] = diag("真实", off, nbr, ptab, deg)
del pre, post, off, nbr, ptab, deg
pre, post, N = load_gen("/tmp/flydata/gen/genv6p4_S1")
off, nbr, ptab, deg = build_und(pre, post, N)
print(f"v6p4: N={N:,} 无向边={len(ptab):,}")
out["v6p4"] = diag("v6p4", off, nbr, ptab, deg)
json.dump(out, open(ROOT + "wedge_ceiling.json", "w"), indent=1)
print("SAVED wedge_ceiling.json")
