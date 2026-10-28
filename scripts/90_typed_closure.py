#!/usr/bin/env python3
"""90_typed_closure.py — γ 缺口的型结构分解: 闭合率按端点 superclass 对 / 三元组
对每个图: 楔形采样 (node-uniform), 记录 (sc_center, sc_v, sc_w) 与闭合;
输出闭合率矩阵 real vs gen 及比值剖面, 判断超额是否均匀摊派。
用法: python3 90_typed_closure.py <out.json> [gen_prefix]
"""
import numpy as np, json, sys, time
from scipy.sparse import csr_matrix
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
sc_names = np.load(D + "sc_names.npy", allow_pickle=True)
N = len(sc); K = int(sc.max()) + 1
print(f"N={N:,} K={K} superclasses", flush=True)

def wedge_table(pre, post, nsamp=400_000, seed=0):
    lo = np.minimum(pre, post).astype(np.int64); hi = np.maximum(pre, post).astype(np.int64)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N)); A.data[:] = 1
    ip, ix = A.indptr, A.indices
    deg = np.diff(ip); cand = np.where(deg >= 2)[0]
    rng = np.random.default_rng(seed)
    tot = np.zeros((K, K, K), np.int32); cls = np.zeros((K, K, K), np.int32)
    t = 0
    while t < nsamp:
        u = int(cand[rng.integers(0, len(cand))])
        ns_ = ix[ip[u]:ip[u + 1]]
        if len(ns_) < 2: continue
        v, w_ = ns_[rng.integers(0, len(ns_), 2)]
        if v == w_: continue
        a_, b_ = (v, w_) if v < w_ else (w_, v)
        hit = np.searchsorted(ix[ip[a_]:ip[a_ + 1]], b_)
        cl = ip[a_] + hit < ip[a_ + 1] and ix[ip[a_] + hit] == b_
        su, sv, sw = sc[u], sc[v], sc[w_]
        e1, e2 = min(sv, sw), max(sv, sw)
        tot[su, e1, e2] += 1; cls[su, e1, e2] += cl
        t += 1
    return tot, cls

tot_r, cls_r = wedge_table(ep, eq)
out = {"K": K, "sc_names": [str(x) for x in sc_names],
       "T_real": float(cls_r.sum() / tot_r.sum())}
# 端点对聚合 (对中心 sc 求和)
pair_r = cls_r.sum(0)   # e1<=e2 已对称化, 上三角
npair_r = tot_r.sum(0)

if len(sys.argv) > 2:
    gp = np.fromfile(sys.argv[2] + "_pre.i32", dtype=np.int32)
    gq = np.fromfile(sys.argv[2] + "_post.i32", dtype=np.int32)
    tot_g, cls_g = wedge_table(gp, gq)
    pair_g = cls_g.sum(0); npair_g = tot_g.sum(0)
    out["T_gen"] = float(cls_g.sum() / tot_g.sum())
    # 端点对剖面: 采样量足够的对
    rows = []
    ok = (npair_r >= 500) & (npair_g >= 500)
    for a in range(K):
        for b in range(a, K):
            if not ok[a, b]: continue
            rows.append({"pair": [str(sc_names[a]), str(sc_names[b])],
                         "n_real": int(npair_r[a, b]),
                         "closure_real": float(pair_r[a, b] / npair_r[a, b]),
                         "closure_gen": float(pair_g[a, b] / npair_g[a, b]),
                         "ratio": float(pair_r[a, b] / max(pair_g[a, b], 1e-9))})
    rows.sort(key=lambda r: -r["n_real"])
    out["endpoint_pairs"] = rows
    # 中心×端点三元组 top
    trip = []
    ok3 = (tot_r >= 300) & (tot_g >= 300)
    ii = np.argwhere(ok3)
    for su, e1, e2 in ii:
        trip.append({"triple": [str(sc_names[su]), str(sc_names[e1]), str(sc_names[e2])],
                     "n_real": int(tot_r[su, e1, e2]),
                     "closure_real": float(cls_r[su, e1, e2] / tot_r[su, e1, e2]),
                     "closure_gen": float(cls_g[su, e1, e2] / tot_g[su, e1, e2]),
                     "ratio": float(cls_r[su, e1, e2] / max(cls_g[su, e1, e2], 1e-9))})
    trip.sort(key=lambda r: -r["n_real"])
    out["triplets"] = trip[:40]
    # 缺口加权: 哪些端点对贡献了绝对缺口
    gap = (pair_r - pair_g)  # 每对闭合楔形差
    frac = []
    G = float(gap.sum())
    for a in range(K):
        for b in range(a, K):
            if npair_r[a, b] >= 200:
                frac.append({"pair": [str(sc_names[a]), str(sc_names[b])],
                             "gap_share": float(gap[a, b] / G)})
    frac.sort(key=lambda r: -r["gap_share"])
    out["gap_shares"] = frac[:20]
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(f"SAVED {sys.argv[1]}  {time.time()-t0:.0f}s  T_real={out['T_real']:.4f}" +
      (f" T_gen={out['T_gen']:.4f}" if "T_gen" in out else ""))
