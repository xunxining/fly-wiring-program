#!/usr/bin/env python3
"""97_anchor_split.py — 坐标模糊对闭合测量的影响: 楔形闭合率按端点锚点状态分层
锚点 = column_assignment 直接标注的细胞 (坐标精确); 其余 = 传播坐标 (中位误差 1.7 列)
用法: python3 97_anchor_split.py <out.json> <gen_prefix>
"""
import numpy as np, json, sys
from scipy.sparse import csr_matrix
D = "/mnt/agents/output/果蝇生成器/data_female/"
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
z = np.load(D + "lat2q_f.npz"); PQ = z["pq"]; HEM = z["hem"]
N = len(sc); om = sc == 5
has = om & (HEM != 0)

# 锚点掩码
import pandas as pd
node_ids = np.fromfile(D + "node_ids.i64", dtype=np.int64)
order = np.argsort(node_ids); sids = node_ids[order]
ca = pd.read_csv(D + "column_assignment.csv.gz")
j = np.searchsorted(sids, ca.root_id.values)
okm = (j < len(sids)) & (sids[np.clip(j, 0, len(sids) - 1)] == ca.root_id.values)
anchor = np.zeros(N, bool); anchor[order[j[okm]]] = True

def run(pre, post, nsamp=1_000_000, seed=0):
    lo = np.minimum(pre, post).astype(np.int64); hi = np.maximum(pre, post).astype(np.int64)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N)); A.data[:] = 1
    ip, ix = A.indptr, A.indices
    deg = np.diff(ip); cand = np.where(deg >= 2)[0]
    rng = np.random.default_rng(seed)
    # (距离档近/远) x (锚点数 0/1/2)
    tot = np.zeros((2, 3), np.int64); cls = np.zeros((2, 3), np.int64)
    t = 0
    while t < nsamp:
        u = int(cand[rng.integers(0, len(cand))])
        ns_ = ix[ip[u]:ip[u + 1]]
        if len(ns_) < 2: continue
        v, w_ = ns_[rng.integers(0, len(ns_), 2)]
        if v == w_: continue
        if not (has[v] and has[w_]) or HEM[v] != HEM[w_]:
            t += 1; continue
        dd = float(np.hypot(PQ[v, 0] - PQ[w_, 0], PQ[v, 1] - PQ[w_, 1]))
        di = 0 if dd <= 1.0 else 1
        if di == 1 and dd > 3.0: t += 1; continue   # 只看 <=1 与 (1,3]
        na = int(anchor[v]) + int(anchor[w_])
        a_, b_ = (v, w_) if v < w_ else (w_, v)
        hit = np.searchsorted(ix[ip[a_]:ip[a_ + 1]], b_)
        cl = int(ip[a_] + hit < ip[a_ + 1] and ix[ip[a_] + hit] == b_)
        tot[di, na] += 1; cls[di, na] += cl
        t += 1
    return tot, cls

tot_r, cls_r = run(ep, eq)
gp = np.fromfile(sys.argv[2] + "_pre.i32", dtype=np.int32)
gq = np.fromfile(sys.argv[2] + "_post.i32", dtype=np.int32)
tot_g, cls_g = run(gp, gq)
out = {"rows": ["d<=1", "1<d<=3"], "cols": ["0 anchors", "1 anchor", "2 anchors"],
       "real_tot": tot_r.tolist(), "real_rate": (cls_r / np.maximum(tot_r, 1)).round(4).tolist(),
       "gen_tot": tot_g.tolist(), "gen_rate": (cls_g / np.maximum(tot_g, 1)).round(4).tolist()}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("            锚点数:  0      1      2")
for di, nm in enumerate(["d<=1 ", "1-3  "]):
    print(f"real {nm}", " ".join(f"{cls_r[di,k]/max(tot_r[di,k],1):.3f}" for k in range(3)),
          " | gen", " ".join(f"{cls_g[di,k]/max(tot_g[di,k],1):.3f}" for k in range(3)))
print("tot real", tot_r.tolist(), "\ntot gen ", tot_g.tolist())
