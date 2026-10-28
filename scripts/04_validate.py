#!/usr/bin/env python3
"""Stage 4: 验证 —— 生成图 vs 真实图的统计对比"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp, spearmanr
import json, os

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"

def log(*a): print(*a, flush=True)

# ---- 真实图(typed-typed) ----
pre = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(D + "e_post.i32", dtype=np.int32, mode="r")
w = np.memmap(D + "e_w.i32", dtype=np.int32, mode="r")
node_type = np.load(D + "node_type.npy")
N = len(node_type)
m = (node_type[pre] >= 0) & (node_type[post] >= 0)
rpre = pre[m]; rpost = post[m]; rw = np.asarray(w[m])
# 映射到 typed 节点空间
typed_nodes = np.flatnonzero(node_type >= 0)
remap = np.full(N, -1, dtype=np.int32)
remap[typed_nodes] = np.arange(len(typed_nodes), dtype=np.int32)
rpre = remap[rpre]; rpost = remap[rpost]
NT = len(typed_nodes)

# ---- 生成图 ----
gpre = np.memmap(OUT + "gen_S1_pre.i32", dtype=np.int32, mode="r")
gpost = np.memmap(OUT + "gen_S1_post.i32", dtype=np.int32, mode="r")
gw = np.memmap(OUT + "gen_S1_w.i32", dtype=np.int32, mode="r")

stats = {}
stats["real"] = {"nodes": NT, "edges": int(len(rpre)), "synapses": int(rw.sum())}
stats["gen"] = {"nodes": NT, "edges": int(len(gpre)), "synapses": int(gw.sum())}

# ---- 度分布 ----
r_out = np.bincount(rpre, minlength=NT); r_in = np.bincount(rpost, minlength=NT)
g_out = np.bincount(gpre, minlength=NT); g_in = np.bincount(gpost, minlength=NT)
stats["ks_outdeg"] = float(ks_2samp(r_out, g_out).statistic)
stats["ks_indeg"] = float(ks_2samp(r_in, g_in).statistic)
stats["ks_weight"] = float(ks_2samp(rw, np.asarray(gw)).statistic)
log("KS outdeg:", round(stats["ks_outdeg"], 4), "indeg:", round(stats["ks_indeg"], 4),
    "weight:", round(stats["ks_weight"], 4))

# ---- 互惠性 ----
def reciprocity(p, q, n):
    key = p.astype(np.int64) * n + q
    key.sort()
    rev = q.astype(np.int64) * n + p
    idx = np.clip(np.searchsorted(key, rev), 0, len(key) - 1)
    return float((key[idx] == rev).mean())
stats["recip_real"] = reciprocity(rpre, rpost, NT)
stats["recip_gen"] = reciprocity(gpre, gpost, NT)
log("reciprocity real/gen:", round(stats["recip_real"], 4), round(stats["recip_gen"], 4))

# ---- 连通分量 ----
A = sp.coo_matrix((np.ones(len(rpre), dtype=np.int8), (rpre, rpost)), shape=(NT, NT)).tocsr()
nc, labels = connected_components(A + A.T, directed=False)
sizes = np.bincount(labels)
stats["components_real"] = int(nc)
stats["giant_real"] = float(sizes.max() / NT)
del A; 
Ag = sp.coo_matrix((np.ones(len(gpre), dtype=np.int8), (gpre, gpost)), shape=(NT, NT)).tocsr()
ncg, lg = connected_components(Ag + Ag.T, directed=False)
stats["components_gen"] = int(ncg)
stats["giant_gen"] = float(np.bincount(lg).max() / NT)
del Ag
log("giant component real/gen:", round(stats["giant_real"], 4), round(stats["giant_gen"], 4))

# ---- 聚类系数(采样) ----
rng = np.random.default_rng(0)
def clustering_sample(p, q, n, k=3000):
    A = sp.coo_matrix((np.ones(len(p), dtype=np.int8), (p, q)), shape=(n, n)).tocsr()
    Au = A + A.T
    Au.data[:] = 1
    A2 = Au @ Au
    nodes_idx = rng.choice(n, k, replace=False)
    vals = []
    deg = np.asarray(Au.sum(1)).ravel()
    for v in nodes_idx:
        d = deg[v]
        if d < 2: continue
        tri = A2[v, v] / 2
        vals.append(2 * tri / (d * (d - 1)))
    return float(np.mean(vals))
stats["cluster_real"] = clustering_sample(rpre, rpost, NT)
stats["cluster_gen"] = clustering_sample(gpre, gpost, NT)
log("clustering real/gen:", round(stats["cluster_real"], 4), round(stats["cluster_gen"], 4))

# ---- 类型对边数相关性 ----
d = np.load(D + "typepair.npz")
uk, cnt = d['uk'], d['cnt']
type_count = d['type_count']
T = len(type_count)
tp, tq = uk // T, uk % T
# 生成图的类型对边数（实例均匀落在类型内，类型偏移即可还原）
offsets = np.zeros(T, dtype=np.int64); offsets[1:] = np.cumsum(type_count)[:-1]
gtp = np.searchsorted(offsets, gpre, side="right") - 1
gtq = np.searchsorted(offsets, gpost, side="right") - 1
gkey = gtp.astype(np.int64) * T + gtq
guk, gcnt = np.unique(gkey, return_counts=True)
gcnt_map = dict(zip(guk.tolist(), gcnt.tolist()))
gcnt_aligned = np.array([gcnt_map.get(int(k), 0) for k in uk])
rho, _ = spearmanr(cnt, gcnt_aligned)
stats["typepair_spearman"] = float(rho)
log("type-pair edge count spearman:", round(rho, 4))

with open(OUT + "validation.json", "w") as f:
    json.dump(stats, f, indent=2)
log(json.dumps(stats, indent=2))
log("VALIDATE DONE")
