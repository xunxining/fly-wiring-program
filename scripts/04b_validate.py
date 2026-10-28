#!/usr/bin/env python3
"""Stage 4b: 聚类系数(内存安全) + 类型对边数相关性"""
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
import json

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"

def log(*a): print(*a, flush=True)

pre = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(D + "e_post.i32", dtype=np.int32, mode="r")
node_type = np.load(D + "node_type.npy")
N = len(node_type)
m = (node_type[pre] >= 0) & (node_type[post] >= 0)
typed_nodes = np.flatnonzero(node_type >= 0)
remap = np.full(N, -1, dtype=np.int32)
remap[typed_nodes] = np.arange(len(typed_nodes), dtype=np.int32)
rpre = remap[pre[m]]; rpost = remap[post[m]]
NT = len(typed_nodes)

gpre = np.memmap(OUT + "gen_S1_pre.i32", dtype=np.int32, mode="r")
gpost = np.memmap(OUT + "gen_S1_post.i32", dtype=np.int32, mode="r")

rng = np.random.default_rng(0)

def clustering_sample(p, q, n, k=2000):
    A = sp.coo_matrix((np.ones(len(p), dtype=np.int8), (p, q)), shape=(n, n)).tocsr()
    Au = (A + A.T)
    Au.data[:] = 1
    Au.setdiag(0); Au.eliminate_zeros()
    indptr, indices = Au.indptr, Au.indices
    deg = np.diff(indptr)
    nodes_idx = rng.choice(n, k, replace=False)
    vals = []
    for v in nodes_idx:
        nb = indices[indptr[v]:indptr[v+1]]
        d = len(nb)
        if d < 2: continue
        sub = Au[nb][:, nb]   # 邻居子图
        tri = sub.nnz / 2
        vals.append(tri / (d * (d - 1) / 2))
    return float(np.mean(vals))

c_real = clustering_sample(rpre, rpost, NT)
log("clustering real:", round(c_real, 4))
c_gen = clustering_sample(np.asarray(gpre), np.asarray(gpost), NT)
log("clustering gen:", round(c_gen, 4))

# 类型对边数相关性
d = np.load(D + "typepair.npz")
uk, cnt = d['uk'], d['cnt']
type_count = d['type_count']
T = len(type_count)
offsets = np.zeros(T, dtype=np.int64); offsets[1:] = np.cumsum(type_count)[:-1]
gtp = np.searchsorted(offsets, gpre, side="right") - 1
gtq = np.searchsorted(offsets, gpost, side="right") - 1
gkey = gtp.astype(np.int64) * T + gtq
guk, gcnt = np.unique(gkey, return_counts=True)
gcnt_map = dict(zip(guk.tolist(), gcnt.tolist()))
gcnt_aligned = np.array([gcnt_map.get(int(k), 0) for k in uk])
rho, _ = spearmanr(cnt, gcnt_aligned)
log("type-pair spearman:", round(float(rho), 4))

with open(OUT + "validation2.json", "w") as f:
    json.dump({"cluster_real": c_real, "cluster_gen": c_gen,
               "typepair_spearman": float(rho)}, f, indent=2)
log("DONE")
