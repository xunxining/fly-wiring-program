#!/usr/bin/env python3
"""通用验证: python3 06_validate_any.py <gen_prefix> <out_json>"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp, spearmanr
import json, sys

D = "/tmp/flydata/"
prefix, outjson = sys.argv[1], sys.argv[2]

pre = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(D + "e_post.i32", dtype=np.int32, mode="r")
w = np.memmap(D + "e_w.i32", dtype=np.int32, mode="r")
node_type = np.load(D + "node_type.npy")
N = len(node_type)
m = (node_type[pre] >= 0) & (node_type[post] >= 0)
typed_nodes = np.flatnonzero(node_type >= 0)
remap = np.full(N, -1, dtype=np.int32)
remap[typed_nodes] = np.arange(len(typed_nodes), dtype=np.int32)
rpre = remap[pre[m]]; rpost = remap[post[m]]; rw = np.asarray(w[m])
NT = len(typed_nodes)

gpre = np.memmap(prefix + "_pre.i32", dtype=np.int32, mode="r")
gpost = np.memmap(prefix + "_post.i32", dtype=np.int32, mode="r")
gw = np.memmap(prefix + "_w.i32", dtype=np.int32, mode="r")

stats = {}
r_out = np.bincount(rpre, minlength=NT); r_in = np.bincount(rpost, minlength=NT)
g_out = np.bincount(gpre, minlength=NT); g_in = np.bincount(gpost, minlength=NT)
stats["ks_outdeg"] = float(ks_2samp(r_out, g_out).statistic)
stats["ks_indeg"] = float(ks_2samp(r_in, g_in).statistic)
stats["ks_weight"] = float(ks_2samp(rw, np.asarray(gw)).statistic)
stats["edges_real"] = int(len(rpre)); stats["edges_gen"] = int(len(gpre))
stats["syn_real"] = int(rw.sum()); stats["syn_gen"] = int(gw.sum())

def reciprocity(p, q, n):
    key = p.astype(np.int64) * n + q
    key.sort()
    rev = q.astype(np.int64) * n + p
    idx = np.clip(np.searchsorted(key, rev), 0, len(key) - 1)
    return float((key[idx] == rev).mean())
stats["recip_real"] = reciprocity(rpre, rpost, NT)
stats["recip_gen"] = reciprocity(np.asarray(gpre), np.asarray(gpost), NT)
print("recip done", flush=True)

rng = np.random.default_rng(0)
def clustering_sample(p, q, n, k=2000):
    A = sp.coo_matrix((np.ones(len(p), dtype=np.int8), (p, q)), shape=(n, n)).tocsr()
    Au = A + A.T
    Au.data[:] = 1; Au.setdiag(0); Au.eliminate_zeros()
    indptr, indices = Au.indptr, Au.indices
    deg = np.diff(indptr)
    vals = []
    for v in rng.choice(n, k, replace=False):
        nb = indices[indptr[v]:indptr[v+1]]
        d = len(nb)
        if d < 2: continue
        tri = Au[nb][:, nb].nnz / 2
        vals.append(tri / (d * (d - 1) / 2))
    return float(np.mean(vals))
stats["cluster_real"] = clustering_sample(rpre, rpost, NT)
print("cluster real done", flush=True)
stats["cluster_gen"] = clustering_sample(np.asarray(gpre), np.asarray(gpost), NT)
print("cluster gen done", flush=True)

with open(outjson, "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2), flush=True)
print("DONE")
