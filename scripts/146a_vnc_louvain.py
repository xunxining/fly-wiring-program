#!/usr/bin/env python3
# 146_vnc_comm.py — ②b-1d (§53): VNC 闭合的中尺度社区(Louvain)载体检验
import json, time
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + "data_male/"
G = ROOT + "gen_edges/g_p9bm2"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

node_sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
N = len(node_sc)
VNC = {15, 16, 17, 21, 22, 23, 24, 25, 26, 27}
is_vnc = np.isin(node_sc, list(VNC))
vidx = np.where(is_vnc)[0]

# ---- real VNC 诱导子图 (无向, 权重=双向和) ----
pre = np.fromfile(D + "e_pre.i32", dtype=np.int32); post = np.fromfile(D + "e_post.i32", dtype=np.int32)
w = np.fromfile(D + "e_w.i32", dtype=np.int32)
m = is_vnc[pre] & is_vnc[post]
lo = np.minimum(pre[m], post[m]); hi = np.maximum(pre[m], post[m]); ww = w[m].astype(np.float64)
key = lo.astype(np.int64) * N + hi
uk, inv = np.unique(key, return_inverse=True)
uw = np.zeros(len(uk)); np.add.at(uw, inv, ww)
del pre, post, w, m, lo, hi, ww, key, inv
Gnx = nx.Graph()
Gnx.add_nodes_from(range(len(vidx)))
pos = {g: i for i, g in enumerate(vidx)}
eu = np.searchsorted(vidx, uk % N); ev = np.searchsorted(vidx, uk // N)
Gnx.add_edges_from(zip(eu.tolist(), ev.tolist(), ({"weight": float(x)} for x in uw)))
log(f"子图: {Gnx.number_of_nodes():,} 节点 {Gnx.number_of_edges():,} 边")
comm = nx.community.louvain_communities(Gnx, weight="weight", resolution=1.0, seed=13)
COM = np.full(N, -1, dtype=np.int32)
for ci, cset in enumerate(comm):
    for li in cset: COM[vidx[li]] = ci
sizes = sorted((len(c) for c in comm), reverse=True)
Q = nx.community.modularity(Gnx, comm, weight="weight")
log(f"Louvain: {len(comm)} 社区, top sizes {sizes[:6]}, Q={Q:.3f}")
del Gnx


np.savez_compressed(D + "vnc_comm_labels.npz", COM=COM, Q=Q, n_comm=len(comm))
log("-> data_male/vnc_comm_labels.npz")
