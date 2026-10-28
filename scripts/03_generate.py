#!/usr/bin/env python3
"""Stage 3: 生成器 —— 用拟合好的"基因组"生成任意规模的果蝇式网络
缩放原则: 度数守恒 —— 实例数 ×S 时, 类型对边数 ×S, 每神经元期望度数不变
"""
import numpy as np
import pandas as pd
import torch
import os, json, time, gc

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"
rng = np.random.default_rng(7)
Q = 20

def log(*a): print(*a, flush=True)

# ---- 载入"基因组" ----
ckpt = torch.load(OUT + "affinity.pt", weights_only=False)
T = ckpt["T"]
type_names = np.load(OUT + "type_names.npy", allow_pickle=True)

d = np.load(D + "typepair.npz")
uk, cnt = d['uk'].astype(np.int64), d['cnt'].astype(np.int64)
type_count = d['type_count'].astype(np.int64)
tp, tq = uk // T, uk % T

pre = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(D + "e_post.i32", dtype=np.int32, mode="r")
w = np.memmap(D + "e_w.i32", dtype=np.int32, mode="r")
node_type = np.load(D + "node_type.npy")
N = len(node_type)
nodes = pd.read_feather(D + "nodes.feather", columns=["superclass"])
sc = nodes["superclass"].fillna("UNK").to_numpy()
del nodes; gc.collect()

outdeg = np.bincount(pre, minlength=N)
indeg = np.bincount(post, minlength=N)

# 每型经验度分布（分位数形式，作为"基因"保存）
qs = np.linspace(0, 1, Q + 1)
out_q = np.zeros((T, Q + 1), dtype=np.float32)
in_q = np.zeros((T, Q + 1), dtype=np.float32)
for t in range(T):
    m = node_type == t
    out_q[t] = np.quantile(outdeg[m], qs)
    in_q[t] = np.quantile(indeg[m], qs)
log("degree quantiles done")

# 权重分布: 按 superclass 对的经验分位数
sc_types = sorted(np.unique(sc))
sc2i = {s: i for i, s in enumerate(sc_types)}
node_sc = np.array([sc2i[x] for x in sc], dtype=np.int32)
NSC = len(sc_types)
m_typed = (node_type[pre] >= 0) & (node_type[post] >= 0)
wkey = node_sc[pre[m_typed]].astype(np.int64) * NSC + node_sc[post[m_typed]]
ws = np.asarray(w[m_typed], dtype=np.float32)
w_q = np.zeros((NSC * NSC, Q + 1), dtype=np.float32)
for k in np.unique(wkey):
    w_q[k] = np.quantile(ws[wkey == k], qs)
log("weight quantiles done")

# 类型 -> 主要 superclass
type_sc = np.zeros(T, dtype=np.int32)
for t in range(T):
    vals, cts = np.unique(node_sc[node_type == t], return_counts=True)
    type_sc[t] = vals[np.argmax(cts)]

np.savez(OUT + "distribs.npz", out_q=out_q, in_q=in_q, w_q=w_q,
         sc_types=np.array(sc_types, dtype=object), type_count=type_count,
         uk=uk, cnt=cnt)
log("genome saved")

def sample_quant_1d(row, n):
    return np.interp(rng.random(n), qs, row).astype(np.float32)

def sample_quant_2d(rows, n):
    jf = rng.random(n) * Q
    j = jf.astype(int); frac = (jf - j).astype(np.float32)
    v0 = np.take_along_axis(rows, j[:, None], axis=1).ravel()
    v1 = np.take_along_axis(rows, np.minimum(j + 1, Q)[:, None], axis=1).ravel()
    return v0 * (1 - frac) + v1 * frac

def generate(S, out_prefix):
    new_count = (type_count * S).astype(np.int64)
    total_nodes = int(new_count.sum())
    offsets = np.zeros(T, dtype=np.int64)
    offsets[1:] = np.cumsum(new_count)[:-1]
    lam = cnt.astype(np.float64) * S           # 度数守恒缩放
    est = int(lam.sum())
    log(f"S={S}: nodes={total_nodes:,}, expected edges={est:,}")

    fpre = open(out_prefix + "_pre.i32", "wb")
    fpost = open(out_prefix + "_post.i32", "wb")
    fw = open(out_prefix + "_w.i32", "wb")
    Bn = 20_000
    n_written = 0
    for i0 in range(0, len(uk), Bn):
        i1 = min(i0 + Bn, len(uk))
        ne = rng.poisson(lam[i0:i1])
        sel = ne > 0
        if not sel.any():
            continue
        tb_s, tb_t, ne = tp[i0:i1][sel], tq[i0:i1][sel], ne[sel]
        edge_pair = np.repeat(np.arange(len(ne)), ne)
        ep_s = tb_s[edge_pair]; ep_t = tb_t[edge_pair]
        pi = offsets[ep_s] + (rng.random(len(edge_pair)) * new_count[ep_s]).astype(np.int64)
        qi = offsets[ep_t] + (rng.random(len(edge_pair)) * new_count[ep_t]).astype(np.int64)
        wkeys = type_sc[ep_s].astype(np.int64) * NSC + type_sc[ep_t]
        ww = np.clip(np.round(sample_quant_2d(w_q[wkeys], len(edge_pair))), 1, None).astype(np.int32)
        pi.astype(np.int32).tofile(fpre)
        qi.astype(np.int32).tofile(fpost)
        ww.tofile(fw)
        n_written += len(edge_pair)
        if i0 % (Bn * 5) == 0:
            log(f"  pairs {i0:,}/{len(uk):,}, edges {n_written:,}")
    fpre.close(); fpost.close(); fw.close()
    log(f"GENERATE DONE S={S}: {n_written:,} edges")
    return total_nodes, n_written

if __name__ == "__main__":
    import sys
    S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    generate(S, f"/tmp/flydata/gen/gen_S{S:g}")
