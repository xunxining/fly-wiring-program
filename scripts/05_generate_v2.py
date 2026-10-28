#!/usr/bin/env python3
"""Stage 5: v2 生成器 —— 亲和码(类型级) + Peters规则(实例级空间邻近)
真实发育的两级机制:
  1) 分子亲和码决定哪些类型兼容 (v1 已拟合)
  2) 轴突/树突的物理重叠决定实例间实际接线 (本脚本用胞体位置近似)
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import os, time, gc

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"
rng = np.random.default_rng(11)

def log(*a): print(*a, flush=True)

d = np.load(D + "typepair.npz")
uk, cnt = d['uk'].astype(np.int64), d['cnt'].astype(np.int64)
type_count = d['type_count'].astype(np.int64)
T = len(type_count)
tp, tq = uk // T, uk % T
type_names = np.load(OUT + "type_names.npy", allow_pickle=True)

# 节点位置
nodes = pd.read_feather(D + "nodes.feather", columns=["type", "somaLocation", "superclass"])
node_type_name = nodes["type"].to_numpy(object)
pos = np.full((len(nodes), 3), np.nan, dtype=np.float32)
sl = nodes["somaLocation"]
has = sl.notna().to_numpy()
pos[has] = np.stack(sl[has].to_numpy())
sc = nodes["superclass"].fillna("UNK").to_numpy()
del nodes, sl; gc.collect()

t2i = {t: i for i, t in enumerate(type_names)}
node_type = np.array([t2i.get(x, -1) for x in node_type_name], dtype=np.int32)

# 类型的 superclass（权重分布用）
sc_types = sorted(np.unique(sc)); sc2i = {s: i for i, s in enumerate(sc_types)}
NSC = len(sc_types)
node_sc = np.array([sc2i[x] for x in sc], dtype=np.int32)
type_sc = np.zeros(T, dtype=np.int32)
for t in range(T):
    vals, c = np.unique(node_sc[node_type == t], return_counts=True)
    type_sc[t] = vals[np.argmax(c)]

distribs = np.load(OUT + "distribs.npz", allow_pickle=True)
w_q = distribs["w_q"]
Q = 20
qs = np.linspace(0, 1, Q + 1)

def sample_quant_2d(rows, n):
    jf = rng.random(n) * Q
    j = jf.astype(int); frac = (jf - j).astype(np.float32)
    v0 = np.take_along_axis(rows, j[:, None], axis=1).ravel()
    v1 = np.take_along_axis(rows, np.minimum(j + 1, Q)[:, None], axis=1).ravel()
    return v0 * (1 - frac) + v1 * frac

def generate(S, out_prefix, k_near=4):
    new_count = (type_count * S).astype(np.int64)
    total_nodes = int(new_count.sum())
    offsets = np.zeros(T, dtype=np.int64)
    offsets[1:] = np.cumsum(new_count)[:-1]
    lam = cnt.astype(np.float64) * S

    # 为每个类型准备实例位置
    # S=1: 直接用真实实例位置; S>1: 真实位置 bootstrap + 抖动
    type_positions = []
    for t in range(T):
        real_idx = np.flatnonzero(node_type == t)
        P = pos[real_idx]
        valid = ~np.isnan(P[:, 0])
        P = P[valid]
        n_new = new_count[t]
        if n_new == 0:
            type_positions.append(None); continue
        if len(P) == 0:
            type_positions.append(None); continue
        if S == 1 and n_new == len(P):
            Pnew = P
        else:
            ext = np.std(P, axis=0).mean() + 1e-3
            Pnew = P[rng.integers(0, len(P), n_new)] + rng.normal(0, 0.05 * ext, (n_new, 3))
        type_positions.append(Pnew.astype(np.float32))
    # KD树（仅对有位置的类型）
    trees = {}
    for t in range(T):
        P = type_positions[t]
        if P is not None and len(P) >= k_near:
            trees[t] = cKDTree(P)
    log(f"S={S}: nodes={total_nodes:,}, trees={len(trees)}, expected edges={int(lam.sum()):,}")

    fpre = open(out_prefix + "_pre.i32", "wb")
    fpost = open(out_prefix + "_post.i32", "wb")
    fw = open(out_prefix + "_w.i32", "wb")
    Bn = 100_000
    n_written = 0
    for i0 in range(0, len(uk), Bn):
        i1 = min(i0 + Bn, len(uk))
        ne = rng.poisson(lam[i0:i1])
        sel = ne > 0
        if not sel.any(): continue
        tb_s, tb_t, neb = tp[i0:i1][sel], tq[i0:i1][sel], ne[sel]
        pi_all, qi_all = [], []
        for s, t, e in zip(tb_s, tb_t, neb):
            ns_, nt_ = new_count[s], new_count[t]
            pi = rng.integers(0, ns_, e)
            if t in trees and type_positions[s] is not None:
                P_s = type_positions[s]
                # pre 实例位置（S=1 时 pi 直接是实例索引）
                pp = P_s[pi % len(P_s)]
                k = min(k_near, nt_)
                dists, cands = trees[t].query(pp, k=k)
                pick = rng.integers(0, k, e)
                qi = np.take_along_axis(cands, pick[:, None], axis=1).ravel()
            else:
                qi = rng.integers(0, nt_, e)
            pi_all.append(pi); qi_all.append(qi)
        pi = np.concatenate(pi_all); qi = np.concatenate(qi_all)
        ep_s = np.repeat(tb_s, neb); ep_t = np.repeat(tb_t, neb)
        wkeys = type_sc[ep_s].astype(np.int64) * NSC + type_sc[ep_t]
        ww = np.clip(np.round(sample_quant_2d(w_q[wkeys], len(ep_s))), 1, None).astype(np.int32)
        (offsets[ep_s] + pi).astype(np.int32).tofile(fpre)
        (offsets[ep_t] + qi).astype(np.int32).tofile(fpost)
        ww.tofile(fw)
        n_written += len(pi)
        if i0 % (Bn * 10) == 0:
            log(f"  pairs {i0:,}/{len(uk):,}, edges {n_written:,}")
    fpre.close(); fpost.close(); fw.close()
    log(f"GENERATE v2 DONE S={S}: {n_written:,} edges")

if __name__ == "__main__":
    import sys
    S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    generate(S, f"/tmp/flydata/gen/genv2_S{S:g}")
