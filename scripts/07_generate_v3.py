#!/usr/bin/env python3
"""Stage 7: v3 生成器 —— 修复 v2 的两个缺陷
1) 权重: 分位数插值 -> 经验池重采样 (修复突触数10倍虚高)
2) 入度: k近邻候选内按剩余入度配额选择 (修复空间匹配导致的入度边际失真)
机制: 亲和码定类型兼容, 空间邻近(Peters规则)+度配额定实例接线
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import os, time, gc

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"
rng = np.random.default_rng(13)
RES = 256  # 每 superclass 对的经验权重池大小

def log(*a): print(*a, flush=True)

d = np.load(D + "typepair.npz")
uk, cnt = d['uk'].astype(np.int64), d['cnt'].astype(np.int64)
type_count = d['type_count'].astype(np.int64)
T = len(type_count)
tp, tq = uk // T, uk % T
type_names = np.load(OUT + "type_names.npy", allow_pickle=True)

pre = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(D + "e_post.i32", dtype=np.int32, mode="r")
w = np.memmap(D + "e_w.i32", dtype=np.int32, mode="r")

nodes = pd.read_feather(D + "nodes.feather", columns=["type", "somaLocation", "superclass"])
node_type_name = nodes["type"].to_numpy(object)
sl = nodes["somaLocation"]
has = sl.notna().to_numpy()
pos = np.full((len(nodes), 3), np.nan, dtype=np.float32)
pos[has] = np.stack(sl[has].to_numpy())
sc = nodes["superclass"].fillna("UNK").to_numpy()
del nodes, sl; gc.collect()

t2i = {t: i for i, t in enumerate(type_names)}
node_type = np.array([t2i.get(x, -1) for x in node_type_name], dtype=np.int32)
sc_types = sorted(np.unique(sc)); sc2i = {s: i for i, s in enumerate(sc_types)}
NSC = len(sc_types)
node_sc = np.array([sc2i[x] for x in sc], dtype=np.int32)

type_sc = np.zeros(T, dtype=np.int32)
for t in range(T):
    vals, c = np.unique(node_sc[node_type == t], return_counts=True)
    type_sc[t] = vals[np.argmax(c)]

# ---- 经验权重池 ----
m_typed = (node_type[pre] >= 0) & (node_type[post] >= 0)
wkey = node_sc[pre[m_typed]].astype(np.int64) * NSC + node_sc[post[m_typed]]
ws = np.asarray(w[m_typed], dtype=np.int32)
w_emp = np.ones((NSC * NSC, RES), dtype=np.int32)
for k in np.unique(wkey):
    vals = ws[wkey == k]
    w_emp[k] = np.resize(vals, RES) if len(vals) < RES else vals[rng.choice(len(vals), RES, replace=False)]
del ws, wkey; gc.collect()
log("weight pools done")

# ---- 每型经验入度序列（配额重采样源）----
indeg = np.bincount(post, minlength=len(node_type))
type_indeg_seq = []
for t in range(T):
    type_indeg_seq.append(indeg[node_type == t].astype(np.int32))
del indeg; gc.collect()
log("indeg sequences done")

def generate(S, out_prefix, k_near=6):
    new_count = (type_count * S).astype(np.int64)
    total_nodes = int(new_count.sum())
    offsets = np.zeros(T, dtype=np.int64)
    offsets[1:] = np.cumsum(new_count)[:-1]
    lam = cnt.astype(np.float64) * S

    # 实例位置
    type_positions = []
    for t in range(T):
        real_idx = np.flatnonzero(node_type == t)
        P = pos[real_idx]
        P = P[~np.isnan(P[:, 0])]
        n_new = new_count[t]
        if n_new == 0 or len(P) == 0:
            type_positions.append(None); continue
        if S == 1 and n_new == len(P):
            Pnew = P
        else:
            ext = np.std(P, axis=0).mean() + 1e-3
            Pnew = P[rng.integers(0, len(P), n_new)] + rng.normal(0, 0.05 * ext, (n_new, 3))
        type_positions.append(Pnew.astype(np.float32))
    trees = {t: cKDTree(type_positions[t]) for t in range(T)
             if type_positions[t] is not None and len(type_positions[t]) >= k_near}

    # 入度配额: 从每型经验序列重采样
    quota = np.zeros(total_nodes, dtype=np.int32)
    for t in range(T):
        n_new = new_count[t]
        if n_new == 0: continue
        seq = type_indeg_seq[t]
        if len(seq) == 0: continue
        q = seq[rng.integers(0, len(seq), n_new)]
        quota[offsets[t]:offsets[t] + n_new] = q
    log(f"S={S}: nodes={total_nodes:,}, quota_sum={int(quota.sum()):,}, expected_edges={int(lam.sum()):,}")

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
        pi_l, qi_l = [], []
        for s, t, e in zip(tb_s, tb_t, neb):
            ns_, nt_ = int(new_count[s]), int(new_count[t])
            pi = rng.integers(0, ns_, e)
            base_t = offsets[t]
            if t in trees and type_positions[s] is not None:
                P_s = type_positions[s]
                pp = P_s[pi % len(P_s)]
                k = min(k_near, nt_)
                _, cands = trees[t].query(pp, k=k)     # (e, k)
                rem = quota[base_t + cands]            # 剩余配额
                # 配额为0的候选降权: 加小噪声后取最大剩余配额
                score = rem + rng.random(rem.shape) * 0.5
                pick = np.argmax(score, axis=1)
                qi = np.take_along_axis(cands, pick[:, None], axis=1).ravel()
                quota[base_t + qi] -= 1                # 允许为负(软约束)
            else:
                qi = rng.integers(0, nt_, e)
            pi_l.append(pi); qi_l.append(qi)
        pi = np.concatenate(pi_l); qi = np.concatenate(qi_l)
        ep_s = np.repeat(tb_s, neb); ep_t = np.repeat(tb_t, neb)
        wkeys = type_sc[ep_s].astype(np.int64) * NSC + type_sc[ep_t]
        ww = w_emp[wkeys, rng.integers(0, RES, len(ep_s))]
        (offsets[ep_s] + pi).astype(np.int32).tofile(fpre)
        (offsets[ep_t] + qi).astype(np.int32).tofile(fpost)
        ww.astype(np.int32).tofile(fw)
        n_written += len(pi)
        if i0 % (Bn * 10) == 0:
            log(f"  pairs {i0:,}/{len(uk):,}, edges {n_written:,}")
    fpre.close(); fpost.close(); fw.close()
    log(f"GENERATE v3 DONE S={S}: {n_written:,} edges")

if __name__ == "__main__":
    import sys
    S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    generate(S, f"/tmp/flydata/gen/genv3_S{S:g}")
