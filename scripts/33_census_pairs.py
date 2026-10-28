#!/usr/bin/env python3
"""Stage 33: 全脑位置编码普查 —— 逐类型对谱检验(裂半无偏)
对每个实例数足够(n_s>=8,n_t>=8)且边数>=40的类型对(s,t):
  H1 半边 -> 度校正矩阵 SVD -> 实例相位 theta_pre/theta_post (atan2 of top-2 奇异向量)
  H1 拟合相位偏移 phi;  H2 半边(完全出样本)评估:
    A        = |mean exp(i(θ_i-θ_j))|  圆对齐度 (噪声基准 ~1/sqrt(E2))
    auc_lift = AUC(cos(θ_i-θ_j-φ), 边 vs 非边) - 0.5
    sig2     = 第2奇异值 / 随机矩阵体相边缘 (Tracy-Widom 型信号比)
输出 data_census/pair_records.f32 (每对 10 个 float32) + pair_state.json (断点续跑)
"""
import numpy as np
import json, os, time
from scipy.stats import rankdata
from scipy.sparse.linalg import svds, LinearOperator

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
T = 11751
MIN_INST, MIN_E, MIN_HALF = 5, 20, 10
NPERM = 8
DENSE_CAP = 400_000     # m*n 超过则走稀疏 svds
RNG = np.random.default_rng(7)

def log(*a): print(*a, flush=True)

node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
node_ids = np.fromfile(OUT + "node_ids.i64", dtype=np.int64)
# 左右半球标记 (somaSide): L=0, R=1, 其他/缺失=-1
import pandas as pd
_ann = pd.read_feather(ROOT + "data/body-annotations.feather", columns=["bodyId", "somaSide"])
_idx = np.clip(np.searchsorted(node_ids, _ann["bodyId"].to_numpy(np.int64)), 0, len(node_ids)-1)
node_side = np.full(len(node_ids), -1, dtype=np.int8)
_hit = node_ids[_idx] == _ann["bodyId"].to_numpy(np.int64)
_sv = _ann["somaSide"].to_numpy(object)
node_side[_idx[_hit]] = np.where(_sv[_hit] == "L", 0, np.where(_sv[_hit] == "R", 1, -1)).astype(np.int8)
del _ann, _idx, _hit, _sv
pre = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="r")
E_ALL = pre.shape[0]
log("edges:", E_ALL)

t0 = time.time()
keys = node_type[pre].astype(np.int64) * T + node_type[post].astype(np.int64)
order = np.argsort(keys, kind="stable")
sk = keys[order]
bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
log(f"sort done {time.time()-t0:.0f}s, groups={len(bounds)-1}")
del keys

REC = OUT + "pair_records.npy"
STATE = OUT + "pair_state.json"
NF = 13   # s,t,m,n,E,A,lift,phi,z_A,Rth_pre,Rth_post,r_side,sig2
ngroups = len(bounds) - 1
start_g, nrec = 0, 0
recbuf = []
if os.path.exists(STATE):
    st = json.load(open(STATE))
    start_g, nrec = st["groups_done"], st["nrec"]
    recbuf = np.load(REC).tolist() if os.path.exists(REC) else []
    log(f"resume from group {start_g}, nrec={nrec}, buf={len(recbuf)}")

def flush_recs():
    np.save(REC, np.asarray(recbuf, dtype=np.float32).reshape(-1, NF))

def auc_rank(pos, neg):
    r = rankdata(np.r_[pos, neg])
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))

def top2_svd(B, r, c, E1):
    m, n = B.shape
    if m * n <= DENSE_CAP:
        Bc = B - np.outer(r, c) / E1
        U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
        return U[:, :2], S[:2], Vt[:2, :]
    rs, cs = r.copy(), c.copy()
    def mv(x):  return B @ x - rs * (cs @ x) / E1
    def rmv(x): return B.T @ x - cs * (rs @ x) / E1
    LO = LinearOperator((m, n), matvec=mv, rmatvec=rmv)
    U, S, Vt = svds(LO, k=2)
    o = np.argsort(S)[::-1]
    return U[:, o], S[o], Vt[o, :]

t1 = time.time()
tested = 0
for gidx in range(start_g, ngroups):
    lo, hi = bounds[gidx], bounds[gidx + 1]
    E = hi - lo
    if E < MIN_E:
        continue
    ei = order[lo:hi]
    pu, pinv = np.unique(pre[ei], return_inverse=True)
    qu, qinv = np.unique(post[ei], return_inverse=True)
    m, n = len(pu), len(qu)
    if m < MIN_INST or n < MIN_INST:
        continue
    # 裂半
    h1 = RNG.random(E) < 0.5
    if h1.sum() < MIN_HALF or (~h1).sum() < MIN_HALF:
        continue
    r1, c1 = pinv[h1], qinv[h1]
    r2, c2 = pinv[~h1], qinv[~h1]
    B = np.zeros((m, n), dtype=np.float32)
    B[r1, c1] = 1.0
    rdeg, cdeg = B.sum(1), B.sum(0)
    E1 = float(h1.sum())
    try:
        U2, S2, Vt2 = top2_svd(B, rdeg, cdeg, E1)
    except Exception:
        continue
    if len(S2) < 2:
        continue
    th_p = np.arctan2(U2[:, 0], U2[:, 1])
    th_q = np.arctan2(Vt2[0, :], Vt2[1, :])
    # phi 从 H1 拟合
    d1 = th_p[r1] - th_q[c1]
    phi = float(np.angle(np.exp(1j * d1).mean()))
    # H2 出样本评估
    d2 = th_p[r2] - th_q[c2]
    A = float(np.abs(np.exp(1j * d2).mean()))
    # 置换零分布: 打乱 post 相位, 保持相位边际分布
    Ap = np.empty(NPERM)
    for pp_ in range(NPERM):
        Ap[pp_] = np.abs(np.exp(1j * (th_p[r2] - th_q[RNG.permutation(n)[c2]])).mean())
    z_A = float((A - Ap.mean()) / (Ap.std() + 1e-9))
    # 环 vs 块: 相位在圆周上的聚集度 (环≈低, 块/双侧≈高)
    Rp = float(np.abs(np.exp(1j * th_p).mean()))
    Rq = float(np.abs(np.exp(1j * th_q).mean()))
    # 相位与左右半球的相关 (point-biserial on cos θ)
    sd = node_side[pu]
    sm = sd >= 0
    if sm.sum() >= 8 and np.unique(sd[sm]).size == 2:
        x = np.cos(th_p)[sm]; y = sd[sm].astype(np.float32)
        r_side = float(np.corrcoef(x, y)[0, 1])
    else:
        r_side = 0.0
    # 非边采样(与 H2 等量)
    occ = np.zeros(m * n, dtype=bool)
    occ[pinv * n + qinv] = True
    nneg = len(d2)
    na = RNG.integers(0, m, nneg * 3)
    nb = RNG.integers(0, n, nneg * 3)
    keep = ~occ[na * n + nb]
    na, nb = na[keep][:nneg], nb[keep][:nneg]
    if len(na) < 50:
        continue
    sc_pos = np.cos(d2 - phi)
    sc_neg = np.cos(th_p[na] - th_q[nb] - phi)
    auc = auc_rank(sc_pos, sc_neg)
    p1 = E1 / (m * n)
    bulk = np.sqrt(p1 * (1 - p1)) * (np.sqrt(m) + np.sqrt(n))
    sig2 = float(S2[1] / (bulk + 1e-9))
    s, t = divmod(int(sk[lo]), T)
    recbuf.append([float(s), float(t), float(m), float(n), float(E),
                   A, auc - 0.5, phi, z_A, Rp, Rq, r_side, sig2])
    nrec += 1
    tested += 1
    if tested % 20000 == 0:
        flush_recs()
        json.dump({"groups_done": gidx + 1, "nrec": nrec}, open(STATE, "w"))
        log(f"g={gidx}/{ngroups} tested={tested} {time.time()-t1:.0f}s "
            f"rss={__import__('resource').getrusage(0).ru_maxrss//1024}MB")

flush_recs()
json.dump({"groups_done": ngroups, "nrec": nrec}, open(STATE, "w"))
log(f"DONE tested={tested} nrec={nrec} {time.time()-t1:.0f}s")
