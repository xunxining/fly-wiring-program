#!/usr/bin/env python3
"""Stage 36: 导出 v4 位置核参数 —— 对 3,440 个显著对, 用全部边(生成用途, 无需裂半)
每对输出: s,t,is_self,a,phi + 实例相位池 (theta_pre/theta_post; 自对单池)
振幅 a 由矩匹配导出: 连边上 E[cos(Δθ+φ)] = I1(a)/I0(a) 的逆 (密度 ∝ exp(a·cos))
输出 data_census/poskernel.npz
"""
import numpy as np
import json, time
from scipy.special import i0, i1

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
T = 11751
DENSE_CAP = 400_000

def log(*a): print(*a, flush=True)

d = np.load(OUT + "census_pairs.npz")
hs, ht = d['s'][d['hit']], d['t'][d['hit']]
hit_keys = set((hs.astype(np.int64) * T + ht).tolist())
log("hit pairs:", len(hit_keys))

# Bessel 反查表
agrid = np.linspace(0, 8, 1601)
cgrid = i1(agrid) / i0(agrid)
def bessel_inv(c):
    return float(np.interp(min(max(c, 0.0), 0.999), cgrid, agrid))

node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
pre = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="r")

keys = node_type[pre].astype(np.int64) * T + node_type[post].astype(np.int64)
order = np.argsort(keys, kind="stable")
sk = keys[order]
bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
del keys
log("sorted")

from scipy.sparse.linalg import svds, LinearOperator

def top2(B, r, c, E1):
    m, n = B.shape
    if m * n <= DENSE_CAP:
        Bc = B - np.outer(r, c) / E1
        U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
        return U[:, :2], Vt[:2, :]
    def mv(x):  return B @ x - r * (c @ x) / E1
    def rmv(x): return B.T @ x - c * (r @ x) / E1
    U, S, Vt = svds(LinearOperator((m, n), matvec=mv, rmatvec=rmv), k=2)
    o = np.argsort(S)[::-1]
    return U[:, o], Vt[o, :]

pk_s, pk_t, pk_a, pk_phi, pk_self, pk_ns, pk_nt = [], [], [], [], [], [], []
th_pool = []          # 拼接的相位池
th_off = [0]
id_pool = []          # 拼接的实例 node id (pu 然后 qu; 自对只存 pu)
dg_pool = []          # 拼接的对内度数倾向 (outdeg/indeg, 各自均值归一)
t0 = time.time()
done = 0
for gidx in range(len(bounds) - 1):
    lo, hi = bounds[gidx], bounds[gidx + 1]
    if int(sk[lo]) not in hit_keys:
        continue
    ei = order[lo:hi]
    s, t = divmod(int(sk[lo]), T)
    pu, pinv = np.unique(pre[ei], return_inverse=True)
    qu, qinv = np.unique(post[ei], return_inverse=True)
    m, n = len(pu), len(qu)
    E = float(len(ei))
    B = np.zeros((m, n), dtype=np.float32)
    B[pinv, qinv] = 1.0
    r, c = B.sum(1), B.sum(0)
    deg_p = np.bincount(pinv, minlength=m).astype(np.float32)
    deg_q = np.bincount(qinv, minlength=n).astype(np.float32)
    if s == t and m == n and (pu == qu).all():
        Bs = B + B.T
        rs, cs = Bs.sum(1), Bs.sum(0)
        U2, V2 = top2(Bs, rs, cs, Bs.sum())
        th = np.arctan2(U2[:, 0], U2[:, 1])     # 对称化后 U≈V
        dd = th[pinv] - th[qinv]
        phi = 0.0
        C = float(np.cos(dd).mean())
        a = bessel_inv(C)
        th_pool.append(th.astype(np.float32))
        id_pool.append(pu.astype(np.int32))
        dg = (deg_p + deg_q) / 2
        dg_pool.append((dg / max(dg.mean(), 1e-9)).astype(np.float32))
        th_off.append(th_off[-1] + m)
        pk_self.append(1); pk_ns.append(m); pk_nt.append(m)
    else:
        U2, V2 = top2(B, r, c, E)
        th_p = np.arctan2(U2[:, 0], U2[:, 1])
        th_q = np.arctan2(V2[0, :], V2[1, :])
        dd = th_p[pinv] - th_q[qinv]
        M = np.exp(1j * dd).mean()
        phi = float(np.angle(M))
        C = float(np.cos(dd - phi).mean())
        a = bessel_inv(C)
        th_pool.append(np.concatenate([th_p, th_q]).astype(np.float32))
        id_pool.append(np.concatenate([pu, qu]).astype(np.int32))
        dg_pool.append(np.concatenate([deg_p / max(deg_p.mean(), 1e-9),
                                       deg_q / max(deg_q.mean(), 1e-9)]).astype(np.float32))
        th_off.append(th_off[-1] + m + n)
        pk_self.append(0); pk_ns.append(m); pk_nt.append(n)
    pk_s.append(s); pk_t.append(t); pk_a.append(a); pk_phi.append(phi)
    done += 1
    if done % 500 == 0:
        log(f"{done}/{len(hit_keys)}  {time.time()-t0:.0f}s")

TH = np.concatenate(th_pool)
np.savez(OUT + "poskernel.npz",
         s=np.array(pk_s, np.int32), t=np.array(pk_t, np.int32),
         a=np.array(pk_a, np.float32), phi=np.array(pk_phi, np.float32),
         is_self=np.array(pk_self, np.int8), ns=np.array(pk_ns, np.int32),
         nt=np.array(pk_nt, np.int32), th_off=np.array(th_off, np.int64),
         theta=TH, ids=np.concatenate(id_pool), deg=np.concatenate(dg_pool))
log(f"DONE pairs={done}, theta pool={len(TH):,}, a: med={np.median(pk_a):.2f} "
    f"p90={np.quantile(pk_a, 0.9):.2f} max={np.max(pk_a):.2f}")
