#!/usr/bin/env python3
# 61_close_mcmc.py — 机制 v3: 度保持交换 MCMC 闭三角 (§21)
# 同 (s,t) 类型对两条有向边 (i1->j1),(i2->j2) 交换目标 -> (i1->j2),(i2->j1)
#   * 出/入度逐节点保持, 类型对配额保持, 无向度保持 => 楔总数 W 不变, T = 3C_c/W 直接可追
#   * j2 由 i1 的 2-跳引导抽取 (u∈N(i1), j2∈N(u)∩t), e2 由反向段 (j2,s) 抽取
#   * 禁止: 自环/与既有无向边重叠 (同时覆盖重边与新建互惠 => 互惠度冻结)
#   * 普查核对与 CX 相关边不可参与; 贪心接受 ΔC>=1; 每轮换完重建索引
# 用法: python3 61_close_mcmc.py <in_prefix> <out_prefix> [R=24] [M=1500000]
import numpy as np, json, time, sys, os, gc, pickle, shutil

ROOT = os.environ.get("GROOT", "/mnt/agents/output/果蝇生成器/")
IN = sys.argv[1]
OUTP = sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 24
M = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
rng = np.random.default_rng(11)
Tp1 = 11752  # T+1

g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
type_count = g['type_count'].astype(np.int64)
T = len(type_count)
N = int(type_count.sum())
gtype = np.repeat(np.arange(T, dtype=np.int32), type_count)

pre = np.fromfile(IN + "_pre.i32", dtype=np.int32)  # 一次性读入, 之后不依赖 /tmp
post = np.fromfile(IN + "_post.i32", dtype=np.int32).copy()
E = len(post)
Nn = np.int64(N)

# 输出骨架先行落盘 (pre/w/cxtheta 全程不变; post 每 4 轮覆盖写)
os.makedirs(os.path.dirname(OUTP), exist_ok=True)
shutil.copy(IN + "_pre.i32", OUTP + "_pre.i32")
shutil.copy(IN + "_w.i32", OUTP + "_w.i32")
if os.path.exists(IN + "_cxtheta.npz"):
    shutil.copy(IN + "_cxtheta.npz", OUTP + "_cxtheta.npz")

# 不可参与边: 普查核对 / 任一端点属 CX 六型
_pk = np.load(ROOT + "data_census/poskernel_cal.npz", allow_pickle=True)
pk_keys = np.sort(_pk['s'].astype(np.int64) * T + _pk['t'].astype(np.int64))
type_names = np.load(ROOT + "data_census/type_names.npy", allow_pickle=True)
tn_str = [str(x) for x in type_names]
lp = pickle.load(open(ROOT + "latent_params.pkl", "rb"))
CXSET = set(tn_str.index(x) for x in lp['type_list'])
cxmask = np.zeros(T, bool)
for x in CXSET:
    cxmask[x] = True
ekey = gtype[pre].astype(np.int64) * T + gtype[post]
pos = np.searchsorted(pk_keys, ekey)
pos = np.clip(pos, 0, len(pk_keys) - 1)
elig = (pk_keys[pos] != ekey) & ~cxmask[gtype[pre]] & ~cxmask[gtype[post]]
elig_idx = np.flatnonzero(elig).astype(np.int32)
print(f"边 {E:,}, 可参与 {len(elig_idx):,} ({len(elig_idx)/E:.1%}), CX型 {sorted(CXSET)}", flush=True)
del ekey, pk_keys, _pk, elig

def build(first=False):
    """无向 ptab + (u,类型) 分段 tkey/tnbr + 反向 rkey/rval(仅可参与边)"""
    mn = np.minimum(pre, post)
    mx = np.maximum(pre, post)
    ptab = np.unique(mn.astype(np.int64) * Nn + mx)
    del mn, mx
    if first:
        a = (ptab // Nn).astype(np.int32)
        b = (ptab - a.astype(np.int64) * Nn).astype(np.int32)
        deg = np.bincount(np.concatenate([a, b]), minlength=N)
        W = int((deg.astype(np.int64) * (deg.astype(np.int64) - 1) // 2).sum())
        del a, b, deg
    else:
        W = 0
    Mu = len(ptab)
    tk = np.empty(2 * Mu, np.int32)
    tn = np.empty(2 * Mu, np.int32)
    aa = (ptab // Nn).astype(np.int32)
    bb = (ptab - aa.astype(np.int64) * Nn).astype(np.int32)
    tk[:Mu] = (aa.astype(np.int64) * Tp1 + gtype[bb]).astype(np.int32)
    tk[Mu:] = (bb.astype(np.int64) * Tp1 + gtype[aa]).astype(np.int32)
    tn[:Mu] = bb
    tn[Mu:] = aa
    del aa, bb
    od = np.argsort(tk)
    tkey = tk[od]
    tnbr = tn[od]
    del tk, tn, od
    rk = (post[elig_idx].astype(np.int64) * Tp1 + gtype[pre[elig_idx]]).astype(np.int32)
    od = np.argsort(rk)
    rkey = rk[od]
    rval = elig_idx[od]
    del rk, od
    gc.collect()
    return ptab, tkey, tnbr, rkey, rval, W

def Cund(xs, ys, tkey, tnbr, ptab):
    """|N(x) ∩ N(y)| 无向: x 的邻居段取自 tkey/tnbr, 与 y 配成无向键查 ptab"""
    out = np.zeros(len(xs), np.int32)
    B = 8000
    for s0 in range(0, len(xs), B):
        x = xs[s0:s0 + B]
        y = ys[s0:s0 + B].astype(np.int64)
        lo = np.searchsorted(tkey, (x * Tp1).astype(np.int32))
        hi = np.searchsorted(tkey, ((x + 1) * Tp1).astype(np.int32))
        d = (hi - lo).astype(np.int64)
        tot = int(d.sum())
        base = np.repeat(lo - np.concatenate([[0], np.cumsum(d)[:-1]]), d)
        u = tnbr[base + np.arange(tot)].astype(np.int64)
        own = np.repeat(np.arange(len(x)), d)
        yr = y[own]
        kk = np.minimum(u, yr) * Nn + np.maximum(u, yr)
        p = np.searchsorted(ptab, kk)
        p = np.clip(p, 0, len(ptab) - 1)
        hit = ptab[p] == kk
        out[s0:s0 + B] = np.bincount(own[hit], minlength=len(x))
    return out

ptab, tkey, tnbr, rkey, rval, W = build(first=True)
T0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0548
print(f"W={W:,}, C_c0≈{T0*W/3:,.0f} (T0={T0})", flush=True)

sum_dC = 0
hist = []
t_start = time.time()
for r in range(R):
    t0 = time.time()
    e1 = elig_idx[rng.integers(0, len(elig_idx), M)]
    e1 = np.unique(e1)
    ine1 = np.zeros(E, bool)
    ine1[e1] = True
    i1 = pre[e1].astype(np.int64)
    j1 = post[e1].astype(np.int64)
    s = gtype[i1]
    t = gtype[j1]
    # 2-跳引导抽 j2: u∈N(i1), j2∈N(u)∩t
    lo = np.searchsorted(tkey, (i1 * Tp1).astype(np.int32))
    hi = np.searchsorted(tkey, ((i1 + 1) * Tp1).astype(np.int32))
    d = hi - lo
    ok = d > 0
    i1, j1, s, t, e1, lo, d = i1[ok], j1[ok], s[ok], t[ok], e1[ok], lo[ok], d[ok]
    uu = tnbr[lo + (rng.random(len(lo)) * d).astype(np.int64)].astype(np.int64)
    ku = (uu * Tp1 + t).astype(np.int32)
    lo2 = np.searchsorted(tkey, ku)
    hi2 = np.searchsorted(tkey, (ku.astype(np.int64) + 1).astype(np.int32))
    d2 = hi2 - lo2
    ok = d2 > 0
    i1, j1, s, t, e1 = i1[ok], j1[ok], s[ok], t[ok], e1[ok]
    lo2, d2 = lo2[ok], d2[ok]
    j2 = tnbr[lo2 + (rng.random(len(lo2)) * d2).astype(np.int64)].astype(np.int64)
    # 反向抽 e2: (j2, s) 段
    kr = (j2 * Tp1 + s).astype(np.int32)
    lo3 = np.searchsorted(rkey, kr)
    hi3 = np.searchsorted(rkey, (kr.astype(np.int64) + 1).astype(np.int32))
    d3 = hi3 - lo3
    ok = d3 > 0
    i1, j1, j2, s, e1 = i1[ok], j1[ok], j2[ok], s[ok], e1[ok]
    lo3, d3 = lo3[ok], d3[ok]
    e2 = rval[lo3 + (rng.random(len(lo3)) * d3).astype(np.int64)]
    i2 = pre[e2].astype(np.int64)
    # 基本过滤
    ok = (i2 != i1) & (j2 != i1) & (j1 != i2) & ~ine1[e2] & (e2 != e1)
    i1, j1, j2, i2, e1, e2 = i1[ok], j1[ok], j2[ok], i2[ok], e1[ok], e2[ok]
    del ine1
    # 无向重叠过滤 (覆盖重边与新建互惠)
    k1a = i1 * Nn + j2
    k2a = i2 * Nn + j1
    kk = np.concatenate([np.minimum(i1, j2) * Nn + np.maximum(i1, j2),
                         np.minimum(i2, j1) * Nn + np.maximum(i2, j1)])
    p = np.searchsorted(ptab, kk)
    p = np.clip(p, 0, len(ptab) - 1)
    bad = (ptab[p] == kk).reshape(2, -1).any(axis=0)
    keep = ~bad
    i1, j1, j2, i2, e1, e2 = i1[keep], j1[keep], j2[keep], i2[keep], e1[keep], e2[keep]
    k1a, k2a = k1a[keep], k2a[keep]
    n_valid = len(e1)
    n_acc = 0
    if n_valid:
        dC = (Cund(i1, j2, tkey, tnbr, ptab) + Cund(i2, j1, tkey, tnbr, ptab)
              - Cund(i1, j1, tkey, tnbr, ptab) - Cund(i2, j2, tkey, tnbr, ptab))
        acc = dC >= 1
        i1, j1, j2, i2, e1, e2, dC = i1[acc], j1[acc], j2[acc], i2[acc], e1[acc], e2[acc], dC[acc]
        k1a, k2a = k1a[acc], k2a[acc]
        nA = len(e1)
        if nA:
            # 批内冲突去重: 新边键(k1a/k2a)与 e2 唯一, 保留先出现者
            allk = np.concatenate([k1a, k2a, e2.astype(np.int64)])
            _, fidx = np.unique(allk, return_index=True)
            first = np.zeros(3 * nA, bool)
            first[fidx] = True
            pid = np.arange(nA)
            km = first[pid] & first[pid + nA] & first[pid + 2 * nA]
            i1, j2, i2, j1, e1, e2, dC = i1[km], j2[km], i2[km], j1[km], e1[km], e2[km], dC[km]
            post[e1] = j2.astype(np.int32)
            post[e2] = j1.astype(np.int32)
            sum_dC += int(dC.sum())
            n_acc = len(e1)
    Cc_est = (T0 * W / 3.0) + sum_dC
    hist.append({"round": r, "valid": int(n_valid), "acc": n_acc, "sum_dC": sum_dC,
                 "T_est": 3 * Cc_est / W})
    print(f"R{r:02d}: 有效 {n_valid:,} 接受 {n_acc:,} ΣΔC={sum_dC:,} T_est={3*Cc_est/W:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if r >= 2 and hist[-1]["acc"] < 0.02 * M and hist[-2]["acc"] < 0.02 * M:
        print("接受率枯竭, 提前停止", flush=True)
        break
    if (r + 1) % 4 == 0:
        post.tofile(OUTP + "_post.i32")
    if r < R - 1:
        del ptab, tkey, tnbr, rkey, rval
        ptab, tkey, tnbr, rkey, rval, _ = build()

post.tofile(OUTP + "_post.i32")
json.dump({"W": W, "T0": T0, "sum_dC": sum_dC, "T_est": 3 * Cc_est / W, "hist": hist,
           "minutes": (time.time() - t_start) / 60}, open(ROOT + "mcmc_log.json", "w"), indent=1)
print(f"DONE: ΣΔC={sum_dC:,}, T_est={3*Cc_est/W:.4f}, {(time.time()-t_start)/60:.1f}min", flush=True)
