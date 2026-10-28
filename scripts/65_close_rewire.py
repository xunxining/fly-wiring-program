#!/usr/bin/env python3
# 65_close_rewire.py — 机制 v5: 全量批量楔重连后处理 (§21)
# 对每条可参与边 (i->j): 经 2-跳走 u∈N(i), j'∈N(u)∩type(j) 重放目标
#   * 出度逐节点保持, 类型对配额保持 (j' 同型), 边数不变
#   * 最终密度图上操作 => 无中途稀疏; 全量覆盖 => 无资格门
#   * 跳过: j'==i / (i,j') 已无向相邻 (保持无重边, 互惠仅能微升)
# 用法: python3 65_close_rewire.py <in_prefix> <out_prefix> [p=1.0] [seed]
import numpy as np, json, time, sys, os, gc, pickle, shutil

ROOT = "/mnt/agents/output/果蝇生成器/"
IN = sys.argv[1]
OUTP = sys.argv[2]
P_CLOSE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 21)
Tp1 = 11752

g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
type_count = g['type_count'].astype(np.int64)
T = len(type_count)
N = int(type_count.sum())
gtype = np.repeat(np.arange(T, dtype=np.int32), type_count)

pre = np.fromfile(IN + "_pre.i32", dtype=np.int32)
post = np.fromfile(IN + "_post.i32", dtype=np.int32).copy()
E = len(post)
Nn = np.int64(N)

os.makedirs(os.path.dirname(OUTP), exist_ok=True)
shutil.copy(IN + "_pre.i32", OUTP + "_pre.i32")
shutil.copy(IN + "_w.i32", OUTP + "_w.i32")
if os.path.exists(IN + "_cxtheta.npz"):
    shutil.copy(IN + "_cxtheta.npz", OUTP + "_cxtheta.npz")

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
pp = np.searchsorted(pk_keys, ekey)
pp = np.clip(pp, 0, len(pk_keys) - 1)
elig = (pk_keys[pp] != ekey) & ~cxmask[gtype[pre]] & ~cxmask[gtype[post]]
elig_idx = np.flatnonzero(elig).astype(np.int32)
print(f"边 {E:,}, 可参与 {len(elig_idx):,}", flush=True)
del ekey, pk_keys, _pk, elig

# 结构: 无向 ptab + (u,类型) 分段 tkey/tnbr
t0 = time.time()
mn = np.minimum(pre, post)
mx = np.maximum(pre, post)
key = mn.astype(np.int64) * Nn + mx
del mn, mx
ptab = np.unique(key)
del key
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
gc.collect()
print(f"结构就绪 ({time.time()-t0:.0f}s, 无向边 {Mu:,})", flush=True)

# 全量批量重连
sel = elig_idx[rng.random(len(elig_idx)) < P_CLOSE]
n_conv = n_skip = 0
B = 2_000_000
for s0 in range(0, len(sel), B):
    e = sel[s0:s0 + B]
    i = pre[e].astype(np.int64)
    t = gtype[post[e]]
    lo = np.searchsorted(tkey, (i * Tp1).astype(np.int32))
    hi = np.searchsorted(tkey, ((i + 1) * Tp1).astype(np.int32))
    d = hi - lo
    ok = d > 0
    e, i, t, lo, d = e[ok], i[ok], t[ok], lo[ok], d[ok]
    uu = tnbr[lo + (rng.random(len(lo)) * d).astype(np.int64)].astype(np.int64)
    ku = (uu * Tp1 + t).astype(np.int32)
    lo2 = np.searchsorted(tkey, ku)
    hi2 = np.searchsorted(tkey, (ku.astype(np.int64) + 1).astype(np.int32))
    d2 = hi2 - lo2
    ok = d2 > 0
    e, i = e[ok], i[ok]
    lo2, d2 = lo2[ok], d2[ok]
    j2 = tnbr[lo2 + (rng.random(len(lo2)) * d2).astype(np.int64)].astype(np.int64)
    ok = j2 != i
    e, i, j2 = e[ok], i[ok], j2[ok]
    kk = np.minimum(i, j2) * Nn + np.maximum(i, j2)
    p = np.searchsorted(ptab, kk)
    p = np.clip(p, 0, len(ptab) - 1)
    free = ptab[p] != kk
    n_skip += int((~free).sum())
    e, j2 = e[free], j2[free]
    post[e] = j2.astype(np.int32)
    n_conv += len(e)
    print(f"  {s0 + len(e):,}/{len(sel):,} 重连 {n_conv:,} 跳过 {n_skip:,} ({time.time()-t0:.0f}s)", flush=True)

post.tofile(OUTP + "_post.i32")
json.dump({"eligible": int(len(elig_idx)), "converted": n_conv, "skipped": n_skip,
           "p_close": P_CLOSE}, open(ROOT + "rewire_log.json", "w"))
print(f"DONE: 重连 {n_conv:,} ({n_conv/max(len(sel),1):.1%}), 跳过 {n_skip:,}, {time.time()-t0:.0f}s", flush=True)
