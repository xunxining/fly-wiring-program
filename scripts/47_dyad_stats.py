#!/usr/bin/env python3
"""Stage 47: 二元亲和残差统计 —— 真实互惠 dyad 的结构
产出:
1) 每无序类型对 {s,t}: n_single / n_mutual (dyad 级, 生成器 v4.2 的配额输入)
2) 互惠 vs 相位: 普查核对内 P(mutual|connected, cos(Δθ+φ)) 形状 (检验残差是否独立于坐标)
3) 谱系检验 (若注释含 hemilineage 列): 同型内 同谱系 vs 异谱系 的互惠率比
"""
import numpy as np
import json, gc, os, sys

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
T = 11751

def log(*a): print(*a, flush=True)

def drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass

node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
tm = node_type >= 0
remap = np.cumsum(tm, dtype=np.int64) - 1
tnode = node_type[tm]                      # typed 子图类型
N = int(tm.sum())
rpre = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="r")
rpost = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="r")
pre = remap[np.asarray(rpre)].astype(np.int64)
post = remap[np.asarray(rpost)].astype(np.int64)
del rpre, rpost, remap; gc.collect()
drop_cache(OUT + "e_pre.i32"); drop_cache(OUT + "e_post.i32")
log(f"typed N={N:,} edges={len(pre):,}")

# ---- 有向对去重 + und 状态 ----
k = pre * N + post
del pre, post; gc.collect()
k.sort()
keep = np.ones(len(k), dtype=bool)
np.not_equal(k[1:], k[:-1], out=keep[1:])
ku = k[keep]; del k, keep; gc.collect()
E_pairs = len(ku)
und_all = np.empty(E_pairs, dtype=np.int64)
bit_all = np.empty(E_pairs, dtype=np.int8)
CH = 4_000_000
for s in range(0, E_pairs, CH):
    e_ = min(s + CH, E_pairs)
    kk = ku[s:e_]
    a = kk // N; b = kk - a * N
    mn = np.minimum(a, b); mx = np.maximum(a, b)
    und_all[s:e_] = mn * N + mx
    bit_all[s:e_] = np.where(a < b, 1, 2)
    del kk, a, b, mn, mx
del ku; gc.collect()
o = np.argsort(und_all)
und_s = und_all[o]; bit_s = bit_all[o]
del und_all, bit_all, o; gc.collect()
first = np.ones(len(und_s), dtype=bool)
np.not_equal(und_s[1:], und_s[:-1], out=first[1:])
u_und = und_s[first]
fi = np.flatnonzero(first)
state = bit_s[fi].copy()
li = np.r_[fi[1:], len(und_s)] - 1
state |= bit_s[li]
del und_s, bit_s, first, fi, li; gc.collect()
log(f"und dyads={len(u_und):,} mutual={int((state==3).sum()):,}")

# ---- 1) 每无序类型对 {s,t} 统计 (排序分组, 不用 T*T 稠密表) ----
i_mn = (u_und // N).astype(np.int32)
i_mx = (u_und % N).astype(np.int32)
st_ = tnode[i_mn].astype(np.int64); tt_ = tnode[i_mx].astype(np.int64)
tmn = np.minimum(st_, tt_); tmx = np.maximum(st_, tt_)
tp_key = tmn * T + tmx
del tmn, tmx; gc.collect()
o_tp = np.argsort(tp_key)
tp_sorted = tp_key[o_tp]
imn_s = i_mn[o_tp]; imx_s = i_mx[o_tp]; st_s = state[o_tp]
del tp_key, o_tp; gc.collect()
first_ = np.ones(len(tp_sorted), dtype=bool)
np.not_equal(tp_sorted[1:], tp_sorted[:-1], out=first_[1:])
fi_ = np.flatnonzero(first_)
uk_pair = tp_sorted[fi_]
li_ = np.r_[fi_[1:], len(tp_sorted)]
is_mut = (st_s == 3).astype(np.int64)
cs_mut = np.cumsum(is_mut)                 # inclusive
n_mut = cs_mut[li_ - 1] - cs_mut[fi_] + is_mut[fi_]
n_tot = (li_ - fi_).astype(np.int64)
n_sgl = n_tot - n_mut
log(f"类型对 (dyad 连通): {len(uk_pair):,}, 总体 P(mutual|connected)={n_mut.sum()/n_tot.sum():.4f}")
np.savez_compressed(ROOT + "dyad_stats.npz",
                    uk=uk_pair.astype(np.int64), n_single=n_sgl.astype(np.int32),
                    n_mutual=n_mut.astype(np.int32))
del cs_mut, is_mut, n_tot, fi_, li_, first_; gc.collect()

# ---- 2) 互惠 vs 相位 (普查核对) ----
_pk = np.load(OUT + "poskernel_cal.npz")
pk = {kk: _pk[kk] for kk in _pk.files}
# 原图 id -> typed 重标定: poskernel 存的 ids 是原始 node_ids 还是 typed 索引? 36 导出时存的是
# census 节点索引 (0..N_real-1 全图 id). 需 remap. 重建 typed remap:
remap_full = np.cumsum(tm, dtype=np.int64) - 1   # 全图 id -> typed id
id2typed = remap_full

# 预分组已在上方完成 (tp_sorted / imn_s / imx_s / st_s)
kmap_rows = [(int(pk['s'][kidx]), int(pk['t'][kidx]), kidx) for kidx in range(len(pk['s']))]
NBIN = 6
bin_mut = np.zeros(NBIN); bin_tot = np.zeros(NBIN)
n_pairs_used = 0
for (s_, t_, kidx) in kmap_rows:
    o0, o1 = int(pk['th_off'][kidx]), int(pk['th_off'][kidx + 1])
    ids = pk['ids'][o0:o1]; th = pk['theta'][o0:o1]
    nsr = int(pk['ns'][kidx])
    ids_p, ids_q = ids[:nsr], ids[nsr:]
    th_p, th_q = th[:nsr], th[nsr:]
    if len(ids_p) < 20 or len(ids_q) < 20:
        continue
    phi = float(pk['phi'][kidx])
    tip = id2typed[ids_p]; tiq = id2typed[ids_q]
    tpk = (min(s_, t_)) * T + max(s_, t_)
    b0 = np.searchsorted(tp_sorted, tpk); b1 = np.searchsorted(tp_sorted, tpk + 1)
    if b1 - b0 < 200:
        continue
    dm = imn_s[b0:b1]; dx = imx_s[b0:b1]; sm = st_s[b0:b1]
    # 相位查表 (typed -> 相位): 用 searchsorted 建映射
    op = np.argsort(tip); oq = np.argsort(tiq)
    def lookup(arr_ids, arr_th, o, x):
        i = np.searchsorted(arr_ids[o], x)
        i = np.clip(i, 0, len(arr_ids) - 1)
        ok = arr_ids[o][i] == x
        return np.where(ok, arr_th[o][i], np.nan)
    thm_p = lookup(tip, th_p, op, dm)   # mn 侧若在 s 池
    thm_q = lookup(tiq, th_q, oq, dm)   # mn 侧若在 t 池
    thx_p = lookup(tip, th_p, op, dx)
    thx_q = lookup(tiq, th_q, oq, dx)
    # dyad (mn,mx): 情况A mn∈s,mx∈t; 情况B mn∈t,mx∈s (s≠t); 自对 s==t 两侧同池
    if s_ == t_:
        dth = thm_p - thx_p
    else:
        dA = ~np.isnan(thm_p) & ~np.isnan(thx_q)
        dth = np.where(dA, thm_p - thx_q, thx_p - thm_q)
    ok = ~np.isnan(dth)
    if ok.sum() < 200:
        continue
    cosv = np.cos(dth[ok] + phi)
    sm2 = sm[ok]
    bi = np.clip(((cosv + 1) / 2 * NBIN).astype(int), 0, NBIN - 1)
    for b in range(NBIN):
        mb = bi == b
        bin_tot[b] += mb.sum()
        bin_mut[b] += (sm2[mb] == 3).sum()
    n_pairs_used += 1
log(f"相位-互惠分析: {n_pairs_used} 个核对, dyads={int(bin_tot.sum()):,}")
curve = bin_mut / np.maximum(bin_tot, 1)
log("P(mutual|connected, cos bin):", np.round(curve, 4), " bins: [-1..1] 6 等分")

# ---- 3) 谱系检验 (可选) ----
lineage = {"available": False}
try:
    import pandas as pd
    ann = pd.read_feather(ROOT + "data/body-annotations.feather")
    cols = [c for c in ann.columns if "hemi" in c.lower() or "lineage" in c.lower() or "group" in c.lower()]
    log("注释列含谱系候选:", cols)
    if cols:
        c0 = cols[0]
        node_ids = np.fromfile(OUT + "node_ids.i64", dtype=np.int64)
        lin_map = dict(zip(ann["bodyId"], ann[c0].astype(str)))
        lin = np.array([lin_map.get(int(b), "") for b in node_ids], dtype=object)
        lin_t = lin[tm]
        same = lin_t[i_mn] == lin_t[i_mx]
        valid = (lin_t[i_mn] != "") & (st_ == tt_)   # 同型对内比谱系
        for grp, msk in [("同型同谱系", valid & same), ("同型异谱系", valid & ~same)]:
            mm = state[msk]
            lineage[grp] = {"dyads": int(msk.sum()), "p_mutual": float((mm == 3).mean())}
        lineage["available"] = True
        log("谱系:", lineage)
except Exception as ex:
    log("谱系检验跳过:", repr(ex)[:120])

json.dump({"p_mutual_overall": float(curve.mean()),
           "phase_mutual_curve": curve.tolist(), "phase_bin_tot": bin_tot.tolist(),
           "n_pairs_used": n_pairs_used, "lineage": lineage},
          open(ROOT + "dyad_phase_lineage.json", "w"), indent=1)
log("DYAD STATS DONE")
