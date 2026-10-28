#!/usr/bin/env python3
"""Stage 45: Motif 全谱 —— 16 类 triad census, 真实 vs v4.1 (S=1)
采样估计 (Horvitz-Thompson):
  - 楔采样 (中心 ∝ C(deg,2), 两邻居均匀): 覆盖 ≥2 条无向边的 triad (开楔权1, 三角权3)
  - 均匀三元组采样: 覆盖 003/102/012
分类: 64 状态查表 (s_uv*16+s_uw*4+s_vw -> MAN 类)
内存: 顺序处理 real 再 gen, 每图建 无向对->状态 排序表
"""
import numpy as np
import json, time, os, gc

ROOT = "/mnt/agents/output/果蝇生成器/"
NWED = 2_000_000
NTRI = 2_000_000
CLASSES = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
           "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]
CID = {c: i for i, c in enumerate(CLASSES)}

def log(*a): print(*a, flush=True)

# ---------- 64 状态分类表 ----------
def classify(s01, s02, s12):
    """节点 0,1,2; s_ij: 0 无, 1 i->j, 2 j->i, 3 互. 返回 MAN 类名."""
    st = [s01, s02, s12]
    prs = [(0, 1), (0, 2), (1, 2)]
    m = sum(1 for s in st if s == 3)
    a = sum(1 for s in st if s in (1, 2))
    n = sum(1 for s in st if s == 0)
    out_d = [0, 0, 0]; in_d = [0, 0, 0]
    for (i, j), s in zip(prs, st):
        if s == 1: out_d[i] += 1; in_d[j] += 1
        elif s == 2: out_d[j] += 1; in_d[i] += 1
        elif s == 3: out_d[i] += 1; in_d[j] += 1; out_d[j] += 1; in_d[i] += 1
    if (m, a, n) == (0, 0, 3): return "003"
    if (m, a, n) == (0, 1, 2): return "102"
    if (m, a, n) == (1, 0, 2): return "012"
    if (m, a, n) == (2, 0, 1): return "201"
    if (m, a, n) == (2, 1, 0): return "210"
    if (m, a, n) == (3, 0, 0): return "300"
    if (m, a, n) == (0, 2, 1):
        if 2 in out_d: return "021U"
        if 2 in in_d: return "021D"
        return "021C"
    if (m, a, n) == (1, 1, 1):
        # 互对对 (p,q), 不对称边的源若在互对内 -> U, 否则 D
        for (i, j), s in zip(prs, st):
            if s in (1, 2):
                src = i if s == 1 else j
                mp = [k for k, s2 in enumerate(st) if s2 == 3][0]
                return "111U" if src in prs[mp] else "111D"
    if (m, a, n) == (0, 3, 0):
        return "030T" if 2 in out_d else "030C"
    if (m, a, n) == (1, 2, 0):
        mp = [k for k, s2 in enumerate(st) if s2 == 3][0]
        dy = set(prs[mp])
        srcs = [ (i if s == 1 else j) for (i, j), s in zip(prs, st) if s in (1, 2)]
        tgts = [ (j if s == 1 else i) for (i, j), s in zip(prs, st) if s in (1, 2)]
        if all(s_ in dy for s_ in srcs): return "120U"
        if all(t_ in dy for t_ in tgts): return "120D"
        return "120C"
    raise ValueError((m, a, n))

TAB = np.zeros(64, dtype=np.int8)
for x in range(64):
    TAB[x] = CID[classify(x // 16, (x // 4) % 4, x % 4)]
# 自检
assert classify(3, 1, 0) == "111U"   # 0<->1, 0->2
assert classify(3, 2, 0) == "111D"   # 0<->1, 2->0
assert classify(1, 0, 1) == "021C"   # 0->1, 1->2 (链)
assert classify(1, 1, 0) == "021U"   # 0->1, 0->2
assert classify(2, 2, 0) == "021D"   # 1->0, 2->0
assert classify(1, 2, 2) == "030T"   # 0->1, 0->2, 1->2 传递
assert classify(1, 2, 1) == "030C"   # 0->1, 2->0, 1->2 三环
assert classify(3, 1, 1) == "120U"   # 0<->1, 0->2, 1->2
assert classify(3, 2, 2) == "120D"   # 0<->1, 2->0, 2->1
assert classify(3, 1, 2) == "120C"   # 0<->1, 0->2, 2->1
log("triad table self-test passed")

# ---------- 单图 census ----------
def census(pre, post, N, tag):
    """pre/post: int32 ndarray (已物化). 全程省内存: 原地运算 + 分块"""
    t0 = time.time()
    E = len(pre)
    k = pre.astype(np.int64)              # 200MB
    k *= N; k += post                     # 原地
    del pre, post; gc.collect()
    k.sort()                              # 原地 introsort, 无大临时
    keep = np.ones(E, dtype=bool)
    np.not_equal(k[1:], k[:-1], out=keep[1:])
    ku = k[keep]
    del k, keep; gc.collect()
    E_pairs = len(ku)
    # 分块算 und key + 方向位, 避免同时持有 5 个 int64 长数组
    und_all = np.empty(E_pairs, dtype=np.int64)
    bit_all = np.empty(E_pairs, dtype=np.int8)
    CH = 4_000_000
    for s in range(0, E_pairs, CH):
        e_ = min(s + CH, E_pairs)
        kk = ku[s:e_]
        a = kk // N                       # pre
        b = kk - a * N                    # post
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        und_all[s:e_] = mn * N + mx
        bit_all[s:e_] = np.where(a < b, 1, 2)
        del kk, a, b, mn, mx
    del ku; gc.collect()
    o = np.argsort(und_all)               # int64 idx, ~160MB
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
    n_mutual = int((state == 3).sum())
    recip = 2 * n_mutual / E_pairs        # 边级互惠: 有反向的边占比
    log(f"{tag}: N={N:,} directed_pairs={E_pairs:,} und_pairs={len(u_und):,} recip={recip:.4f} ({time.time()-t0:.0f}s)")

    def pstate(x, y):
        mn_ = np.minimum(x, y).astype(np.int64); mx_ = np.maximum(x, y).astype(np.int64)
        kk = mn_ * N + mx_
        i = np.searchsorted(u_und, kk); i = np.clip(i, 0, len(u_und) - 1)
        hit = u_und[i] == kk
        st = np.where(hit, state[i], 0).astype(np.int64)
        sw = np.zeros(len(x), dtype=np.int64)
        fwd = (st == 1) | (st == 3); rev = (st == 2) | (st == 3)
        sw = np.where(fwd & (x < y), 1, 0) + np.where(rev & (x > y), 1, 0) \
           + np.where(fwd & (x > y), 2, 0) + np.where(rev & (x < y), 2, 0)
        return sw.astype(np.int64)

    # 无向 CSR: 打包 int64 key 原地排序 (无 argsort 索引数组), deg 用 searchsorted
    dst_key = (u_und % N) * N + (u_und // N)
    key = np.r_[u_und, dst_key]
    del dst_key; gc.collect()
    key.sort()                            # 原地
    nbr = (key % N).astype(np.int32)
    bounds = np.searchsorted(key, np.arange(N + 1, dtype=np.int64) * N)
    deg = np.diff(bounds)
    off = bounds
    del key, bounds; gc.collect()
    log(f"{tag}: CSR built ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(42)
    # ---- 楔采样 ----
    cw = deg * (deg - 1) // 2
    Wtot = int(cw.sum())
    cdf = np.cumsum(cw)
    u = np.searchsorted(cdf, rng.integers(0, Wtot, NWED))
    b0 = off[u]; dg = deg[u]
    i1 = rng.integers(0, dg); i2 = (i1 + 1 + rng.integers(0, dg - 1)) % dg
    v = nbr[b0 + i1].astype(np.int64); w = nbr[b0 + i2].astype(np.int64)
    s_uv = pstate(u, v); s_uw = pstate(u, w); s_vw = pstate(v, w)
    cls = TAB[s_uv * 16 + s_uw * 4 + s_vw]
    closed = s_vw > 0
    wt = np.where(closed, 3, 1)           # 三角含 3 楔, 开楔含 1
    cnt = np.zeros(16, dtype=np.float64)
    np.add.at(cnt, cls, wt)
    est_wedge = cnt / NWED * Wtot
    log(f"{tag}: wedge census done ({time.time()-t0:.0f}s)")

    # ---- 均匀三元组采样 (003/012/102) ----
    tri = rng.integers(0, N, (NTRI, 3)).astype(np.int64)
    ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[ok]
    s01 = pstate(tri[:, 0], tri[:, 1]); s02 = pstate(tri[:, 0], tri[:, 2]); s12 = pstate(tri[:, 1], tri[:, 2])
    cls2 = TAB[s01 * 16 + s02 * 4 + s12]
    cnt2 = np.bincount(cls2, minlength=16).astype(np.float64)
    C3 = N * (N - 1) * (N - 2) / 6.0
    est_tri = cnt2 / len(tri) * C3
    log(f"{tag}: triple census done ({time.time()-t0:.0f}s)")

    est = est_wedge.copy()
    for c in ["003", "012", "102"]:
        est[CID[c]] = est_tri[CID[c]]
    frac = est / est.sum()
    return {"N": int(N), "directed_pairs": int(E_pairs), "recip": recip,
            "counts": {c: float(est[CID[c]]) for c in CLASSES},
            "frac": {c: float(frac[CID[c]]) for c in CLASSES}}

# ---------- real ----------
import sys
WHICH = sys.argv[1] if len(sys.argv) > 1 else "both"
OUT = ROOT + "data_census/"
node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
def drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass

rpre = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="r")
rpost = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="r")
_r = np.asarray(rpre); _p = np.asarray(rpost)
del rpre, rpost; gc.collect()
drop_cache(OUT + "e_pre.i32"); drop_cache(OUT + "e_post.i32")
# 重标定到 typed 子图 (与 gen 的 N 口径一致: 全部为 typed 节点)
if WHICH in ("real", "both"):
    tm = node_type >= 0
    remap = np.cumsum(tm, dtype=np.int64) - 1
    _r = remap[_r].astype(np.int32); _p = remap[_p].astype(np.int32)
    res_real = census(_r, _p, int(tm.sum()), "real")
    json.dump(res_real, open(ROOT + "triad_real.json", "w"), indent=1)
    del res_real; gc.collect()

# ---------- gen ----------
if WHICH in ("gen", "both"):
    GPREFIX = sys.argv[2] if len(sys.argv) > 2 else "/tmp/flydata/gen/genv4_S1"
    GOUT = sys.argv[3] if len(sys.argv) > 3 else ROOT + "triad_v4.json"
    GTAG = os.path.basename(GPREFIX)
    g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
    N_gen = int(g['type_count'].astype(np.int64).sum())
    gpre = np.fromfile(GPREFIX + "_pre.i32", dtype=np.int32)
    gpost = np.fromfile(GPREFIX + "_post.i32", dtype=np.int32)
    res_gen = census(gpre, gpost, N_gen, GTAG)
    json.dump(res_gen, open(GOUT, "w"), indent=1)
log("TRIAD DONE")
