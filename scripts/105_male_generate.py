#!/usr/bin/env python3
"""75_female_generate.py — 雌性生成器 (v6-lite 架构转移: 配额骨架+CELLQ+ACT+空间核)
所有参数从雌性数据重学 (74_female_genome.py), 不含任何雄性参数。
用法: CELLQ=1 SPATIAL=1 ACT=1 python3 75_female_generate.py <out_prefix>
      全关 = 配额均匀基线 (γ 对照)
"""
import numpy as np, os, sys, time, gc

D = os.environ.get("MALE_D", "/mnt/agents/output/果蝇生成器/data_male/")
rng = np.random.default_rng(13)
RES = 256
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)

CELLQ = os.environ.get("CELLQ", "0") == "1"
SPATIAL = os.environ.get("SPATIAL", "0") == "1"
ACT = os.environ.get("ACT", "0") == "1"
NPOV = os.environ.get("NPOV", "0") == "1"          # 神经毡重叠门控/加权
NPB = float(os.environ.get("NPB", "0"))            # overlap 幂指数 (0=纯硬门)
LAT = os.environ.get("LAT", "0") == "1"            # 视叶二维晶格核
LATSHUF = os.environ.get("LATSHUFFLE", "0") == "1" # 消融: 坐标洗牌
LATOS = int(os.environ.get("LATOS", "96"))         # 晶格对的过采样
out_prefix = sys.argv[1]
SEGPAIRS = int(os.environ.get("SEGPAIRS", "0"))   # >0: 分段模式
CKPT = out_prefix + ".ckpt.pkl"
import pickle, gzip

def loadc(p):
    """npz 全量读入内存并关闭句柄; 挂载断开自动重试"""
    for att in range(100):
        try:
            z = np.load(p, allow_pickle=True)
            d = {k: z[k] for k in z.files}
            z.close()
            return d
        except OSError as e:
            log(f"npz 读取重试{att} {p.split('/')[-1]}: {str(e)[:40]}"); time.sleep(15)
    raise RuntimeError("npz 读失败: " + p)

# popcount 表
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
def popc(a):
    a = np.asarray(a, np.uint64)
    b = np.frombuffer(a.tobytes(), np.uint8).reshape(a.shape + (8,))
    return POP[b].sum(-1)

# ---------- 基因组 ----------
g = loadc(D + "genome_m.npz")
uk, cnt = g['uk'].astype(np.int64), g['cnt'].astype(np.int64)
tc = g['type_count'].astype(np.int64)
T = len(tc); N = int(tc.sum())
tp, tq = uk // T, uk % T
toff = np.concatenate([[0], np.cumsum(tc)])
lam = cnt.astype(np.float64)
log(f"基因组: {len(uk):,} 对, N={N:,}, E={int(cnt.sum()):,}")

node_sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
NSC = int(node_sc.max()) + 1
xyz_g = np.fromfile(D + "xyz.f32", dtype=np.float32).reshape(N, 3)

# ---------- 权重池 (sc,sc) ----------
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
w = np.fromfile(D + "e_w.i32", dtype=np.int32)
wkey = node_sc[ep].astype(np.int64) * NSC + node_sc[eq]
w_emp = np.ones((NSC * NSC, RES), dtype=np.int32)
ws = np.asarray(w)
for k_ in np.unique(wkey):
    vals = ws[wkey == k_]
    w_emp[k_] = np.resize(vals, RES) if len(vals) < RES else vals[rng.choice(len(vals), RES, replace=False)]
del ep, eq, w, ws, wkey; gc.collect()
log("权重池完成")

# ---------- ACT ----------
cw_cache = {}
if ACT:
    a_in = np.fromfile(D + "act_in.f32", dtype=np.float32).astype(np.float64)
    def act_cand(ti_, shape):
        if ti_ not in cw_cache:
            seg = slice(toff[ti_], toff[ti_ + 1])
            cw_cache[ti_] = np.cumsum(a_in[seg])
        cw = cw_cache[ti_]
        return np.searchsorted(cw, rng.random(shape) * cw[-1])
    log("ACT 启用")

# ---------- NPOV (神经毡重叠) ----------
if NPOV:
    _npz = loadc(D + "np_masks_m2.npz")
    OUTM = _npz["OUT"]; INM = _npz["IN"]          # (N,3) uint64, 146 ROI
    def ov3(g1, g2):
        """g1,g2 可广播索引 -> 重叠计数"""
        return (popc(OUTM[g1][..., 0] & INM[g2][..., 0]) +
                popc(OUTM[g1][..., 1] & INM[g2][..., 1]) +
                popc(OUTM[g1][..., 2] & INM[g2][..., 2]))
    CAP = 8.0
    log(f"NPOV 启用: 硬门 overlap>0, beta={NPB}")

    def ov_weight(si, ti_, pi_sub, cand):
        """pi_sub: (k,) 源细胞内局部索引; cand: (k,OS) 候选局部索引 -> (k,OS) 权重"""
        g1 = (toff[si] + pi_sub)[:, None]
        g2 = toff[ti_] + cand
        ol = ov3(g1, g2)
        if NPB > 0:
            return np.minimum(ol, CAP) ** NPB
        return (ol > 0).astype(np.float64)

    def npov_sample(si, ti_, nt_, e, pi_):
        """非 SPATIAL 路径的重叠门控采样 (仅硬门)"""
        q = act_cand(ti_, e).astype(np.int32) if ACT else rng.integers(0, nt_, e, dtype=np.int32)
        g1 = toff[si] + pi_
        for _ in range(8):
            ol = ov3(g1, toff[ti_] + q)
            bad = ol == 0
            if not bad.any(): break
            nb = int(bad.sum())
            q[bad] = act_cand(ti_, nb).astype(np.int32) if ACT else rng.integers(0, nt_, nb, dtype=np.int32)
        return q

# ---------- SPATIAL ----------
if SPATIAL:
    sk = loadc(D + "spatial_kernel_m.npz")
    sp_bins = sk['bins']; sp_uk = sk['big_uk']
    sp_wbig = sk['wbig']; sp_wmax = sk['wmax_big']
    sp_wg = sk['w_glob']; sp_wmax_g = float(sk['wmax_glob'])
    NBSP = len(sp_bins) - 1
    n_sp_left = [0]
    log(f"SPATIAL 启用: {len(sp_uk):,} 大对核")

# ---------- LAT: 视叶二维晶格核 ----------
LATV = os.environ.get("LATV", "1")                 # 2 = 扩展坐标 (感光/VPN) + R=20 核
LATF = os.environ.get("LATF", "")                  # "m2" = P9b: soma直测坐标 + m2 snap 核 (§41)
LAT_SUF = "_m2" if LATF == "m2" else "_m"
if LAT:
    lk = loadc(D + f"lat_snap{LAT_SUF}.npz")
    if LATV == "1":
        l_uk = lk['big_uk']; l_wbig = lk['wbig']; l_wmax = lk['wmax_big']
        l_wg = lk['w_glob']; l_wmax_g = float(lk['wmax_glob'])
    else:
        l_uk = lk['big_uk']; l_wmax_g = 1.0        # v2 仅供 snap2 使用
    LR = int(lk['R']); LNB1 = 2 * LR + 1
    log(f"LAT 文件: lat_snap{LAT_SUF}.npz / lat2q{LAT_SUF}.npz (LATF={LATF or 'default'}, R={LR})")
    zq = loadc(D + f"lat2q{LAT_SUF}.npz")
    PQ_g = zq['pq'].astype(np.float64); HEM_g = zq['hem']
    if LATSHUF:
        ocells = np.where(HEM_g != 0)[0]
        perm = rng.permutation(len(ocells))
        PQ_g[ocells] = PQ_g[ocells[perm]]; HEM_g[ocells] = HEM_g[ocells[perm]]
        log("LATSHUFFLE: 坐标已洗牌 (消融对照)")
    if os.environ.get("LATPOS", "") == "rand":
        # P7 对照: 连续均匀随机位置 (bbox 内), 半球/类型/Δ 表不动
        oc = HEM_g != 0
        r7 = np.random.default_rng(7)
        mn = np.nanmin(PQ_g[oc], axis=0); mx = np.nanmax(PQ_g[oc], axis=0)
        PQ_g[oc] = r7.uniform(mn, mx, size=(int(oc.sum()), 2))
        log(f"LATPOS=rand: {int(oc.sum())} 细胞坐标已替换为 bbox 均匀随机 (P7 对照)")
    # 型级可用标志: >90% 细胞有坐标
    has_c = HEM_g != 0
    t_lat = np.zeros(T, bool)
    for t_ in range(T):
        seg = slice(toff[t_], toff[t_ + 1])
        n_ = toff[t_ + 1] - toff[t_]
        t_lat[t_] = n_ > 0 and has_c[seg].mean() > 0.9
    log(f"LAT 启用: {len(l_uk):,} 大对核, 视叶型 {t_lat.sum():,}")

# ---------- SNAP: 直接晶格放置 (P6) ----------
LATMODE = os.environ.get("LATMODE", "soft")
SNAP = LAT and LATMODE in ("snap", "snap2")
SNAP2 = LAT and LATMODE == "snap2"          # 配额填充: 模态偏移 + 最近 n 个去重
if SNAP:
    sn = loadc(D + f"lat_snap{LAT_SUF}.npz")
    sn_uk = sn['big_uk']; sn_cr = sn['crbig']; sn_cr_g = sn['cr_glob']
    from scipy.spatial import cKDTree
    tree_cache = {}
    def type_tree(t_, h_):
        kk = (t_, h_)
        if kk not in tree_cache:
            seg = np.arange(toff[t_], toff[t_ + 1])
            mk = has_c[seg] & (HEM_g[seg] == h_)
            pts = PQ_g[seg[mk]]                                   # (n,2) float64
            tree_cache[kk] = (cKDTree(pts), (seg[mk] - toff[t_]).astype(np.int32))
        return tree_cache[kk]
    n_snap_left = [0]
    log("SNAP 启用: k-d 最近列吸附")

    def snap_draw(si, ti_, nt_, m, pi):
        key = np.int64(si) * T + np.int64(ti_)
        r2 = int(np.searchsorted(sn_uk, key))
        cr = sn_cr[r2] if (r2 < len(sn_uk) and sn_uk[r2] == key) else sn_cr_g
        gs = (toff[si] + pi).astype(np.int64)
        hs = HEM_g[gs].astype(np.int64); oks = has_c[gs]
        u = rng.random(m)
        bb = np.searchsorted(cr, u)
        dp = (bb // LNB1 - LR).astype(np.float64); dq = (bb % LNB1 - LR).astype(np.float64)
        is_cross = bb == LNB1 * LNB1 + 1
        over = bb == LNB1 * LNB1
        tgt_hem = np.where(is_cross, -hs, hs)                     # 跨半球: 同位镜像
        fall = (~oks) | over
        q = np.empty(m, np.int32)
        done = np.zeros(m, bool)
        for h_ in (-1, 1):
            sub = np.where(~fall & (tgt_hem == h_))[0]
            if len(sub) == 0: continue
            tree, loc = type_tree(ti_, h_)
            if len(loc) == 0: continue
            ptgt = PQ_g[gs[sub]] + np.column_stack([dp[sub], dq[sub]])
            _, ii = tree.query(ptgt, k=1)
            q[sub] = loc[ii]
            done[sub] = True
        todo = np.where(~done)[0]
        if len(todo):
            q[todo] = act_cand(ti_, len(todo)).astype(np.int32) if ACT else \
                rng.integers(0, nt_, len(todo), dtype=np.int32)
            n_snap_left[0] += len(todo)
        return q

    # ---- SNAP2: 配额填充 —— 每源细胞连向 模态偏移处最近的 n 个去重目标 ----
    NB21 = LNB1 * LNB1
    def pair_mode(key):
        r2 = int(np.searchsorted(sn_uk, key))
        cr = sn_cr[r2] if (r2 < len(sn_uk) and sn_uk[r2] == key) else sn_cr_g
        pr = np.diff(np.concatenate([[0.0], cr]))
        bm = int(np.argmax(pr))
        if bm == NB21 + 1: return None, None, True, False     # CROSS
        if bm == NB21: return None, None, False, True          # OVER
        return float(bm // LNB1 - LR), float(bm % LNB1 - LR), False, False

    def snap2_draw(si, ti_, nt_, m, pi):
        key = np.int64(si) * T + np.int64(ti_)
        dp0, dq0, is_cross, is_over = pair_mode(key)
        if dp0 is None: dp0, dq0 = 0.0, 0.0                     # 跨半球: 同位镜像; OVER 不参放置
        gs = (toff[si] + pi).astype(np.int64)
        oks = has_c[gs]
        # 源分组 (pi 由 np.repeat 产生, 组连续但未必有序)
        us, inv = np.unique(pi, return_inverse=True)
        cnts = np.bincount(inv).astype(np.int64)
        srcg = (toff[si] + us).astype(np.int64)
        oks_s = has_c[srcg]
        hs = HEM_g[srcg].astype(np.int64)
        if is_cross: hs = -hs
        kmax = int(min(max(cnts.max(), 1) + 2, 12))
        nn = np.full((len(us), kmax), -1, np.int32)
        for h_ in (-1, 1):
            sub = np.where(oks_s & (hs == h_))[0]
            if len(sub) == 0: continue
            tree, loc = type_tree(ti_, h_)
            if len(loc) == 0: continue
            kk = min(kmax, len(loc))
            ptgt = PQ_g[srcg[sub]] + np.array([dp0, dq0])
            _, ii = tree.query(ptgt, k=kk)
            if kk == 1: ii = ii[:, None]
            nn[sub, :kk] = loc[ii]
        # 每边在其源组内的秩
        ob = np.argsort(inv, kind="stable")
        gr = inv[ob]
        starts = np.flatnonzero(np.r_[True, gr[1:] != gr[:-1]])
        gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, m]))
        rank = np.arange(m) - starts[gid]
        q = np.empty(m, np.int32)
        tgt = nn[gr, np.minimum(rank, kmax - 1)]
        ok_e = (rank < kmax) & (tgt >= 0) & oks_s[gr] & (not is_over)
        q[ob[ok_e]] = tgt[ok_e]
        todo = ob[~ok_e]
        if len(todo):
            q[todo] = act_cand(ti_, len(todo)).astype(np.int32) if ACT else \
                rng.integers(0, nt_, len(todo), dtype=np.int32)
            n_snap_left[0] += len(todo)
        if os.environ.get("DEBUGQ", "0") == "1" and si == 5643 and ti_ == 1092:
            log(f"snap2dbg m={m} ok={ok_e.sum()} todo={len(todo)} kmax={kmax} "
                f"us={len(us)} cntmax={cnts.max()} unassigned={(q<0).sum()+(q>=nt_).sum()}")
        return q

# ---------- SPATIAL 抽样 ----------
if SPATIAL:
    def spatial_draw(si, ti_, ns_, nt_, m, pi):
        key = np.int64(si) * T + np.int64(ti_)
        r = int(np.searchsorted(sp_uk, key))
        if r < len(sp_uk) and sp_uk[r] == key:
            W = sp_wbig[r]; wmax = float(sp_wmax[r])
        else:
            W = sp_wg; wmax = sp_wmax_g
        tb0 = toff[ti_]
        OS = int(min(max(wmax * 1.6, 4), float(os.environ.get("SPATIALOS", "24"))))
        lat_ok = LAT and t_lat[si] and t_lat[ti_]
        if lat_ok:
            r2 = int(np.searchsorted(l_uk, key))
            if r2 < len(l_uk) and l_uk[r2] == key:
                W2 = l_wbig[r2]; lwmax = float(l_wmax[r2])
            else:
                W2 = l_wg; lwmax = l_wmax_g
            OS = max(OS, LATOS)
        q = np.empty(m, np.int32)
        todo = np.arange(m)
        srcx = xyz_g[toff[si] + pi]
        n_rounds = 6 if lat_ok else 4
        CH = 4096                                  # 分块控内存 (memcg 4GB)
        for _ in range(n_rounds):
            k = len(todo)
            if k == 0: break
            filled = np.zeros(k, bool)
            q_new = np.empty(k, np.int32)
            for c0 in range(0, k, CH):
                c1 = min(c0 + CH, k); kc = c1 - c0
                td = todo[c0:c1]
                cand = act_cand(ti_, (kc, OS)) if ACT else rng.integers(0, nt_, (kc, OS), dtype=np.int32)
                dx = xyz_g[tb0 + cand] - srcx[td][:, None, :]
                d = np.sqrt((dx * dx).sum(-1))
                b = np.clip(np.searchsorted(sp_bins, d, side="right") - 1, 0, NBSP - 1)
                ww = W[b].astype(np.float32)
                ww[np.isnan(d)] = 1.0
                wmax_e = wmax
                if NPOV:
                    ow = ov_weight(si, ti_, pi[td], cand)
                    if NPB > 0:
                        ww = ww * ow; wmax_e = wmax * np.maximum(ow.max(axis=1), 1.0)[:, None]
                    else:
                        ww = ww * ow
                if lat_ok:
                    gs = (toff[si] + pi[td]).astype(np.int64)
                    gc_ = (tb0 + cand).astype(np.int64)
                    hs_ = HEM_g[gs]; ht_ = HEM_g[gc_]
                    ok2 = (hs_ != 0)[:, None] & (ht_ != 0)
                    same = ok2 & (hs_[:, None] == ht_)
                    dpq = PQ_g[gc_] - PQ_g[gs][:, None, :]
                    dpr = np.rint(dpq[..., 0]).astype(np.int32)
                    dqr = np.rint(dpq[..., 1]).astype(np.int32)
                    inr = same & (np.abs(dpr) <= LR) & (np.abs(dqr) <= LR)
                    bb = np.full((kc, OS), LNB1 * LNB1 + 1, np.int32)   # 跨半球
                    bb[same] = LNB1 * LNB1                               # 同半球溢出
                    bb[inr] = (dpr[inr] + LR) * LNB1 + (dqr[inr] + LR)
                    wl = W2[bb]
                    wl[~ok2] = 1.0                                       # 无坐标: 中性
                    ww = ww * wl
                    wmax_e = wmax_e * lwmax
                acc = rng.random((kc, OS)) < (ww / wmax_e)
                h = acc.any(1)
                q_new[c0:c1][h] = cand[np.arange(kc)[h], acc[h].argmax(1)]
                filled[c0:c1] = h
            q[todo[filled]] = q_new[filled]
            todo = todo[~filled]
        if len(todo):
            q[todo] = rng.integers(0, nt_, len(todo), dtype=np.int32)
            n_sp_left[0] += len(todo)
        return q

# ---------- CELLQ ----------
if CELLQ:
    cq_cell = np.fromfile(D + "cq_cell.i32", dtype=np.int32).astype(np.int64)
    cq_t = np.fromfile(D + "cq_t.i32", dtype=np.int32).astype(np.int64)
    cq_cn = np.fromfile(D + "cq_cnt.i32", dtype=np.int32).astype(np.int64)
    cq_pk = (cq_cell // 1)  # placeholder
    cq_pk = (np.searchsorted(toff, cq_cell, side="right") - 1) * T + cq_t  # pair key via source type
    o2 = np.argsort(cq_pk, kind="stable")
    cq_pk = cq_pk[o2]; cq_cell = cq_cell[o2]; cq_t = cq_t[o2]; cq_cn = cq_cn[o2]
    uq2, st2 = np.unique(cq_pk, return_index=True)
    assert len(uq2) == len(uk) and (uq2 == uk).all(), "细胞配额对与基因组对不齐"
    cq_off = np.append(st2, len(cq_pk))
    log(f"CELLQ: {len(cq_pk):,} 条逐细胞配额")

# ---------- DYAD (互惠 dyad 配额, P8) ----------
DYAD = os.environ.get("DYAD", "0") in ("1", "2")
dy_Ruk = None; dy_R = None; pend = {}
n_dyad_fix = [0]; n_dyad_pairs = [0]; n_dyad_def = [0]; n_dyad_nocand = [0]; n_dyad_exh = [0]
if DYAD:
    _dz = loadc(D + "dyad_m.npz")
    dy_Ruk = _dz['uk'].astype(np.int64); dy_R = _dz['R'].astype(np.int64)
    log(f"DYAD 启用: {len(dy_Ruk):,} 对的互惠配额已载入")
    # P8b: 互惠潜能前瞻 (DYAD=2): 目标细胞对源类型有出配额者优先
    DYAD2 = os.environ.get("DYAD", "0") == "2"
    if DYAD2:
        _dc = loadc(D + "dyad_cap_m.npz")
        dy_Cuk = _dc['uk'].astype(np.int64); dy_C = _dc['cap'].astype(np.int64); dy_E = _dc['E'].astype(np.int64)
        # cq 反向对 -> 潜能细胞集合 (局部索引), 惰性缓存
        cap_seg = {}
        def cap_cells(si_, ti_):
            """放置对 (si_,ti_) 时: type-ti_ 中对 si_ 有出配额的细胞 (局部索引)"""
            kk = (si_, ti_)
            if kk not in cap_seg:
                rk_ = ti_ * T + si_
                pp = np.searchsorted(cq_pk, rk_)
                lo = pp; hi = pp
                while lo > 0 and cq_pk[lo - 1] == rk_: lo -= 1
                while hi < len(cq_pk) and cq_pk[hi] == rk_: hi += 1
                if hi > lo:
                    cap_seg[kk] = (cq_cell[lo:hi] - toff[ti_]).astype(np.int32)
                else:
                    cap_seg[kk] = np.zeros(0, np.int32)
            return cap_seg[kk]
        def cap_frac_of(pk_):
            pp = np.searchsorted(dy_Cuk, pk_)
            if pp < len(dy_Cuk) and dy_Cuk[pp] == pk_:
                return float(dy_C[pp]) / max(float(dy_E[pp]), 1.0)
            return 0.0
        log("DYAD2 前瞻: 互惠潜能目标优先已启用")
        n_prospect = [0]

        def dyad2_prospect(si, ti_, nt_, pi, qi):
            """将部分非潜能边的目标换为潜能细胞 (ACT 加权), 使潜能边占比达实值 rho。"""
            pk_ = si * T + ti_
            if dy_Rof(pk_) == 0 and dy_Rof(ti_ * T + si) == 0:
                return 0
            rho = cap_frac_of(pk_)
            if rho <= 0.0:
                return 0
            caps = np.sort(cap_cells(si, ti_))
            if len(caps) == 0:
                return 0
            e = len(qi)
            pp = np.minimum(np.searchsorted(caps, qi), len(caps) - 1)
            is_cap = caps[pp] == qi
            deficit = int(round(rho * e)) - int(is_cap.sum())
            if deficit <= 0:
                return 0
            nm = np.where(~is_cap)[0]
            n_sw = min(deficit, len(nm))
            sw = nm[rng.choice(len(nm), n_sw, replace=False)]
            # ACT 加权在潜能集内采样新目标
            if ACT:
                cw = np.cumsum(a_in[toff[ti_] + caps])
                newt = caps[np.minimum(np.searchsorted(cw, rng.random(n_sw) * cw[-1]), len(caps) - 1)]
            else:
                newt = caps[rng.integers(0, len(caps), n_sw)]
            # NPOV 门 (向量化)
            if NPOV:
                gsrc = toff[si] + pi[sw]; gtgt = toff[ti_] + newt
                ok = ov3(gsrc, gtgt) > 0
                sw = sw[ok]; newt = newt[ok]
                if len(sw) == 0:
                    return 0
            qi[sw] = newt.astype(np.int32)
            n_prospect[0] += len(sw)
            return len(sw)

    def dy_Rof(pk_):
        """有序对 key -> 互惠边配额 (无则 0)"""
        pp = np.searchsorted(dy_Ruk, pk_)
        if pp < len(dy_Ruk) and dy_Ruk[pp] == pk_:
            return int(dy_R[pp])
        return 0

    def dyad_repair(si, ti_, gp, gq, rp, rq, R_target):
        """gp/gq: 本对全局边 (i->j); rp/rq: 反向对已放置全局边 (j->i)。
        就地修改 gq, 使有反向的边数达到 R_target。返回修复数。"""
        e = len(gp)
        if e == 0 or len(rp) == 0 or R_target <= 0:
            return 0
        # 反向边键集合: j*N+i
        revk = np.sort(rp.astype(np.int64) * N + rq.astype(np.int64))
        pos = np.searchsorted(revk, gq.astype(np.int64) * N + gp.astype(np.int64))
        posc = np.minimum(pos, len(revk) - 1)
        is_mut = revk[posc] == (gq.astype(np.int64) * N + gp.astype(np.int64))
        deficit = R_target - int(is_mut.sum())
        if deficit <= 0:
            return 0
        n_dyad_def[0] += deficit
        # 每个源 i 的反向候选 j': rq==i 分组
        o2 = np.lexsort((rp, rq))
        rq_s = rq[o2]; rp_s = rp[o2]
        uq_i, b0 = np.unique(rq_s, return_index=True)
        b1 = np.append(b0[1:], len(rq_s))
        # 本对全部边按源分组 (一次排序, 供重边排除)
        o4 = np.argsort(gp, kind="stable")
        gp_s = gp[o4].astype(np.int64); gq_s = gq[o4]
        ua, ub = np.unique(gp_s, return_index=True)
        ub1 = np.append(ub[1:], len(gp_s))
        s2g = {int(v): k for k, v in enumerate(ua)}
        nm = np.where(~is_mut)[0]
        # 按源分组非互惠边
        o3 = np.argsort(gp[nm], kind="stable")
        nm = nm[o3]
        gps = gp[nm].astype(np.int64)
        us, st = np.unique(gps, return_index=True)
        st = np.append(st, len(nm))
        fixed = 0
        for k in range(len(us)):
            if fixed >= deficit: break
            i_g = int(us[k])
            ei = nm[st[k]:st[k+1]]                 # 该源的非互惠边
            pp = np.searchsorted(uq_i, i_g)
            if pp >= len(uq_i) or uq_i[pp] != i_g: continue
            cand = rp_s[b0[pp]:b1[pp]]             # j' 全局 (j'->i 已放置)
            if len(cand) == 0: n_dyad_nocand[0] += len(ei); continue
            # 排除该源已连的 j (避免重边): 该源当前目标集
            g2 = s2g.get(i_g)
            cur = set(gq_s[ub[g2]:ub1[g2]].tolist()) if g2 is not None else set()
            cand_l = [c for c in cand.tolist() if c not in cur]
            if not cand_l: n_dyad_exh[0] += len(ei); continue
            if len(cand_l) < len(ei): n_dyad_exh[0] += len(ei) - len(cand_l)
            n_take = min(len(ei), deficit - fixed, len(cand_l))
            if n_take <= 0: continue
            cand_a = np.array(cand_l, dtype=np.int64)
            # P8c: 晶格对齐优先 — 距模态偏移目标位 p_i+Δ_st 最近; 无坐标退回 xyz
            dp0 = dq0 = None
            if LAT:
                dp0, dq0, _cr, _ov = pair_mode(np.int64(si) * T + np.int64(ti_))
            if dp0 is not None and not _ov and has_c[i_g]:
                pstar = PQ_g[i_g] + np.array([dp0, dq0])
                okc = has_c[cand_a]
                d2 = np.where(okc, ((PQ_g[cand_a] - pstar) ** 2).sum(1),
                                  ((xyz_g[cand_a] - xyz_g[i_g]) ** 2).sum(1) * 1e6)
            else:
                d2 = ((xyz_g[cand_a] - xyz_g[i_g]) ** 2).sum(1)
            take = cand_a[np.argsort(d2)[:n_take]]
            # NPOV 门
            if NPOV:
                okm = []
                for c in take:
                    okm.append(ov3(i_g, c) > 0)
                take = take[np.array(okm, bool)]
                n_take = len(take)
                if n_take == 0: continue
            gq[ei[:n_take]] = take.astype(np.int32)
            fixed += n_take
        return fixed

# ---------- 主循环 ----------
# 暂存模式: 计算产物先写本地 /tmp(不受 portal 抖动影响), 由独立 flusher 进程同步到 portal
STAGE = os.environ.get("STAGE", "")                 # 非空则启用暂存
l_prefix = os.path.join(STAGE, os.path.basename(out_prefix)) if STAGE else out_prefix
if STAGE: os.makedirs(STAGE, exist_ok=True)
CKPT_R = CKPT                                     # ckpt 读取: portal(持久)
CKPT_W = l_prefix + ".ckpt.pkl" if STAGE else CKPT  # ckpt 写入: 暂存
wstate = {"pre": {"off": 0}, "post": {"off": 0}, "w": {"off": 0}}
def wbatch(kind, arr):
    path = l_prefix + "_" + kind + ".i32"
    data = arr.tobytes()
    stt = wstate[kind]
    if STAGE and stt["off"] > 0 and not os.path.exists(path):
        log("暂存被清空, 退出待重启")
        os._exit(3)
    for att in range(150):
        try:
            sz0 = os.path.getsize(path) if os.path.exists(path) else 0
            if sz0 > stt["off"]:
                os.truncate(path, stt["off"]); sz0 = stt["off"]
            with open(path, "r+b" if sz0 else "wb") as f:
                f.seek(stt["off"]); f.write(data); f.flush(); os.fsync(f.fileno())
            stt["off"] += len(data)
            return
        except OSError as e:
            log(f"写重试{att} {kind}: {str(e)[:40]}"); time.sleep(15)
    raise RuntimeError("portal 写失败 " + kind)
Bn = 100_000
MAXP = int(os.environ.get("MAXP", "0"))            # 冒烟测试: 仅 MAXP 对
STARTP = int(os.environ.get("STARTP", "0"))        # 冒烟测试: 起始对
n_pairs = min(len(uk), STARTP + MAXP) if MAXP else len(uk)
n_written = 0
t0 = time.time()
seg_start = STARTP
if SEGPAIRS:
    _ck_src = CKPT_W if (STAGE and os.path.exists(CKPT_W)) else CKPT_R
    if os.path.exists(_ck_src):
        for _a in range(200):
            try:
                raw = open(_ck_src, "rb").read()
                ck = pickle.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
                break
            except OSError as e:
                log(f"ckpt 读取重试{_a}: {str(e)[:40]}"); time.sleep(10)
        rng.bit_generator.state = ck["rng"]
        if "pend2" in ck:
            pk_a, ln_a, gp_a, gq_a = ck["pend2"]
            _o = 0
            for _k, _l in zip(pk_a, ln_a):
                pend[int(_k)] = (gp_a[_o:_o+_l], gq_a[_o:_o+_l]); _o += _l
        else:
            pend.update(ck["pend"])
        seg_start = ck["cursor"]
        for k_, v_ in ck.get("wstate", {}).items(): wstate[k_]["off"] = v_
        if STAGE:
            # 稀疏物化: 截断到 off (洞区字节不会被读: flusher 只追加 portal 缺口, portal 保证 >= off)
            for k_ in ("pre", "post", "w"):
                lp = l_prefix + "_" + k_ + ".i32"
                need = wstate[k_]["off"]
                have = os.path.getsize(lp) if os.path.exists(lp) else 0
                if have != need:
                    with open(lp, "wb") as fw: fw.truncate(need)
                    log(f"稀疏物化 {k_}: -> {need:,}")
        n_dyad_fix[0], n_dyad_def[0] = ck.get("ndf", 0), ck.get("ndd", 0)
        n_dyad_nocand[0], n_dyad_exh[0] = ck.get("ndnc", 0), ck.get("ndex", 0)
        n_dyad_pairs[0] = ck.get("ndp", 0)
        if DYAD and DYAD2: n_prospect[0] = ck.get("npro", 0)
        if SPATIAL: n_sp_left[0] = ck.get("nspl", 0)
        if SNAP: n_snap_left[0] = ck.get("nsnl", 0)
        log(f"断点恢复: cursor={seg_start:,} pend={len(pend):,} off={ck.get('wstate')}")
    seg_end = min(seg_start + SEGPAIRS, n_pairs)
else:
    seg_end = n_pairs
for i0 in range(seg_start, seg_end, Bn):
    i1 = min(i0 + Bn, len(uk), seg_end)
    ne = rng.poisson(lam[i0:i1])
    sel = np.ones(i1 - i0, bool) if CELLQ else (ne > 0)
    tb_s, tb_t, neb = tp[i0:i1][sel], tq[i0:i1][sel], ne[sel]
    sel_idx = np.flatnonzero(sel)
    buf_p, buf_q, buf_w = [], [], []
    for jj, (s, t, e) in enumerate(zip(tb_s, tb_t, neb)):
        si, ti_ = int(s), int(t)
        ns_, nt_ = int(tc[si]), int(tc[ti_])
        if CELLQ:
            ii = i0 + int(sel_idx[jj])
            seg = slice(cq_off[ii], cq_off[ii + 1])
            pi = np.repeat((cq_cell[seg] - toff[si]).astype(np.int32), cq_cn[seg])
            e = len(pi)
            if e == 0: continue
            if SNAP2 and t_lat[si] and t_lat[ti_]:
                qi = snap2_draw(si, ti_, nt_, e, pi)
            elif SNAP and t_lat[si] and t_lat[ti_]:
                qi = snap_draw(si, ti_, nt_, e, pi)
            elif SPATIAL:
                qi = spatial_draw(si, ti_, ns_, nt_, e, pi)
            elif NPOV:
                qi = npov_sample(si, ti_, nt_, e, pi)
            elif ACT:
                qi = act_cand(ti_, e).astype(np.int32)
            else:
                qi = rng.integers(0, nt_, e, dtype=np.int32)
        else:
            pi = rng.integers(0, ns_, e, dtype=np.int32)
            if SNAP2 and t_lat[si] and t_lat[ti_]:
                qi = snap2_draw(si, ti_, nt_, e, pi)
            elif SNAP and t_lat[si] and t_lat[ti_]:
                qi = snap_draw(si, ti_, nt_, e, pi)
            elif SPATIAL:
                qi = spatial_draw(si, ti_, ns_, nt_, e, pi)
            elif NPOV:
                qi = npov_sample(si, ti_, nt_, e, pi)
            elif ACT:
                qi = act_cand(ti_, e).astype(np.int32)
            else:
                qi = rng.integers(0, nt_, e, dtype=np.int32)
        if os.environ.get("DEBUGQ", "0") == "1":
            bad = (qi < 0) | (qi >= nt_)
            if bad.any():
                log(f"BADQ pair s={si} t={ti_} bad={bad.sum()}/{len(qi)} vals={qi[bad][:6]} "
                    f"nt={nt_} snap2={SNAP2 and t_lat[si] and t_lat[ti_]}")
        if DYAD and os.environ.get("DYAD", "0") == "2":
            dyad2_prospect(si, ti_, nt_, pi, qi)
        gp = (toff[si] + pi).astype(np.int32)
        gq = (toff[ti_] + qi).astype(np.int32)
        if DYAD:
            pk_ = si * T + ti_; rk_ = ti_ * T + si
            rpq = pend.pop(rk_, None)
            if rpq is not None:
                Rt = dy_Rof(pk_)
                if Rt > 0:
                    nf = dyad_repair(si, ti_, gp, gq, rpq[0], rpq[1], Rt)
                    n_dyad_fix[0] += nf; n_dyad_pairs[0] += 1
            if dy_Rof(rk_) > 0 or dy_Rof(pk_) > 0:
                pend[pk_] = (gp, gq)
        wv = w_emp[int(node_sc[gp[0]]) * NSC + int(node_sc[gq[0]])][rng.integers(0, RES, e)]
        buf_p.append(gp); buf_q.append(gq); buf_w.append(wv.astype(np.int32))
    if buf_p:
        bp = np.concatenate(buf_p); bq = np.concatenate(buf_q); bw = np.concatenate(buf_w)
        wbatch("pre", bp); wbatch("post", bq); wbatch("w", bw)
        n_written += len(bp)
    if (i0 // Bn) % 10 == 0:
        log(f"  {i1:,}/{len(uk):,} 对, 已写 {n_written:,} 边, {time.time()-t0:.0f}s"
            + (f" dyad_fix={n_dyad_fix[0]:,}" if DYAD else ""))
if SPATIAL:
    for att in range(20):
        try:
            with open(l_prefix + ".fallback.txt", "w") as ff: ff.write(str(n_sp_left[0]))
            break
        except OSError: time.sleep(10)
if SEGPAIRS and seg_end < n_pairs:
    if pend:
        pk_a = np.fromiter(pend.keys(), dtype=np.int64, count=len(pend))
        ln_a = np.fromiter((len(v_[0]) for v_ in pend.values()), dtype=np.int64, count=len(pend))
        gp_a = np.concatenate([v_[0] for v_ in pend.values()])
        gq_a = np.concatenate([v_[1] for v_ in pend.values()])
    else:
        pk_a = np.empty(0, np.int64); ln_a = np.empty(0, np.int64)
        gp_a = np.empty(0, np.int32); gq_a = np.empty(0, np.int32)
    ck = {"rng": rng.bit_generator.state, "pend2": (pk_a, ln_a, gp_a, gq_a), "cursor": seg_end,
          "wstate": {k_: v_["off"] for k_, v_ in wstate.items()},
          "ndf": n_dyad_fix[0], "ndd": n_dyad_def[0], "ndnc": n_dyad_nocand[0],
          "ndex": n_dyad_exh[0], "ndp": n_dyad_pairs[0],
          "npro": n_prospect[0] if (DYAD and DYAD2) else 0,
          "nspl": n_sp_left[0] if SPATIAL else 0, "nsnl": n_snap_left[0] if SNAP else 0}
    blob = gzip.compress(pickle.dumps(ck, protocol=4), compresslevel=3)
    CH_ = 2 * 1024 * 1024
    okc = False
    for _a in range(200):
        try:
            cur = os.path.getsize(CKPT_W + ".tmp") if os.path.exists(CKPT_W + ".tmp") else 0
            if cur > len(blob):
                os.truncate(CKPT_W + ".tmp", 0); cur = 0
            with open(CKPT_W + ".tmp", "r+b" if cur else "wb") as fc:
                fc.seek(cur)
                while cur < len(blob):
                    seg = blob[cur:cur+CH_]
                    fc.write(seg); fc.flush(); os.fsync(fc.fileno())
                    cur += len(seg)
            if os.path.getsize(CKPT_W + ".tmp") == len(blob):
                os.replace(CKPT_W + ".tmp", CKPT_W)
                okc = True; break
        except OSError as e:
            log(f"ckpt 写重试{_a}: {str(e)[:40]}"); time.sleep(10)
    if not okc: log("ckpt 写失败(本段)")
    log(f"SEGMENT DONE: {seg_start:,}..{seg_end:,} ckpt saved ({time.time()-t0:.0f}s)")
else:
    if SEGPAIRS:
        try: os.remove(CKPT_W)
        except OSError: pass
    log(f"DONE: {n_written:,} 边 -> {out_prefix} ({time.time()-t0:.0f}s)")
if SPATIAL: log("spatial 兜底边:", n_sp_left[0])
if SNAP: log("snap 兜底边:", n_snap_left[0])
if DYAD: log(f"DYAD: 修复 {n_dyad_fix[0]:,} 边 / 总缺口 {n_dyad_def[0]:,} (达成 {n_dyad_fix[0]/max(n_dyad_def[0],1):.1%}), 无候选 {n_dyad_nocand[0]:,}, 候选耗尽 {n_dyad_exh[0]:,}, 覆盖 {n_dyad_pairs[0]:,} 对, pend 残余 {len(pend):,}")
if DYAD and os.environ.get("DYAD", "0") == "2": log(f"DYAD2: 前瞻换入 {n_prospect[0]:,} 边")
