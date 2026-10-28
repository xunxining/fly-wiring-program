#!/usr/bin/env python3
"""Stage 38 v4.1: 生成器 = 均匀基线 + 普查核(3,440对) + CX 联合模型盒
证据: (1) 真实类型内入度 cv≈0.19 —— 均匀分配即泊松, 边际天然吻合 (v3 配额/kNN 反而破坏)
      (2) 核对采样: 均匀提案 + exp(a(cosΔθ+φ)-1) 接受, 相位按类型级缓存跨对一致
      (3) CX 六型的对一律用 §12 联合拟合 (a,φ + 152 细胞相位), 覆盖普查漏检对
用法: python3 38_generate_v4.py [S] [out_prefix]
"""
import numpy as np
import pickle, os, sys, time, gc
from scipy.special import i0 as besseli0

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
rng = np.random.default_rng(13)
RES = 256

def log(*a): print(*a, flush=True)

# ---------- 基因组 ----------
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
uk, cnt = g['uk'].astype(np.int64), g['cnt'].astype(np.int64)
type_count = g['type_count'].astype(np.int64)
T = len(type_count)
tp, tq = uk // T, uk % T
sc_types = list(g['sc_types'])
NSC = len(sc_types)
type_names = np.load(OUT + "type_names.npy", allow_pickle=True)
tn_str = [str(x) for x in type_names]

# ---------- 真实图(仅用于权重池/类型-sc映射) ----------
pre = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="r")
w = np.memmap(OUT + "e_w.i32", dtype=np.int32, mode="r")
node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
node_sc = np.fromfile(OUT + "node_sc.i32", dtype=np.int32)
N_real = len(node_type)

type_sc = np.zeros(T, dtype=np.int32)
for t in range(T):
    vals, c = np.unique(node_sc[node_type == t], return_counts=True)
    type_sc[t] = vals[np.argmax(c)]

wkey = node_sc[pre].astype(np.int64) * NSC + node_sc[post]
w_emp = np.ones((NSC * NSC, RES), dtype=np.int32)
ws = np.asarray(w)
for k in np.unique(wkey):
    vals = ws[wkey == k]
    w_emp[k] = np.resize(vals, RES) if len(vals) < RES else vals[rng.choice(len(vals), RES, replace=False)]
del ws, wkey; gc.collect()
log("weight pools done")

def drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass

# 生成阶段不再需要真实边表: 释放并驱逐其页缓存
del pre, post, w; gc.collect()
for _f in ["e_pre.i32", "e_post.i32", "e_w.i32"]:
    drop_cache(OUT + _f)

# ---------- 普查位置核 ----------
_pk = np.load(OUT + "poskernel_cal.npz")
pk = {key: _pk[key] for key in _pk.files}
_cons = np.load(OUT + "consensus.npz")
cons = {key: _cons[key] for key in _cons.files}
KMAP = {(int(pk['s'][k]), int(pk['t'][k])): k for k in range(len(pk['s']))}
log(f"poskernel pairs: {len(KMAP)}")

# ---------- CX 联合模型盒 ----------
lp = pickle.load(open(ROOT + "latent_params.pkl", "rb"))
cx_types = list(lp['type_list'])
cx_tidx = [tn_str.index(x) for x in cx_types]
CXSET = set(cx_tidx)
import pandas as pd
cx_cells = pd.read_csv(ROOT + "data_cx/cx_cells.csv")
th_learned = np.load(ROOT + "theta_learned.npy")
cx_theta_pool = {}
for i, x in enumerate(cx_types):
    cx_theta_pool[x] = th_learned[(cx_cells['type'] == x).to_numpy()]
a_par, phi_par = lp['a'], lp['phi']
log(f"CX cartridge: types={cx_types}, pairs covered all CXxCX with |a|>0.05")

def generate(S, out_prefix, jitter=0.1):
    new_count = (type_count * S).astype(np.int64)
    total_nodes = int(new_count.sum())
    offsets = np.zeros(T, dtype=np.int64)
    offsets[1:] = np.cumsum(new_count)[:-1]
    lam = cnt.astype(np.float64) * S
    log(f"S={S}: nodes={total_nodes:,}, expected_edges={int(lam.sum()):,}")

    # 类型级相位缓存 (普查共识池 + CX 联合池)
    t2ci = {int(t): i for i, t in enumerate(cons['ctype'])}
    def type_pool_cons(ty):
        i = t2ci[ty]
        o0, o1 = cons['t_off'][i], cons['t_off'][i + 1]
        return cons['t_theta'][o0:o1]
    tp_cache = {}
    def type_phase_cons(ty):
        if ty not in tp_cache:
            th_r = type_pool_cons(ty)
            n_new = int(new_count[ty])
            dd = rng.integers(0, len(th_r), n_new)
            tp_cache[ty] = th_r[dd] + rng.normal(0, jitter, n_new)
        return tp_cache[ty]
    cx_cache = {}
    def type_phase_cx(ty):
        if ty not in cx_cache:
            x = tn_str[ty]
            th_r = cx_theta_pool[x]
            n_new = int(new_count[ty])
            dd = rng.integers(0, len(th_r), n_new)
            cx_cache[ty] = th_r[dd] + rng.normal(0, jitter, n_new)
        return cx_cache[ty]

    pair_cache = {}
    def get_pair(k, s, t):
        """普查核对: (thp, thq, phi)"""
        if k in pair_cache: return pair_cache[k]
        ns_new, nt_new = int(new_count[s]), int(new_count[t])
        if cons['pair_use_cons'][k]:
            thp = type_phase_cons(s)
            thq = thp if pk['is_self'][k] else type_phase_cons(t)
            phi = float(cons['pair_phi_new'][k])
        else:
            o0, o1 = pk['th_off'][k], pk['th_off'][k + 1]
            nsr = int(pk['ns'][k])
            th = pk['theta'][o0:o1]
            thp_r = th[:nsr]
            thq_r = th[:nsr] if pk['is_self'][k] else th[nsr:]
            phi = float(pk['phi'][k])
            dp = rng.integers(0, len(thp_r), ns_new)
            thp = thp_r[dp] + rng.normal(0, jitter, ns_new)
            if pk['is_self'][k] and ns_new == nt_new:
                thq = thp
            else:
                dq = rng.integers(0, len(thq_r), nt_new)
                thq = thq_r[dq] + rng.normal(0, jitter, nt_new)
        pair_cache[k] = (thp, thq, phi)
        return pair_cache[k]

    # ---- 闭楔增强 (GAMMA>1 启用, §18.4/§21): 候选 (i,j) 若在已生成图中闭楔则接受率 ×γ ----
    GAMMA = float(os.environ.get("GAMMA", "0"))
    PCLOSE = float(os.environ.get("PCLOSE", "0"))
    keep_p, keep_q = [], []          # 已冲刷边 (GAMMA 时用于 booster 重建)
    booster = {"nu": None, "tkey": None, "tnbr": None,
               "gtype": np.repeat(np.arange(T, dtype=np.int32), new_count)}

    def rebuild_booster(extra_p=None, extra_q=None):
        """(u,type) 分段 CSR: tkey=u*(T+1)+tn 排序, tnbr=邻居id; 另留 nu 无向对表做邻接测试"""
        ps = keep_p + ([extra_p] if extra_p is not None and len(extra_p) else [])
        qs = keep_q + ([extra_q] if extra_q is not None and len(extra_q) else [])
        if not ps:
            return
        ap = np.concatenate(ps); aq = np.concatenate(qs)
        Nn = total_nodes
        mn = np.minimum(ap, aq).astype(np.int64)
        mx = np.maximum(ap, aq).astype(np.int64)
        keep = mn != mx
        und = mn[keep] * Nn + mx[keep]
        del mn, mx, keep, ap, aq
        und.sort()
        neq = np.ones(len(und), bool)
        np.not_equal(und[1:], und[:-1], out=neq[1:])
        und = und[neq]
        booster["nu"] = und
        # (u,类型) 分段 CSR: tkey = u*(T+1)+邻居类型 (<=1.57e9, int32 可容), tnbr = 邻居 id
        uu = (und // Nn).astype(np.int32)
        vv = (und - uu.astype(np.int64) * Nn).astype(np.int32)
        gtype = booster["gtype"]
        M = len(und)
        tkey = np.empty(2 * M, np.int32)
        tnbr = np.empty(2 * M, np.int32)
        tkey[:M] = (uu.astype(np.int64) * (T + 1) + gtype[vv]).astype(np.int32)
        tkey[M:] = (vv.astype(np.int64) * (T + 1) + gtype[uu]).astype(np.int32)
        tnbr[:M] = vv
        tnbr[M:] = uu
        del uu, vv
        od = np.argsort(tkey)
        booster["tkey"] = tkey[od]
        booster["tnbr"] = tnbr[od]
        del tkey, tnbr, od
        gc.collect()

    def adjacent_any(gi, gj):
        """(gi,gj) 是否已在 booster 无向图中相邻 (nu 表二分, 全向量化)"""
        nu = booster.get("nu")
        if nu is None or len(gi) == 0:
            return np.zeros(len(gi), bool)
        Nn = total_nodes
        mn = np.minimum(gi, gj).astype(np.int64)
        mx = np.maximum(gi, gj).astype(np.int64)
        kk = mn * Nn + mx
        i = np.searchsorted(nu, kk)
        i = np.clip(i, 0, len(nu) - 1)
        return nu[i] == kk

    def wedge_place(si, ti_, ns_, nt_, m, thp=None, thq=None, a=0.0, phi=0.0):
        """闭楔放置 (§21): i 均匀, u∈N(i), j∈N(u)∩类型t (段直取), (i,j) 空, 核接受(可选)"""
        tkey = booster.get("tkey")
        if tkey is None or m <= 0:
            return None, None
        tnbr = booster["tnbr"]
        Nn = total_nodes
        Tp1 = T + 1
        base_i = np.int64(offsets[si]); base_j = np.int64(offsets[ti_])
        acc_i, acc_j = [], []
        got = 0
        tries = 0
        maxtries = max(4 * m, 2000)
        while got < m and tries < maxtries:
            B = min(3 * (m - got) + 32, 400_000)
            tries += B
            ii = rng.integers(0, ns_, B)
            gi = base_i + ii
            lo = np.searchsorted(tkey, (gi * Tp1).astype(np.int32))
            hi = np.searchsorted(tkey, ((gi + 1) * Tp1).astype(np.int32))
            d = hi - lo
            ok = d > 0
            if not ok.any():
                continue
            ii, gi, lo, d = ii[ok], gi[ok], lo[ok], d[ok]
            uu = tnbr[lo + (rng.random(len(lo)) * d).astype(np.int64)].astype(np.int64)
            ku = (uu * Tp1 + ti_).astype(np.int32)
            lo2 = np.searchsorted(tkey, ku)
            hi2 = np.searchsorted(tkey, (ku.astype(np.int64) + 1).astype(np.int32))
            d2 = hi2 - lo2
            ok = d2 > 0
            if not ok.any():
                continue
            ii, uu, lo2, d2 = ii[ok], uu[ok], lo2[ok], d2[ok]
            gj = tnbr[lo2 + (rng.random(len(lo2)) * d2).astype(np.int64)].astype(np.int64)
            jj = gj - base_j
            ok = (jj >= 0) & (jj < nt_) & (gj != (base_i + ii))
            if not ok.any():
                continue
            ii, jj, gi, gj = ii[ok], jj[ok], (base_i + ii[ok]), gj[ok]
            adj = adjacent_any(gi.astype(np.int32), gj.astype(np.int32))
            ii, jj = ii[~adj], jj[~adj]
            if len(ii) == 0:
                continue
            if thp is not None:
                w = np.exp(np.float32(a) * (np.cos(thp[ii] - thq[jj] + np.float32(phi)) - np.float32(1.0)))
                keepk = rng.random(len(ii), dtype=np.float32) < w
                ii, jj = ii[keepk], jj[keepk]
            acc_i.append(ii.astype(np.int32)); acc_j.append(jj.astype(np.int32))
            got += len(ii)
        if not acc_i:
            return None, None
        ii = np.concatenate(acc_i)[:m]
        jj = np.concatenate(acc_j)[:m]
        return ii, jj

    def kernel_sample(ns_, nt_, thp, thq, a, phi, e, boost=None):
        acc = max(0.05, float(besseli0(a) / np.exp(a)))
        thp32 = np.asarray(thp, dtype=np.float32)
        thq32 = np.asarray(thq, dtype=np.float32)
        gp, gq = [], []
        need = int(e)
        while need > 0:
            M = min(int(need / acc * 1.25) + 64, 3_000_000)
            ii = rng.integers(0, ns_, M, dtype=np.int32)
            jj = rng.integers(0, nt_, M, dtype=np.int32)
            dth = thp32[ii] - thq32[jj] + np.float32(phi)
            wij = np.exp(np.float32(a) * (np.cos(dth) - np.float32(1.0)))
            if boost is not None:
                cl = adjacent_any((boost[0] + ii).astype(np.int32), (boost[1] + jj).astype(np.int32))
                wij *= np.where(cl, np.float32(1.0), np.float32(1.0 / GAMMA))
            keep = rng.random(M, dtype=np.float32) < wij
            gp.append(ii[keep]); gq.append(jj[keep])
            need -= int(keep.sum())
        return np.concatenate(gp)[:e], np.concatenate(gq)[:e]

    def cx_place_sys(ns_, nt_, thp, thq, a, phi, e):
        """v5 低噪声 CX 放置 (§19/§20): a>0.05 -> 核 CDF 系统化采样(每胞计数误差<1);
        其余(含负 a, 保持 38 退化均匀语义) -> 随机起点平衡轮转。"""
        ncell = ns_ * nt_
        if a <= 0.05:
            u0 = int(rng.integers(0, ncell))
            idx = (u0 + np.arange(e)) % ncell
            return idx // nt_, idx % nt_
        Wij = np.exp(a * (np.cos(thp[:, None] - thq[None, :] + phi) - 1.0))
        flat = Wij.ravel()
        cdf = np.cumsum(flat); cdf /= cdf[-1]
        pos = (rng.random() + np.arange(e)) / e
        idx = np.searchsorted(cdf, pos)
        return idx // nt_, idx % nt_

    CXSYS = os.environ.get("CXSYS", "0") == "1"

    # ---- §21 机制 v4: 神经毡共定位同配放置 (SCHOM=1) ----
    # 均匀对的 target 以逐对概率 p_st (=真实同 sc 边占比, m0 收缩) 从同 sc 段抽取
    SCHOM = os.environ.get("SCHOM", "0") == "1"
    SCHOMF = float(os.environ.get("SCHOMF", "0"))
    if SCHOM:
        sh_uk = np.fromfile(ROOT + "schom_uk.i64", dtype=np.int64)
        sh_f = np.fromfile(ROOT + "schom_f.f32", dtype=np.float32)
        sh_cnt = np.fromfile(ROOT + "schom_cnt.i32", dtype=np.int32)
        scm_g = np.fromfile(ROOT + "node_sc_gen.i32", dtype=np.int32)
        F_GLOB, M0 = 0.7215, 30.0
        log(f"SCHOM: 同配表 {len(sh_uk):,} 对, F_GLOB={F_GLOB}")

        def hom_draw(si, ti_, ns_, nt_, m, pi):
            """pi 为已抽源 (局部), 返回同 sc 富集的 target 局部索引"""
            if SCHOMF > 0:
                p_st = SCHOMF  # 剂量反应: 恒定同配强度
            else:
                key = np.int64(si) * T + np.int64(ti_)
                p = int(np.searchsorted(sh_uk, key))
                if p < len(sh_uk) and sh_uk[p] == key:
                    c, f = float(sh_cnt[p]), float(sh_f[p])
                else:
                    c, f = 0.0, F_GLOB
                p_st = (c * f + M0 * F_GLOB) / (c + M0)
            if p_st <= 1e-3:
                return rng.integers(0, nt_, m, dtype=np.int32)
            tsc = scm_g[offsets[ti_]:offsets[ti_] + nt_]
            order = np.argsort(tsc, kind="stable")
            scs = tsc[order]
            glo = np.searchsorted(scs, np.arange(29))
            ghi = np.searchsorted(scs, np.arange(1, 30))
            sci = scm_g[offsets[si] + pi]
            lo = glo[sci]; ln = ghi[sci] - lo
            hom = (rng.random(m) < p_st) & (ln > 0)
            q = rng.integers(0, nt_, m, dtype=np.int32)
            hidx = np.minimum(lo + (rng.random(m) * np.maximum(ln, 1)).astype(np.int64), nt_ - 1)
            q[hom] = order[hidx[hom]].astype(np.int32)
            return q

    # ---- §22 空间距离核 (SPATIAL=1): 均匀对按 w_st(d) 拒绝采样放置 ----
    SPATIAL = os.environ.get("SPATIAL", "0") == "1"
    if SPATIAL:
        sk = np.load(ROOT + "spatial_kernel.npz")
        sp_bins = sk["bins"]; sp_uk = sk["big_uk"]
        sp_wbig = sk["wbig"]; sp_wmax = sk["wmax_big"]
        sp_wg = sk["w_glob"]; sp_wmax_g = float(sk["wmax_glob"])
        xyz_g = np.fromfile(ROOT + os.environ.get("SPATIALXYZ", "xyz_gen.f32"),
                            dtype=np.float32).reshape(-1, 3)
        NBSP = len(sp_bins) - 1
        n_sp_left = [0]  # 拒绝采样 4 轮后仍空、退回均匀的边数
        # §23b 目标活跃度加权候选 (ACT=1, 需 act_in.f32)
        ACT = os.environ.get("ACT", "0") == "1"
        act_in = cw_cache = None
        if ACT:
            act_in = np.fromfile(ROOT + "act_in.f32", dtype=np.float32)
            cw_cache = {}
            log(f"ACT: 目标活跃度加权, a_in 中位 {float(np.median(act_in)):.2f}")

        def act_cand(ti_, shape):
            """类型 ti_ 内按 a_in 加权抽候选 (局部索引)"""
            if ti_ not in cw_cache:
                seg = act_in[offsets[ti_]:offsets[ti_] + int(new_count[ti_])]
                cw_cache[ti_] = np.cumsum(seg)
            cw = cw_cache[ti_]
            tot = cw[-1]
            if tot <= 0:
                return rng.integers(0, len(cw), shape, dtype=np.int32)
            return np.searchsorted(cw, rng.random(shape) * tot).astype(np.int32)

        log(f"SPATIAL: 大对核 {len(sp_uk):,}, wmax_glob {sp_wmax_g:.2f}")

        def spatial_draw(si, ti_, ns_, nt_, m, pi):
            """pi 为已抽源 (局部), 返回按逐对距离核拒绝采样的 target 局部索引"""
            key = np.int64(si) * T + np.int64(ti_)
            r = int(np.searchsorted(sp_uk, key))
            if r < len(sp_uk) and sp_uk[r] == key:
                W = sp_wbig[r]; wmax = float(sp_wmax[r])
            else:
                W = sp_wg; wmax = sp_wmax_g
            srcx = xyz_g[offsets[si] + pi]            # (m,3) um, NaN=缺坐标
            tb0 = offsets[ti_]
            q = np.empty(m, np.int32)
            todo = np.arange(m)
            OS = int(min(max(wmax * 1.6, 4), float(os.environ.get("SPATIALOS", "24"))))
            for _ in range(4):
                k = len(todo)
                if k == 0:
                    break
                cand = act_cand(ti_, (k, OS)) if ACT else rng.integers(0, nt_, (k, OS), dtype=np.int32)
                dx = xyz_g[tb0 + cand] - srcx[todo][:, None, :]
                d = np.sqrt((dx * dx).sum(-1))
                b = np.clip(np.searchsorted(sp_bins, d, side="right") - 1, 0, NBSP - 1)
                w = W[b]
                w[np.isnan(d)] = 1.0                  # 缺坐标: 中性
                acc = rng.random((k, OS)) < (w / wmax)
                has = acc.any(1)
                got = todo[has]
                q[got] = cand[np.arange(k)[has], acc[has].argmax(1)]
                todo = todo[~has]
            if len(todo):
                q[todo] = rng.integers(0, nt_, len(todo), dtype=np.int32)
                n_sp_left[0] += len(todo)
            return q

        def spatial_refilter(si, ti_, pi, qi, redraw, fix_src=False):
            """对已有抽样 (pi,qi) 按距离核拒绝-重抽
            redraw(m)->(pi2,qi2) 联合重抽; fix_src=True 时 redraw(sp)->qi2 仅重抽目标
            过采样 OS 倍每槽取首个通过者, 已接受冻结; 3 轮后残余保留兜底候选"""
            key = np.int64(si) * T + np.int64(ti_)
            r = int(np.searchsorted(sp_uk, key))
            if r < len(sp_uk) and sp_uk[r] == key:
                W = sp_wbig[r]; wmax = float(sp_wmax[r])
            else:
                W = sp_wg; wmax = sp_wmax_g
            so, to0 = offsets[si], offsets[ti_]
            OS = int(min(max(wmax * 1.6, 4), float(os.environ.get("SPATIALOS", "24"))))
            out_p = pi.astype(np.int32).copy(); out_q = qi.astype(np.int32).copy()

            def acc_mask(p_, q_):
                d = np.sqrt(((xyz_g[to0 + q_] - xyz_g[so + p_]) ** 2).sum(-1))
                b = np.clip(np.searchsorted(sp_bins, d, side="right") - 1, 0, NBSP - 1)
                w = W[b]
                w[np.isnan(d)] = 1.0
                return rng.random(len(p_)) < (w / wmax)

            todo = np.flatnonzero(~acc_mask(out_p, out_q))
            for _ in range(3):
                nb = len(todo)
                if nb == 0:
                    break
                if fix_src:
                    sp = out_p[todo].astype(np.int64)
                    cp = np.repeat(sp, OS).reshape(nb, OS)
                    cq = redraw(np.repeat(sp, OS))[:nb * OS].reshape(nb, OS).astype(np.int64)
                else:
                    cp, cq = redraw(nb * OS)
                    cp = cp[:nb * OS].reshape(nb, OS).astype(np.int64)
                    cq = cq[:nb * OS].reshape(nb, OS).astype(np.int64)
                acc = acc_mask(cp.ravel(), cq.ravel()).reshape(nb, OS)
                has = acc.any(1)
                first = acc[has].argmax(1)
                idx = np.arange(nb)[has]
                if not fix_src:
                    out_p[todo[idx]] = cp[idx, first].astype(np.int32)
                out_q[todo[idx]] = cq[idx, first].astype(np.int32)
                nh = ~has
                if not fix_src:
                    out_p[todo[nh]] = cp[nh, 0].astype(np.int32)   # 兜底
                out_q[todo[nh]] = cq[nh, 0].astype(np.int32)
                todo = todo[nh]
            return out_p, out_q

    # ---- §23 逐细胞配额 (CELLQ=1): 源侧精确兑现真实逐细胞出度 ----
    CELLQ = os.environ.get("CELLQ", "0") == "1"
    if CELLQ:
        cq_cell = np.fromfile(ROOT + "cq_cell.i32", dtype=np.int32).astype(np.int64)  # 紧凑序细胞
        cq_t = np.fromfile(ROOT + "cq_t.i32", dtype=np.int32).astype(np.int64)
        cq_cn = np.fromfile(ROOT + "cq_cnt.i32", dtype=np.int32).astype(np.int64)
        nt_full = np.fromfile(ROOT + "data_census/node_type.i32", dtype=np.int32)
        ntc_c = nt_full[nt_full >= 0].astype(np.int64)          # 紧凑序真实类型标签
        perm_g = np.argsort(ntc_c, kind="stable")               # 基因组序 -> 紧凑序
        c2g = np.empty(len(ntc_c), np.int64); c2g[perm_g] = np.arange(len(ntc_c))
        g_of = c2g[cq_cell]                                     # 配额细胞的基因组序
        cq_loc = (g_of - offsets[ntc_c[cq_cell]]).astype(np.int32)   # 型内局部源索引
        cq_pk = ntc_c[cq_cell] * T + cq_t
        o2 = np.argsort(cq_pk, kind="stable")
        cq_pk = cq_pk[o2]; cq_loc = cq_loc[o2]; cq_cn = cq_cn[o2]
        uq2, st2 = np.unique(cq_pk, return_index=True)
        assert len(uq2) == len(uk) and (uq2 == uk).all(), "细胞配额对与基因组对不齐"
        cq_off = np.append(st2, len(cq_pk))
        log(f"CELLQ: 逐细胞配额 {len(cq_pk):,} 条载入, 与基因组对全等")

        def row_draw(sp, thp, thq, a, phi):
            """给定源局部索引 sp, 逐源按其行核密度系统化抽目标 (每源误差<1)"""
            us, inv, cs = np.unique(sp, return_inverse=True, return_counts=True)
            out = np.empty(len(sp), np.int32)
            od = np.argsort(inv, kind="stable")
            pos = 0
            for u_, c_ in zip(us, cs):
                wr = np.exp(a * (np.cos(thp[u_] - thq + phi) - 1.0))
                cdf = np.cumsum(wr); cdf /= cdf[-1]
                pp = (rng.random() + np.arange(c_)) / c_
                out[od[pos:pos + c_]] = np.searchsorted(cdf, pp)
                pos += c_
            return out

    fpre = open(out_prefix + "_pre.i32", "wb")
    fpost = open(out_prefix + "_post.i32", "wb")
    fw = open(out_prefix + "_w.i32", "wb")
    Bn = 100_000
    n_written = n_kernel = n_cx = n_wedge = 0
    t0 = time.time()
    for i0 in range(0, len(uk), Bn):
        i1 = min(i0 + Bn, len(uk))
        ne = rng.poisson(lam[i0:i1])
        sel = np.ones(i1 - i0, bool) if CELLQ else (ne > 0)   # CELLQ: 配额精确, 不做泊松稀释
        if not sel.any(): continue
        tb_s, tb_t, neb = tp[i0:i1][sel], tq[i0:i1][sel], ne[sel]
        buf_p, buf_q, buf_w, buf_n = [], [], [], 0

        def flush_buf():
            nonlocal buf_n
            if not buf_p: return
            bp = np.concatenate(buf_p); bq = np.concatenate(buf_q); bw = np.concatenate(buf_w)
            bp.tofile(fpre); bq.tofile(fpost); bw.tofile(fw)
            if GAMMA > 1 or PCLOSE > 0:
                keep_p.append(bp); keep_q.append(bq)
            for fh in (fpre, fpost, fw):
                fh.flush()
            os.fsync(fpre.fileno()); os.fsync(fpost.fileno()); os.fsync(fw.fileno())
            os.posix_fadvise(fpre.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fpost.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fw.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            buf_p.clear(); buf_q.clear(); buf_w.clear()
            buf_n = 0

        if (GAMMA > 1 or PCLOSE > 0) and (i0 // Bn) % 4 == 0:
            rebuild_booster(np.concatenate(buf_p) if buf_p else None,
                            np.concatenate(buf_q) if buf_q else None)

        sel_idx = np.flatnonzero(sel)
        for jj, (s, t, e) in enumerate(zip(tb_s, tb_t, neb)):
            ns_, nt_ = int(new_count[s]), int(new_count[t])
            si, ti_ = int(s), int(t)
            pi_q = None
            if CELLQ:
                ii = i0 + int(sel_idx[jj])
                seg = slice(cq_off[ii], cq_off[ii + 1])
                pi_q = np.repeat(cq_loc[seg], cq_cn[seg]).astype(np.int32)
                e = len(pi_q)
                if e == 0:
                    continue
            if si in CXSET and ti_ in CXSET:
                # ---- CX 联合模型盒 (CXSYS=1 时启用 v5 低噪声放置) ----
                xs, xt = cx_tidx.index(si), cx_tidx.index(ti_)
                a_cx = float(a_par[xs, xt])
                if CELLQ:
                    pi = pi_q
                    if CXSYS and a_cx > 0.05:
                        thp = type_phase_cx(si)
                        thq = thp if si == ti_ else type_phase_cx(ti_)
                        qi = row_draw(pi, thp, thq, a_cx, float(phi_par[xs, xt]))
                    else:
                        qi = rng.integers(0, nt_, e, dtype=np.int32)
                    n_cx += e
                elif CXSYS:
                    thp = type_phase_cx(si)
                    thq = thp if si == ti_ else type_phase_cx(ti_)
                    pi, qi = cx_place_sys(ns_, nt_, thp, thq, a_cx, float(phi_par[xs, xt]), e)
                    n_cx += e
                elif abs(a_cx) > 0.05:
                    thp = type_phase_cx(si)
                    thq = thp if si == ti_ else type_phase_cx(ti_)
                    pi, qi = kernel_sample(ns_, nt_, thp, thq, a_cx, float(phi_par[xs, xt]), e)
                    n_cx += e
                else:
                    pi = rng.integers(0, ns_, e); qi = rng.integers(0, nt_, e)
            elif (si, ti_) in KMAP:
                k = KMAP[(si, ti_)]
                thp, thq, phi = get_pair(k, si, ti_)
                bst = (offsets[si], offsets[ti_]) if GAMMA > 1 else None
                m = int(round(e * PCLOSE)) if (PCLOSE > 0 and e >= 10 and nt_ >= 200) else 0
                pw = qw = None
                if m:
                    pw, qw = wedge_place(si, ti_, ns_, nt_, m, thp, thq, float(pk['a'][k]), phi)
                g = 0 if pw is None else len(pw)
                if CELLQ:
                    pi = pi_q
                    qi = row_draw(pi, thp, thq, float(pk['a'][k]), phi)
                    if SPATIAL and e > 0:
                        aa = float(pk['a'][k])
                        pi, qi = spatial_refilter(si, ti_, pi, qi,
                                    lambda sp: row_draw(sp, thp, thq, aa, phi), fix_src=True)
                else:
                    pi, qi = kernel_sample(ns_, nt_, thp, thq, float(pk['a'][k]), phi, e - g, boost=bst)
                    if SPATIAL and e - g > 0:
                        aa = float(pk['a'][k])
                        pi, qi = spatial_refilter(si, ti_, pi, qi,
                                    lambda m: kernel_sample(ns_, nt_, thp, thq, aa, phi, m))
                if g:
                    pi = np.concatenate([pw, pi]); qi = np.concatenate([qw, qi])
                    n_wedge += g
                n_kernel += e
            else:
                m = int(round(e * PCLOSE)) if (PCLOSE > 0 and e >= 10 and nt_ >= 200) else 0
                pw = qw = None
                if m:
                    pw, qw = wedge_place(si, ti_, ns_, nt_, m)
                g = 0 if pw is None else len(pw)
                if CELLQ:
                    pi = pi_q
                else:
                    pi = rng.integers(0, ns_, e - g, dtype=np.int32)
                if SPATIAL and e - g > 0:
                    qi = spatial_draw(si, ti_, ns_, nt_, len(pi), pi)
                elif SCHOM and e - g > 0:
                    qi = hom_draw(si, ti_, ns_, nt_, e - g, pi)
                else:
                    qi = rng.integers(0, nt_, len(pi), dtype=np.int32)
                if g:
                    pi = np.concatenate([pw, pi]); qi = np.concatenate([qw, qi])
                    n_wedge += g
            # 逐对定权入缓冲, 每 ~2M 边冲刷 (防大块瞬时内存)
            wk = int(type_sc[s]) * NSC + int(type_sc[t])
            buf_p.append((offsets[s] + pi).astype(np.int32))
            buf_q.append((offsets[t] + qi).astype(np.int32))
            buf_w.append(w_emp[wk, rng.integers(0, RES, len(pi))].astype(np.int32))
            buf_n += len(pi)
            if buf_n >= 2_000_000:
                n_written += buf_n
                flush_buf()
        n_written += buf_n
        flush_buf()
        if i0 % (Bn * 10) == 0:
            log(f"  pairs {i0:,}/{len(uk):,}, edges {n_written:,} (kernel {n_kernel:,} cx {n_cx:,} wedge {n_wedge:,}), {time.time()-t0:.0f}s")
    fpre.close(); fpost.close(); fw.close()
    if cx_cache:
        tys = sorted(cx_cache)
        np.savez(out_prefix + "_cxtheta.npz", tidx=np.array(tys, np.int32),
                 off=offsets[tys].astype(np.int64),
                 theta=np.concatenate([cx_cache[t] for t in tys]))
    log(f"GENERATE v4.1 DONE S={S}: {n_written:,} edges (kernel {n_kernel:,}, cx {n_cx:,}, wedge {n_wedge:,})")
    if SPATIAL:
        log(f"SPATIAL: 退回均匀 {n_sp_left[0]:,} 边")

if __name__ == "__main__":
    S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    outp = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/flydata/gen/genv4_S{S:g}"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    generate(S, outp)
