#!/usr/bin/env python3
"""Stage 48 v4.2: v4.1 + 二元亲和组件 (互惠 dyad 配额)
机制: 每无序类型对 {s,t} 取真实 n_mutual 为配额; 互 dyad 用核采样(有核时)同时放置双向边,
      其余边按 v4.1 独立放置. 边数/边际守恒 (互 dyad 两个方向各计 1 条).
依据: §17 亏缺按 2.8^k 纯复合 -> 只需修二元; 47 测得互 dyad 相位倾斜温和 (0.21->0.33)
用法: python3 48_generate_v42.py [S] [out_prefix]
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

# ---------- 真实图 (权重池/类型-sc) ----------
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

# ---------- 二元亲和: 每无序类型对 n_mutual ----------
dz = np.load(ROOT + "dyad_stats.npz")
dy_uk = dz['uk'].astype(np.int64)          # min*T+max
dy_nm = dz['n_mutual'].astype(np.int64)
# 基因组每行 (s,t) 的互配额 (按无序对查)
gkey_u = np.minimum(tp, tq) * T + np.maximum(tp, tq)
ii = np.searchsorted(dy_uk, gkey_u)
ii = np.clip(ii, 0, len(dy_uk) - 1)
mut_quota = np.where(dy_uk[ii] == gkey_u, dy_nm[ii], 0)
# 伙伴行索引 (t,s) 的行号
order_uk = np.argsort(uk)
uk_sorted = uk[order_uk]
jj = np.searchsorted(uk_sorted, tq * T + tp)
jj = np.clip(jj, 0, len(uk_sorted) - 1)
has_partner = uk_sorted[jj] == (tq * T + tp)
partner = order_uk[np.where(has_partner, jj, 0)]
log(f"dyad quotas: mutual total={int(mut_quota.sum()):,}")

def generate(S, out_prefix, jitter=0.1):
    new_count = (type_count * S).astype(np.int64)
    total_nodes = int(new_count.sum())
    offsets = np.zeros(T, dtype=np.int64)
    offsets[1:] = np.cumsum(new_count)[:-1]
    lam = cnt.astype(np.float64) * S
    log(f"S={S}: nodes={total_nodes:,}, expected_edges={int(lam.sum()):,}")

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

    def kernel_sample(ns_, nt_, thp, thq, a, phi, e):
        acc = max(0.05, float(besseli0(abs(a)) / np.exp(abs(a))))
        thp32 = np.asarray(thp, dtype=np.float32)
        thq32 = np.asarray(thq, dtype=np.float32)
        gp, gq = [], []
        need = int(e)
        while need > 0:
            M = min(int(need / acc * 1.25) + 64, 3_000_000)
            ii = rng.integers(0, ns_, M, dtype=np.int32)
            jj = rng.integers(0, nt_, M, dtype=np.int32)
            wij = np.exp(np.float32(a) * np.cos(thp32[ii] - thq32[jj] + np.float32(phi)) - np.float32(abs(a)))
            keep = rng.random(M, dtype=np.float32) < wij
            gp.append(ii[keep]); gq.append(jj[keep])
            need -= int(keep.sum())
        return np.concatenate(gp)[:e], np.concatenate(gq)[:e]

    def get_kernel(si, ti_):
        """该有序对的核参数 (thp, thq, a, phi, 类型级一致?); 无核返回 None"""
        if si in CXSET and ti_ in CXSET:
            xs, xt = cx_tidx.index(si), cx_tidx.index(ti_)
            a_cx = float(a_par[xs, xt])
            if abs(a_cx) > 0.05:
                thp = type_phase_cx(si)
                thq = thp if si == ti_ else type_phase_cx(ti_)
                return (thp, thq, a_cx, float(phi_par[xs, xt]), True)
            return None
        if (si, ti_) in KMAP:
            k = KMAP[(si, ti_)]
            thp, thq, phi = get_pair(k, si, ti_)
            return (thp, thq, float(pk['a'][k]), phi, bool(cons['pair_use_cons'][k]))
        return None

    def place(si, ti_, e):
        """v4.1 原放置逻辑: 返回 (pi, qi) 局部索引"""
        ns_, nt_ = int(new_count[si]), int(new_count[ti_])
        kp = get_kernel(si, ti_)
        if kp is None:
            return rng.integers(0, ns_, e, dtype=np.int32), rng.integers(0, nt_, e, dtype=np.int32)
        return kernel_sample(ns_, nt_, kp[0], kp[1], kp[2], kp[3], e)


    def place_mutual(si, ti_, m):
        """互 dyad: 双方向核乘积 w=w_fwd·w_rev 接受采样 (核一致互惠, 护 CX R²)"""
        ns_, nt_ = int(new_count[si]), int(new_count[ti_])
        kf = get_kernel(si, ti_)
        kr = get_kernel(ti_, si) if si != ti_ else kf
        # 双方向核必须同相位坐标系 (类型级一致) 才能相乘; 否则只用前向核
        use_rev = (kr is not None and kf is not None and kr[4] and kf[4])
        if kf is None and kr is None:
            return rng.integers(0, ns_, m, dtype=np.int32), rng.integers(0, nt_, m, dtype=np.int32)
        if kf is None:
            kf = None  # 前向无核: 只用反向 (以反向提案重参数化过于复杂, 退化为均匀)
            return rng.integers(0, ns_, m, dtype=np.int32), rng.integers(0, nt_, m, dtype=np.int32)
        gp, gq = [], []
        need = int(m)
        tot_prop = tot_kept = 0
        while need > 0:
            if tot_prop > 0:
                acc_emp = max(tot_kept / tot_prop, 1e-4)
            else:
                acc_emp = 0.05
            M = min(int(need / acc_emp * 1.3) + 64, 3_000_000)
            ii = rng.integers(0, ns_, M, dtype=np.int32)
            jj = rng.integers(0, nt_, M, dtype=np.int32)
            logw = np.zeros(M, dtype=np.float32)
            thp32 = np.asarray(kf[0], dtype=np.float32); thq32 = np.asarray(kf[1], dtype=np.float32)
            logw += np.float32(kf[2]) * (np.cos(thp32[ii] - thq32[jj] + np.float32(kf[3])) - np.float32(1.0))
            if use_rev:
                # 反向核: pre=t型(实例j), post=s型(实例i) -> cos(θ_t[j]-θ_s[i]+φ_r)
                trp32 = np.asarray(kr[0], dtype=np.float32); trq32 = np.asarray(kr[1], dtype=np.float32)
                logw += np.float32(kr[2]) * (np.cos(trp32[jj] - trq32[ii] + np.float32(kr[3])) - np.float32(1.0))
            keep = rng.random(M, dtype=np.float32) < np.exp(logw)
            gp.append(ii[keep]); gq.append(jj[keep])
            tot_prop += M; tot_kept += int(keep.sum())
            need -= int(keep.sum())
        return np.concatenate(gp)[:m], np.concatenate(gq)[:m]

    fpre = open(out_prefix + "_pre.i32", "wb")
    fpost = open(out_prefix + "_post.i32", "wb")
    fw = open(out_prefix + "_w.i32", "wb")
    Bn = 100_000
    n_written = n_mutual_placed = 0
    t0 = time.time()
    done = np.zeros(len(uk), dtype=bool)

    def emit(buf, s_off, t_off, pi, qi, wk):
        buf[0].append((s_off + pi).astype(np.int32))
        buf[1].append((t_off + qi).astype(np.int32))
        buf[2].append(w_emp[wk, rng.integers(0, RES, len(pi))].astype(np.int32))

    for i0 in range(0, len(uk), Bn):
        i1 = min(i0 + Bn, len(uk))
        ne = rng.poisson(lam[i0:i1])
        buf = [[], [], []]
        buf_n = 0

        def flush_buf():
            nonlocal buf_n
            if not buf[0]: return
            bp = np.concatenate(buf[0]); bq = np.concatenate(buf[1]); bw = np.concatenate(buf[2])
            bp.tofile(fpre); bq.tofile(fpost); bw.tofile(fw)
            for fh in (fpre, fpost, fw):
                fh.flush()
            os.fsync(fpre.fileno()); os.fsync(fpost.fileno()); os.fsync(fw.fileno())
            os.posix_fadvise(fpre.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fpost.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fw.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            buf[0].clear(); buf[1].clear(); buf[2].clear()
            buf_n = 0

        for r in range(i0, i1):
            if done[r]:
                continue
            s, t = int(tp[r]), int(tq[r])
            e = int(ne[r - i0])
            m = int(min(mut_quota[r] * S, e))  # 互配额 (S=1 即真实值)
            wk = int(type_sc[s]) * NSC + int(type_sc[t])

            if s == t:
                # 自对: m 个互 dyad (各 2 边), 其余 e-2m 单边 -> 总数守恒
                if m > 0 and e >= 2 * m:
                    mi, mj = place_mutual(s, t, m)
                    keep = mi != mj
                    mi, mj = mi[keep], mj[keep]
                    emit(buf, offsets[s], offsets[t], mi, mj, wk)
                    emit(buf, offsets[t], offsets[s], mj, mi, wk)
                    n_mutual_placed += len(mi)
                    if e - 2 * m > 0:
                        pi, qi = place(s, t, e - 2 * m)
                        emit(buf, offsets[s], offsets[t], pi, qi, wk)
                else:
                    pi, qi = place(s, t, e)
                    emit(buf, offsets[s], offsets[t], pi, qi, wk)
                buf_n += e
                done[r] = True
            elif s < t or not has_partner[r]:
                # 无序对 {s,t}: 联合生成两个方向; 若无伙伴行则仅本方向
                e2 = 0
                r2 = partner[r] if has_partner[r] else -1
                if 0 <= r2 < len(uk) and not done[r2] and int(tp[r2]) == t and int(tq[r2]) == s:
                    e2 = int(rng.poisson(lam[r2]))
                m2 = min(m, e, e2) if e2 > 0 else 0
                # 互 dyad: 双方向核乘积采样, 同时放双向
                if m2 > 0:
                    mi, mj = place_mutual(s, t, m2)
                    emit(buf, offsets[s], offsets[t], mi, mj, wk)
                    wk2 = int(type_sc[t]) * NSC + int(type_sc[s])
                    emit(buf, offsets[t], offsets[s], mj, mi, wk2)
                    n_mutual_placed += m2
                # 剩余单边
                if e - m2 > 0:
                    pi, qi = place(s, t, e - m2)
                    emit(buf, offsets[s], offsets[t], pi, qi, wk)
                if e2 - m2 > 0:
                    wk2 = int(type_sc[t]) * NSC + int(type_sc[s])
                    pi2, qi2 = place(t, s, e2 - m2)
                    emit(buf, offsets[t], offsets[s], pi2, qi2, wk2)
                buf_n += e + e2
                done[r] = True
                if r2 >= 0:
                    done[r2] = True
            else:
                # s > t 且伙伴存在: 已在伙伴行联合处理
                done[r] = True
                continue
            if buf_n >= 2_000_000:
                n_written += buf_n
                flush_buf()
        n_written += buf_n
        flush_buf()
        if i0 % (Bn * 10) == 0:
            log(f"  rows {i0:,}/{len(uk):,}, edges {n_written:,} (mutual {n_mutual_placed:,}), {time.time()-t0:.0f}s")
    fpre.close(); fpost.close(); fw.close()
    if cx_cache:
        tys = sorted(cx_cache)
        np.savez(out_prefix + "_cxtheta.npz", tidx=np.array(tys, np.int32),
                 off=offsets[tys].astype(np.int64),
                 theta=np.concatenate([cx_cache[t] for t in tys]))
    log(f"GENERATE v4.2 DONE S={S}: {n_written:,} edges, mutual dyads {n_mutual_placed:,}")

if __name__ == "__main__":
    S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    outp = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/flydata/gen/genv42_S{S:g}"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    generate(S, outp)
