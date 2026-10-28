#!/usr/bin/env python3
# 127_trim_lataware.py — §39 P8g 装配忠实修剪: 与 125 全同, 唯候选池改为晶格模态偏移
# 近邻(snap 核 pair_mode + (t,半球) cKDTree, 复用 75 机械; 无坐标/OVER/树空回退类型随机池)。
# 候选内评分仍 argmax cn, 并列回退 |Δdist|。用法: python3 127_trim_lataware.py female
import json, os, sys, time, hashlib, gc
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器"
SEX = sys.argv[1]
if SEX == "female":
    D = f"{ROOT}/data_female"
    SRC = {k: f"{ROOT}/gen_edges/g_mutq_{k}.i32" for k in ("pre", "post", "w")}
    DST = {k: f"{ROOT}/gen_edges/g_mutq4_{k}.i32" for k in ("pre", "post", "w")}
    REAL_PRE, REAL_POST = f"{D}/e_pre.i32", f"{D}/e_post.i32"
    RECIP_REAL = 0.32130410937717807
else:
    D = f"{ROOT}/data_male"
    SRC = {k: f"{ROOT}/gen_edges/g_mutqm_{k}.i32" for k in ("pre", "post", "w")}
    DST = {k: f"{ROOT}/gen_edges/g_mutqm3_{k}.i32" for k in ("pre", "post", "w")}
    REAL_PRE, REAL_POST = f"{D}/e_pre_s.i32", f"{D}/e_post_s.i32"
    RECIP_REAL = 0.2997500837059255

t0 = time.time()
def log(*a):
    try:  # EIO 窗口写日志会抛错杀进程; 日志可丢, 命要保
        print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    except OSError:
        pass

def write_atomic(writer, dst, want):
    # 直写门户(暂存也在门户: /tmp 挂载会整个消失, errno 2 风暴的根因)
    att = 0
    tdst = dst + ".tmp"
    while True:
        try:
            with open(tdst, "wb") as f:
                writer(f)
            if os.path.getsize(tdst) == want:
                os.replace(tdst, dst)
                return
        except OSError as e:
            att += 1
            if att % 10 == 1:
                log(f"write_atomic {dst} retry {att}: {e!r}")
            time.sleep(3)

def write_portal(arr, dst):
    # 分块追加装配: 单个大写会在 ~3.25MB 处被门户掐断; 1MB 块能落进门抖间隙
    att = 0
    tdst = dst + ".asm"
    M = (1 << 20) // 4  # 1MB = 262144 个 int32
    while True:
        try:
            with open(tdst, "wb") as f:
                for a in range(0, arr.size, M):
                    f.write(arr[a:a + M])
            if os.path.getsize(tdst) == arr.nbytes:
                os.replace(tdst, dst)
                return
        except OSError as e:
            att += 1
            if att % 10 == 1:
                log(f"write_portal {dst} retry {att}: {e!r}")
            time.sleep(3)

CHUNK = 1 << 20  # 8MB: 小段写能落进门抖间隙
NC = None  # 运行时按 E 计算

def part_path(dst, tag, k):
    return f"{dst}.{tag}pt{k:02d}"

def write_parts(arr, dst, tag):
    # 分段直写; 全部落盘后即为该 tag 的一份完整 ckpt
    nc = (arr.nbytes + CHUNK - 1) // CHUNK
    for k in range(nc):
        sl = arr[k * (CHUNK // 4):(k + 1) * (CHUNK // 4)]
        write_atomic(lambda f, sl=sl: sl.tofile(f), part_path(dst, tag, k), sl.nbytes)
    return nc

def read_parts(dst, tag, nbytes):
    nc = (nbytes + CHUNK - 1) // CHUNK
    outs = []
    for k in range(nc):
        p = part_path(dst, tag, k)
        want = min(CHUNK, nbytes - k * CHUNK)
        if stat_retry(p, tries=3) != want:
            return None
        outs.append(np.fromfile(p, dtype=np.int32))
    return np.concatenate(outs) if outs else None

def copy_portal(src, dst):
    def w(f):
        with open(src, "rb") as s_:
            while True:
                b = s_.read(1 << 24)
                if not b:
                    break
                f.write(b)
    write_atomic(w, dst, stat_retry(src) or -1)

def stat_retry(path, tries=20):
    for _ in range(tries):
        try:
            return os.path.getsize(path)
        except OSError:
            time.sleep(3)
    return None

nt = np.fromfile(f"{D}/node_type.i32", dtype=np.int32)
N = int(nt.size)
xyz = np.fromfile(f"{D}/xyz.f32", dtype=np.float32).reshape(N, 3)
pre_r = np.fromfile(REAL_PRE, dtype=np.int32)
post_r = np.fromfile(REAL_POST, dtype=np.int32)
gp = np.fromfile(SRC["pre"], dtype=np.int32)
E = gp.size
log(f"load ok  N={N:,} E={E:,} recip_real={RECIP_REAL:.4f}")

# 真实互惠配额 m_i(真实边已排序 ⇒ 行切片合法)
kr = np.unique(pre_r.astype(np.int64) * N + post_r)
rk = post_r.astype(np.int64) * N + pre_r
p = np.clip(np.searchsorted(kr, rk), 0, kr.size - 1)
mok = kr[p] == rk
m_i = np.bincount(pre_r[mok], minlength=N).astype(np.int64)
del kr, rk, p, mok
gc.collect()
log("real m_i done")

# 类型成员表(修剪候选采样用)
torder = np.argsort(nt, kind="stable")
tuniq, tstart, tcount = np.unique(nt[torder], return_index=True, return_counts=True)
tpos = np.full(int(nt.max()) + 1, -1, dtype=np.int64)
tpos[tuniq] = np.arange(tuniq.size)
log("type member table done")

# ---------- 工作态 ----------
ADDF = DST["post"] + ".added2"
RMF = DST["post"] + ".removed"
CURF = DST["post"] + ".cur2"
# 先读游标取轮替 tag(4 字段新格式; 3 字段旧格式用无后缀 sidecar)
ckpt_tag = None
if stat_retry(CURF, tries=3):
    try:
        _p = open(CURF).read().split()
        if len(_p) == 4 and _p[0] in ("A", "B"):
            ckpt_tag = _p[0]
    except OSError:
        pass
SFX = "." + ckpt_tag if ckpt_tag else ""
gq = None
resumed = False
if ckpt_tag:
    gq = read_parts(DST["post"], ckpt_tag, E * 4)
    if gq is not None:
        resumed = True
        log(f"resume: loaded parts set {ckpt_tag}")
if gq is None and stat_retry(DST["post"]) == E * 4:
    gq = np.fromfile(DST["post"], dtype=np.int32).copy()
    resumed = True
    log("resume: loaded existing DST post")
# 一致性校验(仅续跑态; 源态信任): 基座天生含重复键/自环(多突触对, 实测签名见下),
# 修剪只会增 unique、减自环 ⇒ 合法工作态 unique >= BASE_U 且 self_loops <= BASE_SL;
# 越界即分段集/文件损坏, 回退源态重跑
BASE_U, BASE_SL = (18638016, 63765) if SEX == "female" else (23426144, 84444)
if resumed and gq is not None:
    _kk = gp.astype(np.int64) * N + gq
    _u = int(np.unique(_kk).size)
    _sl = int((gp == gq).sum())
    del _kk
    gc.collect()
    if _u < BASE_U or _sl > BASE_SL:
        log(f"!! insane resumed state (unique={_u:,} self_loops={_sl:,}; base {BASE_U:,}/{BASE_SL:,}) — discard")
        gq = None
if gq is None:
    gq = np.fromfile(SRC["post"], dtype=np.int32).copy()
    for f_ in (ADDF, RMF, CURF, ADDF + ".A", ADDF + ".B", RMF + ".A", RMF + ".B",
               DST["post"] + ".surplus", DST["post"] + ".surplus.A", DST["post"] + ".surplus.B"):
        try:
            os.remove(f_)
        except OSError:
            pass
    for f_ in ([DST["post"], DST["post"] + ".asm"] +
               [part_path(DST["post"], tg, k) for tg in ("A", "B") for k in range(200)]):
        try:
            os.remove(f_)
        except OSError:
            pass
    log("fresh state from source (sidecars/parts purged)")
for k in ("pre", "w"):
    if stat_retry(DST[k]) != E * 4:
        copy_portal(SRC[k], DST[k])
        log(f"copied {k} -> DST")

def csr_starts(arr):
    return np.concatenate([[0], np.cumsum(np.bincount(arr, minlength=N))]).astype(np.int64)

added, removed = set(), set()
_asz = stat_retry(ADDF + SFX, tries=3)
if _asz:
    added = set(np.fromfile(ADDF + SFX, dtype=np.int64).tolist())
_rsz = stat_retry(RMF + SFX, tries=3)
if _rsz:
    removed = set(np.fromfile(RMF + SFX, dtype=np.int64).tolist())

order = np.argsort(gq, kind="stable")
in_adj = gp[order]          # 行内按 source 升序(gp 排序基 + stable)
in_start = csr_starts(gq)
# P8f: 对称共同邻居用静态 CSR(MUTQ 基座, 与工作态解耦 ⇒ 续跑/重生一致; union(out,in) 去重, 行内升序)
_gqb = np.fromfile(SRC["post"], dtype=np.int32)
_sym_e = np.unique(np.concatenate([gp.astype(np.int64) * N + _gqb,
                                   _gqb.astype(np.int64) * N + gp]))
del _gqb
sym_adj = (_sym_e % N).astype(np.int32)
sym_start = csr_starts((_sym_e // N).astype(np.int32))
del _sym_e
gc.collect()
mark = np.zeros(N, dtype=np.int32)  # epoch 戳记, 免清零
log(f"sym csr done: {sym_adj.size:,} sym edges")

# ---- P8g: 晶格模态偏移候选池(复用 75/96b 机械) ----
from scipy.spatial import cKDTree
_zq = np.load(f"{D}/lat2q_f3.npz")
PQ, HEM = _zq["pq"].astype(np.float64), _zq["hem"]
has_c = HEM != 0
del _zq
_sn = np.load(f"{D}/lat_snap_f2.npz")
sn_uk, sn_cr, sn_cr_g, LR = _sn["big_uk"], _sn["crbig"], _sn["cr_glob"], int(_sn["R"])
LNB1 = 2 * LR + 1
NB21 = LNB1 * LNB1
del _sn
T_nt = int(nt.max()) + 1

def pair_mode(key):
    r2 = int(np.searchsorted(sn_uk, key))
    cr = sn_cr[r2] if (r2 < len(sn_uk) and sn_uk[r2] == key) else sn_cr_g
    pr = np.diff(np.concatenate([[0.0], cr]))
    bm = int(np.argmax(pr))
    if bm == NB21 + 1:
        return None, None, True, False   # CROSS
    if bm == NB21:
        return None, None, False, True   # OVER
    return float(bm // LNB1 - LR), float(bm % LNB1 - LR), False, False

_tree_cache = {}
def type_tree(t_, h_):
    kk_ = (t_, h_)
    if kk_ not in _tree_cache:
        tp_ = tpos[t_]
        seg = torder[tstart[tp_]:tstart[tp_] + tcount[tp_]] if tp_ >= 0 else np.array([], dtype=np.int64)
        mk = has_c[seg] & (HEM[seg] == h_)
        pts = PQ[seg[mk]]
        _tree_cache[kk_] = (cKDTree(pts), seg[mk].astype(np.int64)) if mk.sum() else (None, None)
    return _tree_cache[kk_]
log("lattice pool machinery done")
del order
out_start = csr_starts(gp)
kk = np.unique(gp.astype(np.int64) * N + gq)   # 当前已占键(初始态; 增量由 added/removed 维护)
gc.collect()
log("csr + key set done")

# 当前互惠计数 cur_i
gk = gq.astype(np.int64) * N + gp
pp = np.clip(np.searchsorted(kk, gk), 0, kk.size - 1)
gmut = kk[pp] == gk
cur = np.bincount(gp[gmut], minlength=N).astype(np.int64)
del gk, pp, gmut
gc.collect()
surplus = cur - m_i
log(f"cur done: surplus cells={int((surplus>0).sum()):,}  mass={int(np.clip(surplus,0,None).sum()):,}")

rng = np.random.default_rng(20261004 if SEX == "female" else 20261005)
CKPT_SECS = 30
start_i = 0
trims_total, trims_paired = 0, 0
if stat_retry(CURF, tries=3):
    try:
        with open(CURF) as f:
            _p = f.read().split()
        start_i, trims_total, trims_paired = (int(x) for x in _p[-3:])
        # 游标续跑时 surplus 无法精确复原 ⇒ 重新载入 sidecar surplus 文件(与游标同 tag)
        sf = np.fromfile(DST["post"] + ".surplus" + SFX, dtype=np.int64)
        surplus = sf
        log(f"resume cursor: cell {start_i:,}  trims={trims_total:,} (paired {trims_paired:,})")
    except Exception:
        start_i, trims_total, trims_paired = 0, 0, 0

cn_sum, cn_n, cn_zero, cn_max = 0, 0, 0, 0
lat_served_g, lat_fallback_g = [0], [0]
CNF = DST["post"] + ".cnstat"
if stat_retry(CNF, tries=3):
    try:
        cn_sum, cn_n, cn_zero, cn_max = (int(x) for x in open(CNF).read().split())
    except Exception:
        pass
last_ckpt = time.time()
last_log = time.time()
ckpt_no = 0
for i in range(start_i, N):
    if surplus[i] <= 0:
        continue
    o0, o1 = out_start[i], out_start[i + 1]
    if o1 == o0:
        continue
    js = gq[o0:o1].copy()
    # 当前互惠掩码: 初始键 ∪ added − removed
    rv = js.astype(np.int64) * N + i
    pq = np.clip(np.searchsorted(kk, rv), 0, kk.size - 1)
    mut = kk[pq] == rv
    if added:
        mut |= np.fromiter((x in added for x in rv), dtype=bool, count=rv.size)
    if removed:
        mut &= ~np.fromiter((x in removed for x in rv), dtype=bool, count=rv.size)
    midx = np.nonzero(mut)[0]
    if midx.size == 0:
        continue
    outset = set(js.tolist())
    i0, i1 = in_start[i], in_start[i + 1]
    inset = set(in_adj[i0:i1].tolist())
    # 已占邻接(当前): 出集 ∪ 入集; added/removed 修正入侧
    xi = xyz[i]
    # 优先: 双端盈余对(伙伴 surplus>0); 兜底: 其余互惠边
    part = surplus[js[midx]] > 0
    order_idx = list(midx[part]) + list(midx[~part])
    did = 0
    for idx in order_idx:
        if surplus[i] <= 0:
            break
        j_old = int(js[idx])
        paired = surplus[j_old] > 0
        t = int(nt[j_old])
        d_old = float(np.sqrt(((xyz[j_old] - xi) ** 2).sum()))
        # 候选: 同型、与 i 完全不邻接(当前)、距离最近
        tp = tpos[t]
        if tp < 0:
            continue
        members = torder[tstart[tp]:tstart[tp] + tcount[tp]]
        cand = None
        if has_c[i]:
            dp0, dq0, is_cross, is_over = pair_mode(int(nt[i]) * T_nt + t)
            if dp0 is None:
                dp0, dq0 = 0.0, 0.0  # CROSS: 同位镜像(半球翻转, 偏移取 0)
            if not is_over:
                h_ = int(-HEM[i] if is_cross else HEM[i])
                tree, loc = type_tree(t, h_)
                if tree is not None:
                    k16 = min(16, loc.size)
                    _, ii = tree.query(PQ[i] + np.array([dp0, dq0]), k=k16)
                    cand = loc[np.atleast_1d(ii)]
                    lat_served_g[0] += 1
        if cand is None:
            cand = members[rng.integers(0, members.size, size=min(64, members.size))]
            lat_fallback_g[0] += 1
        epoch = i + 1  # 戳记(i 单增不撞)
        mark[sym_adj[sym_start[i]:sym_start[i + 1]]] = epoch
        best, bestd, bestcn = -1, np.inf, -1
        for c in cand.tolist():
            if c == i or c in outset:
                continue
            key_r = c * N + i
            in_cur = (c in inset and key_r not in removed) or (key_r in added)
            if in_cur:
                continue
            cn = int((mark[sym_adj[sym_start[c]:sym_start[c + 1]]] == epoch).sum())
            dd = abs(float(np.sqrt(((xyz[c] - xi) ** 2).sum())) - d_old)
            if cn > bestcn or (cn == bestcn and dd < bestd):
                best, bestd, bestcn = c, dd, cn
        if best < 0:
            continue
        gq[o0 + idx] = best
        outset.discard(j_old)
        outset.add(best)
        removed.add(i * N + j_old)
        added.add(i * N + best)
        cn_sum += bestcn; cn_n += 1
        cn_zero += (bestcn == 0); cn_max = max(cn_max, bestcn)
        surplus[i] -= 1
        if paired:
            surplus[j_old] -= 1
            trims_paired += 1
        trims_total += 1
        did += 1
    if did:
        now = time.time()
        if now - last_log > 20:
            log(f"  progress: cell {i:,}/{N:,}  trims={trims_total:,} (paired {trims_paired:,})")
            last_log = now
        if now - last_ckpt > CKPT_SECS:
            # 全量轮替: 撕开写永远落在非活动集, 活动集始终完整一致(增量原位写曾造成跨代混杂)
            tag = "B" if ckpt_no % 2 else "A"
            write_parts(gq, DST["post"], tag)
            _a = np.fromiter(added, dtype=np.int64, count=len(added))
            write_atomic(lambda f: _a.tofile(f), ADDF + "." + tag, _a.nbytes)
            _r = np.fromiter(removed, dtype=np.int64, count=len(removed))
            write_atomic(lambda f: _r.tofile(f), RMF + "." + tag, _r.nbytes)
            write_atomic(lambda f: surplus.tofile(f), DST["post"] + ".surplus." + tag, surplus.nbytes)
            _cn = f"{cn_sum} {cn_n} {cn_zero} {cn_max}".encode()
            write_atomic(lambda f: f.write(_cn), CNF, len(_cn))
            cur_txt = f"{tag} {i + 1} {trims_total} {trims_paired}".encode()
            write_atomic(lambda f: f.write(cur_txt), CURF, len(cur_txt))
            ckpt_no += 1
            last_ckpt = time.time()
            log(f"  ckpt[{tag}]: cell {i + 1:,}  trims={trims_total:,}")

write_portal(gq, DST["post"])
_a = np.fromiter(added, dtype=np.int64, count=len(added))
write_atomic(lambda f: _a.tofile(f), ADDF, _a.nbytes)
_r = np.fromiter(removed, dtype=np.int64, count=len(removed))
write_atomic(lambda f: _r.tofile(f), RMF, _r.nbytes)
cur_txt = f"Z {N} {trims_total} {trims_paired}".encode()
write_atomic(lambda f: f.write(cur_txt), CURF, len(cur_txt))
log(f"trim done: trims={trims_total:,} (paired {trims_paired:,})")

# 终态 recip
kk2 = np.unique(gp.astype(np.int64) * N + gq)
gk2 = gq.astype(np.int64) * N + gp
p2 = np.clip(np.searchsorted(kk2, gk2), 0, kk2.size - 1)
recip_new = float((kk2[p2] == gk2).sum()) / E
log(f"recip: -> {recip_new:.5f}  (real {RECIP_REAL:.4f}, band ±0.005)")

h = hashlib.md5()
with open(DST["post"], "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
res = dict(sex=SEX, E=int(E), trims=int(trims_total), trims_paired=int(trims_paired),
           cn_mean=round(cn_sum / max(cn_n, 1), 3), cn_zero_frac=round(cn_zero / max(cn_n, 1), 4),
           cn_max=int(cn_max), lat_served=int(lat_served_g[0]), lat_fallback=int(lat_fallback_g[0]),
           recip_new=round(recip_new, 5), recip_real=round(RECIP_REAL, 5),
           gate=f"|recip-real|<=0.005", passed=bool(abs(recip_new - RECIP_REAL) <= 0.005),
           post_md5=h.hexdigest(), ts=time.strftime("%F %T"))
with open(f"{D}/trim_result.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log("saved", f"{D}/trim_result.json")
