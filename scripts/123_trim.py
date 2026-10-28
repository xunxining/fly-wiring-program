#!/usr/bin/env python3
# 123_trim.py — §36 P8e 互惠盈余修剪(对级贪心 + 单端兜底), 抗收割机(原子 ckpt + 游标)
# 对 cur_i>m_i 的细胞 i: 互惠出边 i→j 改挂到同型、与 i 完全不邻接、距离相近的 j''
# (构造上: 新边非互惠、零碰撞、CELLQ/出度/w 不变)。双端盈余对优先(一次修剪服务两端配额)。
# 用法: python3 123_trim.py female|male
import json, os, sys, time, hashlib, gc
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器"
SEX = sys.argv[1]
if SEX == "female":
    D = f"{ROOT}/data_female"
    SRC = {k: f"{ROOT}/gen_edges/g_mutq_{k}.i32" for k in ("pre", "post", "w")}
    DST = {k: f"{ROOT}/gen_edges/g_mutq2_{k}.i32" for k in ("pre", "post", "w")}
    REAL_PRE, REAL_POST = f"{D}/e_pre.i32", f"{D}/e_post.i32"
    RECIP_REAL = 0.32130410937717807
else:
    D = f"{ROOT}/data_male"
    SRC = {k: f"{ROOT}/gen_edges/g_mutqm_{k}.i32" for k in ("pre", "post", "w")}
    DST = {k: f"{ROOT}/gen_edges/g_mutqm2_{k}.i32" for k in ("pre", "post", "w")}
    REAL_PRE, REAL_POST = f"{D}/e_pre_s.i32", f"{D}/e_post_s.i32"
    RECIP_REAL = 0.2997500837059255
if len(sys.argv) >= 4:
    SRC = {k: f"{ROOT}/gen_edges/{sys.argv[2]}_{k}.i32" for k in ("pre", "post", "w")}
    DST = {k: f"{ROOT}/gen_edges/{sys.argv[3]}_{k}.i32" for k in ("pre", "post", "w")}
JSON_NAME = sys.argv[4] if len(sys.argv) >= 5 else "trim_result.json"

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
        cand = members[rng.integers(0, members.size, size=min(64, members.size))]
        best, bestd = -1, np.inf
        for c in cand.tolist():
            if c == i or c in outset:
                continue
            key_r = c * N + i
            in_cur = (c in inset and key_r not in removed) or (key_r in added)
            if in_cur:
                continue
            dd = abs(float(np.sqrt(((xyz[c] - xi) ** 2).sum())) - d_old)
            if dd < bestd:
                best, bestd = c, dd
        if best < 0:
            continue
        gq[o0 + idx] = best
        outset.discard(j_old)
        outset.add(best)
        removed.add(i * N + j_old)
        added.add(i * N + best)
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
           recip_new=round(recip_new, 5), recip_real=round(RECIP_REAL, 5),
           gate=f"|recip-real|<=0.005", passed=bool(abs(recip_new - RECIP_REAL) <= 0.005),
           post_md5=h.hexdigest(), ts=time.strftime("%F %T"))
with open(f"{D}/{JSON_NAME}", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log("saved", f"{D}/{JSON_NAME}")
