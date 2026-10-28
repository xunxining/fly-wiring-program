#!/usr/bin/env python3
# 117_mutq_repair.py — §35 P8d MUTQ 后处理输运修复(幂等可续跑, 稀疏类型版)
# 缺口细胞 i 的同型非互惠出边 i→j0 改挂到同型候选 j'(已有 j'→i 且 j' 非 i 出邻居
# ⇒ 构造上零碰撞、出度分布不变);候选按 |dist(j',i)-dist(j0,i)| 最小优先(护 dist guard)。
# 工作文件即门户 DST post:启动时若已存在则载入续跑(已修复细胞自然跳过)。
# 用法: python3 117_mutq_repair.py female|male
import json, os, sys, time, hashlib, gc
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器"
SEX = sys.argv[1]
if SEX == "female":
    D = f"{ROOT}/data_female"
    SRC = {k: f"{ROOT}/gen_edges/g_dyad3s_{k}.i32" for k in ("pre", "post", "w")}
    DST = {k: f"{ROOT}/gen_edges/g_mutq_{k}.i32" for k in ("pre", "post", "w")}
    GATE = 0.31
    REAL_PRE, REAL_POST = f"{D}/e_pre.i32", f"{D}/e_post.i32"  # 雌性真实边已验证升序
else:
    D = f"{ROOT}/data_male"
    SRC = {k: f"{ROOT}/gen_edges/gm_p8cs_{k}.i32" for k in ("pre", "post", "w")}
    DST = {k: f"{ROOT}/gen_edges/g_mutqm_{k}.i32" for k in ("pre", "post", "w")}
    GATE = 0.297
    REAL_PRE, REAL_POST = f"{D}/e_pre_s.i32", f"{D}/e_post_s.i32"  # 120 排序产物

# §36 P8e 补填轮: 可选 argv[2]=src_prefix argv[3]=dst_prefix argv[4]=json_name 覆盖(默认即 MUTQ 行为)
if len(sys.argv) >= 4:
    SRC = {k: f"{ROOT}/gen_edges/{sys.argv[2]}_{k}.i32" for k in ("pre", "post", "w")}
    DST = {k: f"{ROOT}/gen_edges/{sys.argv[3]}_{k}.i32" for k in ("pre", "post", "w")}
JSON_NAME = sys.argv[4] if len(sys.argv) >= 5 else "mutq_repair.json"

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
    while True:  # 门户抖动: 重试直到整块落盘+原子改名(ckpt 绝不截断已有状态)
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

def part_path(dst, tag, k):
    return f"{dst}.{tag}pt{k:02d}"

def write_parts(arr, dst, tag):
    nc = (arr.nbytes + CHUNK - 1) // CHUNK
    for k in range(nc):
        sl = arr[k * (CHUNK // 4):(k + 1) * (CHUNK // 4)]
        write_atomic(lambda f, sl=sl: sl.tofile(f), part_path(dst, tag, k), sl.nbytes)

def read_parts(dst, tag, nbytes):
    nc = (nbytes + CHUNK - 1) // CHUNK
    outs = []
    for k in range(nc):
        p = part_path(dst, tag, k)
        if stat_retry(p, tries=3) != min(CHUNK, nbytes - k * CHUNK):
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

def save_added(added, ADDF):
    _a = np.fromiter(added, dtype=np.int64, count=len(added))
    write_atomic(lambda f: _a.tofile(f), ADDF, _a.nbytes)

nt = np.fromfile(f"{D}/node_type.i32", dtype=np.int32)
N = int(nt.size)
xyz = np.fromfile(f"{D}/xyz.f32", dtype=np.float32).reshape(N, 3)
pre_r = np.fromfile(REAL_PRE, dtype=np.int32)
post_r = np.fromfile(REAL_POST, dtype=np.int32)
E_r = pre_r.size
gp = np.fromfile(SRC["pre"], dtype=np.int32)
E = gp.size
log(f"load ok  N={N:,} E_real={E_r:,} E_gen={E:,} gate={GATE}")

# ---------- 工作态 gq_new: 存在则续跑,否则从源初始化 ----------
ADDF = DST["post"] + ".added"
def stat_retry(path, tries=20):
    for _ in range(tries):  # EIO 时重试; 绝不把抖动误判为"不存在"
        try:
            return os.path.getsize(path)
        except OSError:
            time.sleep(3)
    return None

CURF = DST["post"] + ".cur"
ckpt_tag = None
if stat_retry(CURF, tries=3):
    try:
        _p = open(CURF).read().split()
        if len(_p) == 4 and _p[0] in ("A", "B"):
            ckpt_tag = _p[0]
    except OSError:
        pass
SFX = "." + ckpt_tag if ckpt_tag else ""
gq_new = None
resumed = False
if ckpt_tag:
    gq_new = read_parts(DST["post"], ckpt_tag, E * 4)
    if gq_new is not None:
        resumed = True
        log(f"resume: loaded parts set {ckpt_tag}")
if gq_new is None and stat_retry(DST["post"]) == E * 4:
    gq_new = np.fromfile(DST["post"], dtype=np.int32).copy()
    resumed = True
    log("resume: loaded existing DST post as working state")
# 一致性校验(仅续跑态; 源态信任): 基座天生含重复键/自环(多突触对, 实测签名见下,
# 阈值为 P8e 基座 g_mutq/g_mutqm 的签名; 补填只增 unique、减自环)
BASE_U, BASE_SL = (18638016, 63765) if SEX == "female" else (23426144, 84444)
if resumed and gq_new is not None:
    _kk = gp.astype(np.int64) * N + gq_new
    _u = int(np.unique(_kk).size)
    _sl = int((gp == gq_new).sum())
    del _kk
    gc.collect()
    if _u < BASE_U or _sl > BASE_SL:
        log(f"!! insane resumed state (unique={_u:,} self_loops={_sl:,}; base {BASE_U:,}/{BASE_SL:,}) — discard")
        gq_new = None
if gq_new is None:
    if SRC["post"] == DST["post"]:
        # SRC==DST(P8e 轮): 源即工作文件。已判损或缺失 ⇒ 无可回退, 致命退出(看门狗/日志可见)
        log("FATAL: SRC==DST and no sane source state available — aborting")
        sys.exit(1)
    gq_new = np.fromfile(SRC["post"], dtype=np.int32).copy()
    PURGE = ([ADDF, CURF, ADDF + ".A", ADDF + ".B", DST["post"] + ".asm"] +
             [part_path(DST["post"], tg, k) for tg in ("A", "B") for k in range(200)])
    PURGE.append(DST["post"])  # 仅 SRC!=DST 时安全(SRC==DST 已在上面致命退出)
    for f_ in PURGE:
        try:
            if os.path.exists(f_):
                os.remove(f_)
        except OSError:
            pass  # 工作态重置 ⇒ 旧的保护键/游标/分段/成品全部作废
    log("initialized from source (parts-only ckpt from here on)")
# ---------- gen/real csr(bincount 起点 + stable argsort ⇒ 行内 source 升序) ----------
def csr_starts(arr):
    return np.concatenate([[0], np.cumsum(np.bincount(arr, minlength=N))]).astype(np.int64)

order = np.argsort(gq_new, kind="stable")
in_adj = gp[order]
in_start = csr_starts(gq_new)      # 行内 in_adj 按 source 升序(stable 保持原 pre 序)
del order
out_start = csr_starts(gp)
rorder = np.argsort(post_r, kind="stable")
rin_adj = pre_r[rorder]
rin_start = csr_starts(post_r)
del rorder
ro_start = csr_starts(pre_r)
gc.collect()
_asz = stat_retry(ADDF + SFX, tries=3)
if _asz:
    added = set(np.fromfile(ADDF + SFX, dtype=np.int64).tolist())
    log(f"resume: loaded {len(added):,} added mutual keys")
else:
    added = set()  # 修复新建的无序对键(保护新建互惠不被后续细胞当 donor 拆走)
gc.collect()
log("gen csr + key set done")

# ---------- 逐缺口细胞修复(类型全程稀疏; 游标续跑免重扫) ----------
CKPT_SECS = 30
start_i, don_total, cells_done = 0, 0, 0
if stat_retry(CURF, tries=3):
    try:
        with open(CURF) as f:
            _p = f.read().split()
        start_i, don_total, cells_done = (int(x) for x in _p[-3:])
        log(f"resume cursor: cell {start_i:,}  retargets={don_total:,}")
    except Exception:
        start_i, don_total, cells_done = 0, 0, 0
last_ckpt = time.time()
last_log = time.time()
ckpt_no = 0
for i in range(start_i, N):
    o0, o1 = out_start[i], out_start[i + 1]
    if o1 == o0:
        continue
    i0, i1 = in_start[i], in_start[i + 1]
    js = gq_new[o0:o1]
    ots = nt[js]
    # 互惠掩码: out-邻居 j 出现在 i 的 in-行(升序) ⇒ 反向边存在
    irow = in_adj[i0:i1]
    if irow.size:
        pos_clip = np.clip(np.searchsorted(irow, js), 0, irow.size - 1)
        mut = irow[pos_clip] == js
    else:
        mut = np.zeros(js.size, dtype=bool)
    rv = js.astype(np.int64) * N + i
    if added:
        mut |= np.fromiter((x in added for x in rv), dtype=bool, count=rv.size)
    # 真实互惠(逐细胞, 同样用 in-行二分)
    rjs = post_r[ro_start[i]:ro_start[i + 1]]
    rrow = rin_adj[rin_start[i]:rin_start[i + 1]]
    if rrow.size:
        rpos = np.clip(np.searchsorted(rrow, rjs), 0, rrow.size - 1)
        rmut = rrow[rpos] == rjs
    else:
        rmut = np.zeros(rjs.size, dtype=bool)
    m_ii = int(rmut.sum())
    if m_ii == 0:
        continue
    deficit = m_ii - int(mut.sum())
    if deficit <= 0:
        continue
    if i1 == i0:
        continue
    ins = irow
    its = nt[ins]
    mt, mc = np.unique(nt[rjs[rmut]], return_counts=True)
    need = dict(zip(mt.tolist(), mc.tolist()))
    ct_, cc_ = np.unique(ots[mut], return_counts=True)
    cur = dict(zip(ct_.tolist(), cc_.tolist()))
    # donor(非互惠出边) / 候选(in-neighbor 且非出邻居非自身), 按类型分组
    di = np.nonzero(~mut)[0]
    ci = np.nonzero(~(np.isin(ins, js) | (ins == i)))[0]
    if di.size == 0 or ci.size == 0:
        continue
    don_t, can_t = ots[di], its[ci]
    xi = xyz[i]
    d_don = np.sqrt(((xyz[js[di]] - xi) ** 2).sum(1))
    d_can = np.sqrt(((xyz[ins[ci]] - xi) ** 2).sum(1))
    don_by_t, can_by_t = {}, {}
    s = np.argsort(don_t, kind="stable")
    for t, a, b in zip(*np.unique(don_t[s], return_index=True, return_counts=True)):
        don_by_t[int(t)] = list(s[a:a + b])
    s = np.argsort(can_t, kind="stable")
    for t, a, b in zip(*np.unique(can_t[s], return_index=True, return_counts=True)):
        can_by_t[int(t)] = list(s[a:a + b])
    present = set(need) | set(don_by_t) | set(can_by_t)
    take = {t: max(0, min(need.get(t, 0) - cur.get(t, 0),
                          len(don_by_t.get(t, [])), len(can_by_t.get(t, []))))
            for t in present}
    got = 0
    def grab(t, k):
        g = 0
        dl = don_by_t.get(t)
        cl = can_by_t.get(t)
        if not dl or not cl:
            return 0
        while g < k and dl and cl:
            dd = dl.pop(0)
            b = int(np.argmin(np.abs(d_can[cl] - d_don[dd])))
            cc = cl.pop(b)
            gq_new[o0 + di[dd]] = ins[ci[cc]]
            added.add(int(i) * N + int(ins[ci[cc]]))
            g += 1
        return g
    for t in sorted(present, key=lambda t: -take[t]):
        if got >= deficit:
            break
        if take[t] > 0:
            got += grab(t, min(take[t], deficit - got))
    if got < deficit:  # float: 剩余缺口用任意类型剩余额度
        for t in sorted(present, key=lambda t: -(len(don_by_t.get(t, [])) *
                                                 len(can_by_t.get(t, [])))):
            if got >= deficit:
                break
            got += grab(t, deficit - got)
    don_total += got
    cells_done += 1
    now = time.time()
    if now - last_log > 20:
        log(f"  progress: cell {i:,}/{N:,}  done={cells_done:,}  retargets={don_total:,}")
        last_log = now
    if now - last_ckpt > CKPT_SECS:
        # 全量轮替: 撕开写永远落在非活动集, 活动集始终完整一致(增量原位写曾造成跨代混杂)
        tag = "B" if ckpt_no % 2 else "A"
        write_parts(gq_new, DST["post"], tag)
        save_added(added, ADDF + "." + tag)
        cur_txt = f"{tag} {i + 1} {don_total} {cells_done}".encode()
        write_atomic(lambda f: f.write(cur_txt), CURF, len(cur_txt))
        ckpt_no += 1
        last_ckpt = time.time()
        log(f"  ckpt[{tag}]: cell {i + 1:,}/{N:,}  done={cells_done:,}  retargets={don_total:,}")

write_portal(gq_new, DST["post"])
save_added(added, ADDF)
cur_txt = f"Z {N} {don_total} {cells_done}".encode()
write_atomic(lambda f: f.write(cur_txt), CURF, len(cur_txt))
log(f"repair done: cells={cells_done:,}  retargets={don_total:,}")

# pre/w 收尾拷贝(与源一致, 循环不需要 ⇒ 挪到最后, 省每生命 60-130s)
for k in ("pre", "w"):
    if stat_retry(DST[k]) != E * 4:
        copy_portal(SRC[k], DST[k])
        log(f"copied {k} -> DST")

# ---------- 终态 recip ----------
uk2 = np.unique(gp.astype(np.int64) * N + gq_new)
rv2 = gq_new.astype(np.int64) * N + gp
p3 = np.clip(np.searchsorted(uk2, rv2), 0, uk2.size - 1)
recip_new = float((uk2[p3] == rv2).sum()) / E
del uk2, rv2, p3
gc.collect()
src_post = np.fromfile(SRC["post"], dtype=np.int32)
uk0 = np.unique(gp.astype(np.int64) * N + src_post)
p4 = np.clip(np.searchsorted(uk0, src_post.astype(np.int64) * N + gp), 0, uk0.size - 1)
recip_old = float((uk0[p4] == src_post.astype(np.int64) * N + gp).sum()) / E
log(f"recip: {recip_old:.4f} -> {recip_new:.4f}  (gate {GATE})")

h = hashlib.md5(open(DST["post"], "rb").read()).hexdigest()
res = dict(sex=SEX, E=int(E), retargets=int(don_total), cells=int(cells_done),
           recip_old=round(recip_old, 5), recip_new=round(recip_new, 5), gate=GATE,
           post_md5=h, ts=time.strftime("%F %T"))
with open(f"{D}/{JSON_NAME}", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log("saved", f"{D}/{JSON_NAME}")
