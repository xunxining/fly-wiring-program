#!/usr/bin/env python3
"""Stage 9: S=10 分段生成器 —— v1 均匀实例分配 + v3 经验权重池
用法: python3 09_generate_s10.py <start_pair> <end_pair>
追加写入 gen_S10_{pre,post,w}.i32; 每分段独立进程, 规避内存累积 OOM
缩放原则: 度数守恒 —— 实例数 ×S 时, 类型对边数 ×S, 每神经元期望度数不变
"""
import numpy as np
import sys, os, gc, ctypes

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"
S = 10
Bn = 20_000

def log(*a): print(*a, flush=True)

def trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

start, end = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) > 3:
    Bn = int(sys.argv[3])
rng = np.random.default_rng(1000 + start)

d = np.load(D + "typepair.npz")
uk, cnt = d['uk'].astype(np.int64), d['cnt'].astype(np.int64)
type_count = d['type_count'].astype(np.int64)
T = len(type_count)
tp, tq = uk // T, uk % T
del d; trim()

w_emp = np.load(OUT + "w_emp.npy")
type_sc = np.load(OUT + "type_sc.npy")
NSC = int(round(np.sqrt(w_emp.shape[0])))

new_count = (type_count * S).astype(np.int64)
offsets = np.zeros(T, dtype=np.int64)
offsets[1:] = np.cumsum(new_count)[:-1]

end = min(end, len(uk))
# 写本段独立临时文件, 成功后由驱动脚本并入主文件 —— 主文件永远只含完整段
fpre = open(OUT + f"seg{start}_pre.i32", "wb")
fpost = open(OUT + f"seg{start}_post.i32", "wb")
fw = open(OUT + f"seg{start}_w.i32", "wb")
# 每块写完立即 fsync + fadvise(DONTNEED), 防止脏页/页缓存堆积触发 memcg OOM
def flush_advise(f, off, nbytes):
    f.flush()
    os.fsync(f.fileno())
    os.posix_fadvise(f.fileno(), off, nbytes, os.POSIX_FADV_DONTNEED)
    return off + nbytes
off_pre = off_post = off_w = 0

n_written = 0
for i0 in range(start, end, Bn):
    i1 = min(i0 + Bn, end)
    ne = rng.poisson(cnt[i0:i1].astype(np.float64) * S)
    sel = ne > 0
    if not sel.any():
        continue
    tb_s, tb_t, ne = tp[i0:i1][sel], tq[i0:i1][sel], ne[sel]
    edge_pair = np.repeat(np.arange(len(ne)), ne)
    ep_s = tb_s[edge_pair]; ep_t = tb_t[edge_pair]
    pi = offsets[ep_s] + (rng.random(len(edge_pair)) * new_count[ep_s]).astype(np.int64)
    qi = offsets[ep_t] + (rng.random(len(edge_pair)) * new_count[ep_t]).astype(np.int64)
    wkeys = type_sc[ep_s].astype(np.int64) * NSC + type_sc[ep_t]
    ww = w_emp[wkeys, rng.integers(0, w_emp.shape[1], len(ep_s))]
    pi.astype(np.int32).tofile(fpre)
    qi.astype(np.int32).tofile(fpost)
    ww.astype(np.int32).tofile(fw)
    nb = len(edge_pair) * 4
    off_pre = flush_advise(fpre, off_pre, nb)
    off_post = flush_advise(fpost, off_post, nb)
    off_w = flush_advise(fw, off_w, nb)
    n_written += len(edge_pair)
    del edge_pair, ep_s, ep_t, pi, qi, wkeys, ww
    trim()
    if (i0 - start) % (Bn * 5) == 0:
        log(f"  pairs {i0:,}/{len(uk):,}, seg edges {n_written:,}")

fpre.close(); fpost.close(); fw.close()
log(f"SEGMENT DONE [{start:,},{end:,}): {n_written:,} edges")
