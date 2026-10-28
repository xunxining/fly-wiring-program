#!/usr/bin/env python3
"""流式预处理：feather -> 映射后的二进制边表 + 节点表"""
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import os, gc, resource

D = "/tmp/flydata/"
SRC = D + "connectome-weights.feather"

def log(*a):
    print(*a, flush=True)

# ---------- Pass 1: 唯一 id ----------
mm = pa.memory_map(SRC)
reader = ipc.open_file(mm)
n_batches = reader.num_record_batches
uniq = set()
for i in range(n_batches):
    b = reader.get_batch(i)
    uniq.update(np.unique(b.column("body_pre").to_numpy(zero_copy_only=False)).tolist())
    uniq.update(np.unique(b.column("body_post").to_numpy(zero_copy_only=False)).tolist())
    del b
    if i % 400 == 0:
        log(f"pass1 batch {i}/{n_batches}, uniq={len(uniq)}, rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB")
ids = np.array(sorted(uniq), dtype=np.int64)
np.save(D + "node_ids.npy", ids)
log("unique bodies:", len(ids))
del uniq
gc.collect()

# ---------- Pass 2: 映射为 int32 索引并落盘 ----------
n_nodes = len(ids)
# 先数行数
total_rows = 0
for i in range(n_batches):
    total_rows += reader.get_batch(i).num_rows
log("total rows:", total_rows)

pre_mm = np.memmap(D + "edges_pre.i32", dtype=np.int32, mode="w+", shape=(total_rows,))
post_mm = np.memmap(D + "edges_post.i32", dtype=np.int32, mode="w+", shape=(total_rows,))
w_mm = np.memmap(D + "edges_w.i32", dtype=np.int32, mode="w+", shape=(total_rows,))

# 度数统计
indeg_e = np.zeros(n_nodes, dtype=np.int64)
outdeg_e = np.zeros(n_nodes, dtype=np.int64)
indeg_s = np.zeros(n_nodes, dtype=np.int64)   # 突触数计
outdeg_s = np.zeros(n_nodes, dtype=np.int64)
WMAX = int(1e4)
w_hist = np.zeros(WMAX + 2, dtype=np.int64)   # 权重直方图(截断到1e4)

off = 0
for i in range(n_batches):
    b = reader.get_batch(i)
    p = np.searchsorted(ids, b.column("body_pre").to_numpy(zero_copy_only=False)).astype(np.int32)
    q = np.searchsorted(ids, b.column("body_post").to_numpy(zero_copy_only=False)).astype(np.int32)
    ww = b.column("weight").to_numpy(zero_copy_only=False)
    n = len(p)
    pre_mm[off:off+n] = p
    post_mm[off:off+n] = q
    w_mm[off:off+n] = ww.astype(np.int32)
    np.add.at(outdeg_e, p, 1)
    np.add.at(indeg_e, q, 1)
    np.add.at(outdeg_s, p, ww)
    np.add.at(indeg_s, q, ww)
    wc = np.clip(ww, 0, WMAX + 1)
    np.add.at(w_hist, wc, 1)
    off += n
    del b, p, q, ww
    if i % 200 == 0:
        log(f"pass2 batch {i}/{n_batches}, rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB")

pre_mm.flush(); post_mm.flush(); w_mm.flush()
np.save(D + "indeg_e.npy", indeg_e)
np.save(D + "outdeg_e.npy", outdeg_e)
np.save(D + "indeg_s.npy", indeg_s)
np.save(D + "outdeg_s.npy", outdeg_s)
np.save(D + "w_hist.npy", w_hist)
log("PREP DONE")
