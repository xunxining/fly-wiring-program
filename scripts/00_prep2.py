#!/usr/bin/env python3
"""过滤到神经元级图(两端均注释)并映射为连续索引，落盘二进制"""
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import numpy as np
import gc, resource

D = "/tmp/flydata/"

def log(*a):
    print(*a, flush=True)

ann = pd.read_feather(D + "body-annotations.feather")
ann = ann.sort_values("bodyId").reset_index(drop=True)
node_ids = ann["bodyId"].to_numpy(np.int64)
ann_ids = set(node_ids.tolist())
N = len(node_ids)
log("nodes:", N)

mm = pa.memory_map(D + "connectome-weights.feather")
reader = ipc.open_file(mm)
n_batches = reader.num_record_batches

M = 26028386  # 已统计
pre_mm = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="w+", shape=(M,))
post_mm = np.memmap(D + "e_post.i32", dtype=np.int32, mode="w+", shape=(M,))
w_mm = np.memmap(D + "e_w.i32", dtype=np.int32, mode="w+", shape=(M,))

off = 0
for i in range(n_batches):
    b = reader.get_batch(i)
    p = b.column("body_pre").to_numpy(zero_copy_only=False)
    q = b.column("body_post").to_numpy(zero_copy_only=False)
    ww = b.column("weight").to_numpy(zero_copy_only=False)
    mask = np.isin(p, list(ann_ids), assume_unique=False) & np.isin(q, list(ann_ids), assume_unique=False)
    # np.isin with big list is slow; use searchsorted-based membership instead
    mask = (node_ids[np.clip(np.searchsorted(node_ids, p), 0, N-1)] == p) & \
           (node_ids[np.clip(np.searchsorted(node_ids, q), 0, N-1)] == q)
    pi = np.searchsorted(node_ids, p[mask])
    qi = np.searchsorted(node_ids, q[mask])
    n = int(mask.sum())
    pre_mm[off:off+n] = pi.astype(np.int32)
    post_mm[off:off+n] = qi.astype(np.int32)
    w_mm[off:off+n] = ww[mask].astype(np.int32)
    off += n
    del b, p, q, ww, mask, pi, qi
    if i % 200 == 0:
        log(f"batch {i}/{n_batches}, kept={off}, rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB")

pre_mm.flush(); post_mm.flush(); w_mm.flush()
assert off == M, f"{off} != {M}"
ann.to_feather(D + "nodes.feather")
log("PREP DONE, edges:", off)
