#!/usr/bin/env python3
"""Stage 32: 全脑位置编码普查 —— 预处理(持久化版)
产物(全部落 /mnt 持久目录 data_census/):
  node_type.i32  (N,)  类型索引(sorted unique types), 无类型=-1
  node_sc.i32    (N,)  superclass索引(基因组 sc_types 顺序, NaN->UNK)
  node_ids.i64   (N,)  bodyId(升序)
  e_pre.i32/e_post.i32/e_w.i32  (E,) 两端均有类型的边, E=25,135,616
  type_names.npy (T,) object
"""
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import numpy as np
import os, resource

ROOT = "/mnt/agents/output/果蝇生成器/"
D = "/tmp/flydata/"
OUT = ROOT + "data_census/"
os.makedirs(OUT, exist_ok=True)

def log(*a): print(*a, flush=True)

# ---------- 节点 ----------
ann = pd.read_feather(ROOT + "data/body-annotations.feather",
                      columns=["bodyId", "type", "superclass"])
ann = ann.sort_values("bodyId").reset_index(drop=True)
node_ids = ann["bodyId"].to_numpy(np.int64)
N = len(node_ids)
types = sorted(ann["type"].dropna().unique())
t2i = {t: i for i, t in enumerate(types)}
node_type = np.full(N, -1, dtype=np.int32)
m = ann["type"].notna().to_numpy()
node_type[m] = ann.loc[m, "type"].map(t2i).to_numpy(np.int32)

g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
sc_types = list(g["sc_types"])
sc2i = {s: i for i, s in enumerate(sc_types)}
node_sc = ann["superclass"].fillna("UNK").map(sc2i).to_numpy(np.int32)
assert len(types) == 11751 and len(sc_types) == 28
node_type.tofile(OUT + "node_type.i32")
node_sc.tofile(OUT + "node_sc.i32")
node_ids.tofile(OUT + "node_ids.i64")
np.save(OUT + "type_names.npy", np.array(types, dtype=object))
log(f"nodes={N}, typed={int((node_type>=0).sum())}, T={len(types)}, SC={len(sc_types)}")

# ---------- 边: 仅保留两端有类型 ----------
E_TOT = 25135616
pre_mm = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="w+", shape=(E_TOT,))
post_mm = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="w+", shape=(E_TOT,))
w_mm = np.memmap(OUT + "e_w.i32", dtype=np.int32, mode="w+", shape=(E_TOT,))

mm = pa.memory_map(D + "connectome-weights.feather")
reader = ipc.open_file(mm)
off = 0
for i in range(reader.num_record_batches):
    b = reader.get_batch(i)
    p = b.column("body_pre").to_numpy(zero_copy_only=False)
    q = b.column("body_post").to_numpy(zero_copy_only=False)
    ww = b.column("weight").to_numpy(zero_copy_only=False)
    pi = np.clip(np.searchsorted(node_ids, p), 0, N - 1)
    qi = np.clip(np.searchsorted(node_ids, q), 0, N - 1)
    ok = (node_ids[pi] == p) & (node_ids[qi] == q) & \
         (node_type[pi] >= 0) & (node_type[qi] >= 0)
    n = int(ok.sum())
    pre_mm[off:off+n] = pi[ok].astype(np.int32)
    post_mm[off:off+n] = qi[ok].astype(np.int32)
    w_mm[off:off+n] = ww[ok].astype(np.int32)
    off += n
    del b, p, q, ww, pi, qi, ok
    if i % 300 == 0:
        log(f"batch {i}, kept={off}, rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB")
pre_mm.flush(); post_mm.flush(); w_mm.flush()
assert off == E_TOT, f"{off} != {E_TOT}"
log("PREP DONE, typed-typed edges:", off)
