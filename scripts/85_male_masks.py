#!/usr/bin/env python3
"""85_male_masks.py — 从 syn-partners feather 流式构建雄性逐节点神经毡掩膜 (census 序)
输出: data_census/np_masks_male.npz  {OUT:(N,K) bool, IN:(N,K) bool, roi_names}
用法: python3 85_male_masks.py /tmp/synp_male.feather
"""
import numpy as np, sys, time
import pyarrow.ipc as ipc
from pyarrow import memory_map

ROOT = "/mnt/agents/output/果蝇生成器/"
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synp_male.feather"
t0 = time.time()
KMAX = 512

node_ids = np.fromfile(ROOT + "data_census/node_ids.i64", dtype=np.int64)
N = len(node_ids)
assert (np.diff(node_ids) > 0).all(), "node_ids 必须有序"
print(f"nodes: {N:,}", flush=True)

f = memory_map(path)
rd = ipc.RecordBatchFileReader(f)
nb = rd.num_record_batches
print(f"batches: {nb}", flush=True)

OUT = np.zeros((N, KMAX), bool); IN = np.zeros((N, KMAX), bool)
roi_code = {}
nrow = 0
for bi in range(nb):
    b = rd.get_batch(bi)
    d = b.column("primary_post")
    cats = d.dictionary.to_pylist()
    codes = np.empty(len(cats), np.int64)
    for ci, name in enumerate(cats):
        if name not in roi_code:
            if len(roi_code) >= KMAX: raise RuntimeError("ROI>KMAX")
            roi_code[name] = len(roi_code)
        codes[ci] = roi_code[name]
    idx = np.asarray(d.indices)
    valid = idx >= 0
    g = np.zeros(len(idx), np.int64); g[valid] = codes[idx[valid]]
    bp = np.asarray(b.column("body_pre")); bq = np.asarray(b.column("body_post"))
    ip = np.searchsorted(node_ids, bp); iq = np.searchsorted(node_ids, bq)
    ip_c = np.minimum(ip, N - 1); iq_c = np.minimum(iq, N - 1)
    okp = valid & (ip < N) & (node_ids[ip_c] == bp)
    okq = valid & (iq < N) & (node_ids[iq_c] == bq)
    OUT[ip_c[okp], g[okp]] = True
    IN[iq_c[okq], g[okq]] = True
    nrow += len(bp)
    if bi % 100 == 0:
        print(f"  batch {bi}/{nb}, rows {nrow:,}, ROI {len(roi_code)}, {time.time()-t0:.0f}s", flush=True)
    del b, bp, bq, ip, iq, ip_c, iq_c
K = len(roi_code)
print(f"rows: {nrow:,}, ROIs: {K}, {time.time()-t0:.0f}s", flush=True)
names = sorted(roi_code, key=roi_code.get)
OUT = OUT[:, :K]; IN = IN[:, :K]
np.savez_compressed(ROOT + "data_census/np_masks_male.npz",
                    OUT=OUT, IN=IN, roi_names=np.array(names))
print(f"OUT 覆盖: {OUT.any(1).mean():.3f}; IN 覆盖: {IN.any(1).mean():.3f}")
print("SAVED data_census/np_masks_male.npz")
