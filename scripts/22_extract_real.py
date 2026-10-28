#!/usr/bin/env python3
"""Stage 22: 从重下的连接表提取真实嗅觉子网络边
流式扫 feather 行组, 过滤 pre∈{PN,KC}, post∈{KC,MBON} 的边
输出: /tmp/flydata/olf/real_edges.npz (bodyId 空间)
"""
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pickle

D = "/tmp/flydata/"
OUT = "/mnt/agents/output/果蝇生成器/"

def log(*a): print(*a, flush=True)

types = np.load(D + "type_names.npy", allow_pickle=True)
t2i = {t: i for i, t in enumerate(types)}
sets = pickle.load(open(D + "olf_sets.pkl", "rb"))
pn, kc, mbon = sets["pn"], sets["kc"], sets["mbon"]

ann = pd.read_feather(OUT + "data/body-annotations.feather", columns=["bodyId", "type"])
ann = ann.dropna(subset=["type"])
body2t = dict(zip(ann["bodyId"].to_numpy(), ann["type"].map(t2i).to_numpy()))

pn_set = set(ann.loc[ann["type"].isin(pn), "bodyId"].tolist())
kc_set = set(ann.loc[ann["type"].isin(kc), "bodyId"].tolist())
mbon_set = set(ann.loc[ann["type"].isin(mbon), "bodyId"].tolist())
log(f"bodies: PN={len(pn_set)}, KC={len(kc_set)}, MBON={len(mbon_set)}")

src_set = pn_set | kc_set
dst_set = kc_set | mbon_set

f = feather.read_table  # placeholder
# 流式: 用 pyarrow 的 RecordBatchFileReader 逐批读
import pyarrow as pa
reader = pa.ipc.open_file(D + "connectome-weights.feather")
log("batches:", reader.num_record_batches)

keep_pre, keep_post, keep_w = [], [], []
cols = None
for bi in range(reader.num_record_batches):
    b = reader.get_batch(bi)
    if cols is None:
        cols = b.schema.names
        log("cols:", cols)
    df = b.to_pandas()
    pcol, qcol = cols[0], cols[1]
    wcol = cols[2] if len(cols) > 2 else None
    m = df[pcol].isin(src_set) & df[qcol].isin(dst_set)
    if m.any():
        sub = df[m]
        keep_pre.append(sub[pcol].to_numpy())
        keep_post.append(sub[qcol].to_numpy())
        keep_w.append(sub[wcol].to_numpy() if wcol else np.ones(m.sum(), np.int32))
    if bi % 200 == 0:
        log(f"  batch {bi}/{reader.num_record_batches}, kept {sum(len(x) for x in keep_pre):,}")

pre = np.concatenate(keep_pre) if keep_pre else np.array([], np.int64)
post = np.concatenate(keep_post) if keep_post else np.array([], np.int64)
w = np.concatenate(keep_w) if keep_w else np.array([], np.int32)
log(f"extracted {len(pre):,} edges, synapses {w.astype(np.int64).sum():,}")
np.savez(D + "olf/real_edges.npz", pre=pre, post=post, w=w)
log("EXTRACT DONE")
