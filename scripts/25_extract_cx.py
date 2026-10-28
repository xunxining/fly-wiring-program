#!/usr/bin/env python3
"""Stage 25: 提取中央复合体航向环路 (EPG/PEN/PEG/Delta7) 真实边 + 递质符号"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pickle

D = "/tmp/flydata/"
OUT = "/mnt/agents/output/果蝇生成器/"

def log(*a): print(*a, flush=True)

CX_TYPES = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]
ann = pd.read_feather(OUT + "data/body-annotations.feather",
                      columns=["bodyId", "type", "instance", "somaSide"])
cx = ann[ann["type"].isin(CX_TYPES)].dropna(subset=["type"]).copy()
log("CX cells:", cx["type"].value_counts().to_dict())

cx_set = set(cx["bodyId"].tolist())
log("total CX bodies:", len(cx_set))

# 递质
nt = pd.read_feather(D + "nt.feather")
nt_cx = nt[nt["body"].isin(cx_set)]
# 每个 body 取置信度最高的 consensus
g = nt_cx.sort_values("celltype_predicted_nt_confidence", ascending=False).drop_duplicates("body")
body_nt = dict(zip(g["body"].tolist(), g["consensus_nt"].tolist()))
log("NT coverage:", len(body_nt), "/", len(cx_set),
    "dist:", pd.Series(body_nt).value_counts().to_dict())

# 流式提取 CX 内部边
reader = pa.ipc.open_file(D + "connectome-weights.feather")
keep = []
for bi in range(reader.num_record_batches):
    df = reader.get_batch(bi).to_pandas()
    m = df["body_pre"].isin(cx_set) & df["body_post"].isin(cx_set)
    if m.any():
        keep.append(df[m][["body_pre", "body_post", "weight"]].to_numpy())
    if bi % 400 == 0:
        log(f"  batch {bi}/{reader.num_record_batches}, kept {sum(len(x) for x in keep):,}")
e = np.concatenate(keep) if keep else np.zeros((0, 3))
log(f"CX internal edges: {len(e):,}, synapses {e[:,2].sum():,}")

np.savez(D + "cx_edges.npz",
         pre=e[:, 0].astype(np.int64), post=e[:, 1].astype(np.int64),
         w=e[:, 2].astype(np.int32))
cx[["bodyId", "type", "instance", "somaSide"]].to_csv(D + "cx_cells.csv", index=False)
pickle.dump(body_nt, open(D + "cx_nt.pkl", "wb"))
log("CX EXTRACT DONE")
