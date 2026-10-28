#!/usr/bin/env python3
"""Stage 9a: S=10 生成前置 —— 一次性构建经验权重池与类型->superclass 映射
保存 w_emp.npy / type_sc.npy, 供分段生成器直接加载 (避免每个分段重建)
"""
import numpy as np
import pandas as pd
import gc

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"
rng = np.random.default_rng(13)
RES = 256

def log(*a): print(*a, flush=True)

d = np.load(D + "typepair.npz")
type_count = d['type_count'].astype(np.int64)
T = len(type_count)
type_names = np.load(OUT + "type_names.npy", allow_pickle=True)

pre = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(D + "e_post.i32", dtype=np.int32, mode="r")
w = np.memmap(D + "e_w.i32", dtype=np.int32, mode="r")

nodes = pd.read_feather(D + "nodes.feather", columns=["type", "superclass"])
node_type_name = nodes["type"].to_numpy(object)
sc = nodes["superclass"].fillna("UNK").to_numpy()
del nodes; gc.collect()

t2i = {t: i for i, t in enumerate(type_names)}
node_type = np.array([t2i.get(x, -1) for x in node_type_name], dtype=np.int32)
sc_types = sorted(np.unique(sc)); sc2i = {s: i for i, s in enumerate(sc_types)}
NSC = len(sc_types)
node_sc = np.array([sc2i[x] for x in sc], dtype=np.int32)

type_sc = np.zeros(T, dtype=np.int32)
for t in range(T):
    vals, c = np.unique(node_sc[node_type == t], return_counts=True)
    type_sc[t] = vals[np.argmax(c)]
np.save(OUT + "type_sc.npy", type_sc)
log("type_sc saved", NSC, "superclasses")

m_typed = (node_type[pre] >= 0) & (node_type[post] >= 0)
wkey = node_sc[pre[m_typed]].astype(np.int64) * NSC + node_sc[post[m_typed]]
ws = np.asarray(w[m_typed], dtype=np.int32)
w_emp = np.ones((NSC * NSC, RES), dtype=np.int32)
for k in np.unique(wkey):
    vals = ws[wkey == k]
    w_emp[k] = np.resize(vals, RES) if len(vals) < RES else vals[rng.choice(len(vals), RES, replace=False)]
np.save(OUT + "w_emp.npy", w_emp)
log("w_emp saved", w_emp.shape)
log("PREP DONE")
