#!/usr/bin/env python3
"""Stage 21: 生成嗅觉蘑菇体子网络的 v1/v2 版本
v1: 类型对计数 Poisson + 均匀实例分配
v2: Peters 规则 —— post 实例从胞体 kNN(k=4) 中均匀选 (S=1 时位置即真实位置)
输出: /tmp/flydata/olf/gen_v1.npz, gen_v2.npz
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import os

OUT = "/mnt/agents/output/果蝇生成器/"
D = "/tmp/flydata/"
rng = np.random.default_rng(7)

def log(*a): print(*a, flush=True)

types = np.load(D + "type_names.npy", allow_pickle=True)
t2i = {t: i for i, t in enumerate(types)}
T = len(types)

d = np.load(OUT + "基因组_distribs.npz")
uk, cnt = d['uk'].astype(np.int64), d['cnt'].astype(np.int64)
type_count = d['type_count'].astype(np.int64)
tp_all, tq_all = uk // T, uk % T

# ---- 嗅觉类型集合 ----
pn = [t for t in types if 'PN' in t and any(g in t for g in ['_adPN', '_lPN', '_vPN', '_iPN', '_mPN'])]
kc = [t for t in types if t.startswith('KC')]
mbon = [t for t in types if t.startswith('MBON')]
pn_i = np.array([t2i[t] for t in pn])
kc_i = np.array([t2i[t] for t in kc])
mbon_i = np.array([t2i[t] for t in mbon])
log(f"types: PN={len(pn)}, KC={len(kc)}, MBON={len(mbon)}")

# 注释表: 每类型的实例 bodyId 与胞体位置
ann = pd.read_feather(OUT + "data/body-annotations.feather",
                      columns=["bodyId", "type", "somaLocation"])
ann = ann[ann["type"].isin(set(pn) | set(kc) | set(mbon))].reset_index(drop=True)
sl = ann["somaLocation"]
has = sl.notna().to_numpy()
pos = np.full((len(ann), 3), np.nan, dtype=np.float32)
if has.any():
    pos[has] = np.stack(sl[has].to_numpy())
ann["tidx"] = ann["type"].map(t2i)
log(f"annotated cells: {len(ann)}, with soma pos: {has.sum()}")

# 每类型的实例局部索引 (按 bodyId 排序, 确定性)
inst = {}   # tidx -> (bodyId array, pos array)
for t in np.concatenate([pn_i, kc_i, mbon_i]):
    sub = ann[ann["tidx"] == t].sort_values("bodyId")
    inst[t] = (sub["bodyId"].to_numpy(), pos[sub.index.to_numpy()] if False else None)
# 重新按行号对齐 pos
inst = {}
for t in np.concatenate([pn_i, kc_i, mbon_i]):
    sub = ann[ann["tidx"] == t].sort_values("bodyId")
    inst[t] = (sub["bodyId"].to_numpy(), np.stack(sub["somaLocation"].to_numpy()) if sub["somaLocation"].notna().all() else None)

def gen_edges(pairs, peters=False, k_nn=4):
    """pairs: (tp, tq, cnt) 已过滤的类型对; 返回局部实例索引边 (pre_local, post_local, pre_tidx, post_tidx)"""
    out = []
    for s, t, c in pairs:
        n_s = len(inst[s][0]); n_t = len(inst[t][0])
        if n_s == 0 or n_t == 0:
            continue
        ne = rng.poisson(float(c))
        if ne == 0:
            continue
        pre = rng.integers(0, n_s, ne)
        if peters and inst[t][1] is not None and n_t > k_nn:
            pos_t = inst[t][1]
            tree = cKDTree(pos_t)
            # 每个 pre 实例的胞体 -> post 的 kNN
            pos_s = inst[s][1]
            post = np.empty(ne, dtype=np.int64)
            if pos_s is not None:
                _, nb = tree.query(pos_s[pre], k=k_nn)
                post = np.take_along_axis(nb, rng.integers(0, k_nn, ne)[:, None], axis=1).ravel()
            else:
                post = rng.integers(0, n_t, ne)
        else:
            post = rng.integers(0, n_t, ne)
        out.append(np.stack([pre, post,
                             np.full(ne, s), np.full(ne, t)], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 4), np.int64)

# ---- 筛选嗅觉类型对 ----
sel_pk = np.isin(tp_all, pn_i) & np.isin(tq_all, kc_i)
sel_km = np.isin(tp_all, kc_i) & np.isin(tq_all, mbon_i)
pk_pairs = [(tp_all[i], tq_all[i], cnt[i]) for i in np.where(sel_pk)[0]]
km_pairs = [(tp_all[i], tq_all[i], cnt[i]) for i in np.where(sel_km)[0]]
log(f"type pairs: PN->KC={len(pk_pairs)} (real edges {cnt[sel_pk].sum():,}), "
    f"KC->MBON={len(km_pairs)} (real edges {cnt[sel_km].sum():,})")

for tag, peters in [("v1", False), ("v2", True)]:
    e_pk = gen_edges(pk_pairs, peters=peters)
    e_km = gen_edges(km_pairs, peters=peters)
    log(f"gen_{tag}: PN->KC {len(e_pk):,} edges, KC->MBON {len(e_km):,} edges")
    np.savez(D + f"olf_gen_{tag}.npz", e_pk=e_pk, e_km=e_km)

# 保存实例表供后续组装
import pickle
pickle.dump({int(k): (v[0].tolist(),) for k, v in inst.items()},
            open(D + "olf_inst.pkl", "wb"))
np.save(D + "olf_type_sets.npy", np.concatenate([pn_i, kc_i, mbon_i]))
pickle.dump({"pn": pn, "kc": kc, "mbon": mbon}, open(D + "olf_sets.pkl", "wb"))
log("GEN OLFACTORY DONE")
