#!/usr/bin/env python3
"""Stage 24: 消融 —— 真实 PN->KC 二值化拓扑 (剥离突触权重, 只留连接位置)
若 arm1b ≈ arm5a: 真实电路 ≈ 随机稀疏投影, 权重是噪声
若 arm1b ≈ arm1:   拓扑位置本身略逊于均匀随机
"""
import numpy as np
import pandas as pd
import pickle, json, sys
sys.path.insert(0, "/mnt/agents/output/果蝇生成器")
from importlib import import_module
core = import_module("20_func_core")

D = "/tmp/flydata/"
OUT = "/mnt/agents/output/果蝇生成器/"
rng = np.random.default_rng(0)

def log(*a): print(*a, flush=True)

sets = pickle.load(open(D + "olf_sets.pkl", "rb"))
ann = pd.read_feather(OUT + "data/body-annotations.feather", columns=["bodyId", "type"]).dropna(subset=["type"])
pn_ids = np.sort(ann.loc[ann["type"].isin(sets["pn"]), "bodyId"].to_numpy())
kc_ids = np.sort(ann.loc[ann["type"].isin(sets["kc"]), "bodyId"].to_numpy())
N_PN, N_KC = len(pn_ids), len(kc_ids)
pn_pos = {b: i for i, b in enumerate(pn_ids)}
kc_pos = {b: i for i, b in enumerate(kc_ids)}

e = np.load(D + "olf/real_edges.npz")
pre, post = e["pre"], e["post"]
W = np.zeros((N_KC, N_PN), np.float32)
for p, q in zip(pre, post):
    if p in pn_pos and q in kc_pos:
        W[kc_pos[q], pn_pos[p]] = 1.0   # 二值化
log(f"binarized real PN->KC nnz={np.count_nonzero(W)}, mean fan-in={(W.sum(1)>0).mean()*0 + W.sum(1).mean():.2f}")

# KC 入度分布 vs arm5a 的均匀 fan-in=5
deg = W.sum(1)
log(f"KC fan-in: mean={deg.mean():.2f}, std={deg.std():.2f}, min={deg.min()}, max={deg.max()}")

F = rng.standard_normal((N_PN, 784)).astype(np.float32) / np.sqrt(784)
def frontend(X):
    h = X @ F.T
    return np.maximum(h - h.mean(axis=1, keepdims=True), 0)

Xtr, ytr, Xte, yte = core.load_mnist()
Xtr_f, Xte_f = frontend(Xtr), frontend(Xte)

topo = np.ones((10, N_KC), np.float32)
res = {}
log("=== arm1b_real_binary ===")
ckt = core.OlfactoryCircuit(W, topo, n_cls=10, kc_frac=0.05, seed=3)
res["arm1b_real_binary"] = core.fewshot_eval(ckt, Xtr_f, ytr, Xte_f, yte)

# 追加: 度异质性匹配随机 (每 KC 的 fan-in 从真实度序列重采样, 位置随机)
rng2 = np.random.default_rng(11)
Wd = np.zeros((N_KC, N_PN), np.float32)
for k in range(N_KC):
    d = int(deg[k])
    if d > 0:
        Wd[k, rng2.choice(N_PN, min(d, N_PN), replace=False)] = 1.0
log("=== arm5c_rand_degmatched ===")
ckt = core.OlfactoryCircuit(Wd, topo, n_cls=10, kc_frac=0.05, seed=3)
res["arm5c_rand_degmatched"] = core.fewshot_eval(ckt, Xtr_f, ytr, Xte_f, yte)

old = json.load(open(D + "func_results.json"))
old.update(res)
json.dump(old, open(D + "func_results.json", "w"), indent=2)
log("ABLATION DONE")
for k, v in res.items():
    log(k, {s: f"{v[s][0]:.3f}" for s in v})
