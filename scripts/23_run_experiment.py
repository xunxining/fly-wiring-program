#!/usr/bin/env python3
"""Stage 23: 功能对照实验主控
七组:
  arm0 线性感知机(像素直读, 下界)
  arm1 真实 PN->KC 拓扑+权重 (MaleCNS)
  arm2 生成 v1 (均匀实例)
  arm3 生成 v2 (Peters 空间)
  arm4 真实图度保持重连 (剥离特定布线, 保留度分布)
  arm5a 随机稀疏投影 (fan-in 匹配)
  arm5b 随机高斯稠密投影 (经典随机特征, 最强 baseline)
读出: 全部用全连接可训练 KC->10 读出 (隔离表征质量变量)
次级分析: 真实 KC->MBON 稀疏拓扑读出 vs 全连接读出 (在 arm1 上)
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

# ---------- 载入实例表 ----------
types = np.load(D + "type_names.npy", allow_pickle=True)
t2i = {t: i for i, t in enumerate(types)}
sets = pickle.load(open(D + "olf_sets.pkl", "rb"))
pn_names, kc_names, mbon_names = sets["pn"], sets["kc"], sets["mbon"]

ann = pd.read_feather(OUT + "data/body-annotations.feather", columns=["bodyId", "type"])
ann = ann.dropna(subset=["type"])
pn_ids = np.sort(ann.loc[ann["type"].isin(pn_names), "bodyId"].to_numpy())
kc_ids = np.sort(ann.loc[ann["type"].isin(kc_names), "bodyId"].to_numpy())
mbon_ids = np.sort(ann.loc[ann["type"].isin(mbon_names), "bodyId"].to_numpy())
N_PN, N_KC, N_MB = len(pn_ids), len(kc_ids), len(mbon_ids)
log(f"N_PN={N_PN}, N_KC={N_KC}, N_MB={N_MB}")

pn_pos = {b: i for i, b in enumerate(pn_ids)}
kc_pos = {b: i for i, b in enumerate(kc_ids)}
mb_pos = {b: i for i, b in enumerate(mbon_ids)}

# ---------- 真实图矩阵 ----------
def build_mats(edges_npz):
    e = np.load(edges_npz)
    pre, post, w = e["pre"], e["post"], e["w"].astype(np.float32)
    m_pk = np.array([(p in pn_pos) and (q in kc_pos) for p, q in zip(pre, post)])
    m_km = np.array([(p in kc_pos) and (q in mb_pos) for p, q in zip(pre, post)])
    W_pk = np.zeros((N_KC, N_PN), np.float32)
    i = np.where(m_pk)[0]
    for j in i:
        W_pk[kc_pos[post[j]], pn_pos[pre[j]]] += w[j]
    W_km = np.zeros((N_MB, N_KC), np.float32)
    i = np.where(m_km)[0]
    for j in i:
        W_km[mb_pos[post[j]], kc_pos[pre[j]]] += w[j]
    return W_pk, W_km

# ---------- 生成图矩阵 (局部索引 -> 实例顺序与 bodyId 排序一致) ----------
def build_mats_gen(gen_npz):
    e = np.load(gen_npz)
    # 实例局部索引: 需要每个类型的实例排序 -> 与 21 脚本一致 (bodyId 排序)
    ann2 = pd.read_feather(OUT + "data/body-annotations.feather", columns=["bodyId", "type"])
    ann2 = ann2.dropna(subset=["type"])
    inst_order = {}
    for t in np.concatenate([[t2i[x] for x in pn_names],
                             [t2i[x] for x in kc_names],
                             [t2i[x] for x in mbon_names]]):
        ids = np.sort(ann2.loc[ann2["type"].map(t2i) == t, "bodyId"].to_numpy())
        inst_order[int(t)] = ids
    W_pk = np.zeros((N_KC, N_PN), np.float32)
    e_pk = e["e_pk"]
    for row in e_pk:
        s, t = int(row[2]), int(row[3])
        pb = inst_order[s][row[0]]; qb = inst_order[t][row[1]]
        W_pk[kc_pos[qb], pn_pos[pb]] += 1.0
    W_km = np.zeros((N_MB, N_KC), np.float32)
    e_km = e["e_km"]
    for row in e_km:
        s, t = int(row[2]), int(row[3])
        pb = inst_order[s][row[0]]; qb = inst_order[t][row[1]]
        W_km[mb_pos[qb], kc_pos[pb]] += 1.0
    return W_pk, W_km

# ---------- 组装 ----------
log("loading real edges...")
W_pk_real, W_km_real = build_mats(D + "olf/real_edges.npz")
log(f"real: PN->KC nnz={np.count_nonzero(W_pk_real)}, KC->MBON nnz={np.count_nonzero(W_km_real)}")

arms = {}
arms["arm1_real"] = W_pk_real
for tag in ["v1", "v2"]:
    log(f"loading gen_{tag}...")
    W_pk_g, _ = build_mats_gen(D + f"olf_gen_{tag}.npz")
    arms[f"arm2_gen_{tag}" if tag == "v1" else f"arm3_gen_{tag}"] = W_pk_g
    log(f"gen_{tag}: PN->KC nnz={np.count_nonzero(W_pk_g)}")

log("rewiring...")
arms["arm4_rewired"] = core.rewire_deg_preserving(W_pk_real, n_swap=400000, seed=1)
deg_in = (W_pk_real > 0).sum(1)
arms["arm5a_rand_sparse"] = core.make_pn_kc_random(N_PN, N_KC, fanin=int(round(deg_in.mean())), seed=2)
Wg = rng.standard_normal((N_KC, N_PN)).astype(np.float32)
arms["arm5b_rand_gauss"] = Wg

# ---------- 前端: 784 像素 -> PN 活动 (所有组共享) ----------
F = rng.standard_normal((N_PN, 784)).astype(np.float32) / np.sqrt(784)
def frontend(X):
    h = X @ F.T
    return np.maximum(h - h.mean(axis=1, keepdims=True), 0)  # 去均值+ReLU

Xtr, ytr, Xte, yte = core.load_mnist()
Xtr_f, Xte_f = frontend(Xtr), frontend(Xte)

results = {}
# arm0: 线性感知机下界
class LinArm:
    def __init__(s): s.W = np.zeros((10, 784), np.float32)
    def encode(s, X): return X
    def train_sample(s, x, y, lr=0.05):
        sc = s.W @ x; p = int(np.argmax(sc))
        if p != y:
            s.W[y] += lr * x; s.W[p] -= lr * x; return 0
        return 1
    def predict(s, X, bs=2000):
        return np.argmax(X @ s.W.T, axis=1)

log("=== arm0 linear ===")
results["arm0_linear"] = core.fewshot_eval(LinArm(), Xtr, ytr, Xte, yte)

for name, W_pk in arms.items():
    log(f"=== {name} ===")
    Wn = W_pk.copy()
    if Wn.max() > 0:
        Wn = Wn / np.percentile(Wn[Wn > 0], 90)  # 归一化权重尺度
    topo = np.ones((10, N_KC), np.float32)
    ckt = core.OlfactoryCircuit(Wn, topo, n_cls=10, kc_frac=0.05, seed=3)
    results[name] = core.fewshot_eval(ckt, Xtr_f, ytr, Xte_f, yte)

json.dump(results, open(D + "func_results.json", "w"), indent=2)
log("EXPERIMENT DONE")
for k, v in results.items():
    log(k, {s: f"{v[s][0]:.3f}" for s in v})
