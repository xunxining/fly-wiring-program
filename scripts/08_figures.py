#!/usr/bin/env python3
"""生成报告图表"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import json, os

D = "/tmp/flydata/"
OUT = "/mnt/agents/output/果蝇生成器/figures/"
os.makedirs(OUT, exist_ok=True)

pre = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(D + "e_post.i32", dtype=np.int32, mode="r")
w = np.memmap(D + "e_w.i32", dtype=np.int32, mode="r")
node_type = np.load(D + "node_type.npy")
N = len(node_type)
m = (node_type[pre] >= 0) & (node_type[post] >= 0)
gpre = np.memmap("/tmp/flydata/gen/gen_S1_pre.i32", dtype=np.int32, mode="r")
gpost = np.memmap("/tmp/flydata/gen/gen_S1_post.i32", dtype=np.int32, mode="r")
gw = np.memmap("/tmp/flydata/gen/gen_S1_w.i32", dtype=np.int32, mode="r")

# 真实图 typed 子图度数
outdeg = np.bincount(pre[m], minlength=N)
indeg = np.bincount(post[m], minlength=N)
typed = node_type >= 0
g_out = np.bincount(gpre, minlength=int(typed.sum()))
g_in = np.bincount(gpost, minlength=int(typed.sum()))

def ccdf(x):
    v = np.sort(x[x > 0])
    return v, 1 - np.arange(len(v)) / len(v)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (r, g, name) in zip(axes, [(outdeg[typed], g_out, "出度 out-degree"),
                                   (indeg[typed], g_in, "入度 in-degree")]):
    v, c = ccdf(r); ax.loglog(v, c, ".", ms=2, label="真实 MaleCNS", alpha=0.6)
    v, c = ccdf(g); ax.loglog(v, c, ".", ms=2, label="生成器 (S=1)", alpha=0.6)
    ax.set_xlabel("度 k"); ax.set_ylabel("P(K ≥ k)"); ax.set_title(name); ax.legend()
fig.suptitle("度分布：真实连接组 vs 生成网络")
fig.tight_layout(); fig.savefig(OUT + "fig1_degree.png", dpi=150); plt.close(fig)

# 权重分布
fig, ax = plt.subplots(figsize=(5.5, 4.2))
bins = np.logspace(0, np.log10(2600), 40)
ax.hist(np.asarray(w[m]), bins=bins, density=True, alpha=0.6, label="真实 MaleCNS")
ax.hist(np.asarray(gw), bins=bins, density=True, alpha=0.6, label="生成器 (S=1)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("连接权重（突触数）"); ax.set_ylabel("概率密度"); ax.legend()
ax.set_title("突触权重分布")
fig.tight_layout(); fig.savefig(OUT + "fig2_weight.png", dpi=150); plt.close(fig)
print("fig1/2 done", flush=True)

# 高阶结构对比
v1 = {"recip": 0.0828, "cluster": 0.0835}
v2 = {"recip": 0.1295, "cluster": 0.1494}
v3 = json.load(open("/tmp/flydata/gen/validation_v3.json"))
real = {"recip": 0.2998, "cluster": 0.2602}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(2); wd = 0.2
ax.bar(x - 1.5*wd, [real["recip"], real["cluster"]], wd, label="真实 MaleCNS")
ax.bar(x - 0.5*wd, [v1["recip"], v1["cluster"]], wd, label="v1 类型亲和")
ax.bar(x + 0.5*wd, [v2["recip"], v2["cluster"]], wd, label="v2 +空间匹配")
ax.bar(x + 1.5*wd, [v3["recip_gen"], v3["cluster_gen"]], wd, label="v3 +度配额")
ax.set_xticks(x); ax.set_xticklabels(["互惠性 reciprocity", "聚类系数 clustering"])
ax.legend(); ax.set_title("二阶结构（motif 级）：生成器各版本 vs 真实")
fig.tight_layout(); fig.savefig(OUT + "fig3_motifs.png", dpi=150); plt.close(fig)
print("fig3 done", flush=True)

# 低秩结构
d = np.load(D + "typepair.npz")
uk, cnt = d['uk'].astype(np.int64), d['cnt']
type_count = d['type_count'].astype(np.int64)
T = len(type_count)
tp, tq = uk // T, uk % T
possible = type_count[tp] * type_count[tq]
p_st = cnt / possible
logit = np.log(np.clip(p_st, 1e-4, 1-1e-4) / (1 - np.clip(p_st, 1e-4, 1-1e-4)))
A = sp.csr_matrix((logit.astype(np.float32), (tp, tq)), shape=(T, T))
u, s, vt = svds(A, k=64)
s = s[::-1]
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot(np.arange(1, 65), s, "o-", ms=4)
ax.set_xlabel("秩 k"); ax.set_ylabel("奇异值")
ax.set_title("类型对连接矩阵的奇异值衰减\n（低秩 = “化学亲和码”存在的证据，AUC=0.946）")
fig.tight_layout(); fig.savefig(OUT + "fig4_svd.png", dpi=150); plt.close(fig)
print("fig4 done", flush=True)
print("ALL FIGURES DONE")
