#!/usr/bin/env python3
"""Stage 35: 普查图 fig10 (2x2)
(a) 28x28 通道位置编码强度热图 (edge加权 lift)
(b) 全体类型对 A vs lift 散点, 形态着色 + 锚点高亮
(c) 锚点组条形图
(d) 四个代表对的相位重排矩阵: CX环 / KC同型 / T4视觉 / PN->KC零对照
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"

d = np.load(OUT + "census_pairs.npz")
s, t, E, A, lift = d['s'], d['t'], d['E'], d['A'], d['lift']
zA, hit, ring, block = d['zA'], d['hit'], d['ring'], d['block']
tn = np.load(OUT + "type_names.npy", allow_pickle=True)
tn_str = [str(x) for x in tn]

ch = np.load(OUT + "census_channels.npz", allow_pickle=True)
sc_types = [str(x) for x in ch['sc_types']]
ml = ch['mean_lift']; tested = ch['tested']

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(17, 15))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)
ax = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
      [fig.add_subplot(gs[1, 0]), None]]

# ---- (a) 通道热图 ----
a = ax[0][0]
M = ml.copy()
M[tested == 0] = np.nan
im = a.imshow(M, cmap="RdYlBu_r", vmin=-0.05, vmax=0.35)
short = [x.replace("_tbc", "*").replace("descending", "desc").replace("ascending", "asc")
         .replace("_neuron", "").replace("visual_", "vis_").replace("_intrinsic", "_int")
         .replace("efferent", "eff").replace("sensory", "sens") for x in sc_types]
a.set_xticks(range(28), short, rotation=90, fontsize=6.5)
a.set_yticks(range(28), short, fontsize=6.5)
a.set_xlabel("post superclass"); a.set_ylabel("pre superclass")
a.set_title("(a) 位置编码强度地图: 边加权 AUC lift (灰=无可检类型对)", fontsize=11)
plt.colorbar(im, ax=a, shrink=0.8)

# ---- (b) 散点 ----
a = ax[0][1]
oth = ~(hit | ring | block)
a.scatter(A[oth], lift[oth], s=3, c="#cccccc", alpha=0.4, label=f"未检出 ({oth.sum()})")
a.scatter(A[block], lift[block], s=6, c="#9467bd", alpha=0.6, label=f"块/双侧型 ({block.sum()})")
a.scatter(A[ring], lift[ring], s=6, c="#2ca02c", alpha=0.6, label=f"环/连续坐标型 ({ring.sum()})")
CX = {"EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"}
is_cx = np.array([x in CX for x in tn_str])
is_kc = np.array([x.startswith("KC") for x in tn_str])
is_t45 = np.array([re.fullmatch(r"T[45][a-d]", x) is not None for x in tn_str])
is_pn = np.array([x.endswith("PN") for x in tn_str])
mk_cx = is_cx[s] & is_cx[t]
mk_pk = is_pn[s] & is_kc[t]
mk_kk = is_kc[s] & is_kc[t]
mk_t4 = is_t45[s] | is_t45[t]
a.scatter(A[mk_t4], lift[mk_t4], s=14, facecolors="none", edgecolors="#1f77b4", label="T4/T5 相关")
a.scatter(A[mk_kk], lift[mk_kk], s=14, facecolors="none", edgecolors="#ff7f0e", label="KC→KC")
a.scatter(A[mk_cx], lift[mk_cx], s=30, facecolors="none", edgecolors="red", linewidths=1.5, label="中央复合体")
a.scatter(A[mk_pk], lift[mk_pk], s=30, marker="x", c="black", label="PN→KC (零对照)")
a.axhline(0, color="k", lw=0.5)
a.set_xlabel("圆对齐度 A (裂半出样本)"); a.set_ylabel("AUC lift")
a.set_title("(b) 13,212 个可检类型对: 26% 显著位置编码", fontsize=11)
a.legend(fontsize=8, loc="upper left")

# ---- (c) 锚点条形 ----
a = ax[1][0]
anchors = json_load = __import__("json").load(open(OUT + "census_anchors.json"))["anchors"]
names = [x["name"].split(" (")[0] for x in anchors]
vals = [x.get("wmean_lift", 0) for x in anchors]
hrs = [x.get("hit_rate", 0) for x in anchors]
colors = ["#d62728", "#1f77b4", "#1f77b4", "#444444", "#ff7f0e", "#888888"]
bars = a.bar(range(len(vals)), vals, color=colors, alpha=0.85)
for i, (v, h) in enumerate(zip(vals, hrs)):
    a.text(i, v + 0.008 if v > 0 else 0.008, f"hit={h:.0%}", ha="center", fontsize=9)
a.axhline(0, color="k", lw=0.8)
a.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=9)
a.set_ylabel("边加权 AUC lift")
a.set_title("(c) 三组锚点验证: 符号与文献预期全部一致", fontsize=11)

# ---- (d) 代表对相位重排矩阵 ----
gs_d = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 1],
                                        hspace=0.55, wspace=0.25)
inset = [fig.add_subplot(gs_d[i, j]) for i in range(2) for j in range(2)]
fig.text(0.545, 0.485, "(d) 代表类型对的相位重排连接矩阵（上：视觉/嗅觉；下：CX/KC）", fontsize=11)

node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
pre = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="r")
post = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="r")

def pair_matrix(ts, tt):
    """裂半诚实展示: H1 学相位排序, 只画 H2 的边 (出样本, 过拟合不会产生假条带)"""
    si, ti = tn_str.index(ts), tn_str.index(tt)
    mk = (node_type[pre] == si) & (node_type[post] == ti)
    ei = np.where(mk)[0]
    pu, pinv = np.unique(pre[ei], return_inverse=True)
    qu, qinv = np.unique(post[ei], return_inverse=True)
    rng = np.random.default_rng(1)
    h1 = rng.random(len(ei)) < 0.5
    B1 = np.zeros((len(pu), len(qu)), np.float32)
    B1[pinv[h1], qinv[h1]] = 1.0
    r, c = B1.sum(1), B1.sum(0)
    Bc = B1 - np.outer(r, c) / B1.sum()
    U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
    op = np.argsort(np.arctan2(U[:, 0], U[:, 1]))
    oq = np.argsort(np.arctan2(Vt[0, :], Vt[1, :]))
    B2 = np.zeros((len(pu), len(qu)), np.float32)
    B2[pinv[~h1], qinv[~h1]] = 1.0
    return B2[np.ix_(op, oq)]

CASES = [("PEN_a(PEN1)", "EPG", "CX: PEN_a→EPG  A=0.51 lift=0.44 (环)"),
         ("KCab-m", "KCab-m", "KC: KCab-m→KCab-m  A=0.80 (分层)"),
         ("T5c", "Y11", "视觉: T5c→Y11  A=0.89 (视网膜定位)"),
         ("DA1_lPN", "KCg-m", "嗅觉: DA1_lPN→KCg-m  lift=0.03 (零对照)")]
for axi, (ts, tt, ttl) in zip(inset, CASES):
    try:
        Bs = pair_matrix(ts, tt)
        axi.imshow(Bs, cmap="Greys", aspect="auto", interpolation="nearest")
    except Exception as e:
        axi.text(0.5, 0.5, "无数据", ha="center")
    axi.set_title(ttl, fontsize=8.5)
    axi.set_xticks([]); axi.set_yticks([])
fig.suptitle("图10  全脑位置编码普查: 13,212 类型对 × 28×28 通道", fontsize=14, y=0.98)
fig.savefig(ROOT + "figures/fig10_census.png", dpi=150, bbox_inches="tight")
print("fig saved")
