#!/usr/bin/env python3
"""Stage 41: fig11 v4 三联图
(a) CX 等效核: 真实 vs v4.1 (环相位重排热图)
(b) 关键指标对照: real / v3 / v4.1
(c) 度数分布 CCDF: real vs v4.1 (S=1)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP","DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
GEN = "/tmp/flydata/gen/genv4_S1"
T = 11751

tn = [str(x) for x in np.load(OUT + "type_names.npy", allow_pickle=True)]
CX = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]
cx_t = [tn.index(x) for x in CX]

def cx_kernel(pre, post, typ, N):
    sel = np.isin(typ[pre], cx_t) & np.isin(typ[post], cx_t)
    cp, cq = np.asarray(pre[sel]), np.asarray(post[sel])
    cells = np.unique(np.r_[cp, cq])
    bidx = {b: i for i, b in enumerate(cells)}
    ctname = np.array([tn[typ[b]] for b in cells])
    W = np.zeros((len(cells), len(cells)), np.float32)
    for p, q in zip(cp, cq):
        W[bidx[q], bidx[p]] += 1
    epg = np.where(ctname == "EPG")[0]
    K = W[np.ix_(epg, epg)].copy()
    for x in ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]:
        mid = np.where(ctname == x)[0]
        if len(mid):
            K += W[np.ix_(epg, mid)] @ W[mid][:, epg]
    S_ = np.abs(K) + np.abs(K).T
    A_ = S_ / (S_.sum(1, keepdims=True) + 1e-9)
    ev, evec = np.linalg.eigh(A_)
    th = np.arctan2(evec[:, -2], evec[:, -3])
    o = np.argsort(th)
    return K[np.ix_(o, o)]

node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
rpre = np.memmap(OUT + "e_pre.i32", dtype=np.int32, mode="r")
rpost = np.memmap(OUT + "e_post.i32", dtype=np.int32, mode="r")
gpre = np.memmap(GEN + "_pre.i32", dtype=np.int32, mode="r")
gpost = np.memmap(GEN + "_post.i32", dtype=np.int32, mode="r")
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
gen_type = np.repeat(np.arange(T, dtype=np.int32), g['type_count'].astype(np.int64))

Kr = cx_kernel(rpre, rpost, node_type, len(node_type))
Kg = cx_kernel(gpre, gpost, gen_type, len(gen_type))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
im0 = ax[0].imshow(np.log1p(Kr), cmap="magma")
ax[0].set_title("(a) 真实 CX 等效核 (log, 环相位重排)", fontsize=10)
plt.colorbar(im0, ax=ax[0], shrink=0.8)
im1 = ax[1].imshow(np.log1p(Kg), cmap="magma")
ax[1].set_title("(a') v4.1 生成 CX 等效核 —— 环结构重建 (R²=0.50)", fontsize=10)
plt.colorbar(im1, ax=ax[1], shrink=0.8)

# (b) 指标 (CX R² 采用 §16 分解后的诚实同口径: 连接计数二值核)
labels = ["入度 KS↓", "CX 核 R²\n(连接计数口径)", "互惠性", "传递性"]
real_v = [0.0, 0.617, 0.300, 0.256]
v4_v = [0.0088, 0.508, 0.0885, 0.0798]
x = np.arange(len(labels))
ax[2].bar(x - 0.2, real_v, 0.36, label="真实", color="#2ca02c")
ax[2].bar(x + 0.2, v4_v, 0.36, label="v4.1 生成", color="#1f77b4")
for i, (r, v) in enumerate(zip(real_v, v4_v)):
    ax[2].text(i - 0.2, r + 0.012, f"{r:.2f}" if r else "—", ha="center", fontsize=8)
    ax[2].text(i + 0.2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
ax[2].annotate("CX R² 旧口径(真实带符号突触权重)=0.696,\n同尺度二值口径真实上限=0.617(§16 分解)",
               xy=(1, 0.62), xytext=(1.45, 0.72), fontsize=7.5,
               arrowprops=dict(arrowstyle="->", lw=0.7))
ax[2].set_xticks(x, labels)
ax[2].set_title("(b) v4.1 vs 真实: 边际/功能/基序", fontsize=10)
ax[2].legend()
ax[2].set_ylim(0, 0.82)
fig.suptitle("图11  v4.1 生成器验证: 均匀基线 + 普查核 + CX 联合盒", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(ROOT + "figures/fig11_v4.png", dpi=150, bbox_inches="tight")
print("fig saved")
