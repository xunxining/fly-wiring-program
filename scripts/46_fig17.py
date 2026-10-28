#!/usr/bin/env python3
"""Stage 46: fig13 —— 外潜空间基线 + triad 全谱 (§17)"""
import numpy as np, json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = "/mnt/agents/output/果蝇生成器/"
ol = json.load(open(ROOT + "outer_latent.json"))
tr = json.load(open(ROOT + "triad_real.json"))
tg = json.load(open(ROOT + "triad_v4.json"))

fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})

# ---- (a) 外潜空间基线 ----
models = [("M1 cos环(§12)", "M1_cos_ring(§12)", "#d62728"),
          ("M2 2D欧氏距离", "M2_euclid_2d", "#1f77b4"),
          ("M4 1D线性距离", "M4_linear_1d", "#1f77b4"),
          ("M3 点积K=4", "M3_dotprod_k4_clamped", "#1f77b4"),
          ("M0 类型级", "M0_type_level", "#7f7f7f"),
          ("M5 θ打乱", "M5_theta_scrambled", "#7f7f7f")]
y = np.arange(len(models))[::-1]
for yy, (lab, key, c) in zip(y, models):
    ax[0].barh(yy, ol[key]["auc"], color=c, alpha=0.85, height=0.6)
    ax[0].text(ol[key]["auc"] + 0.004, yy, f'{ol[key]["auc"]:.3f}', va="center", fontsize=9)
ax[0].set_yticks(y, [m[0] for m in models], fontsize=9)
ax[0].set_xlim(0.65, 0.95)
ax[0].set_xlabel("留出 AUC（同一切分, §12 CX 数据）")
ax[0].set_title("(a) 外潜空间基线：cos 环是最优潜几何", fontsize=11)

# ---- (b) triad 全谱 log2 比 ----
order = ["102", "021D", "021U", "021C", "030C", "030T", "012", "111D", "111U",
         "120C", "120D", "120U", "201", "210", "300"]
MC = {"102": 0, "021D": 0, "021U": 0, "021C": 0, "030C": 0, "030T": 0,
      "012": 1, "111D": 1, "111U": 1, "120C": 1, "120D": 1, "120U": 1,
      "201": 2, "210": 2, "300": 3}
cmap = {0: "#7f7f7f", 1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
x = np.arange(len(order))
lr = [np.log2(max(tr["frac"][c], 1e-13) / max(tg["frac"][c], 1e-13)) for c in order]
ax[1].bar(x, lr, color=[cmap[MC[c]] for c in order], alpha=0.88)
for xi, v in zip(x, lr):
    ax[1].text(xi, v + 0.06, f"{v:.1f}", ha="center", fontsize=7.5)
ax[1].set_xticks(x, order, fontsize=8)
ax[1].set_ylabel("log2(真实占比 / v4.1 占比)")
ax[1].set_title("(b) Motif 全谱：亏缺按互边数 k 复合（≈2.8^k）", fontsize=11)
for k, xv in [(0, 2.5), (1, 8.5), (2, 12.5), (3, 14)]:
    ax[1].axvline(xv + 0.5, color="k", lw=0.5, ls=":")
ax[1].text(2.5, 4.3, "k=0 (无互边)", ha="center", fontsize=8.5)
ax[1].text(9, 4.3, "k=1", ha="center", fontsize=8.5)
ax[1].text(12.5, 4.3, "k=2", ha="center", fontsize=8.5)
ax[1].text(14, 4.3, "k=3", ha="center", fontsize=8.5)
ax[1].set_ylim(-0.8, 5.3)
fig.suptitle("图13  防御双实验：潜几何模型类裁决 + 十六类 triad 全谱（§17）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(ROOT + "figures/fig13_defense.png", dpi=150, bbox_inches="tight")
print("fig13 saved")
