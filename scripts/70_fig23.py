#!/usr/bin/env python3
# 70_fig23.py — §23 逐细胞配额三联图
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = "/mnt/agents/output/果蝇生成器/"
cv = np.load(ROOT + "cvdump.npz")["cv"]   # (type, [real, v9d, v10c]) 出度CV
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# A: 逐型 CV 散点 (v9d vs v10c 对 y=x)
ax = axes[0]
idx = np.random.default_rng(0).choice(len(cv), 3000, replace=False)
ax.scatter(cv[idx, 0], cv[idx, 1], s=3, alpha=0.25, color="gray", label="v9d (型对配额)")
ax.scatter(cv[idx, 0], cv[idx, 2], s=3, alpha=0.35, color="#c0392b", label="v10c (逐细胞配额)")
ax.plot([0, 0.6], [0, 0.6], "k-", lw=1)
ax.set_xlabel("真实逐型出度 CV"); ax.set_ylabel("生成逐型出度 CV")
ax.set_title("A. 逐型度离散度: 逐细胞配额对上 y=x")
ax.legend(fontsize=9, markerscale=3); ax.grid(alpha=0.3); ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)

# B: 度轴收口阶梯
ax = axes[1]
names = ["v5", "v9d", "v10b", "v10c"]
outks = [0.0134, 0.0110, 0.0000, 0.0000]
inks = [0.0088, 0.0085, 0.0093, 0.0181]
x = np.arange(len(names)); w = 0.35
ax.bar(x - w / 2, outks, w, label="出度 KS", color="#2c3e50")
ax.bar(x + w / 2, inks, w, label="入度 KS", color="#95a5a6")
for xi, (a, b) in enumerate(zip(outks, inks)):
    ax.text(xi - w / 2, a + 0.0004, f"{a:.4f}", ha="center", fontsize=7.5)
    ax.text(xi + w / 2, b + 0.0004, f"{b:.4f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_title("B. 度分布 KS: 出度精确兑现 (0.0000)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# C: γ 轨迹与守卫
ax = axes[2]
Ts = [0.0536, 0.0598, 0.0675, 0.0680]
cxr = [0.474, 0.576, 0.652, 0.694]
ax.plot(names, Ts, "s-", color="#c0392b", lw=2, label="全局传递性 T")
ax.axhline(0.1124, color="#c0392b", ls="--", lw=1, alpha=0.6)
ax.text(0.02, 0.115, "真实 T=0.112", fontsize=8.5, color="#c0392b")
ax.plot(names, cxr, "o-", color="#2c3e50", lw=2, label="CX R²")
ax.axhline(0.617, color="#2c3e50", ls="--", lw=1, alpha=0.6)
ax.text(0.02, 0.60, "真实 CX R²=0.617", fontsize=8.5, color="#2c3e50")
for xi, t_ in zip(x, Ts):
    ax.text(xi, t_ + 0.006, f"{t_:.3f}", ha="center", fontsize=8)
ax.set_ylim(0, 0.8); ax.set_title("C. T 缺口收窄 24%, CX 保真超真实")
ax.legend(fontsize=9, loc="center left"); ax.grid(alpha=0.3)

fig.suptitle("§23 逐细胞度配额 (CELLQ): EPG CV 精确兑现, 度轴收口", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT + "figures/fig23_cellquota.png", dpi=150)
print("saved figures/fig23_cellquota.png")
