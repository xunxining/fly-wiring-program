#!/usr/bin/env python3
# 62_fig21.py — fig21: 三角闭合 γ 五机制审判 (§21)
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = "/mnt/agents/output/果蝇生成器/"
wc = json.load(open(ROOT + "wedge_ceiling.json"))
T_real = 0.1124

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

# (a) 传递性阶梯: 五机制全败
names = ["v4.1\n基线", "v4.2\n互反", "v6p4\n楔路由", "v7\n同配f_st", "v7b\n同配0.95", "v8\n全量重连"]
vals = [0.0536, 0.0463, 0.0548, 0.0530, 0.0539, 0.0576]
cols = ["#8da0cb", "#8da0cb", "#fc8d62", "#fc8d62", "#fc8d62", "#fc8d62"]
b = ax[0].bar(names, vals, color=cols)
ax[0].axhline(T_real, color="#e41a1c", ls="--", lw=1.5, label=f"真实 {T_real}")
for r, v in zip(b, vals):
    ax[0].text(r.get_x() + r.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=8.5)
ax[0].set_ylabel("全局传递性 T")
ax[0].set_title("(a) 六族闭三角机制 vs 真实 (v8 转换率仅29.7%)")
ax[0].legend(loc="lower right"); ax[0].set_ylim(0, 0.125)

# (b) 2-跳公共邻居天花板诊断
labels = ["单次抽均值", "max-of-8均值", "P(C≥5)", "P(C≥20)"]
rv = [wc["real"]["C1_mean"], wc["real"]["Cmax8_mean"], wc["real"]["P5"], wc["real"]["P20"]]
gv = [wc["v6p4"]["C1_mean"], wc["v6p4"]["Cmax8_mean"], wc["v6p4"]["P5"], wc["v6p4"]["P20"]]
x = np.arange(4); wd = 0.36
ax[1].bar(x - wd / 2, rv, wd, label="真实", color="#e41a1c", alpha=0.85)
ax[1].bar(x + wd / 2, gv, wd, label="生成", color="#377eb8", alpha=0.85)
for xi, (a, c) in enumerate(zip(rv, gv)):
    ax[1].text(xi - wd / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
    ax[1].text(xi + wd / 2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9)
ax[1].set_title("(b) 2-跳可达对公共邻居 C(i,j): 生成仅及真实一半")
ax[1].set_yscale("log"); ax[1].set_ylabel("值 (log)")
ax[1].legend()
ax[1].annotate("真实边上 C 均值≈69 > 真实2-跳抽 32.6\n→ 生物的边超集中于高C对", xy=(0.02, 0.95),
               xycoords="axes fraction", fontsize=8.5, va="top",
               bbox=dict(boxstyle="round", fc="#fff3cd", ec="#ccaa00", lw=0.6))

# (c) 同配剂量-反应: 零弹性
dose_x = [0.363, 0.545, 0.599]
dose_y = [0.0536, 0.0530, 0.0539]
ax[2].plot(dose_x, dose_y, "o-", color="#377eb8", ms=7, lw=2, label="生成 (同配强度递增)")
ax[2].scatter([0.7215], [T_real], color="#e41a1c", s=120, marker="*", zorder=5, label="真实")
for xi, yi, lab in zip(dose_x, dose_y, ["v4.1", "v7", "v7b"]):
    ax[2].annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, -12), fontsize=9)
ax[2].set_xlabel("同 superclass 边占比")
ax[2].set_ylabel("全局传递性 T")
ax[2].set_title("(c) 同配剂量-反应: 零弹性 (sc 尺度被排除)")
ax[2].set_xlim(0.3, 0.8); ax[2].set_ylim(0, 0.125)
ax[2].legend(loc="lower right")

fig.tight_layout()
fig.savefig(ROOT + "figures/fig21_closure.png", dpi=150)
print("SAVED figures/fig21_closure.png")
