#!/usr/bin/env python3
# fig14: v4.2 dyad-repair verdict — triad prediction test + CX cost
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

real = json.load(open("triad_real.json"))
v41 = json.load(open("triad_v4.json"))
v42 = json.load(open("triad_v42.json"))
val = json.load(open("v42_validation.json"))

order = ["003", "012", "102", "021D", "021U", "021C", "030T", "030C",
         "201", "120D", "120U", "120C", "111D", "111U", "210", "300"]
NULLS = {c: int(c[2]) for c in order}
r41 = [real["frac"][c] / v41["frac"][c] for c in order]
r42 = [real["frac"][c] / v42["frac"][c] for c in order]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

# (a) reciprocity
ax = axes[0]
rec = [real["recip"], v41["recip"], v42["recip"]]
bars = ax.bar(["真实", "v4.1", "v4.2"], rec, color=["#333333", "#d95f02", "#1b9e77"], width=0.55)
for b, v in zip(bars, rec):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 0.36)
ax.set_ylabel("互惠度 reciprocity")
ax.set_title("(a) 二元互反修复：配额机制达标", fontsize=11)

# (b) triad ratios
ax = axes[1]
x = np.arange(len(order))
w = 0.38
c41 = ["#7570b3" if NULLS[c] > 0 else "#e7298a" for c in order]
c42 = ["#7570b3" if NULLS[c] > 0 else "#e7298a" for c in order]
ax.bar(x - w / 2, np.log2(r41), w, color=c41, alpha=0.45, label="v4.1（仅单发）")
ax.bar(x + w / 2, np.log2(r42), w, color=c42, alpha=0.95, label="v4.2（互反配额）")
ax.axhline(0, color="k", lw=0.8)
ax.axhline(np.log2(3.0), color="#e7298a", lw=0.9, ls="--")
ax.text(15.4, np.log2(3.0) + 0.08, "×3", color="#e7298a", fontsize=9, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("log2（真实 / 生成）")
ax.set_title("(b) 16 类三元组赤字：开放类塌缩到 ~1，闭合类残留 ~×3", fontsize=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#7570b3", alpha=0.7, label="含空位（开放）"),
                   Patch(color="#e7298a", alpha=0.7, label="无空位（闭合三角）")],
          fontsize=9, loc="upper left")

# (c) cost panel: CX R2 + transitivity
ax = axes[2]
labels = ["CX R²", "全局传递性"]
real_v = [0.617, 0.1124]
v41_v = [0.508, 0.0536]
v42_v = [val["cx_r2_v42"], 0.0463]
xx = np.arange(2)
w = 0.26
ax.bar(xx - w, real_v, w, color="#333333", label="真实")
ax.bar(xx, v41_v, w, color="#d95f02", label="v4.1")
ax.bar(xx + w, v42_v, w, color="#1b9e77", label="v4.2")
for i, (a, b, c) in enumerate(zip(real_v, v41_v, v42_v)):
    for dx, v in [(-w, a), (0, b), (w, c)]:
        ax.text(i + dx, v + 0.012, f"{v:.3f}" if v < 0.2 else f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(xx)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.72)
ax.set_title("(c) 代价：CX 环幅与传递性的权衡", fontsize=11)
ax.legend(fontsize=9)

fig.suptitle("图14  v4.2 互反配额实验：2.8^k 预测的判决", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig14_v42.png", dpi=170)
print("saved fig14_v42.png")
