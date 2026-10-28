#!/usr/bin/env python3
# fig16: v5 低噪声 CX 放置 — 前后对比
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

v5 = json.load(open("v5_validation.json"))
cxvar = json.load(open("cxvar.json"))
terms41 = json.load(open("cxterms.json"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) CX R2 ladder
ax = axes[0]
names = ["v4.1", "v4.2\n(互反)", "v5\n(低噪声)", "真实"]
vals = [0.474, 0.336, v5["cx_r2_spectral"], 0.617]
cols = ["#d95f02", "#1b9e77", "#7570b3", "#333333"]
bars = ax.bar(names, vals, color=cols, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10)
ax.axhline(0.617, color="#333333", ls=":", lw=1)
ax.set_ylim(0, 0.72)
ax.set_ylabel("CX R²（谱 θ 口径）")
ax.set_title("(a) CX R²：0.474 → 0.585（真实的 95%）", fontsize=11)

# (b) resid variance
ax = axes[1]
names2 = ["v4.1", "v4.2", "v5", "真实"]
resids = [cxvar["v4.1"]["resid_var"], cxvar["v4.2"]["resid_var"],
          v5["cx_true"]["resid_var"], cxvar["real"]["resid_var"]]
bars = ax.bar(names2, resids, color=cols, width=0.55)
for b, v in zip(bars, resids):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontsize=10)
ax.set_ylabel("K 残差方差（真θ/谱θ 自由相位拟合）")
ax.set_title("(b) 噪声地板：87.7 → 46.7（真实 10.4）", fontsize=11)

# (c) per-term
ax = axes[2]
term_names = ["direct", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]
v41_t = [terms41["v4.1"]["direct_EPG_EPG"], terms41["v4.1"]["2hop_via_PEG"],
         terms41["v4.1"]["2hop_via_PEN_a(PEN1)"], terms41["v4.1"]["2hop_via_PEN_b(PEN2)"]]
v5_t = [v5["cx_terms"]["direct"], v5["cx_terms"]["PEG"],
        v5["cx_terms"]["PEN_a(PEN1)"], v5["cx_terms"]["PEN_b(PEN2)"]]
x = np.arange(4)
w = 0.36
ax.bar(x - w / 2, v41_t, w, color="#d95f02", label="v4.1")
ax.bar(x + w / 2, v5_t, w, color="#7570b3", label="v5")
for i in range(4):
    ax.text(x[i] - w / 2, v41_t[i] + 0.012, f"{v41_t[i]:.2f}", ha="center", fontsize=8)
    ax.text(x[i] + w / 2, v5_t[i] + 0.012, f"{v5_t[i]:.2f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(["直接", "经PEG", "经PEN_a", "经PEN_b"], fontsize=9)
ax.set_ylabel("各项 R²（真θ）")
ax.set_title("(c) 全部两跳项同步增强", fontsize=11)
ax.legend(fontsize=9)

fig.suptitle("图16  v5 低噪声实例化：配额化放置兑现 §19 预测", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig20_v5.png", dpi=170)
print("saved fig20_v5.png")
