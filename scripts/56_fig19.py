#!/usr/bin/env python3
# fig15: CX R² 的信噪比控制律 + 三假设排除
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

cv = json.load(open("cxvar.json"))
tags = ["real", "v4.1", "v4.2"]
labels = {"real": "真实", "v4.1": "v4.1", "v4.2": "v4.2"}
colors = {"real": "#333333", "v4.1": "#d95f02", "v4.2": "#1b9e77"}
amp = [cv[t]["amp"] for t in tags]
resid = [cv[t]["resid_var"] for t in tags]
r2 = [cv[t]["R2"] for t in tags]
snr = [a * a / r for a, r in zip(amp, resid)]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax = axes[0]
x = np.arange(3)
w = 0.38
ax.bar(x - w / 2, amp, w, color=[colors[t] for t in tags], alpha=0.9, label="cos 振幅 amp")
ax2 = ax.twinx()
ax2.bar(x + w / 2, resid, w, color=[colors[t] for t in tags], alpha=0.35, label="残差方差")
for i in range(3):
    ax.text(x[i] - w / 2, amp[i] + 0.3, f"{amp[i]:.1f}", ha="center", fontsize=9)
    ax2.text(x[i] + w / 2, resid[i] + 4, f"{resid[i]:.0f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([labels[t] for t in tags])
ax.set_ylabel("cos 振幅（自由相位拟合）")
ax2.set_ylabel("残差方差")
ax.set_ylim(0, 16)
ax.set_title("(a) 振幅没塌，噪声炸了：残差方差 10.4 → 87.7 → 193.2", fontsize=11)

ax = axes[1]
for i, t in enumerate(tags):
    ax.scatter(snr[i], r2[i], s=180, color=colors[t], zorder=5)
    ax.annotate(f"{labels[t]}\nSNR={snr[i]:.2f}, R²={r2[i]:.3f}",
                (snr[i], r2[i]), textcoords="offset points", xytext=(12, -6), fontsize=10)
ax.plot([0.8, 3.4], [0.30, 0.65], ls="--", color="gray", lw=1, zorder=1)
ax.set_xlabel("信噪比 SNR = amp² / 残差方差")
ax.set_ylabel("CX R²")
ax.set_title("(b) R² 由 SNR 单调控制：差距=实例化噪声，不是环缺失", fontsize=11)

fig.suptitle("图15  CX 退化的真机制：互反注入的不是相位错、不是度地板，是重尾噪声", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig19_cxvar.png", dpi=170)
print("saved fig19_cxvar.png")
