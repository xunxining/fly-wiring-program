#!/usr/bin/env python3
# 68_fig22.py — §22 空间距离核三联图
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = "/mnt/agents/output/果蝇生成器/"
real = json.load(open(ROOT + "real_dist.json"))
d9 = json.load(open(ROOT + "dist_v9.json"))
d9d = json.load(open(ROOT + "dist_v9d.json"))
dg = json.load(open(ROOT + "spatial_kernel_diag.json"))

rs = [1, 2, 5, 10, 20, 40, 80, 160]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# A: 累积剖面
ax = axes[0]
bins = np.array(dg["bins"]); Hbg = np.array(dg["Hbg"], float)
cdf_b = np.cumsum(Hbg / Hbg.sum())
ax.plot(bins[1:], cdf_b, "o--", color="gray", ms=4, lw=1.2,
        label=f"配额均匀基线 (中位 26.1µm)")
for dd, nm, c in [(d9, "v9 (仅均匀支)", "#e6a23c"), (d9d, "v9d (+角度支重抽)", "#c0392b")]:
    ax.plot(rs, [dd["profile"]["gen"][str(r)] for r in rs], "s-", color=c, ms=4,
            label=f"{nm} (中位 {dd['gen_median']:.1f}µm)")
ax.plot(rs, [real["profile"][str(r)] for r in rs], "o-", color="black", lw=2,
        label=f"真实 (中位 {real['median']:.1f}µm)")
ax.set_xscale("log"); ax.set_xlabel("soma 距离 r (µm)"); ax.set_ylabel("P(边距离 < r)")
ax.set_title("A. 边距离累积剖面")
ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3)

# B: 中位距离阶梯 + KS
ax = axes[1]
names = ["基线", "v9", "v9c", "v9d", "真实"]
meds = [26.1, d9["gen_median"], 19.3, d9d["gen_median"], real["median"]]
kss = [None, d9["dist_KS"], 0.1163, d9d["dist_KS"], None]
cols = ["gray", "#e6a23c", "#d4883c", "#c0392b", "black"]
bars = ax.bar(names, meds, color=cols, width=0.6)
for b, m, k in zip(bars, meds, kss):
    lab = f"{m:.1f}" + (f"\nKS={k:.3f}" if k is not None else "")
    ax.text(b.get_x() + b.get_width() / 2, m + 0.3, lab, ha="center", fontsize=9)
ax.set_ylabel("边距离中位 (µm)"); ax.set_ylim(0, 30)
ax.set_title("B. 中位距离阶梯 (v9c=WCAP24)")
ax.grid(alpha=0.3, axis="y")

# C: 守卫指标
ax = axes[2]
metrics = ["全局传递性 T", "CX R²", "互惠度", "出度KS"]
v5v = [0.0536, 0.474, 0.0777, 0.0134]
v9dv = [0.0598, 0.576, 0.0986, 0.0110]
realv = [0.1124, 0.617, 0.2997, 0.0]
x = np.arange(len(metrics)); w = 0.27
ax.bar(x - w, v5v, w, label="v5 (无空间核)", color="gray")
ax.bar(x, v9dv, w, label="v9d (空间核)", color="#c0392b")
ax.bar(x + w, realv, w, label="真实", color="black")
for xi, (a, b, c) in enumerate(zip(v5v, v9dv, realv)):
    ax.text(xi - w, a + 0.008, f"{a:.3f}", ha="center", fontsize=7.5)
    ax.text(xi, b + 0.008, f"{b:.3f}", ha="center", fontsize=7.5)
    ax.text(xi + w, c + 0.008, f"{c:.3f}" if c > 0 else "—", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
ax.set_title("C. 守卫指标: 旧轴无损, CX 反升")
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, axis="y")

fig.suptitle("§22 空间距离核: 逐对 w_st(d) 拒绝采样补齐距离轴 (84.2% soma 覆盖)", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT + "figures/fig22_spatial.png", dpi=150)
print("saved figures/fig22_spatial.png")
