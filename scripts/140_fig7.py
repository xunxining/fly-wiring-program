#!/usr/bin/env python3
"""140_fig7.py — Fig 7: cross-sex conservation of the columnar assembly kernel + P9b circularity control."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = "/mnt/agents/output/果蝇生成器/"
FIG = ROOT + "figures/"
ACC = "#1f4e79"; GRY = "#555555"; RED = "#b03a2e"; LGT = "#7aa5c4"; GRN = "#2e7d32"

kc4 = json.load(open(ROOT + "data_male/kernel_conserve4_m2.json"))
hemi = json.load(open(ROOT + "data_male/kernel_hemi_m2.json"))
heat = np.load(ROOT + "data_male/fig7_heat.npz", allow_pickle=True)
nulls = np.load(ROOT + "data_male/fig7_nulls.npz")

fig = plt.figure(figsize=(11.5, 7.8))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.22,
              left=0.055, right=0.985, top=0.93, bottom=0.075)

# ---------- A: 代表对 lift 热图 ----------
axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
names = [str(x) for x in heat["show_names"]]
sgs = gs[0, 0].subgridspec(2, 4, hspace=0.06, wspace=0.10)
sub = [fig.add_subplot(sgs[i, j]) for i in range(2) for j in range(4)]
vmax = max(float(np.abs(heat[f"heat_{k}_m"]).max()) for k in range(4)) * 0.9
for k in range(4):
    nm_, nf_ = heat[f"heat_{k}_info"]
    pk = heat[f"heat_{k}_pk"]
    for row, tag, ttl in ((0, "m", "male"), (1, "f", "female (harmonized)")):
        ax = sub[row*4 + k]
        img = heat[f"heat_{k}_{tag}"]
        ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
                  extent=[-10.5, 10.5, -10.5, 10.5])
        cx = pk[0] if tag == "m" else pk[2]
        cy = pk[1] if tag == "m" else pk[3]
        ax.plot(cx, cy, marker="o", mfc="none", mec="black", ms=9, mew=1.4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.5)
        if row == 0:
            ax.set_title(names[k].replace("|", "\u2192"), fontsize=8.5)
        if k == 0:
            ax.set_ylabel(ttl, fontsize=7.5)
axA.set_title("a", loc="left", fontsize=12, fontweight="bold", x=-0.06)

# ---------- B: 12 对称一致率 (面板1 全局 + 逐半球) ----------
axB = fig.add_subplot(gs[0, 1])
x = np.arange(12)
p1g = kc4["panel1_selection_set"]["acc_by_symmetry"]
p1R = hemi["panel1_gold_R"]["acc_by_symmetry"]
p1L = hemi["panel1_gold_L"]["acc_by_symmetry"]
w = 0.27
axB.bar(x - w, p1g, w, color=GRY, alpha=0.8, label="pooled (selection panel)")
axB.bar(x, p1R, w, color=ACC, label="right-eye pairs only")
axB.bar(x + w, p1L, w, color=GRN, label="left-eye pairs only")
for xi, v, c, dy in ((3, p1g[3], GRY, 0.04), (6, p1R[6], ACC, 0.04), (3, p1L[3], GRN, 0.10)):
    axB.annotate(f"{v:.2f}", (xi + (0.27 if c == GRN else (-0.27 if c == GRY else 0)), v + dy),
                 ha="center", fontsize=8, fontweight="bold", color=c)
axB.axvspan(5.5, 11.5, color=RED, alpha=0.05)
axB.text(8.5, -0.155, "reflections", fontsize=7.5, color=RED, ha="center")
axB.text(2.5, -0.155, "rotations", fontsize=7.5, color=GRY, ha="center")
axB.set_xticks(x); axB.set_xticklabels([f"#{i}" for i in x], fontsize=7.5)
axB.set_ylim(0, 1.18); axB.set_ylabel("peak agreement (Chebyshev ≤ 1)", fontsize=8.5)
axB.set_xlabel("hex-lattice symmetry", fontsize=8.5)
axB.legend(fontsize=7.2, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
axB.set_title("b", loc="left", fontsize=12, fontweight="bold", x=-0.10)

# ---------- C: 面板2 零分布 vs 观测 ----------
axC = fig.add_subplot(gs[1, 0])
nb = nulls["null_best_p2"]
axC.hist(nb, bins=28, color=LGT, alpha=0.75, edgecolor="white", linewidth=0.4,
         label="shuffle × best-of-12 null (300 reps)")
p99 = np.percentile(nb, 99)
acc = kc4["panel2_validation"]["acc"]; accid = kc4["panel2_validation"]["acc_identity"]
axC.axvline(acc, color=RED, lw=1.8, label=f"observed, R180-harmonized = {acc:.3f}")
axC.axvline(accid, color=GRY, lw=1.4, ls="--", label=f"identity (no harmonization) = {accid:.3f}")
axC.axvline(p99, color=ACC, lw=1.2, ls=":", label=f"null p99 = {p99:.3f}")
axC.set_xlabel("peak agreement on validation panel", fontsize=8.5)
axC.set_ylabel("replicates", fontsize=8.5)
axC.legend(fontsize=7.5, frameon=False, loc="upper right")
axC.set_title("c", loc="left", fontsize=12, fontweight="bold", x=-0.06)

# ---------- D: P9b 循环性对照 ----------
axD = fig.add_subplot(gs[1, 1])
xx = np.arange(2); w2 = 0.34
Tb = [0.182, 0.137]; Ob = [1.29, 2.07]
axD.bar(xx - w2/2, [0.256, 0.256], w2, color=GRY, alpha=0.85, label="real transitivity")
axD.bar(xx + w2/2, Tb, w2, color=ACC, label="generated T")
for xi, v in zip(xx + w2/2, Tb):
    axD.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold", color=ACC)
axD.text(0 - w2/2, 0.262, "0.256", ha="center", fontsize=8, color="white", fontweight="bold")
axD.text(1 - w2/2, 0.262, "0.256", ha="center", fontsize=8, color="white", fontweight="bold")
axD.set_xticks(xx)
axD.set_xticklabels(["old lattice coords\n(harmonic, connection-derived)", "m2 soma coords\n(connection-exogenous)"], fontsize=8)
axD.set_ylim(0, 0.30); axD.set_ylabel("transitivity (node-uniform)", fontsize=8.5)
axD2 = axD.twinx()
axD2.plot(xx, Ob, "s-", color=RED, ms=6, lw=1.6, label="optic–optic closure gap (real/gen)")
axD2.annotate("1.29×", (-0.02, 1.40), fontsize=8.5, color=RED, fontweight="bold")
axD2.annotate("2.07×", (0.86, 1.88), fontsize=8.5, color=RED, fontweight="bold")
axD2.axhline(1.15, color=RED, lw=0.9, ls=":", alpha=0.7)
axD2.text(0.28, 1.10, "registered gate 1.15×", fontsize=7, color=RED)
axD2.set_ylim(1.0, 2.4); axD2.set_ylabel("optic closure deficit ratio", fontsize=8.5, color=RED)
axD2.tick_params(axis="y", colors=RED)
h1, l1 = axD.get_legend_handles_labels(); h2, l2 = axD2.get_legend_handles_labels()
axD.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))
axD.set_title("d", loc="left", fontsize=12, fontweight="bold", x=-0.10)

fig.suptitle("Assembly-kernel conservation across sexes, and the circularity control", fontsize=13, y=0.985)
fig.savefig(FIG + "paper_fig7_conserve.png", dpi=200)
print("saved", FIG + "paper_fig7_conserve.png")
