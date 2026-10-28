#!/usr/bin/env python3
# 80_fig24.py — 跨性别泛化四联图 (§24)
import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + "data_female/"

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

# A. gamma ladders
ax = axes[0]
stages = ["quota\nuniform", "+CELLQ", "+ACT", "+SPATIAL\n(full)"]
male_T = [0.0536, 0.0675, 0.0680, 0.0680]   # v5, v10b, v10c (v10c=full w/ spatial)
male_full = 0.0680
fem = [json.load(open(D + f"valf_{n}.json")) for n in ["cellq", "cellq_act", "full"]]
fem_T = [0.0770, fem[0]["T_gen"], fem[1]["T_gen"], fem[2]["T_gen"]]
x = np.arange(4)
ax.plot(x, male_T, "o-", color="C0", label="male gen")
ax.axhline(0.1124, color="C0", ls="--", alpha=0.7, label="male real (2.10×)")
ax.plot(x, fem_T, "s-", color="C3", label="female gen")
ax.axhline(0.2678, color="C3", ls="--", alpha=0.7, label="female real (3.48×)")
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=8)
ax.set_title("A. Triadic closure gap γ\nreplicates across sexes", fontsize=10)
ax.set_ylabel("transitivity T"); ax.legend(fontsize=7, loc="center left")
ax.annotate("", xy=(3, 0.1045), xytext=(3, 0.2678), arrowprops=dict(arrowstyle="->", color="C3", alpha=0.4))

# B. CV dispersion ratios
ax = axes[1]
labels = ["uniform", "CELLQ", "CELLQ+ACT", "full"]
m_cv = [0.88, 1.18, 1.70, 1.70]  # v5-ish, v10b, v10c
f_cv = [0.857, 1.001, 1.589, 1.688]
ax.plot(x, m_cv, "o-", color="C0", label="male gen")
ax.plot(x, f_cv, "s-", color="C3", label="female gen")
ax.axhline(1.55, color="C0", ls="--", alpha=0.7, label="male real 1.55×")
ax.axhline(1.457, color="C3", ls="--", alpha=0.7, label="female real 1.46×")
ax.axhline(1.0, color="gray", ls=":", alpha=0.6)
ax.text(0.05, 1.02, "Poisson", fontsize=7, color="gray", transform=ax.get_yaxis_transform())
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_title("B. Degree overdispersion\n(real/Poisson floor ratio)", fontsize=10)
ax.set_ylabel("CV ratio (median over types)"); ax.legend(fontsize=7)

# C. female distance CDF
ax = axes[2]
bins = np.exp(np.linspace(np.log(0.5), np.log(200), 25))
kd = json.load(open(D + "kernel_diag_f.json"))
Hr = np.array(kd["Hr"], float); Hbg = np.array(kd["Hbg"], float)
ax.plot(bins[:-1], np.cumsum(Hr / Hr.sum()), color="black", lw=2, label="female real")
ax.plot(bins[:-1], np.cumsum(Hbg / Hbg.sum()), color="gray", ls="--", label="quota-uniform baseline")
vf = fem[2]["dist"]
# gen full CDF not stored; approximate via real-ish line using median markers
ax.axvline(91.9, color="black", ls=":", alpha=0.5)
ax.axvline(91.5, color="C2", ls=":", alpha=0.8)
ax.text(93, 0.25, "real med 91.9µm\ngen med 91.5µm\n(KS=0.030)", fontsize=8, color="C2")
ax.set_xscale("log"); ax.set_xlabel("edge distance (µm, arbor rep-points)")
ax.set_ylabel("CDF"); ax.set_title("C. Female spatial locality:\n2.14× compression, kernel nails it", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# D. CX ring amplitude cross-sex
ax = axes[3]
cx = json.load(open(D + "cx_latent_f.json"))
fa = np.array(cx["a"]); ma = np.array(cx["male_a"])
fi = [0, 2, 3]; mi = [1, 4, 5]
F = fa[np.ix_(fi, fi)].ravel(); M = ma[np.ix_(mi, mi)].ravel()
names = ["EPG", "PEN_a", "PEN_b"]
lab = [f"{names[i]}→{names[j]}" for i in range(3) for j in range(3)]
ax.scatter(M, F, s=60, color="C3", zorder=3)
for xx, yy, ll in zip(M, F, lab):
    ax.annotate(ll, (xx, yy), fontsize=7, xytext=(4, 4), textcoords="offset points")
lim = [0, 10]
ax.plot(lim, lim, "k--", alpha=0.4)
r_ring = np.corrcoef(F, M)[0, 1]
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("male amplitude a"); ax.set_ylabel("female amplitude a")
ax.set_title(f"D. CX ring wiring-gene amplitudes\nsex-invariant (r = {r_ring:.3f})", fontsize=10)

fig.suptitle("Cross-sex transfer of the connectome generator (male MaleCNS → female FlyWire v783, zero male parameters)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(ROOT + "figures/fig24_cross_sex.png", dpi=170)
print("saved figures/fig24_cross_sex.png")
