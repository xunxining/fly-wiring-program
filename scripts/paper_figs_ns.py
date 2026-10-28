"""paper_figs_ns.py — Nature/Science-style figures (overwrite paper_fig*.png).
Style: sans-serif, lowercase bold panel letters, left/bottom spines only,
outward ticks, Okabe-Ito palette, no in-axes titles, minimal grid."""
import json, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + "data_female/"
FIG = ROOT + "figures/"

# ---- Nature-ish rcParams (fonts untouched per env rule) ----
plt.rcParams.update({
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
})

# Okabe-Ito palette
BLU = "#0072B2"; VER = "#D55E00"; GRN = "#009E73"; ORN = "#E69F00"
SKY = "#56B4E9"; GRY = "#7F7F7F"; LGR = "#BBBBBB"; DK = "#333333"

diag = json.load(open(D + "diag_real.json"))
kd   = json.load(open(D + "kernel_diag_f.json"))
vf   = {n: json.load(open(D + f"valf_{n}.json")) for n in ["cellq", "cellq_act", "full"]}
cx   = json.load(open(D + "cx_latent_f.json"))
lp   = pickle.load(open(ROOT + "latent_params.pkl", "rb"))
ol   = json.load(open(ROOT + "outer_latent.json"))  # 面板 b/c 数字从产物现算（勘误同步）
v10c = json.load(open(ROOT + "v10c_validation.json"))
ccv  = json.load(open(ROOT + "cellcv_v10c.json"))
d10  = json.load(open(ROOT + "dist_v10c.json"))

def plabel(ax, s):
    ax.text(-0.20, 1.10, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

def vlab(ax, x, v, fmt="{:.4f}", dy=0, fs=6):
    ax.annotate(fmt.format(v), (x, v), textcoords="offset points",
                xytext=(0, 3 + dy), ha="center", fontsize=fs, color=DK)

# ============================================================
# Fig 1 — schematic (publication-clean)
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

def box(x, y, w, h, title, sub, ec=DK, lw=1.0, tc=DK, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.5,rounding_size=1.0",
                 fc=fill, ec=ec, lw=lw))
    ax.text(x + w/2, y + h - 2.6, title, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=tc)
    if sub:
        ax.text(x + w/2, y + h*0.30, sub, ha="center", va="center",
                fontsize=6.5, color="#555555")

def arr(x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=1.1))

box(1, 26, 17, 16, "Connectome", "typed directed graph\nmale or female", fill="#F2F2F2", ec="#999999")
box(24, 26, 24, 16, "Wiring genome", "type-pair quota table\n+ per-cell quotas (CELLQ)", ec=BLU, tc=BLU)
box(54, 34, 19, 9,  "ACT", "one activity factor per cell")
box(54, 23, 19, 9,  "SPATIAL", "distance kernel w_st(d)")
box(54, 12, 19, 9,  "ANGLE", "CX cosine phase kernel")
box(80, 26, 18, 16, "Quota-exact\nsampler", "rejection sampling", ec=BLU, tc=BLU)
box(80, 6, 18, 11, "Synthetic\nconnectome", "", fill="#F2F2F2", ec="#999999")

arr(18.5, 34, 23.2, 34)
arr(48.8, 38, 53.2, 38.3); arr(48.8, 34, 53.2, 27.5); arr(48.8, 30, 53.2, 16.5)
arr(73.8, 38.3, 79.2, 37); arr(73.8, 27.5, 79.2, 34); arr(73.8, 16.5, 79.2, 31)
arr(89, 25.2, 89, 18)

ax.add_patch(FancyBboxPatch((1, 3), 72, 12, boxstyle="round,pad=0.5,rounding_size=1.0",
                            fc="white", ec=LGR, lw=0.8))
ax.text(37, 11.6, "FALSIFICATION BATTERY", ha="center", fontsize=7.5,
        fontweight="bold", color=VER)
ax.text(37, 6.4, "degree KS | per-type CV vs Poisson floor | edge-distance profile | "
                 "transitivity T | reciprocity | CX kernels | bump dynamics",
        ha="center", fontsize=6.2, color="#555555")
arr(79, 9, 74, 9, color=VER)
ax.text(71.5, 10.6, "mismatch", fontsize=6, color=VER, ha="center")

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig(FIG + "paper_fig1_schematic.png")
plt.close(); print("fig1 ok")

# ============================================================
# Fig 2 — degree determinism
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35))

# a. degree KS
ax = axs[0]
labels = ["male\npre-CELLQ", "male\nv10c", "female\nCELLQ", "female\nfull"]
ks_o = [0.0110, 0.0, 0.0, 0.0]
ks_i = [0.0085, v10c["ks_in"], vf["cellq"]["ks_in"], vf["full"]["ks_in"]]
x = np.arange(4)
ax.bar(x - 0.19, ks_o, 0.38, color=BLU, label="out-degree")
ax.bar(x + 0.19, ks_i, 0.38, color=LGR, label="in-degree")
for xi, v in zip(x, ks_o): vlab(ax, xi - 0.19, v)
for xi, v in zip(x, ks_i): vlab(ax, xi + 0.19, v)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel("KS distance vs real")
ax.set_ylim(0, 0.027); ax.set_yticks([0, 0.01, 0.02])
ax.legend(frameon=False, loc="upper left", handlelength=1.2)
plabel(ax, "a")

# b. dispersion progression
ax = axs[1]
fem_prog = [0.857, 1.001, 1.589, 1.688]
x = np.arange(4)
ax.plot(x, fem_prog, "o-", color=BLU, ms=4, label="female generator")
ax.plot([0, 3], [0.88, ccv["ratio_gen_median"]], "s--", color=VER, ms=4, label="male generator")
ax.axhline(diag["ratio_real_median"], color=BLU, ls=":", lw=0.9)
ax.axhline(ccv["ratio_real_median"], color=VER, ls=":", lw=0.9)
ax.axhline(1.0, color=LGR, lw=0.7)
ax.text(2.35, diag["ratio_real_median"] - 0.09, "female real", fontsize=6, color=BLU, ha="right")
ax.text(1.45, ccv["ratio_real_median"] + 0.03, "male real", fontsize=6, color=VER, ha="right")
ax.text(0.05, 1.035, "Poisson floor", fontsize=6, color="#777777")
ax.set_xticks(x); ax.set_xticklabels(["quota\nuniform", "+CELLQ", "+ACT", "full"], fontsize=6.5)
ax.set_ylabel("median CV / Poisson floor")
ax.set_ylim(0.7, 1.9)
ax.legend(frameon=False, loc="upper left")
plabel(ax, "b")

# c. EPG CV
ax = axs[2]
epg = [("male\n(n = 46)", ccv["EPG"]["out_cv_real"], ccv["EPG"]["out_cv_gen"], 0.0593),
       ("female\n(n = 47)", diag["EPG"]["out_cv"], vf["full"]["EPG"]["out_cv_gen"], diag["EPG"]["poisson_floor"])]
x = np.arange(2)
ax.bar(x - 0.27, [e[1] for e in epg], 0.27, color=DK, label="real")
ax.bar(x,        [e[2] for e in epg], 0.27, color=BLU, label="generator")
ax.bar(x + 0.27, [e[3] for e in epg], 0.27, color=LGR, label="Poisson floor")
for xi, e in zip(x, epg):
    vlab(ax, xi - 0.27, e[1], fs=6); vlab(ax, xi + 0.27, e[3], fs=6)
    ax.annotate("=", (xi, e[2]), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=9, fontweight="bold", color=BLU)
ax.text(0, 0.120, "1.11× floor", fontsize=6.5, color=VER, ha="center")
ax.text(1, 0.120, "1.59× floor", fontsize=6.5, color=VER, ha="center")
ax.set_xticks(x); ax.set_xticklabels([e[0] for e in epg], fontsize=6.5)
ax.set_ylabel("EPG out-degree CV")
ax.set_ylim(0, 0.135)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.00), ncols=3,
          handlelength=1.0, columnspacing=0.8, fontsize=6)
plabel(ax, "c")

plt.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.16, wspace=0.32)
plt.savefig(FIG + "paper_fig2_degree.png")
plt.close(); print("fig2 ok")

# ============================================================
# Fig 3 — space
# ============================================================
bins = np.array(kd["bins"])
Hr = np.array(kd["Hr"], float); Hbg = np.array(kd["Hbg"], float)
cdf_r = np.cumsum(Hr) / Hr.sum(); cdf_b = np.cumsum(Hbg) / Hbg.sum()

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35))

# a. female CDF
ax = axs[0]
ax.plot(bins[1:], cdf_r, color=BLU, lw=1.6, label="real edges")
ax.plot(bins[1:], cdf_b, color=GRY, lw=1.2, ls="--", label="quota-uniform null")
for v, c in [(76.1, BLU), (162.6, GRY), (91.5, VER)]:
    ax.axvline(v, color=c, ls=":", lw=0.8)
ax.text(68, 0.60, "real 76.1", fontsize=6, color=BLU, ha="right")
ax.text(97, 0.33, "gen 91.5", fontsize=6, color=VER, ha="left")
ax.text(170, 0.14, "null 162.6", fontsize=6, color=GRY, ha="left")
ax.set_xscale("log")
ax.set_xlabel("edge distance (µm)")
ax.set_ylabel("cumulative fraction")
ax.set_yticks([0, 0.5, 1.0])
ax.legend(frameon=False, loc="upper left")
plabel(ax, "a")

# b. male quantiles
ax = axs[1]
qs = [0.1, 0.25, 0.5, 0.75, 0.9]
rq = [d10["real_q"][str(q)] for q in qs]
gq = [d10["gen_q"][str(q)] for q in qs]
ax.plot(qs, rq, "o-", color=DK, ms=4, label="real (soma)")
ax.plot(qs, gq, "s--", color=BLU, ms=4, label="generator")
ax.axhline(26.1, color=LGR, ls=":", lw=0.8)
ax.text(0.52, 27.2, "null median 26.1", fontsize=6, color="#777777")
ax.set_xlabel("CDF quantile")
ax.set_ylabel("edge distance (µm)")
ax.set_xticks(qs)
ax.legend(frameon=False, loc="lower right")
plabel(ax, "b")

# c. compression
ax = axs[2]
comp = [("male", 26.1, 15.9, 16.7), ("female", 162.6, 76.1, 91.5)]
x = np.arange(2)
ax.bar(x - 0.27, [c[1] for c in comp], 0.27, color=LGR, label="quota-uniform null")
ax.bar(x,        [c[2] for c in comp], 0.27, color=DK, label="real")
ax.bar(x + 0.27, [c[3] for c in comp], 0.27, color=BLU, label="generator")
for xi, c in zip(x, comp):
    ax.text(xi, c[2] * 1.12, f"{c[1]/c[2]:.2f}×", ha="center", fontsize=7.5,
            color=VER, fontweight="bold")
ax.set_yscale("log"); ax.set_ylim(8, 400)
ax.set_yticks([10, 30, 100, 300]); ax.set_yticklabels(["10", "30", "100", "300"])
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in comp], fontsize=7)
ax.set_ylabel("median edge distance (µm)")
ax.legend(frameon=False, loc="upper left")
plabel(ax, "c")

plt.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.14, wspace=0.32)
plt.savefig(FIG + "paper_fig3_space.png")
plt.close(); print("fig3 ok")

# ============================================================
# Fig 4 — ring-attractor wiring gene
# ============================================================
am = np.asarray(lp["a"]); af = np.asarray(cx["a"])
mtypes = cx["male_types"]; ftypes = cx["types"]
name_map = {"EPG": "EPG", "PEN_a/PEN1": "PEN_a(PEN1)", "PEN_b/PEN2": "PEN_b(PEN2)"}
ring_f = ["EPG", "PEN_a/PEN1", "PEN_b/PEN2"]; ring_m = [name_map[t] for t in ring_f]
fi = [ftypes.index(t) for t in ring_f]; mi = [mtypes.index(t) for t in ring_m]
Fa = af[np.ix_(fi, fi)]; Ma = am[np.ix_(mi, mi)]
fv = Fa.ravel(); mv = Ma.ravel()
r_ring = np.corrcoef(fv, mv)[0, 1]

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35))

# a. cross-sex scatter
ax = axs[0]
ax.plot([-1, 9.5], [-1, 9.5], color=LGR, lw=0.8, zorder=1)
ax.scatter(mv, fv, s=22, color=BLU, zorder=3, label="ring block (3×3)")
lab = {0: ("EPG>EPG", (7, -3)), 2: ("EPG>PENb", (-2, 8)), 6: ("PENb>EPG", (8, -2))}
for k, (t, off) in lab.items():
    ax.annotate(t, (mv[k], fv[k]), textcoords="offset points", xytext=off,
                fontsize=5.5, color=BLU)
d7f = [af[ftypes.index("Delta7"), ftypes.index("EPG")], af[ftypes.index("EPG"), ftypes.index("Delta7")]]
d7m = [am[mtypes.index("Delta7"), mtypes.index("EPG")], am[mtypes.index("EPG"), mtypes.index("Delta7")]]
ax.scatter(d7m, d7f, s=30, marker="D", color=VER, zorder=3, label="Δ7↔EPG")
ax.annotate("D7>EPG", (d7m[1], d7f[1]), textcoords="offset points", xytext=(-2, 7),
            fontsize=5.5, color=VER)
ax.text(0.97, 0.06, f"r = {r_ring:.3f}", transform=ax.transAxes, ha="right",
        fontsize=8, color=DK)
ax.set_xlabel("male amplitude a")
ax.set_ylabel("female amplitude a")
ax.legend(frameon=False, loc="upper left")
plabel(ax, "a")

# b. held-out AUC
ax = axs[1]
x = np.arange(2)
auc_l = [ol["M1_cos_ring(§12)"]["auc"], cx["auc_latent"]]; auc_b = [ol["M0_type_level"]["auc"], cx["auc_type_baseline"]]
ax.bar(x - 0.19, auc_l, 0.38, color=BLU, label="latent-angle model")
ax.bar(x + 0.19, auc_b, 0.38, color=LGR, label="type-level null")
for xi, v in zip(x, auc_l): vlab(ax, xi - 0.19, v, "{:.3f}")
for xi, v in zip(x, auc_b): vlab(ax, xi + 0.19, v, "{:.3f}")
ax.axhline(0.5, color=LGR, lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(["male", "female"], fontsize=7)
ax.set_ylabel("held-out AUC")
ax.set_ylim(0.4, 1.02); ax.set_yticks([0.5, 0.7, 0.9])
ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.22), handlelength=1.2)
plabel(ax, "b")

# c. model selection (male)
ax = axs[2]
models = ["ring\ncos 1D", "Euclid\n2D", "linear\n1D", "inner\nproduct", "scrambled\nθ"]
aucs = [ol["M1_cos_ring(§12)"]["auc"], ol["M2_euclid_2d"]["auc"], ol["M4_linear_1d"]["auc"], ol["M3_dotprod_k4"]["auc"], ol["M5_theta_scrambled"]["auc"]]
cols = [BLU, SKY, SKY, SKY, LGR]
ax.bar(range(5), aucs, 0.62, color=cols)
ax.axhline(ol["M0_type_level"]["auc"], color=GRY, ls=":", lw=0.8)
ax.set_xlim(-0.6, 5.2)
ax.text(4.62, 0.802, "type null", fontsize=6, color=GRY, ha="center")
for i, v in enumerate(aucs):
    ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=6)
ax.set_xticks(range(5)); ax.set_xticklabels(models, fontsize=6)
ax.set_ylabel("held-out AUC (male)")
ax.set_ylim(0.68, 0.95); ax.set_yticks([0.7, 0.8, 0.9])
plabel(ax, "c")

plt.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.17, wspace=0.32)
plt.savefig(FIG + "paper_fig4_ring.png")
plt.close(); print("fig4 ok")

# ============================================================
# Fig 5 — the gamma excess and its resolution (lattice quota-fill)
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.45, 1, 1]})

# a. ladders
ax = axs[0]
mal_steps = [("quota\nunif.", 0.0536), ("+SPAT", 0.0598), ("full\nstack", 0.0680)]
fem_steps = [("quota\nunif.", 0.0770), ("+CQ", 0.0831), ("+ACT", 0.0855), ("full\nstack", 0.1045),
             ("+LATTICE", 0.2396)]
xm = np.arange(3); xf = np.arange(5) + 4
ax.bar(xm, [s[1] for s in mal_steps], 0.62, color="#E5B8B0", label="male generator")
ax.bar(xf[:-1], [s[1] for s in fem_steps[:-1]], 0.62, color="#A6C8E0", label="female generator")
ax.bar(xf[-1], fem_steps[-1][1], 0.62, color="#4E85B6", label="female + lattice")
ax.axhline(0.1124, color=VER, lw=1.3)
ax.axhline(0.2678, color=BLU, lw=1.3)
ax.text(1.4, 0.122, "male real T = 0.112\n(2.10× null)", fontsize=6, color=VER, ha="center", va="bottom")
ax.text(4.55, 0.262, "female real T = 0.268\n(3.48× null)", fontsize=6, color=BLU, ha="center", va="top")
for xi, s in zip(xm, mal_steps): vlab(ax, xi, s[1])
for xi, s in zip(xf, fem_steps): vlab(ax, xi, s[1])
ax.set_xticks(list(xm) + list(xf))
ax.set_xticklabels([s[0] for s in mal_steps] + [s[0] for s in fem_steps], fontsize=5.6, rotation=38, ha="right")
ax.set_ylabel("global transitivity T")
ax.set_ylim(0, 0.31); ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.62), fontsize=6)
plabel(ax, "a")

# b. gap closure by mechanism
ax = axs[1]
mech = ["lattice quota-fill\n(female; confirmed)",
        "γ acceptance\nweighting", "reciprocity-first\nrouting", "motif-directed\nplacement",
        "homophily dose-\nresponse", "supertype quota\nrelaxation", "neuropil-overlap\ngate + weighting",
        "spatial kernel\n(male)", "per-cell quotas\n+ activity (male)", "full stack\n(male)",
        "full stack\n+ gate (female)"]
gap_m = 0.1124 - 0.0536; gap_f = 0.2678 - 0.0770
closed = [100 * (0.2396 - 0.0770) / gap_f,
          0.0, 0.0, 0.0, 0.0, 0.0,
          100 * (0.1123 - 0.1045) / gap_f,
          100 * (0.0598 - 0.0536) / gap_m, 100 * (0.0680 - 0.0598) / gap_m,
          100 * (0.0680 - 0.0536) / gap_m, 100 * (0.1123 - 0.0770) / gap_f]
cols = ["#4E85B6"] + [LGR] * 5 + [LGR, "#E5B8B0", "#E5B8B0", "#D99085", "#A6C8E0"]
y = np.arange(len(mech))[::-1]
ax.barh(y, closed, 0.62, color=cols)
for yi, v in zip(y, closed):
    if v > 0.5:
        ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=6)
ax.set_yticks(y); ax.set_yticklabels(mech, fontsize=5.3)
ax.set_xlabel("% of triadic-closure gap closed")
ax.set_xlim(0, 100)
plabel(ax, "b")

# c. closure-by-overlap law: real vs generator
ax = axs[2]
ovlab = ["0", "1", "2", "3–4", "5–7", "8–14", "15+"]
xx = np.arange(7)
fem_real = [0.000, 0.263, 0.300, 0.287, 0.349, 0.488, 0.691]
fem_gen  = [0.003, 0.083, 0.103, 0.151, 0.282, 0.392, 0.544]
fem_lat  = [0.017, 0.259, 0.284, 0.260, 0.295, 0.415, 0.569]
mal_real = [0.019, 0.207, 0.265, 0.304, 0.333, 0.464, 0.636]
ax.plot(xx, fem_real, "-o", color=BLU, ms=3.5, lw=1.3, label="female real")
ax.plot(xx, mal_real, "-^", color=VER, ms=3.5, lw=1.1, label="male real")
ax.plot(xx, fem_gen, "--s", color="#A6C8E0", ms=3.2, lw=1.1, label="female full stack + gate")
ax.plot(xx, fem_lat, "-D", color="#4E85B6", ms=3.0, lw=1.3, label="female + lattice quota-fill")
ax.set_xticks(xx); ax.set_xticklabels(ovlab, fontsize=6)
ax.set_xlabel("shared neuropils (wedge endpoints)", fontsize=6)
ax.set_ylabel("P(wedge closes)")
ax.set_ylim(-0.03, 0.78)
ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.6)
plabel(ax, "c")

plt.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.17, wspace=0.48)
plt.savefig(FIG + "paper_fig5_gamma.png")
plt.close(); print("fig5 ok")
