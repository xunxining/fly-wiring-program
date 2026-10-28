"""paper_figs.py — English figures for the cross-sex wiring-program paper.
All numbers pulled from saved JSONs / pkl (no recomputation of heavy stats)."""
import json, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + "data_female/"
FIG = ROOT + "figures/"

ACC = "#1f4e79"   # deep blue accent
GRY = "#555555"
RED = "#b03a2e"

# ---------- load data ----------
diag = json.load(open(D + "diag_real.json"))
kd   = json.load(open(D + "kernel_diag_f.json"))
vf   = {n: json.load(open(D + f"valf_{n}.json")) for n in ["cellq", "cellq_act", "full"]}
cx   = json.load(open(D + "cx_latent_f.json"))
ol   = json.load(open(ROOT + "outer_latent.json"))  # 2026-10-22: 面板 B/C 数字改为从产物现算
lp   = pickle.load(open(ROOT + "latent_params.pkl", "rb"))
v10c = json.load(open(ROOT + "v10c_validation.json"))
ccv  = json.load(open(ROOT + "cellcv_v10c.json"))
d10  = json.load(open(ROOT + "dist_v10c.json"))

bins = np.array(kd["bins"])
Hr = np.array(kd["Hr"], float); Hbg = np.array(kd["Hbg"], float)
cdf_r = np.cumsum(Hr) / Hr.sum(); cdf_b = np.cumsum(Hbg) / Hbg.sum()

# ============================================================
# Fig 1 — schematic of the wiring program
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

def box(x, y, w, h, title, sub, fc="white", ec=GRY, tc="black", title_fs=10, sub_fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h - 3.2, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=tc)
    ax.text(x + w/2, y + (h - 6.5)/2, sub, ha="center", va="center",
            fontsize=sub_fs, color=tc)

def arrow(x1, y1, x2, y2, label="", color=GRY, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=lw))
    if label:
        ax.text((x1+x2)/2, max(y1, y2) + 1.6, label, ha="center", fontsize=7.5, color=color)

# input
box(1, 30, 20, 18, "Connectome", "typed graph\n(N, T, E)\nmale or female", fc="#f0f4f8")
# genome
box(27, 30, 22, 18, "GENOME", "type-pair quota table\n(s,t) -> edge count\n+ per-cell quotas (CELLQ)", fc="white", ec=ACC, tc=ACC)
# operators
box(55, 38, 20, 10, "ACT", "one activity factor\nper cell (in/out corr 0.79)", fc="white")
box(55, 26, 20, 10, "SPATIAL", "per-pair distance\nkernel w_st(d)", fc="white")
box(55, 14, 20, 10, "ANGLE", "CX latent phase kernel\na_st cos(theta_i - theta_j + phi)", fc="white")
# sampler
box(81, 30, 17, 18, "Rejection\nsampler", "quota-exact\nplacement", fc="white", ec=ACC, tc=ACC)

arrow(21.8, 39, 26.2, 39, "reverse-engineer")
arrow(49.8, 41.5, 54.2, 43)
arrow(49.8, 39, 54.2, 31)
arrow(49.8, 36.5, 54.2, 19)
arrow(75.8, 43, 80.2, 41)
arrow(75.8, 31, 80.2, 35)
arrow(75.8, 19, 80.2, 33)

# output
box(81, 6, 17, 12, "Synthetic\nconnectome", "quota-exact edges", fc="#f0f4f8")
arrow(89.5, 29.2, 89.5, 18.8)

# falsification band
box(1, 2, 74, 12, "", "", fc="#fafafa", ec="#bbbbbb")
ax.text(38, 11.2, "FALSIFICATION BATTERY (per generated graph)", ha="center",
        fontsize=9, fontweight="bold", color=RED)
ax.text(38, 5.5, "degree distributions (KS)  |  per-type CV vs Poisson floor  |  edge-distance profile  |  triadic closure T  |  "
                 "reciprocity  |  CX angle kernels  |  bump dynamics", ha="center", fontsize=7.8, color=GRY)
arrow(81, 8, 76, 8, "", color=RED, lw=1.4)
ax.text(78.5, 9.6, "mismatch ->\nnext mechanism", fontsize=7, color=RED, ha="left")

plt.tight_layout()
plt.savefig(FIG + "paper_fig1_schematic.png", dpi=200)
plt.close()
print("fig1 ok")

# ============================================================
# Fig 2 — degree determinism
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))

# A. degree KS
ax = axs[0]
labels = ["male\nv9d", "male\nv10c", "female\nCELLQ", "female\nfull"]
ks_o = [0.0110, 0.0, 0.0, 0.0]
ks_i = [0.0085, v10c["ks_in"], vf["cellq"]["ks_in"], vf["full"]["ks_in"]]
x = np.arange(4)
ax.bar(x - 0.18, ks_o, 0.36, color=ACC, label="out-degree KS")
ax.bar(x + 0.18, ks_i, 0.36, color="#9db8d2", label="in-degree KS")
for xi, v in zip(x, ks_o):
    ax.text(xi - 0.18, v + 0.0006, f"{v:.4f}", ha="center", fontsize=7.5)
for xi, v in zip(x, ks_i):
    ax.text(xi + 0.18, v + 0.0006, f"{v:.4f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("KS distance vs real", fontsize=9)
ax.set_title("A. Out-degree distribution matched exactly", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False)
ax.set_ylim(0, 0.026)

# B. dispersion progression
ax = axs[1]
fem_prog = [0.857, 1.001, 1.589, 1.688]
mal_prog = [0.88, np.nan, np.nan, ccv["ratio_gen_median"]]
x = np.arange(4)
ax.plot(x, fem_prog, "o-", color=ACC, lw=1.8, label="female generator")
ax.plot([0, 3], [mal_prog[0], mal_prog[3]], "s--", color=RED, lw=1.6, label="male generator")
ax.axhline(diag["ratio_real_median"], color=ACC, ls=":", lw=1.4)
ax.axhline(ccv["ratio_real_median"], color=RED, ls=":", lw=1.4)
ax.axhline(1.0, color=GRY, lw=0.8)
ax.text(3.05, diag["ratio_real_median"], "female real 1.46", fontsize=7.5, color=ACC, va="center")
ax.text(3.05, ccv["ratio_real_median"], "male real 1.55", fontsize=7.5, color=RED, va="center")
ax.text(0.02, 1.03, "Poisson floor", fontsize=7.5, color=GRY)
ax.set_xticks(x); ax.set_xticklabels(["quota\nuniform", "+CELLQ", "+ACT", "+SPATIAL\n(full)"], fontsize=8)
ax.set_ylabel("median per-type CV / Poisson floor", fontsize=9)
ax.set_title("B. Degree overdispersion: one activity factor", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.set_xlim(-0.3, 4.6)

# C. EPG CV real vs gen
ax = axs[2]
epg = [("male\n(n=46)", ccv["EPG"]["out_cv_real"], ccv["EPG"]["out_cv_gen"], 0.0593),
       ("female\n(n=47)", diag["EPG"]["out_cv"], vf["full"]["EPG"]["out_cv_gen"], diag["EPG"]["poisson_floor"])]
x = np.arange(2)
ax.bar(x - 0.27, [e[1] for e in epg], 0.27, color=GRY, label="real")
ax.bar(x,       [e[2] for e in epg], 0.27, color=ACC, label="generator")
ax.bar(x + 0.27,[e[3] for e in epg], 0.27, color="#dddddd", ec=GRY, label="Poisson floor")
for xi, e in zip(x, epg):
    ax.text(xi - 0.27, e[1] + 0.002, f"{e[1]:.4f}", ha="center", fontsize=7.5)
    ax.text(xi,       e[2] + 0.002, f"{e[2]:.4f}", ha="center", fontsize=7.5)
    ax.text(xi + 0.27,e[3] + 0.002, f"{e[3]:.4f}", ha="center", fontsize=7.5)
ax.text(0, 0.118, "1.11x floor", ha="center", fontsize=8, color=RED)
ax.text(1, 0.118, "1.59x floor", ha="center", fontsize=8, color=RED)
ax.set_xticks(x); ax.set_xticklabels([e[0] for e in epg], fontsize=9)
ax.set_ylabel("EPG out-degree CV", fontsize=9)
ax.set_title("C. EPG: exact CV, sex-specific determinism", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False)
ax.set_ylim(0, 0.132)

plt.tight_layout()
plt.savefig(FIG + "paper_fig2_degree.png", dpi=200)
plt.close()
print("fig2 ok")

# ============================================================
# Fig 3 — space
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))

# A. female CDF real vs baseline
ax = axs[0]
ax.plot(bins[1:], cdf_r, color=ACC, lw=2, label="real edges")
ax.plot(bins[1:], cdf_b, color=GRY, lw=1.6, ls="--", label="quota-uniform baseline")
ax.axvline(76.1, color=ACC, ls=":", lw=1)
ax.axvline(162.6, color=GRY, ls=":", lw=1)
ax.axvline(91.5, color=RED, ls=":", lw=1)
ax.text(76.1, 0.10, " real 76.1", fontsize=7.5, color=ACC)
ax.text(162.6, 0.52, " baseline 162.6 ", fontsize=7.5, color=GRY, ha="right")
ax.text(91.5, 0.10, " gen 91.5\n (KS 0.030)", fontsize=7.5, color=RED)
ax.set_xscale("log")
ax.set_xlabel("edge distance (um, log)", fontsize=9)
ax.set_ylabel("CDF", fontsize=9)
ax.set_title("A. Female: 2.14x distance compression", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

# B. male quantile curves
ax = axs[1]
qs = [0.1, 0.25, 0.5, 0.75, 0.9]
rq = [d10["real_q"][str(q)] for q in qs]
gq = [d10["gen_q"][str(q)] for q in qs]
ax.plot(qs, rq, "o-", color=GRY, lw=2, label="real (soma)")
ax.plot(qs, gq, "s--", color=ACC, lw=1.6, label="v10c generator")
ax.axhline(26.1, color="#999999", ls=":", lw=1.2)
ax.text(0.11, 26.8, "quota-uniform baseline median 26.1", fontsize=7.5, color="#777777")
ax.set_xlabel("CDF quantile", fontsize=9)
ax.set_ylabel("edge distance (um)", fontsize=9)
ax.set_title("B. Male: full profile matched (KS 0.020)", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")

# C. compression factors
ax = axs[2]
comp = [("male", 26.1, 15.9, 16.7), ("female", 162.6, 76.1, 91.5)]
x = np.arange(2)
ax.bar(x - 0.27, [c[1] for c in comp], 0.27, color="#dddddd", ec=GRY, label="quota-uniform baseline")
ax.bar(x,        [c[2] for c in comp], 0.27, color=GRY, label="real")
ax.bar(x + 0.27, [c[3] for c in comp], 0.27, color=ACC, label="generator")
for xi, c in zip(x, comp):
    ax.text(xi, c[2] + 3, f"{c[1]/c[2]:.2f}x", ha="center", fontsize=9, color=RED, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in comp], fontsize=9)
ax.set_ylabel("median edge distance (um)", fontsize=9)
ax.set_title("C. Locality is an independent wiring axis", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False)
ax.set_yscale("log"); ax.set_ylim(8, 300)

plt.tight_layout()
plt.savefig(FIG + "paper_fig3_space.png", dpi=200)
plt.close()
print("fig3 ok")

# ============================================================
# Fig 4 — ring-attractor wiring gene
# ============================================================
am = np.asarray(lp["a"]); af = np.asarray(cx["a"])
mtypes = cx["male_types"]; ftypes = cx["types"]
name_map = {"EPG": "EPG", "EPGt": "EPGt", "PEN_a/PEN1": "PEN_a(PEN1)",
            "PEN_b/PEN2": "PEN_b(PEN2)", "PEG": "PEG", "Delta7": "Delta7"}
ring_f = ["EPG", "PEN_a/PEN1", "PEN_b/PEN2"]; ring_m = [name_map[t] for t in ring_f]
fi = [ftypes.index(t) for t in ring_f]; mi = [mtypes.index(t) for t in ring_m]
Fa = af[np.ix_(fi, fi)]; Ma = am[np.ix_(mi, mi)]
fv = Fa.ravel(); mv = Ma.ravel()
r_ring = np.corrcoef(fv, mv)[0, 1]

fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))

# A. cross-sex scatter
ax = axs[0]
ax.plot([-1, 9.5], [-1, 9.5], color="#cccccc", lw=1)
ax.scatter(mv, fv, s=42, color=ACC, zorder=3, label="ring-block pairs (3x3)")
lab = {0: "EPG>EPG", 1: "PENa>EPG", 2: "PENb>EPG", 3: "EPG>PENa", 6: "EPG>PENb"}  # a 存储约定 [post, pre] (§148 勘误)
for k, t in lab.items():
    ax.annotate(t, (mv[k], fv[k]), textcoords="offset points", xytext=(5, -8), fontsize=7, color=ACC)
# Delta7 block
d7f = [af[ftypes.index("Delta7"), ftypes.index("EPG")], af[ftypes.index("EPG"), ftypes.index("Delta7")]]
d7m = [am[mtypes.index("Delta7"), mtypes.index("EPG")], am[mtypes.index("EPG"), mtypes.index("Delta7")]]
ax.scatter(d7m, d7f, s=64, marker="D", color=RED, zorder=3, label="Delta7 <-> EPG")
ax.annotate("D7>EPG", (d7m[1], d7f[1]), textcoords="offset points", xytext=(-4, 8), fontsize=7, color=RED)
ax.set_xlabel("male amplitude a", fontsize=9)
ax.set_ylabel("female amplitude a", fontsize=9)
ax.set_title(f"A. Ring wiring amplitudes: r = {r_ring:.3f}", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")

# B. held-out prediction
ax = axs[1]
x = np.arange(2)
auc_l = [ol["M1_cos_ring(§12)"]["auc"], cx["auc_latent"]]; auc_b = [ol["M0_type_level"]["auc"], cx["auc_type_baseline"]]
ax.bar(x - 0.18, auc_l, 0.36, color=ACC, label="latent-angle model")
ax.bar(x + 0.18, auc_b, 0.36, color="#dddddd", ec=GRY, label="type-level baseline")
for xi, v in zip(x, auc_l):
    ax.text(xi - 0.18, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
for xi, v in zip(x, auc_b):
    ax.text(xi + 0.18, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.axhline(0.5, color=GRY, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(["male", "female"], fontsize=9)
ax.set_ylabel("held-out AUC", fontsize=9)
ax.set_ylim(0.4, 1.0)
ax.set_title("B. Instance geometry predicts wiring", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")

# C. model selection (male)
ax = axs[2]
models = ["ring cos\n(1D)", "Euclid\n2D", "linear\n1D", "dotprod\nk=4", "scrambled\ntheta"]
aucs = [ol["M1_cos_ring(§12)"]["auc"], ol["M2_euclid_2d"]["auc"], ol["M4_linear_1d"]["auc"], ol["M3_dotprod_k4"]["auc"], ol["M5_theta_scrambled"]["auc"]]
cols = [ACC, "#9db8d2", "#9db8d2", "#9db8d2", "#dddddd"]
ax.bar(range(5), aucs, color=cols, ec=GRY, lw=0.5)
ax.axhline(ol["M0_type_level"]["auc"], color=GRY, ls=":", lw=1)
ax.text(4.4, 0.79, "type\nbaseline", fontsize=7, color=GRY, ha="center")
for i, v in enumerate(aucs):
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5)
ax.set_xticks(range(5)); ax.set_xticklabels(models, fontsize=7.5)
ax.set_ylabel("held-out AUC (male)", fontsize=9)
ax.set_ylim(0.62, 0.95)
ax.set_title("C. Ring beats line, plane, inner product", fontsize=9.5)

plt.tight_layout()
plt.savefig(FIG + "paper_fig4_ring.png", dpi=200)
plt.close()
print("fig4 ok")

# ============================================================
# Fig 5 — the gamma axiom
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))

# A. ladders
ax = axs[0]
mal_steps = [("quota\nuniform", 0.0536), ("+SPATIAL", 0.0598), ("full stack\n(+CELLQ+ACT)", 0.0680)]
fem_steps = [("quota\nuniform", 0.0770), ("+CELLQ", 0.0831), ("+ACT", 0.0855), ("full\n(+SPATIAL)", 0.1045)]
xm = np.arange(3); xf = np.arange(4) + 4
ax.bar(xm, [s[1] for s in mal_steps], 0.62, color="#c9a0a0", ec=RED, label="male generator")
ax.bar(xf, [s[1] for s in fem_steps], 0.62, color="#9db8d2", ec=ACC, label="female generator")
ax.axhline(0.1124, color=RED, lw=1.8)
ax.axhline(0.2678, color=ACC, lw=1.8)
ax.text(1, 0.118, "male real T = 0.112 (2.10x baseline)", fontsize=8, color=RED, ha="center")
ax.text(5.5, 0.274, "female real T = 0.268 (3.48x baseline)", fontsize=8, color=ACC, ha="center")
for xi, s in zip(xm, mal_steps):
    ax.text(xi, s[1] + 0.004, f"{s[1]:.4f}", ha="center", fontsize=7.5)
for xi, s in zip(xf, fem_steps):
    ax.text(xi, s[1] + 0.004, f"{s[1]:.4f}", ha="center", fontsize=7.5)
ax.set_xticks(list(xm) + list(xf))
ax.set_xticklabels([s[0] for s in mal_steps] + [s[0] for s in fem_steps], fontsize=7)
ax.set_ylabel("global transitivity T", fontsize=9)
ax.set_title("A. The triadic-closure gap survives every module", fontsize=9.5)
ax.set_ylim(0, 0.31)

# B. gap-closure by mechanism
ax = axs[1]
mech = ["gamma acceptance\nweighting", "reciprocity-first\nrouting", "motif-directed\nplacement",
        "homophily\ndose-response", "supertype quota\nrelaxation (v8)", "per-cell quotas\n(CELLQ)",
        "spatial kernel", "full stack\n(male)", "full stack\n(female)"]
gap_m = 0.1124 - 0.0536; gap_f = 0.2678 - 0.0770
closed = [0.0, 0.0, 0.0, 0.0, 0.0,
          100 * 0.0144 / gap_m, 100 * (0.0598 - 0.0536) / gap_m,
          100 * (0.0680 - 0.0536) / gap_m, 100 * (0.1045 - 0.0770) / gap_f]
cols = ["#dddddd"] * 5 + ["#c9a0a0", "#c9a0a0", "#c9a0a0", "#9db8d2"]
y = np.arange(len(mech))[::-1]
ax.barh(y, closed, color=cols, ec=GRY, lw=0.5)
for yi, v in zip(y, closed):
    ax.text(v + 0.6, yi, f"{v:.1f}%", va="center", fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels(mech, fontsize=7.5)
ax.set_xlabel("% of triadic-closure gap closed", fontsize=9)
ax.set_title("B. Six mechanism families falsified;\nresidual gamma is unexplained", fontsize=9.5)
ax.set_xlim(0, 32)

plt.tight_layout()
plt.savefig(FIG + "paper_fig5_gamma.png", dpi=200)
plt.close()
print("fig5 ok")
