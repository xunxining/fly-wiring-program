"""114_paper_fig6.py — Fig 6: the program in both directions (male full-stack transfer)."""
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/mnt/agents/output/果蝇生成器/"
FIG = ROOT + "figures/"
ACC = "#1f4e79"; GRY = "#555555"; RED = "#b03a2e"; LGT = "#7aa5c4"

val = json.load(open(ROOT + "data_male/val_gm_p8c.json"))
tg  = json.load(open(ROOT + "data_male/tgap_m.json"))

T_m_gen, R_m_gen = val["T_gen"], val["recip_gen"]
T_m_real, R_m_real = tg["T_real"], val["recip_real"]
T_f_gen, T_f_real = 0.2396, 0.2565
R_f_base, R_f_lat, R_f_gen, R_f_real = 0.128, 0.225, 0.299, 0.321
R_f_mutq, R_m_mutq = 0.348, 0.319
R_f_p8e, R_m_p8e = 0.322, 0.299
R_m_base = 0.112

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
fig.subplots_adjust(wspace=0.30, left=0.06, right=0.99, top=0.86, bottom=0.23)

# ---------- A ----------
ax = axes[0]
x = np.arange(2); w = 0.34
ax.bar(x - w/2, [T_m_real, T_f_real], w, color=GRY, alpha=0.85, label="real")
ax.bar(x + w/2, [T_m_gen, T_f_gen], w, color=ACC, label="generated (full stack)")
for xi, v in zip(x - w/2, [T_m_real, T_f_real]):
    ax.text(xi, v - 0.016, f"{v:.3f}", ha="center", fontsize=8, color="white", fontweight="bold")
for xi, v in zip(x + w/2, [T_m_gen, T_f_gen]):
    ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8, color=ACC, fontweight="bold")
ax.axhline(0.20, color=RED, ls=":", lw=1.2)
ax.axhline(0.19, color=RED, ls=":", lw=0.8, alpha=0.45)
ax.set_xlim(-0.55, 1.55)
ax.set_xticks(x); ax.set_xticklabels(["male (MaleCNS)", "female (FlyWire)"], fontsize=9)
ax.set_ylim(0, 0.31); ax.set_ylabel("transitivity $T$ (node-uniform)", fontsize=9)
ax.set_title("A  Triadic closure", fontsize=10, loc="left", fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", borderaxespad=0.1)
ax.spines[["top", "right"]].set_visible(False)

# ---------- B ----------
ax = axes[1]
blocks = [b for b in tg["blocks"] if b["block"] in ("OPTIC-OPTIC", "CENTRAL-CENTRAL", "VNC-VNC")]
labels = ["optic–optic", "central–central", "VNC–VNC"]
pr = [b["p_real"] for b in blocks]; pg = [b["p_gen"] for b in blocks]
tot = sum(b["gap_contrib"] for b in tg["blocks"])
y = np.arange(len(blocks))[::-1]; h = 0.36
ax.barh(y + h/2, pr, h, color=GRY, alpha=0.85, label="real")
ax.barh(y - h/2, pg, h, color=ACC, label="generated")
for yi, b in zip(y, blocks):
    ax.text(0.345, yi, f"{b['ratio']}×   gap share {b['gap_contrib']/tot*100:.0f}%", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_ylim(-1.35, 2.6)
ax.set_xlim(0, 0.55); ax.set_xlabel("wedge-closure probability", fontsize=9)
ax.set_title("B  Male closure gap by region", fontsize=10, loc="left", fontweight="bold")
ax.text(0.0, -0.78, "lattice held-out median error: male 3.1 vs female 1.7 columns", fontsize=7.4, color=RED)
ax.text(0.0, -1.14, "(23.7k vs 43.0k anchors) — coordinate quality is the binding constraint", fontsize=7.4, color=RED)
ax.spines[["top", "right"]].set_visible(False)

# ---------- C ----------
ax = axes[2]
MUT = "#0e3a5d"; P8E = "#1b7f5c"
xf = np.array([0, 1, 2, 3, 4]); xm = np.array([5.75, 6.75, 7.75, 8.75])
ax.bar(xf, [R_f_base, R_f_lat, R_f_gen, R_f_mutq, R_f_p8e], 0.62, color=[LGT, LGT, ACC, MUT, P8E])
ax.bar(xm, [R_m_base, R_m_gen, R_m_mutq, R_m_p8e], 0.62, color=[LGT, ACC, MUT, P8E])
for xi, v in zip(list(xf) + list(xm), [R_f_base, R_f_lat, R_f_gen, R_f_mutq, R_f_p8e, R_m_base, R_m_gen, R_m_mutq, R_m_p8e]):
    ax.text(xi, v + 0.007, f"{v:.3f}", ha="center", fontsize=7.6, color="#333")
ax.axhline(R_f_real, color=GRY, ls="--", lw=1.1)
ax.text(-0.45, R_f_real + 0.016, "female real 0.321", fontsize=7, color=GRY)
ax.axhline(R_m_real, color=GRY, ls=":", lw=1.1)
ax.text(9.2, 0.372, "male real 0.300", fontsize=7, color=GRY, ha="right")
ax.plot([4.9, 4.9], [0, 0.40], color="#bbbbbb", lw=1)
ax.set_xticks(list(xf) + list(xm))
ax.set_xticklabels(["base", "+LAT", "+DYAD", "+MUTQ", "+P8e", "base", "full\nstack", "+MUTQ", "+P8e"],
                   fontsize=6.8, rotation=24, ha="right")
ax.text(2.0, -0.115, "female", fontsize=9.5, ha="center")
ax.text(7.25, -0.115, "male", fontsize=9.5, ha="center")
ax.set_xlim(-0.6, 9.35)
ax.set_ylim(0, 0.40); ax.set_ylabel("edge reciprocity", fontsize=9)
ax.set_title("C  Reciprocity: quota-exact", fontsize=10, loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

plt.savefig(FIG + "paper_fig6_male_port.png", dpi=200)
print("saved")
