#!/usr/bin/env python3
"""Stage 43: fig12 CX R² 损失分解图 (§16)"""
import numpy as np, json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = "/mnt/agents/output/果蝇生成器/"
d = json.load(open(ROOT + "cx_decompose.json"))

# (标签, 值, sd, 组)
rows = [
    ("L0  §12重生成: 带符号·突触尺度·λ采样", d["L0_poisson_signed(§12原样)"], "model"),
    ("L1  去符号权重", d["L1_poisson_count(去符号)"], "model"),
    ("C0×12  盒式核采样·突触尺度", d["C0_x12计数"], "scale"),
    ("C0×4", d["C0_x4计数"], "scale"),
    ("C0×2", d["C0_x2计数"], "scale"),
    ("C0  v4.1 实测操作点(连接计数)", d["C0_v4原样(转置+负a均匀)"], "scale"),
    ("C0+精确θ (实例化损失≈0)", d["C0_精确θ"], "scale"),
    ("C0核二值化", d["C0_核二值化"], "scale"),
    ("C6  “修正”参数化(被数据否决)", d["C6_双修正(=v4.2盒)"], "reject"),
    ("U0  全均匀地板", d["U0_全均匀"], "floor"),
]
refs = [
    ("真实·带符号 (旧口径 0.696)", d["real_signed"]["full_mean"], "#2ca02c", "--"),
    ("真实·连接二值 (同口径上限 0.617)", d["real_binary"]["full_mean"], "#2ca02c", "-"),
    ("真实·连接计数 (0.543)", d["real_count"]["full_mean"], "#2ca02c", ":"),
]
C = {"model": "#9467bd", "scale": "#1f77b4", "reject": "#d62728", "floor": "#7f7f7f"}

fig, ax = plt.subplots(figsize=(10.5, 5.6))
ypos = np.arange(len(rows))[::-1]
for y, (lab, rec, grp) in zip(ypos, rows):
    ax.barh(y, rec["full_mean"], xerr=rec.get("full_sd", 0), color=C[grp], alpha=0.85, height=0.62)
    ax.text(rec["full_mean"] + 0.012, y, f'{rec["full_mean"]:.3f}', va="center", fontsize=8.5)
ax.set_yticks(ypos, [r[0] for r in rows], fontsize=9)
for k, (lab, v, c, ls) in enumerate(refs):
    ax.axvline(v, color=c, ls=ls, lw=1.4)
    ax.text(v, len(rows) + 0.15 + k * 0.55, lab, rotation=0, fontsize=8, color=c,
            va="bottom", ha="center")
ax.set_ylim(-0.6, len(rows) + 2.2)
ax.set_xlabel("CX 等效核 cos R²（自环序，full/masked 口径相同）")
ax.set_xlim(0, 0.85)
ax.set_title("图12  CX 核 R² 损失分解：0.755 → 0.508 的每一分在哪（§16）", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT + "figures/fig12_cx_decompose.png", dpi=150, bbox_inches="tight")
print("fig12 saved")
