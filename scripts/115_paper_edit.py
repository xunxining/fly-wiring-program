#!/usr/bin/env python3
# 115_paper_edit.py — 雄侧全栈结果写入论文 (paper.html)
import os
P = "/mnt/agents/output/果蝇生成器/paper/paper.html"
s = open(P).read()
edits = []

# ---- 1. 封面日期 ----
edits.append(("Compiled September 2026", "Compiled October 2026"))

# ---- 2. 摘要: 雄侧全栈回植 ----
edits.append((
"suggesting that cell-type developmental programs specify connection counts, not just connection probabilities.</p>",
"""suggesting that cell-type developmental programs specify connection counts, not just connection probabilities. Finally, the completed program&mdash;including the lattice and dyad modules discovered on the female side&mdash;ports back to the full male CNS with male-only parameters: edge count and out-degrees are again exact, while transitivity (0.182 against the real 0.256) and reciprocity (0.262 against the real 0.300) fall short of their pre-registered parity gates and are so reported, with the shortfalls measured and attributed&mdash;to noisier male lattice coordinates, an unmodelled nerve cord, and exhausted reverse-partner pools&mdash;rather than left unexplained.</p>"""))

# ---- 3. §2.6 标题 ----
edits.append((
"<h3>2.6 Cross-sex transfer with zero male parameters</h3>",
"<h3>2.6 Cross-sex transfer in both directions</h3>"))

# ---- 4. §2.6 新增雄侧段落 + Finding 4 + Fig 6 ----
edits.append((
"pointing to a per-cell mutual-edge budget as the next layer of the grammar.</p>",
"""pointing to a per-cell mutual-edge budget as the next layer of the grammar.</p>

<p>The reverse direction completes the test. The mechanism families validated on the female side&mdash;the neuropil-overlap gate, lattice quota-fill, and the dyad modules&mdash;were frozen and re-parameterized from the male census alone (male genome and quotas, activity factors, distance kernels, lattice coordinates extended from 23,720 male columnar anchors, dyad tables), then executed on the full male CNS, nerve cord included (Fig.&nbsp;6, Table&nbsp;2). The deterministic skeleton transfers perfectly: the edge count is exact to the unit (25,135,616), out-degree KS&nbsp;=&nbsp;0.0000, and the EPG out-degree CV is exact to every printed digit. The two hard statistics transfer only partially, and both misses are pre-registered ones. Transitivity reaches 0.182 against the male-real 0.256 (node-uniform estimator; Methods&nbsp;4.5), below the registered parity gate of 0.20. The gap decomposes geographically (Fig.&nbsp;6B): 44% sits in optic&ndash;optic wedges, where the closure deficit is 1.29&times; (female after lattice: 1.08&times;); 31% in central-brain wedges (1.52&times;); and 17% in the ventral nerve cord (1.98&times;), which carries no lattice machinery at all. For the optic residual the instrument points at coordinate quality rather than a missing mechanism: male lattice anchors are half as numerous (23.7k vs 43.0k) and held-out placement error twice as large (3.1 vs 1.7 columns), diluting same-column assembly. Reciprocity reaches 0.262 against the real 0.300 (Fig.&nbsp;6C), missing the registered gate (0.297, the female level); the male dyad stage closes 64% of its measured mutual-edge deficit (female 75%) with the same signature failure&mdash;zero sources lacking reverse partners, exhaustion of the per-source reverse-partner pool in the largest pairs&mdash;which strengthens, rather than weakens, the case that the residual grammar is a per-cell mutual-edge budget in both sexes. Guard statistics under the male full stack remain exact where construction forces them (edge count, out-degrees, EPG CV) and degrade modestly where redirected placements cost realism (in-degree KS 0.067, distance KS 0.034, per-type CV correlation 0.771); we record the degradation rather than tune it away.</p>

<div class="finding">
<span class="f-title">Finding 4.&nbsp;</span>
<span class="f-body">A wiring program completed on one sex ports to the other with its deterministic skeleton intact: on both directions of transfer, edge counts, out-degree distributions, and EPG degree precision are exact. The closure and reciprocity components transfer partially (male transitivity 0.182 vs real 0.256; reciprocity 0.262 vs 0.300&mdash;both below pre-registered parity gates and so recorded), with the residuals measured and attributed to lattice-coordinate quality, ventral-cord scope, and per-cell mutual budgets: each is a registered next step, not an unexplained mismatch.</span>
</div>

<figure>
  <img src="../figures/paper_fig6_male_port.png" alt="male full-stack port">
  <figcaption><b>Figure 6 | The completed program in both directions: the male full-stack port.</b> <b>A</b>, Transitivity under the node-uniform wedge estimator (identical sampler on real and generated graphs): the male full stack reaches 0.182 against the male-real 0.256; dotted lines mark the pre-registered parity gates (male 0.20; female 0.19, passed at 0.240). <b>B</b>, The male closure gap by endpoint region&mdash;wedge-closure probabilities in the real (gray) and generated (blue) graphs; annotations give the real/generated ratio and each block&rsquo;s share of the total transitivity gap. The male lattice coordinate is measurably noisier than the female (held-out median error 3.1 vs 1.7 columns; 23.7k vs 43.0k anchors). <b>C</b>, Edge reciprocity across module stages for the female (base &rarr; +lattice &rarr; +dyad) and the male (base &rarr; full stack); dashed and dotted lines mark the two real values. Both pre-registered reciprocity gates (female 0.31; male 0.297) were narrowly missed and are so recorded.</figcaption>
</figure>"""))

# ---- 5. Table 2: caption ----
edits.append((
"<caption><b>Table 2 | Cross-sex transfer battery.</b> Female generator parameters were estimated from female data only; the architecture (module set and functional forms) is the male program frozen.</caption>",
"<caption><b>Table 2 | Cross-sex transfer battery.</b> Female parameters were estimated from female data only, male full-stack parameters from male data only; the architecture (module set and functional forms) is frozen across sexes.</caption>"))

# ---- 6. Table 2: 表头加列 ----
edits.append((
"<thead><tr><th>Diagnostic</th><th class=\"num\">male real</th><th class=\"num\">male gen (v10c)</th><th class=\"num\">female real</th><th class=\"num\">female gen (full)</th><th>verdict</th></tr></thead>",
"<thead><tr><th>Diagnostic</th><th class=\"num\">male real</th><th class=\"num\">male gen (v10c)</th><th class=\"num\">male gen (full)</th><th class=\"num\">female real</th><th class=\"num\">female gen (full)</th><th>verdict</th></tr></thead>"))

# ---- 7. Table 2 行更新 (逐行: 在 v10c 列后插入 male full 列) ----
edits.append((
"<tr><td>Edge count</td><td class=\"num\">25,135,616</td><td class=\"num\">25,135,616</td><td class=\"num\">19,656,020</td><td class=\"num\">19,656,020</td><td>exact</td></tr>",
"<tr><td>Edge count</td><td class=\"num\">25,135,616</td><td class=\"num\">25,135,616</td><td class=\"num\">25,135,616</td><td class=\"num\">19,656,020</td><td class=\"num\">19,656,020</td><td>exact</td></tr>"))
edits.append((
"<tr><td>Out-degree KS</td><td class=\"num\">&mdash;</td><td class=\"num\">0.0000</td><td class=\"num\">&mdash;</td><td class=\"num\">0.0000</td><td>exact, both sexes</td></tr>",
"<tr><td>Out-degree KS</td><td class=\"num\">&mdash;</td><td class=\"num\">0.0000</td><td class=\"num\">0.0000</td><td class=\"num\">&mdash;</td><td class=\"num\">0.0000</td><td>exact, both sexes</td></tr>"))
edits.append((
"<tr><td>In-degree KS</td><td class=\"num\">&mdash;</td><td class=\"num\">0.0181</td><td class=\"num\">&mdash;</td><td class=\"num\">0.0235</td><td>transfers</td></tr>",
"<tr><td>In-degree KS</td><td class=\"num\">&mdash;</td><td class=\"num\">0.0181</td><td class=\"num\">0.067&sect;</td><td class=\"num\">&mdash;</td><td class=\"num\">0.0235</td><td>transfers; degrades under male full stack&sect;</td></tr>"))
edits.append((
"<tr><td>Per-type CV / Poisson floor (median)</td><td class=\"num\">1.55</td><td class=\"num\">1.70</td><td class=\"num\">1.46</td><td class=\"num\">1.69</td><td>transfers (incl. overshoot)</td></tr>",
"<tr><td>Per-type CV / Poisson floor (median)</td><td class=\"num\">1.55</td><td class=\"num\">1.70</td><td class=\"num\">1.60</td><td class=\"num\">1.46</td><td class=\"num\">1.69</td><td>transfers (incl. overshoot)</td></tr>"))
edits.append((
"<tr><td>Per-type CV correlation (gen vs real)</td><td class=\"num\">&mdash;</td><td class=\"num\">0.953</td><td class=\"num\">&mdash;</td><td class=\"num\">0.911</td><td>transfers</td></tr>",
"<tr><td>Per-type CV correlation (gen vs real)</td><td class=\"num\">&mdash;</td><td class=\"num\">0.953</td><td class=\"num\">0.771&sect;</td><td class=\"num\">&mdash;</td><td class=\"num\">0.911</td><td>transfers</td></tr>"))
edits.append((
"<tr><td>EPG out-degree CV</td><td class=\"num\">0.0658</td><td class=\"num\">0.0658</td><td class=\"num\">0.1032</td><td class=\"num\">0.1032</td><td>exact, both sexes</td></tr>",
"<tr><td>EPG out-degree CV</td><td class=\"num\">0.0658</td><td class=\"num\">0.0658</td><td class=\"num\">0.0658</td><td class=\"num\">0.1032</td><td class=\"num\">0.1032</td><td>exact, both sexes</td></tr>"))
edits.append((
"<tr><td>Median edge distance</td><td class=\"num\">15.9 &micro;m</td><td class=\"num\">16.7 &micro;m</td><td class=\"num\">76.1 &micro;m&dagger;</td><td class=\"num\">79.0 &micro;m&dagger;&Dagger;</td><td>transfers (improved)</td></tr>",
"<tr><td>Median edge distance</td><td class=\"num\">15.9 &micro;m</td><td class=\"num\">16.7 &micro;m</td><td class=\"num\">&mdash;</td><td class=\"num\">76.1 &micro;m&dagger;</td><td class=\"num\">79.0 &micro;m&dagger;&Dagger;</td><td>transfers (improved)</td></tr>"))
edits.append((
"<tr><td>Distance profile KS</td><td class=\"num\">&mdash;</td><td class=\"num\">0.020</td><td class=\"num\">&mdash;</td><td class=\"num\">0.014&Dagger;</td><td>transfers (improved)</td></tr>",
"<tr><td>Distance profile KS</td><td class=\"num\">&mdash;</td><td class=\"num\">0.020</td><td class=\"num\">0.034&sect;</td><td class=\"num\">&mdash;</td><td class=\"num\">0.014&Dagger;</td><td>transfers (improved)</td></tr>"))
edits.append((
"<tr><td>CX latent-angle held-out AUC</td><td class=\"num\">&mdash;</td><td class=\"num\">0.906</td><td class=\"num\">&mdash;</td><td class=\"num\">0.849</td><td>transfers</td></tr>",
"<tr><td>CX latent-angle held-out AUC</td><td class=\"num\">&mdash;</td><td class=\"num\">0.906</td><td class=\"num\">&mdash;</td><td class=\"num\">&mdash;</td><td class=\"num\">0.849</td><td>transfers</td></tr>"))
edits.append((
"<tr><td>Ring-block amplitude corr. (cross-sex)</td><td colspan=\"4\" style=\"text-align:center\">$r = 0.985$ (EPG/PEN 3&times;3); 0.688 including &Delta;7</td><td>invariant</td></tr>",
"<tr><td>Ring-block amplitude corr. (cross-sex)</td><td colspan=\"5\" style=\"text-align:center\">$r = 0.985$ (EPG/PEN 3&times;3); 0.688 including &Delta;7</td><td>invariant</td></tr>"))
edits.append((
"<tr><td>Global transitivity $T$</td><td class=\"num\">0.112</td><td class=\"num\">0.068</td><td class=\"num\">0.268</td><td class=\"num\">0.240&Dagger;</td><td><b>resolved (Sec. 2.5)</b></td></tr>",
"<tr><td>Global transitivity $T$ (node-uniform)&para;</td><td class=\"num\">0.256</td><td class=\"num\">&mdash;</td><td class=\"num\">0.182&sect;</td><td class=\"num\">0.257</td><td class=\"num\">0.240&Dagger;</td><td><b>female resolved (Sec. 2.5); male partial&sect;</b></td></tr>"))
edits.append((
"<tr><td>Edge reciprocity</td><td class=\"num\">0.300</td><td class=\"num\">0.112</td><td class=\"num\">0.321</td><td class=\"num\">0.299&Dagger;</td><td><b>partial (gate missed)</b></td></tr>",
"<tr><td>Edge reciprocity</td><td class=\"num\">0.300</td><td class=\"num\">0.112</td><td class=\"num\">0.262&sect;</td><td class=\"num\">0.321</td><td class=\"num\">0.299&Dagger;</td><td><b>partial (both gates missed)</b></td></tr>"))

# ---- 8. Table 2 脚注 ----
edits.append((
"Without the lattice module: $T=0.112$, reciprocity 0.128, distance KS 0.026; with the lattice module but without dyad modules: reciprocity 0.225.</p>",
"""Without the lattice module: $T=0.112$, reciprocity 0.128, distance KS 0.026; with the lattice module but without dyad modules: reciprocity 0.225. &para; Node-uniform wedge estimator (Methods 4.5) used for all full-stack transitivity values; classic wedge-uniform transitivity reads 0.112 (male real), 0.068 (male v10c gen), 0.268 (female real). &sect; Male full stack (neuropil gate + lattice quota-fill + dyad modules, male-only parameters; Sec. 2.6, Fig. 6): edge count exact (25,135,616), out-degree KS 0.0000, EPG out-degree CV exact; both pre-registered parity gates missed and so recorded ($T$ 0.182 vs gate 0.20; reciprocity 0.262 vs gate 0.297); the male dyad stage closes 64% of its measured mutual-edge deficit (female 75%) with the same pool-exhaustion signature; in-degree KS, distance KS, and CV correlation degrade modestly relative to the male v10c stack (0.067 / 0.034 / 0.771) as the price of redirected placements.</p>"""))

# ---- 9. Limitations (ii) 雄侧状态更新 ----
edits.append((
"The male generator remains at 0.112 pending the lattice port.",
"The male full-stack port (male-only parameters; Sec.&nbsp;2.6) now reaches 0.262&mdash;64% of its measured dyad deficit closed, versus 75% in the female&mdash;with transitivity 0.182 against the male-real 0.256; both male parity gates were pre-registered and missed, and the shortfalls carry measured attributions (lattice-coordinate quality, unmodelled ventral cord, reverse-pool exhaustion) rather than free parameters."))

# ---- 10. Outlook 互惠句 ----
edits.append((
"The one open term&mdash;reciprocity&mdash;is now three-quarters explained by census-level dyad structure, with the residual measured, localized, and attributed to per-cell mutual budgets under registered falsification criteria.",
"The one open term&mdash;reciprocity&mdash;is now three-quarters explained in the female and two-thirds in the male by census-level dyad structure, with the residual in both sexes carrying the same measured signature&mdash;exhausted reverse-partner pools&mdash;and attributed to per-cell mutual budgets under registered falsification criteria."))

# ---- 11. Methods 4.3 雄侧验证器 ----
edits.append((
"All generation runs use fixed seeds; edge counts are exact by construction; no male parameter entered any female fit.</p>",
"All generation runs use fixed seeds; edge counts are exact by construction; no male parameter entered any female fit. The male full-stack battery (edge count, degree KS, dispersion, transitivity, reciprocity, distance histogram, EPG CV) is the same instrument with identical estimator settings, run on MaleCNS; the male node-uniform real transitivity is 0.2545&ndash;0.2560 across sampler seeds (500k wedges), reported as 0.256.</p>"))

# ---- 12. Methods 4.6 雄侧晶格 ----
edits.append((
"Pairs with overflow or cross-hemisphere modes, and cells without coordinates, fall back to the distance-kernel sampler (7.0% of edges).",
"Pairs with overflow or cross-hemisphere modes, and cells without coordinates, fall back to the distance-kernel sampler (7.0% of edges). <b>Male port.</b> The male lattice was built identically, from 23,720 male medulla columnar anchors (MaleCNS column assignment): harmonic extension over the male visual-subsystem adjacency graph on a 90% edge split, coverage 99.8&ndash;100% of the four visual superclasses, held-out anchor error median 3.1 columns (female 1.7) with exact hemisphere assignment, and modal-offset quota-fill tables for 8,439 pairs covering 82% of eligible edges."))

# ---- 13. Methods 4.7 雄侧 dyad ----
edits.append((
"Both components use only census- and quota-layer information; no cell-identity correspondences are introduced.",
"Both components use only census- and quota-layer information; no cell-identity correspondences are introduced. <b>Male port.</b> The male dyad tables were derived from the male census and executed with the same DYAD1+2 stack: the repair stage closed 64.2% of the measured male mutual-edge deficit (745,628 of 1,161,560 edges), with attribution counters again recording zero no-candidate sources and reverse-pool exhaustion (675,664 edge-equivalents) as the dominant failure mode&mdash;the same signature as the female, at a lower closure rate (75%)."))

# ================= 应用 =================
for old, new in edits:
    assert s.count(old) == 1, f"target not unique/found ({s.count(old)}): {old[:70]!r}"
    s = s.replace(old, new)
tmp = P + ".tmp"
open(tmp, "w").write(s)
os.replace(tmp, P)
print(f"OK: {len(edits)} edits applied, {len(s)} bytes")
