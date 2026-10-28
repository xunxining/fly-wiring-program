#!/usr/bin/env python3
"""100_dyad_attain.py — P8b 达成度诊断: 生成图逐对互惠数 G_st vs 实值配额 R_st。
输出: 总体达成率、缺口分布 (按对规模)、缺口对的结构归因。-> dyad_attain_diag.json"""
import numpy as np, json, time, gc
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
N = 138327
gp = np.fromfile(D + "../gen_edges/g_dyad2_pre.i32", dtype=np.int32).astype(np.int64)
gq = np.fromfile(D + "../gen_edges/g_dyad2_post.i32", dtype=np.int32).astype(np.int64)
nt = np.fromfile(D + "node_type.i32", dtype=np.int32).astype(np.int64)
T = int(nt.max()) + 1
E = len(gp)
dz = np.load(D + "dyad_f.npz")
Ruk, RR = dz['uk'].astype(np.int64), dz['R'].astype(np.int64)

# 生成图互惠
key = gp * N + gq
ks = np.sort(key)
rev = gq * N + gp
pos = np.minimum(np.searchsorted(ks, rev), E - 1)
is_mut = ks[pos] == rev
del key, ks, rev, pos; gc.collect()

# 逐对聚合
pair = nt[gp] * T + nt[gq]
del gp, gq; gc.collect()
up, cnt = np.unique(pair, return_counts=True)
upm, cmut = np.unique(pair[is_mut], return_counts=True)
del pair, is_mut; gc.collect()
G = np.zeros(len(up), dtype=np.int64)
G[np.searchsorted(up, upm)] = cmut
# 对齐 R
pp = np.minimum(np.searchsorted(Ruk, up), len(Ruk) - 1)
hit = Ruk[pp] == up
R_al = np.where(hit, RR[np.minimum(pp, len(Ruk) - 1)], 0)

tot_R = int(R_al.sum()); tot_G = int(G.sum())
attain = float(np.minimum(G, R_al).sum() / tot_R)
short = R_al - np.minimum(G, R_al)          # 每对缺口
nz = short > 0
# 缺口按对规模分桶
bins = [(1, 5), (5, 20), (20, 100), (100, 1000), (1000, 10**9)]
bd = {}
for lo, hi in bins:
    m = nz & (cnt >= lo) & (cnt < hi)
    bd[f"{lo}-{hi if hi<10**9 else 'inf'}"] = dict(pairs=int(m.sum()), short=int(short[m].sum()))
# 超额对 (G>R, 结构性多出的互惠)
over = int(np.maximum(G - R_al, 0).sum())
diag = dict(E=E, recip_gen=float(tot_G / E), recip_real=float(tot_R / E),
            attain_vs_R=attain, total_short_dyads=int(short.sum() // 2),
            total_R=int(tot_R), total_G=int(tot_G),
            pairs_with_short=int(nz.sum()), pairs_total=int(len(up)),
            over_dyads=int(over // 2), short_by_pairsize=bd,
            runtime_s=round(time.time() - t0, 1))
json.dump(diag, open(D + "dyad_attain_diag.json", "w"), indent=1)
print(json.dumps(diag, indent=1), flush=True)
