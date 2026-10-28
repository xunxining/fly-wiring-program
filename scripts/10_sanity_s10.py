#!/usr/bin/env python3
"""Stage 10: S=10 网络 sanity check —— 全 memmap 流式, 小内存安全
输出: gen/validation_s10.json + gen/s10_degrees.npz (供绘图)
"""
import numpy as np
import json, gc

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"
rng = np.random.default_rng(99)

def log(*a): print(*a, flush=True)

def ks(a, b):
    a = np.sort(a); b = np.sort(b)
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))

# ---- 真实图 ----
rpre = np.memmap(D + "e_pre.i32", dtype=np.int32, mode="r")
rpost = np.memmap(D + "e_post.i32", dtype=np.int32, mode="r")
rw = np.memmap(D + "e_w.i32", dtype=np.int32, mode="r")
node_type = np.load(D + "node_type.npy")
Nr = len(node_type)
Er = len(rpre)
rout = np.bincount(rpre, minlength=Nr)
rin = np.bincount(rpost, minlength=Nr)
syn_real = int(np.asarray(rw, dtype=np.int64).sum())
log("real loaded:", Er, "edges")

# ---- S=10 生成图 ----
gpre = np.memmap(OUT + "gen_S10_pre.i32", dtype=np.int32, mode="r")
gpost = np.memmap(OUT + "gen_S10_post.i32", dtype=np.int32, mode="r")
gw = np.memmap(OUT + "gen_S10_w.i32", dtype=np.int32, mode="r")
Eg = len(gpre)
d = np.load(OUT + "distribs.npz")
type_count = d['type_count'].astype(np.int64)
T = len(type_count)
Ng = int(type_count.sum() * 10)
log("gen loaded:", Eg, "edges,", Ng, "nodes")

gout = np.bincount(gpre, minlength=Ng)
gin = np.bincount(gpost, minlength=Ng)
log("degrees done")

# 突触总数 (分块)
syn_gen = 0
for i0 in range(0, Eg, 8_000_000):
    syn_gen += int(np.asarray(gw[i0:i0+8_000_000], dtype=np.int64).sum())
log("synapses done")

# 权重 KS (抽样)
ns = 2_000_000
idx_r = np.sort(rng.integers(0, Er, ns)); idx_g = np.sort(rng.integers(0, Eg, ns))
ks_w = ks(np.asarray(rw[idx_r], dtype=np.float32), np.asarray(gw[idx_g], dtype=np.float32))
log("weight ks done")

# 互惠性 (1M 边样本, 哈希集合)
def recip_sample(pre, post, E, n=1_000_000):
    idx = rng.integers(0, E, n)
    a = np.asarray(pre[idx], dtype=np.int64); b = np.asarray(post[idx], dtype=np.int64)
    keys = set((a * (np.iinfo(np.int32).max + 1) + b).tolist())
    hit = 0
    for x, y in zip(a.tolist(), b.tolist()):
        if (y * (np.iinfo(np.int32).max + 1) + x) in keys:
            hit += 1
    return hit / n
recip_r = recip_sample(rpre, rpost, Er)
recip_g = recip_sample(gpre, gpost, Eg)
log("reciprocity done")

# 类型对边数 Spearman (分块累计)
offsets = np.zeros(T, dtype=np.int64)
offsets[1:] = np.cumsum(type_count * 10)[:-1]
from collections import Counter
pair_cnt = Counter()
for i0 in range(0, Eg, 10_000_000):
    i1 = min(i0 + 10_000_000, Eg)
    p = np.asarray(gpre[i0:i1], dtype=np.int64)
    q = np.asarray(gpost[i0:i1], dtype=np.int64)
    tp = np.searchsorted(offsets, p, side="right") - 1
    tq = np.searchsorted(offsets, q, side="right") - 1
    uk, c = np.unique(tp * T + tq, return_counts=True)
    for k, v in zip(uk.tolist(), c.tolist()):
        pair_cnt[k] += v
    if i0 % 50_000_000 == 0:
        log(f"  pair agg {i0:,}/{Eg:,}")
# 真实类型对: 从 typepair.npz
dd = np.load(D + "typepair.npz")
uk_r, cnt_r = dd['uk'].astype(np.int64), dd['cnt'].astype(np.int64)
cnt_g = np.array([pair_cnt.get(int(k), 0) for k in uk_r], dtype=np.float64) / 10.0
from scipy.stats import spearmanr
spr = float(spearmanr(cnt_r, cnt_g).statistic)
log("spearman done")

res = {
    "edges_real": int(Er), "edges_gen_S10": int(Eg), "edges_ratio": Eg / Er,
    "nodes_real": int(Nr), "nodes_gen_S10": int(Ng),
    "syn_real": syn_real, "syn_gen_S10": syn_gen, "syn_ratio": syn_gen / syn_real,
    "ks_outdeg": ks(rout, gout), "ks_indeg": ks(rin, gin),
    "ks_weight": ks_w,
    "recip_real_sample": recip_r, "recip_gen_S10_sample": recip_g,
    "typepair_spearman": spr,
}
json.dump(res, open(OUT + "validation_s10.json", "w"), indent=2)
np.savez(OUT + "s10_degrees.npz", rout=rout.astype(np.int32), rin=rin.astype(np.int32),
         gout=gout.astype(np.int32), gin=gin.astype(np.int32))
log("SANITY DONE")
print(json.dumps(res, indent=2))
