#!/usr/bin/env python3
"""99_dyad_stats.py — P8 预备: 实测雌性逐类型对互惠配额 R_st 与全局诊断。
R_st = #(i->j, i∈s, j∈t 且 j->i 存在)。输出 dyad_f.npz + dyad_diag.json。"""
import numpy as np, json, time, gc
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32).astype(np.int64)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32).astype(np.int64)
N = 138327
nt = np.fromfile(D + "node_type.i32", dtype=np.int32).astype(np.int64)
T = int(nt.max()) + 1
E = len(ep)
print(f"E={E} T={T}", flush=True)

# 全局互惠率: 排序键 + searchsorted
key = ep * N + eq
ks = np.sort(key)
rev = eq * N + ep
pos = np.searchsorted(ks, rev)
pos_c = np.minimum(pos, E - 1)
is_mut = ks[pos_c] == rev          # 每条边: 反向是否存在
recip = float(is_mut.mean())
print(f"recip_real={recip:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# 逐有序类型对: 边数 E_st 与互惠边数 R_st (稀疏聚合, 不开 T*T 稠密数组)
sp = nt[ep]; tp = nt[eq]
pair = sp * T + tp
del sp, tp, ep, eq, key, ks, rev, pos, pos_c; gc.collect()
up, cnt = np.unique(pair, return_counts=True)
upm, cmut = np.unique(pair[is_mut], return_counts=True)
del pair; gc.collect()
R_al = np.zeros(len(up), dtype=np.int64)
idx = np.searchsorted(up, upm); R_al[idx] = cmut
# 对称性检查: R_st 应等于 R_ts
rev_pair = (up % T) * T + (up // T)
pp = np.searchsorted(up, rev_pair)
ppc = np.minimum(pp, len(up) - 1)
hit = up[ppc] == rev_pair
sym_diff = int(np.abs(R_al[hit] - R_al[ppc[hit]]).max()) if hit.any() else 0
# 互惠率分布 (仅 E_st>=10 的对)
big = cnt >= 10
rr = R_al[big] / cnt[big]
np.savez_compressed(D + "dyad_f.npz", uk=up.astype(np.int64), E=cnt.astype(np.int32),
                    R=R_al.astype(np.int32))
diag = dict(recip_real=recip, E=E, T=T, n_pairs=int(len(up)),
            sym_maxdiff=int(sym_diff),
            recip_pair_mean=float(rr.mean()), recip_pair_p50=float(np.median(rr)),
            recip_pair_p90=float(np.percentile(rr, 90)),
            frac_pairs_recip0=float((R_al == 0).mean()),
            edges_in_mut=float(is_mut.sum() / E),
            runtime_s=round(time.time() - t0, 1))
json.dump(diag, open(D + "dyad_diag.json", "w"), indent=1)
print(json.dumps(diag, indent=1), flush=True)
