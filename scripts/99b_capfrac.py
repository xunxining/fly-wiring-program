#!/usr/bin/env python3
"""99b_capfrac.py — P8b 预备: 逐有序类型对的「互惠潜能」统计。
对每条实边 (i->j, 类型 s->t): j 是否有向 s 的出配额 (cq 记录 (j,s)) → capable。
cap_frac[s,t] = capable 边占比。另存全局均值。输出 dyad_cap_f.npz + diag。"""
import numpy as np, json, time, gc
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32).astype(np.int64)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32).astype(np.int64)
nt = np.fromfile(D + "node_type.i32", dtype=np.int32).astype(np.int64)
T = int(nt.max()) + 1
E = len(ep)
cq_cell = np.fromfile(D + "cq_cell.i32", dtype=np.int32).astype(np.int64)
cq_t = np.fromfile(D + "cq_t.i32", dtype=np.int32).astype(np.int64)
# 细胞级潜能键: (细胞, 出配额伙伴类型)
ck = np.sort(cq_cell * T + cq_t)
del cq_cell, cq_t; gc.collect()
si_ = nt[ep]; ti_ = nt[eq]
key2 = eq * T + si_                      # (j, s)
pos = np.minimum(np.searchsorted(ck, key2), len(ck) - 1)
capable = ck[pos] == key2
del key2, pos, ck; gc.collect()
pair = si_ * T + ti_
del si_, ti_; gc.collect()
up, cnt = np.unique(pair, return_counts=True)
upc, ccap = np.unique(pair[capable], return_counts=True)
cap_al = np.zeros(len(up), dtype=np.int64)
cap_al[np.searchsorted(up, upc)] = ccap
cap_frac = cap_al / cnt
np.savez_compressed(D + "dyad_cap_f.npz", uk=up.astype(np.int64),
                    cap=cap_al.astype(np.int32), E=cnt.astype(np.int32))
diag = dict(E=E, n_pairs=int(len(up)),
            cap_frac_global=float(capable.mean()),
            cap_frac_pair_mean=float(cap_frac[cnt >= 10].mean()),
            cap_frac_pair_p50=float(np.median(cap_frac[cnt >= 10])),
            runtime_s=round(time.time() - t0, 1))
json.dump(diag, open(D + "dyad_cap_diag.json", "w"), indent=1)
print(json.dumps(diag, indent=1), flush=True)
