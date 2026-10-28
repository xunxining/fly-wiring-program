#!/usr/bin/env python3
# 122_trim_precheck.py — §36 P8e 预检: 盈余配对率与可修剪上界(解析)
# 盈余细胞 cur_i>m_i; 盈余对 = 双端皆盈余的互惠对. 判: 双端盈余覆盖 >=50% ?
import json, sys, time
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

for SEX, GEN, DR in (("female", "g_mutq", "data_female"), ("male", "g_mutqm", "data_male")):
    D = f"{ROOT}/{DR}"
    nt = np.fromfile(f"{D}/node_type.i32", dtype=np.int32)
    N = int(nt.size)
    rp = np.fromfile(f"{D}/e_pre_s.i32" if SEX == "male" else f"{D}/e_pre.i32", dtype=np.int32)
    rq = np.fromfile(f"{D}/e_post_s.i32" if SEX == "male" else f"{D}/e_post.i32", dtype=np.int32)
    gp = np.fromfile(f"{ROOT}/gen_edges/{GEN}_pre.i32", dtype=np.int32)
    gq = np.fromfile(f"{ROOT}/gen_edges/{GEN}_post.i32", dtype=np.int32)
    # 真实 m_i
    kr = np.unique(rp.astype(np.int64) * N + rq)
    rk = rq.astype(np.int64) * N + rp
    p = np.clip(np.searchsorted(kr, rk), 0, kr.size - 1)
    mok = kr[p] == rk
    m_i = np.bincount(rp[mok], minlength=N).astype(np.int64)
    del kr, rk, p, mok, rp, rq
    # 当前 cur_i 与互惠对表
    kk = np.unique(gp.astype(np.int64) * N + gq)
    gk = gq.astype(np.int64) * N + gp
    p = np.clip(np.searchsorted(kk, gk), 0, kk.size - 1)
    gmut = kk[p] == gk
    cur = np.bincount(gp[gmut], minlength=N).astype(np.int64)
    surplus = np.clip(cur - m_i, 0, None)
    deficit = np.clip(m_i - cur, 0, None)
    # 盈余对: 互惠对中双端皆有盈余
    ma, mb = gp[gmut], gq[gmut]  # 互惠有向边(双向各计一次)
    both = (surplus[ma] > 0) & (surplus[mb] > 0)
    # 每细胞可修剪量 = min(自身盈余, 双端盈余对数(以其为 source))
    both_cnt = np.bincount(ma[both], minlength=N)
    trimmable = np.minimum(surplus, both_cnt)
    tot_surplus_pairs = int(surplus.sum())
    both_cover = int(trimmable.sum())
    res = dict(
        sex=SEX, gen=GEN,
        recip_cur=round(float(gmut.sum()) / gp.size, 5),
        surplus_cells=int((surplus > 0).sum()), deficit_cells=int((deficit > 0).sum()),
        surplus_mass=tot_surplus_pairs, deficit_mass=int(deficit.sum()),
        both_surplus_cover=both_cover,
        cover_ratio=round(both_cover / max(tot_surplus_pairs, 1), 4),
        recip_after_pure_trim=round(float(gmut.sum() - 2 * both_cover) / gp.size, 5),
        recip_real=round(float(m_i.sum()) / gp.size, 5),
    )
    log(f"{SEX}: {json.dumps(res, ensure_ascii=False)}")
    with open(f"{D}/trim_precheck.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
log("done")
