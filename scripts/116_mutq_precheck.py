#!/usr/bin/env python3
# 116_mutq_precheck.py — §35 P8d 预检: 后处理 MUTQ 的逐细胞可达互惠上界 (解析, 免修复)
# U_i = min(m_i, Σ_t min(out_i(t), revpool_i(t)))   [gen 图内可达]
# recip 上界 = Σ_i U_i / E ; 与注册门 (雌 0.31 / 雄 0.297) 比较
import numpy as np, json, time, gc, sys
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)
R = "/mnt/agents/output/果蝇生成器/"

def run(tag, D, GEN, gate):
    log(f"=== {tag} ===")
    nt = np.fromfile(D + "node_type.i32", dtype=np.int32)
    N = len(nt); T = int(nt.max()) + 1
    # --- real: 逐细胞互惠预算 m_i (census 事实) ---
    rp = np.fromfile(D + "e_pre.i32", dtype=np.int32); rq = np.fromfile(D + "e_post.i32", dtype=np.int32)
    E = len(rp)
    kk = rp.astype(np.int64) * N + rq; kk.sort()
    rk = rq.astype(np.int64) * N + rp
    hit = np.searchsorted(kk, rk)
    ok = (hit < len(kk)) & (kk[np.clip(hit, 0, len(kk) - 1)] == rk)
    m_i = np.bincount(rp[ok], minlength=N).astype(np.int64)
    recip_real = float(ok.mean())
    del kk, rk, hit, ok, rp, rq; gc.collect()
    # --- gen: 反向伙伴池 revpool_i(t) = #{j: type t, j→i placed} ---
    gp = np.fromfile(GEN + "_pre.i32", dtype=np.int32); gq = np.fromfile(GEN + "_post.i32", dtype=np.int32)
    log(f"{tag}: E={E:,} gen E={len(gp):,} recip_real={recip_real:.4f}")
    # 当前 gen recip ( sanity )
    kk = gp.astype(np.int64) * N + gq; kk.sort()
    rk = gq.astype(np.int64) * N + gp
    hit = np.searchsorted(kk, rk)
    ok = (hit < len(kk)) & (kk[np.clip(hit, 0, len(kk) - 1)] == rk)
    recip_gen = float(ok.mean()); del hit, ok, rk; gc.collect()
    # 唯一反向对 (post, pre)
    uk = gq.astype(np.int64) * N + gp
    uk = np.unique(uk)
    del gp, gq; gc.collect()
    b = uk // N; a = uk % N
    gk = b * T + nt[a]            # (细胞, 伙伴型) 反向池条目
    del uk, b, a; gc.collect()
    gu, gc_ = np.unique(gk, return_counts=True)   # 每 (i,t) 池规模
    del gk; gc.collect()
    log(f"{tag}: recip_gen={recip_gen:.4f}  revpool entries={len(gu):,}")
    # --- cq: out_i(t) 与 pool 合并 ---
    cc = np.fromfile(D + "cq_cell.i32", dtype=np.int32)
    ct = np.fromfile(D + "cq_t.i32", dtype=np.int32)
    cn = np.fromfile(D + "cq_cnt.i32", dtype=np.int32)
    ck = cc.astype(np.int64) * T + ct
    h = np.searchsorted(gu, ck)
    pool = np.where((h < len(gu)) & (gu[np.clip(h, 0, len(gu) - 1)] == ck), gc_[np.clip(h, 0, len(gu) - 1)], 0)
    cap = np.minimum(cn, pool)                      # min(out_i(t), pool_i(t))
    capsum = np.bincount(cc, weights=cap, minlength=N)
    U = np.minimum(m_i, capsum.astype(np.int64))
    ub = float(U.sum() / E)
    res = dict(E=int(E), recip_real=round(recip_real, 4), recip_gen=round(recip_gen, 4),
               recip_upper=round(ub, 4), gate=gate,
               achievable=round(float(U.sum() / m_i.sum()), 4),
               deficit_mass=round(float((m_i - U).sum() / E), 4),
               cells_full=int((U == m_i).sum()), cells_partial=int(((U < m_i) & (U > 0)).sum()),
               cells_zero=int(((U == 0) & (m_i > 0)).sum()),
               verdict="后处理可达" if ub >= gate else "后处理上界不足→机制类证伪, 转放置期池工程")
    log(f"{tag}: recip_upper={ub:.4f} vs gate {gate}  m_i 可兑现 {res['achievable']:.1%}  "
        f"缺口质量 {res['deficit_mass']:.4f}  cells full/partial/zero={res['cells_full']:,}/{res['cells_partial']:,}/{res['cells_zero']:,}")
    log(f"{tag}: {res['verdict']}")
    json.dump(res, open(D + "mutq_precheck.json", "w"), indent=1)
    del nt, m_i, gu, gc_, cc, ct, cn, ck, h, pool, cap, capsum, U; gc.collect()

run("female", R + "data_female/", R + "gen_edges/g_dyad3", 0.31)
run("male", R + "data_male/", R + "gen_edges/gm_p8c", 0.297)
log("DONE")
