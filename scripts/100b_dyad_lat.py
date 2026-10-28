#!/usr/bin/env python3
"""100b_dyad_lat.py — P8c 可行性: 实值互惠 dyad 的晶格偏移结构。
对每个大对 (s,t): 互惠边 (i->j) 的偏移 Δ_st 分布 vs 全部边的 Δ_st 分布;
以及互惠 dyad 的 Δ_st+Δ_ts 偏移和分布 (0 = 同柱往返)。
输出 dyad_lat_diag.json"""
import numpy as np, json, time
D = "/mnt/agents/output/果蝇生成器/data_female/"
t0 = time.time()
N = 138327
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32).astype(np.int64)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32).astype(np.int64)
nt = np.fromfile(D + "node_type.i32", dtype=np.int32).astype(np.int64)
T = int(nt.max()) + 1
zq = np.load(D + "lat2q_f3.npz")
PQ = zq['pq']; HEM = zq['hem']
has = HEM != 0

# 互惠标记
key = ep * N + eq
ks = np.sort(key)
rev = eq * N + ep
pos = np.minimum(np.searchsorted(ks, rev), len(key) - 1)
is_mut = ks[pos] == rev

sp = nt[ep]; tp = nt[eq]
pair = sp * T + tp

# 选 30 个最大且有坐标的对做详细统计
dz = np.load(D + "dyad_f.npz")
Ruk, RE, RR = dz['uk'].astype(np.int64), dz['E'].astype(np.int64), dz['R'].astype(np.int64)
# 类型覆盖标记
toff = np.concatenate([[0], np.cumsum(np.bincount(nt, minlength=T))])
# (节点已按类型排序: 以 toff 段均值判覆盖)
tcnt = np.bincount(nt, minlength=T)
t_lat = np.zeros(T, bool)
for t_ in range(T):
    seg = slice(toff[t_], toff[t_] + tcnt[t_])
    if tcnt[t_] > 0:
        t_lat[t_] = has[seg].mean() > 0.9
o = np.argsort(RE)[::-1]
big = [pk for pk in o if RR[pk] > 0 and t_lat[int(Ruk[pk] // T)] and t_lat[int(Ruk[pk] % T)]][:120]

res_off = []   # 互惠边 Δ 分布锐度 vs 全部
res_sum = []   # dyad 偏移和
for pk in big[:200]:
    s, t = int(pk // T), int(pk % T)
    pk_ = Ruk[pk] if isinstance(pk, (int, np.integer)) and pk < len(Ruk) else pk
    s, t = int(pk_ // T), int(pk_ % T)
    m = pair == pk_
    if m.sum() < 500: continue
    ei = ep[m]; ej = eq[m]; em = is_mut[m]
    ok = has[ei] & has[ej] & (HEM[ei] == HEM[ej])
    if ok.sum() < 200: continue
    d_all = PQ[ej[ok]] - PQ[ei[ok]]                       # (n,2)
    d_mut = PQ[ej[ok & em]] - PQ[ei[ok & em]] if (ok & em).any() else None
    # 模态偏移
    da = np.round(d_all).astype(np.int64)
    uk_, cn_ = np.unique(da, axis=0, return_counts=True)
    mp, mq = uk_[cn_.argmax()]
    at_mode_all = float(((da[:, 0] == mp) & (da[:, 1] == mq)).mean())
    if d_mut is not None and len(d_mut) >= 50:
        dm = np.round(d_mut).astype(np.int64)
        at_mode_mut = float(((dm[:, 0] == mp) & (dm[:, 1] == mq)).mean())
        res_off.append((at_mode_all, at_mode_mut, int(m.sum()), int(em.sum())))
    # dyad 偏移和: 对互惠边 i->j, 找 j->i 的偏移
    # 用键匹配: j->i 边
    if em.sum() >= 50:
        mi = ei[em]; mj = ej[em]
        # 建 (j,i)->row 映射 (全部边中)
        rk = ej * N + ei
        rks = np.sort(rk)
        q = mj * N + mi
        pp = np.minimum(np.searchsorted(rks, q), len(rks) - 1)
        hit = rks[pp] == q
        okh = hit & has[mi] & has[mj] & (HEM[mi] == HEM[mj])
        if okh.sum() >= 50:
            d_fwd = PQ[mj[okh]] - PQ[mi[okh]]
            # 反向边行号: pp[okh] 是排序数组位置 -> 原边索引需 rev? rks 是 rk 的排序
            # 直接重建: 反向边 (mj->mi) 的偏移 = -(d_fwd) 若同细胞对
            # 但我们需要实测反向边的偏移来验证同细胞对: hit 保证同 (j,i) 键 => 同一对细胞
            # 因此反向偏移恒等于 -d_fwd (同细胞对间), 此项无需计算; 改测 |d_fwd - modal|
            dist_mode = np.abs(np.round(d_fwd) - np.array([mp, mq])).sum(1)
            res_sum.append((float(np.median(dist_mode)), float((dist_mode == 0).mean())))

off_arr = np.array([(a, b) for a, b, _, _ in res_off])
diag = dict(
    n_pairs_used=len(res_off),
    modal_share_all_mean=float(off_arr[:, 0].mean()) if len(off_arr) else None,
    modal_share_mut_mean=float(off_arr[:, 1].mean()) if len(off_arr) else None,
    dyad_offset_at_mode_median=float(np.median([a for a, _ in res_sum])) if res_sum else None,
    dyad_at_mode_frac=float(np.mean([b for _, b in res_sum])) if res_sum else None,
    runtime_s=round(time.time() - t0, 1))
json.dump(diag, open(D + "dyad_lat_diag.json", "w"), indent=1)
print(json.dumps(diag, indent=1), flush=True)
