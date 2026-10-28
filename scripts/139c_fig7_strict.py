#!/usr/bin/env python3
"""139c_fig7_strict.py — Fig 7 严格版 v2: 全局背景与每对窗内直方图分离 (内存安全)
与 135 面板2 完全同管线(节点集/边/背景定义), 但只对 Nm≥100 的对建 ±10 窗直方图。
输出: data_male/fig7_strict.npz
"""
import sys; sys.path.insert(0, "/mnt/agents/pylibs")
import numpy as np, json, time, gc
import pandas as pd
R = "/mnt/agents/output/果蝇生成器/"
DM, DF = R+"data_male/", R+"data_female/"
t0 = time.time()
def log(*a):
    try: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
    except OSError: pass

B = 40; NB1 = 2*B+1; W = 10
WIN = np.array([(dp+B)*NB1+(dq+B) for dp in range(-W, W+1) for dq in range(-W, W+1)
                if not (dp == 0 and dq == 0)])
NB = NB1*NB1
MIN_E = 100; MIN_OFFCENTER = 0.02; MIN_PEAK = 0.01

def rots(dp, dq):
    return [(dp, dq), (-dq, dp+dq), (-dp-dq, dp), (-dp, -dq), (dq, -dp-dq), (dp+dq, -dq)]
SYMS = [lambda dp, dq, k=k: rots(dp, dq)[k] for k in range(6)] + \
       [lambda dp, dq, k=k: rots(dq, dp)[k] for k in range(6)]
MP_BIN = np.zeros((12, NB), np.int64)   # 同 135: bin 级置换
for b_ in range(NB):
    dp, dq = b_//NB1-B, b_%NB1-B
    for si, S in enumerate(SYMS):
        sp, sq = S(dp, dq)
        if abs(sp) <= B and abs(sq) <= B:
            MP_BIN[si, (sp+B)*NB1+(sq+B)] = b_

def edges_hist10(DX, g_arr, hexs, hems, tpre, tpost):
    """返回 uk, n(每对边数, ±40窗内), h10(uk_keep × 440 lift 用), g(±40全局背景)"""
    ep = np.fromfile(DX+"e_pre.i32", dtype=np.int32).astype(np.int64)
    eq = np.fromfile(DX+"e_post.i32", dtype=np.int32).astype(np.int64)
    gsort = np.sort(g_arr)
    ip_ = np.searchsorted(gsort, ep); iq_ = np.searchsorted(gsort, eq)
    okp = (ip_ < len(gsort)) & (gsort[np.clip(ip_, 0, len(gsort)-1)] == ep)
    okq = (iq_ < len(gsort)) & (gsort[np.clip(iq_, 0, len(gsort)-1)] == eq)
    m = okp & okq
    order = np.argsort(g_arr)
    ip = order[ip_[m]]; iq = order[iq_[m]]
    same = hems[ip] == hems[iq]
    ip, iq = ip[same], iq[same]
    dp = np.rint(hexs[iq, 0]-hexs[ip, 0]).astype(np.int64)
    dq = np.rint(hexs[iq, 1]-hexs[ip, 1]).astype(np.int64)
    tp, tq = tpre[ip], tpost[iq]
    del ep, eq, ip_, iq_, okp, okq, m, order, ip, iq; gc.collect()
    inr40 = (np.abs(dp) <= B) & (np.abs(dq) <= B)
    tp, tq, dp, dq = tp[inr40], tq[inr40], dp[inr40], dq[inr40]
    key = tp*100000 + tq
    bb = (dp+B)*NB1 + (dq+B)
    del tp, tq, dp, dq; gc.collect()
    log(f"  {DX.split('/')[-2]} 窗边 {len(key):,}")
    uk, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv, minlength=len(uk)).astype(np.int64)
    g = np.bincount(bb, minlength=NB).astype(np.float64); g /= g.sum()
    # 只对 n≥MIN_E 的对建 ±10 窗直方图
    keep = np.where(n >= MIN_E)[0]
    kmap = np.full(len(uk), -1, np.int64); kmap[keep] = np.arange(len(keep))
    in10 = np.isin(bb, WIN)
    m2 = in10 & (kmap[inv] >= 0)
    widx = np.searchsorted(WIN, bb[m2])
    h10 = np.bincount(kmap[inv[m2]]*len(WIN) + widx, minlength=len(keep)*len(WIN))\
              .reshape(len(keep), len(WIN)).astype(np.float64)
    del key, inv, bb, widx, m2, in10; gc.collect()
    n_k = n[keep]
    hn = h10/np.maximum(n_k, 1)[:, None]
    lift = hn - g[WIN][None, :]
    return uk[keep], hn, lift, n_k, g

def cheb(pkm, pkf):
    d = np.maximum(np.abs(pkm//NB1 - pkf//NB1), np.abs(pkm%NB1 - pkf%NB1))
    return d <= 1

# ---- 雄性 m2 节点集 (同 135) ----
ann = pd.read_feather(R+"data/body-annotations.feather", columns=["bodyId","type","flywireType"])
type_m_all = ann.type.to_numpy().astype(str); fwt_all = ann.flywireType.to_numpy().astype(str)
fw_map = {}
for t_ in np.unique(type_m_all):
    m = type_m_all == t_
    u, c = np.unique(fwt_all[m], return_counts=True)
    ok_ = u != "None"
    if ok_.sum(): fw_map[t_] = u[ok_][c[ok_].argmax()]
vocab = {t: i+1 for i, t in enumerate(sorted(set(fwt_all))) if t != "None"}
inv_vocab = {v: k for k, v in vocab.items()}
del ann, type_m_all, fwt_all; gc.collect()

zm = np.load(DM+"lat2q_m2.npz"); PQm, HEMm, SRCm = zm["pq"], zm["hem"], zm["src"]
nt_m = np.fromfile(DM+"node_type.i32", dtype=np.int32)
tn_m = np.load(DM+"type_names.npy", allow_pickle=True).astype(str)
tcnt = np.bincount(nt_m)
extra = [t for t in range(len(tcnt)) if 1600 <= tcnt[t] <= 1960]
ann2 = pd.read_feather(R+"data/body-annotations.feather", columns=["type"])
col_ids = sorted({int(t) for t in extra} | {int(np.where(tn_m == t_)[0][0]) for t_ in set(ann2.type.astype(str)) if (tn_m == t_).any()})
del ann2; gc.collect()
m2mask = (SRCm > 0) & np.isin(nt_m, col_ids)
g2 = np.where(m2mask)[0]
hex2 = PQm[g2].astype(np.float64); hem2 = HEMm[g2]
t2s = np.array([fw_map.get(tn_m[t_], "None") for t_ in nt_m[g2]])
okt = t2s != "None"
g2, hex2, hem2, t2s = g2[okt], hex2[okt], hem2[okt], t2s[okt]
t2 = np.array([vocab.get(t, 0) for t in t2s], dtype=np.int64)
okt2 = t2 > 0
g2, hex2, hem2, t2 = g2[okt2], hex2[okt2], hem2[okt2], t2[okt2]
log(f"雄 m2 节点 {len(g2):,}")

ca = pd.read_csv(DF+"column_assignment.csv.gz")
nids_f = np.fromfile(DF+"node_ids.i64", dtype=np.int64)
of = np.argsort(nids_f); sf = nids_f[of]
jf = np.searchsorted(sf, ca.root_id.values)
okf = (jf < len(sf)) & (sf[np.clip(jf, 0, len(sf)-1)] == ca.root_id.values)
gf = of[jf[okf]]
hex_f = np.column_stack([ca.p.values[okf].astype(np.float64), ca.q.values[okf].astype(np.float64)])
hem_f = np.where(ca.hemisphere.values[okf]=="right", 1, -1)
type_f = np.array([vocab.get(t, 0) for t in ca.type.values[okf].astype(str)], dtype=np.int64)
del ca; gc.collect()
log("雄性直方图…")
ukm, HNm, LM, NM, GM = edges_hist10(DM, g2, hex2, hem2, t2, t2)
del hex2, hem2; gc.collect()
log(f"  雄 keep 对 {len(ukm)}")
log("雌性直方图…")
ukf, HNf, LF, NF, GF = edges_hist10(DF, gf, hex_f, hem_f, type_f, type_f)
del hex_f, hem_f; gc.collect()
log(f"  雌 keep 对 {len(ukf)}")

# ---- 资格 (同 135, 全部基于 ±10 窗) ----
shared, im, if_ = np.intersect1d(ukm, ukf, return_indices=True)
Lm, Lf = LM[im], LF[if_]
Hm_, Hf_ = HNm[im], HNf[if_]
Nm, Nf = NM[im], NF[if_]
offm, offf = Hm_.sum(1), Hf_.sum(1)
ph_m = Lm.max(1); ph_f = Lf.max(1)
qual = (Nm >= MIN_E) & (Nf >= MIN_E) & (offm >= MIN_OFFCENTER) & (offf >= MIN_OFFCENTER) \
       & (ph_m >= MIN_PEAK) & (ph_f >= MIN_PEAK)
qi = np.where(qual)[0]
log(f"n_shared {len(shared)} 合格 {len(qi)} (应 225/67)")

# 对称后雌 lift: Lf[:, MW_S] 即 bin 置换在 WIN 子列上等价于 MP_BIN[si] 再取 WIN
def lf_sym(si):
    return Lf[:, np.searchsorted(WIN, 0)] if False else Lf[:, :][:, WINmap_idx(si)]
def WINmap_idx(si):
    # WIN 的 bin 经 MP_BIN[si] 置换后仍在 WIN 内的索引序列 (与 135 的 Lf[:, MP[si]][:, WIN] 一致)
    return np.array([np.where(WIN == MP_BIN[si, b])[0][0] if (WIN == MP_BIN[si, b]).any() else -1 for b in WIN])
WM = {si: WINmap_idx(si) for si in range(12)}
# 校验: 135 语义是 Lf_full[:, MP[si]] 再取 WIN 列 => 元素 Lf_full[MP[si][b]], b∈WIN
# 即 idx_in_WIN = position of MP_BIN[si][b] in WIN

pk_m_bins = WIN[Lm[qi].argmax(1)]
def acc_at(si):
    wm = WM[si]
    ok = wm >= 0
    pkf = wm[Lf[qi].argmax(1)]
    # 注意: argmax 在 wm 重排后的窗上进行, 与 135 的 Lf[:, MP[si]][:, WIN].argmax 一致
    lf_perm = Lf[qi][:, wm]
    pkf = wm[lf_perm.argmax(1)]
    return cheb(pk_m_bins, WIN[pkf]).mean()
rng = np.random.default_rng(0)
pk_mq = pk_m_bins
acc_obs = acc_at(3); acc_id = acc_at(0)
nulls = []
Lfq = Lf[qi]
for rep in range(500):
    prm = rng.permutation(len(qi))
    best = 0.0
    for si in range(12):
        wm = WM[si]
        lf_perm = Lfq[prm][:, wm]
        best = max(best, cheb(pk_mq, WIN[wm[lf_perm.argmax(1)]]).mean())
    nulls.append(best)
nulls = np.array(nulls)
log(f"观测 {acc_obs:.3f} (应 0.687) 恒等 {acc_id:.3f} (应 0.597) null p99 {np.percentile(nulls,99):.3f} (应≈0.672)")

# ---- 代表对热图 ----
CAND_STR = ["C3|Tm4", "Tm4|Tm4", "L5|Mi4", "L5|L5", "Tm20|Tm20", "L5|Mi1", "L2|Tm2", "L1|C2", "C2|C3", "L1|L2"]
CAND = [vocab[a]*100000+vocab[b] for a, b in (p.split("|") for p in CAND_STR)]
wr = WIN//NB1-B+W; wc = WIN%NB1-B+W
wm3 = WM[3]
heat = {}
for pstr, p in zip(CAND_STR, CAND):
    j = np.searchsorted(shared, p)
    if j >= len(shared) or shared[j] != p: continue
    qm = np.where(qi == j)[0]
    if not len(qm): continue
    qpos = qm[0]
    lm10 = Lm[j]; lf10 = Lf[j][wm3]
    pkm = WIN[lm10.argmax()]; pkf = WIN[wm3[lf10.argmax()]]
    hit = bool(cheb(np.array([pkm]), np.array([pkf]))[0])
    lm = np.zeros((2*W+1, 2*W+1)); lm[wr, wc] = lm10
    lf = np.zeros((2*W+1, 2*W+1)); lf[wr, wc] = lf10
    heat[pstr] = (lm, lf, int(Nm[j]), int(Nf[j]), hit,
                  (int(pkm//NB1-B), int(pkm%NB1-B)), (int(pkf//NB1-B), int(pkf%NB1-B)))
    log(f"{pstr}: 雄峰 {heat[pstr][5]} 雌峰(调和) {heat[pstr][6]} 命中 {hit} n_m {Nm[j]:,}")
hits = sorted([p for p in heat if heat[p][4]], key=lambda p: -heat[p][2])[:4]
log(f"选中: {hits}")
out = {"nulls": nulls, "acc_obs": acc_obs, "acc_id": acc_id, "show_names": np.array(hits)}
for k, p in enumerate(hits):
    lm, lf, nm_, nf_, hit, pm, pf = heat[p]
    out[f"heat_{k}_m"] = lm; out[f"heat_{k}_f"] = lf
    out[f"heat_{k}_info"] = np.array([nm_, nf_])
    out[f"heat_{k}_pk"] = np.array([pm[0], pm[1], pf[0], pf[1]])
np.savez(R+"data_male/fig7_strict.npz", **out)
log("DONE")
