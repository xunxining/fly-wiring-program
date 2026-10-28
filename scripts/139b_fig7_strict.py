#!/usr/bin/env python3
"""139b_fig7_strict.py — Fig 7 严格版: 完整复刻 135 面板2 数据路径(字符串键, 105k 节点)
输出: 代表对 lift 热图(命中优先) + best-of-12 完整零分布(500 reps) -> data_male/fig7_strict.npz
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

B = 40; NB1 = 2*B+1; NB = NB1*NB1; CENTER = B*NB1+B; W = 10
WIN = np.array([(dp+B)*NB1+(dq+B) for dp in range(-W, W+1) for dq in range(-W, W+1)
                if not (dp == 0 and dq == 0)])
MIN_E = 100; MIN_OFFCENTER = 0.02; MIN_PEAK = 0.01

def rots(dp, dq):
    return [(dp, dq), (-dq, dp+dq), (-dp-dq, dp), (-dp, -dq), (dq, -dp-dq), (dp+dq, -dq)]
SYMS = [lambda dp, dq, k=k: rots(dp, dq)[k] for k in range(6)] + \
       [lambda dp, dq, k=k: rots(dq, dp)[k] for k in range(6)]
MP = np.zeros((12, NB), np.int64)
for b_ in range(NB):
    dp, dq = b_//NB1-B, b_%NB1-B
    for si, S in enumerate(SYMS):
        sp, sq = S(dp, dq)
        if abs(sp) <= B and abs(sq) <= B:
            MP[si, (sp+B)*NB1+(sq+B)] = b_
MP[:, CENTER] = CENTER

def pair_hists(tpre, tpost, dp, dq):
    inr = (np.abs(dp) <= B) & (np.abs(dq) <= B)
    key = tpre[inr]*100000 + tpost[inr]
    dp, dq = dp[inr], dq[inr]
    uk, inv = np.unique(key, return_inverse=True)
    b = (dp+B)*NB1 + (dq+B)
    h = np.bincount(inv*NB + b, minlength=len(uk)*NB).reshape(len(uk), NB).astype(np.float64)
    n = h.sum(1)
    g = h.sum(0); g /= g.sum()
    hn = h/np.maximum(n, 1)[:, None]
    return uk, hn, hn-g, n.astype(np.int64)

def anchor_edges(DX, g_arr, hexs, hems, types):
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
    return types[ip], types[iq], dp, dq

# ---- 雄性 m2 (同 135 面板2: 全部 flywireType≠None 节点, 字符串键) ----
ann = pd.read_feather(R+"data/body-annotations.feather", columns=["bodyId","type","flywireType"])
type_m_all = ann.type.to_numpy().astype(str); fwt_all = ann.flywireType.to_numpy().astype(str)
fw_map = {}
for t_ in np.unique(type_m_all):
    m = type_m_all == t_
    u, c = np.unique(fwt_all[m], return_counts=True)
    ok_ = u != "None"
    if ok_.sum(): fw_map[t_] = u[ok_][c[ok_].argmax()]
del ann
# 联合词表(保留全部雄性 flywireType, 与 135 字符串键等价编码)
vocab = {t: i+1 for i, t in enumerate(sorted(set(fwt_all) | set()))}
vocab.pop("None", None)
inv_vocab = {v: k for k, v in vocab.items()}
fwt_code = np.array([vocab.get(t, 0) for t in fwt_all], dtype=np.int32)
del type_m_all, fwt_all; gc.collect()

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
del t2s; gc.collect()
log(f"雄 m2 节点 {len(g2):,} (应≈104,869)")

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

log("雄性边…"); a2 = anchor_edges(DM, g2, hex2, hem2, t2)
log("雌性边…"); b1 = anchor_edges(DF, gf, hex_f, hem_f, type_f)
log("雄性直方图…", f"{len(a2[0]):,} 边")
ukm, Hm, Lm, Nm = pair_hists(a2[0], a2[1], a2[2], a2[3])
del a2; gc.collect()
log("雄性直方图完成", len(ukm), "对")
ukf, Hf, Lf, Nf = pair_hists(b1[0], b1[1], b1[2], b1[3])
del b1; gc.collect()
log("雌性直方图完成", len(ukf), "对")
log(f"对数: 雄 {len(ukm)} 雌 {len(ukf)}")

# ---- 资格 (同 135) ----
shared, im, if_ = np.intersect1d(ukm, ukf, return_indices=True)
Hm2, Lm2, Nm2 = Hm[im], Lm[im], Nm[im]
Hf2, Lf2, Nf2 = Hf[if_], Lf[if_], Nf[if_]
offm = Hm2[:, WIN].sum(1); offf = Hf2[:, WIN].sum(1)
LF_all0 = Lf2[:, MP[0]]
pk_m = Lm2[:, WIN].argmax(1)
ph_m = Lm2[:, WIN].max(1); ph_f0 = LF_all0[:, WIN].max(1)
qual = (Nm2 >= MIN_E) & (Nf2 >= MIN_E) & (offm >= MIN_OFFCENTER) & (offf >= MIN_OFFCENTER) \
       & (ph_m >= MIN_PEAK) & (ph_f0 >= MIN_PEAK)
qi = np.where(qual)[0]
log(f"n_shared {len(shared)} 合格 {len(qi)} (应 225/67)")

def cheb(pkm, pkf):
    d = np.maximum(np.abs(pkm//NB1 - pkf//NB1), np.abs(pkm%NB1 - pkf%NB1))
    return d <= 1

# ---- 观测 + 完整零分布 (500 reps, S=3) ----
rng = np.random.default_rng(0)
Lfq = Lf2[qi]; pk_mq = WIN[pk_m[qi]]
pkf_obs = WIN[Lfq[:, MP[3]][:, WIN].argmax(1)]
acc_obs = float(cheb(pk_mq, pkf_obs).mean())
acc_id = float(cheb(pk_mq, WIN[Lfq[:, MP[0]][:, WIN].argmax(1)]).mean())
nulls = []
for rep in range(500):
    prm = rng.permutation(len(qi))
    Lp = Lfq[prm]
    best = max(cheb(pk_mq, WIN[Lp[:, MP[si]][:, WIN].argmax(1)]).mean() for si in range(12))
    nulls.append(best)
nulls = np.array(nulls)
log(f"观测 {acc_obs:.3f} (应 0.687) 恒等 {acc_id:.3f} (应 0.597) null p99 {np.percentile(nulls,99):.3f} (应≈0.672)")

# ---- 代表对热图: 候选大对, 算命中性, 挑 4 个命中 ----
CAND_STR = ["C3|Tm4", "Tm4|Tm4", "L5|Mi4", "L5|L5", "Tm20|Tm20", "L5|Mi1", "L2|Tm2", "L1|C2", "C2|C3", "L1|L2"]
CAND = [vocab[a]*100000+vocab[b] for a, b in (p.split("|") for p in CAND_STR)]
cname = {v: s for v, s in zip(CAND, CAND_STR)}
wr = WIN//NB1-B+W; wc = WIN%NB1-B+W
heat = {}
for p in CAND:
    j = np.searchsorted(shared, p)
    if j >= len(shared) or shared[j] != p: continue
    if j not in set(qi.tolist()): continue
    qpos = np.where(qi == j)[0][0]
    pkm_b = pk_mq[qpos]; pkf_b = pkf_obs[qpos]
    hit = bool(cheb(np.array([pkm_b]), np.array([pkf_b]))[0])
    i_m, i_f = im[j], if_[j]
    lm = np.zeros((2*W+1, 2*W+1)); lm[wr, wc] = Lm[i_m, WIN]
    lf = np.zeros((2*W+1, 2*W+1)); lf[wr, wc] = Lf2[j, MP[3]][WIN]
    heat[cname[p]] = (lm, lf, int(Nm[i_m]), int(Nf[i_f]), hit,
               (int(pkm_b//NB1-B), int(pkm_b%NB1-B)), (int(pkf_b//NB1-B), int(pkf_b%NB1-B)))
    log(f"{cname[p]}: 雄峰 {heat[cname[p]][5]} 雌峰(调和) {heat[cname[p]][6]} 命中 {hit} n_m {Nm[i_m]:,}")

# 挑 4 个命中且 n 最大
hits = sorted([cname[p] for p in CAND if cname[p] in heat and heat[cname[p]][4]], key=lambda p: -heat[p][2])[:4]
log(f"选中: {hits}")
out = {"nulls": nulls, "acc_obs": acc_obs, "acc_id": acc_id,
       "show_names": np.array(hits)}
for k, p in enumerate(hits):
    lm, lf, nm_, nf_, hit, pm, pf = heat[p]
    out[f"heat_{k}_m"] = lm; out[f"heat_{k}_f"] = lf
    out[f"heat_{k}_info"] = np.array([nm_, nf_])
    out[f"heat_{k}_pk"] = np.array([pm[0], pm[1], pf[0], pf[1]])
np.savez(R+"data_male/fig7_strict.npz", **out)
log("DONE")
