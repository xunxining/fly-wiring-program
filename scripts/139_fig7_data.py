#!/usr/bin/env python3
"""139_fig7_data.py — Fig 7 数据: 代表对 lift 热图 + 面板2 best-of-12 完整零分布"""
import sys; sys.path.insert(0, "/mnt/agents/pylibs")
import numpy as np, json, time
import pandas as pd
R = "/mnt/agents/output/果蝇生成器/"
DM, DF = R+"data_male/", R+"data_female/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

B = 40; NB1 = 2*B+1; NB = NB1*NB1; W = 10
WIN = np.array([(dp+B)*NB1+(dq+B) for dp in range(-W, W+1) for dq in range(-W, W+1)
                if not (dp == 0 and dq == 0)])

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

# 雄性 m2 锚点集 (面板2)
ann = pd.read_feather(R+"data/body-annotations.feather", columns=["bodyId","type","flywireType"])
nt_m = np.fromfile(DM+"node_type.i32", dtype=np.int32)
tn_m = np.load(DM+"type_names.npy", allow_pickle=True).astype(str)
type_m_all = ann.type.to_numpy().astype(str); fwt_all = ann.flywireType.to_numpy().astype(str)
fw_map = {}
for t_ in np.unique(type_m_all):
    m = type_m_all == t_
    u, c = np.unique(fwt_all[m], return_counts=True)
    ok_ = u != "None"
    if ok_.sum(): fw_map[t_] = u[ok_][c[ok_].argmax()]

ca = pd.read_csv(DF+"column_assignment.csv.gz")
nids_f = np.fromfile(DF+"node_ids.i64", dtype=np.int64)
of = np.argsort(nids_f); sf = nids_f[of]
jf = np.searchsorted(sf, ca.root_id.values)
okf = (jf < len(sf)) & (sf[np.clip(jf, 0, len(sf)-1)] == ca.root_id.values)
gf = of[jf[okf]]
hex_f = np.column_stack([ca.p.values[okf].astype(np.float64), ca.q.values[okf].astype(np.float64)])
hem_f = np.where(ca.hemisphere.values[okf]=="right", 1, -1)
type_f_raw = ca.type.values[okf].astype(str)
tcode_f = {t: i+1 for i, t in enumerate(np.unique(type_f_raw))}
type_f = np.array([tcode_f[t] for t in type_f_raw], dtype=np.int64)
inv_code = {v: k for k, v in tcode_f.items()}


zm = np.load(DM+"lat2q_m2.npz"); PQm, HEMm, SRCm = zm["pq"], zm["hem"], zm["src"]
tcnt = np.bincount(nt_m)
extra = [t for t in range(len(tcnt)) if 1600 <= tcnt[t] <= 1960]
col_ids = sorted({int(t) for t in extra} | {int(np.where(tn_m == t_)[0][0]) for t_ in set(type_m_all) if (tn_m == t_).any()})
m2mask = (SRCm > 0) & np.isin(nt_m, col_ids)
g2 = np.where(m2mask)[0]
hex2 = PQm[g2].astype(np.float64); hem2 = HEMm[g2]
t2s = np.array([fw_map.get(tn_m[t_], "None") for t_ in nt_m[g2]])
okt = t2s != "None"
g2, hex2, hem2, t2s = g2[okt], hex2[okt], hem2[okt], t2s[okt]
t2 = np.array([tcode_f.get(t, 0) for t in t2s], dtype=np.int64)
okt2 = t2 > 0
g2, hex2, hem2, t2 = g2[okt2], hex2[okt2], hem2[okt2], t2[okt2]

log("雄性边…"); a2 = anchor_edges(DM, g2, hex2, hem2, t2)
log("雌性边…"); b1 = anchor_edges(DF, gf, hex_f, hem_f, type_f)
ukm, Hm, Lm, Nm = pair_hists(a2[0], a2[1], a2[2], a2[3])
ukf, Hf, Lf, Nf = pair_hists(b1[0], b1[1], b1[2], b1[3])
del a2, b1
log(f"对数: 雄 {len(ukm)} 雌 {len(ukf)}")

# ---- 代表对 lift 热图 (±10 窗, 雌经 S=3 调和) ----
SHOW_STR = ["C3|Tm4", "L2|Tm4", "L2|Tm2", "L1|C2"]
SHOW = [tcode_f[a]*100000 + tcode_f[b] for a, b in (p.split("|") for p in SHOW_STR)]
S = 3
heat = {}
for p in SHOW:
    im = np.searchsorted(ukm, p); if_ = np.searchsorted(ukf, p)
    if im >= len(ukm) or ukm[im] != p or if_ >= len(ukf) or ukf[if_] != p:
        log(f"skip {p}"); continue
    wr = WIN//NB1-B+W; wc = WIN%NB1-B+W
    lm = np.zeros((2*W+1, 2*W+1)); lm[wr, wc] = Lm[im, WIN]
    lf = np.zeros((2*W+1, 2*W+1)); lf[wr, wc] = Lf[if_, MP[S]][WIN]
    pname = inv_code[p//100000] + "|" + inv_code[p%100000]
    heat[pname] = (lm, lf, int(Nm[im]), int(Nf[if_]))
    log(f"{pname}: 雄峰 {WIN[Lm[im,WIN].argmax()]//NB1-B},{WIN[Lm[im,WIN].argmax()]%NB1-B}  "
        f"雌峰(调和) {WIN[Lf[if_,MP[S]][WIN].argmax()]//NB1-B},{WIN[Lf[if_,MP[S]][WIN].argmax()]%NB1-B}")

# ---- 面板2 best-of-12 完整零分布 (与 135 同资格) ----
MIN_E = 100; MIN_OFFCENTER = 0.02; MIN_PEAK = 0.01
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
log(f"合格对 {len(qi)} (应=67)")
def cheb(pkm, pkf):
    d = np.maximum(np.abs(pkm//NB1 - pkf//NB1), np.abs(pkm%NB1 - pkf%NB1))
    return d <= 1
rng = np.random.default_rng(0)
Lfq = Lf2[qi]; pk_mq = WIN[pk_m[qi]]
nulls = []
for rep in range(500):
    prm = rng.permutation(len(qi))
    Lp = Lfq[prm]
    best = max(cheb(pk_mq, WIN[Lp[:, MP[si]][:, WIN].argmax(1)]).mean() for si in range(12))
    nulls.append(best)
nulls = np.array(nulls)
pkf_obs = WIN[Lfq[:, MP[3]][:, WIN].argmax(1)]
acc_obs = float(cheb(pk_mq, pkf_obs).mean())
acc_id = float(cheb(pk_mq, WIN[Lfq[:, MP[0]][:, WIN].argmax(1)]).mean())
log(f"观测 {acc_obs:.3f} 恒等 {acc_id:.3f} null p99 {np.percentile(nulls,99):.3f}")

np.savez(R+"data_male/fig7_data.npz",
         **{f"heat_{i}_{k}": v for i, (p, (lm, lf, nm_, nf_)) in enumerate(heat.items())
            for k, v in (("m", lm), ("f", lf))},
         show_names=np.array(list(heat.keys())),
         show_counts=np.array([[v[2], v[3]] for v in heat.values()]),
         nulls=nulls, acc_obs=acc_obs, acc_id=acc_id)
log("DONE")
