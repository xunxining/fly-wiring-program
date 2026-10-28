#!/usr/bin/env python3
"""136b_kernel_hemi2.py — 半球条件变换调和的终审统计 (§40g)
对全局合格大对(MIN_E=100)拆半球(各≥50边), R 用 S=6(轴交换反射)/L 用 S=3(R180)
(S_R/S_L 均由面板1选定), 在面板2(m2)评定合并峰一致率;
零分布: 逐半球配对洗牌 × 逐半球 best-of-12, 500 reps。
输出: data_male/kernel_hemi2_m2.json
"""
import sys; sys.path.insert(0, "/mnt/agents/pylibs")
import numpy as np, json, time
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
MIN_E = 100; MIN_HEMI = int(__import__("os").environ.get("MINH", "50")); MIN_OFFCENTER = 0.02; MIN_PEAK = 0.01
S_R, S_L = 6, 3   # 面板1逐半球 argmax (136)

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
    return types[ip], types[iq], dp, dq, hems[ip]

def cheb_agree(pkm, pkf):
    d = np.maximum(np.abs(pkm//NB1 - pkf//NB1), np.abs(pkm%NB1 - pkf%NB1))
    return d <= 1

# ---- 数据加载 ----
ann = pd.read_feather(R+"data/body-annotations.feather",
                      columns=["bodyId","somaSide","type","flywireType"])
nids = np.fromfile(DM+"node_ids.i64", dtype=np.int64)
om = np.argsort(nids); sm = nids[om]
jm = np.searchsorted(sm, ann.bodyId.values)
okm = sm[np.clip(jm, 0, len(sm)-1)] == ann.bodyId.values
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

zm = np.load(DM+"lat2q_m2.npz"); PQm, HEMm, SRCm = zm["pq"], zm["hem"], zm["src"]
nt_m = np.fromfile(DM+"node_type.i32", dtype=np.int32)
tn_m = np.load(DM+"type_names.npy", allow_pickle=True).astype(str)
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
log(f"m2 节点 {len(g2):,}")

log("雄性(m2)边…"); a2 = anchor_edges(DM, g2, hex2, hem2, t2)
log("雌性边…");     b1 = anchor_edges(DF, gf, hex_f, hem_f, type_f)

# 全局直方图 (资格用) + 分半球直方图 (评定用)
ukmg, Hmg, Lmg, Nmg = pair_hists(a2[0], a2[1], a2[2], a2[3])
ukfg, Hfg, Lfg, Nfg = pair_hists(b1[0], b1[1], b1[2], b1[3])
hemi = {}
for hv, name in ((1, "R"), (-1, "L")):
    m_ = a2[4] == hv; f_ = b1[4] == hv
    hemi[name] = (pair_hists(a2[0][m_], a2[1][m_], a2[2][m_], a2[3][m_]),
                  pair_hists(b1[0][f_], b1[1][f_], b1[2][f_], b1[3][f_]))

# 全局合格对
shared, im, if_ = np.intersect1d(ukmg, ukfg, return_indices=True)
Nm_g, Nf_g = Nmg[im], Nfg[if_]
Hm_g, Hf_g = Hmg[im], Hfg[if_]
offm = Hm_g[:, WIN].sum(1); offf = Hf_g[:, WIN].sum(1)
qual = (Nm_g >= MIN_E) & (Nf_g >= MIN_E) & (offm >= MIN_OFFCENTER) & (offf >= MIN_OFFCENTER)
qset = set(shared[qual].tolist())
log(f"全局合格对 {len(qset)}")

# 逐半球合格 (全局合格 ∩ 半球边数达标)
sel = {}   # name -> (pairs, pk_m_bins, LF_all[12,n,NB] of female lift)
for name in ("R", "L"):
    (ukm, Hm, Lm, Nm), (ukf, Hf, Lf, Nf) = hemi[name]
    sh, im2, if2 = np.intersect1d(ukm, ukf, return_indices=True)
    keep = np.array([p in qset for p in sh]) & (Nm[im2] >= MIN_HEMI) & (Nf[if2] >= MIN_HEMI)
    sh, im2, if2 = sh[keep], im2[keep], if2[keep]
    Lm_, Lf_ = Lm[im2], Lf[if2]
    ph_m = Lm_[:, WIN].max(1); ph_f = Lf_[:, WIN].max(1)
    kp = (ph_m >= MIN_PEAK) & (ph_f >= MIN_PEAK)
    sh, Lm_, Lf_ = sh[kp], Lm_[kp], Lf_[kp]
    pk_m = WIN[Lm_[:, WIN].argmax(1)]
    LF_all = np.stack([Lf_[:, MP[si]] for si in range(12)])
    sel[name] = (sh, pk_m, LF_all)
    log(f"半球 {name}: 双合格对 {len(sh)}")

common = np.intersect1d(sel["R"][0], sel["L"][0])
log(f"两半球均合格: {len(common)} 对")

def acc_for(name, pairs, pk_m, LF_all, S, use_pairs):
    idx = np.where(np.isin(pairs, use_pairs))[0]
    pkf = WIN[LF_all[S][idx][:, WIN].argmax(1)]
    return cheb_agree(pk_m[idx], pkf), idx

hitR, iR = acc_for("R", *sel["R"], S_R, common)
hitL, iL = acc_for("L", *sel["L"], S_L, common)
acc_obs = float(np.concatenate([hitR, hitL]).mean())
log(f"观测合并一致率 (R@S6 + L@S3): {acc_obs:.3f}  (R {hitR.mean():.3f} / L {hitL.mean():.3f}, 各 {len(common)} 对)")

# 零分布: 逐半球配对洗牌 × 逐半球 best-of-12
rng = np.random.default_rng(0)
nulls = []
pR, pkR, LFR = sel["R"]; pL, pkL, LFL = sel["L"]
LFRc = LFR[:, iR]; LFLc = LFL[:, iL]; pkRc, pkLc = pkR[iR], pkL[iL]
for rep in range(500):
    prmR = rng.permutation(len(common)); prmL = rng.permutation(len(common))
    bestR = max(cheb_agree(pkRc, WIN[LFRc[si][prmR][:, WIN].argmax(1)]).mean() for si in range(12))
    bestL = max(cheb_agree(pkLc, WIN[LFLc[si][prmL][:, WIN].argmax(1)]).mean() for si in range(12))
    nulls.append((bestR+bestL)/2)
nulls = np.array(nulls)
p99 = float(np.percentile(nulls, 99))
pval = float((nulls >= acc_obs).mean())
log(f"零分布 best-of-12×2半球: 均值 {nulls.mean():.3f} p99 {p99:.3f}; 观测 {acc_obs:.3f} → p≈{pval:.3f}")

inv_code = {v: k for k, v in tcode_f.items()}
def pname(k): return f"{inv_code.get(k//100000,'?')}|{inv_code.get(k%100000,'?')}"
json.dump({"protocol": "S_R=6/S_L=3 由面板1(136)选定; 评定在面板2(m2)全局合格大对的双半球子集; 零=逐半球洗牌×best-of-12 (500 reps)",
           "n_common_pairs": int(len(common)),
           "acc_R_at_S6": float(hitR.mean()), "acc_L_at_S3": float(hitL.mean()),
           "acc_merged": acc_obs, "null_mean": float(nulls.mean()), "null_p99": p99,
           "p_value_approx": pval,
           "pairs": [pname(int(k)) for k in common]},
          open(DM+"kernel_hemi2_m2.json", "w"), indent=1)
log("DONE")
