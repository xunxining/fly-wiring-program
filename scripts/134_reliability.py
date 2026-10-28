#!/usr/bin/env python3
"""134_reliability.py — M2 信度对照 (§40e 执行附件): 同性别边对半分的峰一致率(噪声天花板)
对每性别独立: rng 对半分边 → 各半 lift 峰位 → 同一组合格对上的一致率。
解释: 跨性别一致率 ≈ 同性别信度 ⇒ 守恒至噪声天花板; 显著低于 ⇒ 真性别差异。
输出: data_male/kernel_reliability.json
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
WIN = []
for dp in range(-W, W+1):
    for dq in range(-W, W+1):
        if dp == 0 and dq == 0: continue
        WIN.append((dp+B)*NB1+(dq+B))
WIN = np.array(WIN)
MIN_E = 100; MIN_OFFCENTER = 0.02; MIN_PEAK = 0.01

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

def half_peaks(tpre, tpost, dp, dq, half):
    inr = (np.abs(dp[half]) <= B) & (np.abs(dq[half]) <= B)
    key = np.array([a+"|"+b for a, b in zip(tpre[half][inr], tpost[half][inr])])
    dph, dqh = dp[half][inr], dq[half][inr]
    uk, inv = np.unique(key, return_inverse=True)
    b = (dph+B)*NB1 + (dqh+B)
    h = np.bincount(inv*NB + b, minlength=len(uk)*NB).reshape(len(uk), NB).astype(np.float64)
    n = h.sum(1)
    g = h.sum(0); g /= g.sum()
    hn = h/np.maximum(n, 1)[:, None]
    lift = hn - g
    out = {}
    for i, k_ in enumerate(uk):
        lw = lift[i][WIN]
        out[k_] = (int(lw.argmax()), float(lw.max()), float(hn[i][WIN].sum()), int(n[i]))
    return out

def reliability(tpre, tpost, dp, dq, label, n_rep=5):
    accs = []
    for rep in range(n_rep):
        rng = np.random.default_rng(rep)
        half = rng.random(len(tpre)) < 0.5
        P1 = half_peaks(tpre, tpost, dp, dq, half)
        P2 = half_peaks(tpre, tpost, dp, dq, ~half)
        shared = sorted(set(P1) & set(P2))
        qs = []
        for k_ in shared:
            a1, a2 = P1[k_], P2[k_]
            if min(a1[3], a2[3]) < MIN_E//2: continue
            if min(a1[2], a2[2]) < MIN_OFFCENTER: continue
            if min(a1[1], a2[1]) < MIN_PEAK: continue
            b1, b2 = WIN[a1[0]], WIN[a2[0]]
            cheb = max(abs(b1//NB1-B - (b2//NB1-B)), abs(b1%NB1-B - (b2%NB1-B)))
            qs.append(cheb <= 1)
        accs.append((float(np.mean(qs)), len(qs)))
        log(f"  {label} rep{rep}: 两半峰一致率 {accs[-1][0]:.3f} (合格 {accs[-1][1]})")
    return float(np.mean([a for a, _ in accs])), int(np.mean([n for _, n in accs]))

# 雄性锚点
ann = pd.read_feather(R+"data/body-annotations.feather",
                      columns=["bodyId","assignedOlHex1","assignedOlHex2","somaSide","type"])
nids = np.fromfile(DM+"node_ids.i64", dtype=np.int64)
om = np.argsort(nids); sm = nids[om]
jm = np.searchsorted(sm, ann.bodyId.values)
okm = sm[np.clip(jm, 0, len(sm)-1)] == ann.bodyId.values
gm_all = np.full(len(ann), -1, np.int64); gm_all[okm] = om[jm[okm]]
has = ann.assignedOlHex1.notna().to_numpy() & ann.assignedOlHex2.notna().to_numpy() & (gm_all >= 0)
gm = gm_all[has]
hem_m = np.where(ann.somaSide.to_numpy()[has]=="R", 1, np.where(ann.somaSide.to_numpy()[has]=="L", -1, 0))
km = hem_m != 0; gm, hem_m = gm[km], hem_m[km]
hex_m = np.column_stack([ann.assignedOlHex1.to_numpy()[has][km].astype(np.float64),
                         ann.assignedOlHex2.to_numpy()[has][km].astype(np.float64)])
type_m = ann.type.to_numpy()[has][km].astype(str)
am = anchor_edges(DM, gm, hex_m, hem_m, type_m)
log(f"雄性锚点边 {len(am[0]):,}")
acc_m, n_m = reliability(am[0], am[1], am[2], am[3], "雄性")

ca = pd.read_csv(DF+"column_assignment.csv.gz")
nids_f = np.fromfile(DF+"node_ids.i64", dtype=np.int64)
of = np.argsort(nids_f); sf = nids_f[of]
jf = np.searchsorted(sf, ca.root_id.values)
okf = (jf < len(sf)) & (sf[np.clip(jf, 0, len(sf)-1)] == ca.root_id.values)
gf = of[jf[okf]]
hex_f = np.column_stack([ca.p.values[okf].astype(np.float64), ca.q.values[okf].astype(np.float64)])
hem_f = np.where(ca.hemisphere.values[okf]=="right", 1, -1)
type_f = ca.type.values[okf].astype(str)
af = anchor_edges(DF, gf, hex_f, hem_f, type_f)
log(f"雌性锚点边 {len(af[0]):,}")
acc_f, n_f = reliability(af[0], af[1], af[2], af[3], "雌性")

out = {"male_within_sex_peak_agreement": acc_m, "male_n_qual_avg": n_m,
       "female_within_sex_peak_agreement": acc_f, "female_n_qual_avg": n_f,
       "cross_sex_ref": {"panel1": 0.483, "panel2": 0.597},
       "interpretation": "跨性别 ≈ 同性别信度 ⇒ 守恒至噪声天花板; 显著低于 ⇒ 真差异"}
json.dump(out, open(DM+"kernel_reliability.json", "w"), indent=1)
log(json.dumps(out, indent=1))
log("DONE")
