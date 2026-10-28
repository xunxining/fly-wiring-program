#!/usr/bin/env python3
"""132_kernel_conserve2.py — 跨性别装配核守恒检验 v2 (§40d): 偏移直方图相关
主检验: 逐对偏移直方图(81×81)跨性别 Pearson r, 剔除 (0,0) bin; 合格对: 非中心质量≥2% 且边数≥100
门: 中位 r ≥ 0.7 且 ≫ 洗牌基线(1000 次置换)
次级: 含中心 r_full; 非零模态对一致率; 峰度对照
面板1 金-金(两性注释锚点), 面板2 扩展(雌锚 × 雄 m2 柱状)
输出: data_male/kernel_conserve2_m2.json + kernel_conserve2_pairs.csv
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

B = 40; NB1 = 2*B+1; NB = NB1*NB1
CENTER = B*NB1 + B
MIN_E = 100; MIN_OFFCENTER = 0.02

def pair_hists(tpre, tpost, dp, dq):
    inr = (np.abs(dp) <= B) & (np.abs(dq) <= B)
    key = np.array([a+"|"+b for a, b in zip(tpre, tpost)])
    key, dp, dq = key[inr], dp[inr], dq[inr]
    uk, inv = np.unique(key, return_inverse=True)
    b = (dp+B)*NB1 + (dq+B)
    h = np.bincount(inv*NB + b, minlength=len(uk)*NB).reshape(len(uk), NB).astype(np.float64)
    n = h.sum(1)
    keep = n >= MIN_E
    uk, h, n = uk[keep], h[keep], n[keep]
    hn = h/n[:, None]
    return {k_: (hn[i], int(n[i])) for i, k_ in enumerate(uk)}

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

def corr_panels(M, F, label):
    shared = sorted(set(M) & set(F))
    rows, rs_off, rs_full, quals = [], [], [], []
    for k_ in shared:
        hm, nm = M[k_]; hf, nf = F[k_]
        offm = 1-hm[CENTER]; offf = 1-hf[CENTER]
        r_full = float(np.corrcoef(hm, hf)[0, 1])
        mo = np.concatenate([hm[:CENTER], hm[CENTER+1:]])
        fo = np.concatenate([hf[:CENTER], hf[CENTER+1:]])
        r_off = float(np.corrcoef(mo, fo)[0, 1]) if mo.std() > 0 and fo.std() > 0 else np.nan
        q = (offm >= MIN_OFFCENTER) and (offf >= MIN_OFFCENTER)
        quals.append(q)
        rows.append({"pair": k_, "panel": label, "n_male": nm, "n_female": nf,
                     "offcenter_male": float(offm), "offcenter_female": float(offf),
                     "r_off": r_off, "r_full": r_full,
                     "mode_male": f"{int(hm.argmax()//NB1-B)},{int(hm.argmax()%NB1-B)}",
                     "mode_female": f"{int(hf.argmax()//NB1-B)},{int(hf.argmax()%NB1-B)}"})
        rs_full.append(r_full); rs_off.append(r_off)
    quals = np.array(quals); rs_off = np.array(rs_off); rs_full = np.array(rs_full)
    med_off = float(np.nanmedian(rs_off[quals])) if quals.sum() else np.nan
    med_full = float(np.nanmedian(rs_full)) if len(rs_full) else np.nan
    # 洗牌基线 (合格对子集上置换雌性直方图)
    rng = np.random.default_rng(0)
    qidx = np.where(quals)[0]
    base = []
    Ms = [np.concatenate([M[k_][0][:CENTER], M[k_][0][CENTER+1:]]) for k_ in shared]
    Fs = [np.concatenate([F[k_][0][:CENTER], F[k_][0][CENTER+1:]]) for k_ in shared]
    for rep in range(1000):
        prm = rng.permutation(len(shared))
        rr = []
        for i in qidx:
            a, b_ = Ms[i], Fs[prm[i]]
            if a.std() > 0 and b_.std() > 0:
                rr.append(np.corrcoef(a, b_)[0, 1])
        base.append(np.median(rr))
    base = np.array(base)
    log(f"面板{label}: 共享对 {len(shared)}, 合格 {quals.sum()}, 中位 r_off={med_off:.3f} "
        f"r_full={med_full:.3f} (洗牌 {base.mean():.3f}±{base.std():.3f})")
    return rows, {"n_shared": len(shared), "n_qual": int(quals.sum()),
                  "median_r_off": med_off, "median_r_full": med_full,
                  "shuffle_median_mean": float(base.mean()), "shuffle_median_std": float(base.std()),
                  "shuffle_p99": float(np.percentile(base, 99)),
                  "verdict": ("PASS" if med_off >= 0.7 and med_off > float(np.percentile(base, 99)) else
                              "PARTIAL" if med_off > float(np.percentile(base, 99)) else "FAIL")}

# ---- 数据加载 (同 131) ----
ann = pd.read_feather(R+"data/body-annotations.feather",
                      columns=["bodyId","assignedOlHex1","assignedOlHex2","somaSide","type","flywireType"])
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
fwt_m = ann.flywireType.to_numpy()[has][km].astype(str)
fw_map = {}
for t_ in np.unique(type_m):
    m = type_m == t_
    u, c = np.unique(fwt_m[m], return_counts=True)
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
type_f = ca.type.values[okf].astype(str)
log(f"锚点: 雄 {len(gm):,} / 雌 {len(gf):,}")

log("面板1 雄性…"); a1 = anchor_edges(DM, gm, hex_m, hem_m, type_m)
M1 = pair_hists(np.array([fw_map.get(x, "None") for x in a1[0]]), a1[1], a1[2], a1[3])
log("面板1 雌性…"); b1 = anchor_edges(DF, gf, hex_f, hem_f, type_f)
F1 = pair_hists(b1[0], b1[1], b1[2], b1[3])
rows1, res1 = corr_panels(M1, F1, 1)

# ---- 面板2: 雌锚 × 雄 m2 柱状 ----
zm = np.load(DM+"lat2q_m2.npz"); PQm, HEMm, SRCm = zm["pq"], zm["hem"], zm["src"]
nt_m = np.fromfile(DM+"node_type.i32", dtype=np.int32)
tn_m = np.load(DM+"type_names.npy", allow_pickle=True).astype(str)
tcnt = np.bincount(nt_m)
extra = [t for t in range(len(tcnt)) if 1600 <= tcnt[t] <= 1960]
col_ids = sorted({int(t) for t in extra} | {int(np.where(tn_m == t_)[0][0]) for t_ in set(type_m) if (tn_m == t_).any()})
m2mask = (SRCm > 0) & np.isin(nt_m, col_ids)
g2 = np.where(m2mask)[0]
hex2 = PQm[g2].astype(np.float64); hem2 = HEMm[g2]
t2 = np.array([fw_map.get(tn_m[t_], "None") for t_ in nt_m[g2]])
okt = t2 != "None"
g2, hex2, hem2, t2 = g2[okt], hex2[okt], hem2[okt], t2[okt]
log("面板2 雄性(m2)…"); a2 = anchor_edges(DM, g2, hex2, hem2, t2)
M2 = pair_hists(a2[0], a2[1], a2[2], a2[3])
rows2, res2 = corr_panels(M2, F1, 2)

pd.DataFrame(rows1+rows2).to_csv(DM+"kernel_conserve2_pairs.csv", index=False)
out = {"metric": "per-pair offset-histogram Pearson r, center bin excluded (primary) / included (r_full)",
       "MIN_E": MIN_E, "MIN_OFFCENTER": MIN_OFFCENTER, "offset_window": B,
       "gate": "median r_off >= 0.7 AND > shuffle p99",
       "panel1_gold": res1, "panel2_ext": res2}
json.dump(out, open(DM+"kernel_conserve2_m2.json", "w"), indent=1)
log(json.dumps(out, indent=1))
log("DONE")
