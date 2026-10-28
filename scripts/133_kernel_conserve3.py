#!/usr/bin/env python3
"""133_kernel_conserve3.py — 跨性别装配核守恒检验 v3 (§40e): lift 峰 + lift 相关
lift = 对直方图 − 该性别全局背景(同面板锚点边全局偏移直方图)
M2-peak 主门: 合格对(n≥100, ±10窗非中心质量≥2%, 两性 lift 峰高≥1%)峰位 Cheb≤1 一致率 ≥60%
M2-lift 次门: ±10 窗 lift 图 Pearson r 中位 ≥0.4 且 > 洗牌 p99
输出: data_male/kernel_conserve3_m2.json + kernel_conserve3_pairs.csv
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
W = 10                                   # lift 分析窗 ±10 hex
WIN = []                                 # 窗内 bin 索引(除中心)
for dp in range(-W, W+1):
    for dq in range(-W, W+1):
        if dp == 0 and dq == 0: continue
        WIN.append((dp+B)*NB1 + (dq+B))
WIN = np.array(WIN)
MIN_E = 100; MIN_OFFCENTER = 0.02; MIN_PEAK = 0.01

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
    g = h.sum(0); g /= g.sum()             # 全局背景
    hn = h/n[:, None]
    return {k_: (hn[i], hn[i]-g, int(n[i])) for i, k_ in enumerate(uk)}, g

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

def lift_analysis(M, F, label):
    shared = sorted(set(M) & set(F))
    rows, qual = [], []
    for k_ in shared:
        hm, lm, nm = M[k_]; hf, lf, nf = F[k_]
        lmw, lfw = lm[WIN], lf[WIN]
        pm, pf = int(lmw.argmax()), int(lfw.argmax())
        hm_, hf_ = float(lmw[pm]), float(lfw[pf])
        offm, offf = float(hm[WIN].sum()), float(hf[WIN].sum())
        q = (offm >= MIN_OFFCENTER) and (offf >= MIN_OFFCENTER) and (hm_ >= MIN_PEAK) and (hf_ >= MIN_PEAK)
        r_lift = float(np.corrcoef(lmw, lfw)[0, 1]) if lmw.std() > 0 and lfw.std() > 0 else np.nan
        bpm, bpf = WIN[pm], WIN[pf]
        dpm, dqm = bpm//NB1-B, bpm%NB1-B
        dpf, dqf = bpf//NB1-B, bpf%NB1-B
        cheb = max(abs(dpm-dpf), abs(dqm-dqf))
        qual.append(q)
        rows.append({"pair": k_, "panel": label, "n_male": nm, "n_female": nf,
                     "peak_male": f"{dpm},{dqm}", "peak_female": f"{dpf},{dqf}",
                     "peak_h_male": hm_, "peak_h_female": hf_, "cheb": cheb, "r_lift": r_lift,
                     "qualified": bool(q)})
    qual = np.array(qual)
    qs = [r_ for r_, q_ in zip(rows, qual) if q_]
    agree = np.array([r_["cheb"] <= 1 for r_ in qs])
    acc = float(agree.mean()) if len(qs) else np.nan
    r_lifts = np.array([r_["r_lift"] for r_ in qs])
    med_lift = float(np.nanmedian(r_lifts)) if len(qs) else np.nan
    # 洗牌基线
    rng = np.random.default_rng(0)
    Lm = [M[k_][1][WIN] for k_ in shared]
    Lf = [F[k_][1][WIN] for k_ in shared]
    qidx = np.where(qual)[0]
    base_acc, base_r = [], []
    for rep in range(1000):
        prm = rng.permutation(len(shared))
        aa, rr = [], []
        for i in qidx:
            a, b_ = Lm[i], Lf[prm[i]]
            ba, bb = WIN[a.argmax()], WIN[b_.argmax()]
            dpa, dqa = ba//NB1 - B, ba % NB1 - B
            dpb, dqb = bb//NB1 - B, bb % NB1 - B
            aa.append(max(abs(dpa-dpb), abs(dqa-dqb)) <= 1)
            if a.std() > 0 and b_.std() > 0:
                rr.append(np.corrcoef(a, b_)[0, 1])
        base_acc.append(np.mean(aa)); base_r.append(np.median(rr))
    base_acc, base_r = np.array(base_acc), np.array(base_r)
    log(f"面板{label}: 共享 {len(shared)}, 合格 {qual.sum()}, 峰一致率 {acc:.3f} (基线 {base_acc.mean():.3f}±{base_acc.std():.3f}), "
        f"lift-r 中位 {med_lift:.3f} (基线 {base_r.mean():.3f}±{base_r.std():.3f})")
    res = {"n_shared": len(shared), "n_qual": int(qual.sum()),
           "peak_agreement": acc, "peak_shuffle_mean": float(base_acc.mean()),
           "peak_shuffle_p99": float(np.percentile(base_acc, 99)),
           "median_r_lift": med_lift, "lift_shuffle_mean": float(base_r.mean()),
           "lift_shuffle_p99": float(np.percentile(base_r, 99)),
           "gate_peak": "PASS" if (acc >= 0.60 and acc > float(np.percentile(base_acc, 99))) else "FAIL",
           "gate_lift": "PASS" if (med_lift >= 0.4 and med_lift > float(np.percentile(base_r, 99))) else "FAIL"}
    return rows, res

# ---- 数据加载 (同 131/132) ----
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
M1, gm_glob = pair_hists(np.array([fw_map.get(x, "None") for x in a1[0]]), a1[1], a1[2], a1[3])
log("面板1 雌性…"); b1 = anchor_edges(DF, gf, hex_f, hem_f, type_f)
F1, gf_glob = pair_hists(b1[0], b1[1], b1[2], b1[3])
rows1, res1 = lift_analysis(M1, F1, 1)

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
M2d, _ = pair_hists(a2[0], a2[1], a2[2], a2[3])
rows2, res2 = lift_analysis(M2d, F1, 2)

pd.DataFrame(rows1+rows2).to_csv(DM+"kernel_conserve3_pairs.csv", index=False)
out = {"metric": "lift = pair hist − sex-global background; peak=argmax lift in ±10 hex window (center excluded)",
       "MIN_E": MIN_E, "MIN_OFFCENTER": MIN_OFFCENTER, "MIN_PEAK": MIN_PEAK, "window": W,
       "gates": {"peak": "agreement>=0.60 AND > shuffle p99", "lift": "median r>=0.4 AND > shuffle p99"},
       "panel1_gold": res1, "panel2_ext": res2}
json.dump(out, open(DM+"kernel_conserve3_m2.json", "w"), indent=1)
log(json.dumps(out, indent=1))
log("DONE")
