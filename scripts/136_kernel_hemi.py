#!/usr/bin/env python3
"""136_kernel_hemi.py — 逐半球×逐对称核峰一致分析 (§40g 机制诊断)
假设: 雌性 hex 标记相对雄性逐眼镜像 → 混合半球后无单一全局变换可同时调和,
面板1 选中 #3(R180) 是混合伪影; 若成立, R/L 子集应各自偏好不同的反射且一致率更高。
方法: 复用 135 的数据管线, anchor_edges 返回边半球, 分半球重建对直方图,
对 hem∈{R,L} × panel∈{1,2} 各算 12 对称峰一致率 + best-of-12 洗牌零 (200 reps)。
输出: data_male/kernel_hemi_m2.json
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
MIN_E = 50; MIN_OFFCENTER = 0.02; MIN_PEAK = 0.01

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
    key = np.array([a+"|"+b for a, b in zip(tpre, tpost)])
    key, dp, dq = key[inr], dp[inr], dq[inr]
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

def hemi_panel(ukm, Hm, Lm, Nm, ukf, Hf, Lf, Nf, label):
    shared, im, if_ = np.intersect1d(ukm, ukf, return_indices=True)
    Hm, Lm, Nm = Hm[im], Lm[im], Nm[im]
    Hf, Lf, Nf = Hf[if_], Lf[if_], Nf[if_]
    n = len(shared)
    if n == 0:
        log(f"{label}: 无共享对"); return None
    offm = Hm[:, WIN].sum(1); offf = Hf[:, WIN].sum(1)
    LF_all = np.stack([Lf[:, MP[si]] for si in range(12)])
    pk_m = Lm[:, WIN].argmax(1)
    pkf_all = LF_all[:, :, WIN].argmax(2)
    ph_m = Lm[:, WIN].max(1); ph_f0 = LF_all[0][:, WIN].max(1)
    qual = (Nm >= MIN_E) & (Nf >= MIN_E) & (offm >= MIN_OFFCENTER) & (offf >= MIN_OFFCENTER) \
           & (ph_m >= MIN_PEAK) & (ph_f0 >= MIN_PEAK)
    qi = np.where(qual)[0]
    if len(qi) < 8:
        log(f"{label}: 合格对太少 {len(qi)}/{n}"); return {"n_shared": int(n), "n_qual": int(len(qi))}
    acc_by_S = np.array([cheb_agree(WIN[pk_m[qi]], WIN[pkf_all[si][qi]]).mean() for si in range(12)])
    rng = np.random.default_rng(0)
    Lfq = Lf[qi]; pk_mq = WIN[pk_m[qi]]; LfqW = Lfq[:, WIN]
    null_best = []
    for rep in range(200):
        prm = rng.permutation(len(qi))
        Lp = Lfq[prm]
        best = 0.0
        for si in range(12):
            pkf = Lp[:, MP[si]][:, WIN].argmax(1)
            best = max(best, cheb_agree(pk_mq, WIN[pkf]).mean())
        null_best.append(best)
    null_best = np.array(null_best)
    bestS = int(np.argmax(acc_by_S))
    log(f"{label}: 合格 {len(qi)}/{n}; acc_by_S = {np.round(acc_by_S,3).tolist()}; "
        f"最优 #{bestS} {acc_by_S[bestS]:.3f} (零 p99 {np.percentile(null_best,99):.3f})")
    # 逐对最优对称 (诊断: 哪些对在哪个 S 下命中)
    hit = np.zeros((len(qi), 12), bool)
    for si in range(12):
        hit[:, si] = cheb_agree(WIN[pk_m[qi]], WIN[pkf_all[si][qi]])
    return {"n_shared": int(n), "n_qual": int(len(qi)),
            "acc_by_symmetry": acc_by_S.round(3).tolist(), "best_S": bestS,
            "acc_best": float(acc_by_S[bestS]), "acc_identity": float(acc_by_S[0]),
            "null_bestof12_p99": float(np.percentile(null_best, 99)),
            "pairs": shared[qi].tolist(),
            "hits_by_S": hit.astype(int).tolist()}

# ---- 数据加载 (同 135) ----
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

log("面板1 雄性(金锚)…"); a1 = anchor_edges(DM, gm, hex_m, hem_m, type_m)
log("面板1 雌性…");       b1 = anchor_edges(DF, gf, hex_f, hem_f, type_f)

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

def split_hemi(a, tmap=None):
    tpre, tpost, dp, dq, hh = a
    if tmap is not None:
        tpre = np.array([tmap.get(x, "None") for x in tpre])
    out = {}
    for hv, name in ((1, "R"), (-1, "L")):
        m = hh == hv
        out[name] = pair_hists(tpre[m], tpost[m], dp[m], dq[m])
    return out

m1 = split_hemi(a1, fw_map); f1 = split_hemi(b1)
m2 = split_hemi(a2);        f2 = split_hemi(b1)

res = {}
for pname, mm, ff in (("panel1_gold", m1, f1), ("panel2_m2", m2, f2)):
    for hv in ("R", "L"):
        r = hemi_panel(*mm[hv], *ff[hv], f"{pname}-{hv}")
        if r: res[f"{pname}_{hv}"] = r

json.dump(res, open(DM+"kernel_hemi_m2.json", "w"), indent=1)
log("DONE")
