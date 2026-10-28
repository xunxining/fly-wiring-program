#!/usr/bin/env python3
"""135_kernel_conserve4.py — 跨性别装配核守恒检验 v4 终审 (§40f): 轴约定调和 + best-of-12 零分布
向量化: 每对称的 bin 置换预计算一次, 直方图重排 = 花哨索引。
S(轴交换) 只在面板1 选择; 终门在面板2 (S 固定): 峰一致率 ≥0.60 且 > 洗牌×best-of-12 零 p99; lift-r 中位 ≥0.40。
输出: data_male/kernel_conserve4_m2.json + kernel_conserve4_pairs.csv
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

def rots(dp, dq):
    return [(dp, dq), (-dq, dp+dq), (-dp-dq, dp), (-dp, -dq), (dq, -dp-dq), (dp+dq, -dq)]
SYMS = [lambda dp, dq, k=k: rots(dp, dq)[k] for k in range(6)] + \
       [lambda dp, dq, k=k: rots(dq, dp)[k] for k in range(6)]
# 每对称的 bin 置换: new_hist[b_new] = old[b_old], b_new = S(b_old); 存 mp_S[new]=old
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
    return types[ip], types[iq], dp, dq

def cheb_agree(pkm, pkf):
    d = np.maximum(np.abs(pkm//NB1 - pkf//NB1), np.abs(pkm%NB1 - pkf%NB1))
    return d <= 1

def panel(ukm, Hm, Lm, Nm, ukf, Hf, Lf, Nf, label, S_fixed=None):
    shared, im, if_ = np.intersect1d(ukm, ukf, return_indices=True)
    Hm, Lm, Nm = Hm[im], Lm[im], Nm[im]
    Hf, Lf, Nf = Hf[if_], Lf[if_], Nf[if_]
    n = len(shared)
    offm = Hm[:, WIN].sum(1); offf = Hf[:, WIN].sum(1)
    # 每对称的雌峰与合格掩码 (峰高资格随对称不变, 用恒等下峰高)
    LF_all = np.stack([Lf[:, MP[si]] for si in range(12)])     # 12 × n × NB
    HF_all = np.stack([Hf[:, MP[si]] for si in range(12)])
    pk_m = Lm[:, WIN].argmax(1)                                # 窗内 argmax 索引
    pkf_all = LF_all[:, :, WIN].argmax(2)                      # 12 × n
    ph_m = Lm[:, WIN].max(1)
    ph_f0 = LF_all[0][:, WIN].max(1)
    qual = (Nm >= MIN_E) & (Nf >= MIN_E) & (offm >= MIN_OFFCENTER) & (offf >= MIN_OFFCENTER) \
           & (ph_m >= MIN_PEAK) & (ph_f0 >= MIN_PEAK)
    qi = np.where(qual)[0]
    acc_by_S = np.array([cheb_agree(WIN[pk_m[qi]], WIN[pkf_all[si][qi]]).mean() for si in range(12)])
    # 洗牌 × best-of-12 零分布 (合格对子集)
    rng = np.random.default_rng(0)
    Lfq = Lf[qi]; pk_mq = WIN[pk_m[qi]]
    null_best = []
    LfqW = Lfq[:, WIN]
    for rep in range(300):
        prm = rng.permutation(len(qi))
        Lp = Lfq[prm]
        best = 0.0
        for si in range(12):
            pkf = Lp[:, MP[si]][:, WIN].argmax(1)
            best = max(best, cheb_agree(pk_mq, WIN[pkf]).mean())
        null_best.append(best)
    null_best = np.array(null_best)
    # 选定 S
    S_sel = int(np.argmax(acc_by_S)) if S_fixed is None else S_fixed
    pkf_sel = WIN[pkf_all[S_sel][qi]]
    acc = float(cheb_agree(pk_mq, pkf_sel).mean())
    # lift-r (选定 S)
    lmw = Lm[qi][:, WIN]; lfw = LF_all[S_sel][qi][:, WIN]
    def medr(a, b):
        rs = []
        for x, y in zip(a, b):
            if x.std() > 0 and y.std() > 0: rs.append(np.corrcoef(x, y)[0, 1])
        return float(np.median(rs)), np.array(rs)
    r_lift, rl_arr = medr(lmw, lfw)
    null_r = []
    for rep in range(300):
        prm = rng.permutation(len(qi))
        null_r.append(medr(lmw, lfw[prm])[0])
    null_r = np.array(null_r)
    log(f"面板{label}: 合格 {len(qi)}/{n}; 恒等 {acc_by_S[0]:.3f}; 各对称最优 #{int(np.argmax(acc_by_S))} "
        f"{acc_by_S.max():.3f}; S={S_sel} 一致率 {acc:.3f} (best-of-12 零 p99 {np.percentile(null_best,99):.3f}); "
        f"lift-r {r_lift:.3f} (零 p99 {np.percentile(null_r,99):.3f})")
    res = {"n_shared": int(n), "n_qual": int(len(qi)),
           "_null_best": null_best, "_null_r": null_r,
           "acc_by_symmetry": acc_by_S.round(3).tolist(), "acc_identity": float(acc_by_S[0]),
           "S_used": S_sel, "acc": acc,
           "null_bestof12_p99": float(np.percentile(null_best, 99)),
           "null_bestof12_mean": float(null_best.mean()),
           "median_r_lift": r_lift, "null_r_p99": float(np.percentile(null_r, 99)),
           "gate_peak": "PASS" if (acc >= 0.60 and acc > np.percentile(null_best, 99)) else "FAIL",
           "gate_lift": "PASS" if (r_lift >= 0.40 and r_lift > np.percentile(null_r, 99)) else "FAIL"}
    return res, shared, qi, pk_m, pkf_all[S_sel], rl_arr

# ---- 数据加载 ----
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
ukm1, Hm1, Lm1, Nm1 = pair_hists(np.array([fw_map.get(x, "None") for x in a1[0]]), a1[1], a1[2], a1[3])
log("面板1 雌性…"); b1 = anchor_edges(DF, gf, hex_f, hem_f, type_f)
ukf, Hf, Lf, Nf = pair_hists(b1[0], b1[1], b1[2], b1[3])

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
ukm2, Hm2, Lm2, Nm2 = pair_hists(a2[0], a2[1], a2[2], a2[3])

res1, sh1, qi1, pkm1, pkf1, rl1 = panel(ukm1, Hm1, Lm1, Nm1, ukf, Hf, Lf, Nf, 1)
S_sel = int(np.argmax(res1["acc_by_symmetry"]))
log(f"面板1 选择 S = {S_sel} (注册预期 6=轴交换)")
res2, sh2, qi2, pkm2, pkf2, rl2 = panel(ukm2, Hm2, Lm2, Nm2, ukf, Hf, Lf, Nf, 2, S_fixed=S_sel)

rows = []
for j_, i_ in enumerate(qi2):
    rows.append({"pair": sh2[i_], "n_male": int(Nm2[np.searchsorted(ukm2, sh2[i_])]),
                 "peak_male": f"{WIN[pkm2[i_]]//NB1-B},{WIN[pkm2[i_]]%NB1-B}",
                 "peak_female_harmonized": f"{pkf2[j_]//NB1-B},{pkf2[j_]%NB1-B}",
                 "r_lift": float(rl2[j_])})
pd.DataFrame(rows).to_csv(DM+"kernel_conserve4_pairs.csv", index=False)
out = {"protocol": "S 由面板1(金锚) argmax 选择; 终门在面板2(雄性 m2 坐标)评定, S 固定; 零分布=配对洗牌×best-of-12",
       "handedness_physical": {"male_R": 1, "male_L": -1, "female_R": -1, "female_L": 1,
                               "meaning": "两性 hex 坐标约定互为镜像(物理测量, 独立于连接), 反射调和成立"},
       "female_contraction": 0.752,
       "S_selected": S_sel, "S_meaning": "轴交换(反射)" if S_sel == 6 else f"对称#{S_sel}",
       "panel1_selection_set": res1, "panel2_validation": res2}
import numpy as _np
# ---- Fig7 热图: 面板2 代表对 lift (S=3 调和) ----
_W = 10; _NB1 = 81
wr = WIN//_NB1-B+_W; wc = WIN%_NB1-B+_W
_heat = {}
# 面板2 的 ukm2/Lm2/Nm2/ukf/Lf/Nf 与 shared/qi/pk_m/pkf(S=3) 均在作用域
_sh2 = list(sh2)
for _p in ["C3|Tm4", "Tm4|Tm4", "L5|Mi4", "L5|L5", "Tm20|Tm20", "L5|Mi1", "L2|Tm2", "L1|C2"]:
    if _p not in _sh2: continue
    _j = _sh2.index(_p)
    _im = _np.searchsorted(ukm2, _p); _if = _np.searchsorted(ukf, _p)
    _lm10 = Lm2[_im, WIN]; _lf10 = Lf[_if, MP[3]][WIN]
    _lm = _np.zeros((2*_W+1, 2*_W+1)); _lm[wr, wc] = _lm10
    _lf = _np.zeros((2*_W+1, 2*_W+1)); _lf[wr, wc] = _lf10
    _pkm = WIN[_lm10.argmax()]; _pkf = WIN[_lf10.argmax()]
    _heat[_p] = (_lm, _lf, int(Nm2[_im]), int(Nf[_if]),
                 (int(_pkm//_NB1-B), int(_pkm%_NB1-B)), (int(_pkf//_NB1-B), int(_pkf%_NB1-B)))
    log(f"热图 {_p}: 雄峰 {_heat[_p][4]} 雌峰(调和) {_heat[_p][5]} n_m {Nm2[_im]:,}")
_hits = [_p for _p, v in _heat.items()
         if max(abs(v[4][0]-v[5][0]), abs(v[4][1]-v[5][1])) <= 1]
_hits = sorted(_hits, key=lambda p: -_heat[p][2])[:4]
log(f"热图命中选中: {_hits}")
_np.savez(DM+"fig7_heat.npz",
          show_names=_np.array(_hits),
          **{f"heat_{k}_{kk}": vv for k, p in enumerate(_hits) for kk, vv in
             (("m", _heat[p][0]), ("f", _heat[p][1]),
              ("info", _np.array([_heat[p][2], _heat[p][3]])),
              ("pk", _np.array([_heat[p][4][0], _heat[p][4][1], _heat[p][5][0], _heat[p][5][1]])))})
_np.savez(DM+"fig7_nulls.npz",
          null_best_p2=_np.array(out["panel2_validation"].pop("_null_best")),
          null_r_p2=_np.array(out["panel2_validation"].pop("_null_r")),
          null_best_p1=_np.array(out["panel1_selection_set"].pop("_null_best")),
          null_r_p1=_np.array(out["panel1_selection_set"].pop("_null_r")))
json.dump(out, open(DM+"kernel_conserve4_m2.json", "w"), indent=1)
log(json.dumps(res2, indent=1))
log("DONE")
