#!/usr/bin/env python3
"""131_kernel_conserve.py — 跨性别装配核守恒检验 (P9a-3, §40/§40c, M2 门)
面板1(金-金, 主检验): 两性注释锚点细胞(真实整数 hex 位)间真实边的有序类型对模态偏移
面板2(扩展): 雌性锚点 × 雄性 m2 全柱状细胞(直测); 报全局尺度因子 s(吸收收缩+眼尺寸差)
一致 = 模态偏移 Chebyshev ≤1 hex; 基线 = 配对洗牌 ×1000
同源映射: 雄性 ann.type → ann.flywireType 众数投票
输出: data_male/kernel_conserve_m2.json + kernel_conserve_pairs.csv
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

B = 40; NB1 = 2*B+1; NB = NB1*NB1   # 偏移窗 ±40 hex
MIN_E = 100

def pair_modes(tpre, tpost, dp, dq):
    """返回 {pair_key(str)->(mod_dp,mod_dq,n)} 对同半身旁边"""
    inr = (np.abs(dp) <= B) & (np.abs(dq) <= B)
    key = np.array([a+"|"+b for a, b in zip(tpre, tpost)])
    key, dp, dq = key[inr], dp[inr], dq[inr]
    uk, inv = np.unique(key, return_inverse=True)
    b = (dp+B)*NB1 + (dq+B)
    h = np.bincount(inv*NB + b, minlength=len(uk)*NB).reshape(len(uk), NB)
    n = h.sum(1)
    mode = h.argmax(1)
    out = {}
    for i, k_ in enumerate(uk):
        if n[i] >= MIN_E:
            out[k_] = (int(mode[i]//NB1-B), int(mode[i]%NB1-B), int(n[i]))
    return out

# ============ 雄性锚点 ============
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
km = hem_m != 0
gm, hem_m = gm[km], hem_m[km]
hex_m = np.column_stack([ann.assignedOlHex1.to_numpy()[has][km].astype(np.float64),
                         ann.assignedOlHex2.to_numpy()[has][km].astype(np.float64)])
type_m = ann.type.to_numpy()[has][km].astype(str)
fwt_m = ann.flywireType.to_numpy()[has][km].astype(str)
# 雄性类型→flywire 众数映射
fw_map = {}
for t_ in np.unique(type_m):
    m = type_m == t_
    u, c = np.unique(fwt_m[m], return_counts=True)
    ok_ = u != "None"
    if ok_.sum() == 0: continue
    fw_map[t_] = u[ok_][c[ok_].argmax()]
log(f"雄性锚点 {len(gm):,}; 类型映射 {len(fw_map)} 个")

# ============ 雌性锚点 ============
ca = pd.read_csv(DF+"column_assignment.csv.gz")
nids_f = np.fromfile(DF+"node_ids.i64", dtype=np.int64)
of = np.argsort(nids_f); sf = nids_f[of]
jf = np.searchsorted(sf, ca.root_id.values)
okf = (jf < len(sf)) & (sf[np.clip(jf, 0, len(sf)-1)] == ca.root_id.values)
gf = of[jf[okf]]
hex_f = np.column_stack([ca.p.values[okf].astype(np.float64), ca.q.values[okf].astype(np.float64)])
hem_f = np.where(ca.hemisphere.values[okf]=="right", 1, -1)
type_f = ca.type.values[okf].astype(str)
log(f"雌性锚点 {len(gf):,}")

# ============ 面板 1: 金-金 ============
def anchor_edges(DX, g_arr, hexs, hems, types):
    """g_arr: 锚点细胞全局索引(任意序); 返回 (pre类型, post类型, dp, dq) 同半身旁边"""
    ep = np.fromfile(DX+"e_pre.i32", dtype=np.int32).astype(np.int64)
    eq = np.fromfile(DX+"e_post.i32", dtype=np.int32).astype(np.int64)
    gsort = np.sort(g_arr)
    ip_ = np.searchsorted(gsort, ep); iq_ = np.searchsorted(gsort, eq)
    okp = (ip_ < len(gsort)) & (gsort[np.clip(ip_, 0, len(gsort)-1)] == ep)
    okq = (iq_ < len(gsort)) & (gsort[np.clip(iq_, 0, len(gsort)-1)] == eq)
    m = okp & okq
    # 全局索引 → 锚点序号
    order = np.argsort(g_arr)
    ip = order[ip_[m]]; iq = order[iq_[m]]
    same = hems[ip] == hems[iq]
    ip, iq = ip[same], iq[same]
    dp = np.rint(hexs[iq, 0]-hexs[ip, 0]).astype(np.int64)
    dq = np.rint(hexs[iq, 1]-hexs[ip, 1]).astype(np.int64)
    return types[ip], types[iq], dp, dq

log("面板1 雄性锚点边…")
tm_pre, tm_post, dm_p, dm_q = anchor_edges(DM, gm, hex_m, hem_m, type_m)
tm_pre_fw = np.array([fw_map.get(x, "None") for x in tm_pre])  # 雄性侧映射为 flywire 名
M1 = pair_modes(tm_pre_fw, tm_post, dm_p, dm_q)
log(f"雄性金面板合格对: {len(M1)}")
log("面板1 雌性锚点边…")
tf_pre, tf_post, df_p, df_q = anchor_edges(DF, gf, hex_f, hem_f, type_f)
F1 = pair_modes(tf_pre, tf_post, df_p, df_q)
log(f"雌性金面板合格对: {len(F1)}")

shared1 = sorted(set(M1) & set(F1))
log(f"同源合格对交集: {len(shared1)}")
rng = np.random.default_rng(0)
agree1, d_raw = [], []
for k_ in shared1:
    md = M1[k_]; fd = F1[k_]
    cheb = max(abs(md[0]-fd[0]), abs(md[1]-fd[1]))
    d_raw.append({"pair": k_, "male_dp": md[0], "male_dq": md[1], "male_n": md[2],
                  "female_dp": fd[0], "female_dq": fd[1], "female_n": fd[2], "cheb": cheb})
    agree1.append(cheb <= 1)
agree1 = np.array(agree1); acc1 = agree1.mean() if len(agree1) else np.nan
# 洗牌基线
mm = np.array([[M1[k_][0], M1[k_][1]] for k_ in shared1])
ff = np.array([[F1[k_][0], F1[k_][1]] for k_ in shared1])
base = []
for rep in range(1000):
    prm = rng.permutation(len(shared1))
    base.append((np.maximum(np.abs(mm[:, 0]-ff[prm, 0]), np.abs(mm[:, 1]-ff[prm, 1])) <= 1).mean())
base = np.array(base)
log(f"面板1: 一致率 {acc1:.3f} (洗牌基线 {base.mean():.3f} ± {base.std():.3f})")

# ============ 面板 2: 雌锚 × 雄 m2 全柱状 ============
zm = np.load(DM+"lat2q_m2.npz"); PQm, HEMm, SRCm = zm["pq"], zm["hem"], zm["src"]
nt_m = np.fromfile(DM+"node_type.i32", dtype=np.int32)
tn_m = np.load(DM+"type_names.npy", allow_pickle=True).astype(str)
col_t = sorted(set(type_m))  # 14 注释柱状型
# 扩展到计数≈1784 的柱状型
tcnt = np.bincount(nt_m)
extra = [t for t in range(len(tcnt)) if 1600 <= tcnt[t] <= 1960]
col_ids = sorted({int(t) for t in extra} | {int(np.where(tn_m == t_)[0][0]) for t_ in col_t if (tn_m == t_).any()})
m2mask = (SRCm > 0) & np.isin(nt_m, col_ids)
g2 = np.where(m2mask)[0]
hex2 = PQm[g2].astype(np.float64); hem2 = HEMm[g2]
t2 = np.array([fw_map.get(tn_m[t_], "None") for t_ in nt_m[g2]])
okt = t2 != "None"
g2, hex2, hem2, t2 = g2[okt], hex2[okt], hem2[okt], t2[okt]
log(f"面板2 雄性侧细胞: {len(g2):,} (柱状 {len(col_ids)} 型)")
t2_pre, t2_post, d2_p, d2_q = anchor_edges(DM, g2, hex2, hem2, t2)
M2d = pair_modes(t2_pre, t2_post, d2_p, d2_q)
shared2 = sorted(set(M2d) & set(F1))
mm2 = np.array([[M2d[k_][0], M2d[k_][1]] for k_ in shared2], float)
ff2 = np.array([[F1[k_][0], F1[k_][1]] for k_ in shared2], float)
acc2 = (np.maximum(np.abs(mm2[:, 0]-ff2[:, 0]), np.abs(mm2[:, 1]-ff2[:, 1])) <= 1).mean() if len(shared2) else np.nan
s_fit = float((mm2*ff2).sum()/np.maximum((ff2**2).sum(), 1e-9))
cheb_s = np.maximum(np.abs(mm2[:, 0]-s_fit*ff2[:, 0]), np.abs(mm2[:, 1]-s_fit*ff2[:, 1]))
acc2s = float((cheb_s <= 1).mean()) if len(shared2) else np.nan
base2 = []
for rep in range(1000):
    prm = rng.permutation(len(shared2))
    base2.append((np.maximum(np.abs(mm2[:, 0]-ff2[prm, 0]), np.abs(mm2[:, 1]-ff2[prm, 1])) <= 1).mean())
base2 = np.array(base2)
log(f"面板2: 合格对 {len(shared2)}, 一致率 原始 {acc2:.3f} / 尺度校正(s={s_fit:.3f}) {acc2s:.3f} (基线 {base2.mean():.3f})")

rows = []
for i, k_ in enumerate(shared2):
    rows.append({"pair": k_, "panel": 2, "male_dp": mm2[i, 0], "male_dq": mm2[i, 1],
                 "female_dp": ff2[i, 0], "female_dq": ff2[i, 1],
                 "cheb_raw": max(abs(mm2[i, 0]-ff2[i, 0]), abs(mm2[i, 1]-ff2[i, 1])),
                 "cheb_scaled": float(cheb_s[i])})
for d_ in d_raw: d_["panel"] = 1; rows.append(d_)
pd.DataFrame(rows).to_csv(DM+"kernel_conserve_pairs.csv", index=False)

def verdict(acc, n):
    if np.isnan(acc): return "NO_DATA"
    return "PASS" if acc >= 0.70 else ("PARTIAL" if acc >= 0.15 else "FAIL")
out = {"MIN_E": MIN_E, "offset_window": B,
       "panel1_gold": {"n_pairs": len(shared1), "agreement": float(acc1),
                       "shuffle_mean": float(base.mean()), "shuffle_std": float(base.std()),
                       "verdict_70": verdict(acc1, len(shared1))},
       "panel2_ext": {"n_pairs": len(shared2), "agreement_raw": float(acc2),
                      "scale_s": s_fit, "agreement_scaled": acc2s,
                      "shuffle_mean": float(base2.mean()),
                      "verdict_70_raw": verdict(acc2, len(shared2)),
                      "verdict_70_scaled": verdict(acc2s, len(shared2))},
       "note": "面板1=两性注释锚点金标准(零收缩); 面板2=雌锚×雄m2直测; s 吸收雌坐标收缩(0.752)与眼柱数差(796 vs 892)"}
json.dump(out, open(DM+"kernel_conserve_m2.json", "w"), indent=1)
log(json.dumps(out, indent=1))
log("DONE")
