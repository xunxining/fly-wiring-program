#!/usr/bin/env python3
"""129c_male_latq_final.py — 雄侧晶格 v3 终版 (P9a-1, §40): soma 直测 + 传播补位
v1 (129)  自训练加密: 饱和, 3.071→3.067 (FAIL)
v2 (129b) soma 软锚+传播: 2.546 (FAIL) — 传播本身再引入收缩
v3: 有 soma 细胞直接用 kNN(k=12,逆平方,分半球) 映射坐标 (LOO 中位 0.884);
    NaN-soma 细胞用谐传播补位 (硬锚 21,375 + soma 软锚浮点位);
    半球归属: 有 soma 者取两映射 d12 较近侧, 其余取传播符号。
调参全程训练集 LOO; rng7 留出 2,345 锚点仅终评一次。
输出: data_male/lat2q_m2.npz {pq f32, hem i8, src i8(1注释/2soma直测/3传播补位)} + lat2q_dense_diag_m.json
"""
import sys; sys.path.insert(0, "/mnt/agents/pylibs")
import numpy as np, json, time
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
R = "/mnt/agents/output/果蝇生成器/"
D = R + "data_male/"
t0 = time.time()
def log(*a):
    try: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
    except OSError: pass

ep = np.fromfile(D+"e_pre.i32", dtype=np.int32)
eq = np.fromfile(D+"e_post.i32", dtype=np.int32)
ew = np.fromfile(D+"e_w.i32", dtype=np.int32).astype(np.float64)
sc = np.fromfile(D+"node_sc.i32", dtype=np.int32)
N = len(sc)
vis = np.isin(sc, [13, 14, 18, 19])
oi = np.where(vis)[0]; no = len(oi)
xyz = np.fromfile(D+"xyz.f32", dtype=np.float32).reshape(N, 3)
log(f"visual-set cells: {no:,}")

rng42 = np.random.default_rng(42)
sel = vis[ep] & vis[eq]
p, q, w = ep[sel], eq[sel], ew[sel]
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)
keep = p2 != q2
p2, q2, w = p2[keep], q2[keep], w[keep]
fit = rng42.random(len(p2)) < 0.9
r = np.concatenate([p2[fit], q2[fit]]); c = np.concatenate([q2[fit], p2[fit]])
ww = np.concatenate([w[fit], w[fit]])
A = csr_matrix((ww, (r, c)), shape=(no, no)); A.sum_duplicates()
dvec = np.asarray(A.sum(1)).ravel(); dvec[dvec <= 0] = 1e-9
Dinv = 1.0/dvec
del r, c, ww, p, q, w, p2, q2, sel, keep, fit
import gc; gc.collect()

# ---- 注释锚点 ----
ann = pd.read_feather(R+"data/body-annotations.feather",
                      columns=["bodyId","assignedOlHex1","assignedOlHex2","somaSide"])
node_ids = np.fromfile(D+"node_ids.i64", dtype=np.int64)
order = np.argsort(node_ids); sids = node_ids[order]
j = np.searchsorted(sids, ann.bodyId.values)
okm = sids[np.clip(j, 0, len(sids)-1)] == ann.bodyId.values
glo = np.full(len(ann), -1, np.int64); glo[okm] = order[j[okm]]
has_anchor = ann.assignedOlHex1.notna().to_numpy() & ann.assignedOlHex2.notna().to_numpy() & (glo >= 0)
glo_a = glo[has_anchor]; glo_a = glo_a[vis[glo_a]]
hemraw = ann.somaSide.to_numpy()[has_anchor][vis[glo[has_anchor]]]
hemv = np.where(hemraw == "R", 1.0, np.where(hemraw == "L", -1.0, 0.0))
kh = hemv != 0; glo_a = glo_a[kh]; hemv = hemv[kh]
labels = np.column_stack([ann.assignedOlHex1.to_numpy()[has_anchor][vis[glo[has_anchor]]][kh].astype(np.float64),
                          ann.assignedOlHex2.to_numpy()[has_anchor][vis[glo[has_anchor]]][kh].astype(np.float64), hemv])
lok = np.searchsorted(oi, glo_a)
log(f"anchors: {len(lok):,}")
rng2 = np.random.default_rng(7)
hold = rng2.random(len(lok)) < 0.10
HL = lok[hold]; TRUE = labels[hold]
tr = ~hold
sx_a = xyz[glo_a]

# ---- soma→hex 映射 (分半球 kNN k=12) ----
K = 12
maps = {}
for hemi in [-1.0, 1.0]:
    m = tr & (labels[:, 2] == hemi) & np.isfinite(sx_a).all(1)
    maps[hemi] = (cKDTree(sx_a[m]), labels[m][:, :2])
    log(f"hem {hemi:+.0f}: 映射锚点 {m.sum():,}")

# 全体视觉细胞直测
sx_v = xyz[oi]
fin_v = np.isfinite(sx_v).all(1)
dL = np.full(no, np.inf); dR = np.full(no, np.inf)
dL[fin_v] = maps[-1.0][0].query(sx_v[fin_v], k=K)[0][:, -1]
dR[fin_v] = maps[ 1.0][0].query(sx_v[fin_v], k=K)[0][:, -1]
hem_direct = np.where(dR <= dL, 1.0, -1.0)
pq_direct = np.full((no, 2), np.nan)
for hemi in [-1.0, 1.0]:
    mm = fin_v & (hem_direct == hemi)
    if mm.sum() == 0: continue
    tree, hexlab = maps[hemi]
    d, k = tree.query(sx_v[mm], k=K)
    w_ = 1.0/np.maximum(d, 1e-6)**2; w_ /= w_.sum(1, keepdims=True)
    pq_direct[mm] = (hexlab[k]*w_[..., None]).sum(1)
log(f"soma 直测覆盖: {fin_v.sum():,}/{no:,} ({fin_v.mean():.3f})")

# ---- 传播补位 (硬锚 + soma 软锚) ----
anch = np.zeros(no, bool); anch[lok[tr]] = True
Y = np.zeros((no, 3)); Y[lok[tr]] = labels[tr]
soft = fin_v & ~anch
soft[HL] = False
Y[soft, :2] = pq_direct[soft]; Y[soft, 2] = hem_direct[soft]
anch[soft] = True
log(f"传播锚点合计: {anch.sum():,} (硬 {(~hold).sum():,} + 软 {soft.sum():,})")
X = Y.copy()
for it in range(1500):
    Xn = Dinv[:, None] * (A @ X)
    Xn[anch] = Y[anch]
    mx = np.abs(Xn - X).max()
    X = Xn
    if mx < 1e-4: break
log(f"传播收敛 it={it}")

# ---- 合成终坐标 ----
PQf = np.where(fin_v[:, None], pq_direct, X[:, :2])          # 有 soma 直测优先
HEMf = np.where(fin_v, hem_direct, np.sign(X[:, 2]))
deg0 = np.asarray((A != 0).sum(1)).ravel() == 0
src_loc = np.zeros(no, np.int8)
src_loc[fin_v] = 2; src_loc[~fin_v] = 3
src_loc[lok] = 1                                            # 注释锚点标记(含留出, 仅 provenance)
PQf[deg0] = np.nan; HEMf[deg0] = 0; src_loc[deg0] = 0

# ---- 终评 (留出, 唯一一次) ----
err = np.sqrt((PQf[HL, 0] - TRUE[:, 0])**2 + (PQf[HL, 1] - TRUE[:, 1])**2)
med, p1 = float(np.median(err)), float((err <= 1).mean())
hacc = float((np.sign(HEMf[HL]) == TRUE[:, 2]).mean())
fin_hl = fin_v[HL]
err_soma = err[fin_hl]; err_prop = err[~fin_hl]
log(f"终评(留出 {len(HL):,}): 中位 {med:.3f}  P≤1 {p1:.3f}  hem {hacc:.4f}")
log(f"  其中 soma 直测 {fin_hl.sum():,}: 中位 {np.median(err_soma):.3f}; 传播补位 {(~fin_hl).sum():,}: 中位 {np.median(err_prop):.3f}")

PQ = np.full((N, 2), np.nan, np.float32); HEM = np.zeros(N, np.int8); SRC = np.zeros(N, np.int8)
PQ[oi] = PQf.astype(np.float32); HEM[oi] = HEMf.astype(np.int8); SRC[oi] = src_loc
for att in range(20):
    try:
        np.savez_compressed(D+"lat2q_m2.npz", pq=PQ, hem=HEM, src=SRC); break
    except OSError as e:
        log("写入重试", att, str(e)[:40]); time.sleep(10)
diag = {"n_visual": int(no), "anchors_ann_train": int(tr.sum()), "heldout_n": int(len(HL)),
        "soma_direct_coverage": float(fin_v.mean()),
        "held_anchor_median_err_hex": med, "held_anchor_P<=1": p1, "held_anchor_hem_acc": hacc,
        "held_median_soma_subset": float(np.median(err_soma)),
        "held_median_prop_subset": float(np.median(err_prop)),
        "map": {"k": K, "weight": "inverse-square", "per_hemisphere": True,
                "tuning": "训练集 LOO: k12 全体中位 0.884 (k6 0.896, k8 0.889); 支撑域截断无影响"},
        "history": {"v0_103": {"median": 3.0711, "P1": 0.1365},
                    "v1_129_selftrain": {"median": 3.067, "P1": 0.135, "fail": "自训练饱和"},
                    "v2_129b_soma_anchor_prop": {"median": 2.546, "P1": 0.176, "fail": "传播再收缩"},
                    "female_ref": {"median": 1.6598, "P1": 0.3022}},
        "gate_M1": {"median<=2.0": bool(med <= 2.0), "P1>=0.22": bool(p1 >= 0.22),
                    "verdict": "PASS" if (med <= 2.0 and p1 >= 0.22) else "FAIL"}}
json.dump(diag, open(D+"lat2q_dense_diag_m.json", "w"), indent=1)
log("SAVED lat2q_m2.npz ; M1 门:", diag["gate_M1"]["verdict"])
