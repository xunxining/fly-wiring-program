#!/usr/bin/env python3
"""129b_male_latq_soma.py — 雄侧晶格 v2 (P9a-1 修订, §40): soma 空间先验 + 图传播
v1 (129) 教训: 自训练加密饱和(只加已准细胞, 留出误差不动 3.071→3.067)。
v1 诊断: 误差=系统性向心收缩(半径比 0.70) + 髓质类型/边缘带最差;
        soma-kNN(k=8,逆平方) 独立信号中位 0.85 hex 且全边缘带平坦 ⇒ 空间先验主导。
v2: (1) 分半球从训练锚点学 soma→hex kNN 映射;
    (2) 柱状类型(27类)×有soma×在支撑域 的细胞以浮点预测位晋升软锚点;
    (3) 重传播(图平滑正则化 soma 噪声, 填补 NaN-soma 细胞)。
泄漏控制: rng7 留出 10% 锚点全程不参与映射拟合/锚定/评估。
输出: data_male/lat2q_m2.npz {pq f32, hem i8, src i8(1注释/2soma晋升/0其余)} + lat2q_dense_diag_m.json
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
nt = np.fromfile(D+"node_type.i32", dtype=np.int32)
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

# ---- 锚点 (复刻 103) ----
ann = pd.read_feather(R+"data/body-annotations.feather",
                      columns=["bodyId","assignedOlHex1","assignedOlHex2","somaSide"])
node_ids = np.fromfile(D+"node_ids.i64", dtype=np.int64)
order = np.argsort(node_ids); sids = node_ids[order]
j = np.searchsorted(sids, ann.bodyId.values)
okm = sids[np.clip(j, 0, len(sids)-1)] == ann.bodyId.values
glo = np.full(len(ann), -1, np.int64); glo[okm] = order[j[okm]]
has_anchor = ann.assignedOlHex1.notna().to_numpy() & ann.assignedOlHex2.notna().to_numpy() & (glo >= 0)
glo_a = glo[has_anchor]; isl = vis[glo_a]; glo_a = glo_a[isl]
lok = np.searchsorted(oi, glo_a)
hemraw = ann.somaSide.to_numpy()[has_anchor][isl]
hemv = np.where(hemraw == "R", 1.0, np.where(hemraw == "L", -1.0, 0.0))
keeph = hemv != 0; lok = lok[keeph]; hemv = hemv[keeph]
labels = np.column_stack([ann.assignedOlHex1.to_numpy()[has_anchor][isl][keeph].astype(np.float64),
                          ann.assignedOlHex2.to_numpy()[has_anchor][isl][keeph].astype(np.float64), hemv])
log(f"anchors: {len(lok):,}")
rng2 = np.random.default_rng(7)
hold = rng2.random(len(lok)) < 0.10
HL = lok[hold]; TRUE = labels[hold]
tr = ~hold
sx_all = xyz[glo_a]                      # 锚点 soma (含 NaN)

# ---- 柱状类型 ----
type_names = np.load(D+"type_names.npy", allow_pickle=True)
tcnt = np.bincount(nt, minlength=int(nt.max())+1)
u, cts = np.unique(nt[glo_a], return_counts=True)
anch_types = {int(t_) for t_, c_ in zip(u, cts) if c_ >= 100}
COLT = sorted(anch_types | {t for t in range(len(tcnt)) if 1600 <= tcnt[t] <= 1960})
colt_mask = np.isin(nt, np.array(COLT, np.int32))
log(f"columnar types ({len(COLT)})")

# ---- soma→hex 映射 (分半球 kNN k=8 逆平方) ----
maps = {}
for hemi in [-1.0, 1.0]:
    m = tr & (labels[:, 2] == hemi) & np.isfinite(sx_all).all(1)
    tree = cKDTree(sx_all[m])
    d9, _ = tree.query(sx_all[m], k=9)   # LOO 支撑域: 训练点第8近邻距离分布
    sup = float(np.percentile(d9[:, 8], 95))
    maps[hemi] = (tree, labels[m][:, :2], sup)
    log(f"hem {hemi:+.0f}: 映射锚点 {m.sum():,} 支撑域d8p95 {sup:.2f}µm")

# ---- 软锚点晋升 ----
anch = np.zeros(no, bool); anch[lok[tr]] = True
Y = np.zeros((no, 3)); Y[lok[tr]] = labels[tr]
src_loc = np.zeros(no, np.int8); src_loc[lok] = 1
cand_glo = oi[(~anch) & colt_mask[oi]]
# 半球判定: 两半球映射各查一次, 取第8近邻距离更近者; 支撑域 = 该侧 d8 ≤ 其锚点分布 p95
sxc = xyz[cand_glo]
fin = np.isfinite(sxc).all(1)
dL = np.full(len(cand_glo), np.inf); dR = np.full(len(cand_glo), np.inf)
dL[fin] = maps[-1.0][0].query(sxc[fin], k=8)[0][:, 7]
dR[fin] = maps[ 1.0][0].query(sxc[fin], k=8)[0][:, 7]
hem_c = np.where(dR <= dL, 1.0, -1.0)
in_sup = fin & np.where(hem_c > 0, dR <= maps[1.0][2], dL <= maps[-1.0][2])
# 用所判半球的映射预测 hex
pred_hex = np.full((len(cand_glo), 2), np.nan)
for hemi in [-1.0, 1.0]:
    mm = (hem_c == hemi) & in_sup
    if mm.sum() == 0: continue
    tree, hexlab, _ = maps[hemi]
    d, k = tree.query(xyz[cand_glo][mm], k=8)
    w_ = 1.0/np.maximum(d, 1e-6)**2; w_ /= w_.sum(1, keepdims=True)
    pred_hex[mm] = (hexlab[k]*w_[..., None]).sum(1)
# 留出细胞永不晋升
pl = np.searchsorted(oi, cand_glo)
keep_p = in_sup & ~np.isin(pl, HL)
final_promo = pl[keep_p]
final_hex = pred_hex[keep_p]
final_hem = hem_c[keep_p]
Y[final_promo, 0] = final_hex[:, 0]; Y[final_promo, 1] = final_hex[:, 1]; Y[final_promo, 2] = final_hem
anch[final_promo] = True; src_loc[final_promo] = 2
log(f"soma 晋升软锚点: {len(final_promo):,} (柱状候选 {len(cand_glo):,}, 支撑域内 {int(in_sup.sum()):,})")

# ---- 重传播 ----
X = Y.copy()
for it in range(1500):
    Xn = Dinv[:, None] * (A @ X)
    Xn[anch] = Y[anch]
    mx = np.abs(Xn - X).max()
    X = Xn
    if mx < 1e-4: break
log(f"传播收敛 it={it}")

err = np.sqrt(((X[HL, :2] - TRUE[:, :2])**2).sum(1))
med, p1 = float(np.median(err)), float((err <= 1).mean())
hacc = float((np.sign(X[HL, 2]) == TRUE[:, 2]).mean())
log(f"终评(留出 {len(HL):,}): 中位 {med:.3f}  P≤1 {p1:.3f}  hem {hacc:.4f}")

deg0 = np.asarray((A != 0).sum(1)).ravel() == 0
pq = X[:, :2].astype(np.float32); hem = np.sign(X[:, 2]).astype(np.int8)
pq[deg0] = np.nan; hem[deg0] = 0; src_loc[deg0] = 0
PQ = np.full((N, 2), np.nan, np.float32); HEM = np.zeros(N, np.int8); SRC = np.zeros(N, np.int8)
PQ[oi] = pq; HEM[oi] = hem; SRC[oi] = src_loc
for att in range(20):
    try:
        np.savez_compressed(D+"lat2q_m2.npz", pq=PQ, hem=HEM, src=SRC); break
    except OSError as e:
        log("写入重试", att, str(e)[:40]); time.sleep(10)
diag = {"n_visual": int(no), "anchors_ann": int((~hold).sum()), "anchors_soma_promoted": int(len(final_promo)),
        "anchors_total": int(anch.sum()), "heldout_n": int(len(HL)),
        "held_anchor_median_err_hex": med, "held_anchor_P<=1": p1, "held_anchor_hem_acc": hacc,
        "columnar_types": [str(type_names[t]) for t in COLT],
        "v1_selftrain": {"median": 3.067, "P1": 0.135, "note": "饱和: 只晋升已准细胞"},
        "baseline_103": {"median": 3.0711, "P1": 0.1365}, "female_ref": {"median": 1.6598, "P1": 0.3022},
        "soma_knn_alone": {"median_L": 0.824, "median_R": 0.886},
        "gate_M1": {"median<=2.0": bool(med <= 2.0), "P1>=0.22": bool(p1 >= 0.22),
                    "verdict": "PASS" if (med <= 2.0 and p1 >= 0.22) else "FAIL"}}
json.dump(diag, open(D+"lat2q_dense_diag_m.json", "w"), indent=1)
log("SAVED lat2q_m2.npz ; M1 门:", diag["gate_M1"]["verdict"])
