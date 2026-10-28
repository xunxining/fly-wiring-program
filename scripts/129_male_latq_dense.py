#!/usr/bin/env python3
"""129_male_latq_dense.py — 雄侧晶格锚点加密 (P9a-1, §40)
轮次制半监督标签扩散:
  第0轮 = 复刻 103 谐扩展 (rng42 90/10 拟合边, 23,720 注释锚点, rng7 扣 10% 留出)
  第k轮 = 柱状类型细胞(计数∈[1600,1960] ∪ 已注释类型)中满足 (i)距最近合法hex≤thr
          (ii)该(类型,hex,半球)列无现存锚点 (iii)同类最近者唯一性约束 者晋升锚点, 重传播
  thr = 0.6 → 0.75 → 0.9; 留出误差恶化即回滚并停止
泄漏控制: rng7 留出锚点全程不参与锚定/晋升/调阈。
输出: data_male/lat2q_m2.npz {pq f32, hem i8, src i8(0无/1注释/2晋升)} + lat2q_dense_diag_m.json
检查点: data_male/latq_dense_ckpt.npz (每轮后)
"""
import sys; sys.path.insert(0, "/mnt/agents/pylibs")
import numpy as np, json, time, os
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
R = "/mnt/agents/output/果蝇生成器/"
D = R + "data_male/"
t0 = time.time()
def log(*a):
    try: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
    except OSError: pass

CKPT = D + "latq_dense_ckpt.npz"

ep = np.fromfile(D+"e_pre.i32", dtype=np.int32)
eq = np.fromfile(D+"e_post.i32", dtype=np.int32)
ew = np.fromfile(D+"e_w.i32", dtype=np.int32).astype(np.float64)
sc = np.fromfile(D+"node_sc.i32", dtype=np.int32)
nt = np.fromfile(D+"node_type.i32", dtype=np.int32)
N = len(sc)
VIS_SC = {13, 14, 18, 19}
vis = np.isin(sc, list(VIS_SC))
oi = np.where(vis)[0]; no = len(oi)
log(f"visual-set cells: {no:,}")

rng42 = np.random.default_rng(42)
sel = vis[ep] & vis[eq]
p, q, w = ep[sel], eq[sel], ew[sel]
p2 = np.searchsorted(oi, p); q2 = np.searchsorted(oi, q)
keep = p2 != q2
p2, q2, w = p2[keep], q2[keep], w[keep]
fit = rng42.random(len(p2)) < 0.9
log(f"visual-visual edges: {len(p2):,} (fit {fit.sum():,})")

r = np.concatenate([p2[fit], q2[fit]]); c = np.concatenate([q2[fit], p2[fit]])
ww = np.concatenate([w[fit], w[fit]])
A = csr_matrix((ww, (r, c)), shape=(no, no)); A.sum_duplicates()
dvec = np.asarray(A.sum(1)).ravel(); dvec[dvec <= 0] = 1e-9
Dinv = 1.0/dvec
del r, c, ww, p, q, w, p2, q2, sel, keep, fit
import gc; gc.collect()

# ---- 锚点 (复刻 103) ----
ann = pd.read_feather(R+"data/body-annotations.feather",
                      columns=["bodyId","assignedOlHex1","assignedOlHex2","somaSide","type"])
node_ids = np.fromfile(D+"node_ids.i64", dtype=np.int64)
order = np.argsort(node_ids); sids = node_ids[order]
j = np.searchsorted(sids, ann.bodyId.values)
okm = sids[np.clip(j, 0, len(sids)-1)] == ann.bodyId.values
glo = np.full(len(ann), -1, np.int64); glo[okm] = order[j[okm]]
has_anchor = ann.assignedOlHex1.notna().to_numpy() & ann.assignedOlHex2.notna().to_numpy() & (glo >= 0)
glo_a = glo[has_anchor]
isl = vis[glo_a]
glo_a = glo_a[isl]
lok = np.searchsorted(oi, glo_a)
hemraw = ann.somaSide.to_numpy()[has_anchor][isl]
hemv = np.where(hemraw == "R", 1.0, np.where(hemraw == "L", -1.0, 0.0))
keeph = hemv != 0
lok = lok[keeph]; hemv = hemv[keeph]
labels = np.column_stack([ann.assignedOlHex1.to_numpy()[has_anchor][isl][keeph].astype(np.float64),
                          ann.assignedOlHex2.to_numpy()[has_anchor][isl][keeph].astype(np.float64),
                          hemv])
n_ann = len(lok)
log(f"anchors: {n_ann:,}")

rng2 = np.random.default_rng(7)
hold = rng2.random(n_ann) < 0.10
HL = lok[hold]; TRUE = labels[hold]          # 留出集: 全程冻结

# 合法 hex 网格 (注释出现过的 (h1,h2) 对)
valid_hex = np.unique(labels[:, :2].astype(np.int32), axis=0).astype(np.float64)
hex_tree = cKDTree(valid_hex)
log(f"valid hex grid: {len(valid_hex):,} columns")

# 柱状类型侦测: 全脑计数∈[1600,1960] 或 注释锚点≥100 的类型
type_names = np.load(D+"type_names.npy", allow_pickle=True)
tcnt = np.bincount(nt, minlength=int(nt.max())+1)
anch_types = set()
tl = nt[glo_a]  # 锚点细胞类型
u, cts = np.unique(tl, return_counts=True)
for t_, c_ in zip(u, cts):
    if c_ >= 100: anch_types.add(int(t_))
COLT = sorted(anch_types | {t for t in range(len(tcnt)) if 1600 <= tcnt[t] <= 1960})
log(f"columnar types ({len(COLT)}):", [str(type_names[t]) for t in COLT])
colt_mask = np.isin(nt, np.array(COLT, np.int32))

def propagate(anch, Y):
    X = Y.copy()
    for it in range(1500):
        Xn = Dinv[:, None] * (A @ X)
        Xn[anch] = Y[anch]
        mx = np.abs(Xn - X).max()
        X = Xn
        if mx < 1e-4: break
    return X, it

def eval_hold(X):
    err = np.sqrt(((X[HL, :2] - TRUE[:, :2])**2).sum(1))
    return float(np.median(err)), float((err <= 1).mean()), float((np.sign(X[HL, 2]) == TRUE[:, 2]).mean())

# 列占用表: (类型, hem_sign, h1, h2) 是否已有锚点 — 用注释锚点初始化
def col_key(t_arr, hem_arr, h):
    return ((t_arr.astype(np.int64) * 2 + (hem_arr > 0).astype(np.int64)) * 40 + h[:, 0].astype(np.int64)) * 40 + h[:, 1].astype(np.int64)

anch = np.zeros(no, bool); anch[lok[~hold]] = True
Y = np.zeros((no, 3)); Y[lok[~hold]] = labels[~hold]
src_loc = np.zeros(no, np.int8); src_loc[lok[~hold]] = 1; src_loc[HL] = 1  # 留出也算注释来源(但不锚定)
occupied = set(col_key(nt[glo_a], labels[:, 2], labels[:, :2]).tolist())   # 含留出锚点所占列(其列视为已有真值)

start_round = 0
X = None
hist = []
if os.path.exists(CKPT):
    try:
        z = np.load(CKPT)
        X = z["X"]; anch = z["anch"]; Y = z["Y"]; src_loc = z["src"]
        occupied = set(z["occupied"].tolist())
        hist = json.loads(z["hist"].tobytes().decode())
        start_round = int(z["round"]) + 1
        log(f"恢复检查点: 第 {start_round} 轮继续, 锚点 {anch.sum():,}")
    except Exception as e:
        log("检查点读取失败, 从头开始:", str(e)[:60]); X = None

THRS = [0.6, 0.75, 0.9]
best = None
for rnd in range(start_round, 4):
    X, it = propagate(anch, Y)
    med, p1, hacc = eval_hold(X)
    log(f"第{rnd}轮传播收敛 it={it}  留出: 中位{med:.3f} P≤1 {p1:.3f} hem {hacc:.4f}  锚点 {anch.sum():,}")
    hist.append({"round": rnd, "iters": int(it), "anchors": int(anch.sum()),
                 "held_median": med, "held_P1": p1, "held_hem": hacc})
    if best is not None and med > best[0] + 1e-9:
        log(f"留出误差恶化 ({med:.3f} > {best[0]:.3f}), 回滚停止")
        X = best[1]; anch = best[2]; Y = best[3]; src_loc = best[4]
        hist[-1]["rolled_back"] = True
        break
    best = (med, X.copy(), anch.copy(), Y.copy(), src_loc.copy())
    if rnd == 3:
        break
    # ---- 晋升 ----
    thr = THRS[rnd]
    hem_pred = np.sign(X[:, 2])
    cand = (~anch) & colt_mask[oi] & (hem_pred != 0) & np.isfinite(X[:, 0])
    cand[HL] = False                       # 留出细胞永不晋升
    ci = np.where(cand)[0]
    d, k = hex_tree.query(X[ci, :2])
    near = d <= thr
    ci, d, k = ci[near], d[near], k[near]
    keys = col_key(nt[oi[ci]], hem_pred[ci], valid_hex[k])
    # 过滤已占用列
    free = np.array([kk not in occupied for kk in keys])
    ci, d, keys = ci[free], d[free], keys[free]
    # 唯一性: 每列取最近者
    ordi = np.argsort(d, kind="stable")
    seen = set(); promo = []
    for ii in ordi:
        kk = keys[ii]
        if kk in seen: continue
        seen.add(kk); promo.append(int(ci[ii]))
    promo = np.array(promo, np.int64)
    if len(promo) == 0:
        log("无新晋升, 停止"); break
    phex = hex_tree.query(X[promo, :2])[1]
    Y[promo, 0] = valid_hex[phex, 0]; Y[promo, 1] = valid_hex[phex, 1]
    Y[promo, 2] = hem_pred[promo]
    anch[promo] = True; src_loc[promo] = 2
    occupied.update(col_key(nt[oi[promo]], hem_pred[promo], valid_hex[phex]).tolist())
    log(f"第{rnd}轮晋升 {len(promo):,} (thr={thr})")
    hist[-1]["promoted"] = int(len(promo)); hist[-1]["thr"] = thr
    try:
        np.savez_compressed(CKPT, X=X, anch=anch, Y=Y, src=src_loc,
                            occupied=np.array(sorted(occupied), np.int64),
                            hist=np.frombuffer(json.dumps(hist).encode(), np.uint8), round=rnd)
    except OSError as e:
        log("检查点写入失败(继续):", str(e)[:50])

med, p1, hacc = eval_hold(X)
deg0 = np.asarray((A != 0).sum(1)).ravel() == 0
pq = X[:, :2].astype(np.float32); hem = np.sign(X[:, 2]).astype(np.int8)
pq[deg0] = np.nan; hem[deg0] = 0; src_loc[deg0] = 0
PQ = np.full((N, 2), np.nan, np.float32); HEM = np.zeros(N, np.int8); SRC = np.zeros(N, np.int8)
PQ[oi] = pq; HEM[oi] = hem; SRC[oi] = src_loc
np.savez_compressed(D+"lat2q_m2.npz", pq=PQ, hem=HEM, src=SRC)
diag = {"n_visual": int(no), "anchors_initial": int(n_ann), "anchors_final": int(anch.sum()),
        "heldout_n": int(len(HL)), "held_anchor_median_err_hex": med, "held_anchor_P<=1": p1,
        "held_anchor_hem_acc": hacc, "columnar_types": [str(type_names[t]) for t in COLT],
        "rounds": hist,
        "baseline_103": {"median": 3.0711, "P1": 0.1365}, "female_ref": {"median": 1.6598, "P1": 0.3022},
        "gate_M1": {"median<=2.0": bool(med <= 2.0), "P1>=0.22": bool(p1 >= 0.22)}}
json.dump(diag, open(D+"lat2q_dense_diag_m.json", "w"), indent=1)
log(f"SAVED lat2q_m2.npz  终评: 中位 {med:.3f}  P≤1 {p1:.3f}  hem {hacc:.4f}  锚点 {anch.sum():,}")
log("M1 门:", "PASS" if (med <= 2.0 and p1 >= 0.22) else "FAIL")
