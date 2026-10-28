#!/usr/bin/env python3
"""103_male_latq.py — 雄侧谐扩展 (M2a): hex 柱锚点沿视觉连接图扩散
雌性 93c 的雄性版: 视觉集 = {13 ol_intrinsic, 14 ol_sensory, 18 visual_centrifugal, 19 visual_projection}
锚点: body-annotations 的 assignedOlHex1/2 (p,q) + somaSide (hem)
泄漏控制: rng42 视觉边集 90/10 拟合划分; rng7 扣 10% 锚点做恢复检验
输出: data_male/lat2q_m.npz {pq:(N,2) f32, hem:(N,) i8} + lat2q_diag_m.json
"""
import sys; sys.path.insert(0, "/mnt/agents/pylibs")
import numpy as np, json, time
import pandas as pd
from scipy.sparse import csr_matrix
R = "/mnt/agents/output/果蝇生成器/"
D = R + "data_male/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

ep = np.fromfile(D+"e_pre.i32", dtype=np.int32)
eq = np.fromfile(D+"e_post.i32", dtype=np.int32)
ew = np.fromfile(D+"e_w.i32", dtype=np.int32).astype(np.float64)
sc = np.fromfile(D+"node_sc.i32", dtype=np.int32)
N = len(sc)
VIS_SC = {13, 14, 18, 19}
vis = np.isin(sc, list(VIS_SC))
oi = np.where(vis)[0]; no = len(oi)
log(f"male visual-set cells: {no:,}")

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

# ---- 锚点 ----
ann = pd.read_feather(R+"data/body-annotations.feather",
                      columns=["bodyId","assignedOlHex1","assignedOlHex2","somaSide"])
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
log(f"anchors: {len(lok):,}")

rng2 = np.random.default_rng(7)
hold = rng2.random(len(lok)) < 0.10
anch = np.zeros(no, bool); anch[lok[~hold]] = True
Y = np.zeros((no, 3)); Y[lok[~hold]] = labels[~hold]

X = Y.copy()
for it in range(1500):
    Xn = Dinv[:, None] * (A @ X)
    Xn[anch] = Y[anch]
    mx = np.abs(Xn - X).max()
    X = Xn
    if mx < 1e-4: break
log(f"converged it={it} maxdelta={mx:.2e}")

hl = lok[hold]; true = labels[hold]
err = np.sqrt(((X[hl, :2] - true[:, :2])**2).sum(1))
hem_acc = float((np.sign(X[hl, 2]) == true[:, 2]).mean())
log(f"heldout anchors: median err {np.median(err):.3f} hex, P<=1 {np.mean(err<=1):.3f}, hem {hem_acc:.4f}")

pq = X[:, :2].astype(np.float32); hem = np.sign(X[:, 2]).astype(np.int8)
deg0 = np.asarray((A != 0).sum(1)).ravel() == 0
pq[deg0] = np.nan; hem[deg0] = 0
PQ = np.full((N, 2), np.nan, np.float32); HEM = np.zeros(N, np.int8)
PQ[oi] = pq; HEM[oi] = hem

scn = np.load(D+"sc_names_m.npy", allow_pickle=True)
cov = {}
for s in sorted(VIS_SC):
    m = sc == s
    cov[str(scn[s])] = [int(m.sum()), float((HEM[m] != 0).mean())]
diag = {"n_visual": int(no), "anchors": int(len(lok)),
        "held_anchor_median_err_hex": float(np.median(err)),
        "held_anchor_P<=1": float(np.mean(err<=1)),
        "held_anchor_hem_acc": hem_acc, "coverage_by_sc": cov, "iters": int(it)}
np.savez_compressed(D+"lat2q_m.npz", pq=PQ, hem=HEM)
json.dump(diag, open(D+"lat2q_diag_m.json", "w"), indent=1)
log(f"SAVED lat2q_m.npz cov={cov}")
