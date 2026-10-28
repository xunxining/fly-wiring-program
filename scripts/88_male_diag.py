#!/usr/bin/env python3
"""88_male_diag.py — 雄性 MaleCNS γ×神经毡诊断 (复刻雌性 81 的 D1/D2/D3)
输出: data_census/gamma_neuropil_diag_male.json
"""
import numpy as np, json, time
from scipy.sparse import csr_matrix
ROOT = "/mnt/agents/output/果蝇生成器/"
OUTD = ROOT + "data_census/"
t0 = time.time()

ep = np.fromfile(OUTD + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(OUTD + "e_post.i32", dtype=np.int32)
node_type = np.fromfile(OUTD + "node_type.i32", dtype=np.int32)
N = len(node_type)
E = len(ep)
print(f"N={N:,} E={E:,}", flush=True)

_z = np.load(OUTD + "np_masks_male.npz")
OUTM, INM = _z["OUT"], _z["IN"]          # (N,K) bool
AM = OUTM | INM                           # 无向 arbor 掩膜
K = AM.shape[1]
res = {"N": int(N), "E": int(E), "K_roi": int(K)}
rng = np.random.default_rng(0)

# ---------- D1: 边重叠 vs 同型零模型 ----------
ns = 1_000_000
idx = rng.integers(0, E, ns)
ov_real = (OUTM[ep[idx]] & INM[eq[idx]]).sum(1)   # 有向口径: pre 的 OUT ∩ post 的 IN
tj = node_type[eq[idx]]
# 同型随机靶: 预建类型成员表直接抽样
Tmax = int(node_type.max()) + 1
members = [np.flatnonzero(node_type == t) for t in range(Tmax)]
cj = np.empty(ns, np.int64)
for t in np.unique(tj):
    m = tj == t; mem = members[t]
    cj[m] = mem[rng.integers(0, len(mem), int(m.sum()))] if len(mem) else 0
ov_null = (OUTM[ep[idx]] & INM[cj]).sum(1)
res["D1"] = {
    "edge_ov0_frac": float((ov_real == 0).mean()),
    "null_ov0_frac": float((ov_null == 0).mean()),
    "edge_ov_mean": float(ov_real.mean()),
    "null_ov_mean": float(ov_null.mean()),
}
print("D1:", res["D1"], flush=True)
del ov_real, ov_null, idx, cj, tj

# ---------- D2: 楔形闭合率 × 无向重叠 ----------
lo_ = np.minimum(ep, eq).astype(np.int64); hi_ = np.maximum(ep, eq).astype(np.int64)
r = np.concatenate([lo_, hi_]); c = np.concatenate([hi_, lo_])
A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
A.data[:] = 1
indptr, indices = A.indptr, A.indices
deg = np.diff(indptr)
cand = np.where(deg >= 2)[0]
bins = [0, 1, 2, 3, 5, 8, 15, 100]
TRIALS = 300_000
tot = np.zeros(len(bins) - 1, np.int64); cls = np.zeros(len(bins) - 1, np.int64)
t = 0
while t < TRIALS:
    u = int(cand[rng.integers(0, len(cand))])
    ns_ = indices[indptr[u]:indptr[u + 1]]
    if len(ns_) < 2: continue
    v, w_ = ns_[rng.integers(0, len(ns_), 2)]
    if v == w_: continue
    ov = int((AM[v] & AM[w_]).sum())
    bi = np.searchsorted(bins, ov, side="right") - 1
    a_, b_ = (v, w_) if v < w_ else (w_, v)
    hit = np.searchsorted(indices[indptr[a_]:indptr[a_ + 1]], b_)
    cl = indptr[a_] + hit < indptr[a_ + 1] and indices[indptr[a_] + hit] == b_
    tot[bi] += 1; cls[bi] += cl; t += 1
labels = ["[0,1)", "[1,2)", "[2,3)", "[3,5)", "[5,8)", "[8,15)", "[15,100)"]
res["D2"] = {"nsamp": TRIALS, "T_check": float(cls.sum() / tot.sum()),
             "closure_by_overlap": [{"ov": labels[i], "n": int(tot[i]), "closure": float(cls[i] / max(tot[i], 1))} for i in range(len(labels))],
             "p_wedge_ov0": float(tot[0] / tot.sum())}
print("D2:", json.dumps(res["D2"])[:400], flush=True)

# ---------- D3: 闭合三角的单神经毡共定位 ----------
tri = 0; same = 0
t = 0
while t < 60_000:
    u = int(cand[rng.integers(0, len(cand))])
    ns_ = indices[indptr[u]:indptr[u + 1]]
    if len(ns_) < 2: continue
    v, w_ = ns_[rng.integers(0, len(ns_), 2)]
    if v == w_: continue
    a_, b_ = (v, w_) if v < w_ else (w_, v)
    hit = np.searchsorted(indices[indptr[a_]:indptr[a_ + 1]], b_)
    if not (indptr[a_] + hit < indptr[a_ + 1] and indices[indptr[a_] + hit] == b_):
        t += 1; continue
    tri += 1
    inter = AM[u] & AM[v] & AM[w_]
    if inter.any(): same += 1
    t += 1
res["D3"] = {"closed_tri": int(tri), "same_neuropil_frac": float(same / max(tri, 1))}
print("D3:", res["D3"], flush=True)

json.dump(res, open(OUTD + "gamma_neuropil_diag_male.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED  {time.time()-t0:.0f}s")
