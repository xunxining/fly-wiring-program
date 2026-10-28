#!/usr/bin/env python3
"""81_neuropil_gamma.py — γ 的神经毡分层检验（雌性 FAFB v783, 第一步诊断）
注册预测: 真实闭合三角形的边集中在共享神经毡; 重叠(overlap)=0 的楔几乎不闭合。
输出: data_female/gamma_neuropil_diag.json
"""
import numpy as np, pandas as pd, json, time, sys
from scipy.sparse import csr_matrix

D = "/mnt/agents/output/果蝇生成器/data_female/"
CONN = "/tmp/np/conn.csv.gz"
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)

node_ids = np.fromfile(D + "node_ids.i64", dtype=np.int64)
N = len(node_ids)
order = np.argsort(node_ids); ids_sorted = node_ids[order]
ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
E0 = len(ep)
log(f"N={N} census edges E0={E0}")

# census pair keys, sorted for searchsorted mapping
K0 = ep.astype(np.int64) * N + eq.astype(np.int64)
korder = np.argsort(K0); K0s = K0[korder]
assert len(np.unique(K0)) == E0, "census pairs not unique!"

# global neuropil code dict
ncode = {}
def codes_for(s):
    out = np.empty(len(s), np.int16)
    for i, v in enumerate(s):
        c = ncode.get(v)
        if c is None:
            c = len(ncode); ncode[v] = c
        out[i] = c
    return out

POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
def popcount64(a):
    a = np.asarray(a, dtype=np.uint64)
    if a.ndim == 0:
        return int(a).bit_count()
    b = np.frombuffer(a.tobytes(), np.uint8).reshape(a.shape + (8,))
    return POP[b].sum(-1)

OUT_lo = np.zeros(N, np.uint64); OUT_hi = np.zeros(N, np.uint64)
IN_lo  = np.zeros(N, np.uint64); IN_hi  = np.zeros(N, np.uint64)
PM_lo  = np.zeros(E0, np.uint64); PM_hi = np.zeros(E0, np.uint64)
Psyn   = np.zeros(E0, np.int64)

rows_tot = 0; rows_kept = 0
t0 = time.time()
for ci, ch in enumerate(pd.read_csv(CONN, chunksize=2_000_000,
        usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
        dtype={"pre_root_id": np.int64, "post_root_id": np.int64,
               "neuropil": str, "syn_count": np.int64})):
    rows_tot += len(ch)
    pre_v = ch["pre_root_id"].to_numpy(); post_v = ch["post_root_id"].to_numpy()
    p1 = np.searchsorted(ids_sorted, pre_v); p2 = np.searchsorted(ids_sorted, post_v)
    p1c = np.clip(p1, 0, N-1); p2c = np.clip(p2, 0, N-1)
    ok = (ids_sorted[p1c] == pre_v) & (ids_sorted[p2c] == post_v)
    i = order[p1c[ok]]; j = order[p2c[ok]]
    cc = codes_for(ch["neuropil"].to_numpy()[ok])
    w = ch["syn_count"].to_numpy()[ok]
    rows_kept += len(i)
    lo = cc < 64
    bit_lo = np.where(lo, 1, 0).astype(np.uint64) << np.where(lo, cc, 0).astype(np.uint64)
    bit_hi = np.where(lo, 0, 1).astype(np.uint64) << np.where(lo, 0, cc-64).astype(np.uint64)
    np.bitwise_or.at(OUT_lo, i, bit_lo); np.bitwise_or.at(OUT_hi, i, bit_hi)
    np.bitwise_or.at(IN_lo, j, bit_lo);  np.bitwise_or.at(IN_hi, j, bit_hi)
    # pair-level: unique keys within chunk, OR masks, then merge into global arrays
    k = i.astype(np.int64) * N + j.astype(np.int64)
    srt = np.argsort(k); ks = k[srt]
    b0 = np.concatenate([[True], ks[1:] != ks[:-1]])
    starts = np.where(b0)[0]; ends = np.concatenate([starts[1:], [len(ks)]])
    ukey = ks[starts]
    ulo = np.bitwise_or.reduceat(bit_lo[srt], starts)
    uhi = np.bitwise_or.reduceat(bit_hi[srt], starts)
    uw = np.add.reduceat(w[srt], starts)
    pos = np.searchsorted(K0s, ukey)
    posc = np.clip(pos, 0, E0-1)
    hit = K0s[posc] == ukey
    gi = korder[posc[hit]]
    np.bitwise_or.at(PM_lo, gi, ulo[hit]); np.bitwise_or.at(PM_hi, gi, uhi[hit])
    np.add.at(Psyn, gi, uw[hit])
    if ci % 3 == 0:
        log(f"chunk {ci} rows={rows_tot/1e6:.1f}M kept={rows_kept/rows_tot:.3f} "
            f"np={len(ncode)} miss_pair={(~hit).sum()} t={time.time()-t0:.0f}s")

K = len(ncode)
log(f"READ DONE rows={rows_tot} kept={rows_kept} neuropils={K}")
assert K <= 128

# sanity: pair syn sums vs census e_w
ew = np.fromfile(D + "e_w.i32", dtype=np.int32).astype(np.int64)
covered = Psyn > 0
log(f"pair coverage: {covered.mean():.4f} of census edges have neuropil rows; "
    f"syn match: {np.abs(Psyn-ew).max() if covered.all() else 'NA (partial)'}")

res = {"N": int(N), "E": int(E0), "neuropils": K,
       "neuropil_names": {v: k for k, v in ncode.items()},
       "rows_total": int(rows_tot), "rows_kept": int(rows_kept),
       "pair_coverage": float(covered.mean())}

# 立即持久化掩码（防后续 OOM 丢失）
np.savez_compressed(D + "np_masks.npz", OUT_lo=OUT_lo, OUT_hi=OUT_hi,
                    IN_lo=IN_lo, IN_hi=IN_hi, PM_lo=PM_lo, PM_hi=PM_hi,
                    Psyn=Psyn, ncode=json.dumps({v: k for k, v in ncode.items()}))
log("masks saved -> np_masks.npz")

# ---------- D1: 真实边的 overlap 富集 vs 同型配额零模型 ----------
rng = np.random.default_rng(0)
node_type = np.fromfile(D + "node_type.i32", dtype=np.int32)
types = np.load(D + "type_names.npy", allow_pickle=True)
T = len(types)
tc = np.bincount(node_type, minlength=T)
toff = np.concatenate([[0], np.cumsum(tc)])

def ov(a, b):  # popcount of shared neuropils (a 输出 ∩ b 输入)
    return popcount64((OUT_lo[a] & IN_lo[b]) | ((OUT_hi[a] & IN_hi[b]) << 0)) \
           + 0  # hi part shifted separately below

def overlap_dir(a, b):
    return popcount64(OUT_lo[a] & IN_lo[b]) + popcount64(OUT_hi[a] & IN_hi[b])

S = 1_000_000
es = rng.integers(0, E0, S)
i_s = ep[es].astype(np.int64); j_s = eq[es].astype(np.int64)
ov_real = overlap_dir(i_s, j_s)
# null: j' 同型随机
tj = node_type[j_s]
cj = rng.random(S)
j_null = (toff[tj] + (cj * tc[tj]).astype(np.int64))
ov_null = overlap_dir(i_s, j_null)
res["D1"] = {
  "ov_real_mean": float(ov_real.mean()), "ov_null_mean": float(ov_null.mean()),
  "p_ov0_real": float((ov_real == 0).mean()), "p_ov0_null": float((ov_null == 0).mean()),
  "ov_real_q": [float(np.quantile(ov_real, q)) for q in [0.5, 0.9, 0.99]],
  "ov_null_q": [float(np.quantile(ov_null, q)) for q in [0.5, 0.9, 0.99]],
}
log("D1:", {k: round(v, 4) if isinstance(v, float) else v for k, v in res["D1"].items()})

# 边本身的神经毡广度
pm_pc = popcount64(PM_lo) + popcount64(PM_hi)
res["edge_np_breadth"] = {"median": float(np.median(pm_pc)),
                          "p_eq1": float((pm_pc == 1).mean()),
                          "p_le2": float((pm_pc <= 2).mean())}
log("edge neuropil breadth:", res["edge_np_breadth"])

# ---------- D2: 楔形闭合率 vs overlap ----------
import gc
del K0, Psyn, ew, pm_pc, ov_real, ov_null, i_s, j_s, tj, cj, j_null
gc.collect()
# 无向化(与 T 估计同口径)
lo_ = np.minimum(ep, eq); hi_ = np.maximum(ep, eq)
r = np.concatenate([lo_, hi_]).astype(np.int64)
c = np.concatenate([hi_, lo_]).astype(np.int64)
A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
A.data[:] = 1
indptr, indices = A.indptr, A.indices
deg = np.diff(indptr)
cand = np.where(deg >= 2)[0]

def overlap_undir(a, b):
    am = (OUT_lo[a] | IN_lo[a], OUT_hi[a] | IN_hi[a])
    bm = (OUT_lo[b] | IN_lo[b], OUT_hi[b] | IN_hi[b])
    return popcount64(am[0] & bm[0]) + popcount64(am[1] & bm[1])

nsamp = 300_000
closed = np.zeros(nsamp, bool); ovv = np.zeros(nsamp, np.int16)
t = 0
while t < nsamp:
    u = int(cand[rng.integers(0, len(cand))])
    ns_ = indices[indptr[u]:indptr[u+1]]
    if len(ns_) < 2: continue
    v, w_ = ns_[rng.integers(0, len(ns_), 2)]
    if v == w_: continue
    a_, b_ = (v, w_) if v < w_ else (w_, v)
    hit = np.searchsorted(indices[indptr[a_]:indptr[a_+1]], b_)
    cl = indptr[a_]+hit < indptr[a_+1] and indices[indptr[a_]+hit] == b_
    closed[t] = cl; ovv[t] = overlap_undir(np.int64(v), np.int64(w_))
    t += 1
res["D2"] = {"nsamp": nsamp, "T_check": float(closed.mean())}
bins = [0, 1, 2, 3, 5, 8, 15, 100]
by = []
for b0_, b1_ in zip(bins[:-1], bins[1:]):
    m = (ovv >= b0_) & (ovv < b1_)
    if m.sum() > 50:
        by.append({"ov": f"[{b0_},{b1_})", "n": int(m.sum()),
                   "closure": float(closed[m].mean())})
res["D2"]["closure_by_overlap"] = by
res["D2"]["p_wedge_ov0"] = float((ovv == 0).mean())
log("D2 closure by overlap:", [(b["ov"], round(b["closure"], 4)) for b in by])

# ---------- D3: 闭合三角形的同神经毡集中度 ----------
# 对闭合楔 (u 与 v,w 相邻, v-w 闭合): 三边是否有共同神经毡
# 边 -> census 位置映射 (无向: 两端都查)
Kmap = {}
def pair_pm(a, b):
    k1 = a.astype(np.int64)*N + b.astype(np.int64)
    p1 = np.searchsorted(K0s, k1); p1c = np.clip(p1, 0, E0-1)
    h1 = K0s[p1c] == k1
    k2 = b.astype(np.int64)*N + a.astype(np.int64)
    p2 = np.searchsorted(K0s, k2); p2c = np.clip(p2, 0, E0-1)
    h2 = K0s[p2c] == k2
    idx = np.where(h1, korder[p1c], np.where(h2, korder[p2c], -1))
    return idx

ntri = 0; same_np = 0; trials = 0
t = 0
while trials < 60_000:
    u = int(cand[rng.integers(0, len(cand))])
    ns_ = indices[indptr[u]:indptr[u+1]]
    if len(ns_) < 2: continue
    v, w_ = ns_[rng.integers(0, len(ns_), 2)]
    if v == w_: continue
    a_, b_ = (v, w_) if v < w_ else (w_, v)
    hit = np.searchsorted(indices[indptr[a_]:indptr[a_+1]], b_)
    if not (indptr[a_]+hit < indptr[a_+1] and indices[indptr[a_]+hit] == b_):
        trials += 1; continue
    # 闭合三角形 u-v, u-w, v-w
    e1 = pair_pm(np.int64(min(u,v)), np.int64(max(u,v)))
    e2 = pair_pm(np.int64(min(u,w_)), np.int64(max(u,w_)))
    e3 = pair_pm(np.int64(a_), np.int64(b_))
    if e1 >= 0 and e2 >= 0 and e3 >= 0:
        mlo = PM_lo[e1] & PM_lo[e2] & PM_lo[e3]
        mhi = PM_hi[e1] & PM_hi[e2] & PM_hi[e3]
        ntri += 1
        if mlo != 0 or mhi != 0: same_np += 1
    trials += 1
res["D3"] = {"closed_tris": ntri, "same_neuropil_frac": float(same_np/max(ntri,1))}
log("D3 same-neuropil triangle frac:", res["D3"])

json.dump(res, open(D + "gamma_neuropil_diag.json", "w"), indent=1)
log("DONE -> gamma_neuropil_diag.json")
