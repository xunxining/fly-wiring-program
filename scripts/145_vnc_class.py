#!/usr/bin/env python3
# 145_vnc_class.py — ②b-1c (§51): VNC 闭合的回路类别分解 + 楔强度剖面
import json, time
import numpy as np
from scipy.sparse import csr_matrix

ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + "data_male/"
G = ROOT + "gen_edges/g_p9bm2"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

node_sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
N = len(node_sc)
VNC = {15, 16, 17, 21, 22, 23, 24, 25, 26, 27}
is_vnc = np.isin(node_sc, list(VNC))
CLS = np.full(N, -1, dtype=np.int8)          # 0 SENS, 1 IN, 2 MN, 3 EFF, 4 OTHER
for sc, c in [(25, 0), (26, 0), (15, 0), (16, 0), (17, 0), (23, 1), (24, 2), (21, 3), (22, 4), (27, 4)]:
    CLS[node_sc == sc] = c
CN = ["SENS", "IN", "MN", "EFF", "OTHER"]

def analyze(pre_f, post_f, w_f, nsamp, seed):
    pre = np.fromfile(pre_f, dtype=np.int32); post = np.fromfile(post_f, dtype=np.int32)
    w = np.fromfile(w_f, dtype=np.float32) if w_f.endswith(".f32") else np.fromfile(w_f, dtype=np.int32)
    lo = np.minimum(pre, post); hi = np.maximum(pre, post)
    # 无向权重 = 双向求和
    key = lo.astype(np.int64) * N + hi
    uk, inv = np.unique(key, return_inverse=True)
    uw = np.zeros(len(uk), np.float64)
    np.add.at(uw, inv, w)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    ek = uk
    ekw = uw
    del pre, post, lo, hi, r, c, A, key, inv, uk, uw, w
    rng = np.random.default_rng(seed)
    cand = np.where(np.diff(indptr) >= 2)[0]
    U = np.empty(0, np.int64); V = np.empty(0, np.int64); W = np.empty(0, np.int64)
    need = nsamp
    while len(U) < nsamp:
        u = cand[rng.integers(0, len(cand), need)]
        s0, s1 = indptr[u], indptr[u + 1]
        ln = (s1 - s0)
        o1 = rng.integers(0, ln); o2 = rng.integers(0, ln)
        v = indices[s0 + o1]; w2 = indices[s0 + o2]
        ok = v != w2
        U = np.concatenate([U, u[ok]]); V = np.concatenate([V, v[ok]]); W = np.concatenate([W, w2[ok]])
    U, V, W = U[:nsamp], V[:nsamp], W[:nsamp]
    a = np.minimum(V, W).astype(np.int64); b = np.maximum(V, W).astype(np.int64)
    kk = a * N + b
    hit = np.searchsorted(ek, kk)
    closed = (hit < len(ek)) & (ek[np.clip(hit, 0, len(ek) - 1)] == kk)
    # 楔两沿权重
    k1 = np.minimum(U, V).astype(np.int64) * N + np.maximum(U, V)
    k2 = np.minimum(U, W).astype(np.int64) * N + np.maximum(U, W)
    w1 = ekw[np.searchsorted(ek, k1)]
    w2v = ekw[np.searchsorted(ek, k2)]
    del ek, ekw, indptr, indices
    return dict(V=V, W=W, closed=closed, w1=w1, w2=w2v)

NS = 6_000_000
log("抽样 real ..."); re_ = analyze(D + "e_pre.i32", D + "e_post.i32", D + "e_w.i32", NS, 1)
log("抽样 gen  ..."); ge = analyze(G + "_pre.i32", G + "_post.i32", G + "_w.i32", NS, 1)

m_r = is_vnc[re_["V"]] & is_vnc[re_["W"]]
m_g = is_vnc[ge["V"]] & is_vnc[ge["W"]]
log(f"VNC-VNC 楔: real {m_r.sum():,} gen {m_g.sum():,}")

# ---------- T-C: 类别分解 ----------
cr, cg = re_["closed"][m_r], ge["closed"][m_g]
pr = np.minimum(CLS[re_["V"][m_r]], CLS[re_["W"][m_r]]) * 5 + np.maximum(CLS[re_["V"][m_r]], CLS[re_["W"][m_r]])
pairs = [(i, j) for i in range(5) for j in range(i, 5)]
rows = []
for i, j in pairs:
    mr = (pr == i * 5 + j)
    # gen 端用同一对类别
    pg = np.minimum(CLS[ge["V"][m_g]], CLS[ge["W"][m_g]]) * 5 + np.maximum(CLS[ge["V"][m_g]], CLS[ge["W"][m_g]])
    mg = (pg == i * 5 + j)
    if mr.sum() < 200: continue
    pr_, pg_ = cr[mr].mean(), cg[mg].mean()
    share_wedge = mr.sum() / m_r.sum()
    gap = share_wedge * (pr_ - pg_)
    rows.append(dict(pair=f"{CN[i]}-{CN[j]}", n_real=int(mr.sum()),
                     p_real=float(pr_), p_gen=float(pg_), ratio=float(pr_ / pg_),
                     wedge_share=float(share_wedge), gap_contrib=float(gap)))
tot_gap = sum(r["gap_contrib"] for r in rows)
for r in rows: r["gap_share"] = r["gap_contrib"] / tot_gap
rows.sort(key=lambda r: -r["gap_share"])
top = rows[0]
log("T-C top:", top["pair"], f"ratio {top['ratio']:.2f} gap_share {top['gap_share']:.2f}")

# ---------- T-S: 强度剖面 ----------
STR = 5.0
def sbin(w1, w2):
    s = np.zeros(len(w1), np.int8)
    s[(w1 >= STR) | (w2 >= STR)] = 1
    s[(w1 >= STR) & (w2 >= STR)] = 2
    return s
sr, sg = sbin(re_["w1"][m_r], re_["w2"][m_r]), sbin(ge["w1"][m_g], ge["w2"][m_g])
prof = []
for b, nm in enumerate(["weak-weak", "mixed", "strong-strong"]):
    pr_ = cr[sr == b].mean() if (sr == b).sum() else np.nan
    pg_ = cg[sg == b].mean() if (sg == b).sum() else np.nan
    share = (sr == b).sum() / m_r.sum()
    prof.append(dict(bin=nm, n_real=int((sr == b).sum()), p_real=float(pr_), p_gen=float(pg_),
                     ratio=float(pr_ / pg_), wedge_share=float(share),
                     gap_contrib=float(share * (pr_ - pg_))))
tot = sum(p["gap_contrib"] for p in prof)
for p in prof: p["gap_share"] = p["gap_contrib"] / tot
grad_r = prof[2]["p_real"] / prof[0]["p_real"]
grad_g = prof[2]["p_gen"] / prof[0]["p_gen"]
log(f"T-S: real {prof[0]['p_real']:.3f}->{prof[2]['p_real']:.3f} ({grad_r:.2f}x), gen ({grad_g:.2f}x), ss gap {prof[2]['gap_share']:.2f}")

out = dict(protocol="§51 ②b-1c class decomposition + strength profile", nsamp=NS,
           TC=dict(rows=rows, top_pair=top["pair"], top_ratio=top["ratio"], top_gap_share=top["gap_share"]),
           TS=dict(bins=prof, grad_real=float(grad_r), grad_gen=float(grad_g),
                   strong_gap_share=prof[2]["gap_share"]),
           gates=dict(G1c=bool(top["gap_share"] >= 0.40 and top["ratio"] >= 2.5),
                      G2c=bool(grad_r >= 2.0 and grad_g <= 1.3 and prof[2]["gap_share"] >= 0.35)))
json.dump(out, open(D + "vnc_class.json", "w"), indent=1)
log("-> data_male/vnc_class.json")
print(json.dumps(out["gates"], indent=1))
for r in rows[:6]:
    print(f"  {r['pair']:10s} p {r['p_real']:.3f}/{r['p_gen']:.3f} = {r['ratio']:.2f}x  gap_share {r['gap_share']:.2f}")
for p in prof:
    print(f"  {p['bin']:14s} p {p['p_real']:.3f}/{p['p_gen']:.3f} = {p['ratio']:.2f}x  gap_share {p['gap_share']:.2f}")
