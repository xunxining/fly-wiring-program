#!/usr/bin/env python3
# 148_t_se.py — §56③: 头条 T 数字的 Monte-Carlo SE (113 楔抽样, 多种子配对设计, 增量落盘抗收割)
import json, os, time
import numpy as np
from scipy.sparse import csr_matrix

ROOT = "/mnt/agents/output/果蝇生成器/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

GRAPHS = {
    "female_real":  (ROOT + "data_female/", ROOT + "data_female/e_pre.i32", ROOT + "data_female/e_post.i32"),
    "female_flag":  (ROOT + "data_female/", ROOT + "gen_edges/g_mutq4_pre.i32", ROOT + "gen_edges/g_mutq4_post.i32"),
    "male_real":    (ROOT + "data_male/", ROOT + "data_male/e_pre.i32", ROOT + "data_male/e_post.i32"),
    "male_flag":    (ROOT + "data_male/", ROOT + "gen_edges/g_mutqm2_pre.i32", ROOT + "gen_edges/g_mutqm2_post.i32"),
    "male_p9b":     (ROOT + "data_male/", ROOT + "gen_edges/g_p9bm2_pre.i32", ROOT + "gen_edges/g_p9bm2_post.i32"),
    "male_p9d":     (ROOT + "data_male/", ROOT + "gen_edges/gm_p9d_pre.i32", ROOT + "gen_edges/gm_p9d_post.i32"),
    "male_p8c_raw": (ROOT + "data_male/", ROOT + "gen_edges/gm_p8c_pre.i32", ROOT + "gen_edges/gm_p8c_post.i32"),
    "female_lat":   (ROOT + "data_female/", ROOT + "gen_edges/g_snap2v2_pre.i32", ROOT + "gen_edges/g_snap2v2_post.i32"),
}
SEEDS = list(range(1, 13))
NS = 500_000
OUT = ROOT + "data_male/t_se_runs.jsonl"

done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        try:
            r = json.loads(ln); done.add((r["graph"], r["seed"]))
        except Exception: pass

def sample_T(ddir, pre_f, post_f, nsamp, seed):
    node_sc = np.fromfile(ddir + "node_sc.i32", dtype=np.int32)
    N = len(node_sc)
    pre = np.fromfile(pre_f, dtype=np.int32); post = np.fromfile(post_f, dtype=np.int32)
    lo = np.minimum(pre, post); hi = np.maximum(pre, post)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    ek = np.unique(lo.astype(np.int64) * N + hi)
    del pre, post, lo, hi, r, c, A
    rng = np.random.default_rng(seed)
    cand = np.where(np.diff(indptr) >= 2)[0]
    U = np.empty(0, np.int64); V = np.empty(0, np.int64); W = np.empty(0, np.int64)
    need = nsamp
    while len(U) < nsamp:
        u = cand[rng.integers(0, len(cand), need)]
        s0, s1 = indptr[u], indptr[u + 1]
        ln = (s1 - s0)
        o1 = rng.integers(0, ln); o2 = rng.integers(0, ln)
        v = indices[s0 + o1]; w = indices[s0 + o2]
        ok = v != w
        U = np.concatenate([U, u[ok]]); V = np.concatenate([V, v[ok]]); W = np.concatenate([W, w[ok]])
    U, V, W = U[:nsamp], V[:nsamp], W[:nsamp]
    a = np.minimum(V, W).astype(np.int64); b = np.maximum(V, W).astype(np.int64)
    kk = a * N + b
    hit = np.searchsorted(ek, kk)
    closed = (hit < len(ek)) & (ek[np.clip(hit, 0, len(ek) - 1)] == kk)
    return float(closed.mean())

for name, (ddir, pf, qf) in GRAPHS.items():
    for sd in SEEDS:
        if (name, sd) in done: continue
        T = sample_T(ddir, pf, qf, NS, sd)
        with open(OUT, "a") as f:
            f.write(json.dumps({"graph": name, "seed": sd, "T": T, "nsamp": NS}) + "\n")
        log(f"{name} seed={sd} T={T:.5f}")
log("ALL DONE")
