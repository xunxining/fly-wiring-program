#!/usr/bin/env python3
# 113_male_tgap.py — 雄性 T 缺口分解诊断: T_real + 按端点区域块(real vs gen)闭合率
# 估计口径与 106_male_validate.py 一致(节点均匀楔形, 对称化图)
import numpy as np, json, time, os
from scipy.sparse import csr_matrix

D = "/mnt/agents/output/果蝇生成器/data_male/"
import sys
G = "/mnt/agents/output/果蝇生成器/gen_edges/" + (sys.argv[1] if len(sys.argv) > 1 else "gm_p8c")
OUTJ = sys.argv[2] if len(sys.argv) > 2 else None
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)

node_sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
N = len(node_sc)
NSC = 28
OPTIC = {13, 14, 18, 19, 20}
VNC = {15, 16, 17, 21, 22, 23, 24, 25, 26, 27}
def region_of(sc):
    if sc in OPTIC: return 0   # OPTIC
    if sc in VNC: return 1     # VNC
    return 2                   # CENTRAL
reg_lut = np.array([region_of(i) for i in range(NSC)], np.int8)
node_reg = reg_lut[node_sc]
REGN = ["OPTIC", "VNC", "CENTRAL"]

def analyze(pre_f, post_f, nsamp=500_000, seed=1):
    pre = np.fromfile(pre_f, dtype=np.int32); post = np.fromfile(post_f, dtype=np.int32)
    lo = np.minimum(pre, post); hi = np.maximum(pre, post)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    # 唯一无向边键(闭合一查)
    ek = lo.astype(np.int64) * N + hi
    ek = np.unique(ek)
    del pre, post, lo, hi, r, c, A
    rng = np.random.default_rng(seed)
    cand = np.where(np.diff(indptr) >= 2)[0]
    # 向量化楔形抽样
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
    del ek, indptr, indices
    return dict(U=U, V=V, W=W, closed=closed)

out = {}
for tag, (pf, qf) in {"real": (D + "e_pre.i32", D + "e_post.i32"),
                      "gen": (G + "_pre.i32", G + "_post.i32")}.items():
    log(f"分析 {tag} ...")
    R = analyze(pf, qf)
    out[tag] = R
    log(f"{tag} T = {R['closed'].mean():.4f} (n={len(R['closed']):,})")

# 块分解: 端点区域对 (无序) × 闭合率
res = {"T_real": float(out["real"]["closed"].mean()),
       "T_gen": float(out["gen"]["closed"].mean()),
       "blocks": [], "sc_pairs": []}
gap_total = res["T_real"] - res["T_gen"]
rows = []
for i in range(3):
    for j in range(i, 3):
        mr = (node_reg[out["real"]["V"]] == i) & (node_reg[out["real"]["W"]] == j) | \
             (node_reg[out["real"]["V"]] == j) & (node_reg[out["real"]["W"]] == i)
        mg = (node_reg[out["gen"]["V"]] == i) & (node_reg[out["gen"]["W"]] == j) | \
             (node_reg[out["gen"]["V"]] == j) & (node_reg[out["gen"]["W"]] == i)
        nr, ng = int(mr.sum()), int(mg.sum())
        pr = float(out["real"]["closed"][mr].mean()) if nr else float("nan")
        pg = float(out["gen"]["closed"][mg].mean()) if ng else float("nan")
        share = nr / len(out["real"]["closed"])
        rows.append(dict(block=f"{REGN[i]}-{REGN[j]}", n_real=nr, n_gen=ng,
                         share_real=round(share, 4), share_gen=round(ng / len(out["gen"]["closed"]), 4),
                         p_real=round(pr, 4), p_gen=round(pg, 4),
                         ratio=round(pr / pg, 2) if pg > 0 else None,
                         gap_contrib=round(share * (pr - pg), 5)))
rows.sort(key=lambda x: -x["gap_contrib"])
res["blocks"] = rows
log("区域块分解(按绝对缺口贡献排序):")
for r_ in rows:
    print(f"  {r_['block']:16s} share={r_['share_real']:.3f}/{r_['share_gen']:.3f} "
          f"p={r_['p_real']:.3f}/{r_['p_gen']:.3f} ratio={r_['ratio']} gap={r_['gap_contrib']:+.5f}", flush=True)
print(f"  T_real={res['T_real']:.4f} T_gen={res['T_gen']:.4f} 总缺口={gap_total:+.4f} "
      f"缺口闭合率={res['T_gen']/res['T_real']*100:.1f}%", flush=True)

# sc 级 top 对
sp = {}
for tag in ("real", "gen"):
    sv, sw = node_sc[out[tag]["V"]], node_sc[out[tag]["W"]]
    key = np.minimum(sv, sw) * NSC + np.maximum(sv, sw)
    for k in np.unique(key):
        m = key == k
        d = sp.setdefault(int(k), {})
        d[tag + "_n"] = int(m.sum()); d[tag + "_p"] = float(out[tag]["closed"][m].mean())
scn = np.load(D + "sc_names_m.npy", allow_pickle=True)
sc_rows = []
for k, d in sp.items():
    if "real_n" in d and "gen_n" in d and d["real_n"] >= 3000 and d["gen_n"] >= 3000:
        share = d["real_n"] / len(out["real"]["closed"])
        sc_rows.append(dict(pair=f"{scn[k//NSC]}-{scn[k%NSC]}", n_real=d["real_n"], n_gen=d["gen_n"],
                            p_real=round(d["real_p"], 4), p_gen=round(d["gen_p"], 4),
                            ratio=round(d["real_p"] / d["gen_p"], 2) if d["gen_p"] > 0 else None,
                            gap_contrib=round(share * (d["real_p"] - d["gen_p"]), 5)))
sc_rows.sort(key=lambda x: -x["gap_contrib"])
res["sc_pairs"] = sc_rows[:15]
log("sc 级 top10 (按绝对缺口贡献):")
for r_ in sc_rows[:10]:
    print(f"  {r_['pair']:38s} share_n={r_['n_real']:,} p={r_['p_real']:.3f}/{r_['p_gen']:.3f} "
          f"ratio={r_['ratio']} gap={r_['gap_contrib']:+.5f}", flush=True)

_out = OUTJ or (D + "tgap_m.json")
with open(_out, "w") as f:
    json.dump(res, f, indent=1)
log("-> " + _out)
