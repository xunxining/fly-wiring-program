#!/usr/bin/env python3
# 143_vnc_lineage.py — ②b-1 (§47): VNC-VNC 闭合的半血统(hemilineage)集中度诊断
# 楔抽样与 113 逐行相同(无向化/中心度≥2/节点均匀), 过滤两端点皆 VNC 且皆已解析 HL
import json, re, time
import numpy as np
from scipy.sparse import csr_matrix

ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + "data_male/"
G = ROOT + "gen_edges/g_p9bm2"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

node_sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
node_ty = np.fromfile(D + "node_type.i32", dtype=np.int32)
tn = np.load(D + "type_names.npy", allow_pickle=True)
N = len(node_sc)
VNC = {15, 16, 17, 21, 22, 23, 24, 25, 26, 27}          # 与 113 一致
is_vnc = np.isin(node_sc, list(VNC))

# ---- 半血统解析: ^(IN|AN|EN|TN)(\d{2}[AB]) ----
pat = re.compile(r"^(?:IN|AN|EN|TN)(\d{2}[AB])")
hl_of_type = {}
for t in range(len(tn)):
    m = pat.match(str(tn[t]))
    if m: hl_of_type[t] = m.group(1)
HL = np.full(N, -1, dtype=np.int32)
hl_ids = {h: i for i, h in enumerate(sorted(set(hl_of_type.values())))}
for t, h in hl_of_type.items():
    HL[node_ty == t] = hl_ids[h]
parsed = HL >= 0
log(f"VNC cells {is_vnc.sum():,}; HL-parsed 全体 {parsed.sum():,} / VNC 内 {(parsed & is_vnc).sum():,} "
    f"({(parsed & is_vnc).sum()/max(is_vnc.sum(),1):.1%}); 血统数 {len(hl_ids)}")

def analyze(pre_f, post_f, nsamp, seed):
    pre = np.fromfile(pre_f, dtype=np.int32); post = np.fromfile(post_f, dtype=np.int32)
    lo = np.minimum(pre, post); hi = np.maximum(pre, post)
    r = np.concatenate([lo, hi]); c = np.concatenate([hi, lo])
    A = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(N, N))
    A.data[:] = 1
    indptr, indices = A.indptr, A.indices
    ek = lo.astype(np.int64) * N + hi
    ek = np.unique(ek)
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
    del ek, indptr, indices
    return dict(U=U, V=V, W=W, closed=closed)

def cond_stats(V, W, closed, hl_arr):
    m = is_vnc[V] & is_vnc[W] & (hl_arr[V] >= 0) & (hl_arr[W] >= 0)
    V, W, cl = V[m], W[m], closed[m]
    same = hl_arr[V] == hl_arr[W]
    st = node_ty[V] == node_ty[W]
    def p(mask):
        return (cl[mask].mean() if mask.sum() else float("nan")), int(mask.sum())
    p_same, n_same = p(same)
    p_diff, n_diff = p(~same)
    p_st, n_st = p(st)
    return dict(n_wedges=int(m.sum()), p_close_sameHL=p_same, n_sameHL=n_same,
                p_close_diffHL=p_diff, n_diffHL=n_diff,
                p_close_sametype=p_st, n_sametype=n_st,
                sameHL_share=float(same.mean()),
                closed_deficit_sameHL=float((~cl[same]).sum()),   # 供缺口分解
                closed_same=float(cl[same].sum()), closed_diff=float(cl[~same].sum()))

NS = 6_000_000
log("抽样 real ..."); re_ = analyze(D + "e_pre.i32", D + "e_post.i32", NS, seed=1)
log("抽样 gen(g_p9bm2) ..."); ge = analyze(G + "_pre.i32", G + "_post.i32", NS, seed=1)

rs = cond_stats(re_["V"], re_["W"], re_["closed"], HL)
gs = cond_stats(ge["V"], ge["W"], ge["closed"], HL)
E_real = rs["p_close_sameHL"] / rs["p_close_diffHL"]
E_gen = gs["p_close_sameHL"] / gs["p_close_diffHL"]
# 缺口分解: 缺额楔 = real 闭合率 - gen 闭合率, 按楔数加权
gap_same = rs["sameHL_share"] * (rs["p_close_sameHL"] - gs["p_close_sameHL"])
gap_diff = (1 - rs["sameHL_share"]) * (rs["p_close_diffHL"] - gs["p_close_diffHL"])
gap_total = gap_same + gap_diff
same_gap_share = gap_same / gap_total if gap_total > 0 else float("nan")

# ---- 零分布: HL 标签在 VNC 已解析细胞间置换 ×200 (复用 real 楔) ----
rng = np.random.default_rng(7)
m = is_vnc[re_["V"]] & is_vnc[re_["W"]] & parsed[re_["V"]] & parsed[re_["W"]]
Vr, Wr, clr = re_["V"][m], re_["W"][m], re_["closed"][m]
cells = np.where(parsed & is_vnc)[0]
nulls = np.empty(200)
for i in range(200):
    perm = rng.permutation(len(cells))
    hlp = np.full(N, -1, dtype=np.int32); hlp[cells] = HL[cells[perm]]
    same = hlp[Vr] == hlp[Wr]
    nulls[i] = clr[same].mean() / clr[~same].mean()
log(f"null done: mean {nulls.mean():.3f} p99 {np.percentile(nulls, 99):.3f}")

out = dict(protocol="§47 ②b-1 hemilineage concentration diagnostic",
           nsamp=NS, vnc_cells=int(is_vnc.sum()),
           vnc_hl_parsed=int((parsed & is_vnc).sum()), n_hemilineages=len(hl_ids),
           real=rs, gen=gs, E_real=E_real, E_gen=E_gen,
           E_ratio_real_over_gen=E_real / E_gen,
           gap_same_share=same_gap_share, gap_total_perwedge=gap_total,
           null_mean=float(nulls.mean()), null_p99=float(np.percentile(nulls, 99)),
           gates=dict(G1_E_real_ge2=bool(E_real >= 2.0 and E_real > np.percentile(nulls, 99)),
                      G2_ratio_ge15=bool(E_real / E_gen >= 1.5),
                      G3_leverage=bool(same_gap_share >= 0.35 and rs["sameHL_share"] < 0.30)))
json.dump(out, open(D + "vnc_lineage.json", "w"), indent=1)
log("-> data_male/vnc_lineage.json")
print(json.dumps({k: out[k] for k in ("E_real", "E_gen", "E_ratio_real_over_gen", "gap_same_share", "gates")}, indent=1))
print("real:", {k: rs[k] for k in ("n_wedges", "p_close_sameHL", "p_close_diffHL", "p_close_sametype", "sameHL_share")})
print("gen :", {k: gs[k] for k in ("n_wedges", "p_close_sameHL", "p_close_diffHL", "sameHL_share")})
