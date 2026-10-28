#!/usr/bin/env python3
# 144_vnc_soma.py — ②b-1b (§49): VNC 闭合的胞体位置载体检验
# T-A 胞体距离剖面 (8 分箱); T-B 半血统×AP神经节(3分位) 同籍富集
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
xyz = np.fromfile(D + "xyz.f32", dtype=np.float32).reshape(-1, 3)
N = len(node_sc)
VNC = {15, 16, 17, 21, 22, 23, 24, 25, 26, 27}
is_vnc = np.isin(node_sc, list(VNC))
has_xyz = ~(np.isnan(xyz).any(1) | (xyz == 0).all(1))

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

# AP 轴 = VNC 胞体 std 最大轴; 三分位 (在 VNC∩parsed∩has_xyz 上)
vz = xyz[is_vnc & parsed & has_xyz]
AP = int(np.argmax(vz.std(0)))
q1, q2 = np.percentile(vz[:, AP], [33.333, 66.667])
SEG = np.full(N, -1, dtype=np.int32)
zv = xyz[:, AP]
SEG[is_vnc & has_xyz] = np.digitize(zv[is_vnc & has_xyz], [q1, q2])
log(f"AP axis={AP}, tertiles at {q1:.0f}/{q2:.0f}; VNC∩xyz {int((is_vnc & has_xyz).sum()):,}")

def analyze(pre_f, post_f, nsamp, seed):
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
    del ek, indptr, indices
    return dict(V=V, W=W, closed=closed)

NS = 6_000_000
log("抽样 real ..."); re_ = analyze(D + "e_pre.i32", D + "e_post.i32", NS, 1)
log("抽样 gen  ..."); ge = analyze(G + "_pre.i32", G + "_post.i32", NS, 1)

def wedge_dist(V, W):
    return np.sqrt(((xyz[V] - xyz[W]) ** 2).sum(1))

# ---------- T-A: 距离剖面 ----------
mA_r = is_vnc[re_["V"]] & is_vnc[re_["W"]] & has_xyz[re_["V"]] & has_xyz[re_["W"]]
mA_g = is_vnc[ge["V"]] & is_vnc[ge["W"]] & has_xyz[ge["V"]] & has_xyz[ge["W"]]
dr = wedge_dist(re_["V"][mA_r], re_["W"][mA_r]); cr = re_["closed"][mA_r]
dg = wedge_dist(ge["V"][mA_g], ge["W"][mA_g]); cg = ge["closed"][mA_g]
edges = np.percentile(dr, np.linspace(0, 100, 9)); edges[0] = 0
br = np.clip(np.digitize(dr, edges[1:-1]), 0, 7)
bg = np.clip(np.digitize(dg, edges[1:-1]), 0, 7)
prof_r = np.array([cr[br == i].mean() if (br == i).sum() else np.nan for i in range(8)])
prof_g = np.array([cg[bg == i].mean() if (bg == i).sum() else np.nan for i in range(8)])
nb_r = np.array([(br == i).sum() for i in range(8)])
binmid = (edges[:-1] + edges[1:]) / 2
ratio_r = prof_r[0] / prof_r[-1]; ratio_g = prof_g[0] / prof_g[-1]
gap_bins = nb_r / nb_r.sum() * (prof_r - prof_g)
gap_near_share = gap_bins[0] / gap_bins.sum()
log(f"T-A: real {prof_r[0]:.3f}->{prof_r[-1]:.3f} ({ratio_r:.2f}x), gen {prof_g[0]:.3f}->{prof_g[-1]:.3f} ({ratio_g:.2f}x), near-gap {gap_near_share:.2f}")

# ---------- T-B: HL × AP 同籍 ----------
mB_r = is_vnc[re_["V"]] & is_vnc[re_["W"]] & parsed[re_["V"]] & parsed[re_["W"]] & (SEG[re_["V"]] >= 0) & (SEG[re_["W"]] >= 0)
mB_g = is_vnc[ge["V"]] & is_vnc[ge["W"]] & parsed[ge["V"]] & parsed[ge["W"]] & (SEG[ge["V"]] >= 0) & (SEG[ge["W"]] >= 0)
Vr, Wr, clr = re_["V"][mB_r], re_["W"][mB_r], re_["closed"][mB_r]
Vg, Wg, clg = ge["V"][mB_g], ge["W"][mB_g], ge["closed"][mB_g]
sameB_r = (HL[Vr] == HL[Wr]) & (SEG[Vr] == SEG[Wr])
sameB_g = (HL[Vg] == HL[Wg]) & (SEG[Vg] == SEG[Wg])
sameS_r = SEG[Vr] == SEG[Wr]                       # 纯 AP 同箱 (描述性)
E_B_real = clr[sameB_r].mean() / clr[~sameB_r].mean()
E_B_gen = clg[sameB_g].mean() / clg[~sameB_g].mean()
E_S_real = clr[sameS_r].mean() / clr[~sameS_r].mean()
gapB = sameB_r.mean() * (clr[sameB_r].mean() - clg[sameB_g].mean())
gapB_total = gapB + (1 - sameB_r.mean()) * (clr[~sameB_r].mean() - clg[~sameB_g].mean())
gapB_share = gapB / gapB_total

rng = np.random.default_rng(7)
cells = np.where(parsed & is_vnc & has_xyz)[0]
COH = HL * 3 + np.clip(SEG, 0, 2)
nulls = np.empty(200)
for i in range(200):
    perm = rng.permutation(len(cells))
    cp = np.full(N, -1, dtype=np.int32); cp[cells] = COH[cells[perm]]
    same = (cp[Vr] == cp[Wr]) & (cp[Vr] >= 0) & (cp[Wr] >= 0)
    nulls[i] = clr[same].mean() / clr[~same].mean()
log(f"T-B: E_real {E_B_real:.2f} E_gen {E_B_gen:.2f} (pure-AP {E_S_real:.2f}), gap share {gapB_share:.2f}, null p99 {np.percentile(nulls,99):.3f}")

out = dict(protocol="§49 ②b-1b soma-position carrier tests", nsamp=NS, ap_axis=AP,
           ap_tertiles=[float(q1), float(q2)],
           vnc_xyz_cells=int((is_vnc & has_xyz).sum()),
           TA=dict(bin_mid_um=[float(x) for x in binmid],
                   prof_real=[float(x) for x in prof_r], prof_gen=[float(x) for x in prof_g],
                   n_real=[int(x) for x in nb_r],
                   ratio_real=float(ratio_r), ratio_gen=float(ratio_g),
                   near_gap_share=float(gap_near_share),
                   n_wedges_real=int(mA_r.sum()), n_wedges_gen=int(mA_g.sum())),
           TB=dict(n_wedges_real=int(mB_r.sum()), sameB_share=float(sameB_r.mean()),
                   p_real_same=float(clr[sameB_r].mean()), p_real_diff=float(clr[~sameB_r].mean()),
                   p_gen_same=float(clg[sameB_g].mean()), p_gen_diff=float(clg[~sameB_g].mean()),
                   E_real=float(E_B_real), E_gen=float(E_B_gen),
                   E_pureAP_real=float(E_S_real),
                   gap_share=float(gapB_share),
                   null_mean=float(nulls.mean()), null_p99=float(np.percentile(nulls, 99))),
           gates=dict(G1b=bool(ratio_r >= 1.5 and ratio_g <= 1.2 and gap_near_share >= 0.30),
                      G2b=bool(E_B_real >= 2.0 and E_B_real / E_B_gen >= 1.5 and gapB_share >= 0.35
                               and E_B_real > np.percentile(nulls, 99))))
json.dump(out, open(D + "vnc_soma.json", "w"), indent=1)
log("-> data_male/vnc_soma.json")
print(json.dumps(out["gates"], indent=1))
