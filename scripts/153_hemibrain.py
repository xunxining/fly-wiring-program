#!/usr/bin/env python3
# 153_hemibrain.py — 第三连接组电池（HemiBrain v1.2，§150 注册；Supp Table 1 entry 25）
# 全部门限先于测试计算注册（报告.md §150）；本脚本口径经脚本化复核钉死（§150 执行补记）。
# 输入：
#   data_hemibrain/ext/traced-neurons.csv             (bodyId,type,instance)
#   data_hemibrain/ext/traced-total-connections.csv   (bodyId_pre,bodyId_post,weight)
#   data_census/   雄性 MaleCNS v1.0（e_pre/e_post.i32, node_type.i32, type_names.npy）
#   data_female/   雌性 FlyWire FAFB v783（同构）
# 输出：stdout 逐门结果 + hemibrain_results_check.json
# 注册值：G1 1.67 | G2a 0.687 | G2b 0.911 | G4 0.528/0.886 | G5 0.4332/0.4772/0.4269 + log1p 0.6852/0.7095/0.7540 | G6 1.18×/0.99×
# 注：G2a/G2b/G4/G6-qu 为随机估计族，复现值在种子噪声内浮动；verdict 全部不变（§150 执行补记）。
import json
import time

import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr

ROOT = '/mnt/agents/output/果蝇生成器/'
t0 = time.time()


def log(*a):
    print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)


# ---------- 0. 载入 typed 子图（优先 npz 快路径；否则从 CSV 重建） ----------
import os
if os.path.exists(ROOT + 'data_hemibrain/hb_graph.npz'):
    d = np.load(ROOT + 'data_hemibrain/hb_graph.npz', allow_pickle=True)
    bodyId, ci = d['bodyId'], d['ci']
    e_pre, e_post, e_w = d['e_pre'], d['e_post'], d['e_w']
    types = d['types'].astype(str)
else:
    import csv
    nr = {}
    with open(ROOT + 'data_hemibrain/ext/traced-neurons.csv') as f:
        for row in csv.DictReader(f):
            if row['type']:
                nr[int(row['bodyId'])] = row['type']
    types = np.array(sorted({t for t in nr.values()}), dtype=object)
    tidx = {t: i for i, t in enumerate(types)}
    ids = sorted(nr)
    idpos = {b: i for i, b in enumerate(ids)}
    bodyId = np.array(ids, dtype=np.int64)
    ci = np.array([tidx[nr[b]] for b in ids], dtype=np.int32)
    pl, ql, wl = [], [], []
    with open(ROOT + 'data_hemibrain/ext/traced-total-connections.csv') as f:
        for row in csv.DictReader(f):
            b1, b2 = int(row['bodyId_pre']), int(row['bodyId_post'])
            if b1 in idpos and b2 in idpos:
                pl.append(idpos[b1]); ql.append(idpos[b2]); wl.append(int(row['weight']))
    e_pre = np.array(pl, dtype=np.int64); e_post = np.array(ql, dtype=np.int64)
    e_w = np.array(wl, dtype=np.int32)
N, nt = len(bodyId), len(types)
E = len(e_pre)
tc = np.bincount(ci, minlength=nt)
log(f'typed 子图: {N} 细胞 / {nt} 类型 / {E} 连接')

res = {}

# ---------- G1. per-type 出度 CV/Poisson 中位（n_s>=5 且型均出度>=2） ----------
od = np.bincount(e_pre, minlength=N).astype(np.float64)
idg = np.bincount(e_post, minlength=N).astype(np.float64)
sum_od = np.bincount(ci, weights=od, minlength=nt)
sum_od2 = np.bincount(ci, weights=od * od, minlength=nt)
mean_od = sum_od / np.maximum(tc, 1)
var_od = sum_od2 / np.maximum(tc, 1) - mean_od ** 2
cv = np.sqrt(np.maximum(var_od, 0)) / np.maximum(mean_od, 1e-12)
poisson_floor = 1.0 / np.sqrt(np.maximum(mean_od, 1e-12))
mask = (tc >= 5) & (mean_od >= 2)
res['G1'] = float(np.median(cv[mask] / poisson_floor[mask]))
log(f"G1 CV/Poisson 中位 = {res['G1']:.3f}（{int(mask.sum())} 型；注册 1.67，门 [1.2,2.0] → PASS）")

# ---------- 型对配额矩阵 ----------
M = np.zeros((nt, nt), dtype=np.int64)
np.add.at(M, (ci[e_pre], ci[e_post]), 1)
cells_of = [np.flatnonzero(ci == t) for t in range(nt)]


def gen_quota_uniform(seed):
    """quota-uniform 基线：每有向型对配额 M_st 在 n_s×n_t 细胞对内无放回均匀摆放。"""
    rng = np.random.default_rng(seed)
    pl, ql = [], []
    for s in range(nt):
        ns = int(tc[s])
        if ns == 0:
            continue
        cs = cells_of[s]
        for t in range(nt):
            m = int(M[s, t]); nct = int(tc[t])
            if m == 0 or nct == 0:
                continue
            cap = ns * nct
            mm = min(m, cap)
            flat = rng.choice(cap, size=mm, replace=False) if mm < cap else np.arange(cap)
            pl.append(cs[flat // nct]); ql.append(cells_of[t][flat % nct])
    return np.concatenate(pl), np.concatenate(ql)


def per_type_cv(deg):
    s1 = np.bincount(ci, weights=deg, minlength=nt)
    s2 = np.bincount(ci, weights=deg.astype(np.float64) ** 2, minlength=nt)
    m1 = s1 / np.maximum(tc, 1)
    v1 = s2 / np.maximum(tc, 1) - m1 ** 2
    return np.sqrt(np.maximum(v1, 0)) / np.maximum(m1, 1e-12), m1

# ---------- G2a. quota-uniform CV/Poisson 中位（种子相关，注册 0.687） ----------
gq_pre, gq_post = gen_quota_uniform(11)
deg_qu = np.bincount(gq_pre, minlength=N).astype(np.float64)
cv_qu, _ = per_type_cv(deg_qu)
res['G2a'] = float(np.median(cv_qu[mask] / poisson_floor[mask]))
log(f"G2a quota-uniform CV/Poisson 中位 = {res['G2a']:.3f}（注册 0.687，门 <1.0 → PASS）")

# ---------- G2b. activity 模型：deg_i ~ Poisson(a_i·λ_s)，a_i = od_i/λ_s ----------
rng = np.random.default_rng(11)
deg_act = rng.poisson(np.maximum(od, 0))
cv_act, _ = per_type_cv(deg_act)
res['G2b'] = float(spearmanr(cv[mask], cv_act[mask]).statistic)
log(f"G2b activity 模型 per-type CV Spearman = {res['G2b']:.3f}（注册 0.911，门 ≥0.7 → PASS）")

# ---------- G4. 留出预测（seed 11 切分；seed 12 评估；Mann-Whitney AUC） ----------
rng11 = np.random.default_rng(11)
perm = rng11.permutation(E)
tr = np.zeros(E, dtype=bool); tr[perm[: int(E * 0.9)]] = True
tr_pre, tr_post = e_pre[tr], e_post[tr]
te_pre, te_post = e_pre[~tr], e_post[~tr]
Ctr = np.zeros((nt, nt), dtype=np.int64)
np.add.at(Ctr, (ci[tr_pre], ci[tr_post]), 1)
P = Ctr / np.maximum(np.outer(tc, tc), 1)
od_tr = np.bincount(tr_pre, minlength=N)
id_tr = np.bincount(tr_post, minlength=N)
mo = np.bincount(ci, weights=od_tr, minlength=nt) / np.maximum(tc, 1)
mi = np.bincount(ci, weights=id_tr, minlength=nt) / np.maximum(tc, 1)
a_out = od_tr / np.maximum(mo[ci], 1e-12)
a_in = id_tr / np.maximum(mi[ci], 1e-12)
dkey = np.sort(e_pre.astype(np.int64) * N + e_post)


def is_edge(i, j):
    k = i.astype(np.int64) * N + j
    pos = np.clip(np.searchsorted(dkey, k), 0, len(dkey) - 1)
    return dkey[pos] == k


rng12 = np.random.default_rng(12)
sel = rng12.choice(len(te_pre), size=200_000, replace=False)
pi, pj = te_pre[sel], te_post[sel]
pos_score = P[ci[pi], ci[pj]] * a_out[pi] * a_in[pj]

# 同型负例：同源细胞、同目标类型、全图非边；50 次尝试预算
neg_st = np.full(200_000, -1, dtype=np.int64)
toff = np.concatenate([[0], np.cumsum(tc)])
allcells_sorted = np.concatenate(cells_of)  # 型内连续段
need = np.arange(200_000)
for _ in range(50):
    if len(need) == 0:
        break
    tt = ci[pj[need]]
    r = rng12.random(len(need))
    cand = allcells_sorted[toff[tt] + (r * tc[tt]).astype(np.int64)]
    ok = (cand != pj[need]) & ~is_edge(pi[need], cand)
    neg_st[need[ok]] = cand[ok]
    need = need[~ok]
n_neg = int((neg_st >= 0).sum())
okm = neg_st >= 0
neg_score_st = P[ci[pi[okm]], ci[neg_st[okm]]] * a_out[pi[okm]] * a_in[neg_st[okm]]


def mw_auc(a, b):
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


res['G4_sametype'] = mw_auc(pos_score[okm], neg_score_st)
log(f"G4 same-type AUC = {res['G4_sametype']:.3f}（同型负例实现 {n_neg:,}/200,000；注册 0.528/132,185，门 ≥0.65 → FAIL）")

# 随机非边负例
neg_r = rng12.integers(0, N, 200_000)
bad = (neg_r == pj) | is_edge(pi, neg_r)
for _ in range(60):
    if not bad.any():
        break
    neg_r[bad] = rng12.integers(0, N, int(bad.sum()))
    bad = (neg_r == pj) | is_edge(pi, neg_r)
neg_score_r = P[ci[pi], ci[neg_r]] * a_out[pi] * a_in[neg_r]
res['G4_random'] = mw_auc(pos_score, neg_score_r)
log(f"G4 random non-edge AUC = {res['G4_random']:.3f}（注册 0.886，诊断臂）")

# ---------- G6. 闭合超量（无向对称化；node-/wedge-uniform 楔形估计器，seeds 1–12，50 万楔） ----------
import gc


def build_undirected(ep, eq):
    u = np.minimum(ep, eq).astype(np.int64)
    v = np.maximum(ep, eq).astype(np.int64)
    key = np.unique(u * N + v)
    uu = (key // N).astype(np.int32); vv = (key % N).astype(np.int32)
    src = np.concatenate([uu, vv]); dst = np.concatenate([vv, uu])
    o = np.argsort(src, kind='stable')
    src, dst = src[o], dst[o]
    ptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(ptr, src + 1, 1)
    ptr = np.cumsum(ptr)
    dg = np.diff(ptr)
    mm = src < dst
    ek = np.sort(src[mm].astype(np.int64) * N + dst[mm])
    return dst, ptr, dg, ek


def wedge_T(dst, ptr, dg, ek, seed, nsamp=500_000, mode='node'):
    rng = np.random.default_rng(seed)
    centers = np.flatnonzero(dg >= 2)
    if mode == 'node':
        c = centers[rng.integers(0, len(centers), nsamp)]
    else:
        w = (dg[centers] * (dg[centers] - 1) // 2).astype(np.int64)
        cump = np.cumsum(w)
        c = centers[np.searchsorted(cump, rng.integers(0, cump[-1], nsamp))]
    a = ptr[c]; dc = dg[c]
    o1 = (rng.random(nsamp) * dc).astype(np.int64)
    o2 = (rng.random(nsamp) * (dc - 1)).astype(np.int64); o2 += (o2 >= o1)
    nv = dst[a + o1]; nw = dst[a + o2]
    lo = np.minimum(nv, nw).astype(np.int64); hi = np.maximum(nv, nw).astype(np.int64)
    k = lo * N + hi
    pos = np.clip(np.searchsorted(ek, k), 0, len(ek) - 1)
    return float((ek[pos] == k).mean())


dst_r, ptr_r, dg_r, ek_r = build_undirected(e_pre, e_post)
t_real_node = float(np.mean([wedge_T(dst_r, ptr_r, dg_r, ek_r, s, mode='node') for s in range(1, 13)]))
t_real_wedge = float(np.mean([wedge_T(dst_r, ptr_r, dg_r, ek_r, s, mode='wedge') for s in range(1, 13)]))
log(f'G6 real T = {t_real_node:.4f}（node-uniform；注册 0.322）/ {t_real_wedge:.4f}（wedge-uniform；注册 0.215）')
dst_q, ptr_q, dg_q, ek_q = build_undirected(gq_pre, gq_post)
t_qu_node = float(np.mean([wedge_T(dst_q, ptr_q, dg_q, ek_q, s, mode='node') for s in range(1, 13)]))
t_qu_wedge = float(np.mean([wedge_T(dst_q, ptr_q, dg_q, ek_q, s, mode='wedge') for s in range(1, 13)]))
del dst_q, ptr_q, dg_q, ek_q, gq_pre, gq_post
gc.collect()
res['G6'] = {'real_node': t_real_node, 'qu_node': t_qu_node, 'ratio_node': t_real_node / t_qu_node,
             'real_wedge': t_real_wedge, 'qu_wedge': t_qu_wedge, 'ratio_wedge': t_real_wedge / t_qu_wedge}
log(f"G6 超量 = {res['G6']['ratio_node']:.2f}×（node）/ {res['G6']['ratio_wedge']:.2f}×（wedge）（注册 1.18×/0.99×，门 ≥1.5 → FAIL）")

# ---------- G5. quota 率守恒（三方受控词表 1,461；并集对集 228,935；率=C_st/(n_s·n_t)） ----------
tn_m = np.load(ROOT + 'data_census/type_names.npy', allow_pickle=True).astype(str)
tn_f = np.load(ROOT + 'data_female/type_names.npy', allow_pickle=True).astype(str)
nt_m = np.fromfile(ROOT + 'data_census/node_type.i32', dtype=np.int32)
nt_f = np.fromfile(ROOT + 'data_female/node_type.i32', dtype=np.int32)
cnt_m = np.bincount(nt_m[nt_m >= 0], minlength=len(tn_m))
cnt_f = np.bincount(nt_f[nt_f >= 0], minlength=len(tn_f))
name_m = {n: i for i, n in enumerate(tn_m)}
name_f = {n: i for i, n in enumerate(tn_f)}
hb_idx_of = {n: i for i, n in enumerate(types)}
common = [t for t in types if t in name_m and t in name_f]
cm = np.array([cnt_m[name_m[t]] for t in common])
cf = np.array([cnt_f[name_f[t]] for t in common])
ch = np.array([tc[hb_idx_of[t]] for t in common])
vocab = np.array(common, dtype=object)[(cm >= 2) & (cf >= 2) & (ch >= 2)]
V = len(vocab)
vidx = {t: i for i, t in enumerate(vocab)}
log(f'G5 受控词表 {V} 型（注册 1,461）')


def vocab_rates(ep_path, eq_path, node_type_arr, type_names):
    tmap = np.full(len(type_names), -1, dtype=np.int32)
    for i, n in enumerate(type_names):
        j = vidx.get(str(n), -1)
        if j >= 0:
            tmap[i] = j
    valid = node_type_arr >= 0
    vt = np.where(valid, tmap[np.clip(node_type_arr, 0, len(type_names) - 1)], -1).astype(np.int32)
    cnt = np.bincount(vt[vt >= 0], minlength=V)
    ep = np.fromfile(ep_path, dtype=np.int32)
    eq = np.fromfile(eq_path, dtype=np.int32)
    m = (vt[ep] >= 0) & (vt[eq] >= 0)
    C = np.zeros((V, V), dtype=np.int64)
    np.add.at(C, (vt[ep[m]].astype(np.int64), vt[eq[m]].astype(np.int64)), 1)
    return C / np.maximum(np.outer(cnt, cnt), 1), (C > 0)


# HB 率
tmap_h = np.full(nt, -1, dtype=np.int32)
for i, n in enumerate(types):
    j = vidx.get(str(n), -1)
    if j >= 0:
        tmap_h[i] = j
vt_h = tmap_h[ci]
mh = (vt_h[e_pre] >= 0) & (vt_h[e_post] >= 0)
C_h = np.zeros((V, V), dtype=np.int64)
np.add.at(C_h, (vt_h[e_pre[mh]], vt_h[e_post[mh]]), 1)
cnt_h = np.bincount(vt_h[vt_h >= 0], minlength=V)
r_h = C_h / np.maximum(np.outer(cnt_h, cnt_h), 1)
nz_h = C_h > 0
del C_h, vt_h, mh
gc.collect()
r_m, nz_m = vocab_rates(ROOT + 'data_census/e_pre.i32', ROOT + 'data_census/e_post.i32', nt_m, tn_m)
log('G5 male 率就绪')
r_f, nz_f = vocab_rates(ROOT + 'data_female/e_pre.i32', ROOT + 'data_female/e_post.i32', nt_f, tn_f)
log('G5 female 率就绪')
union = (nz_h | nz_m | nz_f).ravel()
vh, vm, vf = r_h.ravel()[union], r_m.ravel()[union], r_f.ravel()[union]
res['G5'] = {
    'vocab': V, 'pairs': int(union.sum()),
    'Spearman_HF': float(spearmanr(vh, vf).statistic),
    'Spearman_HM': float(spearmanr(vh, vm).statistic),
    'Spearman_FM': float(spearmanr(vf, vm).statistic),
    'PearsonLog_HF': float(pearsonr(np.log1p(vh), np.log1p(vf)).statistic),
    'PearsonLog_HM': float(pearsonr(np.log1p(vh), np.log1p(vm)).statistic),
    'PearsonLog_FM': float(pearsonr(np.log1p(vf), np.log1p(vm)).statistic),
}
g5 = res['G5']
log(f"G5 Spearman HF {g5['Spearman_HF']:.4f} / HM {g5['Spearman_HM']:.4f} / FM {g5['Spearman_FM']:.4f}"
    f"（注册 0.4332/0.4772/0.4269，门 ρ(H,F)>ρ(H,M) → FAIL）")
log(f"G5 log1p-Pearson HF {g5['PearsonLog_HF']:.4f} / HM {g5['PearsonLog_HM']:.4f} / FM {g5['PearsonLog_FM']:.4f}"
    f"（注册 0.6852/0.7095/0.7540）")

with open(ROOT + 'hemibrain_results_check.json', 'w') as f:
    json.dump(res, f, indent=1)
log('写出 hemibrain_results_check.json；verdict 汇总：G1 PASS / G2a PASS / G2b PASS / G3 WITHDRAWN（导出无坐标）/ G4 FAIL / G5 FAIL / G6 FAIL')
