#!/usr/bin/env python3
# 147_p10_edgepred.py — P10 (§57): 边级留出预测验证, 单性别一跑
# usage: python3 147_p10_edgepred.py female|male
import json, sys, time
import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

SEX = sys.argv[1] if len(sys.argv) > 1 else "female"
ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + ("data_female/" if SEX == "female" else "data_male/")
LATF = "lat2q_f3.npz" if SEX == "female" else "lat2q_m.npz"
BIG = 500          # 大对阈值 (逐对核)
R = 20             # 晶格偏移窗口
NB = 24            # 距离分箱
NTEST = 200_000

ep = np.fromfile(D + "e_pre.i32", dtype=np.int32)
eq = np.fromfile(D + "e_post.i32", dtype=np.int32)
nty = np.fromfile(D + "node_type.i32", dtype=np.int32)
xyz = np.fromfile(D + "xyz.f32", dtype=np.float32).reshape(-1, 3)
zq = np.load(D + LATF)
PQ = zq["pq"].astype(np.float64); HEM = zq["hem"]
N = len(nty)
E = len(ep)
log(f"{SEX}: N={N:,} E={E:,}")

# ---- 90/10 切分 (seed 11) ----
rng = np.random.default_rng(11)
is_test = rng.random(E) < 0.10
tr = ~is_test
tp_, tq_ = ep[tr], eq[tr]
# 全图边集合 (负例校验用)
fullset = set((ep.astype(np.int64) * N + eq).tolist()) if False else None
key_all = np.unique(ep.astype(np.int64) * N + eq)   # 排序键集合

# ---- 训练集统计 ----
tpre, tpost = nty[tp_], nty[tq_]
pair_id = tpre.astype(np.int64) * 100_000 + tpost     # 类型对 id (类型数<100000)
upt, cnt = np.unique(pair_id, return_counts=True)
q_map = dict(zip(upt.tolist(), cnt.tolist()))
# 类型对全细胞对数 M_st
tall_pre, tall_post = np.unique(nty, return_counts=True)
tmap = dict(zip(tall_pre.tolist(), tall_post.tolist()))
od = np.bincount(tp_, minlength=N).astype(np.float64)
idg = np.bincount(tq_, minlength=N).astype(np.float64)
od_mu = {t: od[nty == t].mean() for t in tmap}
id_mu = {t: idg[nty == t].mean() for t in tmap}
act_o = od / np.array([od_mu[t] for t in nty]).clip(1e-9)
act_i = idg / np.array([id_mu[t] for t in nty]).clip(1e-9)

# 距离 (soma 欧氏, 缺 xyz 置 NaN 箱)
def dist_of(a, b):
    d = np.sqrt(((xyz[a] - xyz[b]) ** 2).sum(1))
    bad = np.isnan(d) | (xyz[a] == 0).all(1) | (xyz[b] == 0).all(1)
    d[bad] = -1
    return d
d_tr = dist_of(tp_, tq_)
dmax = np.nanmax(d_tr[d_tr >= 0])
def dbin(d):
    b = np.clip((d / dmax * NB).astype(np.int32), 0, NB - 1)
    b[d < 0] = NB                       # 缺失箱
    return b
db_tr = dbin(d_tr)
# 全局距离 LR: P(d|edge)/P(d|any) — any 用训练边端点的边际近似: 随机细胞对采样
rng2 = np.random.default_rng(21)
samp_a = rng2.integers(0, N, 2_000_000); samp_b = rng2.integers(0, N, 2_000_000)
ok = samp_a != samp_b
samp_a, samp_b = samp_a[ok][:1_000_000], samp_b[ok][:1_000_000]
db_bg = dbin(dist_of(samp_a, samp_b))
h_e = np.bincount(db_tr, minlength=NB + 1) + 0.5
h_b = np.bincount(db_bg, minlength=NB + 1) + 0.5
LR_glob = (h_e / h_e.sum()) / (h_b / h_b.sum())
# 大对逐对距离 LR
big_pairs = set(upt[cnt >= BIG].tolist())
LR_dist = {}
for p in big_pairs:
    m = pair_id == p
    h = np.bincount(db_tr[m], minlength=NB + 1) + 0.5
    LR_dist[p] = (h / h.sum()) / (h_b / h_b.sum())
log(f"距离核: 大对 {len(big_pairs):,}")

# 晶格偏移 LR (两端 hem≠0 且同半球)
pq_i = np.floor(PQ).astype(np.int64)
def off_of(a, b):
    d = pq_i[b] - pq_i[a]
    ok = (HEM[a] != 0) & (HEM[b] != 0) & (HEM[a] == HEM[b])
    d[~ok] = 9999
    return np.clip(d, -R, R)
off_tr = off_of(tp_, tq_)
ob_tr = (off_tr[:, 0] + R) * (2 * R + 1) + (off_tr[:, 1] + R)
ob_tr[(off_tr == 9999 - 0).all(1)] = -1   # 占位修正
valid = (HEM[tp_] != 0) & (HEM[tq_] != 0) & (HEM[tp_] == HEM[tq_])
ob_tr[~valid] = -1
NBIN = (2 * R + 1) ** 2
LR_lat = {}
for p in big_pairs:
    m = (pair_id == p) & valid
    if m.sum() < 50: continue
    h = np.bincount(ob_tr[m], minlength=NBIN) + 0.25
    LR_lat[p] = h / h.sum() * NBIN        # 相对均匀
log(f"晶格核: {len(LR_lat):,} 大对")

# ---- 测试集: 20 万留出边 + 同型负例 + 随机负例 ----
te = np.where(is_test)[0]
rng3 = np.random.default_rng(12)
te = te[rng3.permutation(len(te))[:NTEST]]
ti, tj = ep[te], eq[te]
tt = nty[tj]
# 同型负例: 同 i, j' 同类型, (i,j') 全图非边
type_members = {}
neg_j = np.empty(len(te), np.int64)
need = np.ones(len(te), bool)
for rnd in range(60):
    if not need.any(): break
    cand = rng3.integers(0, N, need.sum())
    # 类型匹配
    mt = nty[cand] == tt[need]
    idx = np.where(need)[0][mt]
    cand = cand[mt]
    # 全图非边校验
    k = ti[idx].astype(np.int64) * N + cand
    h = np.searchsorted(key_all, k)
    free = (h >= len(key_all)) | (key_all[np.clip(h, 0, len(key_all) - 1)] != k)
    neg_j[idx[free]] = cand[free]
    need[idx[free]] = False
log(f"同型负例未配 {need.sum()}")
neg_j[need] = (neg_j[need] * 0 + 1)     # 兜底 (极少)
# 随机负例
rj = rng3.integers(0, N, len(te))
k = ti.astype(np.int64) * N + rj
h = np.searchsorted(key_all, k)
bad = (h < len(key_all)) & (key_all[np.clip(h, 0, len(key_all) - 1)] == k) | (rj == ti)
rj[bad] = (rj[bad] + 7) % N

def scores(a, b):
    pid = nty[a].astype(np.int64) * 100_000 + nty[b]
    q = np.array([q_map.get(p, 0.0) for p in pid]) + 1e-9
    M = np.array([tmap.get(nty[x], 1) for x in a], np.float64) * np.array([tmap.get(nty[y], 1) for y in b], np.float64)
    q = q / np.clip(M, 1, None) * 1e6    # 每百万细胞对配额率
    ac = act_o[a] * act_i[b]
    db = dbin(dist_of(a, b))
    dl = np.array([LR_dist.get(p, LR_glob)[bb] for p, bb in zip(pid, db)])
    off = off_of(a, b)
    ob = (off[:, 0] + R) * (2 * R + 1) + (off[:, 1] + R)
    v = (HEM[a] != 0) & (HEM[b] != 0) & (HEM[a] == HEM[b])
    ll = np.ones(len(a))
    for p in LR_lat:                     # 向量化查表
        m = (pid == p) & v
        if m.any(): ll[m] = LR_lat[p][ob[m]]
    deg = od[a] * idg[b]
    return dict(quota=q, act=ac, dist=dl, lat=ll, deg=deg,
                q_act=q * ac, q_dist=q * dl, q_lat=q * ll, full=q * ac * dl * ll)

sp = scores(ti, tj); sn = scores(ti, neg_j); sr_ = scores(ti, rj)
def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float((p > n).mean() + 0.5 * (p == n).mean())
arms = ["quota", "q_act", "q_dist", "q_lat", "full", "deg"]
res = {a: dict(sametype=auc(sp[a], sn[a]), random=auc(sp[a], sr_[a])) for a in arms}
log("done")
out = dict(protocol="§57 P10 edge-level held-out prediction", sex=SEX,
           n_test=int(len(te)), n_train_edges=int(tr.sum()), n_big_pairs=len(big_pairs),
           auc=res,
           gates=dict(G1=bool(res["full"]["sametype"] >= 0.65),
                      G2=bool(res["full"]["sametype"] - max(res[a]["sametype"] for a in ["quota", "q_act", "q_dist", "q_lat"]) >= 0.05),
                      G3=bool(res["full"]["sametype"] >= res["deg"]["sametype"] + 0.10),
                      guard=bool(res["quota"]["random"] >= 0.95)))
json.dump(out, open(D + f"p10_edgepred_{SEX}.json", "w"), indent=1)
log(f"-> {D}p10_edgepred_{SEX}.json")
print(json.dumps(res, indent=1))
print(json.dumps(out["gates"], indent=1))
