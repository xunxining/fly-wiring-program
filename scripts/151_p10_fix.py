#!/usr/bin/env python3
# 151_p10_fix.py — §71 勘误修正: P10 边级留出预测, 正确类型桶同型负例
# 与 147 逐字一致(seed 11 切分/线性距离箱/活动+晶格臂/配对AUC), 唯一差异=负例采样器
# usage: python3 151_p10_fix.py female|male
import json, sys, time
import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

SEX = sys.argv[1]
ROOT = "/mnt/agents/output/果蝇生成器/"
D = ROOT + ("data_female/" if SEX == "female" else "data_male/")
LATF = "lat2q_f3.npz" if SEX == "female" else "lat2q_m.npz"
BIG, R, NB, NTEST = 500, 20, 24, 200_000

ep = np.fromfile(D+"e_pre.i32", dtype=np.int32)
eq = np.fromfile(D+"e_post.i32", dtype=np.int32)
nty = np.fromfile(D+"node_type.i32", dtype=np.int32)
xyz = np.fromfile(D+"xyz.f32", dtype=np.float32).reshape(-1,3)
zq = np.load(D+LATF)
PQ = zq["pq"].astype(np.float64); HEM = zq["hem"]
N = len(nty); E = len(ep)
log(f"{SEX}: N={N:,} E={E:,}")

rng = np.random.default_rng(11)
is_test = rng.random(E) < 0.10
tr = ~is_test
tp_, tq_ = ep[tr], eq[tr]
key_all = np.unique(ep.astype(np.int64)*N + eq)

tpre, tpost = nty[tp_], nty[tq_]
pair_id = tpre.astype(np.int64)*100_000 + tpost
upt, cnt = np.unique(pair_id, return_counts=True)
q_map = dict(zip(upt.tolist(), cnt.tolist()))
tall_pre, tall_post = np.unique(nty, return_counts=True)
tmap = dict(zip(tall_pre.tolist(), tall_post.tolist()))
od = np.bincount(tp_, minlength=N).astype(np.float64)
idg = np.bincount(tq_, minlength=N).astype(np.float64)
od_mu = {t: od[nty==t].mean() for t in tmap}
id_mu = {t: idg[nty==t].mean() for t in tmap}
act_o = od / np.array([od_mu[t] for t in nty]).clip(1e-9)
act_i = idg / np.array([id_mu[t] for t in nty]).clip(1e-9)

def dist_of(a, b):
    d = np.sqrt(((xyz[a]-xyz[b])**2).sum(1))
    bad = np.isnan(d) | (xyz[a]==0).all(1) | (xyz[b]==0).all(1)
    d[bad] = -1
    return d
d_tr = dist_of(tp_, tq_)
dmax = np.nanmax(d_tr[d_tr >= 0])
def dbin(d):
    b = np.clip((d/dmax*NB).astype(np.int32), 0, NB-1)
    b[d < 0] = NB
    return b
db_tr = dbin(d_tr)
rng2 = np.random.default_rng(21)
samp_a = rng2.integers(0, N, 2_000_000); samp_b = rng2.integers(0, N, 2_000_000)
ok = samp_a != samp_b
samp_a, samp_b = samp_a[ok][:1_000_000], samp_b[ok][:1_000_000]
db_bg = dbin(dist_of(samp_a, samp_b))
h_e = np.bincount(db_tr, minlength=NB+1) + 0.5
h_b = np.bincount(db_bg, minlength=NB+1) + 0.5
LR_glob = (h_e/h_e.sum())/(h_b/h_b.sum())
big_pairs = set(upt[cnt >= BIG].tolist())
LR_dist = {}
for p in big_pairs:
    m = pair_id == p
    h = np.bincount(db_tr[m], minlength=NB+1) + 0.5
    LR_dist[p] = (h/h.sum())/(h_b/h_b.sum())
log(f"距离核: 大对 {len(big_pairs):,}")

pq_i = np.floor(PQ).astype(np.int64)
def off_of(a, b):
    d = pq_i[b] - pq_i[a]
    ok = (HEM[a] != 0) & (HEM[b] != 0) & (HEM[a] == HEM[b])
    d[~ok] = 9999
    return np.clip(d, -R, R)
off_tr = off_of(tp_, tq_)
ob_tr = (off_tr[:,0]+R)*(2*R+1) + (off_tr[:,1]+R)
valid = (HEM[tp_] != 0) & (HEM[tq_] != 0) & (HEM[tp_] == HEM[tq_])
ob_tr[~valid] = -1
NBIN = (2*R+1)**2
LR_lat = {}
for p in big_pairs:
    m = (pair_id == p) & valid
    if m.sum() < 50: continue
    h = np.bincount(ob_tr[m], minlength=NBIN) + 0.25
    LR_lat[p] = h/h.sum()*NBIN
log(f"晶格核: {len(LR_lat):,} 大对")

# ---- 测试集 ----
te = np.where(is_test)[0]
rng3 = np.random.default_rng(12)
te = te[rng3.permutation(len(te))[:NTEST]]
ti, tj = ep[te], eq[te]
tt = nty[tj]

# ---- 正确同型负例: 类型桶采样 + 全图非边校验 ----
order = np.argsort(nty, kind="stable")
cnts = np.bincount(nty)
starts = np.concatenate([[0], np.cumsum(cnts)[:-1]])
neg_j = np.empty(len(te), np.int64)
member_ok = cnts[tt] >= 2                  # 桶内至少要有非自身的候选
dropped = int((~member_ok).sum())
matched = np.zeros(len(te), bool)
active = np.where(member_ok)[0]
for rnd in range(40):
    if not len(active): break
    t_n = tt[active]
    cand = order[starts[t_n] + rng3.integers(0, cnts[t_n])]
    k = ti[active].astype(np.int64)*N + cand
    h = np.searchsorted(key_all, k)
    free = (cand != ti[active]) & ((h >= len(key_all)) | (key_all[np.clip(h,0,len(key_all)-1)] != k))
    neg_j[active[free]] = cand[free]
    matched[active[free]] = True
    active = active[~free]
    if rnd in (0, 1, 2, 4, 9, 19, 39): log(f"负例轮{rnd}: 剩余 {len(active):,}")
unmatched = int(len(active))
log(f"同型负例: 剔除单成员类型 {dropped}, 40轮后未配 {unmatched} -> 均剔除")
km = member_ok & matched
ti, tj, neg_j = ti[km], tj[km], neg_j[km]
log(f"最终测试条目 {len(ti):,}")
# 随机负例
rj = rng3.integers(0, N, len(ti))
k = ti.astype(np.int64)*N + rj
h = np.searchsorted(key_all, k)
bad = (h < len(key_all)) & (key_all[np.clip(h,0,len(key_all)-1)] == k) | (rj == ti)
rj[bad] = (rj[bad]+7) % N

def scores(a, b):
    pid = nty[a].astype(np.int64)*100_000 + nty[b]
    q = np.array([q_map.get(p, 0.0) for p in pid]) + 1e-9
    Mm = np.array([tmap.get(nty[x],1) for x in a], np.float64)*np.array([tmap.get(nty[y],1) for y in b], np.float64)
    q = q / np.clip(Mm, 1, None) * 1e6
    ac = act_o[a]*act_i[b]
    db = dbin(dist_of(a, b))
    dl = np.array([LR_dist.get(p, LR_glob)[bb] for p, bb in zip(pid, db)])
    off = off_of(a, b)
    ob = (off[:,0]+R)*(2*R+1) + (off[:,1]+R)
    v = (HEM[a] != 0) & (HEM[b] != 0) & (HEM[a] == HEM[b])
    ll = np.ones(len(a))
    for p in LR_lat:
        m = (pid == p) & v
        if m.any(): ll[m] = LR_lat[p][ob[m]]
    deg = od[a]*idg[b]
    return dict(quota=q, act=ac, dist=dl, lat=ll, deg=deg,
                q_act=q*ac, q_dist=q*dl, q_lat=q*ll, full=q*ac*dl*ll)

sp = scores(ti, tj); sn = scores(ti, neg_j); sr_ = scores(ti, rj)
def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float((p > n).mean() + 0.5*(p == n).mean())
arms = ["quota", "q_act", "q_dist", "q_lat", "full", "deg"]
res = {a: dict(sametype=auc(sp[a], sn[a]), random=auc(sp[a], sr_[a])) for a in arms}
log("done")
out = dict(protocol="§71 P10 fixed (type-bucket same-type negatives)", sex=SEX,
           n_test=int(len(ti)), n_dropped_single_member=dropped, n_unmatched=unmatched,
           n_train_edges=int(tr.sum()), n_big_pairs=len(big_pairs), auc=res,
           gates=dict(G1=bool(res["full"]["sametype"] >= 0.65),
                      G2=bool(res["full"]["sametype"] - max(res[a]["sametype"] for a in ["quota","q_act","q_dist","q_lat"]) >= 0.05),
                      G3=bool(res["full"]["sametype"] >= res["deg"]["sametype"] + 0.10),
                      guard=bool(res["quota"]["random"] >= 0.95)))
json.dump(out, open(D+f"p10_fixed_{SEX}.json", "w"), indent=1)
log(f"-> {D}p10_fixed_{SEX}.json")
print(json.dumps(res, indent=1))
print(json.dumps(out["gates"], indent=1))
